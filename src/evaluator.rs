//! 評価器 — perplexity 自動算出とベストチェックポイント管理。
//!
//! 学習中に定期的に評価を行い、ベスト loss のチェックポイントを自動保存する。

use crate::checkpoint::CheckpointData;
use crate::trainer::TrainableNetwork;
use alice_ml::training::LossResult;
use std::path::{Path, PathBuf};

/// 評価結果。
#[derive(Clone, Debug)]
pub struct EvalResult {
    /// エポック番号。
    pub epoch: usize,
    /// 評価データセットの平均 loss。
    pub avg_loss: f32,
    /// Perplexity (exp(avg_loss))。
    pub perplexity: f32,
}

impl EvalResult {
    /// 新しい評価結果を作成する。
    #[must_use]
    pub fn new(epoch: usize, avg_loss: f32) -> Self {
        Self {
            epoch,
            avg_loss,
            perplexity: avg_loss.exp(),
        }
    }
}

/// ベストチェックポイント追跡器。
///
/// これまでの最小 loss を記録し、ベスト更新時にチェックポイントを保存する。
pub struct BestCheckpointTracker {
    /// ベスト loss。
    best_loss: f32,
    /// ベストのエポック番号。
    best_epoch: usize,
    /// 保存先ディレクトリ。
    save_dir: PathBuf,
}

impl BestCheckpointTracker {
    /// 新しいトラッカーを構築する。
    #[must_use]
    pub fn new<P: AsRef<Path>>(save_dir: P) -> Self {
        Self {
            best_loss: f32::MAX,
            best_epoch: 0,
            save_dir: save_dir.as_ref().to_path_buf(),
        }
    }

    /// 現在のベスト loss を返す。
    #[must_use]
    pub fn best_loss(&self) -> f32 {
        self.best_loss
    }

    /// ベスト更新したエポックを返す。
    #[must_use]
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    /// 評価結果を受け取り、ベスト更新ならチェックポイントを保存する。
    ///
    /// # Returns
    ///
    /// ベストが更新されたら `true`。
    ///
    /// # Errors
    ///
    /// チェックポイント保存時の I/O エラー。
    pub fn update(
        &mut self,
        eval: &EvalResult,
        weights: &[f32],
        optimizer_state: &[f32],
        learning_rate: f32,
    ) -> std::io::Result<bool> {
        if eval.avg_loss < self.best_loss {
            self.best_loss = eval.avg_loss;
            self.best_epoch = eval.epoch;

            std::fs::create_dir_all(&self.save_dir)?;
            let path = self.save_dir.join("best.ckpt");
            let ckpt = CheckpointData::new(
                eval.epoch,
                0,
                eval.avg_loss,
                learning_rate,
                weights.to_vec(),
                optimizer_state.to_vec(),
            );
            ckpt.save_to_file(path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// 評価データセットでモデルを評価する。
///
/// # 引数
///
/// - `network` — 評価対象のネットワーク
/// - `inputs` — 評価入力データ
/// - `targets` — 評価ターゲットデータ
/// - `loss_fn` — loss 関数
/// - `epoch` — 現在のエポック番号
///
/// # Panics
///
/// `inputs` と `targets` の長さが異なる場合。
pub fn evaluate<N, L>(
    network: &N,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
    loss_fn: L,
    epoch: usize,
) -> EvalResult
where
    N: TrainableNetwork,
    L: Fn(&[f32], &[f32], &mut [f32]) -> LossResult,
{
    assert_eq!(inputs.len(), targets.len());

    if inputs.is_empty() {
        return EvalResult::new(epoch, 0.0);
    }

    let out_size = network.output_size();
    let mut total_loss = 0.0_f32;

    for (input, target) in inputs.iter().zip(targets.iter()) {
        let mut output = vec![0.0_f32; out_size];
        network.forward(input, &mut output);

        let mut grad_dummy = vec![0.0_f32; out_size];
        let loss = loss_fn(&output, target, &mut grad_dummy);
        total_loss += loss.value;
    }

    let avg_loss = total_loss / inputs.len() as f32;
    EvalResult::new(epoch, avg_loss)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alice_ml::training::mse_loss;

    // テスト用の簡易ネットワーク（y = w * x）
    struct IdentityNet {
        weight: f32,
    }

    impl TrainableNetwork for IdentityNet {
        fn forward(&self, input: &[f32], output: &mut [f32]) {
            output[0] = self.weight * input[0];
        }
        fn backward(&mut self, _input: &[f32], _grad_output: &[f32], _grad_input: &mut [f32]) {}
        fn update_params(&mut self, _lr: f32) {}
        fn zero_grad(&mut self) {}
        fn output_size(&self) -> usize {
            1
        }
        fn input_size(&self) -> usize {
            1
        }
    }

    // --- EvalResult ---

    #[test]
    fn eval_result_perplexity() {
        let r = EvalResult::new(0, 1.0);
        assert!((r.perplexity - std::f32::consts::E).abs() < 0.01);
    }

    #[test]
    fn eval_result_zero_loss() {
        let r = EvalResult::new(0, 0.0);
        assert!((r.perplexity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn eval_result_fields() {
        let r = EvalResult::new(5, 0.42);
        assert_eq!(r.epoch, 5);
        assert!((r.avg_loss - 0.42).abs() < 1e-6);
    }

    #[test]
    fn eval_result_clone() {
        let r = EvalResult::new(1, 0.5);
        let r2 = r.clone();
        assert_eq!(r2.epoch, 1);
    }

    #[test]
    fn eval_result_debug() {
        let r = EvalResult::new(0, 1.0);
        let s = format!("{r:?}");
        assert!(s.contains("perplexity"));
    }

    // --- evaluate ---

    #[test]
    fn evaluate_perfect_model() {
        // y = 2x のモデルで target = 2x → loss ≈ 0
        let net = IdentityNet { weight: 2.0 };
        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![vec![2.0], vec![4.0], vec![6.0]];
        let result = evaluate(&net, &inputs, &targets, mse_loss, 0);
        assert!(
            result.avg_loss < 1e-6,
            "perfect model loss: {}",
            result.avg_loss
        );
    }

    #[test]
    fn evaluate_imperfect_model() {
        // y = 1*x だが target = 2*x → loss > 0
        let net = IdentityNet { weight: 1.0 };
        let inputs = vec![vec![1.0], vec![2.0]];
        let targets = vec![vec![2.0], vec![4.0]];
        let result = evaluate(&net, &inputs, &targets, mse_loss, 3);
        assert!(result.avg_loss > 0.0);
        assert_eq!(result.epoch, 3);
    }

    #[test]
    fn evaluate_empty_data() {
        let net = IdentityNet { weight: 1.0 };
        let result = evaluate(&net, &[], &[], mse_loss, 0);
        assert!((result.avg_loss - 0.0).abs() < 1e-6);
    }

    #[test]
    fn evaluate_single_sample() {
        let net = IdentityNet { weight: 0.0 };
        let inputs = vec![vec![1.0]];
        let targets = vec![vec![5.0]];
        let result = evaluate(&net, &inputs, &targets, mse_loss, 0);
        // (0 - 5)^2 = 25
        assert!((result.avg_loss - 25.0).abs() < 1e-4);
    }

    #[test]
    fn evaluate_perplexity_finite() {
        let net = IdentityNet { weight: 1.0 };
        let inputs = vec![vec![1.0]];
        let targets = vec![vec![2.0]];
        let result = evaluate(&net, &inputs, &targets, mse_loss, 0);
        assert!(result.perplexity.is_finite());
        assert!(result.perplexity > 0.0);
    }

    // --- BestCheckpointTracker ---

    #[test]
    fn tracker_initial_state() {
        let dir = tempfile::tempdir().unwrap();
        let tracker = BestCheckpointTracker::new(dir.path());
        assert_eq!(tracker.best_loss(), f32::MAX);
        assert_eq!(tracker.best_epoch(), 0);
    }

    #[test]
    fn tracker_updates_on_improvement() {
        let dir = tempfile::tempdir().unwrap();
        let mut tracker = BestCheckpointTracker::new(dir.path());

        let eval = EvalResult::new(1, 0.5);
        let updated = tracker.update(&eval, &[1.0, 2.0], &[], 0.001).unwrap();
        assert!(updated);
        assert!((tracker.best_loss() - 0.5).abs() < 1e-6);
        assert_eq!(tracker.best_epoch(), 1);
    }

    #[test]
    fn tracker_no_update_on_worse() {
        let dir = tempfile::tempdir().unwrap();
        let mut tracker = BestCheckpointTracker::new(dir.path());

        let eval1 = EvalResult::new(1, 0.3);
        tracker.update(&eval1, &[1.0], &[], 0.001).unwrap();

        let eval2 = EvalResult::new(2, 0.5);
        let updated = tracker.update(&eval2, &[1.0], &[], 0.001).unwrap();
        assert!(!updated);
        assert!((tracker.best_loss() - 0.3).abs() < 1e-6);
        assert_eq!(tracker.best_epoch(), 1);
    }

    #[test]
    fn tracker_saves_checkpoint_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut tracker = BestCheckpointTracker::new(dir.path());

        let eval = EvalResult::new(5, 0.1);
        tracker
            .update(&eval, &[1.0, 2.0, 3.0], &[0.1], 0.001)
            .unwrap();

        let ckpt_path = dir.path().join("best.ckpt");
        assert!(ckpt_path.exists());

        let loaded = CheckpointData::load_from_file(&ckpt_path).unwrap();
        assert_eq!(loaded.meta.epoch, 5);
        assert_eq!(loaded.weights.len(), 3);
        assert_eq!(loaded.optimizer_state.len(), 1);
    }

    #[test]
    fn tracker_overwrites_on_new_best() {
        let dir = tempfile::tempdir().unwrap();
        let mut tracker = BestCheckpointTracker::new(dir.path());

        let eval1 = EvalResult::new(1, 0.5);
        tracker.update(&eval1, &[1.0], &[], 0.001).unwrap();

        let eval2 = EvalResult::new(3, 0.2);
        tracker.update(&eval2, &[2.0], &[], 0.001).unwrap();

        let loaded = CheckpointData::load_from_file(dir.path().join("best.ckpt")).unwrap();
        assert_eq!(loaded.meta.epoch, 3);
        assert!((loaded.weights[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn tracker_successive_improvements() {
        let dir = tempfile::tempdir().unwrap();
        let mut tracker = BestCheckpointTracker::new(dir.path());

        for i in (1..=5).rev() {
            let eval = EvalResult::new(5 - i, i as f32 * 0.1);
            let updated = tracker.update(&eval, &[1.0], &[], 0.001).unwrap();
            assert!(updated, "epoch {} should improve", 5 - i);
        }
        assert!((tracker.best_loss() - 0.1).abs() < 1e-6);
    }
}
