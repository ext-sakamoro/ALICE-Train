//! 学習ループ — forward → loss → backward → optimizer step。
//!
//! `TrainableNetwork` トレイトを実装すれば任意のネットワーク構造を学習可能。

use alice_ml::training::{AdamConfig, LossResult, SgdConfig};

/// 学習対象ネットワークのトレイト。
///
/// ternary 重みの学習は STE (Straight-Through Estimator) で行うため、
/// 潜在 FP32 重みを保持し、forward 時にternary量子化する。
pub trait TrainableNetwork {
    /// forward パス。入力 → 出力。
    fn forward(&self, input: &[f32], output: &mut [f32]);

    /// backward パス。出力勾配 → 入力勾配。
    /// 内部で重み勾配も蓄積する。
    fn backward(&mut self, input: &[f32], grad_output: &[f32], grad_input: &mut [f32]);

    /// 蓄積された重み勾配を optimizer に渡してパラメータ更新。
    fn update_params(&mut self, learning_rate: f32);

    /// 蓄積された勾配をゼロクリア。
    fn zero_grad(&mut self);

    /// 出力次元数。
    fn output_size(&self) -> usize;

    /// 入力次元数。
    fn input_size(&self) -> usize;
}

/// Optimizer の種別。
#[derive(Debug, Clone)]
pub enum OptimizerConfig {
    /// SGD (momentum 対応)。
    Sgd(SgdConfig),
    /// Adam。
    Adam(AdamConfig),
}

/// 学習設定。
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// エポック数。
    pub epochs: usize,
    /// バッチサイズ。
    pub batch_size: usize,
    /// 学習率。
    pub learning_rate: f32,
    /// ログ出力間隔 (エポック単位)。
    pub log_interval: usize,
}

impl TrainConfig {
    /// デフォルト設定 (epochs=100, batch=32, lr=0.001, log=10)。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            log_interval: 10,
        }
    }

    /// エポック数を指定して設定を作成。
    #[must_use]
    pub const fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// 学習率を指定して設定を作成。
    #[must_use]
    pub const fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// 学習ループの実行器。
pub struct Trainer {
    /// 設定。
    pub config: TrainConfig,
}

/// 1エポックの学習結果。
#[derive(Debug, Clone, Copy)]
pub struct EpochResult {
    /// エポック番号 (0-indexed)。
    pub epoch: usize,
    /// 平均 loss。
    pub avg_loss: f32,
}

impl EpochResult {
    /// 新しい `EpochResult` を作成。
    #[must_use]
    pub const fn new(epoch: usize, avg_loss: f32) -> Self {
        Self { epoch, avg_loss }
    }
}

impl Trainer {
    /// 新しい `Trainer` を作成。
    #[must_use]
    pub const fn new(config: TrainConfig) -> Self {
        Self { config }
    }

    /// 学習ループを実行。
    ///
    /// # 引数
    ///
    /// - `network` — `TrainableNetwork` を実装したモデル
    /// - `inputs` — 入力データ (各要素は1サンプル)
    /// - `targets` — ターゲットデータ (各要素は1サンプル)
    /// - `loss_fn` — loss 関数 (predictions, targets, `grad_out`) → `LossResult`
    ///
    /// # Panics
    ///
    /// - `inputs` と `targets` の長さが異なる場合
    /// - データが空の場合
    /// - 各サンプルの長さがネットワークの入出力次元と異なる場合
    ///
    /// # 戻り値
    ///
    /// 各エポックの `EpochResult` のベクトル。
    pub fn train<N, L>(
        &self,
        network: &mut N,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        loss_fn: L,
    ) -> Vec<EpochResult>
    where
        N: TrainableNetwork,
        L: Fn(&[f32], &[f32], &mut [f32]) -> LossResult,
    {
        assert_eq!(inputs.len(), targets.len());
        assert!(!inputs.is_empty());

        let out_size = network.output_size();
        let in_size = network.input_size();
        let n_samples = inputs.len();

        let mut results = Vec::with_capacity(self.config.epochs);

        for epoch in 0..self.config.epochs {
            let mut total_loss = 0.0_f32;

            for sample_idx in 0..n_samples {
                let input = &inputs[sample_idx];
                let target = &targets[sample_idx];

                assert_eq!(input.len(), in_size);
                assert_eq!(target.len(), out_size);

                let mut output = vec![0.0_f32; out_size];
                network.forward(input, &mut output);

                let mut grad_output = vec![0.0_f32; out_size];
                let loss = loss_fn(&output, target, &mut grad_output);
                total_loss += loss.value;

                let mut grad_input = vec![0.0_f32; in_size];
                network.backward(input, &grad_output, &mut grad_input);

                network.update_params(self.config.learning_rate);
                network.zero_grad();
            }

            let avg_loss = total_loss / n_samples as f32;
            results.push(EpochResult { epoch, avg_loss });
        }

        results
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::redundant_clone)]
mod tests {
    use super::*;
    use alice_ml::training::{cross_entropy_loss, mae_loss, mse_loss};

    struct SimpleLinear {
        weights: Vec<f32>,
        bias: Vec<f32>,
        grad_w: Vec<f32>,
        grad_b: Vec<f32>,
        in_size: usize,
        out_size: usize,
    }

    impl SimpleLinear {
        fn new(in_size: usize, out_size: usize) -> Self {
            let weights: Vec<f32> = (0..out_size * in_size)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
                .collect();
            let bias = vec![0.0_f32; out_size];
            Self {
                grad_w: vec![0.0; out_size * in_size],
                grad_b: vec![0.0; out_size],
                weights,
                bias,
                in_size,
                out_size,
            }
        }

        fn with_weights(in_size: usize, out_size: usize, weights: Vec<f32>) -> Self {
            let bias = vec![0.0_f32; out_size];
            Self {
                grad_w: vec![0.0; out_size * in_size],
                grad_b: vec![0.0; out_size],
                weights,
                bias,
                in_size,
                out_size,
            }
        }
    }

    impl TrainableNetwork for SimpleLinear {
        fn forward(&self, input: &[f32], output: &mut [f32]) {
            for j in 0..self.out_size {
                let mut sum = self.bias[j];
                for i in 0..self.in_size {
                    sum += self.weights[j * self.in_size + i] * input[i];
                }
                output[j] = sum;
            }
        }

        fn backward(&mut self, input: &[f32], grad_output: &[f32], grad_input: &mut [f32]) {
            for i in 0..self.in_size {
                let mut sum = 0.0_f32;
                for j in 0..self.out_size {
                    sum += self.weights[j * self.in_size + i] * grad_output[j];
                }
                grad_input[i] = sum;
            }
            for j in 0..self.out_size {
                for i in 0..self.in_size {
                    self.grad_w[j * self.in_size + i] += grad_output[j] * input[i];
                }
                self.grad_b[j] += grad_output[j];
            }
        }

        fn update_params(&mut self, learning_rate: f32) {
            for i in 0..self.weights.len() {
                self.weights[i] -= learning_rate * self.grad_w[i];
            }
            for i in 0..self.bias.len() {
                self.bias[i] -= learning_rate * self.grad_b[i];
            }
        }

        fn zero_grad(&mut self) {
            self.grad_w.fill(0.0);
            self.grad_b.fill(0.0);
        }

        fn output_size(&self) -> usize {
            self.out_size
        }

        fn input_size(&self) -> usize {
            self.in_size
        }
    }

    // ---- TrainConfig ----

    #[test]
    fn train_config_default() {
        let cfg = TrainConfig::default();
        assert_eq!(cfg.epochs, 100);
        assert_eq!(cfg.batch_size, 32);
        assert!((cfg.learning_rate - 0.001).abs() < 1e-6);
        assert_eq!(cfg.log_interval, 10);
    }

    #[test]
    fn train_config_new() {
        let cfg = TrainConfig::new();
        assert_eq!(cfg.epochs, 100);
    }

    #[test]
    fn train_config_with_epochs() {
        let cfg = TrainConfig::new().with_epochs(50);
        assert_eq!(cfg.epochs, 50);
    }

    #[test]
    fn train_config_with_learning_rate() {
        let cfg = TrainConfig::new().with_learning_rate(0.01);
        assert!((cfg.learning_rate - 0.01).abs() < 1e-6);
    }

    #[test]
    fn train_config_builder_chain() {
        let cfg = TrainConfig::new().with_epochs(10).with_learning_rate(0.1);
        assert_eq!(cfg.epochs, 10);
        assert!((cfg.learning_rate - 0.1).abs() < 1e-6);
    }

    #[test]
    fn train_config_clone() {
        let cfg = TrainConfig::new();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.epochs, cfg2.epochs);
    }

    #[test]
    fn train_config_debug() {
        let cfg = TrainConfig::new();
        let s = format!("{cfg:?}");
        assert!(s.contains("epochs"));
    }

    // ---- EpochResult ----

    #[test]
    fn epoch_result_fields() {
        let r = EpochResult::new(5, 0.42);
        assert_eq!(r.epoch, 5);
        assert!((r.avg_loss - 0.42).abs() < 1e-6);
    }

    #[test]
    fn epoch_result_clone() {
        let r = EpochResult::new(1, 0.5);
        let r2 = r;
        assert_eq!(r2.epoch, 1);
    }

    #[test]
    fn epoch_result_debug() {
        let r = EpochResult::new(0, 1.0);
        let s = format!("{r:?}");
        assert!(s.contains("epoch"));
    }

    // ---- OptimizerConfig ----

    #[test]
    fn optimizer_config_sgd() {
        let cfg = OptimizerConfig::Sgd(SgdConfig::new(0.01));
        let s = format!("{cfg:?}");
        assert!(s.contains("Sgd"));
    }

    #[test]
    fn optimizer_config_adam() {
        let cfg = OptimizerConfig::Adam(AdamConfig::new(0.001));
        let s = format!("{cfg:?}");
        assert!(s.contains("Adam"));
    }

    #[test]
    fn optimizer_config_clone() {
        let cfg = OptimizerConfig::Sgd(SgdConfig::new(0.01));
        let _cfg2 = cfg.clone();
    }

    // ---- Trainer ----

    #[test]
    fn trainer_loss_decreases_mse() {
        let mut net = SimpleLinear::new(1, 1);
        let config = TrainConfig {
            epochs: 50,
            batch_size: 1,
            learning_rate: 0.01,
            log_interval: 10,
        };
        let trainer = Trainer::new(config);
        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![vec![2.0], vec![4.0], vec![6.0]];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        assert!(!results.is_empty());
        let first_loss = results[0].avg_loss;
        let last_loss = results.last().unwrap().avg_loss;
        assert!(last_loss < first_loss);
    }

    #[test]
    fn trainer_loss_decreases_mae() {
        let mut net = SimpleLinear::new(1, 1);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(50).with_learning_rate(0.01));
        let inputs = vec![vec![1.0], vec![2.0]];
        let targets = vec![vec![2.0], vec![4.0]];
        let results = trainer.train(&mut net, &inputs, &targets, mae_loss);
        let first = results[0].avg_loss;
        let last = results.last().unwrap().avg_loss;
        assert!(last < first, "MAE loss should decrease");
    }

    #[test]
    fn trainer_converges() {
        let mut net = SimpleLinear::new(2, 1);
        let trainer = Trainer::new(TrainConfig {
            epochs: 200,
            batch_size: 1,
            learning_rate: 0.01,
            log_interval: 50,
        });
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let targets = vec![vec![1.0], vec![1.0], vec![2.0]];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        let last_loss = results.last().unwrap().avg_loss;
        assert!(last_loss < 0.1);
    }

    #[test]
    fn trainer_single_sample() {
        let mut net = SimpleLinear::new(1, 1);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(10));
        let inputs = vec![vec![1.0]];
        let targets = vec![vec![2.0]];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn trainer_multiple_outputs() {
        let mut net = SimpleLinear::new(2, 3);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(50).with_learning_rate(0.01));
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let targets = vec![vec![1.0, 0.0, -1.0], vec![0.0, 1.0, 0.0]];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        let first = results[0].avg_loss;
        let last = results.last().unwrap().avg_loss;
        assert!(last < first);
    }

    #[test]
    fn trainer_zero_epochs() {
        let mut net = SimpleLinear::new(1, 1);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(0));
        let results = trainer.train(&mut net, &[vec![1.0]], &[vec![2.0]], mse_loss);
        assert!(results.is_empty());
    }

    #[test]
    fn trainer_epoch_results_count() {
        let mut net = SimpleLinear::new(1, 1);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(7));
        let results = trainer.train(&mut net, &[vec![1.0]], &[vec![2.0]], mse_loss);
        assert_eq!(results.len(), 7);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.epoch, i);
        }
    }

    #[test]
    fn trainer_results_have_finite_loss() {
        let mut net = SimpleLinear::new(2, 1);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(20).with_learning_rate(0.001));
        let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let targets = vec![vec![5.0], vec![11.0]];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        for r in &results {
            assert!(
                r.avg_loss.is_finite(),
                "loss at epoch {} is not finite",
                r.epoch
            );
        }
    }

    #[test]
    fn trainer_identity_learn() {
        // y = x の学習（初期重みがずれている場合）
        let mut net = SimpleLinear::with_weights(1, 1, vec![0.0]);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(100).with_learning_rate(0.1));
        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![vec![1.0], vec![2.0], vec![3.0]];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        let last = results.last().unwrap().avg_loss;
        assert!(last < 0.01, "should learn identity: last_loss={last}");
    }

    #[test]
    fn trainer_constant_target() {
        // 全ターゲットが同じ定数
        let mut net = SimpleLinear::new(2, 1);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(100).with_learning_rate(0.01));
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let targets = vec![vec![5.0], vec![5.0], vec![5.0]];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        let last = results.last().unwrap().avg_loss;
        assert!(last < 0.5, "should learn constant: last_loss={last}");
    }

    #[test]
    fn trainer_cross_entropy() {
        // 2クラス分類
        let mut net = SimpleLinear::new(2, 2);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(50).with_learning_rate(0.01));
        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let results = trainer.train(&mut net, &inputs, &targets, cross_entropy_loss);
        let first = results[0].avg_loss;
        let last = results.last().unwrap().avg_loss;
        assert!(last < first, "cross-entropy loss should decrease");
    }

    #[test]
    fn trainer_large_learning_rate_still_finite() {
        let mut net = SimpleLinear::new(1, 1);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(5).with_learning_rate(1.0));
        let results = trainer.train(&mut net, &[vec![1.0]], &[vec![2.0]], mse_loss);
        for r in &results {
            assert!(r.avg_loss.is_finite());
        }
    }

    // ---- パニックテスト ----

    #[test]
    #[should_panic(expected = "assertion")]
    fn trainer_panics_on_empty_data() {
        let mut net = SimpleLinear::new(1, 1);
        let trainer = Trainer::new(TrainConfig::new());
        let empty: Vec<Vec<f32>> = vec![];
        trainer.train(&mut net, &empty, &empty, mse_loss);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn trainer_panics_on_length_mismatch() {
        let mut net = SimpleLinear::new(1, 1);
        let trainer = Trainer::new(TrainConfig::new());
        trainer.train(&mut net, &[vec![1.0], vec![2.0]], &[vec![1.0]], mse_loss);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn trainer_panics_on_wrong_input_dim() {
        let mut net = SimpleLinear::new(2, 1);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(1));
        trainer.train(&mut net, &[vec![1.0]], &[vec![2.0]], mse_loss); // in_size=2だが1
    }
}
