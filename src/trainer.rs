//! 学習ループ — forward → loss → backward → optimizer step。
//!
//! `TrainableNetwork` トレイトを実装すれば任意のネットワーク構造を学習可能。
//!
//! # 学習メソッド
//!
//! | メソッド | 用途 |
//! |---------|------|
//! | [`Trainer::train`] | 固定 LR、`Vec<f32>` データ |
//! | [`Trainer::train_with_scheduler`] | LR スケジューラ統合 |
//! | [`Trainer::train_tokens`] | `DataLoader` + `MmapDataset` ベース |

use alice_ml::training::{AdamConfig, LossResult, SgdConfig};

use crate::checkpoint::CheckpointData;
use crate::dataloader::{DataLoader, MmapDataset};
use crate::logger::{LogEntry, TrainLog};
use crate::scheduler::LrScheduler;

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
    /// 勾配累積ステップ数。
    /// micro-batch を `gradient_accumulation_steps` 回累積してから
    /// パラメータ更新する。実効バッチ = `batch_size * gradient_accumulation_steps`。
    pub gradient_accumulation_steps: usize,
    /// チェックポイント保存間隔（エポック単位）。`None` = 保存しない。
    pub checkpoint_interval: Option<usize>,
    /// チェックポイント保存先ディレクトリ。
    pub checkpoint_dir: Option<String>,
}

impl TrainConfig {
    /// デフォルト設定 (epochs=100, batch=32, lr=0.001, log=10, grad_accum=1)。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            log_interval: 10,
            gradient_accumulation_steps: 1,
            checkpoint_interval: None,
            checkpoint_dir: None,
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

    /// 勾配累積ステップ数を指定して設定を作成。
    #[must_use]
    pub const fn with_gradient_accumulation(mut self, steps: usize) -> Self {
        self.gradient_accumulation_steps = if steps == 0 { 1 } else { steps };
        self
    }

    /// チェックポイント保存間隔を設定。
    #[must_use]
    pub fn with_checkpoint(mut self, interval: usize, dir: &str) -> Self {
        self.checkpoint_interval = Some(interval);
        self.checkpoint_dir = Some(dir.to_owned());
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

        let accum_steps = self.config.gradient_accumulation_steps.max(1);

        for epoch in 0..self.config.epochs {
            let mut total_loss = 0.0_f32;
            let mut micro_count = 0_usize;

            network.zero_grad();

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

                micro_count += 1;

                // 累積ステップ到達 or エポック最終サンプルで更新
                if micro_count >= accum_steps || sample_idx == n_samples - 1 {
                    network.update_params(self.config.learning_rate);
                    network.zero_grad();
                    micro_count = 0;
                }
            }

            let avg_loss = total_loss / n_samples as f32;
            results.push(EpochResult { epoch, avg_loss });
        }

        results
    }

    /// スケジューラ統合学習ループ。
    ///
    /// ステップ毎に `scheduler.get_lr(step)` で学習率を取得する。
    /// チェックポイント保存、ログ記録に対応。
    ///
    /// # 引数
    ///
    /// - `network` — 学習対象
    /// - `inputs` / `targets` — 学習データ
    /// - `loss_fn` — loss 関数
    /// - `scheduler` — 学習率スケジューラ
    /// - `weight_extractor` — ネットワークから重みスライスを取り出すコールバック
    ///   （チェックポイント保存時に使用。`None` ならチェックポイント無効）
    ///
    /// # Returns
    ///
    /// `(Vec<EpochResult>, TrainLog)`
    pub fn train_with_scheduler<N, L, S>(
        &self,
        network: &mut N,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        loss_fn: L,
        scheduler: &S,
        weight_extractor: Option<&dyn Fn(&N) -> Vec<f32>>,
    ) -> (Vec<EpochResult>, TrainLog)
    where
        N: TrainableNetwork,
        L: Fn(&[f32], &[f32], &mut [f32]) -> LossResult,
        S: LrScheduler,
    {
        assert_eq!(inputs.len(), targets.len());
        assert!(!inputs.is_empty());

        let out_size = network.output_size();
        let in_size = network.input_size();
        let n_samples = inputs.len();
        let accum_steps = self.config.gradient_accumulation_steps.max(1);

        let mut results = Vec::with_capacity(self.config.epochs);
        let mut log = TrainLog::with_capacity(self.config.epochs * n_samples);
        let mut global_step = 0_usize;

        for epoch in 0..self.config.epochs {
            let mut total_loss = 0.0_f32;
            let mut micro_count = 0_usize;

            network.zero_grad();

            for sample_idx in 0..n_samples {
                let lr = scheduler.get_lr(global_step);

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

                log.append(LogEntry::new(epoch, global_step, loss.value, lr, 0.0));

                micro_count += 1;
                global_step += 1;

                if micro_count >= accum_steps || sample_idx == n_samples - 1 {
                    network.update_params(lr);
                    network.zero_grad();
                    micro_count = 0;
                }
            }

            let avg_loss = total_loss / n_samples as f32;
            results.push(EpochResult { epoch, avg_loss });

            // チェックポイント保存
            if let (Some(interval), Some(dir)) =
                (self.config.checkpoint_interval, &self.config.checkpoint_dir)
            {
                if interval > 0 && (epoch + 1) % interval == 0 {
                    if let Some(extractor) = weight_extractor {
                        let weights = extractor(network);
                        let lr = scheduler.get_lr(global_step);
                        let ckpt =
                            CheckpointData::new(epoch, global_step, avg_loss, lr, weights, vec![]);
                        let _ = std::fs::create_dir_all(dir);
                        let path = std::path::Path::new(dir).join(format!("epoch_{epoch:04}.ckpt"));
                        let _ = ckpt.save_to_file(path);
                    }
                }
            }
        }

        (results, log)
    }

    /// `DataLoader` + `MmapDataset` ベースの学習ループ。
    ///
    /// トークンバッチを `token_embed_fn` で f32 に変換し、学習する。
    ///
    /// # 引数
    ///
    /// - `network` — 学習対象
    /// - `dataset` — メモリマップ済みデータセット
    /// - `loader` — データローダー（エポック毎にシャッフル）
    /// - `loss_fn` — loss 関数
    /// - `scheduler` — LR スケジューラ
    /// - `token_embed_fn` — `&[u32] (token_ids) → Vec<f32>` 変換
    /// - `target_embed_fn` — `&[u32] (target_ids) → Vec<f32>` 変換
    ///
    /// # Returns
    ///
    /// `(Vec<EpochResult>, TrainLog)`
    pub fn train_tokens<N, L, S, E, T>(
        &self,
        network: &mut N,
        dataset: &MmapDataset,
        loader: &mut DataLoader,
        loss_fn: L,
        scheduler: &S,
        token_embed_fn: E,
        target_embed_fn: T,
    ) -> (Vec<EpochResult>, TrainLog)
    where
        N: TrainableNetwork,
        L: Fn(&[f32], &[f32], &mut [f32]) -> LossResult,
        S: LrScheduler,
        E: Fn(&[u32]) -> Vec<f32>,
        T: Fn(&[u32]) -> Vec<f32>,
    {
        let out_size = network.output_size();
        let in_size = network.input_size();
        let accum_steps = self.config.gradient_accumulation_steps.max(1);

        let mut results = Vec::with_capacity(self.config.epochs);
        let mut log = TrainLog::with_capacity(self.config.epochs * loader.num_batches());
        let mut global_step = 0_usize;

        for epoch in 0..self.config.epochs {
            loader.shuffle_epoch();
            let mut total_loss = 0.0_f32;
            let mut total_samples = 0_usize;
            let mut micro_count = 0_usize;

            network.zero_grad();

            let n_batches = loader.num_batches();
            for batch_idx in 0..n_batches {
                let batch = loader.get_batch(batch_idx, dataset);
                let seq_len = batch.input_ids.len() / batch.actual_batch_size;

                for b in 0..batch.actual_batch_size {
                    let lr = scheduler.get_lr(global_step);
                    let offset = b * seq_len;
                    let input_tokens = &batch.input_ids[offset..offset + seq_len];
                    let target_tokens = &batch.target_ids[offset..offset + seq_len];

                    let input_f32 = token_embed_fn(input_tokens);
                    let target_f32 = target_embed_fn(target_tokens);

                    assert_eq!(input_f32.len(), in_size);
                    assert_eq!(target_f32.len(), out_size);

                    let mut output = vec![0.0_f32; out_size];
                    network.forward(&input_f32, &mut output);

                    let mut grad_output = vec![0.0_f32; out_size];
                    let loss = loss_fn(&output, &target_f32, &mut grad_output);
                    total_loss += loss.value;
                    total_samples += 1;

                    let mut grad_input = vec![0.0_f32; in_size];
                    network.backward(&input_f32, &grad_output, &mut grad_input);

                    log.append(LogEntry::new(epoch, global_step, loss.value, lr, 0.0));

                    micro_count += 1;
                    global_step += 1;

                    if micro_count >= accum_steps {
                        network.update_params(lr);
                        network.zero_grad();
                        micro_count = 0;
                    }
                }
            }

            // エポック末残りの勾配をフラッシュ
            if micro_count > 0 {
                let lr = scheduler.get_lr(global_step.saturating_sub(1));
                network.update_params(lr);
                network.zero_grad();
            }

            let avg_loss = if total_samples > 0 {
                total_loss / total_samples as f32
            } else {
                0.0
            };
            results.push(EpochResult { epoch, avg_loss });
        }

        (results, log)
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
            gradient_accumulation_steps: 1,
            checkpoint_interval: None,
            checkpoint_dir: None,
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
        let trainer = Trainer::new(TrainConfig::new().with_epochs(200).with_learning_rate(0.01));
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

    // ---- 追加テスト ----

    #[test]
    fn train_config_batch_size_field() {
        // batch_size フィールドが正しくアクセスできることを検証
        let cfg = TrainConfig {
            epochs: 10,
            batch_size: 64,
            learning_rate: 0.01,
            log_interval: 5,
            gradient_accumulation_steps: 1,
            checkpoint_interval: None,
            checkpoint_dir: None,
        };
        assert_eq!(cfg.batch_size, 64);
        assert_eq!(cfg.log_interval, 5);
    }

    #[test]
    fn trainer_negative_target() {
        // 負のターゲット値でも収束することを検証
        let mut net = SimpleLinear::with_weights(1, 1, vec![0.0]);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(100).with_learning_rate(0.1));
        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![vec![-1.0], vec![-2.0], vec![-3.0]];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        let last = results.last().unwrap().avg_loss;
        assert!(
            last < 0.01,
            "should learn negative targets: last_loss={last}"
        );
    }

    #[test]
    fn trainer_high_dimensional() {
        // 高次元 (4入力, 3出力) で loss が減少することを検証
        let mut net = SimpleLinear::new(4, 3);
        let trainer = Trainer::new(
            TrainConfig::new()
                .with_epochs(100)
                .with_learning_rate(0.001),
        );
        let inputs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let targets = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        let first = results[0].avg_loss;
        let last = results.last().unwrap().avg_loss;
        assert!(
            last < first,
            "high-dim loss should decrease: first={first}, last={last}"
        );
    }

    // ---- 勾配累積テスト ----

    #[test]
    fn train_config_with_gradient_accumulation() {
        let cfg = TrainConfig::new().with_gradient_accumulation(4);
        assert_eq!(cfg.gradient_accumulation_steps, 4);
    }

    #[test]
    fn train_config_grad_accum_zero_becomes_one() {
        let cfg = TrainConfig::new().with_gradient_accumulation(0);
        assert_eq!(cfg.gradient_accumulation_steps, 1);
    }

    #[test]
    fn trainer_grad_accum_loss_decreases() {
        // 勾配累積ありでも loss が減少することを検証
        let mut net = SimpleLinear::new(2, 1);
        let trainer = Trainer::new(
            TrainConfig::new()
                .with_epochs(100)
                .with_learning_rate(0.01)
                .with_gradient_accumulation(3),
        );
        let inputs = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
            vec![0.5, 0.5],
        ];
        let targets = vec![
            vec![1.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![3.0],
            vec![1.0],
        ];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        let first = results[0].avg_loss;
        let last = results.last().unwrap().avg_loss;
        assert!(
            last < first,
            "grad accum loss should decrease: first={first}, last={last}"
        );
    }

    #[test]
    fn trainer_grad_accum_one_equals_no_accum() {
        // accum=1 は従来と同じ挙動
        let mut net1 = SimpleLinear::with_weights(1, 1, vec![0.5]);
        let mut net2 = SimpleLinear::with_weights(1, 1, vec![0.5]);

        let cfg1 = TrainConfig::new()
            .with_epochs(10)
            .with_learning_rate(0.01)
            .with_gradient_accumulation(1);
        let cfg2 = TrainConfig::new().with_epochs(10).with_learning_rate(0.01); // default accum=1

        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![vec![2.0], vec![4.0], vec![6.0]];

        let r1 = Trainer::new(cfg1).train(&mut net1, &inputs, &targets, mse_loss);
        let r2 = Trainer::new(cfg2).train(&mut net2, &inputs, &targets, mse_loss);

        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!(
                (a.avg_loss - b.avg_loss).abs() < 1e-6,
                "accum=1 should match default: {} vs {}",
                a.avg_loss,
                b.avg_loss
            );
        }
    }

    #[test]
    fn trainer_grad_accum_large_step() {
        // accum > n_samples の場合、エポック末で1回だけ更新
        let mut net = SimpleLinear::with_weights(1, 1, vec![0.0]);
        let trainer = Trainer::new(
            TrainConfig::new()
                .with_epochs(50)
                .with_learning_rate(0.1)
                .with_gradient_accumulation(100), // サンプル数より大きい
        );
        let inputs = vec![vec![1.0], vec![2.0]];
        let targets = vec![vec![1.0], vec![2.0]];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        let first = results[0].avg_loss;
        let last = results.last().unwrap().avg_loss;
        assert!(
            last < first,
            "large accum should still learn: first={first}, last={last}"
        );
    }

    #[test]
    fn trainer_first_epoch_loss_positive() {
        // 初期状態で target と出力が異なるため、最初のエポックの loss は正であること
        let mut net = SimpleLinear::with_weights(1, 1, vec![0.0]);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(1));
        let inputs = vec![vec![1.0]];
        let targets = vec![vec![5.0]];
        let results = trainer.train(&mut net, &inputs, &targets, mse_loss);
        assert!(
            results[0].avg_loss > 0.0,
            "first epoch loss should be positive: {}",
            results[0].avg_loss
        );
    }

    // ---- Phase 1.5: 結合テスト ----

    #[test]
    fn train_config_with_checkpoint() {
        let cfg = TrainConfig::new().with_checkpoint(5, "/tmp/ckpt");
        assert_eq!(cfg.checkpoint_interval, Some(5));
        assert_eq!(cfg.checkpoint_dir.as_deref(), Some("/tmp/ckpt"));
    }

    #[test]
    fn train_with_scheduler_loss_decreases() {
        use crate::scheduler::WarmupCosineScheduler;

        let mut net = SimpleLinear::new(2, 1);
        let scheduler = WarmupCosineScheduler::new(0.01, 0.0001, 10, 200);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(50));

        let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let targets = vec![vec![1.0], vec![1.0], vec![2.0]];

        let (results, log) =
            trainer.train_with_scheduler(&mut net, &inputs, &targets, mse_loss, &scheduler, None);

        let first = results[0].avg_loss;
        let last = results.last().unwrap().avg_loss;
        assert!(
            last < first,
            "scheduler train loss should decrease: first={first}, last={last}"
        );
        assert!(!log.is_empty());
    }

    #[test]
    fn train_with_scheduler_log_records_lr() {
        use crate::scheduler::ConstantScheduler;

        let mut net = SimpleLinear::new(1, 1);
        let scheduler = ConstantScheduler::new(0.042);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(2));

        let inputs = vec![vec![1.0]];
        let targets = vec![vec![2.0]];

        let (_results, log) =
            trainer.train_with_scheduler(&mut net, &inputs, &targets, mse_loss, &scheduler, None);

        for entry in log.entries() {
            assert!(
                (entry.learning_rate - 0.042).abs() < 1e-6,
                "lr should be 0.042, got {}",
                entry.learning_rate
            );
        }
    }

    #[test]
    fn train_with_scheduler_grad_accum() {
        use crate::scheduler::ConstantScheduler;

        let mut net = SimpleLinear::new(1, 1);
        let scheduler = ConstantScheduler::new(0.01);
        let trainer = Trainer::new(
            TrainConfig::new()
                .with_epochs(50)
                .with_gradient_accumulation(2),
        );

        let inputs = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let targets = vec![vec![2.0], vec![4.0], vec![6.0], vec![8.0]];

        let (results, _log) =
            trainer.train_with_scheduler(&mut net, &inputs, &targets, mse_loss, &scheduler, None);

        let first = results[0].avg_loss;
        let last = results.last().unwrap().avg_loss;
        assert!(last < first);
    }

    #[test]
    fn train_with_scheduler_saves_checkpoints() {
        use crate::scheduler::ConstantScheduler;

        let dir = tempfile::tempdir().unwrap();
        let dir_str = dir.path().to_str().unwrap();

        let mut net = SimpleLinear::with_weights(1, 1, vec![0.5]);
        let scheduler = ConstantScheduler::new(0.01);
        let trainer = Trainer::new(
            TrainConfig::new()
                .with_epochs(10)
                .with_checkpoint(5, dir_str),
        );

        let inputs = vec![vec![1.0]];
        let targets = vec![vec![2.0]];

        let extractor = |n: &SimpleLinear| n.weights.clone();
        trainer.train_with_scheduler(
            &mut net,
            &inputs,
            &targets,
            mse_loss,
            &scheduler,
            Some(&extractor),
        );

        // epoch 4 (0-indexed, (4+1) % 5 == 0) → epoch_0004.ckpt
        let ckpt_path = dir.path().join("epoch_0004.ckpt");
        assert!(ckpt_path.exists(), "checkpoint at epoch 4 should exist");

        let ckpt_path2 = dir.path().join("epoch_0009.ckpt");
        assert!(ckpt_path2.exists(), "checkpoint at epoch 9 should exist");
    }

    #[test]
    fn train_tokens_basic() {
        use crate::dataloader::DataLoaderConfig;
        use crate::scheduler::ConstantScheduler;
        use std::io::Write as IoWrite;

        let dir = tempfile::tempdir().unwrap();
        let token_path = dir.path().join("tokens.bin");

        // 100 トークンのテストデータ
        {
            let mut f = std::fs::File::create(&token_path).unwrap();
            for i in 0u32..100 {
                f.write_all(&i.to_le_bytes()).unwrap();
            }
        }

        let dataset = crate::dataloader::MmapDataset::open(&token_path).unwrap();
        let config = DataLoaderConfig::new()
            .with_seq_len(4)
            .with_batch_size(2)
            .with_shuffle(false);
        let mut loader = crate::dataloader::DataLoader::new(&dataset, config);

        // SimpleLinear: in=4, out=4
        let mut net = SimpleLinear::new(4, 4);
        let scheduler = ConstantScheduler::new(0.01);
        let trainer = Trainer::new(TrainConfig::new().with_epochs(5));

        // トークン ID → f32 変換（簡易: ID / 100.0）
        let embed =
            |tokens: &[u32]| -> Vec<f32> { tokens.iter().map(|&t| t as f32 * 0.01).collect() };

        let (results, log) = trainer.train_tokens(
            &mut net,
            &dataset,
            &mut loader,
            mse_loss,
            &scheduler,
            &embed,
            &embed,
        );

        assert_eq!(results.len(), 5);
        assert!(!log.is_empty());
        // loss が有限であること
        for r in &results {
            assert!(
                r.avg_loss.is_finite(),
                "loss at epoch {} is not finite",
                r.epoch
            );
        }
    }
}
