//! QAT パイプライン — FP16/FP32 → BF16 mixed precision → Ternary 統合ループ。
//!
//! ALICE-Train の全コンポーネントをオーケストレーションする最上位モジュール。
//!
//! # フロー
//!
//! ```text
//! FP32 latent weights
//!   ↓ (BF16 precision simulation if enabled)
//! FakeQuantize → ternary {-1, 0, +1} × scale
//!   ↓ forward
//! output → loss
//!   ↓ LossScaler.scale()
//! scaled gradients → STE backward → unscale
//!   ↓ NaN/Inf check
//! weight gradients → gradient accumulation → SGD update (latent FP32)
//!   ↓ temperature annealing (per epoch)
//! evaluate → BestCheckpointTracker → repeat
//! ```
//!
//! # 構成
//!
//! | コンポーネント | 役割 |
//! |--------------|------|
//! | [`FakeQuantize`] | STE ベースの fake quantization |
//! | [`LossScaler`] | 動的 loss scaling（NaN/Inf 防止） |
//! | [`WarmupCosineScheduler`] | 学習率スケジューリング |
//! | [`TrainLog`] | ステップ毎のメトリクス記録 |
//! | [`CalibrationStats`] | 量子化品質の追跡 |
//! | [`CheckpointData`] | 重み保存/復元 |
//!
//! [`FakeQuantize`]: crate::qat::FakeQuantize
//! [`LossScaler`]: crate::mixed_precision::LossScaler
//! [`WarmupCosineScheduler`]: crate::scheduler::WarmupCosineScheduler
//! [`TrainLog`]: crate::logger::TrainLog
//! [`CalibrationStats`]: crate::qat::CalibrationStats
//! [`CheckpointData`]: crate::checkpoint::CheckpointData

use crate::checkpoint::CheckpointData;
use crate::logger::{compute_grad_norm, LogEntry, TrainLog};
use crate::mixed_precision::{bf16_to_f32_vec, f32_to_bf16_vec, LossScaler, MixedPrecisionConfig};
use crate::qat::{CalibrationStats, FakeQuantize, QatConfig};
use crate::scheduler::{LrScheduler, WarmupCosineScheduler};

/// QAT パイプライン設定。
#[derive(Clone, Debug)]
pub struct QatPipelineConfig {
    /// QAT 量子化設定。
    pub qat: QatConfig,
    /// 混合精度設定。
    pub mixed_precision: MixedPrecisionConfig,
    /// エポック数。
    pub epochs: usize,
    /// 初期学習率（スケジューラのピーク）。
    pub learning_rate: f32,
    /// 最小学習率（cosine decay の下限）。
    pub min_lr: f32,
    /// Warmup ステップ数。
    pub warmup_steps: usize,
    /// 全ステップ数（warmup 含む）。0 = `epochs × data_len` から自動算出。
    pub total_steps: usize,
    /// 勾配累積ステップ数。
    pub gradient_accumulation_steps: usize,
    /// 評価間隔（エポック単位）。0 = 評価なし。
    pub eval_interval: usize,
    /// チェックポイント保存間隔（エポック単位）。0 = 保存なし。
    pub checkpoint_interval: usize,
    /// チェックポイント保存先ディレクトリ。
    pub checkpoint_dir: Option<String>,
}

impl Default for QatPipelineConfig {
    fn default() -> Self {
        Self {
            qat: QatConfig::ternary(),
            mixed_precision: MixedPrecisionConfig::default(),
            epochs: 10,
            learning_rate: 1e-4,
            min_lr: 1e-6,
            warmup_steps: 100,
            total_steps: 0,
            gradient_accumulation_steps: 1,
            eval_interval: 1,
            checkpoint_interval: 0,
            checkpoint_dir: None,
        }
    }
}

impl QatPipelineConfig {
    /// Ternary QAT のデフォルト設定を返す。
    #[must_use]
    pub fn ternary() -> Self {
        Self::default()
    }

    /// チェックポイント設定を追加する。
    #[must_use]
    pub fn with_checkpoint(mut self, interval: usize, dir: &str) -> Self {
        self.checkpoint_interval = interval;
        self.checkpoint_dir = Some(dir.to_owned());
        self
    }
}

/// 1ステップの QAT 結果。
#[derive(Clone, Debug)]
pub struct QatStepResult {
    /// 損失値（unscaled）。
    pub loss: f32,
    /// 量子化 MAE。
    pub quant_mae: f32,
    /// 量子化前後のコサイン類似度。
    pub cosine_sim: f32,
    /// 勾配ノルム（L2）。
    pub grad_norm: f32,
    /// 現在の学習率。
    pub learning_rate: f32,
    /// 現在の loss scale。
    pub loss_scale: f32,
    /// 勾配が有効（NaN/Inf なし）だったか。
    pub gradients_valid: bool,
}

/// エポック集計結果。
#[derive(Clone, Debug)]
pub struct QatEpochSummary {
    /// エポック番号。
    pub epoch: usize,
    /// 平均損失。
    pub avg_loss: f32,
    /// 平均量子化 MAE。
    pub avg_quant_mae: f32,
    /// 平均コサイン類似度。
    pub avg_cosine_sim: f32,
    /// エポック末の temperature。
    pub temperature: f32,
    /// エポック末の scale factor。
    pub scale: f32,
    /// 評価損失（評価実行時のみ）。
    pub eval_loss: Option<f32>,
    /// 評価 perplexity（評価実行時のみ）。
    pub eval_perplexity: Option<f32>,
}

/// QAT パイプライン全体の実行結果。
#[derive(Clone, Debug)]
pub struct QatRunResult {
    /// 各エポックの集計結果。
    pub epoch_summaries: Vec<QatEpochSummary>,
    /// 学習ログ。
    pub log: TrainLog,
    /// 最終 temperature。
    pub final_temperature: f32,
    /// 最終 scale factor。
    pub final_scale: f32,
    /// ベスト損失。
    pub best_loss: f32,
    /// ベスト損失のエポック。
    pub best_epoch: usize,
    /// 総ステップ数。
    pub total_steps: usize,
    /// NaN/Inf でスキップされたステップ数。
    pub skipped_steps: usize,
}

/// QAT パイプライン — 全コンポーネントのオーケストレーター。
///
/// FP32 latent weights を BF16 mixed precision で学習しながら、
/// `FakeQuantize` で ternary {-1, 0, +1} へ押し潰す。
pub struct QatPipeline {
    /// パイプライン設定。
    config: QatPipelineConfig,
    /// Fake quantizer。
    fq: FakeQuantize,
    /// 動的 loss scaler。
    scaler: LossScaler,
    /// 学習ログ。
    log: TrainLog,
    /// Calibration 統計。
    stats: CalibrationStats,
    /// グローバルステップ数。
    global_step: usize,
    /// ベスト損失。
    best_loss: f32,
    /// ベスト損失のエポック。
    best_epoch: usize,
    /// スキップされたステップ数。
    skipped_steps: usize,
}

impl QatPipeline {
    /// 新しいパイプラインを構築する。
    #[must_use]
    pub fn new(config: QatPipelineConfig) -> Self {
        let fq = FakeQuantize::new(config.qat.clone());
        let scaler = LossScaler::new(config.mixed_precision.clone());
        Self {
            fq,
            scaler,
            log: TrainLog::new(),
            stats: CalibrationStats::new(),
            global_step: 0,
            best_loss: f32::MAX,
            best_epoch: 0,
            skipped_steps: 0,
            config,
        }
    }

    /// 現在の `FakeQuantize` への参照を返す。
    #[must_use]
    pub fn fake_quantize(&self) -> &FakeQuantize {
        &self.fq
    }

    /// 現在の `LossScaler` への参照を返す。
    #[must_use]
    pub fn loss_scaler(&self) -> &LossScaler {
        &self.scaler
    }

    /// 学習ログへの参照を返す。
    #[must_use]
    pub fn train_log(&self) -> &TrainLog {
        &self.log
    }

    /// Calibration 統計への参照を返す。
    #[must_use]
    pub fn calibration_stats(&self) -> &CalibrationStats {
        &self.stats
    }

    /// 現在のグローバルステップ数を返す。
    #[must_use]
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    /// 潜在重みを最終 ternary 重みに変換する。
    ///
    /// FakeQuantize を適用し、{-scale, 0, +scale} の離散値に確定させる。
    pub fn finalize_weights(&mut self, latent_weights: &[f32], output: &mut [f32]) {
        assert_eq!(latent_weights.len(), output.len());
        self.fq.calibrate_scale(latent_weights);
        self.fq.fake_quantize_forward(latent_weights, output);
    }

    /// 全データに対して QAT ループを実行する。
    ///
    /// # 引数
    ///
    /// - `latent_weights` — 潜在 FP32 重み（更新対象）
    /// - `data` — `(input, target)` ペアのスライス
    /// - `forward_fn` — forward 関数 `(quantized_weights, input, output)`
    /// - `loss_fn` — loss 関数 `(output, target, grad_output) -> loss`
    /// - `eval_data` — 評価用データ（`None` = 評価なし）
    ///
    /// # Panics
    ///
    /// `data` が空の場合。
    pub fn run(
        &mut self,
        latent_weights: &mut [f32],
        data: &[(Vec<f32>, Vec<f32>)],
        forward_fn: &dyn Fn(&[f32], &[f32], &mut [f32]),
        loss_fn: &dyn Fn(&[f32], &[f32], &mut [f32]) -> f32,
        eval_data: Option<&[(Vec<f32>, Vec<f32>)]>,
    ) -> QatRunResult {
        assert!(!data.is_empty(), "training data must not be empty");

        let n_samples = data.len();
        let total_steps = if self.config.total_steps > 0 {
            self.config.total_steps
        } else {
            self.config.epochs * n_samples
        };

        let scheduler = WarmupCosineScheduler::new(
            self.config.learning_rate,
            self.config.min_lr,
            self.config.warmup_steps.min(total_steps),
            total_steps,
        );

        let out_size = data[0].1.len();
        let weight_count = latent_weights.len();
        let accum_steps = self.config.gradient_accumulation_steps.max(1);

        // ワークバッファ
        let mut quantized_buf = vec![0.0f32; weight_count];
        let mut output_buf = vec![0.0f32; out_size];
        let mut grad_output_buf = vec![0.0f32; out_size];
        let mut grad_weight_buf = vec![0.0f32; weight_count];
        let mut accum_grad = vec![0.0f32; weight_count];

        let mut epoch_summaries = Vec::with_capacity(self.config.epochs);

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0f32;
            let mut micro_count = 0usize;

            accum_grad.fill(0.0);

            for (sample_idx, (input, target)) in data.iter().enumerate() {
                let lr = scheduler.get_lr(self.global_step);

                // 1. Calibrate scale
                self.fq.calibrate_scale(latent_weights);

                // 2. BF16 precision simulation + FakeQuantize
                if self.config.mixed_precision.enabled {
                    let bf16 = f32_to_bf16_vec(latent_weights);
                    let approx = bf16_to_f32_vec(&bf16);
                    self.fq.fake_quantize_forward(&approx, &mut quantized_buf);
                } else {
                    self.fq
                        .fake_quantize_forward(latent_weights, &mut quantized_buf);
                }

                // 3. Forward
                forward_fn(&quantized_buf, input, &mut output_buf);

                // 4. Loss + gradient
                let loss = loss_fn(&output_buf, target, &mut grad_output_buf);

                // 5. Scale gradients (loss scaling)
                let scale = self.scaler.scale();
                for g in &mut grad_output_buf {
                    *g *= scale;
                }

                // 6. STE backward
                let grad_len = grad_weight_buf.len().min(grad_output_buf.len());
                self.fq.ste_backward(
                    &grad_output_buf[..grad_len],
                    &mut grad_weight_buf[..grad_len],
                );

                // 7. Unscale
                self.scaler.unscale_gradients(&mut grad_weight_buf);

                // 8. Accumulate
                for (acc, &g) in accum_grad.iter_mut().zip(grad_weight_buf.iter()) {
                    *acc += g;
                }

                let grad_norm = compute_grad_norm(&grad_weight_buf);

                // ログ記録
                self.log
                    .append(LogEntry::new(epoch, self.global_step, loss, lr, grad_norm));

                epoch_loss += loss;
                self.global_step += 1;
                micro_count += 1;

                // 9. 累積ステップ到達 or エポック末で更新
                if micro_count >= accum_steps || sample_idx == n_samples - 1 {
                    let gradients_valid = LossScaler::check_gradients(&accum_grad);
                    self.scaler.update(gradients_valid);

                    if gradients_valid {
                        let inv_micro = 1.0 / micro_count as f32;
                        for (w, &g) in latent_weights.iter_mut().zip(accum_grad.iter()) {
                            *w -= lr * g * inv_micro;
                        }
                    } else {
                        self.skipped_steps += 1;
                    }

                    accum_grad.fill(0.0);
                    micro_count = 0;
                }
            }

            // Calibration stats 更新
            self.fq.calibrate_scale(latent_weights);
            self.fq
                .fake_quantize_forward(latent_weights, &mut quantized_buf);
            self.stats.update_weights(latent_weights, &quantized_buf);
            let epoch_mae = self.stats.quantization_mae;
            let epoch_cos = self.stats.cosine_similarity;

            // Temperature annealing
            self.fq.step_temperature();

            let avg_loss = epoch_loss / n_samples as f32;

            // 評価
            let (eval_loss, eval_perplexity) =
                if self.config.eval_interval > 0 && (epoch + 1) % self.config.eval_interval == 0 {
                    if let Some(eval) = eval_data {
                        let el = self.evaluate(
                            latent_weights,
                            &mut quantized_buf,
                            &mut output_buf,
                            eval,
                            forward_fn,
                            loss_fn,
                        );
                        let ppl = el.exp();

                        // ベスト更新
                        if el < self.best_loss {
                            self.best_loss = el;
                            self.best_epoch = epoch;
                        }

                        (Some(el), Some(ppl))
                    } else {
                        // eval_data なし → train loss でベスト追跡
                        if avg_loss < self.best_loss {
                            self.best_loss = avg_loss;
                            self.best_epoch = epoch;
                        }
                        (None, None)
                    }
                } else {
                    if avg_loss < self.best_loss {
                        self.best_loss = avg_loss;
                        self.best_epoch = epoch;
                    }
                    (None, None)
                };

            // チェックポイント保存
            if self.config.checkpoint_interval > 0
                && (epoch + 1) % self.config.checkpoint_interval == 0
            {
                if let Some(ref dir) = self.config.checkpoint_dir {
                    let lr = scheduler.get_lr(self.global_step.saturating_sub(1));
                    let ckpt = CheckpointData::new(
                        epoch,
                        self.global_step,
                        avg_loss,
                        lr,
                        latent_weights.to_vec(),
                        vec![],
                    );
                    let _ = std::fs::create_dir_all(dir);
                    let path = std::path::Path::new(dir).join(format!("qat_epoch_{epoch:04}.ckpt"));
                    let _ = ckpt.save_to_file(path);
                }
            }

            epoch_summaries.push(QatEpochSummary {
                epoch,
                avg_loss,
                avg_quant_mae: epoch_mae,
                avg_cosine_sim: epoch_cos,
                temperature: self.fq.temperature(),
                scale: self.fq.scale(),
                eval_loss,
                eval_perplexity,
            });
        }

        QatRunResult {
            epoch_summaries,
            log: self.log.clone(),
            final_temperature: self.fq.temperature(),
            final_scale: self.fq.scale(),
            best_loss: self.best_loss,
            best_epoch: self.best_epoch,
            total_steps: self.global_step,
            skipped_steps: self.skipped_steps,
        }
    }

    /// 評価データに対して平均 loss を計算する。
    fn evaluate(
        &self,
        latent_weights: &[f32],
        quantized_buf: &mut [f32],
        output_buf: &mut [f32],
        eval_data: &[(Vec<f32>, Vec<f32>)],
        forward_fn: &dyn Fn(&[f32], &[f32], &mut [f32]),
        loss_fn: &dyn Fn(&[f32], &[f32], &mut [f32]) -> f32,
    ) -> f32 {
        if eval_data.is_empty() {
            return 0.0;
        }

        // FakeQuantize latent → quantized（BF16 simulation 込み）
        if self.config.mixed_precision.enabled {
            let bf16 = f32_to_bf16_vec(latent_weights);
            let approx = bf16_to_f32_vec(&bf16);
            self.fq.fake_quantize_forward(&approx, quantized_buf);
        } else {
            self.fq.fake_quantize_forward(latent_weights, quantized_buf);
        }

        let mut total_loss = 0.0f32;
        let mut grad_dummy = vec![0.0f32; output_buf.len()];

        for (input, target) in eval_data {
            forward_fn(quantized_buf, input, output_buf);
            total_loss += loss_fn(output_buf, target, &mut grad_dummy);
        }

        total_loss / eval_data.len() as f32
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ヘルパー: y = W * x (element-wise multiply + sum 的な単純モデル)
    fn simple_forward(weights: &[f32], input: &[f32], output: &mut [f32]) {
        for (o, (&w, &x)) in output.iter_mut().zip(weights.iter().zip(input.iter())) {
            *o = w * x;
        }
    }

    // ヘルパー: MSE loss = mean((y - t)^2), grad = 2*(y - t)/n
    fn simple_mse(output: &[f32], target: &[f32], grad: &mut [f32]) -> f32 {
        let n = output.len() as f32;
        let mut loss = 0.0f32;
        for (i, (&y, &t)) in output.iter().zip(target.iter()).enumerate() {
            let diff = y - t;
            loss += diff * diff;
            grad[i] = 2.0 * diff / n;
        }
        loss / n
    }

    fn make_data(n: usize, dim: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        (0..n)
            .map(|i| {
                let input: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.1 + 0.1).collect();
                let target: Vec<f32> = input.iter().map(|&x| x * 2.0).collect();
                (input, target)
            })
            .collect()
    }

    // --- QatPipelineConfig ---

    #[test]
    fn config_default() {
        let c = QatPipelineConfig::default();
        assert_eq!(c.epochs, 10);
        assert!((c.learning_rate - 1e-4).abs() < 1e-8);
        assert_eq!(c.gradient_accumulation_steps, 1);
    }

    #[test]
    fn config_ternary() {
        let c = QatPipelineConfig::ternary();
        assert_eq!(c.epochs, 10);
    }

    #[test]
    fn config_with_checkpoint() {
        let c = QatPipelineConfig::default().with_checkpoint(5, "/tmp/ckpt");
        assert_eq!(c.checkpoint_interval, 5);
        assert_eq!(c.checkpoint_dir.as_deref(), Some("/tmp/ckpt"));
    }

    #[test]
    fn config_clone() {
        let c = QatPipelineConfig::default();
        let c2 = c.clone();
        assert_eq!(c2.epochs, c.epochs);
    }

    #[test]
    fn config_debug() {
        let c = QatPipelineConfig::default();
        let s = format!("{c:?}");
        assert!(s.contains("QatPipelineConfig"));
    }

    // --- QatStepResult ---

    #[test]
    fn step_result_clone_debug() {
        let r = QatStepResult {
            loss: 0.5,
            quant_mae: 0.1,
            cosine_sim: 0.99,
            grad_norm: 1.0,
            learning_rate: 0.001,
            loss_scale: 1024.0,
            gradients_valid: true,
        };
        let r2 = r.clone();
        assert!((r2.loss - 0.5).abs() < 1e-6);
        let s = format!("{r:?}");
        assert!(s.contains("QatStepResult"));
    }

    // --- QatEpochSummary ---

    #[test]
    fn epoch_summary_clone_debug() {
        let s = QatEpochSummary {
            epoch: 0,
            avg_loss: 1.0,
            avg_quant_mae: 0.1,
            avg_cosine_sim: 0.99,
            temperature: 0.9,
            scale: 0.5,
            eval_loss: Some(0.8),
            eval_perplexity: Some(2.23),
        };
        let s2 = s.clone();
        assert_eq!(s2.epoch, 0);
        let d = format!("{s:?}");
        assert!(d.contains("QatEpochSummary"));
    }

    // --- QatRunResult ---

    #[test]
    fn run_result_clone_debug() {
        let r = QatRunResult {
            epoch_summaries: vec![],
            log: TrainLog::new(),
            final_temperature: 0.5,
            final_scale: 0.3,
            best_loss: 0.1,
            best_epoch: 5,
            total_steps: 100,
            skipped_steps: 0,
        };
        let r2 = r.clone();
        assert_eq!(r2.total_steps, 100);
        let d = format!("{r:?}");
        assert!(d.contains("QatRunResult"));
    }

    // --- QatPipeline construction ---

    #[test]
    fn pipeline_new() {
        let p = QatPipeline::new(QatPipelineConfig::default());
        assert_eq!(p.global_step(), 0);
        assert!(p.train_log().is_empty());
        assert_eq!(p.calibration_stats().sample_count, 0);
    }

    #[test]
    fn pipeline_accessors() {
        let p = QatPipeline::new(QatPipelineConfig::default());
        let _fq = p.fake_quantize();
        let _scaler = p.loss_scaler();
        let _log = p.train_log();
        let _stats = p.calibration_stats();
        assert_eq!(p.global_step(), 0);
    }

    // --- finalize_weights ---

    #[test]
    fn finalize_weights_produces_ternary() {
        let mut p = QatPipeline::new(QatPipelineConfig::default());
        let latent = [0.5, -0.3, 0.01, 0.8, -0.9];
        let mut output = [0.0f32; 5];
        p.finalize_weights(&latent, &mut output);

        // 各値は {-scale, 0, +scale} のいずれか
        let scale = p.fake_quantize().scale();
        for &q in &output {
            let normalized = if scale > 1e-10 {
                (q / scale).round()
            } else {
                0.0
            };
            assert!(
                (normalized - (-1.0)).abs() < 0.01
                    || (normalized - 0.0).abs() < 0.01
                    || (normalized - 1.0).abs() < 0.01,
                "finalized value {q} not ternary (scale={scale})"
            );
        }
    }

    // --- run: basic ---

    #[test]
    fn run_single_epoch() {
        let config = QatPipelineConfig {
            epochs: 1,
            learning_rate: 0.1,
            min_lr: 0.01,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 0,
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.5f32; 2];
        let data = make_data(3, 2);

        let result = pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        assert_eq!(result.epoch_summaries.len(), 1);
        assert!(result.total_steps > 0);
        assert!(!result.log.is_empty());
    }

    #[test]
    fn run_loss_decreases() {
        let config = QatPipelineConfig {
            epochs: 50,
            learning_rate: 0.5,
            min_lr: 0.01,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 0,
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.0f32; 2];
        let data = vec![
            (vec![1.0, 1.0], vec![0.5, 0.5]),
            (vec![2.0, 2.0], vec![1.0, 1.0]),
        ];

        let result = pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        let first = result.epoch_summaries[0].avg_loss;
        let last = result.epoch_summaries.last().unwrap().avg_loss;
        assert!(
            last < first,
            "loss should decrease: first={first}, last={last}"
        );
    }

    // --- run: BF16 ---

    #[test]
    fn run_with_bf16_enabled() {
        let config = QatPipelineConfig {
            epochs: 10,
            learning_rate: 0.5,
            min_lr: 0.01,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 0,
            mixed_precision: MixedPrecisionConfig::default(), // enabled
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.5f32; 2];
        let data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];

        let result = pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        assert_eq!(result.epoch_summaries.len(), 10);
        for s in &result.epoch_summaries {
            assert!(s.avg_loss.is_finite());
        }
    }

    // --- run: gradient accumulation ---

    #[test]
    fn run_with_gradient_accumulation() {
        let config = QatPipelineConfig {
            epochs: 30,
            learning_rate: 0.5,
            min_lr: 0.01,
            warmup_steps: 0,
            total_steps: 0,
            gradient_accumulation_steps: 2,
            eval_interval: 0,
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.0f32; 2];
        let data = vec![
            (vec![1.0, 1.0], vec![0.5, 0.5]),
            (vec![2.0, 2.0], vec![1.0, 1.0]),
            (vec![0.5, 0.5], vec![0.25, 0.25]),
            (vec![3.0, 3.0], vec![1.5, 1.5]),
        ];

        let result = pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        let first = result.epoch_summaries[0].avg_loss;
        let last = result.epoch_summaries.last().unwrap().avg_loss;
        assert!(
            last < first,
            "grad accum loss should decrease: first={first}, last={last}"
        );
    }

    // --- run: temperature annealing ---

    #[test]
    fn run_temperature_decreases() {
        let mut qat = QatConfig::ternary();
        qat.temperature = 1.0;
        qat.temperature_decay = 0.9;

        let config = QatPipelineConfig {
            qat,
            epochs: 5,
            learning_rate: 0.01,
            min_lr: 0.001,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 0,
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.5f32; 2];
        let data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];

        let result = pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        // 5 epochs with decay 0.9 → temp ≈ 1.0 * 0.9^5 ≈ 0.59
        assert!(
            result.final_temperature < 1.0,
            "temperature should decrease: {}",
            result.final_temperature
        );
        assert!(
            (result.final_temperature - 0.9f32.powi(5)).abs() < 0.01,
            "temperature should be ~0.59, got {}",
            result.final_temperature
        );
    }

    // --- run: evaluation ---

    #[test]
    fn run_with_eval_data() {
        let config = QatPipelineConfig {
            epochs: 5,
            learning_rate: 0.1,
            min_lr: 0.01,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 2, // evaluate every 2 epochs
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.5f32; 2];
        let data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];
        let eval_data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];

        let result = pipeline.run(
            &mut weights,
            &data,
            &simple_forward,
            &simple_mse,
            Some(&eval_data),
        );

        // eval_interval=2 → epoch 1 (0-indexed, (1+1)%2==0) と epoch 3 に eval
        let has_eval = result.epoch_summaries.iter().any(|s| s.eval_loss.is_some());
        assert!(has_eval, "should have at least one eval result");

        assert!(result.best_loss < f32::MAX);
    }

    #[test]
    fn run_eval_perplexity_positive() {
        let config = QatPipelineConfig {
            epochs: 3,
            learning_rate: 0.01,
            min_lr: 0.001,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 1,
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.5f32; 2];
        let data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];
        let eval_data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];

        let result = pipeline.run(
            &mut weights,
            &data,
            &simple_forward,
            &simple_mse,
            Some(&eval_data),
        );

        for s in &result.epoch_summaries {
            if let Some(ppl) = s.eval_perplexity {
                assert!(ppl > 0.0, "perplexity should be positive");
                assert!(ppl.is_finite(), "perplexity should be finite");
            }
        }
    }

    // --- run: checkpoint ---

    #[test]
    fn run_saves_checkpoints() {
        let dir = tempfile::tempdir().unwrap();
        let dir_str = dir.path().to_str().unwrap();

        let config = QatPipelineConfig {
            epochs: 6,
            learning_rate: 0.01,
            min_lr: 0.001,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 0,
            checkpoint_interval: 3,
            checkpoint_dir: Some(dir_str.to_owned()),
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.5f32; 2];
        let data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];

        pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        // checkpoint_interval=3 → epoch 2 (0-indexed, (2+1)%3==0), epoch 5
        let ckpt1 = dir.path().join("qat_epoch_0002.ckpt");
        let ckpt2 = dir.path().join("qat_epoch_0005.ckpt");
        assert!(ckpt1.exists(), "checkpoint at epoch 2 should exist");
        assert!(ckpt2.exists(), "checkpoint at epoch 5 should exist");

        // チェックポイントの中身を検証
        let loaded = CheckpointData::load_from_file(&ckpt2).unwrap();
        assert_eq!(loaded.meta.epoch, 5);
        assert_eq!(loaded.weights.len(), 2);
    }

    // --- run: best tracking ---

    #[test]
    fn run_tracks_best_loss() {
        let config = QatPipelineConfig {
            epochs: 20,
            learning_rate: 0.5,
            min_lr: 0.01,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 0,
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.0f32; 2];
        let data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];

        let result = pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        assert!(result.best_loss < f32::MAX);
        assert!(result.best_loss >= 0.0);
        // best は最終 epoch か、途中の最小 loss
        let min_loss = result
            .epoch_summaries
            .iter()
            .map(|s| s.avg_loss)
            .fold(f32::MAX, f32::min);
        assert!((result.best_loss - min_loss).abs() < 1e-6);
    }

    // --- run: log ---

    #[test]
    fn run_log_has_all_steps() {
        let config = QatPipelineConfig {
            epochs: 3,
            learning_rate: 0.01,
            min_lr: 0.001,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 0,
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.5f32; 2];
        let data = vec![
            (vec![1.0, 1.0], vec![0.5, 0.5]),
            (vec![2.0, 2.0], vec![1.0, 1.0]),
        ];

        let result = pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        // 3 epochs × 2 samples = 6 log entries
        assert_eq!(result.log.len(), 6);
        assert_eq!(result.total_steps, 6);
    }

    // --- run: calibration stats ---

    #[test]
    fn run_updates_calibration_stats() {
        let config = QatPipelineConfig {
            epochs: 3,
            learning_rate: 0.01,
            min_lr: 0.001,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 0,
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.5f32, -0.3];
        let data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];

        pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        let stats = pipeline.calibration_stats();
        assert!(stats.weight_mean_abs > 0.0);
        assert!(stats.cosine_similarity > 0.0);
    }

    // --- run: skipped steps ---

    #[test]
    fn run_counts_skipped_steps() {
        // 正常な学習では skipped_steps = 0
        let config = QatPipelineConfig {
            epochs: 5,
            learning_rate: 0.01,
            min_lr: 0.001,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 0,
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.5f32; 2];
        let data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];

        let result = pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        assert_eq!(result.skipped_steps, 0);
    }

    // --- run: epoch summaries ---

    #[test]
    fn run_epoch_summaries_have_quant_metrics() {
        let config = QatPipelineConfig {
            epochs: 3,
            learning_rate: 0.01,
            min_lr: 0.001,
            warmup_steps: 0,
            total_steps: 0,
            eval_interval: 0,
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.5f32, -0.3];
        let data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];

        let result = pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        for s in &result.epoch_summaries {
            assert!(s.avg_quant_mae >= 0.0);
            assert!(s.avg_cosine_sim >= -1.0 && s.avg_cosine_sim <= 1.0);
            assert!(s.temperature > 0.0);
            assert!(s.scale > 0.0);
        }
    }

    // --- run: warmup scheduler integration ---

    #[test]
    fn run_with_warmup() {
        let config = QatPipelineConfig {
            epochs: 10,
            learning_rate: 0.1,
            min_lr: 0.001,
            warmup_steps: 5,
            total_steps: 0,
            eval_interval: 0,
            mixed_precision: MixedPrecisionConfig::disabled(),
            ..QatPipelineConfig::default()
        };
        let mut pipeline = QatPipeline::new(config);
        let mut weights = vec![0.0f32; 2];
        let data = vec![(vec![1.0, 1.0], vec![0.5, 0.5])];

        let result = pipeline.run(&mut weights, &data, &simple_forward, &simple_mse, None);

        // warmup 期間中は lr が小さい → 最初のログエントリの lr は max_lr 未満
        let first_lr = result.log.entries()[0].learning_rate;
        assert!(
            first_lr < 0.1,
            "warmup should start with low lr, got {first_lr}"
        );
    }
}
