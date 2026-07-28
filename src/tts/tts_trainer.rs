//! TTS 学習用 trainer (FastSpeech2 に特化)。
//!
//! Phase T.4a Phase 4 実装:
//! - forward → mel L2 loss → backward_full → SGD update の 1 step パイプライン
//! - MVP: SGD only (AdamW は Phase T.4a 継続 で追加、learning stability の観点で必要)
//! - ProsodyLoss + variance predictor 学習は Phase T.4a 継続 (現状 predictor grad は 0)
//!
//! # 使用例
//!
//! ```rust,no_run
//! # #[cfg(feature = "tts")] {
//! use alice_train::tts::{FastSpeech2, FastSpeech2Config, TtsTrainConfig, TtsTrainer};
//!
//! let cfg = FastSpeech2Config {
//!     vocab_size: 100,
//!     hidden_dim: 128,
//!     num_heads: 2,
//!     num_encoder_layers: 4,
//!     num_decoder_layers: 4,
//!     fft_kernel_size: 3,
//!     fft_expansion: 4,
//!     predictor_kernel_size: 3,
//!     predictor_hidden: 128,
//!     mel_dim: 80,
//!     postnet_kernel_size: 5,
//!     postnet_layers: 5,
//!     postnet_hidden: 512,
//!     max_len: 1024,
//! };
//! let model = FastSpeech2::zeros(cfg).expect("model");
//! let train_cfg = TtsTrainConfig {
//!     learning_rate: 1e-3,
//!     log_interval: 100,
//! };
//! let mut trainer = TtsTrainer::new(model, train_cfg);
//!
//! // Training loop (mora_ids, durations, mel_target は TtsBatch から取得)
//! let mora_ids: Vec<u32> = vec![1, 2, 3];
//! let durations: Vec<u32> = vec![2, 2, 2]; // 6 frames
//! let mel_target = vec![0.0_f32; 6 * 80]; // ground truth mel
//! let result = trainer
//!     .step(&mora_ids, &durations, &mel_target, 1, 3)
//!     .expect("step");
//! println!("step {}: mel_loss={}", result.step, result.mel_loss);
//! # }
//! ```

use crate::tts::fastspeech2::{FastSpeech2, FastSpeech2Error};

/// TTS trainer 設定。
#[derive(Clone, Copy, Debug)]
pub struct TtsTrainConfig {
    /// 学習率 (SGD)。
    pub learning_rate: f32,
    /// log 出力間隔 (step 単位)。
    pub log_interval: usize,
}

impl Default for TtsTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            log_interval: 100,
        }
    }
}

/// 1 step の結果。
#[derive(Clone, Copy, Debug)]
pub struct TtsStepResult {
    /// mel L2 loss (mean squared error)。
    pub mel_loss: f32,
    /// step counter (1-indexed、trainer が自動 increment)。
    pub step: usize,
}

/// TTS 学習 trainer。
///
/// FastSpeech2 model + TtsTrainConfig を保持し、step method で 1 iteration を実行する。
/// MVP: SGD only (AdamW は Phase T.4a 継続 で追加予定)。
#[derive(Debug)]
pub struct TtsTrainer {
    model: FastSpeech2,
    config: TtsTrainConfig,
    step_count: usize,
}

impl TtsTrainer {
    /// 新しい trainer を構築する。
    #[must_use]
    pub fn new(model: FastSpeech2, config: TtsTrainConfig) -> Self {
        Self {
            model,
            config,
            step_count: 0,
        }
    }

    /// 内部 model への参照。
    #[must_use]
    pub fn model(&self) -> &FastSpeech2 {
        &self.model
    }

    /// 内部 model への可変参照 (weight 手動修正用、通常は step が更新する)。
    pub fn model_mut(&mut self) -> &mut FastSpeech2 {
        &mut self.model
    }

    /// config への参照。
    #[must_use]
    pub fn config(&self) -> &TtsTrainConfig {
        &self.config
    }

    /// 現在の step 数 (何 iteration 学習したか)。
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// 1 step: forward → mel L2 loss → backward_full → SGD update。
    ///
    /// # 引数
    ///
    /// - `mora_ids`: `[batch, mora_len]` u32
    /// - `target_durations`: `[batch, mora_len]` u32 (frame 単位)
    /// - `mel_target`: `[batch, frame_len, mel_dim]` f32 (ground truth mel-spectrogram)
    ///
    /// # 戻り値
    ///
    /// [`TtsStepResult`] (mel_loss + step counter)。
    ///
    /// # Errors
    ///
    /// forward / backward の shape 不整合。
    pub fn step(
        &mut self,
        mora_ids: &[u32],
        target_durations: &[u32],
        mel_target: &[f32],
        batch: usize,
        mora_len: usize,
    ) -> Result<TtsStepResult, FastSpeech2Error> {
        // 1. Forward
        let mel_pred = self
            .model
            .forward(mora_ids, batch, mora_len, target_durations)?;
        if mel_pred.len() != mel_target.len() {
            return Err(FastSpeech2Error::ShapeMismatch {
                field: "mel_target",
                expected: mel_pred.len(),
                actual: mel_target.len(),
            });
        }

        // 2. MSE loss + grad_mel = 2/N * (pred - target)
        let n = mel_pred.len() as f32;
        let mut mel_loss_sum = 0.0_f32;
        let mut grad_mel = vec![0.0_f32; mel_pred.len()];
        for i in 0..mel_pred.len() {
            let diff = mel_pred[i] - mel_target[i];
            mel_loss_sum += diff * diff;
            grad_mel[i] = 2.0 * diff / n;
        }
        let mel_loss = mel_loss_sum / n;

        // 3. Backward
        let grads =
            self.model
                .backward_full(mora_ids, target_durations, &grad_mel, batch, mora_len)?;

        // 4. SGD update
        self.model.apply_sgd(&grads, self.config.learning_rate);

        self.step_count += 1;
        Ok(TtsStepResult {
            mel_loss,
            step: self.step_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::FastSpeech2Config;

    fn small_config() -> FastSpeech2Config {
        FastSpeech2Config {
            vocab_size: 20,
            hidden_dim: 8,
            num_heads: 2,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            fft_kernel_size: 3,
            fft_expansion: 2,
            predictor_kernel_size: 3,
            predictor_hidden: 8,
            mel_dim: 16,
            postnet_kernel_size: 5,
            postnet_layers: 3,
            postnet_hidden: 16,
            max_len: 64,
        }
    }

    fn make_random_model() -> FastSpeech2 {
        // Zero model のまま (LayerNorm eps 保護で forward は動く、weight tuning は apply_sgd 側で)
        let cfg = small_config();
        FastSpeech2::zeros(cfg).unwrap()
    }

    #[test]
    fn trainer_construct_and_getters() {
        let model = FastSpeech2::zeros(small_config()).unwrap();
        let trainer = TtsTrainer::new(model, TtsTrainConfig::default());
        assert_eq!(trainer.step_count(), 0);
        assert!((trainer.config().learning_rate - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn trainer_step_increments_counter() {
        let model = make_random_model();
        let mut trainer = TtsTrainer::new(model, TtsTrainConfig::default());
        let mora_ids = vec![1_u32, 2, 3];
        let durations = vec![2_u32, 2, 2];
        let mel_target = vec![0.5_f32; 6 * small_config().mel_dim];
        let result = trainer
            .step(&mora_ids, &durations, &mel_target, 1, 3)
            .expect("step");
        assert_eq!(result.step, 1);
        assert!(result.mel_loss.is_finite());
        assert_eq!(trainer.step_count(), 1);
    }

    #[test]
    fn trainer_multiple_steps_are_finite() {
        // 5 step 走らせて NaN/Inf 出さないこと + step counter が上がることを確認
        let model = make_random_model();
        let mut trainer = TtsTrainer::new(model, TtsTrainConfig::default());
        let mora_ids = vec![1_u32, 2, 3];
        let durations = vec![2_u32, 2, 2];
        let mel_target = vec![0.3_f32; 6 * small_config().mel_dim];

        for i in 1..=5 {
            let result = trainer
                .step(&mora_ids, &durations, &mel_target, 1, 3)
                .expect("step");
            assert!(
                result.mel_loss.is_finite(),
                "step {i} mel_loss not finite: {}",
                result.mel_loss
            );
            assert_eq!(result.step, i);
        }
        assert_eq!(trainer.step_count(), 5);
    }

    #[test]
    fn trainer_step_shape_mismatch_returns_error() {
        let model = FastSpeech2::zeros(small_config()).unwrap();
        let mut trainer = TtsTrainer::new(model, TtsTrainConfig::default());
        let mora_ids = vec![1_u32, 2, 3];
        let durations = vec![2_u32, 2, 2];
        let mel_target = vec![0.0_f32; 100]; // wrong size
        let err = trainer
            .step(&mora_ids, &durations, &mel_target, 1, 3)
            .expect_err("shape mismatch");
        assert!(matches!(err, FastSpeech2Error::ShapeMismatch { .. }));
    }

    #[test]
    fn default_train_config_is_lr_1e_3_log_100() {
        let cfg = TtsTrainConfig::default();
        assert!((cfg.learning_rate - 1e-3).abs() < 1e-9);
        assert_eq!(cfg.log_interval, 100);
    }
}
