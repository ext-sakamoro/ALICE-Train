//! Prosody joint loss — F0 L1 + duration MSE + energy MSE の重み付き和。
//!
//! FastSpeech2 の Variance Adaptor が出力する 3 predictor の同時学習に使用される。
//! 各 predictor の loss を独立に計算し、重み付き和を total loss とする。
//!
//! # Loss 定義
//!
//! - **F0 L1**: `mean(|f0_pred - f0_target|)` (mora-level、mask 有効時は valid position のみ)
//! - **Duration MSE**: `mean((dur_pred - dur_target)²)` (frames 単位、log-domain 前処理は呼び出し側の責任)
//! - **Energy MSE**: `mean((energy_pred - energy_target)²)` (dB 単位)
//! - **Total**: `w_f0 * F0_L1 + w_duration * Duration_MSE + w_energy * Energy_MSE`
//!
//! # Shape 前提
//!
//! すべての予測 / target は `[batch, mora_len]` の `Vec<Vec<f32>>` mora-level 表現を持つ。
//! frame-level (audio_f0 / audio_energy) から mora-level への集約は
//! [`ProsodyTarget::from_batch`] が [`crate::tts::TtsBatch`] の `phoneme_alignment` を使って
//! 実行する。

use crate::tts::batch::TtsBatch;
use serde::{Deserialize, Serialize};

/// FastSpeech2 Variance Adaptor 等から得られる mora-level prosody 予測。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProsodyPrediction {
    /// F0 予測 (`[batch, mora_len]`、Hz または log-Hz、単位は呼び出し側で統一)。
    pub f0: Vec<Vec<f32>>,
    /// Duration 予測 (`[batch, mora_len]`、frame 数、log(frames + 1) 変換は呼び出し側の責任)。
    pub duration_frames: Vec<Vec<f32>>,
    /// Energy 予測 (`[batch, mora_len]`、dB)。
    pub energy: Vec<Vec<f32>>,
}

/// mora-level prosody 教師 target。任意の mask も持つ (padded position 除外用)。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProsodyTarget {
    /// F0 target (`[batch, mora_len]`)。
    pub f0: Vec<Vec<f32>>,
    /// Duration target (`[batch, mora_len]`、frame 数)。
    pub duration_frames: Vec<Vec<f32>>,
    /// Energy target (`[batch, mora_len]`、dB)。
    pub energy: Vec<Vec<f32>>,
    /// mask (`[batch, mora_len]`、true = valid、false = padded)。None なら全 position 使用。
    pub mask: Option<Vec<Vec<bool>>>,
}

impl ProsodyTarget {
    /// [`TtsBatch`] から mora-level prosody target を集約構築する。
    ///
    /// 集約ルール:
    /// - `f0[b][m] = mean(audio_f0[b][start_frame..end_frame])` (voiced のみ、全て unvoiced なら 0)
    /// - `duration_frames[b][m] = durations_ms[b][m] * sample_rate / (hop_length * 1000)`
    /// - `energy[b][m] = mean(audio_energy[b][start_frame..end_frame])`
    /// - `start_frame = phoneme_alignment[b][m]`, `end_frame = phoneme_alignment[b][m+1]`
    ///   (最後の mora は end_frame = num_frames)
    ///
    /// mask は `durations_ms[b][m] > 0` を valid position として自動生成。
    ///
    /// # Errors
    ///
    /// - `hop_length == 0`
    /// - `sample_rate == 0`
    pub fn from_batch(
        batch: &TtsBatch,
        hop_length: usize,
        sample_rate: u32,
    ) -> Result<Self, ProsodyLossError> {
        if hop_length == 0 {
            return Err(ProsodyLossError::InvalidConfig {
                reason: "hop_length must be > 0".to_string(),
            });
        }
        if sample_rate == 0 {
            return Err(ProsodyLossError::InvalidConfig {
                reason: "sample_rate must be > 0".to_string(),
            });
        }

        let bs = batch.batch_size();
        let sr = sample_rate as f32;
        let hop = hop_length as f32;
        let ms_per_frame = hop * 1000.0 / sr;

        let mut f0_out = Vec::with_capacity(bs);
        let mut dur_out = Vec::with_capacity(bs);
        let mut energy_out = Vec::with_capacity(bs);
        let mut mask_out = Vec::with_capacity(bs);

        for b in 0..bs {
            let alignment = &batch.text_phoneme_alignment()[b];
            let mora_len = alignment.len();
            let f0_frame = &batch.audio_f0()[b];
            let voiced_frame = &batch.audio_voiced()[b];
            let energy_frame = &batch.audio_energy()[b];
            let n_frames = f0_frame.len();
            let durations_ms = &batch.durations_ms()[b];

            let mut f0_row = Vec::with_capacity(mora_len);
            let mut dur_row = Vec::with_capacity(mora_len);
            let mut energy_row = Vec::with_capacity(mora_len);
            let mut mask_row = Vec::with_capacity(mora_len);

            for m in 0..mora_len {
                let start = alignment[m].min(n_frames);
                let end = if m + 1 < mora_len {
                    alignment[m + 1].min(n_frames)
                } else {
                    n_frames
                };
                let end = end.max(start);

                // F0: voiced のみ平均、全 unvoiced なら 0
                // slice 長を frame 数と合わせて安全化
                let f0_slice_end = end.min(f0_frame.len());
                let voiced_slice_end = end.min(voiced_frame.len());
                let slice_end = f0_slice_end.min(voiced_slice_end);
                let (mut f0_sum, mut f0_count) = (0.0_f32, 0_usize);
                for (&f0_val, &voiced) in f0_frame[start..slice_end]
                    .iter()
                    .zip(&voiced_frame[start..slice_end])
                {
                    if voiced {
                        f0_sum += f0_val;
                        f0_count += 1;
                    }
                }
                let f0_avg = if f0_count > 0 {
                    f0_sum / f0_count as f32
                } else {
                    0.0
                };
                f0_row.push(f0_avg);

                // Duration: ms → frame 数
                let dur_ms = durations_ms.get(m).copied().unwrap_or(0);
                let dur_frames = dur_ms as f32 / ms_per_frame;
                dur_row.push(dur_frames);

                // Energy: 単純平均 (unvoiced 含む)
                let energy_slice_end = end.min(energy_frame.len());
                let energy_avg = if energy_slice_end > start {
                    let slice = &energy_frame[start..energy_slice_end];
                    let sum: f32 = slice.iter().sum();
                    sum / slice.len() as f32
                } else {
                    -100.0 // silence floor
                };
                energy_row.push(energy_avg);

                mask_row.push(dur_ms > 0);
            }

            f0_out.push(f0_row);
            dur_out.push(dur_row);
            energy_out.push(energy_row);
            mask_out.push(mask_row);
        }

        Ok(Self {
            f0: f0_out,
            duration_frames: dur_out,
            energy: energy_out,
            mask: Some(mask_out),
        })
    }

    /// `from_batch` + `to_log_duration` を 1 step で行う helper (Phase E-next-4)。
    ///
    /// `duration_frames` を `log(duration_frames + 1)` に変換して保存する。
    /// FastSpeech2 論文の慣習で、durations は log-domain で学習される (small dur を強調)。
    /// 変換後は `FastSpeech2::init_variance_biases(1.5, ...)` の bias 事前設定と併用推奨。
    ///
    /// # Errors
    ///
    /// `from_batch` と同じ (hop_length == 0 / sample_rate == 0)。
    pub fn from_batch_log_duration(
        batch: &TtsBatch,
        hop_length: usize,
        sample_rate: u32,
    ) -> Result<Self, ProsodyLossError> {
        let mut target = Self::from_batch(batch, hop_length, sample_rate)?;
        target.to_log_duration();
        Ok(target)
    }

    /// `duration_frames` を `log(duration_frames + 1)` に in-place 変換 (Phase E-next-4)。
    ///
    /// 負値は 0 に clamp してから log 適用 (log(0+1) = 0 で自然に 0 になる)。
    /// mask false 位置は変換されるが loss 計算で除外されるため影響なし。
    pub fn to_log_duration(&mut self) {
        for row in &mut self.duration_frames {
            for v in row {
                let clamped = v.max(0.0);
                *v = clamped.ln_1p();
            }
        }
    }
}

/// [`ProsodyLoss::compute`] の戻り値 — 各 loss 個別 + total。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LossComponents {
    /// F0 の L1 loss (mean absolute error)。
    pub f0_l1: f32,
    /// Duration の MSE (mean squared error)。
    pub duration_mse: f32,
    /// Energy の MSE。
    pub energy_mse: f32,
    /// 重み付き total: `w_f0 * f0_l1 + w_duration * duration_mse + w_energy * energy_mse`。
    pub total: f32,
}

/// Prosody 3 predictor の重み付き joint loss。
///
/// FastSpeech2 論文の default weight を参考に `w_f0 = 1.0`, `w_duration = 1.0`, `w_energy = 0.5`
/// を [`Self::default_weights`] として提供。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ProsodyLoss {
    /// F0 L1 loss の重み。
    pub w_f0: f32,
    /// Duration MSE の重み。
    pub w_duration: f32,
    /// Energy MSE の重み。
    pub w_energy: f32,
}

impl ProsodyLoss {
    /// FastSpeech2 論文 default weight (`w_f0=1.0`, `w_duration=1.0`, `w_energy=0.5`)。
    #[must_use]
    pub fn default_weights() -> Self {
        Self {
            w_f0: 1.0,
            w_duration: 1.0,
            w_energy: 0.5,
        }
    }

    /// 予測と target から loss を計算する。
    ///
    /// # 手続き
    ///
    /// 1. shape 検証 (batch size / mora len 一致)
    /// 2. mask 有効時は valid position のみ集計
    /// 3. F0 L1 / Duration MSE / Energy MSE を独立計算
    /// 4. 重み付き和で total 計算
    ///
    /// # Errors
    ///
    /// [`ProsodyLossError::ShapeMismatch`] on batch / mora dimension mismatch。
    /// [`ProsodyLossError::EmptyBatch`] on empty batch。
    pub fn compute(
        &self,
        pred: &ProsodyPrediction,
        target: &ProsodyTarget,
    ) -> Result<LossComponents, ProsodyLossError> {
        let bs = pred.f0.len();
        if bs == 0 {
            return Err(ProsodyLossError::EmptyBatch);
        }
        check_batch_dim("pred.duration_frames", pred.duration_frames.len(), bs)?;
        check_batch_dim("pred.energy", pred.energy.len(), bs)?;
        check_batch_dim("target.f0", target.f0.len(), bs)?;
        check_batch_dim("target.duration_frames", target.duration_frames.len(), bs)?;
        check_batch_dim("target.energy", target.energy.len(), bs)?;
        if let Some(m) = &target.mask {
            check_batch_dim("target.mask", m.len(), bs)?;
        }

        // per-sample mora length 一致検証 + 損失集計
        let mut f0_sum = 0.0_f64;
        let mut dur_sum = 0.0_f64;
        let mut energy_sum = 0.0_f64;
        let mut count: u64 = 0;

        for b in 0..bs {
            let mora_len = pred.f0[b].len();
            check_mora_dim(
                b,
                "pred.duration_frames",
                pred.duration_frames[b].len(),
                mora_len,
            )?;
            check_mora_dim(b, "pred.energy", pred.energy[b].len(), mora_len)?;
            check_mora_dim(b, "target.f0", target.f0[b].len(), mora_len)?;
            check_mora_dim(
                b,
                "target.duration_frames",
                target.duration_frames[b].len(),
                mora_len,
            )?;
            check_mora_dim(b, "target.energy", target.energy[b].len(), mora_len)?;
            if let Some(m) = &target.mask {
                check_mora_dim(b, "target.mask", m[b].len(), mora_len)?;
            }

            for m in 0..mora_len {
                let valid = target.mask.as_ref().is_none_or(|mk| mk[b][m]);
                if !valid {
                    continue;
                }
                let df = pred.f0[b][m] - target.f0[b][m];
                f0_sum += f64::from(df.abs());

                let dd = pred.duration_frames[b][m] - target.duration_frames[b][m];
                dur_sum += f64::from(dd * dd);

                let de = pred.energy[b][m] - target.energy[b][m];
                energy_sum += f64::from(de * de);

                count += 1;
            }
        }

        if count == 0 {
            // 全 mask false → loss は 0 と定義 (実運用では警告レベル、fail_fast にせず総合 loss 0 で返す)
            return Ok(LossComponents::default());
        }

        let cnt = count as f64;
        let f0_l1 = (f0_sum / cnt) as f32;
        let duration_mse = (dur_sum / cnt) as f32;
        let energy_mse = (energy_sum / cnt) as f32;
        let total = self.w_f0 * f0_l1 + self.w_duration * duration_mse + self.w_energy * energy_mse;

        Ok(LossComponents {
            f0_l1,
            duration_mse,
            energy_mse,
            total,
        })
    }
}

impl Default for ProsodyLoss {
    fn default() -> Self {
        Self::default_weights()
    }
}

fn check_batch_dim(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), ProsodyLossError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ProsodyLossError::ShapeMismatch {
            field,
            axis: "batch",
            expected,
            actual,
        })
    }
}

fn check_mora_dim(
    sample: usize,
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), ProsodyLossError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ProsodyLossError::MoraLenMismatch {
            sample,
            field,
            expected,
            actual,
        })
    }
}

/// [`ProsodyLoss::compute`] / [`ProsodyTarget::from_batch`] で発生し得るエラー。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProsodyLossError {
    /// Batch dimension が field 間で不一致。
    ShapeMismatch {
        /// 不一致 field 名。
        field: &'static str,
        /// axis 名 ("batch" / "mora")。
        axis: &'static str,
        /// 期待値。
        expected: usize,
        /// 実際値。
        actual: usize,
    },
    /// per-sample mora 長が field 間で不一致。
    MoraLenMismatch {
        /// sample index。
        sample: usize,
        /// 不一致 field 名。
        field: &'static str,
        /// 期待値。
        expected: usize,
        /// 実際値。
        actual: usize,
    },
    /// 空 batch。
    EmptyBatch,
    /// 不正 config (hop_length=0 等)。
    InvalidConfig {
        /// 理由。
        reason: String,
    },
}

impl std::fmt::Display for ProsodyLossError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch {
                field,
                axis,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch on {axis} axis: field '{field}' has {actual}, expected {expected}"
            ),
            Self::MoraLenMismatch {
                sample,
                field,
                expected,
                actual,
            } => write!(
                f,
                "mora length mismatch at sample {sample}: field '{field}' has {actual}, expected {expected}"
            ),
            Self::EmptyBatch => write!(f, "empty batch (batch size = 0)"),
            Self::InvalidConfig { reason } => write!(f, "invalid config: {reason}"),
        }
    }
}

impl std::error::Error for ProsodyLossError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::batch::TtsBatch;

    fn make_valid_batch() -> TtsBatch {
        // 1 sample, mel 80 × 4 frames, mora 2, alignment [0, 2] (mora 0 = frame 0-1, mora 1 = frame 2-3)
        TtsBatch::new(
            vec![vec![0.0_f32; 24_000]],
            vec![vec![vec![0.0_f32; 4]; 80]],
            vec![vec![100.0, 105.0, 200.0, 210.0]],
            vec![vec![true; 4]],
            vec![vec![-20.0, -18.0, -15.0, -12.0]],
            vec![vec![1_u32]],
            vec![vec![0_u8, 1]],
            vec![vec![0_u8]],
            vec![vec![0_usize, 2]],
            vec![0_u32],
            vec![vec![50_u32, 40]],
            80,
        )
        .expect("valid batch")
    }

    #[test]
    fn to_log_duration_transforms_in_place() {
        // Phase E-next-4: log(dur + 1) 変換確認
        let mut target = ProsodyTarget {
            f0: vec![vec![100.0, 105.0]],
            duration_frames: vec![vec![3.0, 7.0]],
            energy: vec![vec![-20.0, -15.0]],
            mask: None,
        };
        target.to_log_duration();
        // log(3+1)=log(4)≈1.386、log(7+1)=log(8)≈2.079
        let expected_1 = 4.0_f32.ln();
        let expected_2 = 8.0_f32.ln();
        assert!((target.duration_frames[0][0] - expected_1).abs() < 1e-6);
        assert!((target.duration_frames[0][1] - expected_2).abs() < 1e-6);
        // f0 / energy は不変
        assert!((target.f0[0][0] - 100.0).abs() < 1e-6);
        assert!((target.energy[0][0] - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn to_log_duration_negative_clamps_to_zero() {
        // 負値 → 0 clamp → log(0+1) = 0
        let mut target = ProsodyTarget {
            f0: vec![vec![100.0]],
            duration_frames: vec![vec![-5.0]],
            energy: vec![vec![-20.0]],
            mask: None,
        };
        target.to_log_duration();
        assert!(target.duration_frames[0][0].abs() < 1e-6);
    }

    #[test]
    fn from_batch_log_duration_applies_log_transform() {
        // Phase E-next-4: from_batch + to_log_duration の 1-step helper
        let batch = make_valid_batch();
        let raw = ProsodyTarget::from_batch(&batch, 256, 24_000).expect("raw");
        let log_target = ProsodyTarget::from_batch_log_duration(&batch, 256, 24_000).expect("log");
        // log_target.duration_frames[b][m] = log(raw.duration_frames[b][m] + 1)
        for (raw_row, log_row) in raw.duration_frames.iter().zip(&log_target.duration_frames) {
            for (r, l) in raw_row.iter().zip(log_row) {
                let expected = r.ln_1p();
                assert!(
                    (l - expected).abs() < 1e-6,
                    "log_dur mismatch raw={r} log={l} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn perfect_prediction_gives_zero_loss() {
        let pred = ProsodyPrediction {
            f0: vec![vec![100.0, 105.0]],
            duration_frames: vec![vec![5.0, 4.0]],
            energy: vec![vec![-20.0, -15.0]],
        };
        let target = ProsodyTarget {
            f0: vec![vec![100.0, 105.0]],
            duration_frames: vec![vec![5.0, 4.0]],
            energy: vec![vec![-20.0, -15.0]],
            mask: None,
        };
        let loss = ProsodyLoss::default_weights()
            .compute(&pred, &target)
            .unwrap();
        assert!(loss.f0_l1.abs() < 1e-5);
        assert!(loss.duration_mse.abs() < 1e-5);
        assert!(loss.energy_mse.abs() < 1e-5);
        assert!(loss.total.abs() < 1e-5);
    }

    #[test]
    fn f0_l1_manual_verification() {
        // pred = [10, 20], target = [15, 30] → L1 = mean(|10-15|, |20-30|) = mean(5, 10) = 7.5
        let pred = ProsodyPrediction {
            f0: vec![vec![10.0, 20.0]],
            duration_frames: vec![vec![0.0, 0.0]],
            energy: vec![vec![0.0, 0.0]],
        };
        let target = ProsodyTarget {
            f0: vec![vec![15.0, 30.0]],
            duration_frames: vec![vec![0.0, 0.0]],
            energy: vec![vec![0.0, 0.0]],
            mask: None,
        };
        let loss = ProsodyLoss::default_weights()
            .compute(&pred, &target)
            .unwrap();
        assert!(
            (loss.f0_l1 - 7.5).abs() < 1e-5,
            "expected 7.5, got {}",
            loss.f0_l1
        );
    }

    #[test]
    fn duration_mse_manual_verification() {
        // pred = [3, 5], target = [1, 4] → MSE = mean((3-1)², (5-4)²) = mean(4, 1) = 2.5
        let pred = ProsodyPrediction {
            f0: vec![vec![0.0, 0.0]],
            duration_frames: vec![vec![3.0, 5.0]],
            energy: vec![vec![0.0, 0.0]],
        };
        let target = ProsodyTarget {
            f0: vec![vec![0.0, 0.0]],
            duration_frames: vec![vec![1.0, 4.0]],
            energy: vec![vec![0.0, 0.0]],
            mask: None,
        };
        let loss = ProsodyLoss::default_weights()
            .compute(&pred, &target)
            .unwrap();
        assert!(
            (loss.duration_mse - 2.5).abs() < 1e-5,
            "expected 2.5, got {}",
            loss.duration_mse
        );
    }

    #[test]
    fn weight_config_is_reflected_in_total() {
        let pred = ProsodyPrediction {
            f0: vec![vec![0.0]],
            duration_frames: vec![vec![0.0]],
            energy: vec![vec![0.0]],
        };
        let target = ProsodyTarget {
            f0: vec![vec![10.0]],
            duration_frames: vec![vec![2.0]],
            energy: vec![vec![-4.0]],
            mask: None,
        };
        let loss = ProsodyLoss {
            w_f0: 2.0,
            w_duration: 0.5,
            w_energy: 3.0,
        };
        let LossComponents {
            f0_l1,
            duration_mse,
            energy_mse,
            total,
        } = loss.compute(&pred, &target).unwrap();
        // f0_l1 = 10, duration_mse = 4, energy_mse = 16
        assert!((f0_l1 - 10.0).abs() < 1e-4);
        assert!((duration_mse - 4.0).abs() < 1e-4);
        assert!((energy_mse - 16.0).abs() < 1e-4);
        let expected_total = 2.0 * 10.0 + 0.5 * 4.0 + 3.0 * 16.0;
        assert!(
            (total - expected_total).abs() < 1e-3,
            "expected {expected_total}, got {total}"
        );
    }

    #[test]
    fn mask_excludes_padded_positions() {
        // mora 2 で padded、mask=false → loss は mora 0 と 1 のみで計算 (mora 2 の巨大差は無視)
        let pred = ProsodyPrediction {
            f0: vec![vec![10.0, 20.0, 100.0]],
            duration_frames: vec![vec![0.0, 0.0, 0.0]],
            energy: vec![vec![0.0, 0.0, 0.0]],
        };
        let target = ProsodyTarget {
            f0: vec![vec![10.0, 20.0, 0.0]], // mora 2 は 0 (padded)
            duration_frames: vec![vec![0.0, 0.0, 0.0]],
            energy: vec![vec![0.0, 0.0, 0.0]],
            mask: Some(vec![vec![true, true, false]]),
        };
        let loss = ProsodyLoss::default_weights()
            .compute(&pred, &target)
            .unwrap();
        // mora 0,1 は完全一致 → L1 = 0 (mora 2 の巨大差は mask で無視)
        assert!(
            loss.f0_l1.abs() < 1e-5,
            "expected 0 (masked out), got {}",
            loss.f0_l1
        );
    }

    #[test]
    fn all_masked_out_returns_zero_loss() {
        let pred = ProsodyPrediction {
            f0: vec![vec![10.0]],
            duration_frames: vec![vec![10.0]],
            energy: vec![vec![10.0]],
        };
        let target = ProsodyTarget {
            f0: vec![vec![100.0]],
            duration_frames: vec![vec![100.0]],
            energy: vec![vec![100.0]],
            mask: Some(vec![vec![false]]),
        };
        let loss = ProsodyLoss::default_weights()
            .compute(&pred, &target)
            .unwrap();
        assert_eq!(loss, LossComponents::default());
    }

    #[test]
    fn empty_batch_returns_error() {
        let pred = ProsodyPrediction {
            f0: vec![],
            duration_frames: vec![],
            energy: vec![],
        };
        let target = ProsodyTarget {
            f0: vec![],
            duration_frames: vec![],
            energy: vec![],
            mask: None,
        };
        let err = ProsodyLoss::default_weights()
            .compute(&pred, &target)
            .expect_err("empty batch must fail");
        assert_eq!(err, ProsodyLossError::EmptyBatch);
    }

    #[test]
    fn batch_size_mismatch_returns_error() {
        let pred = ProsodyPrediction {
            f0: vec![vec![1.0]],
            duration_frames: vec![vec![1.0], vec![1.0]], // batch 2, mismatch
            energy: vec![vec![1.0]],
        };
        let target = ProsodyTarget {
            f0: vec![vec![1.0]],
            duration_frames: vec![vec![1.0]],
            energy: vec![vec![1.0]],
            mask: None,
        };
        let err = ProsodyLoss::default_weights()
            .compute(&pred, &target)
            .expect_err("batch mismatch must fail");
        assert!(matches!(
            err,
            ProsodyLossError::ShapeMismatch {
                field: "pred.duration_frames",
                axis: "batch",
                ..
            }
        ));
    }

    #[test]
    fn mora_len_mismatch_returns_error() {
        let pred = ProsodyPrediction {
            f0: vec![vec![1.0, 2.0]],
            duration_frames: vec![vec![1.0]], // mora 1, mismatch
            energy: vec![vec![1.0, 2.0]],
        };
        let target = ProsodyTarget {
            f0: vec![vec![1.0, 2.0]],
            duration_frames: vec![vec![1.0, 2.0]],
            energy: vec![vec![1.0, 2.0]],
            mask: None,
        };
        let err = ProsodyLoss::default_weights()
            .compute(&pred, &target)
            .expect_err("mora len mismatch must fail");
        assert!(matches!(err, ProsodyLossError::MoraLenMismatch { .. }));
    }

    #[test]
    fn from_batch_computes_mora_level_aggregation() {
        // batch: mora 0 = frame 0-1 (f0=100,105 voiced), mora 1 = frame 2-3 (f0=200,210 voiced)
        // durations_ms = [50, 40], sr=24000, hop=256 → ms_per_frame = 256*1000/24000 ≈ 10.667
        // duration_frames: [50/10.667, 40/10.667] ≈ [4.6875, 3.75]
        // f0 avg: [(100+105)/2, (200+210)/2] = [102.5, 205.0]
        // energy avg: [(-20-18)/2, (-15-12)/2] = [-19, -13.5]
        let batch = make_valid_batch();
        let target = ProsodyTarget::from_batch(&batch, 256, 24_000).expect("aggregation");

        assert_eq!(target.f0.len(), 1);
        assert_eq!(target.f0[0].len(), 2);
        assert!((target.f0[0][0] - 102.5).abs() < 1e-3);
        assert!((target.f0[0][1] - 205.0).abs() < 1e-3);
        assert!((target.energy[0][0] - (-19.0)).abs() < 1e-3);
        assert!((target.energy[0][1] - (-13.5)).abs() < 1e-3);
        // duration_frames[0][0] = 50 * 24000 / (256 * 1000) ≈ 4.6875
        assert!((target.duration_frames[0][0] - 4.6875).abs() < 1e-3);
        assert!((target.duration_frames[0][1] - 3.75).abs() < 1e-3);
        assert!(target.mask.is_some());
    }

    #[test]
    fn from_batch_marks_zero_duration_as_padded() {
        // durations_ms[0] = 0 → mask[0] = false (padded)
        let batch = TtsBatch::new(
            vec![vec![0.0_f32; 24_000]],
            vec![vec![vec![0.0_f32; 4]; 80]],
            vec![vec![100.0, 105.0, 200.0, 210.0]],
            vec![vec![true; 4]],
            vec![vec![-20.0, -18.0, -15.0, -12.0]],
            vec![vec![1_u32]],
            vec![vec![0_u8, 1]],
            vec![vec![0_u8]],
            vec![vec![0_usize, 2]],
            vec![0_u32],
            vec![vec![0_u32, 40]], // mora 0 padded
            80,
        )
        .expect("valid batch");
        let target = ProsodyTarget::from_batch(&batch, 256, 24_000).unwrap();
        let mask = target.mask.expect("mask exists");
        assert!(!mask[0][0]);
        assert!(mask[0][1]);
    }

    #[test]
    fn from_batch_rejects_invalid_config() {
        let batch = make_valid_batch();
        let err = ProsodyTarget::from_batch(&batch, 0, 24_000).expect_err("hop=0 must fail");
        assert!(matches!(err, ProsodyLossError::InvalidConfig { .. }));

        let err = ProsodyTarget::from_batch(&batch, 256, 0).expect_err("sr=0 must fail");
        assert!(matches!(err, ProsodyLossError::InvalidConfig { .. }));
    }

    #[test]
    fn default_weights_are_1_1_0_5() {
        let loss = ProsodyLoss::default_weights();
        assert!((loss.w_f0 - 1.0).abs() < f32::EPSILON);
        assert!((loss.w_duration - 1.0).abs() < f32::EPSILON);
        assert!((loss.w_energy - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn default_trait_matches_default_weights() {
        let d = ProsodyLoss::default();
        let w = ProsodyLoss::default_weights();
        assert!((d.w_f0 - w.w_f0).abs() < f32::EPSILON);
        assert!((d.w_duration - w.w_duration).abs() < f32::EPSILON);
        assert!((d.w_energy - w.w_energy).abs() < f32::EPSILON);
    }

    #[test]
    fn error_display_works() {
        let e = ProsodyLossError::EmptyBatch;
        assert_eq!(format!("{e}"), "empty batch (batch size = 0)");
        let e = ProsodyLossError::InvalidConfig {
            reason: "hop_length must be > 0".to_string(),
        };
        assert!(format!("{e}").contains("hop_length"));
    }

    #[test]
    fn round_trip_batch_to_target_to_perfect_pred_zero_loss() {
        let batch = make_valid_batch();
        let target = ProsodyTarget::from_batch(&batch, 256, 24_000).unwrap();
        // perfect prediction = target
        let pred = ProsodyPrediction {
            f0: target.f0.clone(),
            duration_frames: target.duration_frames.clone(),
            energy: target.energy.clone(),
        };
        let loss = ProsodyLoss::default_weights()
            .compute(&pred, &target)
            .unwrap();
        assert!(
            loss.total.abs() < 1e-4,
            "total should be ~0, got {}",
            loss.total
        );
    }
}
