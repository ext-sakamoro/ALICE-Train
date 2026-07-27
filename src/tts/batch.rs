//! TTS 学習用マルチモーダルバッチ型 (Feature Request R1)。
//!
//! [`TtsBatch`] は FastSpeech2 (Phase T.4a) / VITS2 (Phase T.4b) 両方の学習で必要となる
//! 11 種類の modality をひとまとめにする batch 型。構築時に shape 整合性を全 field で
//! 検証し、mismatched batch を build 時点で fail-fast させる。
//!
//! # 例
//!
//! ```rust
//! # #[cfg(feature = "tts")] {
//! use alice_train::tts::TtsBatch;
//!
//! // 1 sample の最小 batch (mel 80 dim × 3 frames, mora 2 個)
//! let batch = TtsBatch::new(
//!     vec![vec![0.0; 24_000]],                                    // audio_waveform (1 sec @ 24 kHz)
//!     vec![vec![vec![0.0; 3]; 80]],                               // audio_mel [80 mel × 3 frames]
//!     vec![vec![100.0, 110.0, 120.0]],                            // audio_f0
//!     vec![vec![true, true, true]],                               // audio_voiced
//!     vec![vec![-20.0, -18.0, -19.0]],                            // audio_energy (dB)
//!     vec![vec![1_u32, 2, 3]],                                    // text_input_ids
//!     vec![vec![0_u8, 1]],                                        // text_moras
//!     vec![vec![0_u8]],                                           // text_accent_types (1 phrase)
//!     vec![vec![0_usize, 2]],                                     // text_phoneme_alignment (mora → frame idx)
//!     vec![0_u32],                                                // speaker_id
//!     vec![vec![50_u32, 40]],                                     // durations_ms
//!     80,                                                         // expected n_mels
//! )
//! .expect("valid batch");
//!
//! assert_eq!(batch.batch_size(), 1);
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// TTS マルチモーダル学習 batch。
///
/// FastSpeech2 (non-AR) / VITS2 (flow) 両方の acoustic model 学習で共通に使用される
/// 11 modality を保持する。すべての field は **outer Vec が batch dimension**、inner Vec が
/// per-sample data を表す。frame-level field (mel/f0/voiced/energy) は同じ frame 数を持つ
/// ことが constructor で保証される。
///
/// # Field 概要
///
/// | Field | Shape | 単位 | 用途 |
/// |---|---|---|---|
/// | `audio_waveform` | `[batch, samples]` | f32, 24 kHz mono | vocoder / audio recon loss |
/// | `audio_mel` | `[batch, n_mels, frames]` | log-mel | acoustic model target |
/// | `audio_f0` | `[batch, frames]` | Hz (voiced) or 0 (unvoiced) | prosody supervision |
/// | `audio_voiced` | `[batch, frames]` | bool | V/UV flag |
/// | `audio_energy` | `[batch, frames]` | dB | prosody supervision |
/// | `text_input_ids` | `[batch, seq_len]` | u32 tokenizer id | text encoder input |
/// | `text_moras` | `[batch, mora_len]` | u8 mora id | G2P output |
/// | `text_accent_types` | `[batch, phrase_len]` | u8 accent type | prosody bias |
/// | `text_phoneme_alignment` | `[batch, mora_len]` | frame index | Length Regulator target |
/// | `speaker_id` | `[batch]` | u32 | speaker embedding lookup |
/// | `durations_ms` | `[batch, mora_len]` | ms | duration predictor target |
///
/// # 検証仕様
///
/// [`TtsBatch::new`] は以下を全て検証する:
///
/// 1. 全 field の batch dimension が一致 (`batch_size` > 0)
/// 2. `audio_mel` の第 2 dim (n_mels) が引数 `expected_n_mels` と一致
/// 3. 各 sample で `audio_mel` frames = `audio_f0` len = `audio_voiced` len = `audio_energy` len
/// 4. 各 sample で `text_moras` len = `text_phoneme_alignment` len = `durations_ms` len
///
/// 失敗時は [`TtsBatchError`] を返す。
///
/// # Feature gate
///
/// このモジュールは `feature = "tts"` でのみ有効。既存 LLM 学習には影響しない。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TtsBatch {
    audio_waveform: Vec<Vec<f32>>,
    audio_mel: Vec<Vec<Vec<f32>>>,
    audio_f0: Vec<Vec<f32>>,
    audio_voiced: Vec<Vec<bool>>,
    audio_energy: Vec<Vec<f32>>,
    text_input_ids: Vec<Vec<u32>>,
    text_moras: Vec<Vec<u8>>,
    text_accent_types: Vec<Vec<u8>>,
    text_phoneme_alignment: Vec<Vec<usize>>,
    speaker_id: Vec<u32>,
    durations_ms: Vec<Vec<u32>>,
    n_mels: usize,
}

impl TtsBatch {
    /// 全 field を検証しつつ新しい `TtsBatch` を構築する。
    ///
    /// `expected_n_mels` は Phase T.5 vocoder の入力 mel 次元と一致する必要がある
    /// (通常 80、Vocos v2.0 baseline は 80 mel @ 22050 Hz、24 kHz 対応は Phase T.6+ 検討)。
    ///
    /// # Errors
    ///
    /// batch dimension 不一致 / mel dim 不一致 / frame 数不一致 / mora 長不一致 / 空 batch のいずれかで
    /// [`TtsBatchError`] を返す。詳細は [`TtsBatchError`] を参照。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        audio_waveform: Vec<Vec<f32>>,
        audio_mel: Vec<Vec<Vec<f32>>>,
        audio_f0: Vec<Vec<f32>>,
        audio_voiced: Vec<Vec<bool>>,
        audio_energy: Vec<Vec<f32>>,
        text_input_ids: Vec<Vec<u32>>,
        text_moras: Vec<Vec<u8>>,
        text_accent_types: Vec<Vec<u8>>,
        text_phoneme_alignment: Vec<Vec<usize>>,
        speaker_id: Vec<u32>,
        durations_ms: Vec<Vec<u32>>,
        expected_n_mels: usize,
    ) -> Result<Self, TtsBatchError> {
        let batch = audio_waveform.len();

        if batch == 0 {
            return Err(TtsBatchError::EmptyBatch);
        }

        // Step 1: 全 field の batch dimension check
        check_batch_dim("audio_mel", audio_mel.len(), batch)?;
        check_batch_dim("audio_f0", audio_f0.len(), batch)?;
        check_batch_dim("audio_voiced", audio_voiced.len(), batch)?;
        check_batch_dim("audio_energy", audio_energy.len(), batch)?;
        check_batch_dim("text_input_ids", text_input_ids.len(), batch)?;
        check_batch_dim("text_moras", text_moras.len(), batch)?;
        check_batch_dim("text_accent_types", text_accent_types.len(), batch)?;
        check_batch_dim(
            "text_phoneme_alignment",
            text_phoneme_alignment.len(),
            batch,
        )?;
        check_batch_dim("speaker_id", speaker_id.len(), batch)?;
        check_batch_dim("durations_ms", durations_ms.len(), batch)?;

        // Step 2: 各 sample の shape 整合性検証
        for (i, mel_sample) in audio_mel.iter().enumerate() {
            // n_mels check
            if mel_sample.len() != expected_n_mels {
                return Err(TtsBatchError::MelDimMismatch {
                    sample: i,
                    got: mel_sample.len(),
                    expected: expected_n_mels,
                });
            }

            // frame count = mel の第 3 dim (0 番目 mel の frame 数を代表として使う)
            let mel_frames = mel_sample.first().map_or(0, Vec::len);

            // 全 mel channel が同じ frame 数か
            for (m, channel) in mel_sample.iter().enumerate() {
                if channel.len() != mel_frames {
                    return Err(TtsBatchError::MelChannelFrameMismatch {
                        sample: i,
                        channel: m,
                        expected: mel_frames,
                        got: channel.len(),
                    });
                }
            }

            // f0 / voiced / energy と mel frames の一致
            check_frame_count(i, "audio_mel", mel_frames, "audio_f0", audio_f0[i].len())?;
            check_frame_count(
                i,
                "audio_mel",
                mel_frames,
                "audio_voiced",
                audio_voiced[i].len(),
            )?;
            check_frame_count(
                i,
                "audio_mel",
                mel_frames,
                "audio_energy",
                audio_energy[i].len(),
            )?;
        }

        // Step 3: mora 長整合性 (text_moras = text_phoneme_alignment = durations_ms)
        for i in 0..batch {
            let mora_len = text_moras[i].len();
            check_mora_len(
                i,
                "text_moras",
                mora_len,
                "text_phoneme_alignment",
                text_phoneme_alignment[i].len(),
            )?;
            check_mora_len(
                i,
                "text_moras",
                mora_len,
                "durations_ms",
                durations_ms[i].len(),
            )?;
        }

        Ok(Self {
            audio_waveform,
            audio_mel,
            audio_f0,
            audio_voiced,
            audio_energy,
            text_input_ids,
            text_moras,
            text_accent_types,
            text_phoneme_alignment,
            speaker_id,
            durations_ms,
            n_mels: expected_n_mels,
        })
    }

    /// Batch 内のサンプル数を返す。
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.audio_waveform.len()
    }

    /// mel 次元数を返す (通常 80)。
    #[must_use]
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// `audio_waveform[batch, samples]` への参照。
    #[must_use]
    pub fn audio_waveform(&self) -> &[Vec<f32>] {
        &self.audio_waveform
    }

    /// `audio_mel[batch, n_mels, frames]` への参照。
    #[must_use]
    pub fn audio_mel(&self) -> &[Vec<Vec<f32>>] {
        &self.audio_mel
    }

    /// `audio_f0[batch, frames]` への参照。Hz (voiced) または 0 (unvoiced)。
    #[must_use]
    pub fn audio_f0(&self) -> &[Vec<f32>] {
        &self.audio_f0
    }

    /// `audio_voiced[batch, frames]` への参照 (V/UV flag)。
    #[must_use]
    pub fn audio_voiced(&self) -> &[Vec<bool>] {
        &self.audio_voiced
    }

    /// `audio_energy[batch, frames]` への参照 (dB)。
    #[must_use]
    pub fn audio_energy(&self) -> &[Vec<f32>] {
        &self.audio_energy
    }

    /// `text_input_ids[batch, seq_len]` への参照 (tokenizer output)。
    #[must_use]
    pub fn text_input_ids(&self) -> &[Vec<u32>] {
        &self.text_input_ids
    }

    /// `text_moras[batch, mora_len]` への参照 (G2P output)。
    #[must_use]
    pub fn text_moras(&self) -> &[Vec<u8>] {
        &self.text_moras
    }

    /// `text_accent_types[batch, phrase_len]` への参照。
    #[must_use]
    pub fn text_accent_types(&self) -> &[Vec<u8>] {
        &self.text_accent_types
    }

    /// `text_phoneme_alignment[batch, mora_len]` への参照 (mora → audio frame index)。
    #[must_use]
    pub fn text_phoneme_alignment(&self) -> &[Vec<usize>] {
        &self.text_phoneme_alignment
    }

    /// `speaker_id[batch]` への参照。
    #[must_use]
    pub fn speaker_id(&self) -> &[u32] {
        &self.speaker_id
    }

    /// `durations_ms[batch, mora_len]` への参照 (duration predictor target)。
    #[must_use]
    pub fn durations_ms(&self) -> &[Vec<u32>] {
        &self.durations_ms
    }
}

fn check_batch_dim(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), TtsBatchError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TtsBatchError::BatchSizeMismatch {
            field,
            expected,
            actual,
        })
    }
}

fn check_frame_count(
    sample: usize,
    field_a: &'static str,
    frames_a: usize,
    field_b: &'static str,
    frames_b: usize,
) -> Result<(), TtsBatchError> {
    if frames_a == frames_b {
        Ok(())
    } else {
        Err(TtsBatchError::FrameCountMismatch {
            sample,
            field_a,
            frames_a,
            field_b,
            frames_b,
        })
    }
}

fn check_mora_len(
    sample: usize,
    field_a: &'static str,
    len_a: usize,
    field_b: &'static str,
    len_b: usize,
) -> Result<(), TtsBatchError> {
    if len_a == len_b {
        Ok(())
    } else {
        Err(TtsBatchError::MoraLengthMismatch {
            sample,
            field_a,
            len_a,
            field_b,
            len_b,
        })
    }
}

/// [`TtsBatch::new`] 検証で発生し得るエラー。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TtsBatchError {
    /// Batch サイズが 0 (最低 1 sample が必要)。
    EmptyBatch,
    /// Batch dimension が field 間で不一致。
    BatchSizeMismatch {
        /// 不一致を起こした field 名。
        field: &'static str,
        /// 期待した batch size (基準は `audio_waveform` の len)。
        expected: usize,
        /// 実際の batch size。
        actual: usize,
    },
    /// mel 第 2 dim が期待値と不一致。
    MelDimMismatch {
        /// 不一致を起こした sample index。
        sample: usize,
        /// 実際の mel dim。
        got: usize,
        /// 期待した mel dim (`expected_n_mels`)。
        expected: usize,
    },
    /// mel channel 間の frame 数が不一致 (同一 sample 内)。
    MelChannelFrameMismatch {
        /// sample index。
        sample: usize,
        /// mel channel index。
        channel: usize,
        /// 期待した frame 数 (channel 0 の frame 数)。
        expected: usize,
        /// 実際の frame 数。
        got: usize,
    },
    /// per-sample frame 数が field 間で不一致 (mel vs f0/voiced/energy)。
    FrameCountMismatch {
        /// sample index。
        sample: usize,
        /// 基準 field 名。
        field_a: &'static str,
        /// 基準 field の frame 数。
        frames_a: usize,
        /// 比較 field 名。
        field_b: &'static str,
        /// 比較 field の frame 数。
        frames_b: usize,
    },
    /// per-sample mora 長が field 間で不一致 (moras vs alignment/durations)。
    MoraLengthMismatch {
        /// sample index。
        sample: usize,
        /// 基準 field 名。
        field_a: &'static str,
        /// 基準 field の mora 長。
        len_a: usize,
        /// 比較 field 名。
        field_b: &'static str,
        /// 比較 field の mora 長。
        len_b: usize,
    },
}

impl fmt::Display for TtsBatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => write!(f, "empty batch: at least one sample required"),
            Self::BatchSizeMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "batch size mismatch: field '{field}' has {actual} samples, expected {expected}"
            ),
            Self::MelDimMismatch {
                sample,
                got,
                expected,
            } => write!(
                f,
                "mel dim mismatch at sample {sample}: got {got}, expected {expected}"
            ),
            Self::MelChannelFrameMismatch {
                sample,
                channel,
                expected,
                got,
            } => write!(
                f,
                "mel channel frame mismatch at sample {sample}, channel {channel}: expected {expected} frames (from channel 0), got {got}"
            ),
            Self::FrameCountMismatch {
                sample,
                field_a,
                frames_a,
                field_b,
                frames_b,
            } => write!(
                f,
                "frame count mismatch at sample {sample}: {field_a}={frames_a}, {field_b}={frames_b}"
            ),
            Self::MoraLengthMismatch {
                sample,
                field_a,
                len_a,
                field_b,
                len_b,
            } => write!(
                f,
                "mora length mismatch at sample {sample}: {field_a}={len_a}, {field_b}={len_b}"
            ),
        }
    }
}

impl std::error::Error for TtsBatchError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1 sample の最小 valid batch を生成するヘルパ。
    fn valid_batch_1sample() -> TtsBatch {
        TtsBatch::new(
            vec![vec![0.0_f32; 24_000]],
            vec![vec![vec![0.0_f32; 3]; 80]],
            vec![vec![100.0_f32, 110.0, 120.0]],
            vec![vec![true, true, true]],
            vec![vec![-20.0_f32, -18.0, -19.0]],
            vec![vec![1_u32, 2, 3]],
            vec![vec![0_u8, 1]],
            vec![vec![0_u8]],
            vec![vec![0_usize, 2]],
            vec![0_u32],
            vec![vec![50_u32, 40]],
            80,
        )
        .expect("valid batch must construct")
    }

    #[test]
    fn valid_1_sample_batch_constructs() {
        let batch = valid_batch_1sample();
        assert_eq!(batch.batch_size(), 1);
        assert_eq!(batch.n_mels(), 80);
        assert_eq!(batch.audio_waveform()[0].len(), 24_000);
        assert_eq!(batch.audio_mel()[0].len(), 80);
        assert_eq!(batch.audio_mel()[0][0].len(), 3);
        assert_eq!(batch.audio_f0()[0].len(), 3);
        assert_eq!(batch.audio_voiced()[0].len(), 3);
        assert_eq!(batch.audio_energy()[0].len(), 3);
        assert_eq!(batch.text_moras()[0].len(), 2);
        assert_eq!(batch.text_phoneme_alignment()[0].len(), 2);
        assert_eq!(batch.durations_ms()[0].len(), 2);
        assert_eq!(batch.speaker_id()[0], 0);
    }

    #[test]
    fn empty_batch_returns_empty_error() {
        let err = TtsBatch::new(
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            80,
        )
        .expect_err("empty batch must be rejected");
        assert_eq!(err, TtsBatchError::EmptyBatch);
    }

    #[test]
    fn mismatched_batch_size_across_fields_is_rejected() {
        // audio_waveform 2 samples, audio_mel 1 sample → mismatch
        let err = TtsBatch::new(
            vec![vec![0.0; 24_000], vec![0.0; 24_000]],
            vec![vec![vec![0.0; 3]; 80]],
            vec![vec![100.0; 3]],
            vec![vec![true; 3]],
            vec![vec![-20.0; 3]],
            vec![vec![1_u32]],
            vec![vec![0_u8]],
            vec![vec![0_u8]],
            vec![vec![0_usize]],
            vec![0_u32],
            vec![vec![50_u32]],
            80,
        )
        .expect_err("mismatched batch size must be rejected");
        assert!(matches!(
            err,
            TtsBatchError::BatchSizeMismatch {
                field: "audio_mel",
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn wrong_n_mels_is_rejected() {
        // expected 80 but got 40
        let err = TtsBatch::new(
            vec![vec![0.0; 24_000]],
            vec![vec![vec![0.0; 3]; 40]],
            vec![vec![100.0; 3]],
            vec![vec![true; 3]],
            vec![vec![-20.0; 3]],
            vec![vec![1_u32]],
            vec![vec![0_u8]],
            vec![vec![0_u8]],
            vec![vec![0_usize]],
            vec![0_u32],
            vec![vec![50_u32]],
            80,
        )
        .expect_err("wrong n_mels must be rejected");
        assert_eq!(
            err,
            TtsBatchError::MelDimMismatch {
                sample: 0,
                got: 40,
                expected: 80,
            }
        );
    }

    #[test]
    fn mel_f0_frame_count_mismatch_is_rejected() {
        // mel 3 frames, f0 5 frames → frame count mismatch
        let err = TtsBatch::new(
            vec![vec![0.0; 24_000]],
            vec![vec![vec![0.0; 3]; 80]],
            vec![vec![100.0; 5]],
            vec![vec![true; 3]],
            vec![vec![-20.0; 3]],
            vec![vec![1_u32]],
            vec![vec![0_u8]],
            vec![vec![0_u8]],
            vec![vec![0_usize]],
            vec![0_u32],
            vec![vec![50_u32]],
            80,
        )
        .expect_err("mel/f0 frame mismatch must be rejected");
        assert_eq!(
            err,
            TtsBatchError::FrameCountMismatch {
                sample: 0,
                field_a: "audio_mel",
                frames_a: 3,
                field_b: "audio_f0",
                frames_b: 5,
            }
        );
    }

    #[test]
    fn mora_alignment_length_mismatch_is_rejected() {
        // text_moras 2, alignment 3 → mismatch
        let err = TtsBatch::new(
            vec![vec![0.0; 24_000]],
            vec![vec![vec![0.0; 3]; 80]],
            vec![vec![100.0; 3]],
            vec![vec![true; 3]],
            vec![vec![-20.0; 3]],
            vec![vec![1_u32]],
            vec![vec![0_u8, 1]],
            vec![vec![0_u8]],
            vec![vec![0_usize, 2, 4]],
            vec![0_u32],
            vec![vec![50_u32, 40]],
            80,
        )
        .expect_err("mora/alignment mismatch must be rejected");
        assert_eq!(
            err,
            TtsBatchError::MoraLengthMismatch {
                sample: 0,
                field_a: "text_moras",
                len_a: 2,
                field_b: "text_phoneme_alignment",
                len_b: 3,
            }
        );
    }

    #[test]
    fn mora_durations_length_mismatch_is_rejected() {
        // text_moras 2, durations 3 → mismatch
        let err = TtsBatch::new(
            vec![vec![0.0; 24_000]],
            vec![vec![vec![0.0; 3]; 80]],
            vec![vec![100.0; 3]],
            vec![vec![true; 3]],
            vec![vec![-20.0; 3]],
            vec![vec![1_u32]],
            vec![vec![0_u8, 1]],
            vec![vec![0_u8]],
            vec![vec![0_usize, 2]],
            vec![0_u32],
            vec![vec![50_u32, 40, 30]],
            80,
        )
        .expect_err("mora/durations mismatch must be rejected");
        assert_eq!(
            err,
            TtsBatchError::MoraLengthMismatch {
                sample: 0,
                field_a: "text_moras",
                len_a: 2,
                field_b: "durations_ms",
                len_b: 3,
            }
        );
    }

    #[test]
    fn mel_channel_frame_mismatch_is_rejected() {
        // mel[0] 3 frames, mel[1] 5 frames → channel-level mismatch
        let mut mel = vec![vec![0.0_f32; 3]; 80];
        mel[1] = vec![0.0_f32; 5];
        let err = TtsBatch::new(
            vec![vec![0.0; 24_000]],
            vec![mel],
            vec![vec![100.0; 3]],
            vec![vec![true; 3]],
            vec![vec![-20.0; 3]],
            vec![vec![1_u32]],
            vec![vec![0_u8]],
            vec![vec![0_u8]],
            vec![vec![0_usize]],
            vec![0_u32],
            vec![vec![50_u32]],
            80,
        )
        .expect_err("mel channel frame mismatch must be rejected");
        assert_eq!(
            err,
            TtsBatchError::MelChannelFrameMismatch {
                sample: 0,
                channel: 1,
                expected: 3,
                got: 5,
            }
        );
    }

    #[test]
    fn error_display_is_human_readable() {
        let err = TtsBatchError::EmptyBatch;
        let s = format!("{err}");
        assert!(s.contains("empty batch"));

        let err = TtsBatchError::BatchSizeMismatch {
            field: "audio_mel",
            expected: 4,
            actual: 3,
        };
        let s = format!("{err}");
        assert!(s.contains("audio_mel"));
        assert!(s.contains('3'));
        assert!(s.contains('4'));
    }

    #[test]
    fn error_implements_std_error() {
        // std::error::Error trait bound を確認 (dyn Error として box 化可能)
        let err: Box<dyn std::error::Error> = Box::new(TtsBatchError::EmptyBatch);
        assert!(err.to_string().contains("empty batch"));
    }

    #[test]
    fn batch_is_cloneable_and_serializable() {
        let batch = valid_batch_1sample();
        let cloned = batch.clone();
        assert_eq!(cloned.batch_size(), batch.batch_size());

        // serde JSON roundtrip
        let json = serde_json::to_string(&batch).expect("serialize");
        let restored: TtsBatch = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.batch_size(), 1);
        assert_eq!(restored.n_mels(), 80);
    }
}
