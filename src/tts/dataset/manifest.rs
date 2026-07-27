//! TTS manifest entry (1 sample per JSON line) + parse エラー型。
//!
//! Manifest JSONL 形式は G2P 済 + speaker アノテーション済のデータセット (JSUT + JVS + JVNV 等) を
//! ALICE-Train へ渡すための中間表現。実 audio ファイルは manifest から相対パスで参照される。

use serde::{Deserialize, Serialize};
use std::io;

/// TTS manifest の 1 サンプル (JSONL の 1 行)。
///
/// # Field 一覧
///
/// | Field | 型 | 用途 |
/// |---|---|---|
/// | `audio_path` | String | `audio_root` からの相対パス (例: `"jvs001/BASIC_0001.wav"`) |
/// | `text_input_ids` | `Vec<u32>` | tokenizer output |
/// | `text_moras` | `Vec<u8>` | G2P output |
/// | `text_accent_types` | `Vec<u8>` | accent phrase 単位 |
/// | `phoneme_alignment_ms` | `Vec<u32>` | mora 開始時刻 (ms、audio 先頭 = 0) |
/// | `speaker_id` | u32 | speaker embedding lookup index |
/// | `durations_ms` | `Vec<u32>` | mora 継続時間 (ms、duration predictor target) |
///
/// `phoneme_alignment_ms` は Load 時に `hop_length / sample_rate` で frame index に変換される。
///
/// # 例
///
/// ```rust
/// # #[cfg(feature = "tts")] {
/// use alice_train::tts::TtsManifestEntry;
///
/// let json = r#"{"audio_path":"a.wav","text_input_ids":[1,2],"text_moras":[0,1],"text_accent_types":[0],"phoneme_alignment_ms":[0,50],"speaker_id":0,"durations_ms":[50,70]}"#;
/// let entry: TtsManifestEntry = serde_json::from_str(json).expect("parse");
/// assert_eq!(entry.audio_path, "a.wav");
/// assert_eq!(entry.text_moras, vec![0, 1]);
/// # }
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TtsManifestEntry {
    /// `audio_root` からの相対 WAV パス。
    pub audio_path: String,
    /// tokenizer output。
    pub text_input_ids: Vec<u32>,
    /// G2P output (mora id 列)。
    pub text_moras: Vec<u8>,
    /// accent phrase 単位の accent type。
    pub text_accent_types: Vec<u8>,
    /// mora 開始時刻 (ms、audio 先頭 = 0)。長さは `text_moras.len()` と一致必須。
    pub phoneme_alignment_ms: Vec<u32>,
    /// speaker embedding lookup index。
    pub speaker_id: u32,
    /// mora 継続時間 (ms)。長さは `text_moras.len()` と一致必須。
    pub durations_ms: Vec<u32>,
}

impl TtsManifestEntry {
    /// entry の内部整合性を検証する。
    ///
    /// # Errors
    ///
    /// - mora 長不一致 (moras / alignment / durations 三者)
    /// - `audio_path` が空文字列
    pub fn validate(&self) -> Result<(), TtsDatasetError> {
        if self.audio_path.is_empty() {
            return Err(TtsDatasetError::InvalidEntry {
                reason: "audio_path is empty".to_string(),
            });
        }
        let mora_len = self.text_moras.len();
        if self.phoneme_alignment_ms.len() != mora_len {
            return Err(TtsDatasetError::InvalidEntry {
                reason: format!(
                    "phoneme_alignment_ms length {} != text_moras length {mora_len}",
                    self.phoneme_alignment_ms.len()
                ),
            });
        }
        if self.durations_ms.len() != mora_len {
            return Err(TtsDatasetError::InvalidEntry {
                reason: format!(
                    "durations_ms length {} != text_moras length {mora_len}",
                    self.durations_ms.len()
                ),
            });
        }
        Ok(())
    }
}

/// TtsDataset 操作で発生し得るエラー。
#[derive(Debug)]
pub enum TtsDatasetError {
    /// I/O エラー (manifest read / WAV load)。
    Io(io::Error),
    /// JSONL parse エラー (1 行内の JSON が不正)。
    JsonParse {
        /// 対象行番号 (1-indexed)。
        line: usize,
        /// serde_json のエラーメッセージ。
        source: serde_json::Error,
    },
    /// Manifest entry の整合性違反 (mora 長不一致等)。
    InvalidEntry {
        /// 具体的な理由。
        reason: String,
    },
    /// WAV フォーマットエラー (非対応の sample format 等)。
    WavFormat {
        /// 対象ファイルパス。
        path: String,
        /// エラー詳細。
        reason: String,
    },
    /// Batch 構築時の TtsBatch validation error。
    BatchConstruction {
        /// 内部 error message。
        source: String,
    },
    /// split 比率が不正 (負値 or 合計が 0)。
    InvalidSplitRatio {
        /// 引数の (train, valid, test) 比率。
        ratios: (f32, f32, f32),
    },
    /// 空 manifest (1 entry も無い)。
    EmptyManifest,
}

impl std::fmt::Display for TtsDatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::JsonParse { line, source } => {
                write!(f, "JSON parse error at line {line}: {source}")
            }
            Self::InvalidEntry { reason } => write!(f, "invalid manifest entry: {reason}"),
            Self::WavFormat { path, reason } => {
                write!(f, "WAV format error in {path}: {reason}")
            }
            Self::BatchConstruction { source } => write!(f, "batch construction failed: {source}"),
            Self::InvalidSplitRatio { ratios } => {
                write!(f, "invalid split ratio: {ratios:?}")
            }
            Self::EmptyManifest => write!(f, "manifest is empty (no entries)"),
        }
    }
}

impl std::error::Error for TtsDatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::JsonParse { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<io::Error> for TtsDatasetError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_manifest_entry() {
        let json = r#"{"audio_path":"a.wav","text_input_ids":[1,2],"text_moras":[0,1],"text_accent_types":[0],"phoneme_alignment_ms":[0,50],"speaker_id":0,"durations_ms":[50,70]}"#;
        let entry: TtsManifestEntry = serde_json::from_str(json).expect("parse");
        assert_eq!(entry.audio_path, "a.wav");
        assert_eq!(entry.text_moras.len(), 2);
        entry.validate().expect("valid");
    }

    #[test]
    fn missing_field_returns_parse_error() {
        // audio_path 抜け → serde_json エラー
        let json = r#"{"text_input_ids":[1],"text_moras":[0],"text_accent_types":[0],"phoneme_alignment_ms":[0],"speaker_id":0,"durations_ms":[50]}"#;
        let result: Result<TtsManifestEntry, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn validate_rejects_empty_audio_path() {
        let entry = TtsManifestEntry {
            audio_path: String::new(),
            text_input_ids: vec![1],
            text_moras: vec![0],
            text_accent_types: vec![0],
            phoneme_alignment_ms: vec![0],
            speaker_id: 0,
            durations_ms: vec![50],
        };
        let err = entry.validate().expect_err("empty path must fail");
        assert!(matches!(err, TtsDatasetError::InvalidEntry { .. }));
    }

    #[test]
    fn validate_rejects_alignment_length_mismatch() {
        let entry = TtsManifestEntry {
            audio_path: "a.wav".to_string(),
            text_input_ids: vec![1],
            text_moras: vec![0, 1], // 2
            text_accent_types: vec![0],
            phoneme_alignment_ms: vec![0], // 1 ← mismatch
            speaker_id: 0,
            durations_ms: vec![50, 70],
        };
        let err = entry.validate().expect_err("length mismatch must fail");
        let msg = format!("{err}");
        assert!(msg.contains("phoneme_alignment_ms"));
    }

    #[test]
    fn validate_rejects_durations_length_mismatch() {
        let entry = TtsManifestEntry {
            audio_path: "a.wav".to_string(),
            text_input_ids: vec![1],
            text_moras: vec![0, 1],
            text_accent_types: vec![0],
            phoneme_alignment_ms: vec![0, 50],
            speaker_id: 0,
            durations_ms: vec![50], // ← mismatch
        };
        let err = entry.validate().expect_err("length mismatch must fail");
        let msg = format!("{err}");
        assert!(msg.contains("durations_ms"));
    }

    #[test]
    fn error_display_and_source_work() {
        let e = TtsDatasetError::EmptyManifest;
        assert_eq!(format!("{e}"), "manifest is empty (no entries)");

        let io_err = io::Error::new(io::ErrorKind::NotFound, "not found");
        let e = TtsDatasetError::Io(io_err);
        assert!(format!("{e}").contains("I/O error"));
        assert!(std::error::Error::source(&e).is_some());
    }

    #[test]
    fn error_from_io_error_works() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "denied");
        let e: TtsDatasetError = io_err.into();
        assert!(matches!(e, TtsDatasetError::Io(_)));
    }
}
