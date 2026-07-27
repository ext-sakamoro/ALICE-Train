//! TTS 学習用 dataset loader (Feature Request R2)。
//!
//! JSUT / JVS / JVNV 等の TTS corpus を manifest jsonl 形式で管理し、
//! [`TtsBatch`](super::TtsBatch) を batch 化して返す loader を提供する。
//!
//! # サブモジュール
//!
//! | Module | 内容 |
//! |---|---|
//! | [`manifest`] | `TtsManifestEntry` / JSONL parse / エラー型 |
//! | [`loader`] | `TtsDataset` (from_manifest / split / iter_batches) |
//!
//! # Manifest 形式
//!
//! 1 行 = 1 sample の JSON。詳細は [`manifest::TtsManifestEntry`] 参照。
//!
//! ```jsonl
//! {"audio_path":"jvs001/BASIC_0001.wav","text_input_ids":[1,2,3],"text_moras":[0,1],"text_accent_types":[0],"phoneme_alignment_ms":[0,50],"speaker_id":0,"durations_ms":[50,70]}
//! ```
//!
//! # 例
//!
//! ```rust,no_run
//! # #[cfg(feature = "tts")] {
//! use alice_train::tts::{AudioFeatureExtractor, TtsDataset};
//!
//! let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
//! let dataset = TtsDataset::from_manifest("data/jvs.jsonl", "/data/jvs", extractor)
//!     .expect("manifest load");
//!
//! for batch_result in dataset.iter_batches(4) {
//!     let batch = batch_result.expect("batch load");
//!     assert!(batch.batch_size() <= 4);
//!     // trainer.step(&batch)?;
//! }
//! # }
//! ```

pub mod loader;
pub mod manifest;

pub use loader::TtsDataset;
pub use manifest::{TtsDatasetError, TtsManifestEntry};
