//! ALICE-TTS v2.0 対応モジュール — TTS 学習用の型・データローダー・損失関数を提供。
//!
//! ALICE-TTS v2.0 (Path α: 完全独自 Neural TTS from-scratch) の Phase T.0-R
//! (Feature Request R1-R4) の実装を担う。既存 LLM 学習パイプラインには影響しないよう
//! `feature = "tts"` gate で切り離されている。
//!
//! # Feature Request
//!
//! - **R1**: [`TtsBatch`] — 音声 + テキスト + prosody + alignment + speaker + duration の
//!   マルチモーダルバッチ型 (実装済)
//! - **R2**: [`TtsDataset`] — JSUT/JVS/JVNV manifest からの batch loader (実装済)
//! - **R3**: `ProsodyLoss` — F0/duration/energy の重み付き joint loss (未実装、Phase T.0-R 継続)
//! - **R4**: [`AudioFeatureExtractor`] — STFT + mel filterbank + F0 (YIN) + energy 抽出 (実装済)
//!
//! # 関連ドキュメント
//!
//! - `~/ALICE-TTS/docs/v2/feature-requests/alice-train-tts-support.md` — Feature Request 詳細
//! - `~/ALICE-TTS/docs/v2/ARCHITECTURE.md` §AD-7 — Rust 学習 pipeline は ALICE-Train 派生

pub mod audio;
pub mod batch;
pub mod dataset;

pub use audio::AudioFeatureExtractor;
pub use batch::{TtsBatch, TtsBatchError};
pub use dataset::{TtsDataset, TtsDatasetError, TtsManifestEntry};
