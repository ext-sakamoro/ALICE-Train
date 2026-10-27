//! TTS 用の音声特徴量と data loader
//!
//! 学習用の batch 型 ([`TtsBatch`])、manifest からの loader ([`TtsDataset`])、音声特徴量
//! ([`AudioFeatureExtractor`]: STFT + mel filterbank + F0 + energy) を提供する F0 の抽出器は
//! [`crate::F0Extractor`] で差し替えられる 既存の LLM 学習 pipeline に影響しないよう
//! `feature = "tts"` で切り離している
//!
//! 音響 model (以前の FastSpeech2) とその学習はこの crate から外した

pub mod audio;
pub mod batch;
pub mod dataset;

pub use audio::{AudioFeatureExtractor, Yin};
pub use batch::{TtsBatch, TtsBatchError};
pub use dataset::{TtsDataset, TtsDatasetError, TtsManifestEntry};
