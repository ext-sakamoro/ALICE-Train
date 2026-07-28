//! TTS 学習用 primitive layer (Phase T.0-P、AD-7 準拠)。
//!
//! FastSpeech2 (Phase T.4a) / VITS2 (Phase T.4b) / Vocos (Phase T.5) の共通レイヤーを
//! `feature = "tts"` gate 内に実装する。既存 Qwen 系 layer とは別に、TTS 汎用 primitive
//! (Conv1D / LayerNorm / MultiHeadAttention 等) を提供する。
//!
//! # サブモジュール
//!
//! | Module | 内容 |
//! |---|---|
//! | [`conv1d`] | 1D convolution (forward + 手書き backward、grouped/depthwise 対応) |
//!
//! # ロードマップ (Phase T.0-P)
//!
//! - ✅ Conv1D + backward (本 module)
//! - ⏳ LayerNorm + backward
//! - ⏳ MultiHeadAttention + backward (causal / bidirectional 両対応)
//! - ⏳ PositionalEncoding (sinusoidal + rotary)
//! - ⏳ WeightNorm (optional、Vocos style)
//!
//! # 参照
//!
//! - `~/ALICE-TTS/docs/v2/ARCHITECTURE.md` §AD-7 — 手書き backward の設計方針
//! - `~/ALICE-TTS/docs/v2/ROADMAP.md` Phase T.0-P
//! - Vocos primitives (`~/ALICE-TTS/crates/alice-tts-vocoder/src/primitives.rs`) — 将来集約対象

pub mod conv1d;

pub use conv1d::{Conv1d, Conv1dConfig, Conv1dError};
