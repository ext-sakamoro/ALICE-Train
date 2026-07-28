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
//! | [`layer_norm`] | LayerNorm (forward + 手書き backward、PyTorch nn.LayerNorm 互換) |
//! | [`multi_head_attention`] | Multi-Head Attention (self + cross、causal / bidirectional、forward + backward) |
//! | [`positional_encoding`] | Sinusoidal (Vaswani 2017) + Rotary (RoPE, Su 2021) positional encoding |
//! | [`weight_norm`] | WeightNorm reparametrization `w = g * v / ||v||` (Salimans & Kingma 2016) |
//!
//! # ロードマップ (Phase T.0-P)
//!
//! - ✅ Conv1D + backward
//! - ✅ LayerNorm + backward
//! - ✅ MultiHeadAttention + backward (causal / bidirectional 両対応、self + cross)
//! - ✅ PositionalEncoding (sinusoidal + rotary)
//! - ✅ WeightNorm (Vocos / HiFi-GAN 用の weight reparametrization)
//!
//! **Phase T.0-P 100% 完了、Phase T.4a FastSpeech2 architecture 実装着手可能。**
//!
//! # 参照
//!
//! - `~/ALICE-TTS/docs/v2/ARCHITECTURE.md` §AD-7 — 手書き backward の設計方針
//! - `~/ALICE-TTS/docs/v2/ROADMAP.md` Phase T.0-P
//! - Vocos primitives (`~/ALICE-TTS/crates/alice-tts-vocoder/src/primitives.rs`) — 将来集約対象

pub mod conv1d;
pub mod layer_norm;
pub mod linear;
pub mod multi_head_attention;
pub mod positional_encoding;
pub mod weight_norm;

pub use conv1d::{Conv1d, Conv1dConfig, Conv1dError};
pub use layer_norm::{LayerNorm, LayerNormConfig, LayerNormError};
pub use linear::{Linear, LinearConfig, LinearError};
pub use multi_head_attention::{MhaError, MhaGrads, MultiHeadAttention, MultiHeadAttentionConfig};
pub use positional_encoding::{
    PosEncError, RotaryEmbedding, RotaryEmbeddingConfig, SinusoidalPositionalEncoding,
    SinusoidalPositionalEncodingConfig,
};
pub use weight_norm::{WeightNorm, WeightNormConfig, WeightNormError};
