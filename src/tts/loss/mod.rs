//! TTS 学習用損失関数 (Feature Request R3)。
//!
//! Phase T.2 (Prosody Model) / Phase T.4a (FastSpeech2) の joint 学習で使用される
//! 損失関数群を提供する。
//!
//! # サブモジュール
//!
//! | Module | 内容 |
//! |---|---|
//! | [`prosody`] | `ProsodyLoss` (F0 L1 + duration MSE + energy MSE 重み付き joint) |
//!
//! # 例
//!
//! ```rust
//! # #[cfg(feature = "tts")] {
//! use alice_train::tts::{LossComponents, ProsodyLoss, ProsodyPrediction, ProsodyTarget};
//!
//! let pred = ProsodyPrediction {
//!     f0: vec![vec![100.0, 110.0]],
//!     duration_frames: vec![vec![5.0, 4.0]],
//!     energy: vec![vec![-20.0, -15.0]],
//! };
//! let target = ProsodyTarget {
//!     f0: vec![vec![105.0, 108.0]],
//!     duration_frames: vec![vec![5.0, 4.0]],
//!     energy: vec![vec![-18.0, -14.0]],
//!     mask: None,
//! };
//!
//! let loss = ProsodyLoss::default_weights();
//! let LossComponents {
//!     f0_l1,
//!     duration_mse,
//!     energy_mse,
//!     total,
//! } = loss.compute(&pred, &target).expect("shape match");
//!
//! assert!(f0_l1 > 0.0);
//! assert!(duration_mse.abs() < 1e-6); // duration 完全一致
//! assert!(energy_mse > 0.0);
//! assert!((total - (1.0 * f0_l1 + 1.0 * duration_mse + 0.5 * energy_mse)).abs() < 1e-4);
//! # }
//! ```

pub mod prosody;

pub use prosody::{
    LossComponents, ProsodyLoss, ProsodyLossError, ProsodyPrediction, ProsodyTarget,
};
