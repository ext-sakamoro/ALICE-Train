//! ALICE-Train: Backpropagation & Training Framework
//!
//! > "Forward is inference. Backward is learning."
//!
//! ALICE-ML の推論エンジンに逆伝播を追加し、学習フレームワークとして完成させる。
//!
//! # 設計
//!
//! - ternary 重み {-1, 0, +1} は離散値 → 直接微分不可
//! - Straight-Through Estimator (STE) で潜在 FP32 重みを保持し、forward 時に量子化
//! - backward は DPS パターン（勾配バッファを呼び出し側が渡す）
//!
//! # モジュール
//!
//! | モジュール | 内容 |
//! |-----------|------|
//! | [`activation`] | 活性化関数の backward (`ReLU`, `SiLU`, GELU) |
//! | [`backward`] | レイヤー逆伝播 (ternary matvec transpose, `BitLinear` backward, STE) |
//! | [`trainer`] | 学習ループ (`TrainableNetwork` トレイト, `Trainer`) |
//!
//! # Quick Start
//!
//! ```rust
//! use alice_train::{ternary_matvec_backward, relu_backward};
//! use alice_ml::ops::TernaryWeightKernel;
//!
//! // ternary 重み W = [[1, -1], [0, 1]]
//! let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
//!
//! // forward: y = W * x → y = [-1, 3] (done by alice-ml)
//! let input = [2.0_f32, 3.0];
//!
//! // backward: 出力勾配 dy → 入力勾配 dx = W^T * dy
//! let grad_output = [1.0_f32, 1.0];
//! let mut grad_input = [0.0_f32; 2];
//! ternary_matvec_backward(&grad_output, &kernel, &mut grad_input);
//!
//! // W^T = [[1, 0], [-1, 1]], dx = [1, 0]
//! assert!((grad_input[0] - 1.0).abs() < 1e-6);
//! assert!((grad_input[1] - 0.0).abs() < 1e-6);
//!
//! // 活性化 backward
//! let activations = [1.0_f32, -0.5, 3.0];
//! let grad_out = [1.0, 1.0, 1.0];
//! let mut grad_in = [0.0_f32; 3];
//! relu_backward(&activations, &grad_out, &mut grad_in);
//! assert_eq!(grad_in, [1.0, 0.0, 1.0]);
//! ```

// clippy 設定: ALICE 品質基準 pedantic+nursery
#![warn(clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::too_many_lines
)]
#![warn(missing_docs)]

pub mod activation;
pub mod backward;
pub mod trainer;

// Re-exports
pub use activation::{gelu_backward, relu_backward, silu_backward};
pub use backward::{bitlinear_backward, ste_weight_grad, ternary_matvec_backward};
pub use trainer::{EpochResult, TrainConfig, TrainableNetwork, Trainer};
