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
//! | [`trainer`] | 学習ループ (`TrainableNetwork` トレイト, `Trainer`, 勾配累積) |
//! | [`qat`] | 量子化再学習 (`FakeQuantize`, `QatTrainer`, `CalibrationStats`) |
//! | [`distill`] | 知識蒸留 (`DistillTrainer`, KL-divergence + hard label 混合損失) |
//! | [`checkpoint`] | 重み保存/復元 (`CheckpointData`, バイナリフォーマット) |
//! | [`dataloader`] | メモリマップデータ読み込み (`MmapDataset`, `DataLoader`) |
//! | [`scheduler`] | 学習率スケジューラ (warmup + cosine decay) |
//! | [`evaluator`] | perplexity 算出、ベストチェックポイント自動保存 |
//! | [`logger`] | loss/lr/grad_norm の CSV/JSON 記録 |
//! | [`mixed_precision`] | BF16 変換、動的 loss scaling |
//! | [`pipeline`] | QAT パイプライン (FP32→BF16→Ternary 統合ループ) |
//! | [`offload`] | ZeRO-Offload (AdamW m/v CPU RAM オフロード) |
//! | [`llama`] | Llama-3 アーキテクチャ定義 (QAT 学習用レイヤー構造) |
//! | `gpu` | GPU コンテキスト (wgpu Device/Queue) *(feature: `gpu`)* |
//! | `gpu_backward` | GPU ternary backward (compute shader) *(feature: `gpu`)* |
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
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::missing_panics_doc,
    clippy::too_many_arguments,
    clippy::suboptimal_flops,
    clippy::type_complexity,
    clippy::match_same_arms
)]
#![warn(missing_docs)]

pub mod activation;
pub mod backward;
pub mod blas;
pub mod checkpoint;
pub mod dataloader;
pub mod deltanet;
pub mod distill;
pub mod evaluator;
pub mod export;
pub mod fast_math;
pub mod fp32_cache;
pub mod inference;
pub mod llama;
pub mod llama_backward;
pub mod llama_forward;
pub mod logger;
pub mod mixed_precision;
pub mod offload;
pub mod pipeline;
pub mod qat;
pub mod qwen35;
pub mod qwen35_backward;
pub mod qwen35_forward;
#[cfg(feature = "qat-cli")]
pub mod safetensors_loader;
pub mod scheduler;
pub mod tokenizer;
pub mod trainer;

// GPU モジュール (feature gate)
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod gpu_backward;
#[cfg(feature = "gpu")]
pub mod gpu_matmul;

// CUDA モジュール (feature gate)
#[cfg(feature = "cuda")]
pub mod cuda_matmul;

// Re-exports
pub use activation::{gelu_backward, relu_backward, silu_backward};
pub use backward::{bitlinear_backward, ste_weight_grad, ternary_matvec_backward};
pub use checkpoint::{CheckpointData, CheckpointMeta};
pub use dataloader::{Batch, DataLoader, DataLoaderConfig, MmapDataset};
pub use distill::{DistillConfig, DistillEpochResult, DistillTrainer};
pub use evaluator::{evaluate, BestCheckpointTracker, EvalResult};
pub use export::{
    dequantize, export_alice_model, pack_ternary, quantize_and_pack, read_alice_meta,
    unpack_ternary, AliceModelMeta, ExportStats, LayerScales,
};
pub use logger::{compute_grad_norm, LogEntry, TrainLog};
pub use mixed_precision::{
    bf16_to_f32_batch, bf16_to_f32_vec, f32_to_bf16_batch, f32_to_bf16_vec, Bf16, LossScaler,
    MixedPrecisionConfig,
};
pub use offload::{MemoryBudget, OffloadConfig, OffloadOptimizer};
pub use pipeline::{QatEpochSummary, QatPipeline, QatPipelineConfig, QatRunResult, QatStepResult};
pub use qat::{CalibrationStats, FakeQuantize, QatConfig, QatEpochResult, QatTrainer, QuantBits};
pub use scheduler::{ConstantScheduler, LrScheduler, WarmupCosineScheduler};
pub use trainer::{EpochResult, TrainConfig, TrainableNetwork, Trainer};

// GPU re-exports
#[cfg(feature = "gpu")]
pub use gpu::GpuContext;
#[cfg(feature = "gpu")]
pub use gpu_backward::GpuBackwardEngine;
#[cfg(feature = "gpu")]
pub use gpu_matmul::GpuMatmul;

// CUDA re-exports
#[cfg(feature = "cuda")]
pub use cuda_matmul::{
    cuda_deltanet_recurrence, CudaLayerWorkspace, CudaMatmul, LayerWeightGrads, VramLayerWeights,
};
