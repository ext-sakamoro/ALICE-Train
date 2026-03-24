# ALICE-Train

Backpropagation & Training Framework for ALICE-ML ternary networks.

## Architecture

```
ALICE-ML (Inference)          ALICE-Train (Learning)
┌──────────────────┐          ┌─────────────────────────────┐
│ BitLinear        │          │ backward.rs                 │
│   forward()      │◀────────│   ternary_matvec_backward   │
│ TernaryWeight    │          │   bitlinear_backward        │
│ Loss / Optimizer │          │   ste_weight_grad           │
└──────────────────┘          ├─────────────────────────────┤
                              │ activation.rs               │
                              │   relu / silu / gelu        │
                              ├─────────────────────────────┤
                              │ trainer.rs                  │
                              │   TrainableNetwork trait    │
                              │   Trainer (grad accumulation)│
                              │   train_with_scheduler()    │
                              │   train_tokens()            │
                              ├─────────────────────────────┤
                              │ scheduler.rs                │
                              │   WarmupCosineScheduler     │
                              │   ConstantScheduler         │
                              ├─────────────────────────────┤
                              │ checkpoint.rs               │
                              │   ALICETRN binary format    │
                              │   save / load               │
                              ├─────────────────────────────┤
                              │ dataloader.rs               │
                              │   MmapDataset (memmap2)     │
                              │   DataLoader + Batch        │
                              ├─────────────────────────────┤
                              │ evaluator.rs                │
                              │   perplexity evaluation     │
                              │   BestCheckpointTracker     │
                              ├─────────────────────────────┤
                              │ logger.rs                   │
                              │   TrainLog (CSV / JSON)     │
                              │   compute_grad_norm         │
                              ├─────────────────────────────┤
                              │ mixed_precision.rs          │
                              │   Bf16 conversion           │
                              │   LossScaler (dynamic)      │
                              ├─────────────────────────────┤
                              │ qat.rs                      │
                              │   FakeQuantize              │
                              │   QatTrainer                │
                              │   CalibrationStats          │
                              ├─────────────────────────────┤
                              │ distill.rs                  │
                              │   DistillTrainer            │
                              │   KL-div + hard label mix   │
                              ├─────────────────────────────┤
                              │ pipeline.rs                 │
                              │   QatPipeline (orchestrator)│
                              │   FP32→BF16→Ternary loop   │
                              ├─────────────────────────────┤
                              │ offload.rs                  │
                              │   OffloadOptimizer (AdamW)  │
                              │   ZeRO-Offload m/v→CPU RAM │
                              ├─────────────────────────────┤
                              │ gpu.rs          [gpu feature]│
                              │   GpuContext (wgpu)         │
                              ├─────────────────────────────┤
                              │ gpu_backward.rs [gpu feature]│
                              │   GpuBackwardEngine         │
                              │   WGSL compute shader       │
                              └─────────────────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| Ternary backward | W^T * dy using add/sub only (no multiplication) |
| RMSNorm backward | Full gradient through pre-normalization |
| STE weight grad | Straight-Through Estimator for latent FP32 weights |
| Activation backward | ReLU, SiLU, GELU with numerical gradient verification |
| Training loop | `TrainableNetwork` trait + `Trainer` with MSE/CE/MAE loss |
| Gradient accumulation | Micro-batch accumulation for effective batch size scaling |
| LR scheduling | Warmup + Cosine Decay / Constant scheduler |
| Checkpoint | Binary format (ALICETRN magic + JSON header + raw weights) |
| Memory-mapped data | `MmapDataset` for large token files via memmap2 |
| Token-based training | `train_tokens()` with DataLoader + scheduler integration |
| Perplexity evaluation | `evaluate()` + `BestCheckpointTracker` for auto-save |
| Training log | CSV / JSON export of loss, lr, grad_norm per step |
| Mixed precision | BF16 conversion + dynamic loss scaling (NaN/Inf detection) |
| QAT | `FakeQuantize`, `QatTrainer`, `CalibrationStats` |
| Knowledge distillation | KL-divergence + hard label mixed loss |
| QAT Pipeline | Full orchestration: FP32→BF16→Ternary with scheduler, checkpoint, eval |
| GPU backward | wgpu compute shader for `ternary_matvec_backward` (feature: `gpu`) |
| ZeRO-Offload | AdamW optimizer state (m/v) offloaded to CPU RAM — 50% VRAM reduction |

## Quick Start

```rust
use alice_train::{ternary_matvec_backward, relu_backward};
use alice_ml::ops::TernaryWeightKernel;

// Ternary weights W = [[1, -1], [0, 1]]
let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);

// Backward: dy -> dx = W^T * dy
let grad_output = [1.0_f32, 1.0];
let mut grad_input = [0.0_f32; 2];
ternary_matvec_backward(&grad_output, &kernel, &mut grad_input);

assert!((grad_input[0] - 1.0).abs() < 1e-6);
assert!((grad_input[1] - 0.0).abs() < 1e-6);
```

### Training with scheduler and checkpoint

```rust
use alice_train::{
    Trainer, TrainConfig, WarmupCosineScheduler,
};

let config = TrainConfig::new()
    .with_epochs(10)
    .with_learning_rate(0.001)
    .with_gradient_accumulation(4)
    .with_checkpoint(5, "checkpoints");
let trainer = Trainer::new(config);

// max_lr, min_lr, warmup_steps, total_steps
let scheduler = WarmupCosineScheduler::new(0.001, 1e-5, 100, 1000);

let (results, log) = trainer.train_with_scheduler(
    &mut network, &inputs, &targets, mse_loss, &scheduler, None,
);
log.save_csv_to_file("train_log.csv").unwrap();
```

### BF16 mixed precision

```rust
use alice_train::{LossScaler, MixedPrecisionConfig, f32_to_bf16_vec};

let config = MixedPrecisionConfig::default(); // dynamic scaling enabled
let mut scaler = LossScaler::new(config);

let weights_bf16 = f32_to_bf16_vec(&weights_f32);
let scaled_loss = scaler.scale_loss(loss);
// ... backward ...
scaler.unscale_gradients(&mut gradients);
let valid = LossScaler::check_gradients(&gradients);
scaler.update(valid);
```

### ZeRO-Offload — VRAM 50% reduction

```rust
use alice_train::{OffloadOptimizer, OffloadConfig, MemoryBudget};

// 7B model memory estimate
let budget = MemoryBudget::estimate(7_000_000_000);
// VRAM: 56 GB (weights + gradients only)
// CPU RAM: 56 GB (m + v offloaded)
// Without offload: 112 GB VRAM needed

let config = OffloadConfig {
    beta1: 0.9,
    beta2: 0.999,
    weight_decay: 0.01,
    max_grad_norm: Some(1.0),
    ..OffloadConfig::default()
};
let mut optimizer = OffloadOptimizer::new(param_count, config);

// Training loop: GPU forward/backward → CPU update
optimizer.step(&mut weights, &mut gradients, lr);
```

### GPU backward (feature: `gpu`)

```rust
use alice_train::{GpuContext, GpuBackwardEngine};

let ctx = GpuContext::new_blocking().expect("GPU required");
let engine = GpuBackwardEngine::new(&ctx);

// GPU-accelerated: dx = W^T * dy
engine.ternary_matvec_backward(&grad_output, &kernel, &mut grad_input);
// Bit-exact match with CPU version
```

### QAT Pipeline — FP32 → Ternary

```rust
use alice_train::pipeline::{QatPipeline, QatPipelineConfig};
use alice_train::mixed_precision::MixedPrecisionConfig;

let config = QatPipelineConfig {
    epochs: 100,
    learning_rate: 1e-4,
    min_lr: 1e-6,
    warmup_steps: 100,
    gradient_accumulation_steps: 4,
    eval_interval: 5,
    mixed_precision: MixedPrecisionConfig::default(), // BF16 enabled
    ..QatPipelineConfig::default()
};
let mut pipeline = QatPipeline::new(config);

let result = pipeline.run(
    &mut latent_weights,
    &train_data,       // &[(Vec<f32>, Vec<f32>)]
    &forward_fn,       // |weights, input, output|
    &loss_fn,          // |output, target, grad| -> loss
    Some(&eval_data),
);

// Export final ternary weights
let mut ternary = vec![0.0f32; latent_weights.len()];
pipeline.finalize_weights(&latent_weights, &mut ternary);
```

## Design Decisions

- **Separate crate**: ALICE-ML stays `no_std` / zero-allocation for inference; training requires `std` and heap allocation for gradient buffers.
- **DPS pattern**: All backward functions write into caller-provided buffers (`grad_input: &mut [f32]`).
- **STE for ternary weights**: Discrete {-1, 0, +1} weights can't be differentiated directly. Latent FP32 weights are maintained and quantized during forward.
- **Loss/Optimizer reuse**: `alice_ml::training` provides MSE, CrossEntropy, MAE, SGD, Adam.
- **Binary checkpoint format**: ALICETRN magic + JSON metadata header + raw f32 weights + optimizer state. Compact and fast to load.
- **Byte-level mmap access**: `MmapDataset` reads tokens via `u32::from_le_bytes()` instead of pointer casting, avoiding alignment issues.
- **Dynamic loss scaling**: `LossScaler` tracks consecutive good steps, grows scale on stability, halves on NaN/Inf. Floor at 1.0.
- **Callback-based token training**: `train_tokens()` takes `token_embed_fn` / `target_embed_fn` closures to decouple token representation from the training loop.
- **GPU backward via wgpu**: WGSL compute shader mirrors CPU logic. Each thread handles one `grad_input[col]`, iterating over all rows. Feature-gated (`gpu`).
- **ZeRO-Offload**: AdamW m/v stored in CPU RAM. Reduces VRAM from 4N to 2N parameters. Includes gradient clipping, bias correction, and memory budget estimation.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `alice-ml` | Inference engine (forward, loss, optimizer) |
| `memmap2` | Memory-mapped file I/O for large datasets |
| `rand` | Shuffle for DataLoader |
| `serde` | Checkpoint metadata serialization |
| `serde_json` | JSON format for checkpoint header and training log |
| `wgpu` | GPU compute (optional, feature: `gpu`) |
| `pollster` | Async→sync bridge for wgpu (optional, feature: `gpu`) |
| `bytemuck` | Zero-copy GPU buffer casting (optional, feature: `gpu`) |

## Quality

| Metric | Value |
|--------|-------|
| Tests | 322 (gpu feature含む) |
| Doc-tests | 1 |
| Clippy (pedantic+nursery) | 0 warnings |
| Doc warnings | 0 |
| fmt | clean |
| Score | 100/100 |

## License

AGPL-3.0
