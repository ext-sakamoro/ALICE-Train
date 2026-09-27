# ALICE-Train

Backpropagation & Training Framework for ALICE-ML ternary networks.

[日本語 README はこちら / Japanese README](./README.ja.md)

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
| **TTS (FastSpeech2)** | Non-autoregressive TTS acoustic model with variance adaptor, `ProsodyLoss` joint training, variable-length + masked attention (feature: `tts`) |

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

### TTS (FastSpeech2) — feature: `tts`

Non-autoregressive TTS acoustic model based on FastSpeech2 (Ren et al. 2021).
Enable with `--features tts`. Reference implementation: [Wataru-Nakata/FastSpeech2-JSUT](https://github.com/Wataru-Nakata/FastSpeech2-JSUT).

**Architecture**:

```
mora_ids [B, N]  ──► Embedding + PE ──► Encoder (FFT blocks × M)
                                        │
                                        ▼
                              Variance Adaptor
                              (Duration / Pitch / Energy predictors)
                                        │  + pitch/energy embed injection
                                        ▼
                              Length Regulator (mora → frame expansion)
                                        │
                                        ▼
                              Decoder (FFT blocks × M) + PE
                                        │
                                        ▼
                              mel_linear → Postnet residual → mel_after
```

**Sub-modules** (`src/tts/`):

| Module | Purpose |
|--------|---------|
| `primitives/` | Conv1D / LayerNorm / MultiHeadAttention / Linear / PositionalEncoding / WeightNorm — hand-written forward+backward |
| `audio/` | STFT + Mel filterbank + F0 (YIN) + Energy — `AudioFeatureExtractor` |
| `dataset/` | Manifest jsonl + streaming loader + WAV I/O |
| `loss/` | `ProsodyLoss` (F0 L1 + Duration MSE + Energy MSE weighted joint) |
| `batch/` | `TtsBatch` (11-field batch struct with validation) |
| `fastspeech2` | `FastSpeech2` (Config, forward/forward_variable/forward_with_prosody, backward_full/backward_full_variable/backward_full_with_prosody, apply_sgd/apply_adamw, save/load safetensors) |
| `tts_trainer` | `TtsTrainer` (SGD/AdamW switch, `step`/`step_variable`/`step_prosody`/`step_variable_prosody`) |

**Quick Start — FastSpeech2 training** (SGD):

```rust
use alice_train::tts::{
    AdamWConfig, FastSpeech2, FastSpeech2Config, ProsodyLoss, ProsodyTarget,
    TtsTrainConfig, TtsTrainer,
};

let cfg = FastSpeech2Config {
    vocab_size: 100, hidden_dim: 256, num_heads: 2,
    num_encoder_layers: 4, num_decoder_layers: 4,
    fft_kernel_size: 3, fft_expansion: 4,
    predictor_kernel_size: 3, predictor_hidden: 256,
    mel_dim: 80, postnet_kernel_size: 5,
    postnet_layers: 5, postnet_hidden: 512, max_len: 2048,
};
let mut model = FastSpeech2::zeros(cfg).expect("model");
model.init_xavier(42);
// Phase E-next-4: log-domain bias tuning (log(4+1)=1.5 duration, log(150Hz)=5.0 pitch, -20dB energy)
model.init_variance_biases(1.5, 5.0, -20.0);

let mut trainer = TtsTrainer::with_adamw(
    model,
    TtsTrainConfig { learning_rate: 1e-4, log_interval: 100 },
    AdamWConfig::default(),
);

// Simple mel-only training:
let mora_ids: Vec<u32> = vec![1, 2, 3];
let durations: Vec<u32> = vec![2, 2, 2];  // 6 frames total
let mel_target = vec![0.0_f32; 6 * 80];
let result = trainer.step(&mora_ids, &durations, &mel_target, 1, 3).unwrap();
println!("step {}: mel_loss={}", result.step, result.mel_loss);

// Joint mel + prosody training (recommended, use log-duration target):
let prosody_target = ProsodyTarget {
    f0: vec![vec![120.0, 130.0, 125.0]],
    duration_frames: vec![vec![2.0_f32.ln_1p(), 2.0_f32.ln_1p(), 2.0_f32.ln_1p()]],
    energy: vec![vec![-20.0, -18.0, -19.0]],
    mask: None,
};
let prosody_cfg = ProsodyLoss::default_weights();  // w_f0=1.0, w_dur=1.0, w_energy=0.5
let result = trainer
    .step_prosody(&mora_ids, &durations, &mel_target, &prosody_target, prosody_cfg, 1, 3)
    .unwrap();
println!("total={}, mel={}, prosody={}", result.total_loss, result.mel_loss, result.prosody.total);
```

**Checkpoint save/load** (safetensors format for Paperspace 6h session resume):

```rust
model.save_safetensors("checkpoints/fs2_step10000.safetensors").unwrap();
// Later:
let mut model2 = FastSpeech2::zeros(cfg).unwrap();
model2.load_safetensors("checkpoints/fs2_step10000.safetensors").unwrap();
```

**Variable-length training** (batch > 1 with different mora/frame lengths):

```rust
let batch = 2;
let max_mora_len = 3;
let mora_ids = vec![1, 2, 3, 4, 5, 0];  // sample 0 = 3 mora, sample 1 = 2 mora + padding
let durations = vec![2, 2, 2, 2, 3, 0];  // padding duration = 0
let mora_lens = vec![3_usize, 2];
let max_frame_len = 6;
let mel_target = vec![0.0_f32; batch * max_frame_len * 80];
let result = trainer
    .step_variable(&mora_ids, &durations, &mel_target, batch, max_mora_len, &mora_lens, max_frame_len)
    .unwrap();
```

**Roadmap** (Phase T.4a completed features):

- ✅ MVP forward-only + backward Phase 1-3
- ✅ Phase 4 `TtsTrainer` (SGD)
- ✅ Phase 5 JSUT pipeline (Python `prepare_jsut_manifest.py`)
- ✅ Phase A Xavier init
- ✅ Phase B AdamW optimizer
- ✅ Phase C Checkpoint save/load (safetensors)
- ✅ Phase D Variable frame length + attention mask
- ✅ Phase E ProsodyLoss joint training
- ✅ Phase E-next-1 pitch/energy embed hidden injection
- ✅ Phase E-next-2 AdamW variance state expansion
- ✅ Phase E-next-3 variable-length prosody
- ✅ Phase E-next-4 VariancePredictor init tune (log-domain bias)
- 🔜 Paperspace JSUT fine-tune (300k steps, ~7 sessions × 6h)

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
- **TTS module isolation**: `feature = "tts"` gate ensures TTS-specific dependencies (rustfft, hound, safetensors, bytemuck) don't bloat the base LLM training binary. Additional deps only pulled when `--features tts`.
- **Log-domain prosody target**: FastSpeech2 paper convention. `ProsodyTarget::to_log_duration()` transforms `duration_frames → log(dur + 1)`, and `FastSpeech2::init_variance_biases(1.5, ...)` pre-sets predictor linear bias to typical log-mean values for faster convergence.

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
| `bytemuck` | Zero-copy GPU buffer casting (optional, features: `gpu` / `tts`) |
| `safetensors` | Model checkpoint I/O (optional, features: `qat-cli` / `tts`) |
| `rustfft` | STFT for `AudioFeatureExtractor` (optional, feature: `tts`) |
| `hound` | WAV I/O for `TtsDataset` (optional, feature: `tts`) |

## Quality

| Metric | Value |
|--------|-------|
| Tests | 467 (default) / 672 (with `tts` feature) |
| Doc-tests | 7 |
| Clippy (pedantic+nursery) | 0 warnings |
| Doc warnings | 0 |
| fmt | clean |
| Score | 100/100 |

## License

`AGPL-3.0 OR LicenseRef-Commercial` — dual-licensed. Pick either.

| Option | Terms | Use it when |
|--------|-------|-------------|
| **AGPL-3.0** | [LICENSE-AGPL](LICENSE-AGPL) — free, no reporting obligation | Your project is itself AGPL-compatible open source, or you are only using it internally |
| **Commercial License** | [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) — paid, removes the copyleft | Closed-source product, proprietary SaaS, edge / firmware distribution, plugin redistribution, or a platform NDA that forbids source disclosure |

AGPL is a strong copyleft: a product, firmware image, or service that links
`alice-train` and is distributed or served to users must be released under the AGPL
as well. That is intentional for the open ecosystem, and the Commercial
License exists for the cases where it is not something you are able to do.

Commercial licence enquiries: <contact@extoria.co.jp>
