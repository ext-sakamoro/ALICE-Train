# ALICE-Train

ALICE-ML 三値ネットワーク向け Backpropagation & 学習フレームワーク

[English README](./README.md)

## アーキテクチャ

```
ALICE-ML (推論)                ALICE-Train (学習)
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
                              │   Trainer (勾配累積)          │
                              │   train_with_scheduler()    │
                              │   train_tokens()            │
                              ├─────────────────────────────┤
                              │ scheduler.rs                │
                              │   WarmupCosineScheduler     │
                              │   ConstantScheduler         │
                              ├─────────────────────────────┤
                              │ checkpoint.rs               │
                              │   ALICETRN バイナリ形式       │
                              │   save / load               │
                              ├─────────────────────────────┤
                              │ dataloader.rs               │
                              │   MmapDataset (memmap2)     │
                              │   DataLoader + Batch        │
                              ├─────────────────────────────┤
                              │ evaluator.rs                │
                              │   perplexity 評価            │
                              │   BestCheckpointTracker     │
                              ├─────────────────────────────┤
                              │ logger.rs                   │
                              │   TrainLog (CSV / JSON)     │
                              │   compute_grad_norm         │
                              ├─────────────────────────────┤
                              │ mixed_precision.rs          │
                              │   BF16 変換                 │
                              │   LossScaler (動的)          │
                              ├─────────────────────────────┤
                              │ qat.rs                      │
                              │   FakeQuantize              │
                              │   QatTrainer                │
                              │   CalibrationStats          │
                              ├─────────────────────────────┤
                              │ distill.rs                  │
                              │   DistillTrainer            │
                              │   KL-div + hard label 混合   │
                              ├─────────────────────────────┤
                              │ pipeline.rs                 │
                              │   QatPipeline (統合)        │
                              │   FP32→BF16→Ternary ループ  │
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
                              ├─────────────────────────────┤
                              │ tts/            [tts feature]│
                              │   FastSpeech2 acoustic model │
                              │   VarianceAdaptor + Postnet │
                              │   ProsodyLoss joint training │
                              │   TtsTrainer (SGD/AdamW)    │
                              └─────────────────────────────┘
```

## 機能一覧

| 機能 | 説明 |
|------|------|
| Ternary backward | W^T * dy を加算/減算のみで計算 (乗算不使用) |
| RMSNorm backward | pre-normalization の完全 gradient |
| STE weight grad | Straight-Through Estimator による latent FP32 weight 勾配 |
| Activation backward | ReLU / SiLU / GELU + finite diff 数値検証 |
| 学習ループ | `TrainableNetwork` trait + `Trainer` (MSE/CE/MAE loss) |
| 勾配累積 | micro-batch 累積で実効 batch size 拡大 |
| LR scheduling | Warmup + Cosine Decay / Constant scheduler |
| Checkpoint | ALICETRN バイナリ形式 (magic + JSON header + raw weights) |
| memmap データ | `MmapDataset` で大規模 token file を memmap2 経由読み込み |
| Token-based 学習 | `train_tokens()` + DataLoader + scheduler 統合 |
| Perplexity 評価 | `evaluate()` + `BestCheckpointTracker` 自動保存 |
| 学習ログ | CSV / JSON で loss / lr / grad_norm を step 毎記録 |
| Mixed precision | BF16 変換 + 動的 loss scaling (NaN/Inf 検知) |
| QAT | `FakeQuantize` / `QatTrainer` / `CalibrationStats` |
| 知識蒸留 | KL-div + hard label 混合 loss |
| QAT Pipeline | FP32→BF16→Ternary 完全パイプライン (scheduler + checkpoint + eval) |
| GPU backward | wgpu compute shader で `ternary_matvec_backward` (feature: `gpu`) |
| ZeRO-Offload | AdamW state (m/v) を CPU RAM に offload — VRAM 50% 削減 |
| **TTS (FastSpeech2)** | Non-autoregressive TTS acoustic model + variance adaptor + `ProsodyLoss` joint + 可変長 masked attention (feature: `tts`) |

## クイックスタート

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

### Scheduler + Checkpoint 付き学習

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

let config = MixedPrecisionConfig::default(); // 動的 scaling 有効
let mut scaler = LossScaler::new(config);

let weights_bf16 = f32_to_bf16_vec(&weights_f32);
let scaled_loss = scaler.scale_loss(loss);
// ... backward ...
scaler.unscale_gradients(&mut gradients);
let valid = LossScaler::check_gradients(&gradients);
scaler.update(valid);
```

### ZeRO-Offload — VRAM 50% 削減

```rust
use alice_train::{OffloadOptimizer, OffloadConfig, MemoryBudget};

// 7B モデルのメモリ見積り
let budget = MemoryBudget::estimate(7_000_000_000);
// VRAM: 56 GB (weights + gradients のみ)
// CPU RAM: 56 GB (m + v を offload)
// offload なしなら: 112 GB VRAM 必要

let config = OffloadConfig {
    beta1: 0.9,
    beta2: 0.999,
    weight_decay: 0.01,
    max_grad_norm: Some(1.0),
    ..OffloadConfig::default()
};
let mut optimizer = OffloadOptimizer::new(param_count, config);

// 学習ループ: GPU forward/backward → CPU update
optimizer.step(&mut weights, &mut gradients, lr);
```

### GPU backward (feature: `gpu`)

```rust
use alice_train::{GpuContext, GpuBackwardEngine};

let ctx = GpuContext::new_blocking().expect("GPU required");
let engine = GpuBackwardEngine::new(&ctx);

// GPU 加速: dx = W^T * dy
engine.ternary_matvec_backward(&grad_output, &kernel, &mut grad_input);
// CPU 版と bit-exact 一致
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
    mixed_precision: MixedPrecisionConfig::default(), // BF16 有効
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

// 最終 ternary weights を書き出し
let mut ternary = vec![0.0f32; latent_weights.len()];
pipeline.finalize_weights(&latent_weights, &mut ternary);
```

### TTS (FastSpeech2) — feature: `tts`

FastSpeech2 (Ren et al. 2021) ベースの non-autoregressive TTS acoustic model。
`--features tts` で有効化。参照実装: [Wataru-Nakata/FastSpeech2-JSUT](https://github.com/Wataru-Nakata/FastSpeech2-JSUT)

**アーキテクチャ**:

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

**サブモジュール** (`src/tts/`):

| モジュール | 役割 |
|-----------|------|
| `primitives/` | Conv1D / LayerNorm / MultiHeadAttention / Linear / PositionalEncoding / WeightNorm — 手書き forward + backward |
| `audio/` | STFT + Mel filterbank + F0 (YIN) + Energy — `AudioFeatureExtractor` |
| `dataset/` | Manifest jsonl + streaming loader + WAV I/O |
| `loss/` | `ProsodyLoss` (F0 L1 + Duration MSE + Energy MSE の重み付き joint) |
| `batch/` | `TtsBatch` (11 field batch struct + 検証) |
| `fastspeech2` | `FastSpeech2` (Config, forward/forward_variable/forward_with_prosody, backward_full/backward_full_variable/backward_full_with_prosody, apply_sgd/apply_adamw, save/load safetensors) |
| `tts_trainer` | `TtsTrainer` (SGD/AdamW 切替、`step`/`step_variable`/`step_prosody`/`step_variable_prosody`) |

**クイックスタート — FastSpeech2 学習**:

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
// Phase E-next-4: log-domain bias 事前設定 (log(4+1)=1.5 duration, log(150Hz)=5.0 pitch, -20dB energy)
model.init_variance_biases(1.5, 5.0, -20.0);

let mut trainer = TtsTrainer::with_adamw(
    model,
    TtsTrainConfig { learning_rate: 1e-4, log_interval: 100 },
    AdamWConfig::default(),
);

// シンプルな mel のみ学習:
let mora_ids: Vec<u32> = vec![1, 2, 3];
let durations: Vec<u32> = vec![2, 2, 2];  // 合計 6 frames
let mel_target = vec![0.0_f32; 6 * 80];
let result = trainer.step(&mora_ids, &durations, &mel_target, 1, 3).unwrap();
println!("step {}: mel_loss={}", result.step, result.mel_loss);

// mel + prosody joint 学習 (推奨、log-duration target 使用):
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

**Checkpoint save/load** (Paperspace 6h セッション対応の safetensors 形式):

```rust
model.save_safetensors("checkpoints/fs2_step10000.safetensors").unwrap();
// 後で:
let mut model2 = FastSpeech2::zeros(cfg).unwrap();
model2.load_safetensors("checkpoints/fs2_step10000.safetensors").unwrap();
```

**可変長学習** (batch > 1 で mora/frame 長が異なる場合):

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

**ロードマップ** (Phase T.4a 完了機能):

- ✅ MVP forward-only + backward Phase 1-3
- ✅ Phase 4 `TtsTrainer` (SGD)
- ✅ Phase 5 JSUT pipeline (Python `prepare_jsut_manifest.py`)
- ✅ Phase A Xavier init
- ✅ Phase B AdamW optimizer
- ✅ Phase C Checkpoint save/load (safetensors)
- ✅ Phase D 可変 frame 長 + attention mask
- ✅ Phase E ProsodyLoss joint 学習
- ✅ Phase E-next-1 pitch/energy embed hidden injection
- ✅ Phase E-next-2 AdamW variance state 拡張
- ✅ Phase E-next-3 可変長 prosody
- ✅ Phase E-next-4 VariancePredictor init tune (log-domain bias)
- 🔜 Paperspace JSUT fine-tune (300k steps、~7 sessions × 6h)

## 設計方針

- **クレート分離**: ALICE-ML は推論特化で `no_std` / ゼロアロケーション、学習は `std` + heap alloc が必要なため本クレートに分離
- **DPS pattern**: 全 backward 関数は呼び出し側から `grad_input: &mut [f32]` を受け取り書き込む
- **STE for ternary weights**: 離散 {-1, 0, +1} は直接微分不能。latent FP32 weights を保持し forward 時に量子化
- **Loss/Optimizer 再利用**: `alice_ml::training` の MSE / CrossEntropy / MAE / SGD / Adam を活用
- **Binary checkpoint 形式**: ALICETRN magic + JSON metadata header + raw f32 weights + optimizer state。コンパクトで高速 load
- **Byte-level mmap access**: `MmapDataset` は `u32::from_le_bytes()` で token を読み、pointer cast による alignment 問題を回避
- **動的 loss scaling**: `LossScaler` が連続成功 step をカウント、安定時は scale を倍増、NaN/Inf 検出時は半減。下限 1.0
- **Callback-based token training**: `train_tokens()` は `token_embed_fn` / `target_embed_fn` closure を受け取り、token 表現と学習ループを分離
- **GPU backward via wgpu**: WGSL compute shader が CPU logic をミラー。各 thread が 1 つの `grad_input[col]` を担当し全 row を走査。feature-gated (`gpu`)
- **ZeRO-Offload**: AdamW m/v を CPU RAM 格納。VRAM を 4N → 2N params に削減。gradient clipping、bias correction、memory budget 見積り込み
- **TTS モジュール隔離**: `feature = "tts"` gate により TTS 特有 dep (rustfft, hound, safetensors, bytemuck) が base LLM 学習 binary を膨らませない。`--features tts` 指定時のみ追加 dep 引き込み
- **Log-domain prosody target**: FastSpeech2 論文慣習。`ProsodyTarget::to_log_duration()` で `duration_frames → log(dur + 1)` 変換し、`FastSpeech2::init_variance_biases(1.5, ...)` で predictor linear bias を典型 log-mean 値に事前設定して序盤収束を改善

## 依存クレート

| Crate | 用途 |
|-------|------|
| `alice-ml` | 推論エンジン (forward, loss, optimizer) |
| `memmap2` | 大規模データセット用の memory-mapped file I/O |
| `rand` | DataLoader の shuffle |
| `serde` | Checkpoint metadata シリアライズ |
| `serde_json` | Checkpoint header と training log の JSON 形式 |
| `wgpu` | GPU compute (optional, feature: `gpu`) |
| `pollster` | wgpu 用 async→sync bridge (optional, feature: `gpu`) |
| `bytemuck` | Zero-copy GPU buffer cast (optional, features: `gpu` / `tts`) |
| `safetensors` | Model checkpoint I/O (optional, features: `qat-cli` / `tts`) |
| `rustfft` | `AudioFeatureExtractor` の STFT (optional, feature: `tts`) |
| `hound` | `TtsDataset` の WAV I/O (optional, feature: `tts`) |

## 品質基準

| 指標 | 値 |
|------|-----|
| テスト数 | 467 (default) / 672 (`tts` feature 込み) |
| Doc-tests | 7 |
| Clippy (pedantic+nursery) | 0 warnings |
| Doc warnings | 0 |
| fmt | clean |
| スコア | 100/100 |

## ライセンス

AGPL-3.0
