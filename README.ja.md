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
                              │   AudioFeatureExtractor     │
                              │   TtsDataset / TtsBatch     │
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
| **TTS の特徴量と data** | 音声特徴量 (STFT + mel filterbank + F0 + energy、F0 抽出器は差し替え可能)、manifest からの dataset loader と batch 型 (feature: `tts`) |

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

### TTS の特徴量と data — feature: `tts`

`--features tts` で有効化 音響 model とその学習はこの crate から外した (以前の FastSpeech2 実装は削除) 残るのは data 側

| Module | 役割 |
|--------|------|
| `tts::audio` | `AudioFeatureExtractor`: STFT + mel filterbank + F0 + energy、F0 抽出器は `F0Extractor` で差し替え可能 (`Yin` を内蔵) |
| `tts::dataset` | `TtsDataset`: manifest jsonl + streaming loader + WAV I/O |
| `tts::batch` | `TtsBatch`: 検証つき batch 型 |

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
- **TTS モジュール隔離**: `feature = "tts"` gate により TTS 特有 dep (rustfft, hound) が base LLM 学習 binary を膨らませない。`--features tts` 指定時のみ追加 dep 引き込み

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
| `bytemuck` | Zero-copy GPU buffer cast (optional, feature: `gpu`) |
| `safetensors` | Model checkpoint I/O (optional, feature: `qat-cli`) |
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

`LicenseRef-Proprietary` 条文は [LICENSE-Proprietary](LICENSE-Proprietary)

Enquiries: https://extoria.co.jp/
