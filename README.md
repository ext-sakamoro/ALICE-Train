# ALICE-Train

Backpropagation & Training Framework for ALICE-ML ternary networks.

## Architecture

```
ALICE-ML (Inference)          ALICE-Train (Learning)
┌──────────────────┐          ┌─────────────────────────┐
│ BitLinear        │          │ backward.rs             │
│   forward()      │◀────────│   ternary_matvec_backward│
│ TernaryWeight    │          │   bitlinear_backward    │
│ Loss / Optimizer │          │   ste_weight_grad       │
└──────────────────┘          ├─────────────────────────┤
                              │ activation.rs           │
                              │   relu_backward         │
                              │   silu_backward         │
                              │   gelu_backward         │
                              ├─────────────────────────┤
                              │ trainer.rs              │
                              │   TrainableNetwork      │
                              │   Trainer               │
                              └─────────────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| Ternary backward | W^T * dy using add/sub only (no multiplication) |
| RMSNorm backward | Full gradient through pre-normalization |
| STE weight grad | Straight-Through Estimator for latent FP32 weights |
| Activation backward | ReLU, SiLU, GELU with numerical gradient verification |
| Training loop | `TrainableNetwork` trait + `Trainer` with MSE/CE/MAE loss |

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

## Design Decisions

- **Separate crate**: ALICE-ML stays `no_std` / zero-allocation for inference; training requires `std` and heap allocation for gradient buffers.
- **DPS pattern**: All backward functions write into caller-provided buffers (`grad_input: &mut [f32]`).
- **STE for ternary weights**: Discrete {-1, 0, +1} weights can't be differentiated directly. Latent FP32 weights are maintained and quantized during forward.
- **Loss/Optimizer reuse**: `alice_ml::training` provides MSE, CrossEntropy, MAE, SGD, Adam.

## Quality

| Metric | Value |
|--------|-------|
| Tests | 100+ |
| Clippy (pedantic+nursery) | 0 warnings |
| Doc warnings | 0 |
| fmt | clean |

## License

AGPL-3.0
