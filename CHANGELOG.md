# Changelog

## [0.1.0] - 2026-03-06

### Added
- `activation` module: `relu_backward`, `silu_backward`, `gelu_backward`
- `backward` module: `ternary_matvec_backward`, `bitlinear_backward`, `ste_weight_grad`
- `trainer` module: `TrainableNetwork` trait, `Trainer`, `TrainConfig`, `EpochResult`
- Full numerical gradient verification for all backward functions
- 100+ tests covering happy path, boundary, error, and convergence
