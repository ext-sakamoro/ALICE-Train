# Contributing to ALICE-Train

## Prerequisites

- Rust 1.75+
- `alice-ml` crate (sibling directory `../ALICE-ML`)

## Build & Test

```bash
cargo test
cargo clippy --all-targets -- -W clippy::pedantic -W clippy::nursery
cargo fmt -- --check
cargo doc --no-deps
```

## Architecture

| Module | Responsibility |
|--------|---------------|
| `activation.rs` | Activation function backward (ReLU, SiLU, GELU) |
| `backward.rs` | Layer backward (ternary matvec transpose, BitLinear, STE) |
| `trainer.rs` | Training loop, `TrainableNetwork` trait |

## Code Style

- Japanese comments (consistent with ALICE project)
- DPS (Destination Passing Style): caller provides output buffers
- All `pub fn` with `# Panics` doc section
- `#[must_use]` on constructors and pure functions
- Clippy pedantic + nursery, 0 warnings
