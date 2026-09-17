//! Analytic oracles — closed-form / independent-reference checks for the
//! numeric laws in ALICE-Train (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Expected values come from closed forms or f64 references written in this
//! file, never from the crate function under test.  Default constructors
//! (`QatConfig::default()` / `::int8()`, `MixedPrecisionConfig::default()`)
//! are the paths a consumer takes first.
//!
//! Oracle sources:
//! - activations: silu' = σ(x)(1 + x(1 − σ(x))), gelu (tanh form) by f64
//!   central difference, relu' = [x > 0]
//! - ternary linear layer y = W x: dx = Wᵀ g, dW = g xᵀ (STE) — exact in f64
//!   for small integers; pre-norm variant by f64 central difference of
//!   gᵀ·W·(x / √(mean x² + ε))
//! - symmetric quantisers: ternary γ = mean|w| (BitNet b1.58), int8 / int4
//!   absmax step Δ = max|w| / half_levels with |w − q(w)| ≤ Δ/2 inside range
//! - bf16: nearest representable (ties to even), exact when ≤ 8 significant bits
//! - dynamic loss scaling: ×growth every `interval` good steps, ×backoff on a
//!   bad step, floor 1
//! - BLAS: f64 naive matmul; rmsnorm x·w/√(mean x² + ε)
//! - warmup-cosine LR: linear to max at warmup, cosine to min at total

use alice_ml::ops::TernaryWeightKernel;
use alice_train::blas::{blas_matmul_bt, blas_matmul_nn, blas_matmul_tn, blas_rmsnorm};
use alice_train::mixed_precision::{Bf16, LossScaler, MixedPrecisionConfig};
use alice_train::qat::{FakeQuantize, QatConfig};
use alice_train::scheduler::{LrScheduler, WarmupCosineScheduler};
use alice_train::{
    bitlinear_backward, gelu_backward, relu_backward, silu_backward, ste_weight_grad,
    ternary_matvec_backward,
};

// ───────────────────────── helpers ────────────────────────────────────────

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0
}

fn sigma(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// GELU, tanh approximation (Hendrycks & Gimpel 2016, the form the crate implements)
fn gelu_tanh(x: f64) -> f64 {
    let k = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
}

fn central_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

// ───────────────────────── activations ────────────────────────────────────

#[test]
fn activation_backward_matches_the_closed_form_derivatives() {
    let xs: Vec<f32> = (-60..=60).map(|i| i as f32 * 0.1).collect();
    let ones = vec![1.0f32; xs.len()];
    let mut g = vec![0.0f32; xs.len()];
    silu_backward(&xs, &ones, &mut g);
    for (&x, &d) in xs.iter().zip(&g) {
        let x = x as f64;
        let s = sigma(x);
        let expected = s * (1.0 + x * (1.0 - s)); // oracle: d/dx x·σ(x)
        assert!(
            (d as f64 - expected).abs() < 1e-5,
            "silu' at {x}: {d} vs {expected}"
        );
    }
    gelu_backward(&xs, &ones, &mut g);
    for (&x, &d) in xs.iter().zip(&g) {
        let expected = central_diff(gelu_tanh, x as f64, 1e-4); // oracle: numeric derivative
        assert!(
            (d as f64 - expected).abs() < 1e-4,
            "gelu' at {x}: {d} vs {expected}"
        );
    }
    relu_backward(&xs, &ones, &mut g);
    for (&x, &d) in xs.iter().zip(&g) {
        assert_eq!(d, if x > 0.0 { 1.0 } else { 0.0 }, "relu' at {x}");
    }
    // chain rule: grad_input = grad_output · f'(x)
    let go: Vec<f32> = xs.iter().map(|x| x.sin()).collect();
    silu_backward(&xs, &go, &mut g);
    for ((&x, &d), &gi) in xs.iter().zip(&go).zip(&g) {
        let x64 = x as f64;
        let s = sigma(x64);
        assert!((gi as f64 - d as f64 * s * (1.0 + x64 * (1.0 - s))).abs() < 1e-5);
    }
}

// ───────────────────────── ternary linear backward ────────────────────────

fn ternary_matrix(out: usize, inp: usize, seed: &mut u64) -> Vec<i8> {
    (0..out * inp)
        .map(|_| {
            let u = lcg(seed);
            if u < -0.33 {
                -1
            } else if u > 0.33 {
                1
            } else {
                0
            }
        })
        .collect()
}

#[test]
fn ternary_matvec_backward_is_w_transpose_times_g() {
    let mut seed = 3u64;
    for (out, inp) in [(2usize, 2usize), (5, 7), (16, 33), (3, 70)] {
        let w = ternary_matrix(out, inp, &mut seed);
        let g: Vec<f32> = (0..out)
            .map(|_| (lcg(&mut seed) * 4.0).round() as f32)
            .collect();
        let kernel = TernaryWeightKernel::from_ternary(&w, out, inp);
        let mut dx = vec![7.0f32; inp]; // must be overwritten, not accumulated
        ternary_matvec_backward(&g, &kernel, &mut dx);
        for i in 0..inp {
            // oracle: dx_i = Σ_j W[j][i] · g_j — integers, exact
            let expected: f64 = (0..out).map(|j| w[j * inp + i] as f64 * g[j] as f64).sum();
            assert_eq!(dx[i] as f64, expected, "{out}x{inp} dx[{i}]");
        }
        // scaled kernel: y = γ·W x ⇒ dx = γ·Wᵀ g
        let gamma = 0.37f32;
        let kernel = TernaryWeightKernel::from_ternary_scaled(&w, out, inp, gamma);
        ternary_matvec_backward(&g, &kernel, &mut dx);
        for i in 0..inp {
            let expected: f64 = (0..out)
                .map(|j| w[j * inp + i] as f64 * g[j] as f64)
                .sum::<f64>()
                * gamma as f64;
            assert!(
                (dx[i] as f64 - expected).abs() < 1e-5 * expected.abs().max(1.0),
                "scaled dx[{i}]: {} vs {expected}",
                dx[i]
            );
        }
    }
}

#[test]
fn ste_weight_grad_is_the_outer_product_and_accumulates() {
    let x = [1.0f32, -2.0, 0.5];
    let g = [3.0f32, -1.0];
    let mut dw = vec![0.0f32; 6];
    ste_weight_grad(&x, &g, &mut dw);
    // oracle: dW[j][i] = g_j · x_i
    assert_eq!(dw, [3.0, -6.0, 1.5, -1.0, 2.0, -0.5]);
    ste_weight_grad(&x, &g, &mut dw);
    assert_eq!(
        dw,
        [6.0, -12.0, 3.0, -2.0, 4.0, -1.0],
        "second call accumulates"
    );
}

#[test]
fn bitlinear_backward_matches_finite_differences_with_and_without_prenorm() {
    let mut seed = 11u64;
    let (out, inp) = (6usize, 9usize);
    let w = ternary_matrix(out, inp, &mut seed);
    let kernel = TernaryWeightKernel::from_ternary(&w, out, inp);
    let x: Vec<f32> = (0..inp).map(|_| lcg(&mut seed) as f32 * 1.5).collect();
    let g: Vec<f32> = (0..out).map(|_| lcg(&mut seed) as f32).collect();
    let eps = 1e-5f32;

    // f64 forward of the pre-norm layer, L = gᵀ · W · (x / rms(x))
    let loss = |xv: &[f64]| -> f64 {
        let ms = xv.iter().map(|v| v * v).sum::<f64>() / xv.len() as f64;
        let inv = 1.0 / (ms + eps as f64).sqrt();
        (0..out)
            .map(|j| {
                g[j] as f64
                    * (0..inp)
                        .map(|i| w[j * inp + i] as f64 * xv[i] * inv)
                        .sum::<f64>()
            })
            .sum()
    };
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();

    let mut dx = vec![0.0f32; inp];
    let mut db = vec![0.0f32; out];
    bitlinear_backward(&x, &g, &kernel, true, eps, &mut dx, Some(&mut db));
    assert_eq!(db, g, "bias gradient is the output gradient");
    for i in 0..inp {
        let mut xp = x64.clone();
        let h = 1e-4;
        xp[i] += h;
        let mut xm = x64.clone();
        xm[i] -= h;
        let expected = (loss(&xp) - loss(&xm)) / (2.0 * h);
        assert!(
            (dx[i] as f64 - expected).abs() < 1e-3 * expected.abs().max(1.0),
            "pre-norm dx[{i}]: {} vs finite difference {expected}",
            dx[i]
        );
    }
    // without pre-norm it is plain Wᵀ g
    bitlinear_backward(&x, &g, &kernel, false, eps, &mut dx, None);
    for i in 0..inp {
        let expected: f64 = (0..out).map(|j| w[j * inp + i] as f64 * g[j] as f64).sum();
        assert!((dx[i] as f64 - expected).abs() < 1e-5, "dx[{i}]");
    }
}

// ───────────────────────── fake quantisation ──────────────────────────────

fn weights(n: usize, seed: &mut u64) -> Vec<f32> {
    // heavy-tailed: mostly small with a few large entries, like real layers
    (0..n)
        .map(|_| {
            let u = lcg(seed);
            (u * u * u * 3.0) as f32
        })
        .collect()
}

#[test]
fn ternary_fake_quantise_is_bitnet_b158() {
    let mut seed = 5u64;
    let w = weights(4096, &mut seed);
    let mut fq = FakeQuantize::new(QatConfig::default()); // ternary
    fq.calibrate_scale(&w);
    // oracle: γ = mean|w|
    let gamma = w.iter().map(|v| v.abs() as f64).sum::<f64>() / w.len() as f64;
    assert!(
        (fq.scale() as f64 - gamma).abs() < 1e-6 * gamma,
        "γ {} vs {gamma}",
        fq.scale()
    );
    let mut q = vec![0.0f32; w.len()];
    fq.fake_quantize_forward(&w, &mut q);
    let mut used = [false; 3];
    for (&wv, &qv) in w.iter().zip(&q) {
        // oracle: q = clamp(round(w/γ), −1, 1) · γ
        let expected = (wv as f64 / gamma).round().clamp(-1.0, 1.0) * gamma;
        assert!(
            (qv as f64 - expected).abs() < 1e-6,
            "w={wv}: {qv} vs {expected}"
        );
        used[if expected < 0.0 {
            0
        } else if expected > 0.0 {
            2
        } else {
            1
        }] = true;
    }
    assert!(used.iter().all(|&u| u), "all three levels appear");
    // STE: gradient passes straight through (grad_clip 0)
    let g: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
    let mut gi = vec![0.0f32; 8];
    fq.ste_backward(&g, &mut gi);
    assert_eq!(gi, g);
}

#[test]
fn int8_and_int4_fake_quantise_reconstruct_within_half_a_step_over_the_full_range() {
    let mut seed = 9u64;
    let w = weights(4096, &mut seed);
    let max_abs = w.iter().map(|v| v.abs() as f64).fold(0.0, f64::max);
    for (cfg, half_levels) in [(QatConfig::int8(), 127.0f64), (QatConfig::int4(), 7.0f64)] {
        let mut fq = FakeQuantize::new(cfg);
        fq.calibrate_scale(&w);
        let mut q = vec![0.0f32; w.len()];
        fq.fake_quantize_forward(&w, &mut q);
        // oracle: a symmetric uniform quantiser covering the tensor has step
        // Δ = max|w| / half_levels and reconstructs every weight within Δ/2;
        // in particular the largest weight must not be clipped
        let step = max_abs / half_levels;
        let mut worst = 0.0f64;
        for (&wv, &qv) in w.iter().zip(&q) {
            worst = worst.max((wv as f64 - qv as f64).abs());
        }
        assert!(
            worst <= step / 2.0 + 1e-6,
            "{half_levels}-level quantiser: worst |w − q| = {worst} > Δ/2 = {} (max|w| {max_abs})",
            step / 2.0
        );
        let levels: std::collections::BTreeSet<i64> = q
            .iter()
            .map(|v| (*v as f64 / step).round() as i64)
            .collect();
        assert!(levels.len() > 2, "more than three levels in use");
    }
    // group-wise int4: per-group absmax, same bound per group
    let mut fq = FakeQuantize::new(QatConfig::int4_grouped(32));
    fq.calibrate_scale(&w);
    let mut q = vec![0.0f32; w.len()];
    fq.fake_quantize_forward(&w, &mut q);
    for (gi, (wg, qg)) in w.chunks(32).zip(q.chunks(32)).enumerate() {
        let gmax = wg.iter().map(|v| v.abs() as f64).fold(0.0, f64::max);
        let step = gmax / 7.0;
        for (&wv, &qv) in wg.iter().zip(qg) {
            assert!(
                (wv as f64 - qv as f64).abs() <= step / 2.0 + 1e-6,
                "group {gi}: w={wv} q={qv}"
            );
        }
    }
}

// ───────────────────────── bf16 / loss scaling ────────────────────────────

#[test]
fn bf16_rounds_to_nearest_even_and_is_exact_for_short_mantissas() {
    let mut seed = 21u64;
    for _ in 0..20_000 {
        let x = (lcg(&mut seed) * 1e3) as f32;
        let r = Bf16::from_f32(x).to_f32();
        // oracle: the two bf16 neighbours are the f32 with the low 16 bits
        // cleared, and the next one up; pick the nearer, ties to even
        let lo = f32::from_bits(x.to_bits() & 0xFFFF_0000);
        let hi = f32::from_bits((x.to_bits() & 0xFFFF_0000).wrapping_add(0x1_0000));
        let (dlo, dhi) = ((x as f64 - lo as f64).abs(), (hi as f64 - x as f64).abs());
        let expected = if dlo < dhi {
            lo
        } else if dhi < dlo {
            hi
        } else if (lo.to_bits() >> 16) & 1 == 0 {
            lo
        } else {
            hi
        };
        assert_eq!(r.to_bits(), expected.to_bits(), "bf16({x})");
        assert!(((r - x) / x).abs() <= 2f32.powi(-8), "relative error bound");
    }
    for x in [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.5,
        3.0,
        1024.0,
        0.001953125,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert_eq!(
            Bf16::from_f32(x).to_f32().to_bits(),
            x.to_bits(),
            "{x} is representable"
        );
    }
    assert!(Bf16::from_f32(f32::NAN).to_f32().is_nan());
    assert!(Bf16::from_f32(0.0).is_zero() && Bf16::from_f32(-0.0).is_zero());
    assert!(!Bf16::from_f32(1e-30).is_zero());
}

#[test]
fn dynamic_loss_scaling_follows_its_closed_form_schedule() {
    let cfg = MixedPrecisionConfig::default();
    let mut s = LossScaler::new(cfg.clone());
    assert_eq!(s.scale(), cfg.loss_scale);
    // scale_loss / unscale are inverses (exact for power-of-two scales)
    let loss = 0.123_456f32;
    assert_eq!(s.scale_loss(loss), loss * cfg.loss_scale);
    let mut g = vec![1.5f32 * cfg.loss_scale, -0.25 * cfg.loss_scale];
    s.unscale_gradients(&mut g);
    assert_eq!(g, [1.5, -0.25]);
    // oracle: after k·interval good steps the scale is loss_scale·growth^k
    for k in 1..=3 {
        for _ in 0..cfg.scale_growth_interval {
            assert!(s.update(true));
        }
        let expected = cfg.loss_scale * cfg.scale_growth_factor.powi(k);
        assert_eq!(s.scale(), expected, "after {k} growth intervals");
    }
    // a bad step multiplies by backoff and resets the counter
    assert!(!s.update(false));
    let expected = cfg.loss_scale * cfg.scale_growth_factor.powi(3) * cfg.scale_backoff_factor;
    assert_eq!(s.scale(), expected);
    for _ in 0..cfg.scale_growth_interval - 1 {
        s.update(true);
    }
    assert_eq!(s.scale(), expected, "counter was reset by the bad step");
    // floor at 1
    for _ in 0..64 {
        s.update(false);
    }
    assert_eq!(s.scale(), 1.0);
    assert!(LossScaler::check_gradients(&[1.0, -2.0]));
    assert!(!LossScaler::check_gradients(&[1.0, f32::NAN]));
    assert!(!LossScaler::check_gradients(&[f32::INFINITY]));
}

// ───────────────────────── BLAS ───────────────────────────────────────────

#[test]
fn matmul_variants_match_the_f64_reference_and_rmsnorm_its_closed_form() {
    let mut seed = 77u64;
    for (m, n, k) in [
        (1usize, 1usize, 1usize),
        (3, 5, 7),
        (8, 8, 8),
        (17, 9, 33),
        (64, 32, 48),
    ] {
        let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) as f32).collect();
        let b_nk: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed) as f32).collect(); // B as [n × k]
        let b_kn: Vec<f32> = (0..k * n).map(|_| lcg(&mut seed) as f32).collect(); // B as [k × n]
        let a_km: Vec<f32> = (0..k * m).map(|_| lcg(&mut seed) as f32).collect(); // A as [k × m]
        let tol = 1e-5 * k as f64;
        // C[m×n] = A[m×k] · B[n×k]ᵀ
        let mut c = vec![0.0f32; m * n];
        blas_matmul_bt(&a, &b_nk, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let e: f64 = (0..k)
                    .map(|l| a[i * k + l] as f64 * b_nk[j * k + l] as f64)
                    .sum();
                assert!(
                    (c[i * n + j] as f64 - e).abs() < tol,
                    "bt {m}x{n}x{k} [{i},{j}]"
                );
            }
        }
        // C[m×n] = A[k×m]ᵀ · B[k×n]
        let mut c = vec![0.0f32; m * n];
        blas_matmul_tn(&a_km, &b_kn, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let e: f64 = (0..k)
                    .map(|l| a_km[l * m + i] as f64 * b_kn[l * n + j] as f64)
                    .sum();
                assert!(
                    (c[i * n + j] as f64 - e).abs() < tol,
                    "tn {m}x{n}x{k} [{i},{j}]"
                );
            }
        }
        // C[m×n] = A[m×k] · B[k×n]
        let mut c = vec![0.0f32; m * n];
        blas_matmul_nn(&a, &b_kn, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let e: f64 = (0..k)
                    .map(|l| a[i * k + l] as f64 * b_kn[l * n + j] as f64)
                    .sum();
                assert!(
                    (c[i * n + j] as f64 - e).abs() < tol,
                    "nn {m}x{n}x{k} [{i},{j}]"
                );
            }
        }
    }
    // rmsnorm: x_i · w_i / √(mean(x²) + ε), row-wise
    let dim = 16;
    let mut x: Vec<f32> = (0..3 * dim).map(|_| lcg(&mut seed) as f32 * 5.0).collect();
    let w: Vec<f32> = (0..dim).map(|i| 0.5 + i as f32 * 0.1).collect();
    let orig = x.clone();
    blas_rmsnorm(&mut x, &w, dim, 1e-6);
    for r in 0..3 {
        let row = &orig[r * dim..(r + 1) * dim];
        let ms = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / dim as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for i in 0..dim {
            let e = row[i] as f64 * inv * w[i] as f64;
            assert!(
                (x[r * dim + i] as f64 - e).abs() < 1e-5,
                "rmsnorm row {r} [{i}]"
            );
        }
    }
}

// ───────────────────────── LR schedule ────────────────────────────────────

#[test]
fn warmup_cosine_schedule_matches_its_closed_form() {
    let s = WarmupCosineScheduler::new(1e-3, 1e-5, 100, 1100);
    for step in 0..1200usize {
        let expected = if step >= 1100 {
            1e-5
        } else if step < 100 {
            1e-3 * step as f64 / 100.0
        } else {
            let p = (step - 100) as f64 / 1000.0;
            1e-5 + (1e-3 - 1e-5) * 0.5 * (1.0 + (std::f64::consts::PI * p).cos())
        };
        let got = s.get_lr(step) as f64;
        assert!(
            (got - expected).abs() < 1e-9,
            "step {step}: {got} vs {expected}"
        );
    }
    assert!(
        (s.get_lr(100) as f64 - 1e-3).abs() < 1e-9,
        "peak at the end of warmup"
    );
    assert!(
        (s.get_lr(600) as f64 - (1e-5 + (1e-3 - 1e-5) * 0.5)).abs() < 1e-9,
        "half way = mid point"
    );
}
