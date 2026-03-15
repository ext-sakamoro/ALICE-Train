//! 活性化関数の逆伝播。
//!
//! 各関数は DPS パターン: `(input, grad_output, grad_input)` で勾配を書き込む。

use std::f32::consts::FRAC_2_SQRT_PI;
use rayon::prelude::*;

/// `ReLU` の逆伝播。
///
/// `grad_input[i] = grad_output[i] if input[i] > 0, else 0`
///
/// # Panics
///
/// 3つのスライスの長さが異なる場合。
pub fn relu_backward(input: &[f32], grad_output: &[f32], grad_input: &mut [f32]) {
    assert_eq!(input.len(), grad_output.len());
    assert_eq!(input.len(), grad_input.len());

    input.par_iter()
        .zip(grad_output.par_iter())
        .zip(grad_input.par_iter_mut())
        .for_each(|((&x, &go), gi)| {
            *gi = if x > 0.0 { go } else { 0.0 };
        });
}

/// `SiLU` (Swish) の逆伝播。
///
/// `silu(x) = x * sigmoid(x)`
/// `silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))`
///
/// # Panics
///
/// 3つのスライスの長さが異なる場合。
pub fn silu_backward(input: &[f32], grad_output: &[f32], grad_input: &mut [f32]) {
    assert_eq!(input.len(), grad_output.len());
    assert_eq!(input.len(), grad_input.len());

    input.par_iter()
        .zip(grad_output.par_iter())
        .zip(grad_input.par_iter_mut())
        .for_each(|((&x, &go), gi)| {
            let sig = 1.0 / (1.0 + (-x).exp());
            let dsilu = sig * x.mul_add(1.0 - sig, 1.0);
            *gi = go * dsilu;
        });
}

/// GELU の逆伝播（tanh 近似版）。
///
/// `gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
///
/// # Panics
///
/// 3つのスライスの長さが異なる場合。
pub fn gelu_backward(input: &[f32], grad_output: &[f32], grad_input: &mut [f32]) {
    assert_eq!(input.len(), grad_output.len());
    assert_eq!(input.len(), grad_input.len());

    let sqrt_2_over_pi = FRAC_2_SQRT_PI * std::f32::consts::FRAC_1_SQRT_2;
    let c = 0.044_715_f32;

    input.par_iter()
        .zip(grad_output.par_iter())
        .zip(grad_input.par_iter_mut())
        .for_each(|((&x, &go), gi)| {
            let arg = sqrt_2_over_pi * (c * x * x).mul_add(x, x);
            let tanh_val = arg.tanh();
            let sech2 = tanh_val.mul_add(-tanh_val, 1.0);
            let d_inner = sqrt_2_over_pi * 3.0_f32.mul_add(c * x * x, 1.0);
            let dgelu = 0.5_f32.mul_add(1.0 + tanh_val, 0.5 * x * sech2 * d_inner);
            *gi = go * dgelu;
        });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    fn numerical_grad(f: impl Fn(f32) -> f32, x: f32, eps: f32) -> f32 {
        (f(x + eps) - f(x - eps)) / (2.0 * eps)
    }

    // ---- ReLU ----

    #[test]
    fn relu_backward_positive() {
        let input = [1.0, 2.0, 3.0];
        let grad_out = [0.5, 1.0, 2.0];
        let mut grad_in = [0.0_f32; 3];
        relu_backward(&input, &grad_out, &mut grad_in);
        assert_eq!(grad_in, [0.5, 1.0, 2.0]);
    }

    #[test]
    fn relu_backward_negative() {
        let input = [-1.0, -0.5, 0.0];
        let grad_out = [1.0, 1.0, 1.0];
        let mut grad_in = [0.0_f32; 3];
        relu_backward(&input, &grad_out, &mut grad_in);
        assert_eq!(grad_in, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn relu_backward_mixed() {
        let input = [1.0, -1.0, 2.0, -2.0, 0.5];
        let grad_out = [10.0, 20.0, 30.0, 40.0, 50.0];
        let mut grad_in = [0.0_f32; 5];
        relu_backward(&input, &grad_out, &mut grad_in);
        assert_eq!(grad_in, [10.0, 0.0, 30.0, 0.0, 50.0]);
    }

    #[test]
    fn relu_backward_numerical() {
        let relu = |x: f32| x.max(0.0);
        for &x in &[0.5_f32, 1.0, 2.0, 5.0] {
            let analytic = if x > 0.0 { 1.0 } else { 0.0 };
            let numeric = numerical_grad(relu, x, 1e-4);
            assert!(
                (analytic - numeric).abs() < 5e-3,
                "relu grad mismatch at x={x}"
            );
        }
    }

    #[test]
    fn relu_backward_large_values() {
        let input = [100.0, -100.0, 1e6, -1e6];
        let grad_out = [1.0; 4];
        let mut grad_in = [0.0_f32; 4];
        relu_backward(&input, &grad_out, &mut grad_in);
        assert_eq!(grad_in, [1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn relu_backward_zero_grad() {
        let input = [1.0, 2.0, 3.0];
        let grad_out = [0.0, 0.0, 0.0];
        let mut grad_in = [0.0_f32; 3];
        relu_backward(&input, &grad_out, &mut grad_in);
        assert_eq!(grad_in, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn relu_backward_single() {
        let mut grad_in = [0.0_f32];
        relu_backward(&[5.0], &[3.0], &mut grad_in);
        assert_eq!(grad_in[0], 3.0);
    }

    #[test]
    fn relu_backward_empty() {
        relu_backward(&[], &[], &mut []);
    }

    // ---- SiLU ----

    #[test]
    fn silu_backward_numerical() {
        let silu = |x: f32| x / (1.0 + (-x).exp());
        for &x in &[-2.0_f32, -1.0, 0.0, 0.5, 1.0, 3.0] {
            let mut grad_in = [0.0_f32];
            silu_backward(&[x], &[1.0], &mut grad_in);
            let numeric = numerical_grad(silu, x, 1e-4);
            assert!(
                (grad_in[0] - numeric).abs() < 1e-3,
                "silu grad mismatch at x={x}: analytic={}, numeric={numeric}",
                grad_in[0]
            );
        }
    }

    #[test]
    fn silu_backward_at_zero() {
        let mut grad_in = [0.0_f32];
        silu_backward(&[0.0], &[1.0], &mut grad_in);
        // silu'(0) = sigmoid(0) * (1 + 0*(1 - sigmoid(0))) = 0.5
        assert!((grad_in[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn silu_backward_large_positive() {
        let mut grad_in = [0.0_f32];
        silu_backward(&[10.0], &[1.0], &mut grad_in);
        // silu'(x) ≈ 1 for large x (sigmoid ≈ 1)
        assert!((grad_in[0] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn silu_backward_large_negative() {
        let mut grad_in = [0.0_f32];
        silu_backward(&[-10.0], &[1.0], &mut grad_in);
        // silu'(x) ≈ 0 for large negative x
        assert!(grad_in[0].abs() < 1e-3);
    }

    #[test]
    fn silu_backward_grad_scaling() {
        let mut grad_in1 = [0.0_f32];
        let mut grad_in2 = [0.0_f32];
        silu_backward(&[1.0], &[1.0], &mut grad_in1);
        silu_backward(&[1.0], &[3.0], &mut grad_in2);
        assert!((grad_in2[0] - 3.0 * grad_in1[0]).abs() < 1e-6);
    }

    #[test]
    fn silu_backward_multi() {
        let input = [0.0, 1.0, -1.0];
        let grad_out = [1.0, 1.0, 1.0];
        let mut grad_in = [0.0_f32; 3];
        silu_backward(&input, &grad_out, &mut grad_in);
        // silu'(0) = 0.5
        assert!((grad_in[0] - 0.5).abs() < 1e-5);
        // silu'(1) > 0.5
        assert!(grad_in[1] > 0.5);
        // silu'(-1) は小さい正の値
        assert!(grad_in[2] > -0.1 && grad_in[2] < 0.3);
    }

    #[test]
    fn silu_backward_empty() {
        silu_backward(&[], &[], &mut []);
    }

    // ---- GELU ----

    #[test]
    fn gelu_backward_numerical() {
        let sqrt_2_over_pi = std::f32::consts::FRAC_2_SQRT_PI * std::f32::consts::FRAC_1_SQRT_2;
        let c = 0.044_715_f32;
        let gelu = |x: f32| 0.5 * x * (1.0 + (sqrt_2_over_pi * (c * x * x).mul_add(x, x)).tanh());
        for &x in &[-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let mut grad_in = [0.0_f32];
            gelu_backward(&[x], &[1.0], &mut grad_in);
            let numeric = numerical_grad(gelu, x, 1e-4);
            assert!(
                (grad_in[0] - numeric).abs() < 1e-2,
                "gelu grad mismatch at x={x}: analytic={}, numeric={numeric}",
                grad_in[0]
            );
        }
    }

    #[test]
    fn gelu_backward_at_zero() {
        let mut grad_in = [0.0_f32];
        gelu_backward(&[0.0], &[1.0], &mut grad_in);
        // gelu'(0) = 0.5
        assert!((grad_in[0] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn gelu_backward_large_positive() {
        let mut grad_in = [0.0_f32];
        gelu_backward(&[5.0], &[1.0], &mut grad_in);
        // gelu'(x) ≈ 1 for large x
        assert!((grad_in[0] - 1.0).abs() < 1e-2);
    }

    #[test]
    fn gelu_backward_large_negative() {
        let mut grad_in = [0.0_f32];
        gelu_backward(&[-5.0], &[1.0], &mut grad_in);
        // gelu'(x) ≈ 0 for large negative x
        assert!(grad_in[0].abs() < 1e-2);
    }

    #[test]
    fn gelu_backward_symmetry() {
        // gelu'(-x) + gelu'(x) は約1になるはず（近似）
        let mut g_pos = [0.0_f32];
        let mut g_neg = [0.0_f32];
        gelu_backward(&[1.0], &[1.0], &mut g_pos);
        gelu_backward(&[-1.0], &[1.0], &mut g_neg);
        // 厳密な対称性はないが、合計は1に近い
        let sum = g_pos[0] + g_neg[0];
        assert!((sum - 1.0).abs() < 0.15, "sum={sum}");
    }

    #[test]
    fn gelu_backward_grad_scaling() {
        let mut g1 = [0.0_f32];
        let mut g2 = [0.0_f32];
        gelu_backward(&[1.0], &[1.0], &mut g1);
        gelu_backward(&[1.0], &[5.0], &mut g2);
        assert!((g2[0] - 5.0 * g1[0]).abs() < 1e-6);
    }

    #[test]
    fn gelu_backward_multi() {
        let input = [-1.0, 0.0, 1.0, 2.0];
        let grad_out = [1.0; 4];
        let mut grad_in = [0.0_f32; 4];
        gelu_backward(&input, &grad_out, &mut grad_in);
        // gelu'(0) ≈ 0.5
        assert!((grad_in[1] - 0.5).abs() < 1e-4);
        // gelu'(x) は x > 0 で 0.5 以上
        assert!(grad_in[2] > 0.5);
        assert!(grad_in[3] > 0.9);
    }

    #[test]
    fn gelu_backward_empty() {
        gelu_backward(&[], &[], &mut []);
    }

    // ---- クロスチェック: 全活性化で zero grad → zero output ----

    #[test]
    fn all_activations_zero_grad_output() {
        let input = [1.0, -1.0, 0.5, -0.5];
        let grad_out = [0.0; 4];
        let mut grad_in = [999.0_f32; 4];

        relu_backward(&input, &grad_out, &mut grad_in);
        assert!(grad_in.iter().all(|&g| g == 0.0));

        silu_backward(&input, &grad_out, &mut grad_in);
        assert!(grad_in.iter().all(|&g| g == 0.0));

        gelu_backward(&input, &grad_out, &mut grad_in);
        assert!(grad_in.iter().all(|&g| g.abs() < 1e-10));
    }

    // ---- 勾配の符号が正しいことを検証 ----

    #[test]
    fn relu_backward_gradient_sign() {
        // 正の入力で正のgrad_out → 正のgrad_in
        let mut g = [0.0_f32];
        relu_backward(&[1.0], &[1.0], &mut g);
        assert!(g[0] > 0.0);
        // 負の入力 → ゼロ
        relu_backward(&[-1.0], &[1.0], &mut g);
        assert_eq!(g[0], 0.0);
    }

    #[test]
    fn silu_backward_gradient_sign_positive_input() {
        let mut g = [0.0_f32];
        silu_backward(&[2.0], &[1.0], &mut g);
        assert!(g[0] > 0.0, "silu'(2.0) should be positive");
    }

    #[test]
    fn gelu_backward_gradient_sign_positive_input() {
        let mut g = [0.0_f32];
        gelu_backward(&[2.0], &[1.0], &mut g);
        assert!(g[0] > 0.0, "gelu'(2.0) should be positive");
    }

    // ---- パニックテスト ----

    #[test]
    #[should_panic(expected = "assertion")]
    fn relu_backward_panics_on_length_mismatch() {
        relu_backward(&[1.0, 2.0], &[1.0], &mut [0.0; 2]);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn silu_backward_panics_on_length_mismatch() {
        silu_backward(&[1.0], &[1.0, 2.0], &mut [0.0]);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn gelu_backward_panics_on_length_mismatch() {
        gelu_backward(&[1.0], &[1.0], &mut [0.0; 2]);
    }

    // ---- 追加テスト ----

    #[test]
    fn relu_backward_negative_grad_output() {
        // 負の grad_output が正の入力に対してそのまま通ることを検証
        let input = [1.0, 2.0, -1.0, 3.0];
        let grad_out = [-5.0, -3.0, -1.0, -7.0];
        let mut grad_in = [0.0_f32; 4];
        relu_backward(&input, &grad_out, &mut grad_in);
        assert_eq!(grad_in, [-5.0, -3.0, 0.0, -7.0]);
    }

    #[test]
    fn silu_backward_numerical_wide_range() {
        // 広範囲 [-5, 5] で SiLU の解析微分と数値微分が一致することを検証
        let silu = |x: f32| x / (1.0 + (-x).exp());
        for i in -50..=50 {
            let x = i as f32 * 0.1;
            let mut grad_in = [0.0_f32];
            silu_backward(&[x], &[1.0], &mut grad_in);
            let numeric = numerical_grad(silu, x, 1e-4);
            assert!(
                (grad_in[0] - numeric).abs() < 5e-3,
                "silu wide range mismatch at x={x}: analytic={}, numeric={numeric}",
                grad_in[0]
            );
        }
    }

    #[test]
    fn gelu_backward_positive_for_positive_input() {
        // GELU' は正の入力領域で常に正であることを検証
        // 注: GELU' は単調ではない（x≈1.5 で一度 1 を超えてから 1 に収束する）
        for i in 1..=100 {
            let x = i as f32 * 0.1;
            let mut g = [0.0_f32];
            gelu_backward(&[x], &[1.0], &mut g);
            assert!(g[0] > 0.0, "gelu'({x}) should be positive, got {}", g[0]);
        }
    }

    #[test]
    fn all_activations_single_vs_vector_consistency() {
        // 単一要素とベクトルの結果が一致することを検証
        let vals = [0.5_f32, -1.0, 2.0];
        let ones = [1.0_f32; 3];

        let mut relu_vec = [0.0_f32; 3];
        let mut silu_vec = [0.0_f32; 3];
        let mut gelu_vec = [0.0_f32; 3];
        relu_backward(&vals, &ones, &mut relu_vec);
        silu_backward(&vals, &ones, &mut silu_vec);
        gelu_backward(&vals, &ones, &mut gelu_vec);

        for i in 0..3 {
            let mut r = [0.0_f32];
            let mut s = [0.0_f32];
            let mut g = [0.0_f32];
            relu_backward(&[vals[i]], &[1.0], &mut r);
            silu_backward(&[vals[i]], &[1.0], &mut s);
            gelu_backward(&[vals[i]], &[1.0], &mut g);
            assert!(
                (r[0] - relu_vec[i]).abs() < 1e-6,
                "relu mismatch at index {i}"
            );
            assert!(
                (s[0] - silu_vec[i]).abs() < 1e-6,
                "silu mismatch at index {i}"
            );
            assert!(
                (g[0] - gelu_vec[i]).abs() < 1e-6,
                "gelu mismatch at index {i}"
            );
        }
    }
}
