//! レイヤー逆伝播 — ternary matvec と `BitLinear` の backward。
//!
//! ternary 重み W ∈ {-1, 0, +1} に対して:
//! - forward: y = W * x
//! - backward (入力勾配): dx = W^T * dy （転置 matvec — add/sub のみ）
//! - backward (重み勾配): dW は STE 用の潜在重みに対して計算
//!
//! 転置 matvec もternary演算のため乗算不要。

use alice_ml::ops::TernaryWeightKernel;

/// ternary matvec の逆伝播: 入力勾配を計算。
///
/// forward: `y[j] = sum_i W[j][i] * x[i]` (`out_features` × `in_features`)
/// backward: `dx[i] = sum_j W[j][i] * dy[j]` = W^T * dy
///
/// ternary なので乗算なし — add/sub のみ。
///
/// # 引数
///
/// - `grad_output` — 出力勾配 dy (長さ = `out_features`)
/// - `weights` — forward と同じ `TernaryWeightKernel`
/// - `grad_input` — 入力勾配 dx の書き込み先 (長さ = `in_features`)
///
/// # Panics
///
/// `grad_output.len() != out_features` または `grad_input.len() != in_features` の場合。
pub fn ternary_matvec_backward(
    grad_output: &[f32],
    weights: &TernaryWeightKernel,
    grad_input: &mut [f32],
) {
    let out_features = weights.out_features();
    let in_features = weights.in_features();

    assert_eq!(grad_output.len(), out_features);
    assert_eq!(grad_input.len(), in_features);

    for g in grad_input.iter_mut() {
        *g = 0.0;
    }

    let words_per_row = weights.words_per_row();
    let plus_bits = weights.plus_bits();
    let minus_bits = weights.minus_bits();

    for (j, &dy_j) in grad_output.iter().enumerate() {
        if dy_j == 0.0 {
            continue;
        }

        let row_offset = j * words_per_row;
        for w in 0..words_per_row {
            let p = plus_bits[row_offset + w];
            let m = minus_bits[row_offset + w];

            if p == 0 && m == 0 {
                continue;
            }

            let base_col = w * 32;
            let mut pbits = p;
            while pbits != 0 {
                let bit = pbits.trailing_zeros() as usize;
                let col = base_col + bit;
                if col < in_features {
                    grad_input[col] += dy_j;
                }
                pbits &= pbits - 1;
            }
            let mut mbits = m;
            while mbits != 0 {
                let bit = mbits.trailing_zeros() as usize;
                let col = base_col + bit;
                if col < in_features {
                    grad_input[col] -= dy_j;
                }
                mbits &= mbits - 1;
            }
        }
    }
}

/// `BitLinear` レイヤーの逆伝播。
///
/// forward: `y = W * norm(x) + bias`
/// backward:
/// - `dx = W^T * dy * inv_rms` (`pre_norm` の場合、スケーリング補正含む)
/// - `d_bias = dy` (bias がある場合)
///
/// # 引数
///
/// - `input` — forward 時の入力 x
/// - `grad_output` — 出力勾配 dy
/// - `weights` — `TernaryWeightKernel`
/// - `pre_norm` — `RMSNorm` が適用されたか
/// - `norm_eps` — `RMSNorm` の epsilon
/// - `grad_input` — 入力勾配 dx の書き込み先
/// - `grad_bias` — bias 勾配の書き込み先 (None なら bias なし)
///
/// # Panics
///
/// `input.len() != in_features`, `grad_output.len() != out_features`,
/// `grad_input.len() != in_features`, `grad_bias` の長さが `out_features` と異なる場合。
pub fn bitlinear_backward(
    input: &[f32],
    grad_output: &[f32],
    weights: &TernaryWeightKernel,
    pre_norm: bool,
    norm_eps: f32,
    grad_input: &mut [f32],
    grad_bias: Option<&mut [f32]>,
) {
    let out_features = weights.out_features();
    let in_features = weights.in_features();

    assert_eq!(input.len(), in_features);
    assert_eq!(grad_output.len(), out_features);
    assert_eq!(grad_input.len(), in_features);

    if let Some(gb) = grad_bias {
        assert_eq!(gb.len(), out_features);
        gb.copy_from_slice(grad_output);
    }

    if pre_norm {
        let mut sum_sq: f32 = 0.0;
        for &x in input {
            sum_sq += x * x;
        }
        let rms = (sum_sq / input.len() as f32 + norm_eps).sqrt();
        let inv_rms = 1.0 / rms;

        let scaled_grad: Vec<f32> = grad_output.iter().map(|&g| g * inv_rms).collect();
        ternary_matvec_backward(&scaled_grad, weights, grad_input);

        let mut wx = vec![0.0_f32; out_features];
        alice_ml::ops::ternary_matvec_kernel(input, weights, &mut wx);

        let dot: f32 = grad_output
            .iter()
            .zip(wx.iter())
            .map(|(&g, &w)| g * w)
            .sum();
        let rms3 = rms * rms * rms;
        let n = input.len() as f32;

        for i in 0..in_features {
            grad_input[i] -= dot * input[i] / (n * rms3);
        }

        drop(scaled_grad);
    } else {
        ternary_matvec_backward(grad_output, weights, grad_input);
    }
}

/// STE (Straight-Through Estimator) による重み勾配計算。
///
/// ternary 重みは離散値なので直接微分できない。
/// 潜在 FP32 重みに対して `dW_latent[j][i] = dy[j] * x[i]` を計算し、
/// STE で勾配をそのまま通す。勾配は累積（+=）される。
///
/// # 引数
///
/// - `input` — forward 時の入力 x (長さ = `in_features`)
/// - `grad_output` — 出力勾配 dy (長さ = `out_features`)
/// - `grad_weights` — 潜在重み勾配 (長さ = `out_features * in_features`, row-major)
///
/// # Panics
///
/// `grad_weights.len() != grad_output.len() * input.len()` の場合。
pub fn ste_weight_grad(input: &[f32], grad_output: &[f32], grad_weights: &mut [f32]) {
    let out_features = grad_output.len();
    let in_features = input.len();
    assert_eq!(grad_weights.len(), out_features * in_features);

    for (j, &dy_j) in grad_output.iter().enumerate() {
        let row_offset = j * in_features;
        for i in 0..in_features {
            grad_weights[row_offset + i] += dy_j * input[i];
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use alice_ml::ops::TernaryWeightKernel;

    // ---- ternary_matvec_backward ----

    #[test]
    fn ternary_backward_identity_like() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, 0, 0, 1], 2, 2);
        let grad_out = [3.0_f32, 7.0];
        let mut grad_in = [0.0_f32; 2];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - 3.0).abs() < 1e-6);
        assert!((grad_in[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn ternary_backward_mixed() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let grad_out = [2.0_f32, 3.0];
        let mut grad_in = [0.0_f32; 2];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - 2.0).abs() < 1e-6);
        assert!((grad_in[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ternary_backward_3x3() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1, 1, -1, 0, -1, 1], 3, 3);
        let grad_out = [1.0_f32, 1.0, 1.0];
        let mut grad_in = [0.0_f32; 3];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - 2.0).abs() < 1e-6);
        assert!((grad_in[1] - (-1.0)).abs() < 1e-6);
        assert!((grad_in[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn ternary_backward_all_zeros() {
        let kernel = TernaryWeightKernel::from_ternary(&[0, 0, 0, 0], 2, 2);
        let grad_out = [5.0_f32, 3.0];
        let mut grad_in = [0.0_f32; 2];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert_eq!(grad_in, [0.0, 0.0]);
    }

    #[test]
    fn ternary_backward_all_plus() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, 1, 1, 1], 2, 2);
        let grad_out = [2.0_f32, 3.0];
        let mut grad_in = [0.0_f32; 2];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        // W^T * [2,3] = [[1,1],[1,1]] * [2,3] = [5, 5]
        assert!((grad_in[0] - 5.0).abs() < 1e-6);
        assert!((grad_in[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn ternary_backward_all_minus() {
        let kernel = TernaryWeightKernel::from_ternary(&[-1, -1, -1, -1], 2, 2);
        let grad_out = [2.0_f32, 3.0];
        let mut grad_in = [0.0_f32; 2];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - (-5.0)).abs() < 1e-6);
        assert!((grad_in[1] - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn ternary_backward_zero_grad_output() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let grad_out = [0.0_f32, 0.0];
        let mut grad_in = [0.0_f32; 2];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert_eq!(grad_in, [0.0, 0.0]);
    }

    #[test]
    fn ternary_backward_single_row() {
        // 1x3 行列: W = [1, -1, 0]
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0], 1, 3);
        let grad_out = [4.0_f32];
        let mut grad_in = [0.0_f32; 3];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - 4.0).abs() < 1e-6);
        assert!((grad_in[1] - (-4.0)).abs() < 1e-6);
        assert!((grad_in[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn ternary_backward_single_col() {
        // 3x1 行列: W = [[1], [-1], [0]]
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0], 3, 1);
        let grad_out = [1.0_f32, 2.0, 3.0];
        let mut grad_in = [0.0_f32; 1];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        // dx = 1*1 + (-1)*2 + 0*3 = -1
        assert!((grad_in[0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn ternary_backward_negative_grad() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let grad_out = [-1.0_f32, -2.0];
        let mut grad_in = [0.0_f32; 2];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        // W^T * [-1,-2] = [[1,0],[-1,1]] * [-1,-2] = [-1, -1]
        assert!((grad_in[0] - (-1.0)).abs() < 1e-6);
        assert!((grad_in[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn ternary_backward_numerical_check() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1, -1, 1], 2, 3);
        let x = [1.0_f32, 2.0, 3.0];
        let eps = 1e-4_f32;

        let forward = |input: &[f32]| -> f32 {
            let mut output = [0.0_f32; 2];
            alice_ml::ops::ternary_matvec_kernel(input, &kernel, &mut output);
            output.iter().sum::<f32>()
        };

        let grad_out = [1.0_f32, 1.0];
        let mut grad_in = [0.0_f32; 3];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);

        for i in 0..3 {
            let mut x_plus = x;
            let mut x_minus = x;
            x_plus[i] += eps;
            x_minus[i] -= eps;
            let numeric = (forward(&x_plus) - forward(&x_minus)) / (2.0 * eps);
            assert!(
                (grad_in[i] - numeric).abs() < 5e-3,
                "dim {i}: analytic={}, numeric={numeric}",
                grad_in[i]
            );
        }
    }

    #[test]
    fn ternary_backward_4x4() {
        // 4x4 行列でより大きなケースを検証
        let vals: Vec<i8> = vec![1, -1, 0, 1, 0, 1, -1, 0, -1, 0, 1, 1, 1, -1, -1, 0];
        let kernel = TernaryWeightKernel::from_ternary(&vals, 4, 4);
        let grad_out = [1.0_f32, 2.0, 3.0, 4.0];
        let mut grad_in = [0.0_f32; 4];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);

        // W^T * [1,2,3,4] を手計算:
        // col 0: W[0][0]=1, W[1][0]=0, W[2][0]=-1, W[3][0]=1
        // dx[0] = 1*1 + 0*2 + (-1)*3 + 1*4 = 2
        assert!((grad_in[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ternary_backward_forward_roundtrip() {
        // forward + backward で入力勾配の整合性を検証
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 1, 0, -1, 1], 2, 3);
        let input = [1.0_f32, 2.0, 3.0];
        let mut output = [0.0_f32; 2];
        alice_ml::ops::ternary_matvec_kernel(&input, &kernel, &mut output);

        let grad_out = [1.0_f32, 1.0];
        let mut grad_in = [0.0_f32; 3];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);

        // backward が NaN/Inf を生成しないことを確認
        for &g in &grad_in {
            assert!(g.is_finite(), "gradient should be finite");
        }
    }

    // ---- bitlinear_backward ----

    #[test]
    fn bitlinear_backward_no_norm() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let input = [2.0_f32, 3.0];
        let grad_out = [1.0_f32, 1.0];
        let mut grad_in = [0.0_f32; 2];
        bitlinear_backward(&input, &grad_out, &kernel, false, 1e-5, &mut grad_in, None);
        assert!((grad_in[0] - 1.0).abs() < 1e-6);
        assert!((grad_in[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn bitlinear_backward_with_bias() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let input = [2.0_f32, 3.0];
        let grad_out = [0.5_f32, 1.5];
        let mut grad_in = [0.0_f32; 2];
        let mut grad_bias = [0.0_f32; 2];
        bitlinear_backward(
            &input,
            &grad_out,
            &kernel,
            false,
            1e-5,
            &mut grad_in,
            Some(&mut grad_bias),
        );
        assert!((grad_bias[0] - 0.5).abs() < 1e-6);
        assert!((grad_bias[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn bitlinear_backward_bias_equals_grad_output() {
        // d_bias は常に grad_output と同じ
        let kernel = TernaryWeightKernel::from_ternary(&[1, 0, -1, 1, 0, -1, 1, 0, -1], 3, 3);
        let input = [1.0_f32, 2.0, 3.0];
        let grad_out = [7.0_f32, -3.0, 0.5];
        let mut grad_in = [0.0_f32; 3];
        let mut grad_bias = [0.0_f32; 3];
        bitlinear_backward(
            &input,
            &grad_out,
            &kernel,
            false,
            1e-5,
            &mut grad_in,
            Some(&mut grad_bias),
        );
        for i in 0..3 {
            assert!(
                (grad_bias[i] - grad_out[i]).abs() < 1e-6,
                "grad_bias[{i}] mismatch"
            );
        }
    }

    #[test]
    fn bitlinear_backward_pre_norm_numerical() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let x = [2.0_f32, 3.0];
        let eps = 1e-4_f32;
        let norm_eps = 1e-5_f32;

        let forward = |input: &[f32]| -> f32 {
            let mut output = [0.0_f32; 2];
            let mut sum_sq: f32 = 0.0;
            for &xi in input {
                sum_sq += xi * xi;
            }
            let rms = (sum_sq / input.len() as f32 + norm_eps).sqrt();
            let inv_rms = 1.0 / rms;
            alice_ml::ops::ternary_matvec_kernel(input, &kernel, &mut output);
            output.iter().map(|&o| o * inv_rms).sum::<f32>()
        };

        let grad_out = [1.0_f32, 1.0];
        let mut grad_in = [0.0_f32; 2];
        bitlinear_backward(&x, &grad_out, &kernel, true, norm_eps, &mut grad_in, None);

        for i in 0..2 {
            let mut x_plus = x;
            let mut x_minus = x;
            x_plus[i] += eps;
            x_minus[i] -= eps;
            let numeric = (forward(&x_plus) - forward(&x_minus)) / (2.0 * eps);
            assert!(
                (grad_in[i] - numeric).abs() < 1e-2,
                "dim {i}: analytic={}, numeric={numeric}",
                grad_in[i]
            );
        }
    }

    #[test]
    fn bitlinear_backward_pre_norm_3x3_numerical() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1, 1, -1, 0, -1, 1], 3, 3);
        let x = [1.0_f32, 2.0, 3.0];
        let eps = 1e-4_f32;
        let norm_eps = 1e-5_f32;

        let forward = |input: &[f32]| -> f32 {
            let mut output = [0.0_f32; 3];
            let mut sum_sq: f32 = 0.0;
            for &xi in input {
                sum_sq += xi * xi;
            }
            let rms = (sum_sq / input.len() as f32 + norm_eps).sqrt();
            let inv_rms = 1.0 / rms;
            alice_ml::ops::ternary_matvec_kernel(input, &kernel, &mut output);
            output.iter().map(|&o| o * inv_rms).sum::<f32>()
        };

        let grad_out = [1.0_f32, 1.0, 1.0];
        let mut grad_in = [0.0_f32; 3];
        bitlinear_backward(&x, &grad_out, &kernel, true, norm_eps, &mut grad_in, None);

        for i in 0..3 {
            let mut x_plus = x;
            let mut x_minus = x;
            x_plus[i] += eps;
            x_minus[i] -= eps;
            let numeric = (forward(&x_plus) - forward(&x_minus)) / (2.0 * eps);
            assert!(
                (grad_in[i] - numeric).abs() < 1e-2,
                "dim {i}: analytic={}, numeric={numeric}",
                grad_in[i]
            );
        }
    }

    #[test]
    fn bitlinear_backward_pre_norm_with_bias_numerical() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let x = [2.0_f32, 3.0];
        let norm_eps = 1e-5_f32;

        let grad_out = [1.0_f32, 1.0];
        let mut grad_in = [0.0_f32; 2];
        let mut grad_bias = [0.0_f32; 2];
        bitlinear_backward(
            &x,
            &grad_out,
            &kernel,
            true,
            norm_eps,
            &mut grad_in,
            Some(&mut grad_bias),
        );

        // bias 勾配は pre_norm に関係なく grad_output と同じ
        assert!((grad_bias[0] - 1.0).abs() < 1e-6);
        assert!((grad_bias[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bitlinear_backward_finite_outputs() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let input = [100.0_f32, -50.0];
        let grad_out = [0.01_f32, -0.02];
        let mut grad_in = [0.0_f32; 2];
        bitlinear_backward(&input, &grad_out, &kernel, true, 1e-5, &mut grad_in, None);
        for &g in &grad_in {
            assert!(g.is_finite());
        }
    }

    // ---- ste_weight_grad ----

    #[test]
    fn ste_weight_grad_basic() {
        let input = [1.0_f32, 2.0];
        let grad_out = [3.0_f32, 5.0];
        let mut grad_w = [0.0_f32; 4];
        ste_weight_grad(&input, &grad_out, &mut grad_w);
        assert!((grad_w[0] - 3.0).abs() < 1e-6);
        assert!((grad_w[1] - 6.0).abs() < 1e-6);
        assert!((grad_w[2] - 5.0).abs() < 1e-6);
        assert!((grad_w[3] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn ste_weight_grad_accumulates() {
        let input = [1.0_f32];
        let grad_out = [2.0_f32];
        let mut grad_w = [1.0_f32];
        ste_weight_grad(&input, &grad_out, &mut grad_w);
        assert!((grad_w[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn ste_weight_grad_3x2() {
        let input = [1.0_f32, 2.0];
        let grad_out = [1.0_f32, 2.0, 3.0];
        let mut grad_w = [0.0_f32; 6]; // 3x2
        ste_weight_grad(&input, &grad_out, &mut grad_w);
        // dW[j][i] = dy[j] * x[i]
        assert!((grad_w[0] - 1.0).abs() < 1e-6); // dy[0]*x[0]
        assert!((grad_w[1] - 2.0).abs() < 1e-6); // dy[0]*x[1]
        assert!((grad_w[2] - 2.0).abs() < 1e-6); // dy[1]*x[0]
        assert!((grad_w[3] - 4.0).abs() < 1e-6); // dy[1]*x[1]
        assert!((grad_w[4] - 3.0).abs() < 1e-6); // dy[2]*x[0]
        assert!((grad_w[5] - 6.0).abs() < 1e-6); // dy[2]*x[1]
    }

    #[test]
    fn ste_weight_grad_zero_grad() {
        let input = [1.0_f32, 2.0];
        let grad_out = [0.0_f32, 0.0];
        let mut grad_w = [0.0_f32; 4];
        ste_weight_grad(&input, &grad_out, &mut grad_w);
        assert!(grad_w.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn ste_weight_grad_zero_input() {
        let input = [0.0_f32, 0.0];
        let grad_out = [5.0_f32, 3.0];
        let mut grad_w = [0.0_f32; 4];
        ste_weight_grad(&input, &grad_out, &mut grad_w);
        assert!(grad_w.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn ste_weight_grad_negative() {
        let input = [-1.0_f32];
        let grad_out = [-2.0_f32];
        let mut grad_w = [0.0_f32; 1];
        ste_weight_grad(&input, &grad_out, &mut grad_w);
        // (-2) * (-1) = 2
        assert!((grad_w[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ste_weight_grad_multi_accumulate() {
        let mut grad_w = [0.0_f32; 2]; // 1x2
        ste_weight_grad(&[1.0, 2.0], &[1.0], &mut grad_w);
        ste_weight_grad(&[3.0, 4.0], &[1.0], &mut grad_w);
        // batch 累積: [1+3, 2+4] = [4, 6]
        assert!((grad_w[0] - 4.0).abs() < 1e-6);
        assert!((grad_w[1] - 6.0).abs() < 1e-6);
    }

    // ---- パニックテスト ----

    #[test]
    #[should_panic(expected = "assertion")]
    fn ternary_backward_panics_on_grad_output_mismatch() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let mut grad_in = [0.0_f32; 2];
        ternary_matvec_backward(&[1.0], &kernel, &mut grad_in); // out_features=2だが1つ
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn ternary_backward_panics_on_grad_input_mismatch() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let mut grad_in = [0.0_f32; 3]; // in_features=2だが3つ
        ternary_matvec_backward(&[1.0, 2.0], &kernel, &mut grad_in);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn bitlinear_backward_panics_on_input_mismatch() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let mut grad_in = [0.0_f32; 2];
        bitlinear_backward(
            &[1.0],
            &[1.0, 2.0],
            &kernel,
            false,
            1e-5,
            &mut grad_in,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn ste_weight_grad_panics_on_size_mismatch() {
        ste_weight_grad(&[1.0, 2.0], &[3.0], &mut [0.0; 3]); // 1*2=2 != 3
    }

    // ---- 追加テスト ----

    #[test]
    fn ternary_backward_rectangular_2x4() {
        // 非正方行列 (2x4) での転置 matvec を検証
        // W = [[1, -1, 0, 1], [0, 1, -1, 0]] (2行4列)
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1, 0, 1, -1, 0], 2, 4);
        let grad_out = [3.0_f32, 5.0];
        let mut grad_in = [0.0_f32; 4];
        ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        // W^T * [3,5]:
        // col0: 1*3 + 0*5 = 3
        // col1: (-1)*3 + 1*5 = 2
        // col2: 0*3 + (-1)*5 = -5
        // col3: 1*3 + 0*5 = 3
        assert!((grad_in[0] - 3.0).abs() < 1e-6);
        assert!((grad_in[1] - 2.0).abs() < 1e-6);
        assert!((grad_in[2] - (-5.0)).abs() < 1e-6);
        assert!((grad_in[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn bitlinear_backward_no_norm_zero_input() {
        // 入力がゼロベクトルの場合、勾配が正しく計算されることを検証
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let input = [0.0_f32, 0.0];
        let grad_out = [1.0_f32, 1.0];
        let mut grad_in = [0.0_f32; 2];
        bitlinear_backward(&input, &grad_out, &kernel, false, 1e-5, &mut grad_in, None);
        // norm なしなので ternary_matvec_backward と同じ結果
        let mut expected = [0.0_f32; 2];
        ternary_matvec_backward(&grad_out, &kernel, &mut expected);
        assert!((grad_in[0] - expected[0]).abs() < 1e-6);
        assert!((grad_in[1] - expected[1]).abs() < 1e-6);
    }

    #[test]
    fn bitlinear_backward_pre_norm_uniform_input() {
        // 均一入力 [c, c] での pre_norm backward が有限値を返すことを検証
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, -1, 1], 2, 2);
        let input = [3.0_f32, 3.0];
        let grad_out = [1.0_f32, -1.0];
        let mut grad_in = [0.0_f32; 2];
        bitlinear_backward(&input, &grad_out, &kernel, true, 1e-5, &mut grad_in, None);
        for &g in &grad_in {
            assert!(g.is_finite(), "gradient should be finite for uniform input");
        }
        // 均一入力でW=[1,-1; -1,1], grad_out=[1,-1] → 勾配は非ゼロ
        assert!(
            grad_in.iter().any(|&g| g.abs() > 1e-8),
            "gradients should be non-zero"
        );
    }

    #[test]
    fn ste_weight_grad_large_matrix() {
        // 4x4 行列での STE 重み勾配を検証
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let grad_out = [0.5_f32, -1.0, 1.5, -0.5];
        let mut grad_w = [0.0_f32; 16]; // 4x4
        ste_weight_grad(&input, &grad_out, &mut grad_w);
        // dW[j][i] = dy[j] * x[i] を全要素検証
        for j in 0..4 {
            for i in 0..4 {
                let expected = grad_out[j] * input[i];
                assert!(
                    (grad_w[j * 4 + i] - expected).abs() < 1e-6,
                    "mismatch at [{j}][{i}]: got={}, expected={expected}",
                    grad_w[j * 4 + i]
                );
            }
        }
    }
}
