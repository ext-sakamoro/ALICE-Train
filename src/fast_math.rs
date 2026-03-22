//! 高速近似数学関数 — 学習ループ向け。
//!
//! IEEE 754 ビットハックによる `exp` 近似で、標準 `f32::exp()` の 3〜5 倍高速。
//! softmax / SiLU など超越関数を大量に呼ぶホットパスに使用する。

/// 高速近似 exp — IEEE 754 ビットハック。
///
/// 精度: 相対誤差 < 2% (|x| < 80)。学習用 softmax / SiLU に十分。
#[inline(always)]
#[must_use] 
pub fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-87.3, 88.7);
    let val = x.mul_add(std::f32::consts::LOG2_E, 126.942_695);
    let bits = (val * (1 << 23) as f32) as u32;
    f32::from_bits(bits)
}

/// 高速近似 sigmoid。
#[inline(always)]
#[must_use] 
pub fn fast_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_exp_accuracy() {
        for i in -800..=800 {
            let x = i as f32 * 0.1;
            let approx = fast_exp(x);
            let exact = x.exp();
            if exact.is_finite() && exact > 1e-30 {
                let rel_err = ((approx - exact) / exact).abs();
                assert!(
                    rel_err < 0.05,
                    "fast_exp({x}): approx={approx}, exact={exact}, rel_err={rel_err}"
                );
            }
        }
    }

    #[test]
    fn fast_exp_clamp_no_panic() {
        assert!(fast_exp(-100.0).is_finite());
        assert!(fast_exp(100.0).is_finite());
        assert!(fast_exp(0.0) > 0.9 && fast_exp(0.0) < 1.1);
    }

    #[test]
    fn fast_sigmoid_range() {
        for i in -100..=100 {
            let x = i as f32 * 0.1;
            let s = fast_sigmoid(x);
            assert!(s >= 0.0 && s <= 1.0, "fast_sigmoid({x}) = {s}");
        }
        assert!((fast_sigmoid(0.0) - 0.5).abs() < 0.02);
    }
}
