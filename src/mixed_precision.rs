//! 混合精度ユーティリティ — BF16 forward + FP32 勾配。
//!
//! 7B クラスの学習ではメモリが制約となる。
//! BF16 (Brain Float 16) で forward パスを行い、勾配は FP32 で保持することで
//! メモリ使用量を約半減させる。
//!
//! BF16 フォーマット: 1 bit sign + 8 bit exponent + 7 bit mantissa
//! f32 と同じ指数部を持つため、範囲は同じだが精度が低い。

/// BF16 を表す newtype（u16 で内部保持）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16(u16);

impl Bf16 {
    /// raw u16 から構築する。
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// raw u16 を返す。
    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// f32 → BF16 変換（round-to-nearest-even）。
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        // NaN の場合は mantissa を保持
        if value.is_nan() {
            return Self((bits >> 16) as u16 | 0x0040); // quiet NaN
        }
        // Round to nearest even
        let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        let rounded = bits.wrapping_add(rounding_bias);
        Self((rounded >> 16) as u16)
    }

    /// BF16 → f32 変換。
    #[must_use]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// ゼロかどうか。
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0.trailing_zeros() >= 15
    }
}

/// 混合精度設定。
#[derive(Clone, Debug)]
pub struct MixedPrecisionConfig {
    /// BF16 forward を有効にするか。
    pub enabled: bool,
    /// Loss scaling の初期値（勾配アンダーフロー防止）。
    pub loss_scale: f32,
    /// Loss scale の動的調整を有効にするか。
    pub dynamic_loss_scaling: bool,
    /// NaN/Inf が発生しなかった連続ステップ数のしきい値
    /// （超えたら scale を増やす）。
    pub scale_growth_interval: usize,
    /// Scale 増加倍率。
    pub scale_growth_factor: f32,
    /// NaN/Inf 発生時の scale 縮小倍率。
    pub scale_backoff_factor: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            loss_scale: 1024.0,
            dynamic_loss_scaling: true,
            scale_growth_interval: 200,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
        }
    }
}

impl MixedPrecisionConfig {
    /// デフォルト設定を返す。
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// 無効（全て FP32）の設定を返す。
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// 動的 loss scaler。
///
/// 勾配の NaN/Inf を検出し、loss scale を自動調整する。
#[derive(Clone, Debug)]
pub struct LossScaler {
    /// 現在の loss scale。
    scale: f32,
    /// 設定。
    config: MixedPrecisionConfig,
    /// NaN/Inf が発生しなかった連続ステップ数。
    good_steps: usize,
}

impl LossScaler {
    /// 新しい `LossScaler` を構築する。
    #[must_use]
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let scale = config.loss_scale;
        Self {
            scale,
            config,
            good_steps: 0,
        }
    }

    /// 現在の loss scale を返す。
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// 勾配にスケールを適用する（loss を scale 倍して backward → 勾配も scale 倍）。
    #[must_use]
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// 勾配のスケールを戻す（scale で割る）。
    pub fn unscale_gradients(&self, gradients: &mut [f32]) {
        if self.scale == 0.0 {
            return;
        }
        let inv_scale = 1.0 / self.scale;
        for g in gradients.iter_mut() {
            *g *= inv_scale;
        }
    }

    /// 勾配に NaN/Inf が含まれるか検査する。
    #[must_use]
    pub fn check_gradients(gradients: &[f32]) -> bool {
        gradients.iter().all(|g| g.is_finite())
    }

    /// ステップ完了後に scale を更新する。
    ///
    /// # Returns
    ///
    /// 勾配が有効（NaN/Inf なし）なら `true`。
    /// `false` の場合はパラメータ更新をスキップすべき。
    pub fn update(&mut self, gradients_valid: bool) -> bool {
        if !self.config.dynamic_loss_scaling {
            return gradients_valid;
        }

        if gradients_valid {
            self.good_steps += 1;
            if self.good_steps >= self.config.scale_growth_interval {
                self.scale *= self.config.scale_growth_factor;
                self.good_steps = 0;
            }
            true
        } else {
            self.scale *= self.config.scale_backoff_factor;
            if self.scale < 1.0 {
                self.scale = 1.0;
            }
            self.good_steps = 0;
            false
        }
    }
}

/// f32 スライスを BF16 に一括変換する。
pub fn f32_to_bf16_batch(src: &[f32], dst: &mut [Bf16]) {
    assert_eq!(src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = Bf16::from_f32(s);
    }
}

/// BF16 スライスを f32 に一括変換する。
pub fn bf16_to_f32_batch(src: &[Bf16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = s.to_f32();
    }
}

/// f32 スライスを BF16 に変換した `Vec<Bf16>` を返す。
#[must_use]
pub fn f32_to_bf16_vec(src: &[f32]) -> Vec<Bf16> {
    src.iter().map(|&v| Bf16::from_f32(v)).collect()
}

/// BF16 スライスを f32 に変換した `Vec<f32>` を返す。
#[must_use]
pub fn bf16_to_f32_vec(src: &[Bf16]) -> Vec<f32> {
    src.iter().map(|v| v.to_f32()).collect()
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- Bf16 ---

    #[test]
    fn bf16_roundtrip_one() {
        let bf = Bf16::from_f32(1.0);
        assert!((bf.to_f32() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bf16_roundtrip_negative() {
        let bf = Bf16::from_f32(-3.14);
        assert!((bf.to_f32() - (-3.14)).abs() < 0.02);
    }

    #[test]
    fn bf16_roundtrip_zero() {
        let bf = Bf16::from_f32(0.0);
        assert!(bf.is_zero());
        assert!((bf.to_f32() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn bf16_negative_zero() {
        let bf = Bf16::from_f32(-0.0);
        assert!(bf.is_zero());
    }

    #[test]
    fn bf16_roundtrip_large() {
        let bf = Bf16::from_f32(65536.0);
        assert!((bf.to_f32() - 65536.0).abs() < 1.0);
    }

    #[test]
    fn bf16_roundtrip_small() {
        let bf = Bf16::from_f32(0.001);
        assert!((bf.to_f32() - 0.001).abs() < 0.0001);
    }

    #[test]
    fn bf16_nan() {
        let bf = Bf16::from_f32(f32::NAN);
        assert!(bf.to_f32().is_nan());
    }

    #[test]
    fn bf16_infinity() {
        let bf_pos = Bf16::from_f32(f32::INFINITY);
        assert!(bf_pos.to_f32().is_infinite());
        assert!(bf_pos.to_f32() > 0.0);

        let bf_neg = Bf16::from_f32(f32::NEG_INFINITY);
        assert!(bf_neg.to_f32().is_infinite());
        assert!(bf_neg.to_f32() < 0.0);
    }

    #[test]
    fn bf16_from_bits() {
        let bf = Bf16::from_bits(0x3F80); // 1.0 in BF16
        assert!((bf.to_f32() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bf16_to_bits() {
        let bf = Bf16::from_f32(1.0);
        assert_eq!(bf.to_bits(), 0x3F80);
    }

    #[test]
    fn bf16_precision_loss() {
        // BF16 は 7bit mantissa → 約 1/128 の精度
        let val = 1.0078125; // 1 + 1/128
        let bf = Bf16::from_f32(val);
        assert!((bf.to_f32() - val).abs() < 0.01);
    }

    #[test]
    fn bf16_eq() {
        let a = Bf16::from_f32(1.0);
        let b = Bf16::from_f32(1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn bf16_copy() {
        let a = Bf16::from_f32(2.0);
        let b = a;
        assert_eq!(a, b);
    }

    // --- バッチ変換 ---

    #[test]
    fn f32_to_bf16_batch_basic() {
        let src = [1.0_f32, -2.0, 0.0, 3.14];
        let mut dst = vec![Bf16::from_bits(0); 4];
        f32_to_bf16_batch(&src, &mut dst);
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!((d.to_f32() - s).abs() < 0.02);
        }
    }

    #[test]
    fn bf16_to_f32_batch_basic() {
        let src = [
            Bf16::from_f32(1.0),
            Bf16::from_f32(-2.0),
            Bf16::from_f32(0.0),
        ];
        let mut dst = vec![0.0_f32; 3];
        bf16_to_f32_batch(&src, &mut dst);
        assert!((dst[0] - 1.0).abs() < 1e-6);
        assert!((dst[1] - (-2.0)).abs() < 1e-6);
        assert!((dst[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn f32_to_bf16_vec_basic() {
        let src = [1.0, 2.0, 3.0];
        let bf = f32_to_bf16_vec(&src);
        assert_eq!(bf.len(), 3);
    }

    #[test]
    fn bf16_to_f32_vec_basic() {
        let bf = vec![Bf16::from_f32(1.0), Bf16::from_f32(2.0)];
        let f = bf16_to_f32_vec(&bf);
        assert_eq!(f.len(), 2);
        assert!((f[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn batch_roundtrip() {
        let original = [0.5_f32, -1.5, 3.0, 0.001, 100.0];
        let bf = f32_to_bf16_vec(&original);
        let restored = bf16_to_f32_vec(&bf);
        for (o, r) in original.iter().zip(restored.iter()) {
            assert!(
                (o - r).abs() < o.abs() * 0.01 + 0.001,
                "roundtrip mismatch: {o} -> {r}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn f32_to_bf16_batch_length_mismatch() {
        let src = [1.0_f32; 3];
        let mut dst = vec![Bf16::from_bits(0); 2];
        f32_to_bf16_batch(&src, &mut dst);
    }

    // --- MixedPrecisionConfig ---

    #[test]
    fn config_default() {
        let c = MixedPrecisionConfig::default();
        assert!(c.enabled);
        assert!((c.loss_scale - 1024.0).abs() < 1e-6);
    }

    #[test]
    fn config_disabled() {
        let c = MixedPrecisionConfig::disabled();
        assert!(!c.enabled);
    }

    #[test]
    fn config_new() {
        let c = MixedPrecisionConfig::new();
        assert!(c.enabled);
    }

    // --- LossScaler ---

    #[test]
    fn scaler_initial_scale() {
        let s = LossScaler::new(MixedPrecisionConfig::default());
        assert!((s.scale() - 1024.0).abs() < 1e-6);
    }

    #[test]
    fn scaler_scale_loss() {
        let s = LossScaler::new(MixedPrecisionConfig::default());
        let scaled = s.scale_loss(0.5);
        assert!((scaled - 512.0).abs() < 1e-4);
    }

    #[test]
    fn scaler_unscale_gradients() {
        let s = LossScaler::new(MixedPrecisionConfig {
            loss_scale: 100.0,
            ..MixedPrecisionConfig::default()
        });
        let mut grads = [100.0, 200.0, -300.0];
        s.unscale_gradients(&mut grads);
        assert!((grads[0] - 1.0).abs() < 1e-4);
        assert!((grads[1] - 2.0).abs() < 1e-4);
        assert!((grads[2] - (-3.0)).abs() < 1e-4);
    }

    #[test]
    fn scaler_check_valid_gradients() {
        assert!(LossScaler::check_gradients(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn scaler_check_nan_gradient() {
        assert!(!LossScaler::check_gradients(&[1.0, f32::NAN, 3.0]));
    }

    #[test]
    fn scaler_check_inf_gradient() {
        assert!(!LossScaler::check_gradients(&[f32::INFINITY]));
    }

    #[test]
    fn scaler_growth_after_interval() {
        let config = MixedPrecisionConfig {
            loss_scale: 100.0,
            scale_growth_interval: 3,
            scale_growth_factor: 2.0,
            ..MixedPrecisionConfig::default()
        };
        let mut s = LossScaler::new(config);
        s.update(true);
        s.update(true);
        assert!((s.scale() - 100.0).abs() < 1e-4); // 2回 → まだ増えない
        s.update(true);
        assert!((s.scale() - 200.0).abs() < 1e-4); // 3回 → 2倍
    }

    #[test]
    fn scaler_backoff_on_nan() {
        let config = MixedPrecisionConfig {
            loss_scale: 100.0,
            scale_backoff_factor: 0.5,
            ..MixedPrecisionConfig::default()
        };
        let mut s = LossScaler::new(config);
        let valid = s.update(false);
        assert!(!valid);
        assert!((s.scale() - 50.0).abs() < 1e-4);
    }

    #[test]
    fn scaler_floor_at_one() {
        let config = MixedPrecisionConfig {
            loss_scale: 1.0,
            scale_backoff_factor: 0.1,
            ..MixedPrecisionConfig::default()
        };
        let mut s = LossScaler::new(config);
        s.update(false);
        assert!((s.scale() - 1.0).abs() < 1e-6); // floor at 1.0
    }

    #[test]
    fn scaler_no_dynamic_scaling() {
        let config = MixedPrecisionConfig {
            loss_scale: 100.0,
            dynamic_loss_scaling: false,
            ..MixedPrecisionConfig::default()
        };
        let mut s = LossScaler::new(config);
        for _ in 0..1000 {
            s.update(true);
        }
        assert!((s.scale() - 100.0).abs() < 1e-6); // 変わらない
    }

    #[test]
    fn scaler_clone() {
        let s = LossScaler::new(MixedPrecisionConfig::default());
        let s2 = s.clone();
        assert!((s2.scale() - s.scale()).abs() < 1e-6);
    }

    #[test]
    fn scaler_debug() {
        let s = LossScaler::new(MixedPrecisionConfig::default());
        let d = format!("{s:?}");
        assert!(d.contains("LossScaler"));
    }
}
