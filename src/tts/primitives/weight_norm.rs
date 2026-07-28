//! WeightNorm — weight reparametrization `w = g * v / ||v||` (Salimans & Kingma 2016)。
//!
//! Vocos ConvNeXt / HiFi-GAN / WaveNet 系の学習安定化に使用される weight reparametrization。
//! 直方向 `v` (同 shape 学習パラメータ) と大きさ `g` (per-output-channel 学習パラメータ) を
//! 保持し、forward 前に `w = g * v / ||v||` を computed weight として出力する。
//!
//! # 数式
//!
//! ```text
//! ||v||_i = sqrt(Σ_j v[i, j]²)     // 各 output channel の L2 norm
//! w[i, j] = g[i] * v[i, j] / ||v||_i
//! ```
//!
//! # backward
//!
//! grad_w を受け取り、grad_v と grad_g を返す:
//!
//! ```text
//! grad_g[i] = Σ_j grad_w[i, j] * v[i, j] / ||v||_i
//! grad_v[i, j] = (g[i] / ||v||_i) * (grad_w[i, j] - grad_g[i] * v[i, j] / ||v||_i)
//! ```
//!
//! # レイアウト
//!
//! - `v`: `[out_channels, dim_per_channel]` flatten
//! - `g`: `[out_channels]` flatten
//! - `w`: `v` と同 shape (compute_weight で生成)
//!
//! # 使用例
//!
//! ```rust
//! # #[cfg(feature = "tts")] {
//! use alice_train::tts::WeightNorm;
//!
//! // 3 output channels × 4 dim per channel
//! let v: Vec<f32> = (0..12).map(|i| (i as f32 + 1.0) * 0.1).collect();
//! let g: Vec<f32> = vec![1.0, 2.0, 3.0];
//! let wn = WeightNorm::new(3, 4, v, g).expect("valid");
//! let w = wn.compute_weight();
//! assert_eq!(w.len(), 12);
//! # }
//! ```

use serde::{Deserialize, Serialize};

/// WeightNorm config。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WeightNormConfig {
    /// output channel 数 (v の第 1 次元)。
    pub out_channels: usize,
    /// 各 output channel あたりの次元数 (v の第 2 次元)。
    /// - Conv1D: `in_channels / groups * kernel_size`
    /// - Linear: `in_features`
    pub dim_per_channel: usize,
    /// 数値安定性のための norm floor (通常 1e-8)。
    pub eps: f32,
}

impl WeightNormConfig {
    /// 最も一般的な config (eps=1e-8)。
    #[must_use]
    pub fn new(out_channels: usize, dim_per_channel: usize) -> Self {
        Self {
            out_channels,
            dim_per_channel,
            eps: 1e-8,
        }
    }

    /// eps を明示指定。
    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// config validity 検証。
    ///
    /// # Errors
    ///
    /// - `out_channels == 0` / `dim_per_channel == 0`
    /// - `eps < 0`
    pub fn validate(&self) -> Result<(), WeightNormError> {
        if self.out_channels == 0 {
            return Err(WeightNormError::InvalidConfig {
                reason: "out_channels must be > 0".to_string(),
            });
        }
        if self.dim_per_channel == 0 {
            return Err(WeightNormError::InvalidConfig {
                reason: "dim_per_channel must be > 0".to_string(),
            });
        }
        if self.eps < 0.0 {
            return Err(WeightNormError::InvalidConfig {
                reason: format!("eps must be >= 0, got {}", self.eps),
            });
        }
        Ok(())
    }
}

/// WeightNorm reparametrization layer (v + g を保持)。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightNorm {
    config: WeightNormConfig,
    v: Vec<f32>,
    g: Vec<f32>,
}

impl WeightNorm {
    /// 与えた v (direction) / g (magnitude) で構築する。
    ///
    /// # 引数
    ///
    /// - `out_channels`: `[out, dim]` の out 次元 (config.out_channels と一致必須)
    /// - `dim_per_channel`: 同 dim 次元 (config.dim_per_channel と一致必須)
    /// - `v`: `[out_channels, dim_per_channel]` flatten
    /// - `g`: `[out_channels]` flatten
    ///
    /// # Errors
    ///
    /// - config validation
    /// - v/g の shape 不整合
    pub fn new(
        out_channels: usize,
        dim_per_channel: usize,
        v: Vec<f32>,
        g: Vec<f32>,
    ) -> Result<Self, WeightNormError> {
        let cfg = WeightNormConfig::new(out_channels, dim_per_channel);
        Self::from_config(cfg, v, g)
    }

    /// config を明示指定して構築する。
    ///
    /// # Errors
    ///
    /// - config validation
    /// - v/g shape 不整合
    pub fn from_config(
        config: WeightNormConfig,
        v: Vec<f32>,
        g: Vec<f32>,
    ) -> Result<Self, WeightNormError> {
        config.validate()?;
        let expected_v = config.out_channels * config.dim_per_channel;
        if v.len() != expected_v {
            return Err(WeightNormError::ShapeMismatch {
                field: "v",
                expected: expected_v,
                actual: v.len(),
            });
        }
        if g.len() != config.out_channels {
            return Err(WeightNormError::ShapeMismatch {
                field: "g",
                expected: config.out_channels,
                actual: g.len(),
            });
        }
        Ok(Self { config, v, g })
    }

    /// unit v (norm = 1) + g = 1 で初期化する (PyTorch weight_norm 初期状態相当)。
    ///
    /// v は最初の要素を 1.0、他を 0.0 とすることで unit norm を実現。
    ///
    /// # Errors
    ///
    /// config validation。
    pub fn default_init(config: WeightNormConfig) -> Result<Self, WeightNormError> {
        config.validate()?;
        let mut v = vec![0.0_f32; config.out_channels * config.dim_per_channel];
        for i in 0..config.out_channels {
            v[i * config.dim_per_channel] = 1.0;
        }
        Ok(Self {
            config,
            v,
            g: vec![1.0; config.out_channels],
        })
    }

    /// config への参照。
    #[must_use]
    pub fn config(&self) -> &WeightNormConfig {
        &self.config
    }

    /// v への参照。
    #[must_use]
    pub fn v(&self) -> &[f32] {
        &self.v
    }

    /// v への可変参照 (optimizer 更新用)。
    pub fn v_mut(&mut self) -> &mut [f32] {
        &mut self.v
    }

    /// g への参照。
    #[must_use]
    pub fn g(&self) -> &[f32] {
        &self.g
    }

    /// g への可変参照 (optimizer 更新用)。
    pub fn g_mut(&mut self) -> &mut [f32] {
        &mut self.g
    }

    /// `w = g * v / ||v||` を計算して返す (v と同 shape)。
    #[must_use]
    pub fn compute_weight(&self) -> Vec<f32> {
        let cfg = self.config;
        let mut w = vec![0.0_f32; self.v.len()];
        for i in 0..cfg.out_channels {
            let norm = self.channel_norm(i);
            let scale = self.g[i] / norm;
            for j in 0..cfg.dim_per_channel {
                let idx = i * cfg.dim_per_channel + j;
                w[idx] = self.v[idx] * scale;
            }
        }
        w
    }

    /// 各 output channel の `||v||` を返す (`[out_channels]` flatten)。
    #[must_use]
    pub fn channel_norms(&self) -> Vec<f32> {
        (0..self.config.out_channels)
            .map(|i| self.channel_norm(i))
            .collect()
    }

    fn channel_norm(&self, channel: usize) -> f32 {
        let cfg = self.config;
        let start = channel * cfg.dim_per_channel;
        let end = start + cfg.dim_per_channel;
        let mut sum_sq = 0.0_f64;
        for &x in &self.v[start..end] {
            sum_sq += f64::from(x * x);
        }
        (sum_sq.sqrt() as f32).max(cfg.eps)
    }

    /// backward: grad_w を受け取り、`(grad_v, grad_g)` を返す。
    ///
    /// # 式
    ///
    /// ```text
    /// grad_g[i]   = Σ_j grad_w[i, j] * v[i, j] / ||v||_i
    /// grad_v[i,j] = (g[i] / ||v||_i) * (grad_w[i, j] - grad_g[i] * v[i, j] / ||v||_i)
    /// ```
    ///
    /// # Errors
    ///
    /// grad_w の shape 不整合。
    pub fn backward(&self, grad_w: &[f32]) -> Result<(Vec<f32>, Vec<f32>), WeightNormError> {
        let cfg = self.config;
        let expected = cfg.out_channels * cfg.dim_per_channel;
        if grad_w.len() != expected {
            return Err(WeightNormError::ShapeMismatch {
                field: "grad_w",
                expected,
                actual: grad_w.len(),
            });
        }

        let mut grad_v = vec![0.0_f32; self.v.len()];
        let mut grad_g = vec![0.0_f32; self.g.len()];

        for (i, grad_g_slot) in grad_g.iter_mut().enumerate() {
            let norm = self.channel_norm(i);
            let scale_gv = self.g[i] / norm;

            // grad_g[i] = Σ_j grad_w[i, j] * v[i, j] / norm
            let mut gg = 0.0_f64;
            for j in 0..cfg.dim_per_channel {
                let idx = i * cfg.dim_per_channel + j;
                gg += f64::from(grad_w[idx] * self.v[idx]);
            }
            gg /= f64::from(norm);
            *grad_g_slot = gg as f32;

            // grad_v[i, j] = scale_gv * (grad_w[i, j] - grad_g[i] * v[i, j] / norm)
            let gg_over_norm = *grad_g_slot / norm;
            for j in 0..cfg.dim_per_channel {
                let idx = i * cfg.dim_per_channel + j;
                grad_v[idx] = scale_gv * (grad_w[idx] - gg_over_norm * self.v[idx]);
            }
        }

        Ok((grad_v, grad_g))
    }
}

/// WeightNorm 操作で発生し得るエラー。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightNormError {
    /// config が不正 (channel/dim = 0、eps < 0)。
    InvalidConfig {
        /// 具体的な理由。
        reason: String,
    },
    /// v / g / grad_w の shape 不整合。
    ShapeMismatch {
        /// 対象 field 名。
        field: &'static str,
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
}

impl std::fmt::Display for WeightNormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig { reason } => write!(f, "invalid WeightNorm config: {reason}"),
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch on '{field}': expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for WeightNormError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_weight_with_unit_v_gives_g() {
        // v is unit vector along dim 0 → norm=1 → w = g * v → w[0]=g, others=0
        let cfg = WeightNormConfig::new(2, 3);
        let wn = WeightNorm::default_init(cfg).unwrap();
        // g=1, v=[1,0,0, 1,0,0] → w = [1,0,0, 1,0,0]
        let w = wn.compute_weight();
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert!(w[1].abs() < 1e-6);
        assert!(w[2].abs() < 1e-6);
        assert!((w[3] - 1.0).abs() < 1e-6);
        assert!(w[4].abs() < 1e-6);
        assert!(w[5].abs() < 1e-6);
    }

    #[test]
    fn compute_weight_scales_by_g() {
        // v=[1,0,0] (unit), g=2.5 → w=[2.5, 0, 0]
        let cfg = WeightNormConfig::new(1, 3);
        let wn = WeightNorm::from_config(cfg, vec![1.0, 0.0, 0.0], vec![2.5]).unwrap();
        let w = wn.compute_weight();
        assert!((w[0] - 2.5).abs() < 1e-6);
        assert!(w[1].abs() < 1e-6);
        assert!(w[2].abs() < 1e-6);
    }

    #[test]
    fn compute_weight_normalizes_by_norm() {
        // v=[3,4] (norm=5), g=10 → w = 10 * [3,4] / 5 = [6, 8]
        let cfg = WeightNormConfig::new(1, 2);
        let wn = WeightNorm::from_config(cfg, vec![3.0, 4.0], vec![10.0]).unwrap();
        let w = wn.compute_weight();
        assert!((w[0] - 6.0).abs() < 1e-5);
        assert!((w[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn channel_norms_computed_correctly() {
        // ch 0: v=[3,4] norm=5、ch 1: v=[1,0] norm=1
        let cfg = WeightNormConfig::new(2, 2);
        let wn = WeightNorm::from_config(cfg, vec![3.0, 4.0, 1.0, 0.0], vec![1.0, 1.0]).unwrap();
        let norms = wn.channel_norms();
        assert!((norms[0] - 5.0).abs() < 1e-5);
        assert!((norms[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn backward_grad_matches_finite_difference() {
        // Numerical gradient check
        let cfg = WeightNormConfig::new(3, 4);
        let v: Vec<f32> = (0..12).map(|i| ((i as f32) * 0.13 + 0.5).sin()).collect();
        let g: Vec<f32> = vec![0.7, 1.2, 0.5];
        let wn = WeightNorm::from_config(cfg, v.clone(), g.clone()).unwrap();
        // 適当な grad_w
        let grad_w: Vec<f32> = (0..12).map(|i| (i as f32 * 0.09).cos() * 0.3).collect();

        let (grad_v_ana, grad_g_ana) = wn.backward(&grad_w).unwrap();

        // Loss = Σ grad_w · w、ここでは Σ w * grad_w
        let h = 1e-3_f32;
        // Check grad_v
        for i in 0..v.len() {
            let mut vp = v.clone();
            vp[i] += h;
            let wp = WeightNorm::from_config(cfg, vp, g.clone())
                .unwrap()
                .compute_weight();
            let mut vm = v.clone();
            vm[i] -= h;
            let wm = WeightNorm::from_config(cfg, vm, g.clone())
                .unwrap()
                .compute_weight();
            let loss_p: f32 = wp.iter().zip(&grad_w).map(|(w, g)| w * g).sum();
            let loss_m: f32 = wm.iter().zip(&grad_w).map(|(w, g)| w * g).sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let ana = grad_v_ana[i];
            let diff = (ana - num).abs();
            let scale = ana.abs().max(num.abs()).max(1e-4);
            assert!(
                diff / scale < 3e-2,
                "grad_v[{i}] analytical={ana}, numerical={num}, rel_err={}",
                diff / scale
            );
        }
        // Check grad_g
        for i in 0..g.len() {
            let mut gp = g.clone();
            gp[i] += h;
            let wp = WeightNorm::from_config(cfg, v.clone(), gp)
                .unwrap()
                .compute_weight();
            let mut gm = g.clone();
            gm[i] -= h;
            let wm = WeightNorm::from_config(cfg, v.clone(), gm)
                .unwrap()
                .compute_weight();
            let loss_p: f32 = wp.iter().zip(&grad_w).map(|(w, g)| w * g).sum();
            let loss_m: f32 = wm.iter().zip(&grad_w).map(|(w, g)| w * g).sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let ana = grad_g_ana[i];
            let diff = (ana - num).abs();
            let scale = ana.abs().max(num.abs()).max(1e-4);
            assert!(
                diff / scale < 3e-2,
                "grad_g[{i}] analytical={ana}, numerical={num}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn zero_v_does_not_panic_thanks_to_eps() {
        // v = 0 だと ||v||=0 で 0 division だが、eps 保護で NaN 出さず
        let cfg = WeightNormConfig::new(1, 3).with_eps(1e-6);
        let wn = WeightNorm::from_config(cfg, vec![0.0, 0.0, 0.0], vec![1.0]).unwrap();
        let w = wn.compute_weight();
        for &v in &w {
            assert!(v.is_finite(), "must not be NaN/Inf");
            assert!(v.abs() < 1e-3, "w should be ~0, got {v}");
        }
    }

    #[test]
    fn v_g_mut_allow_updates() {
        let cfg = WeightNormConfig::new(1, 2);
        let mut wn = WeightNorm::from_config(cfg, vec![3.0, 4.0], vec![1.0]).unwrap();
        wn.v_mut()[0] = 1.0;
        wn.v_mut()[1] = 0.0;
        wn.g_mut()[0] = 5.0;
        // new: v=[1,0] norm=1, g=5 → w=[5, 0]
        let w = wn.compute_weight();
        assert!((w[0] - 5.0).abs() < 1e-5);
        assert!(w[1].abs() < 1e-5);
    }

    #[test]
    fn invalid_config_returns_error() {
        let cfg = WeightNormConfig::new(0, 4);
        assert!(cfg.validate().is_err());

        let cfg = WeightNormConfig::new(4, 0);
        assert!(cfg.validate().is_err());

        let cfg = WeightNormConfig::new(4, 4).with_eps(-1.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let cfg = WeightNormConfig::new(2, 3);
        let err =
            WeightNorm::from_config(cfg, vec![1.0; 5], vec![1.0; 2]).expect_err("v len 5 != 6");
        assert!(matches!(
            err,
            WeightNormError::ShapeMismatch { field: "v", .. }
        ));

        let err =
            WeightNorm::from_config(cfg, vec![1.0; 6], vec![1.0; 3]).expect_err("g len 3 != 2");
        assert!(matches!(
            err,
            WeightNormError::ShapeMismatch { field: "g", .. }
        ));
    }

    #[test]
    fn backward_shape_mismatch_returns_error() {
        let cfg = WeightNormConfig::new(2, 3);
        let wn = WeightNorm::default_init(cfg).unwrap();
        let err = wn.backward(&[0.0; 5]).expect_err("grad_w len 5 != 6");
        assert!(matches!(
            err,
            WeightNormError::ShapeMismatch {
                field: "grad_w",
                ..
            }
        ));
    }

    #[test]
    fn error_display_and_error_trait() {
        let e = WeightNormError::InvalidConfig {
            reason: "test".to_string(),
        };
        let s = format!("{e}");
        assert!(s.contains("invalid WeightNorm"));
        let boxed: Box<dyn std::error::Error> = Box::new(e);
        assert!(boxed.to_string().contains("test"));
    }

    #[test]
    fn config_builder_pattern() {
        let cfg = WeightNormConfig::new(8, 16).with_eps(1e-6);
        cfg.validate().expect("valid");
        assert_eq!(cfg.out_channels, 8);
        assert_eq!(cfg.dim_per_channel, 16);
        assert!((cfg.eps - 1e-6).abs() < 1e-9);
    }

    #[test]
    fn default_init_gives_g_equals_1_and_unit_v() {
        let cfg = WeightNormConfig::new(3, 5);
        let wn = WeightNorm::default_init(cfg).unwrap();
        assert_eq!(wn.g(), &[1.0; 3]);
        // v[0]=1, others=0 で unit norm
        let norms = wn.channel_norms();
        for n in norms {
            assert!((n - 1.0).abs() < 1e-6);
        }
    }
}
