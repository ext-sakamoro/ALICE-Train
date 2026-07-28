//! Linear (fully-connected) layer — forward + 手書き backward。
//!
//! Transformer FFN / FastSpeech2 predictor output / projection 全般で使用される基本 primitive。
//! MHA 内部でも同じ計算を private helper として使用しているが、外部からも使えるよう公開 primitive
//! として整理した。
//!
//! # 演算定義
//!
//! - Input: `[batch, in_features]` (`batch` は 2D outer 次元、higher-dim は flatten して渡す)
//! - Weight: `[out_features, in_features]` (row-major)
//! - Bias: `[out_features]` (optional)
//! - Output: `[batch, out_features]`
//!
//! `y[b, o] = Σ_i x[b, i] * w[o, i] + b[o]`
//!
//! # backward
//!
//! - `grad_input[b, i] = Σ_o grad_output[b, o] * w[o, i]`
//! - `grad_weight[o, i] = Σ_b grad_output[b, o] * x[b, i]`
//! - `grad_bias[o] = Σ_b grad_output[b, o]`

use serde::{Deserialize, Serialize};

/// Linear layer config。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LinearConfig {
    /// 入力次元。
    pub in_features: usize,
    /// 出力次元。
    pub out_features: usize,
    /// bias を持つか。
    pub bias: bool,
}

impl LinearConfig {
    /// 最も一般的な config (bias=true)。
    #[must_use]
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
        }
    }

    /// bias 有無を明示指定。
    #[must_use]
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// config validity 検証。
    ///
    /// # Errors
    ///
    /// - `in_features == 0` / `out_features == 0`
    pub fn validate(&self) -> Result<(), LinearError> {
        if self.in_features == 0 {
            return Err(LinearError::InvalidConfig {
                reason: "in_features must be > 0".to_string(),
            });
        }
        if self.out_features == 0 {
            return Err(LinearError::InvalidConfig {
                reason: "out_features must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

/// Linear layer (weight + bias を保持)。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Linear {
    config: LinearConfig,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Linear {
    /// weight / bias を指定して構築する。
    ///
    /// # Errors
    ///
    /// - config validation
    /// - weight/bias shape 不整合
    pub fn new(
        config: LinearConfig,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, LinearError> {
        config.validate()?;
        let expected_w = config.out_features * config.in_features;
        if weight.len() != expected_w {
            return Err(LinearError::ShapeMismatch {
                field: "weight",
                expected: expected_w,
                actual: weight.len(),
            });
        }
        let expected_b = if config.bias { config.out_features } else { 0 };
        if bias.len() != expected_b {
            return Err(LinearError::ShapeMismatch {
                field: "bias",
                expected: expected_b,
                actual: bias.len(),
            });
        }
        Ok(Self {
            config,
            weight,
            bias,
        })
    }

    /// zero 初期化で構築 (テスト用)。
    ///
    /// # Errors
    ///
    /// config validation。
    pub fn zeros(config: LinearConfig) -> Result<Self, LinearError> {
        config.validate()?;
        let n_w = config.out_features * config.in_features;
        let n_b = if config.bias { config.out_features } else { 0 };
        Ok(Self {
            config,
            weight: vec![0.0; n_w],
            bias: vec![0.0; n_b],
        })
    }

    /// config への参照。
    #[must_use]
    pub fn config(&self) -> &LinearConfig {
        &self.config
    }

    /// weight `[out_features, in_features]` flatten への参照。
    #[must_use]
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// weight への可変参照 (optimizer 更新用)。
    pub fn weight_mut(&mut self) -> &mut [f32] {
        &mut self.weight
    }

    /// bias `[out_features]` への参照 (bias=false なら空)。
    #[must_use]
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// bias への可変参照 (optimizer 更新用)。
    pub fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias
    }

    /// Xavier uniform init: `w ~ U(-a, +a)`, `a = sqrt(6 / (in_features + out_features))`。
    ///
    /// bias は 0 で初期化 (PyTorch default)。
    pub fn init_xavier<R: rand::Rng>(&mut self, rng: &mut R) {
        let fan_in = self.config.in_features;
        let fan_out = self.config.out_features;
        let a = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
        for w in &mut self.weight {
            *w = rng.gen_range(-a..=a);
        }
        for b in &mut self.bias {
            *b = 0.0;
        }
    }

    /// forward pass。
    ///
    /// # 引数
    ///
    /// - `input`: `[batch, in_features]` flatten
    /// - `batch`: 2D outer 次元 (higher-dim tensor は呼び出し側で flatten 済)
    ///
    /// # Errors
    ///
    /// input shape mismatch。
    pub fn forward(&self, input: &[f32], batch: usize) -> Result<Vec<f32>, LinearError> {
        let cfg = self.config;
        if input.len() != batch * cfg.in_features {
            return Err(LinearError::InputShapeMismatch {
                expected: batch * cfg.in_features,
                actual: input.len(),
            });
        }
        let mut output = vec![0.0_f32; batch * cfg.out_features];
        // Phase T.4b: BLAS matmul (macOS Accelerate / Linux OpenBLAS / tiled fallback)
        // output[batch, out] = input[batch, in] × weight[out, in]^T
        crate::blas::blas_matmul_bt(
            input,
            &self.weight,
            &mut output,
            batch,
            cfg.out_features,
            cfg.in_features,
        );
        // bias add
        if cfg.bias {
            for b in 0..batch {
                for o in 0..cfg.out_features {
                    output[b * cfg.out_features + o] += self.bias[o];
                }
            }
        }
        Ok(output)
    }

    /// backward pass — grad_input / grad_weight / grad_bias を計算して返す。
    ///
    /// # Errors
    ///
    /// input / grad_output shape mismatch。
    pub fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        batch: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), LinearError> {
        let cfg = self.config;
        if input.len() != batch * cfg.in_features {
            return Err(LinearError::InputShapeMismatch {
                expected: batch * cfg.in_features,
                actual: input.len(),
            });
        }
        if grad_output.len() != batch * cfg.out_features {
            return Err(LinearError::GradOutputShapeMismatch {
                expected: batch * cfg.out_features,
                actual: grad_output.len(),
            });
        }

        let mut grad_input = vec![0.0_f32; input.len()];
        let mut grad_weight = vec![0.0_f32; self.weight.len()];
        let mut grad_bias = vec![0.0_f32; cfg.out_features];

        // Phase T.4b: BLAS matmul
        // grad_input[batch, in] = grad_output[batch, out] × weight[out, in]
        crate::blas::blas_matmul_nn(
            grad_output,
            &self.weight,
            &mut grad_input,
            batch,
            cfg.in_features,
            cfg.out_features,
        );
        // grad_weight[out, in] = grad_output[batch, out]^T × input[batch, in]
        crate::blas::blas_matmul_tn(
            grad_output,
            input,
            &mut grad_weight,
            cfg.out_features,
            cfg.in_features,
            batch,
        );
        // grad_bias[out] = sum over batch of grad_output[batch, out]
        if cfg.bias {
            for b in 0..batch {
                for o in 0..cfg.out_features {
                    grad_bias[o] += grad_output[b * cfg.out_features + o];
                }
            }
        }

        Ok((grad_input, grad_weight, grad_bias))
    }
}

/// Linear 操作で発生し得るエラー。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinearError {
    /// config が不正 (in_features=0 / out_features=0)。
    InvalidConfig {
        /// 具体的な理由。
        reason: String,
    },
    /// weight/bias shape 不整合。
    ShapeMismatch {
        /// 対象 field 名。
        field: &'static str,
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
    /// input len が `batch * in_features` と不一致。
    InputShapeMismatch {
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
    /// grad_output len が `batch * out_features` と不一致。
    GradOutputShapeMismatch {
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
}

impl std::fmt::Display for LinearError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig { reason } => write!(f, "invalid Linear config: {reason}"),
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch on '{field}': expected {expected}, got {actual}"
            ),
            Self::InputShapeMismatch { expected, actual } => {
                write!(f, "input shape mismatch: expected {expected}, got {actual}")
            }
            Self::GradOutputShapeMismatch { expected, actual } => write!(
                f,
                "grad_output shape mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for LinearError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_linear_reproduces_input() {
        // 2x2 identity + bias 0
        let cfg = LinearConfig::new(2, 2);
        let ln = Linear::new(cfg, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0]).unwrap();
        let input = vec![3.0_f32, 4.0];
        let out = ln.forward(&input, 1).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn bias_only_gives_bias() {
        let cfg = LinearConfig::new(2, 3);
        let ln = Linear::new(cfg, vec![0.0; 6], vec![1.0, 2.0, 3.0]).unwrap();
        let out = ln.forward(&[10.0, 20.0], 1).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn matmul_matches_expected() {
        // w = [[1, 2], [3, 4]], b = [10, 20]
        // input = [5, 6] → out = [1*5 + 2*6 + 10, 3*5 + 4*6 + 20] = [27, 59]
        let cfg = LinearConfig::new(2, 2);
        let ln = Linear::new(cfg, vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0]).unwrap();
        let out = ln.forward(&[5.0, 6.0], 1).unwrap();
        assert!((out[0] - 27.0).abs() < 1e-5);
        assert!((out[1] - 59.0).abs() < 1e-5);
    }

    #[test]
    fn multi_batch_processes_independently() {
        let cfg = LinearConfig::new(2, 1);
        let ln = Linear::new(cfg, vec![1.0, 1.0], vec![0.0]).unwrap();
        let input = vec![
            1.0_f32, 2.0, // batch 0 → 1+2 = 3
            10.0, 20.0, // batch 1 → 10+20 = 30
        ];
        let out = ln.forward(&input, 2).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn backward_gradient_matches_finite_difference() {
        let cfg = LinearConfig::new(3, 2);
        let weight: Vec<f32> = (0..6).map(|i| (i as f32 * 0.2).sin()).collect();
        let bias: Vec<f32> = vec![0.1, -0.2];
        let ln = Linear::new(cfg, weight.clone(), bias.clone()).unwrap();
        let batch = 2;
        let input: Vec<f32> = (0..batch * 3).map(|i| (i as f32 * 0.3).cos()).collect();
        let grad_output: Vec<f32> = (0..batch * 2).map(|i| (i as f32 * 0.15).sin()).collect();

        let (grad_in, grad_w, grad_b) = ln.backward(&input, &grad_output, batch).unwrap();

        let h = 1e-3_f32;
        // grad_input
        for i in 0..input.len() {
            let mut ip = input.clone();
            ip[i] += h;
            let out_p = ln.forward(&ip, batch).unwrap();
            let mut im = input.clone();
            im[i] -= h;
            let out_m = ln.forward(&im, batch).unwrap();
            let loss_p: f32 = out_p.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let loss_m: f32 = out_m.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let ana = grad_in[i];
            let diff = (ana - num).abs();
            let scale = ana.abs().max(num.abs()).max(1e-4);
            assert!(
                diff / scale < 1e-2,
                "grad_input[{i}] analytical={ana}, numerical={num}"
            );
        }
        // grad_weight
        for i in 0..weight.len() {
            let mut wp = weight.clone();
            wp[i] += h;
            let lnp = Linear::new(cfg, wp, bias.clone()).unwrap();
            let out_p = lnp.forward(&input, batch).unwrap();
            let mut wm = weight.clone();
            wm[i] -= h;
            let lnm = Linear::new(cfg, wm, bias.clone()).unwrap();
            let out_m = lnm.forward(&input, batch).unwrap();
            let loss_p: f32 = out_p.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let loss_m: f32 = out_m.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let ana = grad_w[i];
            let diff = (ana - num).abs();
            let scale = ana.abs().max(num.abs()).max(1e-4);
            assert!(
                diff / scale < 1e-2,
                "grad_weight[{i}] analytical={ana}, numerical={num}"
            );
        }
        // grad_bias = Σ grad_output over batch
        for i in 0..bias.len() {
            let expected: f32 = (0..batch).map(|b| grad_output[b * 2 + i]).sum();
            assert!(
                (grad_b[i] - expected).abs() < 1e-5,
                "grad_bias[{i}] = {} expected {}",
                grad_b[i],
                expected
            );
        }
    }

    #[test]
    fn invalid_config_returns_error() {
        let cfg = LinearConfig::new(0, 4);
        assert!(cfg.validate().is_err());
        let cfg = LinearConfig::new(4, 0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let cfg = LinearConfig::new(2, 3);
        // weight expected 6, give 5
        let err = Linear::new(cfg, vec![0.0; 5], vec![0.0; 3]).expect_err("weight shape");
        assert!(matches!(err, LinearError::ShapeMismatch { .. }));
    }

    #[test]
    fn input_shape_mismatch_returns_error() {
        let cfg = LinearConfig::new(2, 3);
        let ln = Linear::zeros(cfg).unwrap();
        let err = ln
            .forward(&[0.0; 5], 1)
            .expect_err("input len 5 vs expected 2");
        assert!(matches!(err, LinearError::InputShapeMismatch { .. }));
    }

    #[test]
    fn weight_mut_bias_mut_allow_updates() {
        let cfg = LinearConfig::new(2, 1);
        let mut ln = Linear::new(cfg, vec![0.0, 0.0], vec![0.0]).unwrap();
        ln.weight_mut()[0] = 3.0;
        ln.weight_mut()[1] = 4.0;
        ln.bias_mut()[0] = 5.0;
        let out = ln.forward(&[1.0, 1.0], 1).unwrap();
        assert!((out[0] - (3.0 + 4.0 + 5.0)).abs() < 1e-6);
    }

    #[test]
    fn error_display_and_error_trait() {
        let e = LinearError::InvalidConfig {
            reason: "test".to_string(),
        };
        let s = format!("{e}");
        assert!(s.contains("invalid Linear config"));
        let boxed: Box<dyn std::error::Error> = Box::new(e);
        assert!(boxed.to_string().contains("test"));
    }
}
