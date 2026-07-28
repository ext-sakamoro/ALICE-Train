//! LayerNorm — forward + 手書き backward (最終次元 normalization、elementwise affine)。
//!
//! Transformer FFT block / Vocos ConvNeXt / FastSpeech2 全 layer で使用される基本 primitive。
//! 入力 `[batch, ..., normalized_shape]` の **最終次元** を normalization し、gamma/beta で
//! affine transform する PyTorch `nn.LayerNorm` と数値互換。
//!
//! # 数式
//!
//! ```text
//! μ = mean(x, dim=-1)              // 各 sample の最終次元平均
//! σ² = mean((x - μ)², dim=-1)      // biased variance
//! x_hat = (x - μ) / sqrt(σ² + ε)
//! y = γ * x_hat + β                // elementwise_affine=true のみ
//! ```
//!
//! # backward (canonical form)
//!
//! ```text
//! grad_gamma[i] = Σ_b grad_y[b, i] * x_hat[b, i]
//! grad_beta[i]  = Σ_b grad_y[b, i]
//! effective_grad = γ * grad_y
//! sum_g          = Σ_i effective_grad
//! sum_g_xh       = Σ_i effective_grad * x_hat
//! grad_x[i] = (1 / (N * sqrt(σ² + ε))) * (N * effective_grad[i] - sum_g - x_hat[i] * sum_g_xh)
//! ```
//!
//! ここで N = `normalized_shape`。
//!
//! # 実装 note
//!
//! - `[batch, normalized_shape]` を flatten 済 `Vec<f32>` として扱う (2D layout)
//! - 高次元入力は呼び出し側で `[..., normalized_shape]` → `[flattened_batch, normalized_shape]` に reshape する責任
//! - f64 accumulator で数値安定性確保 (Welford は 1-pass だが実装複雑度優先)

use serde::{Deserialize, Serialize};

/// LayerNorm config。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LayerNormConfig {
    /// 最終次元サイズ (通常 hidden_dim)。
    pub normalized_shape: usize,
    /// 数値安定性のための ε (通常 1e-5)。
    pub eps: f32,
    /// γ / β を持つか (true = affine、false = normalize のみ)。
    pub elementwise_affine: bool,
}

impl LayerNormConfig {
    /// 最も一般的な config (eps=1e-5, elementwise_affine=true)。
    #[must_use]
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            eps: 1e-5,
            elementwise_affine: true,
        }
    }

    /// eps を明示指定。
    #[must_use]
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// affine の有無を明示指定。
    #[must_use]
    pub fn with_affine(mut self, elementwise_affine: bool) -> Self {
        self.elementwise_affine = elementwise_affine;
        self
    }

    /// config の validity 検証。
    ///
    /// # Errors
    ///
    /// - `normalized_shape == 0`
    /// - `eps < 0`
    pub fn validate(&self) -> Result<(), LayerNormError> {
        if self.normalized_shape == 0 {
            return Err(LayerNormError::InvalidConfig {
                reason: "normalized_shape must be > 0".to_string(),
            });
        }
        if self.eps < 0.0 {
            return Err(LayerNormError::InvalidConfig {
                reason: format!("eps must be >= 0, got {}", self.eps),
            });
        }
        Ok(())
    }
}

/// LayerNorm レイヤー (gamma + beta を保持)。
///
/// `elementwise_affine=false` の場合、gamma/beta は空 `Vec<f32>`。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerNorm {
    config: LayerNormConfig,
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl LayerNorm {
    /// 与えた gamma / beta で新しい layer を構築する。
    ///
    /// `elementwise_affine=true` なら `gamma.len() == beta.len() == normalized_shape`、
    /// `elementwise_affine=false` なら両者 0 長を要求。
    ///
    /// # Errors
    ///
    /// - config validation
    /// - gamma/beta の shape が不整合
    pub fn new(
        config: LayerNormConfig,
        gamma: Vec<f32>,
        beta: Vec<f32>,
    ) -> Result<Self, LayerNormError> {
        config.validate()?;
        let expected = if config.elementwise_affine {
            config.normalized_shape
        } else {
            0
        };
        if gamma.len() != expected {
            return Err(LayerNormError::ShapeMismatch {
                field: "gamma",
                expected,
                actual: gamma.len(),
            });
        }
        if beta.len() != expected {
            return Err(LayerNormError::ShapeMismatch {
                field: "beta",
                expected,
                actual: beta.len(),
            });
        }
        Ok(Self {
            config,
            gamma,
            beta,
        })
    }

    /// `elementwise_affine=true` の場合 gamma=1, beta=0 初期化 (PyTorch default)。
    /// `elementwise_affine=false` なら空初期化。
    ///
    /// # Errors
    ///
    /// config validation。
    pub fn default_init(config: LayerNormConfig) -> Result<Self, LayerNormError> {
        config.validate()?;
        let (gamma, beta) = if config.elementwise_affine {
            (
                vec![1.0; config.normalized_shape],
                vec![0.0; config.normalized_shape],
            )
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Self {
            config,
            gamma,
            beta,
        })
    }

    /// config への参照。
    #[must_use]
    pub fn config(&self) -> &LayerNormConfig {
        &self.config
    }

    /// gamma への参照 (`elementwise_affine=false` なら空 slice)。
    #[must_use]
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    /// gamma への可変参照 (optimizer 更新用)。
    pub fn gamma_mut(&mut self) -> &mut [f32] {
        &mut self.gamma
    }

    /// beta への参照 (`elementwise_affine=false` なら空 slice)。
    #[must_use]
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }

    /// beta への可変参照 (optimizer 更新用)。
    pub fn beta_mut(&mut self) -> &mut [f32] {
        &mut self.beta
    }

    /// forward pass。
    ///
    /// # 引数
    ///
    /// - `input`: `[batch, normalized_shape]` flatten (`input.len() = batch * normalized_shape`)
    /// - `batch`: outer batch dim (row 数)
    ///
    /// # 戻り値
    ///
    /// `[batch, normalized_shape]` flatten `Vec<f32>` (input と同 shape)。
    ///
    /// # Errors
    ///
    /// input len が `batch * normalized_shape` と不一致。
    pub fn forward(&self, input: &[f32], batch: usize) -> Result<Vec<f32>, LayerNormError> {
        let n = self.config.normalized_shape;
        if input.len() != batch * n {
            return Err(LayerNormError::InputShapeMismatch {
                expected: batch * n,
                actual: input.len(),
            });
        }

        let mut output = vec![0.0_f32; input.len()];

        for b in 0..batch {
            let row = &input[b * n..(b + 1) * n];
            let (mean, inv_std) = row_mean_inv_std(row, self.config.eps);
            let out_row = &mut output[b * n..(b + 1) * n];
            if self.config.elementwise_affine {
                for i in 0..n {
                    let x_hat = (row[i] - mean) * inv_std;
                    out_row[i] = self.gamma[i] * x_hat + self.beta[i];
                }
            } else {
                for i in 0..n {
                    out_row[i] = (row[i] - mean) * inv_std;
                }
            }
        }

        Ok(output)
    }

    /// backward pass — grad_input / grad_gamma / grad_beta を計算して返す。
    ///
    /// # 引数
    ///
    /// - `input`: forward 時の入力 (`[batch, normalized_shape]` flatten)
    /// - `grad_output`: 出力勾配 (`[batch, normalized_shape]` flatten)
    /// - `batch`: outer batch dim
    ///
    /// # 戻り値
    ///
    /// `(grad_input, grad_gamma, grad_beta)`:
    /// - `grad_input`: input と同 shape
    /// - `grad_gamma`: `[normalized_shape]` (`elementwise_affine=false` でも len は `normalized_shape`、値は全 0)
    /// - `grad_beta`: 同上
    ///
    /// # Errors
    ///
    /// input / grad_output の shape 不一致。
    pub fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        batch: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), LayerNormError> {
        let n = self.config.normalized_shape;
        if input.len() != batch * n {
            return Err(LayerNormError::InputShapeMismatch {
                expected: batch * n,
                actual: input.len(),
            });
        }
        if grad_output.len() != batch * n {
            return Err(LayerNormError::GradOutputShapeMismatch {
                expected: batch * n,
                actual: grad_output.len(),
            });
        }

        let mut grad_input = vec![0.0_f32; input.len()];
        let mut grad_gamma = vec![0.0_f32; n];
        let mut grad_beta = vec![0.0_f32; n];

        let n_f = n as f32;

        for b in 0..batch {
            let row = &input[b * n..(b + 1) * n];
            let g_row = &grad_output[b * n..(b + 1) * n];
            let (mean, inv_std) = row_mean_inv_std(row, self.config.eps);

            // x_hat, effective_grad, sum_g, sum_g_xh を precompute (f64 accumulate)
            let mut x_hat = vec![0.0_f32; n];
            let mut eff_grad = vec![0.0_f32; n];
            let mut sum_g: f64 = 0.0;
            let mut sum_g_xh: f64 = 0.0;

            for i in 0..n {
                let xh = (row[i] - mean) * inv_std;
                x_hat[i] = xh;
                let gamma_i = if self.config.elementwise_affine {
                    self.gamma[i]
                } else {
                    1.0
                };
                let eg = gamma_i * g_row[i];
                eff_grad[i] = eg;
                sum_g += f64::from(eg);
                sum_g_xh += f64::from(eg * xh);

                if self.config.elementwise_affine {
                    grad_gamma[i] += g_row[i] * xh;
                    grad_beta[i] += g_row[i];
                }
            }

            let sum_g_f = sum_g as f32;
            let sum_g_xh_f = sum_g_xh as f32;
            let scale = inv_std / n_f;

            let out_row = &mut grad_input[b * n..(b + 1) * n];
            for i in 0..n {
                out_row[i] = scale * (n_f * eff_grad[i] - sum_g_f - x_hat[i] * sum_g_xh_f);
            }
        }

        Ok((grad_input, grad_gamma, grad_beta))
    }
}

/// row (長さ N) の mean と inv_std (`1 / sqrt(var + eps)`) を f64 accumulator で計算する。
fn row_mean_inv_std(row: &[f32], eps: f32) -> (f32, f32) {
    let n = row.len();
    let n_f = n as f64;
    let mut sum: f64 = 0.0;
    for &v in row {
        sum += f64::from(v);
    }
    let mean = sum / n_f;
    let mut sq_sum: f64 = 0.0;
    for &v in row {
        let d = f64::from(v) - mean;
        sq_sum += d * d;
    }
    let var = sq_sum / n_f;
    let inv_std = 1.0 / (var + f64::from(eps)).sqrt();
    (mean as f32, inv_std as f32)
}

/// LayerNorm 操作で発生し得るエラー。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerNormError {
    /// config が不正 (normalized_shape=0、eps<0)。
    InvalidConfig {
        /// 具体的な理由。
        reason: String,
    },
    /// gamma / beta の shape が不整合。
    ShapeMismatch {
        /// 対象 field 名 ("gamma" or "beta")。
        field: &'static str,
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
    /// input len が `batch * normalized_shape` と不一致。
    InputShapeMismatch {
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
    /// grad_output len が `batch * normalized_shape` と不一致。
    GradOutputShapeMismatch {
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
}

impl std::fmt::Display for LayerNormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig { reason } => write!(f, "invalid LayerNorm config: {reason}"),
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

impl std::error::Error for LayerNormError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_produces_zero_mean_unit_variance_without_affine() {
        // elementwise_affine=false でチェック (gamma=1, beta=0 と等価)
        let cfg = LayerNormConfig::new(4).with_affine(false);
        let ln = LayerNorm::default_init(cfg).unwrap();
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = ln.forward(&input, 1).unwrap();

        // mean = 2.5, var = 1.25, std ≈ 1.118
        // normalized ≈ [-1.342, -0.447, 0.447, 1.342]
        let sum: f32 = out.iter().sum();
        assert!(sum.abs() < 1e-4, "mean should be 0, got {sum}");
        let sum_sq: f32 = out.iter().map(|&v| v * v).sum();
        // var = sum_sq / N (mean=0), should be ~ 1 (eps 影響で 1 未満)
        assert!(
            (sum_sq / 4.0 - 1.0).abs() < 1e-3,
            "var should be ~1, got {}",
            sum_sq / 4.0
        );
    }

    #[test]
    fn forward_applies_gamma_beta_correctly() {
        // gamma=2, beta=1 で全 output = 2 * x_hat + 1
        let cfg = LayerNormConfig::new(3);
        let ln = LayerNorm::new(cfg, vec![2.0, 2.0, 2.0], vec![1.0, 1.0, 1.0]).unwrap();
        let input = vec![1.0_f32, 2.0, 3.0];
        let out = ln.forward(&input, 1).unwrap();
        // mean=2, var=2/3, x_hat = [-1.225, 0, 1.225] approx
        // y = 2 * x_hat + 1 = [-1.449, 1, 3.449]
        assert!((out[1] - 1.0).abs() < 1e-3);
        assert!(out[0] < 0.0);
        assert!(out[2] > out[1]);
    }

    #[test]
    fn forward_constant_input_gives_beta() {
        // 定数 input → mean=const, var=0, x_hat=0, output=beta
        let cfg = LayerNormConfig::new(4);
        let ln = LayerNorm::new(cfg, vec![3.0; 4], vec![7.0; 4]).unwrap();
        let input = vec![5.0_f32; 4];
        let out = ln.forward(&input, 1).unwrap();
        for &v in &out {
            assert!((v - 7.0).abs() < 1e-3, "expected ~7, got {v}");
        }
    }

    #[test]
    fn multi_batch_processes_rows_independently() {
        let cfg = LayerNormConfig::new(3).with_affine(false);
        let ln = LayerNorm::default_init(cfg).unwrap();
        let input = vec![
            1.0_f32, 2.0, 3.0, // batch 0
            10.0, 20.0, 30.0, // batch 1
        ];
        let out = ln.forward(&input, 2).unwrap();
        // 両 batch とも同じ正規化パターン (等差数列なので mean=中央、std 同じ比率)
        // x_hat = [-sqrt(3/2), 0, sqrt(3/2)] ≈ [-1.225, 0, 1.225]
        assert!((out[0] - out[3]).abs() < 1e-3);
        assert!((out[1] - out[4]).abs() < 1e-3);
        assert!((out[2] - out[5]).abs() < 1e-3);
    }

    #[test]
    fn backward_gradient_matches_finite_difference() {
        // Numerical gradient check
        let cfg = LayerNormConfig::new(4);
        let ln = LayerNorm::new(cfg, vec![1.2, 0.8, 1.5, 0.9], vec![0.1, -0.2, 0.05, 0.3]).unwrap();
        let batch = 3;
        let input: Vec<f32> = (0..batch * 4)
            .map(|i| ((i as f32) * 0.7 - 3.0).sin())
            .collect();
        let grad_output: Vec<f32> = (0..batch * 4)
            .map(|i| ((i as f32) * 0.3 + 0.1).cos())
            .collect();

        let (analytical, _, _) = ln.backward(&input, &grad_output, batch).unwrap();

        let h = 1e-3_f32;
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
            let ana = analytical[i];
            let diff = (ana - num).abs();
            let scale = ana.abs().max(num.abs()).max(1e-4);
            assert!(
                diff / scale < 5e-2,
                "input[{i}] analytical={ana}, numerical={num}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn backward_gamma_grad_matches_finite_difference() {
        let cfg = LayerNormConfig::new(3);
        let gamma_init = vec![1.0_f32, 0.5, 2.0];
        let beta_init = vec![0.1_f32, 0.2, -0.1];
        let batch = 2;
        let input: Vec<f32> = (0..batch * 3).map(|i| (i as f32 * 0.4).cos()).collect();
        let grad_output: Vec<f32> = (0..batch * 3).map(|i| (i as f32 * 0.2).sin()).collect();

        let ln = LayerNorm::new(cfg, gamma_init.clone(), beta_init.clone()).unwrap();
        let (_, analytical_gg, _) = ln.backward(&input, &grad_output, batch).unwrap();

        let h = 1e-3_f32;
        for i in 0..gamma_init.len() {
            let mut gp = gamma_init.clone();
            gp[i] += h;
            let lp = LayerNorm::new(cfg, gp, beta_init.clone()).unwrap();
            let out_p = lp.forward(&input, batch).unwrap();
            let mut gm = gamma_init.clone();
            gm[i] -= h;
            let lm = LayerNorm::new(cfg, gm, beta_init.clone()).unwrap();
            let out_m = lm.forward(&input, batch).unwrap();
            let loss_p: f32 = out_p.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let loss_m: f32 = out_m.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let ana = analytical_gg[i];
            let diff = (ana - num).abs();
            let scale = ana.abs().max(num.abs()).max(1e-4);
            assert!(
                diff / scale < 1e-2,
                "gamma[{i}] analytical={ana}, numerical={num}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn backward_beta_grad_matches_sum_of_grad_output() {
        let cfg = LayerNormConfig::new(3);
        let ln = LayerNorm::default_init(cfg).unwrap();
        let batch = 3;
        let input: Vec<f32> = (0..batch * 3).map(|i| i as f32).collect();
        let grad_output = vec![
            1.0_f32, 2.0, 3.0, // batch 0
            0.5, 1.5, 2.5, // batch 1
            -1.0, 0.0, 1.0, // batch 2
        ];
        let (_, _, gb) = ln.backward(&input, &grad_output, batch).unwrap();
        // ch 0 sum: 1 + 0.5 + (-1) = 0.5
        // ch 1 sum: 2 + 1.5 + 0 = 3.5
        // ch 2 sum: 3 + 2.5 + 1 = 6.5
        assert!((gb[0] - 0.5).abs() < 1e-6);
        assert!((gb[1] - 3.5).abs() < 1e-6);
        assert!((gb[2] - 6.5).abs() < 1e-6);
    }

    #[test]
    fn eps_stabilizes_zero_variance_input() {
        // 定数 input で var=0、eps>0 なので NaN にならず output=beta を返す
        let cfg = LayerNormConfig::new(3).with_eps(1e-5);
        let ln = LayerNorm::new(cfg, vec![1.0; 3], vec![0.0; 3]).unwrap();
        let input = vec![5.0_f32; 3];
        let out = ln.forward(&input, 1).unwrap();
        for &v in &out {
            assert!(v.is_finite(), "must not be NaN/Inf");
            assert!(v.abs() < 1e-3, "expected ~0, got {v}");
        }
    }

    #[test]
    fn invalid_config_returns_error() {
        let cfg = LayerNormConfig::new(0);
        let err = cfg.validate().expect_err("normalized_shape=0 must fail");
        assert!(matches!(err, LayerNormError::InvalidConfig { .. }));

        let cfg = LayerNormConfig::new(3).with_eps(-1.0);
        let err = cfg.validate().expect_err("negative eps must fail");
        assert!(matches!(err, LayerNormError::InvalidConfig { .. }));
    }

    #[test]
    fn gamma_beta_shape_mismatch_returns_error() {
        let cfg = LayerNormConfig::new(4);
        let err = LayerNorm::new(cfg, vec![1.0; 3], vec![0.0; 4]).expect_err("gamma len 3 != 4");
        assert!(matches!(
            err,
            LayerNormError::ShapeMismatch { field: "gamma", .. }
        ));

        let err = LayerNorm::new(cfg, vec![1.0; 4], vec![0.0; 3]).expect_err("beta len 3 != 4");
        assert!(matches!(
            err,
            LayerNormError::ShapeMismatch { field: "beta", .. }
        ));
    }

    #[test]
    fn input_shape_mismatch_returns_error() {
        let cfg = LayerNormConfig::new(4);
        let ln = LayerNorm::default_init(cfg).unwrap();
        let err = ln
            .forward(&[0.0; 5], 1)
            .expect_err("input len 5 vs expected 4");
        assert!(matches!(err, LayerNormError::InputShapeMismatch { .. }));
    }

    #[test]
    fn grad_output_shape_mismatch_returns_error() {
        let cfg = LayerNormConfig::new(4);
        let ln = LayerNorm::default_init(cfg).unwrap();
        let err = ln
            .backward(&[0.0; 4], &[0.0; 5], 1)
            .expect_err("grad_output len mismatch");
        assert!(matches!(
            err,
            LayerNormError::GradOutputShapeMismatch { .. }
        ));
    }

    #[test]
    fn default_init_sets_gamma_1_beta_0() {
        let cfg = LayerNormConfig::new(5);
        let ln = LayerNorm::default_init(cfg).unwrap();
        assert_eq!(ln.gamma(), &[1.0; 5]);
        assert_eq!(ln.beta(), &[0.0; 5]);
    }

    #[test]
    fn default_init_without_affine_gives_empty_gamma_beta() {
        let cfg = LayerNormConfig::new(5).with_affine(false);
        let ln = LayerNorm::default_init(cfg).unwrap();
        assert!(ln.gamma().is_empty());
        assert!(ln.beta().is_empty());
    }

    #[test]
    fn config_builder_pattern_composes() {
        let cfg = LayerNormConfig::new(128).with_eps(1e-6).with_affine(false);
        cfg.validate().expect("valid");
        assert_eq!(cfg.normalized_shape, 128);
        assert!((cfg.eps - 1e-6).abs() < 1e-9);
        assert!(!cfg.elementwise_affine);
    }

    #[test]
    fn error_display_and_error_trait() {
        let e = LayerNormError::InvalidConfig {
            reason: "test".to_string(),
        };
        let s = format!("{e}");
        assert!(s.contains("invalid LayerNorm config"));
        let boxed: Box<dyn std::error::Error> = Box::new(e);
        assert!(boxed.to_string().contains("test"));
    }
}
