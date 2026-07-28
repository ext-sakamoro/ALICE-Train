//! 1D convolution — forward + 手書き backward (grouped / depthwise 対応、stride / padding / dilation)。
//!
//! FastSpeech2 Postnet (kernel 5 / stride 1 / padding 2) / VITS2 duration predictor / Vocos
//! ConvNeXt (depthwise、kernel 7) の共通基盤として `feature = "tts"` gate に実装される。
//!
//! # 演算定義
//!
//! - Input: `[batch, in_channels, in_len]`
//! - Weight: `[out_channels, in_channels / groups, kernel_size]`
//! - Bias (optional): `[out_channels]`
//! - Output: `[batch, out_channels, out_len]`
//!   - `out_len = (in_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1`
//!
//! # backward
//!
//! - `grad_input[b, c_in, t_in] = Σ_{c_out in group} Σ_{k} grad_output[b, c_out, t_out] * weight[c_out, c_in_rel, k]`
//!   where `t_in = t_out * stride - padding + k * dilation`
//! - `grad_weight[c_out, c_in_rel, k] = Σ_{b} Σ_{t_out} grad_output[b, c_out, t_out] * input[b, c_in, t_in]`
//! - `grad_bias[c_out] = Σ_{b} Σ_{t_out} grad_output[b, c_out, t_out]`
//!
//! # 実装 note
//!
//! - Straightforward loops (no im2col / no BLAS)、Phase T.4a で bottleneck 判明したら最適化
//! - `f32` weight を [`Conv1d`] struct に保持 (STE FP32 shadow は本 primitive の責務外)
//! - Grouped conv: `in_channels % groups == 0` かつ `out_channels % groups == 0`

use serde::{Deserialize, Serialize};

/// Conv1D config。builder pattern の代わりに struct をそのまま渡す設計。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Conv1dConfig {
    /// 入力 channel 数。
    pub in_channels: usize,
    /// 出力 channel 数。
    pub out_channels: usize,
    /// カーネルサイズ。
    pub kernel_size: usize,
    /// ストライド。
    pub stride: usize,
    /// パディング (両側同量)。
    pub padding: usize,
    /// 膨張率。
    pub dilation: usize,
    /// グループ数 (1 = 通常 conv、`in_channels` = depthwise)。
    pub groups: usize,
    /// bias を持つか。
    pub bias: bool,
}

impl Conv1dConfig {
    /// 最も一般的な config (stride=1, padding=0, dilation=1, groups=1, bias=true)。
    #[must_use]
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
        }
    }

    /// `Self::new` + padding 指定。
    #[must_use]
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// `Self::new` + groups 指定 (depthwise conv 等)。
    #[must_use]
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// `Self::new` + stride 指定。
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// `Self::new` + dilation 指定。
    #[must_use]
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// `Self::new` + bias 有無指定。
    #[must_use]
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// 出力長を計算する。
    ///
    /// `out_len = (in_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1`
    #[must_use]
    pub fn out_len(&self, in_len: usize) -> usize {
        let effective_kernel = self.dilation * (self.kernel_size - 1) + 1;
        let padded_len = in_len + 2 * self.padding;
        if padded_len < effective_kernel {
            return 0;
        }
        (padded_len - effective_kernel) / self.stride + 1
    }

    /// config の validity 検証。
    ///
    /// # Errors
    ///
    /// - `in_channels == 0` / `out_channels == 0` / `kernel_size == 0`
    /// - `stride == 0` / `dilation == 0` / `groups == 0`
    /// - `in_channels % groups != 0` or `out_channels % groups != 0`
    pub fn validate(&self) -> Result<(), Conv1dError> {
        if self.in_channels == 0 {
            return Err(Conv1dError::InvalidConfig {
                reason: "in_channels must be > 0".to_string(),
            });
        }
        if self.out_channels == 0 {
            return Err(Conv1dError::InvalidConfig {
                reason: "out_channels must be > 0".to_string(),
            });
        }
        if self.kernel_size == 0 {
            return Err(Conv1dError::InvalidConfig {
                reason: "kernel_size must be > 0".to_string(),
            });
        }
        if self.stride == 0 {
            return Err(Conv1dError::InvalidConfig {
                reason: "stride must be > 0".to_string(),
            });
        }
        if self.dilation == 0 {
            return Err(Conv1dError::InvalidConfig {
                reason: "dilation must be > 0".to_string(),
            });
        }
        if self.groups == 0 {
            return Err(Conv1dError::InvalidConfig {
                reason: "groups must be > 0".to_string(),
            });
        }
        if !self.in_channels.is_multiple_of(self.groups) {
            return Err(Conv1dError::InvalidConfig {
                reason: format!(
                    "in_channels {} not divisible by groups {}",
                    self.in_channels, self.groups
                ),
            });
        }
        if !self.out_channels.is_multiple_of(self.groups) {
            return Err(Conv1dError::InvalidConfig {
                reason: format!(
                    "out_channels {} not divisible by groups {}",
                    self.out_channels, self.groups
                ),
            });
        }
        Ok(())
    }
}

/// Conv1D レイヤー (重み + bias を保持)。
///
/// - 重み shape: `[out_channels, in_channels / groups, kernel_size]` の flatten `Vec<f32>`
/// - bias shape: `[out_channels]` の flatten `Vec<f32>` (bias 無効時は空)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conv1d {
    config: Conv1dConfig,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv1d {
    /// 与えた weight / bias で新しい layer を構築する。
    ///
    /// # Errors
    ///
    /// - config validation ([`Conv1dConfig::validate`])
    /// - weight len が `out_channels * (in_channels / groups) * kernel_size` と不一致
    /// - bias len が `out_channels` (bias=true) or 0 (bias=false) と不一致
    pub fn new(
        config: Conv1dConfig,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, Conv1dError> {
        config.validate()?;
        let expected_w =
            config.out_channels * (config.in_channels / config.groups) * config.kernel_size;
        if weight.len() != expected_w {
            return Err(Conv1dError::WeightShapeMismatch {
                expected: expected_w,
                actual: weight.len(),
            });
        }
        let expected_b = if config.bias { config.out_channels } else { 0 };
        if bias.len() != expected_b {
            return Err(Conv1dError::BiasShapeMismatch {
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

    /// zero 初期化で新しい layer を構築する (テスト用途など)。
    ///
    /// # Errors
    ///
    /// config validation。
    pub fn zeros(config: Conv1dConfig) -> Result<Self, Conv1dError> {
        config.validate()?;
        let n_w = config.out_channels * (config.in_channels / config.groups) * config.kernel_size;
        let n_b = if config.bias { config.out_channels } else { 0 };
        Ok(Self {
            config,
            weight: vec![0.0; n_w],
            bias: vec![0.0; n_b],
        })
    }

    /// config への参照。
    #[must_use]
    pub fn config(&self) -> &Conv1dConfig {
        &self.config
    }

    /// weight `Vec<f32>` への参照 (flatten `[out_channels, in_channels/groups, kernel_size]`)。
    #[must_use]
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// weight `Vec<f32>` への可変参照 (optimizer 更新用)。
    pub fn weight_mut(&mut self) -> &mut [f32] {
        &mut self.weight
    }

    /// bias `Vec<f32>` への参照 (bias=false なら空 slice)。
    #[must_use]
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// bias `Vec<f32>` への可変参照 (optimizer 更新用)。
    pub fn bias_mut(&mut self) -> &mut [f32] {
        &mut self.bias
    }

    /// forward pass。
    ///
    /// # 引数
    ///
    /// - `input`: `[batch, in_channels, in_len]` flatten (`Vec<f32>` shape 検証は `in_len` から逆算)
    /// - `batch`: batch size
    /// - `in_len`: 入力時系列長
    ///
    /// # 戻り値
    ///
    /// `[batch, out_channels, out_len]` flatten `Vec<f32>`。
    ///
    /// # Errors
    ///
    /// - input.len() != batch * in_channels * in_len
    /// - out_len == 0 (input too short)
    pub fn forward(
        &self,
        input: &[f32],
        batch: usize,
        in_len: usize,
    ) -> Result<Vec<f32>, Conv1dError> {
        let cfg = &self.config;
        if input.len() != batch * cfg.in_channels * in_len {
            return Err(Conv1dError::InputShapeMismatch {
                expected: batch * cfg.in_channels * in_len,
                actual: input.len(),
            });
        }
        let out_len = cfg.out_len(in_len);
        if out_len == 0 {
            return Err(Conv1dError::InputTooShort {
                in_len,
                min_required: cfg.dilation * (cfg.kernel_size - 1) + 1
                    - 2 * cfg.padding.min(cfg.dilation * (cfg.kernel_size - 1) / 2),
            });
        }

        let in_ch_per_group = cfg.in_channels / cfg.groups;
        let out_ch_per_group = cfg.out_channels / cfg.groups;

        let mut output = vec![0.0_f32; batch * cfg.out_channels * out_len];

        for b in 0..batch {
            let input_batch_offset = b * cfg.in_channels * in_len;
            let output_batch_offset = b * cfg.out_channels * out_len;

            for g in 0..cfg.groups {
                let in_ch_start = g * in_ch_per_group;
                let out_ch_start = g * out_ch_per_group;

                for c_out_g in 0..out_ch_per_group {
                    let c_out = out_ch_start + c_out_g;
                    let bias_val = if cfg.bias { self.bias[c_out] } else { 0.0 };

                    for t_out in 0..out_len {
                        let mut acc = bias_val;
                        for c_in_g in 0..in_ch_per_group {
                            let c_in = in_ch_start + c_in_g;
                            for k in 0..cfg.kernel_size {
                                // t_in = t_out * stride + k * dilation - padding
                                // usize + checked_sub で left/right padding を負値なしで表現
                                let t_offset = t_out * cfg.stride + k * cfg.dilation;
                                let Some(t_in) = t_offset.checked_sub(cfg.padding) else {
                                    continue;
                                };
                                if t_in >= in_len {
                                    continue;
                                }
                                let input_idx = input_batch_offset + c_in * in_len + t_in;
                                // weight[c_out, c_in_g, k]
                                let weight_idx = c_out * in_ch_per_group * cfg.kernel_size
                                    + c_in_g * cfg.kernel_size
                                    + k;
                                acc += input[input_idx] * self.weight[weight_idx];
                            }
                        }
                        let output_idx = output_batch_offset + c_out * out_len + t_out;
                        output[output_idx] = acc;
                    }
                }
            }
        }

        Ok(output)
    }

    /// backward pass — grad_input / grad_weight / grad_bias を計算して返す。
    ///
    /// # 引数
    ///
    /// - `input`: forward 時の入力 (`[batch, in_channels, in_len]` flatten)
    /// - `grad_output`: 出力勾配 (`[batch, out_channels, out_len]` flatten)
    /// - `batch`, `in_len`: forward と同じ
    ///
    /// # 戻り値
    ///
    /// `(grad_input, grad_weight, grad_bias)`:
    /// - `grad_input`: `[batch, in_channels, in_len]` flatten
    /// - `grad_weight`: `[out_channels, in_channels/groups, kernel_size]` flatten
    /// - `grad_bias`: `[out_channels]` flatten (bias=false でも len は out_channels、値は全 0)
    ///
    /// # Errors
    ///
    /// input / grad_output の shape 不一致。
    pub fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        batch: usize,
        in_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Conv1dError> {
        let cfg = &self.config;
        if input.len() != batch * cfg.in_channels * in_len {
            return Err(Conv1dError::InputShapeMismatch {
                expected: batch * cfg.in_channels * in_len,
                actual: input.len(),
            });
        }
        let out_len = cfg.out_len(in_len);
        if grad_output.len() != batch * cfg.out_channels * out_len {
            return Err(Conv1dError::GradOutputShapeMismatch {
                expected: batch * cfg.out_channels * out_len,
                actual: grad_output.len(),
            });
        }

        let in_ch_per_group = cfg.in_channels / cfg.groups;
        let out_ch_per_group = cfg.out_channels / cfg.groups;

        let mut grad_input = vec![0.0_f32; batch * cfg.in_channels * in_len];
        let mut grad_weight = vec![0.0_f32; self.weight.len()];
        let mut grad_bias = vec![0.0_f32; cfg.out_channels];

        for b in 0..batch {
            let input_batch_offset = b * cfg.in_channels * in_len;
            let output_batch_offset = b * cfg.out_channels * out_len;

            for g in 0..cfg.groups {
                let in_ch_start = g * in_ch_per_group;
                let out_ch_start = g * out_ch_per_group;

                for c_out_g in 0..out_ch_per_group {
                    let c_out = out_ch_start + c_out_g;

                    for t_out in 0..out_len {
                        let output_idx = output_batch_offset + c_out * out_len + t_out;
                        let go = grad_output[output_idx];

                        grad_bias[c_out] += go;

                        for c_in_g in 0..in_ch_per_group {
                            let c_in = in_ch_start + c_in_g;
                            for k in 0..cfg.kernel_size {
                                let t_offset = t_out * cfg.stride + k * cfg.dilation;
                                let Some(t_in) = t_offset.checked_sub(cfg.padding) else {
                                    continue;
                                };
                                if t_in >= in_len {
                                    continue;
                                }
                                let input_idx = input_batch_offset + c_in * in_len + t_in;
                                let weight_idx = c_out * in_ch_per_group * cfg.kernel_size
                                    + c_in_g * cfg.kernel_size
                                    + k;

                                grad_weight[weight_idx] += go * input[input_idx];
                                grad_input[input_idx] += go * self.weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }

        Ok((grad_input, grad_weight, grad_bias))
    }
}

/// Conv1D 操作で発生し得るエラー。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Conv1dError {
    /// config が不正 (in/out channel 0、kernel 0、stride 0、groups 整合性など)。
    InvalidConfig {
        /// 具体的な理由。
        reason: String,
    },
    /// weight len が期待値と不一致。
    WeightShapeMismatch {
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
    /// bias len が期待値と不一致。
    BiasShapeMismatch {
        /// 期待 len (`out_channels` or 0)。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
    /// input 長が `batch * in_channels * in_len` と不一致。
    InputShapeMismatch {
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
    /// grad_output 長が `batch * out_channels * out_len` と不一致。
    GradOutputShapeMismatch {
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
    /// 入力時系列長が短すぎて有効な出力を生成できない。
    InputTooShort {
        /// 実際の in_len。
        in_len: usize,
        /// 必要な最小長。
        min_required: usize,
    },
}

impl std::fmt::Display for Conv1dError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig { reason } => write!(f, "invalid Conv1d config: {reason}"),
            Self::WeightShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "weight shape mismatch: expected {expected}, got {actual}"
                )
            }
            Self::BiasShapeMismatch { expected, actual } => {
                write!(f, "bias shape mismatch: expected {expected}, got {actual}")
            }
            Self::InputShapeMismatch { expected, actual } => {
                write!(f, "input shape mismatch: expected {expected}, got {actual}")
            }
            Self::GradOutputShapeMismatch { expected, actual } => write!(
                f,
                "grad_output shape mismatch: expected {expected}, got {actual}"
            ),
            Self::InputTooShort {
                in_len,
                min_required,
            } => write!(
                f,
                "input too short: in_len={in_len}, min_required={min_required}"
            ),
        }
    }
}

impl std::error::Error for Conv1dError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_out_len_matches_pytorch() {
        // kernel=3, padding=1, stride=1 → out_len = in_len (same padding)
        let cfg = Conv1dConfig::new(1, 1, 3).with_padding(1);
        assert_eq!(cfg.out_len(10), 10);

        // kernel=5, padding=2, stride=1 → same
        let cfg = Conv1dConfig::new(1, 1, 5).with_padding(2);
        assert_eq!(cfg.out_len(20), 20);

        // kernel=3, padding=0, stride=1 → out_len = in_len - 2
        let cfg = Conv1dConfig::new(1, 1, 3);
        assert_eq!(cfg.out_len(10), 8);

        // kernel=3, padding=0, stride=2 → out_len = (in - k) / s + 1
        let cfg = Conv1dConfig::new(1, 1, 3).with_stride(2);
        assert_eq!(cfg.out_len(10), (10 - 3) / 2 + 1);
    }

    #[test]
    fn identity_conv_reproduces_input() {
        // kernel=1, weight=[[1.0]], bias=[0.0] で恒等
        let cfg = Conv1dConfig::new(1, 1, 1);
        let conv = Conv1d::new(cfg, vec![1.0], vec![0.0]).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = conv.forward(&input, 1, 4).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn bias_only_conv_adds_bias() {
        // kernel=1, weight=[[0.0]], bias=[2.5] で全 output = 2.5
        let cfg = Conv1dConfig::new(1, 1, 1);
        let conv = Conv1d::new(cfg, vec![0.0], vec![2.5]).unwrap();
        let input = vec![1.0, 2.0, 3.0];
        let output = conv.forward(&input, 1, 3).unwrap();
        for &v in &output {
            assert!((v - 2.5).abs() < 1e-6);
        }
    }

    #[test]
    fn sum_kernel_produces_expected_sums() {
        // kernel=3, weight=[[1,1,1]], bias=0, padding=0 → out[i] = in[i] + in[i+1] + in[i+2]
        let cfg = Conv1dConfig::new(1, 1, 3);
        let conv = Conv1d::new(cfg, vec![1.0, 1.0, 1.0], vec![0.0]).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = conv.forward(&input, 1, 5).unwrap();
        assert_eq!(output.len(), 3); // out_len = 5 - 3 + 1 = 3
        assert!((output[0] - 6.0).abs() < 1e-6); // 1+2+3
        assert!((output[1] - 9.0).abs() < 1e-6); // 2+3+4
        assert!((output[2] - 12.0).abs() < 1e-6); // 3+4+5
    }

    #[test]
    fn same_padding_preserves_length() {
        // kernel=3, padding=1 (same padding) → out_len = in_len
        let cfg = Conv1dConfig::new(1, 1, 3).with_padding(1);
        let conv = Conv1d::new(cfg, vec![1.0, 1.0, 1.0], vec![0.0]).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = conv.forward(&input, 1, 5).unwrap();
        assert_eq!(output.len(), 5);
        // out[0] = 0 + 1 + 2 = 3, out[1] = 1 + 2 + 3 = 6, out[4] = 4 + 5 + 0 = 9
        assert!((output[0] - 3.0).abs() < 1e-6);
        assert!((output[1] - 6.0).abs() < 1e-6);
        assert!((output[4] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn multi_channel_output_stacks_correctly() {
        // in=1, out=2, kernel=1: weight = [[1], [2]], bias = [0, 0]
        let cfg = Conv1dConfig::new(1, 2, 1);
        let conv = Conv1d::new(cfg, vec![1.0, 2.0], vec![0.0, 0.0]).unwrap();
        let input = vec![3.0, 4.0];
        let output = conv.forward(&input, 1, 2).unwrap();
        // output layout [out_ch, len]: [3, 4] (ch 0) + [6, 8] (ch 1)
        assert_eq!(output.len(), 4);
        assert!((output[0] - 3.0).abs() < 1e-6);
        assert!((output[1] - 4.0).abs() < 1e-6);
        assert!((output[2] - 6.0).abs() < 1e-6);
        assert!((output[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn depthwise_conv_operates_per_channel() {
        // in=2, out=2, kernel=1, groups=2 (depthwise) → 各 channel 独立
        let cfg = Conv1dConfig::new(2, 2, 1).with_groups(2);
        // weight shape [out, in/groups, k] = [2, 1, 1]
        let conv = Conv1d::new(cfg, vec![2.0, 3.0], vec![0.0, 0.0]).unwrap();
        // input [batch=1, in_ch=2, len=2]: ch0 = [1, 2], ch1 = [10, 20]
        let input = vec![1.0, 2.0, 10.0, 20.0];
        let output = conv.forward(&input, 1, 2).unwrap();
        // out ch0 = in ch0 * 2 = [2, 4], out ch1 = in ch1 * 3 = [30, 60]
        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 4.0).abs() < 1e-6);
        assert!((output[2] - 30.0).abs() < 1e-6);
        assert!((output[3] - 60.0).abs() < 1e-6);
    }

    #[test]
    fn backward_gradient_matches_finite_difference() {
        // Numerical gradient check: analytical backward vs finite-difference forward
        let cfg = Conv1dConfig::new(2, 3, 3).with_padding(1);
        let n_w = 3 * 2 * 3;
        let weight: Vec<f32> = (0..n_w).map(|i| ((i as f32) * 0.1 - 0.5).sin()).collect();
        let bias: Vec<f32> = vec![0.1, -0.2, 0.05];
        let conv = Conv1d::new(cfg, weight, bias).unwrap();
        let batch = 2;
        let in_len = 5;
        let input: Vec<f32> = (0..batch * cfg.in_channels * in_len)
            .map(|i| ((i as f32) * 0.05).cos() * 0.7)
            .collect();
        // 適当な grad_output
        let out_len = cfg.out_len(in_len);
        let grad_output: Vec<f32> = (0..batch * cfg.out_channels * out_len)
            .map(|i| ((i as f32) * 0.03 + 0.1).sin() * 0.5)
            .collect();

        let (analytical_grad_input, _, _) =
            conv.backward(&input, &grad_output, batch, in_len).unwrap();

        // finite difference: dL/dx_i ≈ (L(x + h*e_i) - L(x - h*e_i)) / (2h)
        // where L = sum(grad_output * forward(x))
        let h = 1e-3_f32;
        let mut numerical = vec![0.0_f32; input.len()];
        for i in 0..input.len() {
            let mut ip = input.clone();
            ip[i] += h;
            let out_p = conv.forward(&ip, batch, in_len).unwrap();
            let mut im = input.clone();
            im[i] -= h;
            let out_m = conv.forward(&im, batch, in_len).unwrap();
            let loss_p: f32 = out_p.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let loss_m: f32 = out_m.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            numerical[i] = (loss_p - loss_m) / (2.0 * h);
        }

        // 相対誤差で検証 (1e-2 精度 h=1e-3 なら十分)
        for (a, n) in analytical_grad_input.iter().zip(&numerical) {
            let diff = (a - n).abs();
            let scale = a.abs().max(n.abs()).max(1e-4);
            assert!(
                diff / scale < 1e-2,
                "analytical={a}, numerical={n}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn backward_grad_weight_matches_finite_difference() {
        let cfg = Conv1dConfig::new(2, 2, 3).with_padding(0);
        let n_w = 2 * 2 * 3;
        let weight: Vec<f32> = (0..n_w).map(|i| (i as f32 * 0.13).cos() * 0.5).collect();
        let bias: Vec<f32> = vec![0.02, -0.03];
        let conv = Conv1d::new(cfg, weight.clone(), bias.clone()).unwrap();
        let batch = 1;
        let in_len = 6;
        let input: Vec<f32> = (0..batch * cfg.in_channels * in_len)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let out_len = cfg.out_len(in_len);
        let grad_output: Vec<f32> = (0..batch * cfg.out_channels * out_len)
            .map(|i| (i as f32 * 0.07 + 0.3).cos())
            .collect();

        let (_, analytical_grad_weight, _) =
            conv.backward(&input, &grad_output, batch, in_len).unwrap();

        let h = 1e-3_f32;
        for i in 0..weight.len() {
            let mut w_p = weight.clone();
            w_p[i] += h;
            let cp = Conv1d::new(cfg, w_p, bias.clone()).unwrap();
            let out_p = cp.forward(&input, batch, in_len).unwrap();
            let mut w_m = weight.clone();
            w_m[i] -= h;
            let cm = Conv1d::new(cfg, w_m, bias.clone()).unwrap();
            let out_m = cm.forward(&input, batch, in_len).unwrap();
            let loss_p: f32 = out_p.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let loss_m: f32 = out_m.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let ana = analytical_grad_weight[i];
            let diff = (ana - num).abs();
            let scale = ana.abs().max(num.abs()).max(1e-4);
            assert!(
                diff / scale < 1e-2,
                "weight[{i}] analytical={ana}, numerical={num}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn backward_grad_bias_is_sum_of_grad_output() {
        let cfg = Conv1dConfig::new(1, 2, 3).with_padding(0);
        let conv = Conv1d::zeros(cfg).unwrap();
        let batch = 2;
        let in_len = 5;
        let input = vec![0.0_f32; batch * cfg.in_channels * in_len];
        // grad_output shape [batch=2, out_ch=2, out_len=3]
        let grad_output = vec![
            // batch 0
            1.0, 2.0, 3.0, // ch 0
            4.0, 5.0, 6.0, // ch 1
            // batch 1
            0.5, 1.5, 2.5, // ch 0
            -1.0, -2.0, -3.0, // ch 1
        ];
        let (_, _, grad_bias) = conv.backward(&input, &grad_output, batch, in_len).unwrap();
        // ch 0 sum: (1+2+3) + (0.5+1.5+2.5) = 6 + 4.5 = 10.5
        // ch 1 sum: (4+5+6) + (-1-2-3) = 15 - 6 = 9.0
        assert!((grad_bias[0] - 10.5).abs() < 1e-6);
        assert!((grad_bias[1] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_config_returns_error() {
        let cfg = Conv1dConfig::new(0, 1, 3);
        let err = cfg.validate().expect_err("in_channels=0 must fail");
        assert!(matches!(err, Conv1dError::InvalidConfig { .. }));

        let cfg = Conv1dConfig::new(3, 2, 3).with_groups(2);
        // 3 % 2 != 0
        let err = cfg.validate().expect_err("groups mismatch must fail");
        assert!(matches!(err, Conv1dError::InvalidConfig { .. }));
    }

    #[test]
    fn weight_shape_mismatch_returns_error() {
        let cfg = Conv1dConfig::new(1, 1, 3);
        // expected weight len = 1 * 1 * 3 = 3、give 5
        let err = Conv1d::new(cfg, vec![1.0; 5], vec![0.0]).expect_err("shape mismatch");
        assert!(matches!(err, Conv1dError::WeightShapeMismatch { .. }));
    }

    #[test]
    fn bias_shape_mismatch_returns_error() {
        let cfg = Conv1dConfig::new(1, 2, 3);
        let err = Conv1d::new(cfg, vec![1.0; 6], vec![0.0; 5])
            .expect_err("bias shape mismatch must fail");
        assert!(matches!(err, Conv1dError::BiasShapeMismatch { .. }));

        // bias=false なら len=0 でないと error
        let cfg = Conv1dConfig::new(1, 2, 3).with_bias(false);
        let err =
            Conv1d::new(cfg, vec![1.0; 6], vec![0.0]).expect_err("bias=false but bias not empty");
        assert!(matches!(err, Conv1dError::BiasShapeMismatch { .. }));
    }

    #[test]
    fn input_shape_mismatch_returns_error() {
        let cfg = Conv1dConfig::new(1, 1, 3);
        let conv = Conv1d::zeros(cfg).unwrap();
        // expected input len = 1 * 1 * 10 = 10、give 5
        let err = conv
            .forward(&[0.0; 5], 1, 10)
            .expect_err("input shape mismatch");
        assert!(matches!(err, Conv1dError::InputShapeMismatch { .. }));
    }

    #[test]
    fn error_display_and_error_trait() {
        let e = Conv1dError::InvalidConfig {
            reason: "test".to_string(),
        };
        let s = format!("{e}");
        assert!(s.contains("invalid Conv1d config"));

        let boxed: Box<dyn std::error::Error> = Box::new(e);
        assert!(boxed.to_string().contains("test"));
    }

    #[test]
    fn config_builder_pattern_composes() {
        let cfg = Conv1dConfig::new(4, 8, 3)
            .with_padding(1)
            .with_stride(2)
            .with_dilation(1)
            .with_groups(1)
            .with_bias(false);
        cfg.validate().expect("valid");
        assert!(!cfg.bias);
        assert_eq!(cfg.stride, 2);
        assert_eq!(cfg.padding, 1);
    }

    #[test]
    fn weight_mut_and_bias_mut_allow_updates() {
        let cfg = Conv1dConfig::new(1, 1, 1);
        let mut conv = Conv1d::new(cfg, vec![1.0], vec![0.0]).unwrap();
        conv.weight_mut()[0] = 5.0;
        conv.bias_mut()[0] = 3.0;
        let out = conv.forward(&[2.0], 1, 1).unwrap();
        assert!((out[0] - (5.0 * 2.0 + 3.0)).abs() < 1e-6);
    }
}
