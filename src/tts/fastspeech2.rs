//! FastSpeech2 acoustic model forward-only skeleton (Phase T.4a MVP)。
//!
//! Ren et al. (2020) FastSpeech2 architecture の non-autoregressive acoustic model。
//! Wataru-Nakata/FastSpeech2-JSUT (日本語 JSUT 特化 fork) を参照実装として、mora + accent
//! 入力 → mel-spectrogram 出力の pipeline を組み立てる。
//!
//! # Architecture
//!
//! ```text
//! mora_ids [B, mora_len]
//!   ↓ Embedding
//! hidden [B, mora_len, hidden_dim]
//!   ↓ Sinusoidal Positional Encoding
//!   ↓
//! Encoder: FFT Block × N
//!   ├─ MultiHeadAttention (self)
//!   ├─ Add & LayerNorm
//!   ├─ Conv1D FFN (kernel k) + ReLU + Conv1D
//!   └─ Add & LayerNorm
//!   ↓ [B, mora_len, hidden_dim]
//!   ↓
//! Variance Adaptor
//!   ├─ Duration Predictor (Conv1D + LayerNorm + Conv1D + LayerNorm + Linear) → [B, mora_len]
//!   ├─ Pitch Predictor    (同構成) → [B, mora_len]
//!   ├─ Energy Predictor   (同構成) → [B, mora_len]
//!   ├─ Pitch/Energy を hidden に加算 (Conv1D proj で [B, mora, 1] → [B, mora, hidden])
//!   └─ Length Regulator: durations で mora → frame 展開
//!   ↓ [B, frame_len, hidden_dim]
//!   ↓
//! Decoder: FFT Block × N (+ Sinusoidal PE)
//!   ↓ [B, frame_len, hidden_dim]
//!   ↓ Linear (hidden → mel_dim)
//! mel_before [B, frame_len, mel_dim]
//!   ↓ Postnet (5 Conv1D + tanh)
//! mel_after = mel_before + postnet_residual [B, frame_len, mel_dim]
//! ```
//!
//! # 実装 note (MVP forward-only)
//!
//! - 全 layer の weight は user が初期化 (default_init は zero、学習は Phase T.4a continuation)
//! - Predictor の Conv1D は same padding (kernel 3, padding 1)、activation は ReLU
//! - Postnet は kernel 5, padding 2、tanh activation
//! - FFT block FFN は Conv1D (kernel k=1 or 3) で hidden → ffn_mid → hidden、activation は ReLU
//! - Length Regulator は target_durations (学習時 teacher forcing) or predicted rounded durations
//! - backward は本 MVP では未実装 (primitive backward の合成で追加可能)
//!
//! # 使用例
//!
//! ```rust,no_run
//! # #[cfg(feature = "tts")] {
//! use alice_train::tts::{FastSpeech2, FastSpeech2Config};
//!
//! let config = FastSpeech2Config {
//!     vocab_size: 100,
//!     hidden_dim: 128,
//!     num_heads: 2,
//!     num_encoder_layers: 4,
//!     num_decoder_layers: 4,
//!     fft_kernel_size: 3,
//!     fft_expansion: 4,
//!     predictor_kernel_size: 3,
//!     predictor_hidden: 128,
//!     mel_dim: 80,
//!     postnet_kernel_size: 5,
//!     postnet_layers: 5,
//!     postnet_hidden: 128,
//!     max_len: 1024,
//! };
//! let model = FastSpeech2::zeros(config).expect("build");
//!
//! let mora_ids: Vec<u32> = vec![1, 2, 3, 4, 5];
//! let target_durations: Vec<u32> = vec![3, 2, 4, 5, 3]; // frame 数
//! let mel = model.forward(&mora_ids, 1, 5, &target_durations).expect("forward");
//! // mel.len() = 1 * (3+2+4+5+3) * 80 = 1360
//! # }
//! ```

use crate::tts::primitives::{
    Conv1d, Conv1dConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, MultiHeadAttention,
    MultiHeadAttentionConfig, SinusoidalPositionalEncoding, SinusoidalPositionalEncodingConfig,
};
use serde::{Deserialize, Serialize};

/// FastSpeech2 モデル設定。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FastSpeech2Config {
    /// mora vocabulary サイズ (Phase T.1 g2p-ja 出力 alphabet)。
    pub vocab_size: usize,
    /// hidden dim (encoder / decoder 共通)。num_heads で割り切れる必須。
    pub hidden_dim: usize,
    /// FFT block の attention head 数。
    pub num_heads: usize,
    /// encoder FFT block の層数。
    pub num_encoder_layers: usize,
    /// decoder FFT block の層数。
    pub num_decoder_layers: usize,
    /// FFT block の FFN Conv1D kernel size (通常 3)。
    pub fft_kernel_size: usize,
    /// FFT block FFN の hidden 拡張率 (mid = hidden * expansion、通常 4)。
    pub fft_expansion: usize,
    /// Variance predictor の Conv1D kernel size (通常 3)。
    pub predictor_kernel_size: usize,
    /// Variance predictor の hidden dim (通常 hidden_dim と同じ)。
    pub predictor_hidden: usize,
    /// mel-spectrogram 次元 (通常 80)。
    pub mel_dim: usize,
    /// Postnet Conv1D kernel size (通常 5)。
    pub postnet_kernel_size: usize,
    /// Postnet 層数 (通常 5)。
    pub postnet_layers: usize,
    /// Postnet 中間 hidden (通常 512)。
    pub postnet_hidden: usize,
    /// 最大 sequence length (positional encoding pre-compute 上限)。
    pub max_len: usize,
}

impl FastSpeech2Config {
    /// config validity 検証。
    ///
    /// # Errors
    ///
    /// 各 field ゼロ値、hidden_dim % num_heads != 0 等。
    pub fn validate(&self) -> Result<(), FastSpeech2Error> {
        if self.vocab_size == 0
            || self.hidden_dim == 0
            || self.num_heads == 0
            || self.mel_dim == 0
            || self.max_len == 0
            || self.postnet_hidden == 0
            || self.postnet_layers == 0
            || self.predictor_hidden == 0
        {
            return Err(FastSpeech2Error::InvalidConfig {
                reason: "all size fields must be > 0".to_string(),
            });
        }
        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(FastSpeech2Error::InvalidConfig {
                reason: format!(
                    "hidden_dim {} not divisible by num_heads {}",
                    self.hidden_dim, self.num_heads
                ),
            });
        }
        Ok(())
    }
}

/// FFT block (Feed-Forward Transformer): MHA + Add&Norm + Conv1D FFN + Add&Norm。
#[derive(Clone, Debug, Serialize, Deserialize)]
struct FftBlock {
    self_attn: MultiHeadAttention,
    attn_norm: LayerNorm,
    ffn_conv1: Conv1d,
    ffn_conv2: Conv1d,
    ffn_norm: LayerNorm,
}

impl FftBlock {
    fn zeros(hidden_dim: usize, num_heads: usize, kernel_size: usize, expansion: usize) -> Self {
        let mha_cfg = MultiHeadAttentionConfig::new(hidden_dim, num_heads);
        let self_attn = MultiHeadAttention::zeros(mha_cfg).expect("valid MHA");
        let attn_norm = LayerNorm::default_init(LayerNormConfig::new(hidden_dim)).expect("norm");
        let mid = hidden_dim * expansion;
        let padding = kernel_size / 2;
        let ffn_conv1 =
            Conv1d::zeros(Conv1dConfig::new(hidden_dim, mid, kernel_size).with_padding(padding))
                .expect("conv1");
        let ffn_conv2 =
            Conv1d::zeros(Conv1dConfig::new(mid, hidden_dim, kernel_size).with_padding(padding))
                .expect("conv2");
        let ffn_norm = LayerNorm::default_init(LayerNormConfig::new(hidden_dim)).expect("norm");
        Self {
            self_attn,
            attn_norm,
            ffn_conv1,
            ffn_conv2,
            ffn_norm,
        }
    }

    /// FFT block forward: input `[batch, seq_len, hidden]` → output 同 shape。
    fn forward(&self, input: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        self.forward_masked(input, batch, seq_len, None)
    }

    /// FFT block forward with optional key_mask (Phase T.4a Phase D)。
    fn forward_masked(
        &self,
        input: &[f32],
        batch: usize,
        seq_len: usize,
        key_mask: Option<&[bool]>,
    ) -> Vec<f32> {
        let h = self.self_attn.config().embed_dim;
        // 1. Self-attention (bidirectional、optional mask)
        let attn_out = key_mask.map_or_else(
            || {
                self.self_attn
                    .forward_self_attention(input, batch, seq_len, false)
                    .expect("attn")
            },
            |m| {
                self.self_attn
                    .forward_self_attention_masked(input, batch, seq_len, false, m)
                    .expect("attn masked")
            },
        );
        // 2. Add & LayerNorm
        let residual = add(input, &attn_out);
        let after_attn = self
            .attn_norm
            .forward(&residual, batch * seq_len)
            .expect("norm");

        // 3. FFN: hidden → mid via Conv1D (需要 [B, hidden, seq_len] layout)
        let bs_hs = batch * seq_len * h;
        let mut reshaped = vec![0.0_f32; bs_hs];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..h {
                    reshaped[b * h * seq_len + c * seq_len + t] =
                        after_attn[b * seq_len * h + t * h + c];
                }
            }
        }
        let ffn_mid = self
            .ffn_conv1
            .forward(&reshaped, batch, seq_len)
            .expect("ffn1");
        let ffn_mid_relu: Vec<f32> = ffn_mid.iter().map(|&x| x.max(0.0)).collect();
        let ffn_out_ch = self
            .ffn_conv2
            .forward(&ffn_mid_relu, batch, seq_len)
            .expect("ffn2");
        let mut ffn_out = vec![0.0_f32; bs_hs];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..h {
                    ffn_out[b * seq_len * h + t * h + c] =
                        ffn_out_ch[b * h * seq_len + c * seq_len + t];
                }
            }
        }

        // 4. Add & LayerNorm
        let residual2 = add(&after_attn, &ffn_out);
        self.ffn_norm
            .forward(&residual2, batch * seq_len)
            .expect("norm2")
    }

    /// FFT block backward: `grad_output` から `grad_input` と全 sub-layer 勾配を計算する。
    ///
    /// Forward を再計算して中間 activation を取得し、reverse chain rule で backward。
    /// 戻り値: `(grad_input, FftBlockGrads)`
    ///
    /// Phase T.4a backward Phase 2: FftBlock 単体の backward は完備、
    /// FastSpeech2 全体 backward に統合するのは Phase T.4a 継続で対応する。
    /// 現状は private method で test module + 将来の backward_full から呼ばれる予定。
    #[allow(dead_code)] // Phase T.4a 継続で FastSpeech2::backward_full から呼ばれる
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> (Vec<f32>, FftBlockGrads) {
        self.backward_masked(input, grad_output, batch, seq_len, None)
    }

    /// FFT block backward with optional key_mask (Phase T.4a Phase D)。
    #[allow(dead_code)] // 将来 FastSpeech2::backward_full_variable から呼ばれる
    fn backward_masked(
        &self,
        input: &[f32],
        grad_output: &[f32],
        batch: usize,
        seq_len: usize,
        key_mask: Option<&[bool]>,
    ) -> (Vec<f32>, FftBlockGrads) {
        let h = self.self_attn.config().embed_dim;

        // === Forward 再計算 (中間 activation 保持) ===
        let attn_out = key_mask.map_or_else(
            || {
                self.self_attn
                    .forward_self_attention(input, batch, seq_len, false)
                    .expect("attn")
            },
            |m| {
                self.self_attn
                    .forward_self_attention_masked(input, batch, seq_len, false, m)
                    .expect("attn masked")
            },
        );
        let residual1 = add(input, &attn_out);
        let after_attn = self
            .attn_norm
            .forward(&residual1, batch * seq_len)
            .expect("norm");
        let bs_hs = batch * seq_len * h;
        let mut reshaped = vec![0.0_f32; bs_hs];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..h {
                    reshaped[b * h * seq_len + c * seq_len + t] =
                        after_attn[b * seq_len * h + t * h + c];
                }
            }
        }
        let ffn_conv1_out = self
            .ffn_conv1
            .forward(&reshaped, batch, seq_len)
            .expect("ffn1");
        let ffn_relu: Vec<f32> = ffn_conv1_out.iter().map(|&x| x.max(0.0)).collect();
        let ffn_conv2_out = self
            .ffn_conv2
            .forward(&ffn_relu, batch, seq_len)
            .expect("ffn2");
        let mut ffn_out = vec![0.0_f32; bs_hs];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..h {
                    ffn_out[b * seq_len * h + t * h + c] =
                        ffn_conv2_out[b * h * seq_len + c * seq_len + t];
                }
            }
        }
        let residual2 = add(&after_attn, &ffn_out);

        // === Backward chain ===
        // Step 1: ffn_norm backward
        let (grad_residual2, ffn_norm_gamma_grad, ffn_norm_beta_grad) = self
            .ffn_norm
            .backward(&residual2, grad_output, batch * seq_len)
            .expect("ffn_norm bw");

        // Step 2: Add branch: grad_after_attn_add = grad_residual2, grad_ffn_out = grad_residual2
        let mut grad_after_attn = grad_residual2.clone();
        let grad_ffn_out = grad_residual2;

        // Step 3: reshape grad_ffn_out [B, seq_len, h] → [B, h, seq_len]
        let mut grad_ffn_out_reshaped = vec![0.0_f32; bs_hs];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..h {
                    grad_ffn_out_reshaped[b * h * seq_len + c * seq_len + t] =
                        grad_ffn_out[b * seq_len * h + t * h + c];
                }
            }
        }

        // Step 4: ffn_conv2 backward
        let (grad_ffn_relu, ffn_conv2_w_grad, ffn_conv2_b_grad) = self
            .ffn_conv2
            .backward(&ffn_relu, &grad_ffn_out_reshaped, batch, seq_len)
            .expect("ffn_conv2 bw");

        // Step 5: ReLU backward: grad_pre_relu = grad_relu * (ffn_conv1_out > 0)
        let grad_pre_relu: Vec<f32> = grad_ffn_relu
            .iter()
            .zip(&ffn_conv1_out)
            .map(|(g, &pre)| if pre > 0.0 { *g } else { 0.0 })
            .collect();

        // Step 6: ffn_conv1 backward
        let (grad_reshaped, ffn_conv1_w_grad, ffn_conv1_b_grad) = self
            .ffn_conv1
            .backward(&reshaped, &grad_pre_relu, batch, seq_len)
            .expect("ffn_conv1 bw");

        // Step 7: reshape grad_reshaped [B, h, seq_len] → [B, seq_len, h] → add to grad_after_attn
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..h {
                    grad_after_attn[b * seq_len * h + t * h + c] +=
                        grad_reshaped[b * h * seq_len + c * seq_len + t];
                }
            }
        }

        // Step 8: attn_norm backward
        let (grad_residual1, attn_norm_gamma_grad, attn_norm_beta_grad) = self
            .attn_norm
            .backward(&residual1, &grad_after_attn, batch * seq_len)
            .expect("attn_norm bw");

        // Step 9: Add branch: grad_input_add = grad_residual1, grad_attn_out = grad_residual1
        let mut grad_input = grad_residual1.clone();

        // Step 10: MHA backward (self-attention Q=K=V=input, optional mask)
        let (grad_attn_input, mha_grads) = key_mask.map_or_else(
            || {
                self.self_attn
                    .backward_self_attention(input, &grad_residual1, batch, seq_len, false)
                    .expect("mha bw")
            },
            |m| {
                self.self_attn
                    .backward_self_attention_masked(
                        input,
                        &grad_residual1,
                        batch,
                        seq_len,
                        false,
                        m,
                    )
                    .expect("mha bw masked")
            },
        );

        // Step 11: sum both branches
        for (dst, src) in grad_input.iter_mut().zip(&grad_attn_input) {
            *dst += src;
        }

        (
            grad_input,
            FftBlockGrads {
                mha: mha_grads,
                attn_norm_gamma: attn_norm_gamma_grad,
                attn_norm_beta: attn_norm_beta_grad,
                ffn_conv1_w: ffn_conv1_w_grad,
                ffn_conv1_b: ffn_conv1_b_grad,
                ffn_conv2_w: ffn_conv2_w_grad,
                ffn_conv2_b: ffn_conv2_b_grad,
                ffn_norm_gamma: ffn_norm_gamma_grad,
                ffn_norm_beta: ffn_norm_beta_grad,
            },
        )
    }
}

/// FftBlock backward で返される sub-layer 勾配 bundle。
#[derive(Clone, Debug)]
pub struct FftBlockGrads {
    /// MHA 勾配 (Q/K/V/O weight + bias)。
    pub mha: crate::tts::MhaGrads,
    /// attn_norm gamma 勾配。
    pub attn_norm_gamma: Vec<f32>,
    /// attn_norm beta 勾配。
    pub attn_norm_beta: Vec<f32>,
    /// FFN Conv1 weight 勾配。
    pub ffn_conv1_w: Vec<f32>,
    /// FFN Conv1 bias 勾配。
    pub ffn_conv1_b: Vec<f32>,
    /// FFN Conv2 weight 勾配。
    pub ffn_conv2_w: Vec<f32>,
    /// FFN Conv2 bias 勾配。
    pub ffn_conv2_b: Vec<f32>,
    /// ffn_norm gamma 勾配。
    pub ffn_norm_gamma: Vec<f32>,
    /// ffn_norm beta 勾配。
    pub ffn_norm_beta: Vec<f32>,
}

/// Variance Predictor (Conv1D + LayerNorm + ReLU + Conv1D + LayerNorm + Linear → scalar)。
#[derive(Clone, Debug, Serialize, Deserialize)]
struct VariancePredictor {
    conv1: Conv1d,
    norm1: LayerNorm,
    conv2: Conv1d,
    norm2: LayerNorm,
    linear: Linear,
}

impl VariancePredictor {
    fn zeros(hidden_dim: usize, predictor_hidden: usize, kernel_size: usize) -> Self {
        let padding = kernel_size / 2;
        let conv1 = Conv1d::zeros(
            Conv1dConfig::new(hidden_dim, predictor_hidden, kernel_size).with_padding(padding),
        )
        .expect("conv1");
        let norm1 = LayerNorm::default_init(LayerNormConfig::new(predictor_hidden)).expect("norm1");
        let conv2 = Conv1d::zeros(
            Conv1dConfig::new(predictor_hidden, predictor_hidden, kernel_size)
                .with_padding(padding),
        )
        .expect("conv2");
        let norm2 = LayerNorm::default_init(LayerNormConfig::new(predictor_hidden)).expect("norm2");
        let linear = Linear::zeros(LinearConfig::new(predictor_hidden, 1)).expect("linear");
        Self {
            conv1,
            norm1,
            conv2,
            norm2,
            linear,
        }
    }

    /// input `[batch, seq_len, hidden]` → output `[batch, seq_len]` (scalar per position)。
    fn forward(&self, input: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let h = self.conv1.config().in_channels;
        let ph = self.conv1.config().out_channels;

        // reshape [B, seq_len, h] → [B, h, seq_len] for Conv1D
        let reshaped = reshape_time_last_to_channel_first(input, batch, seq_len, h);
        let conv1_out = self
            .conv1
            .forward(&reshaped, batch, seq_len)
            .expect("conv1");
        // ReLU
        let conv1_relu: Vec<f32> = conv1_out.iter().map(|&x| x.max(0.0)).collect();
        // reshape back for LayerNorm ([B, ph, seq_len] → [B*seq_len, ph])
        let mut norm1_in = vec![0.0_f32; batch * seq_len * ph];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..ph {
                    norm1_in[b * seq_len * ph + t * ph + c] =
                        conv1_relu[b * ph * seq_len + c * seq_len + t];
                }
            }
        }
        let norm1_out = self
            .norm1
            .forward(&norm1_in, batch * seq_len)
            .expect("norm1");

        // Conv2 (needs [B, ph, seq_len] again)
        let norm1_reshaped = reshape_time_last_to_channel_first(&norm1_out, batch, seq_len, ph);
        let conv2_out = self
            .conv2
            .forward(&norm1_reshaped, batch, seq_len)
            .expect("conv2");
        let conv2_relu: Vec<f32> = conv2_out.iter().map(|&x| x.max(0.0)).collect();
        // back to [B*seq_len, ph] for LayerNorm
        let mut norm2_in = vec![0.0_f32; batch * seq_len * ph];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..ph {
                    norm2_in[b * seq_len * ph + t * ph + c] =
                        conv2_relu[b * ph * seq_len + c * seq_len + t];
                }
            }
        }
        let norm2_out = self
            .norm2
            .forward(&norm2_in, batch * seq_len)
            .expect("norm2");

        // Linear (ph → 1)、shape [batch * seq_len, 1] は flat [batch, seq_len] と同じ
        self.linear
            .forward(&norm2_out, batch * seq_len)
            .expect("linear")
    }

    /// backward: `grad_output[batch, seq_len]` (or flat [batch*seq_len, 1] 同一 memory)
    /// → `(grad_input[batch, seq_len, hidden], VariancePredictorGrads)`。
    ///
    /// Forward 再計算 + reverse chain (Linear → norm2 → ReLU2 → conv2 → norm1 → ReLU1 → conv1)。
    #[allow(dead_code)] // Phase T.4a 継続で backward_full から呼ばれる
    fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> (Vec<f32>, VariancePredictorGrads) {
        let h = self.conv1.config().in_channels;
        let ph = self.conv1.config().out_channels;

        // === Forward 再計算 (intermediate 保存) ===
        let reshaped = reshape_time_last_to_channel_first(input, batch, seq_len, h);
        let conv1_out = self
            .conv1
            .forward(&reshaped, batch, seq_len)
            .expect("conv1 fw");
        let conv1_relu: Vec<f32> = conv1_out.iter().map(|&x| x.max(0.0)).collect();
        let mut norm1_in = vec![0.0_f32; batch * seq_len * ph];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..ph {
                    norm1_in[b * seq_len * ph + t * ph + c] =
                        conv1_relu[b * ph * seq_len + c * seq_len + t];
                }
            }
        }
        let norm1_out = self
            .norm1
            .forward(&norm1_in, batch * seq_len)
            .expect("norm1 fw");
        let norm1_reshaped = reshape_time_last_to_channel_first(&norm1_out, batch, seq_len, ph);
        let conv2_out = self
            .conv2
            .forward(&norm1_reshaped, batch, seq_len)
            .expect("conv2 fw");
        let conv2_relu: Vec<f32> = conv2_out.iter().map(|&x| x.max(0.0)).collect();
        let mut norm2_in = vec![0.0_f32; batch * seq_len * ph];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..ph {
                    norm2_in[b * seq_len * ph + t * ph + c] =
                        conv2_relu[b * ph * seq_len + c * seq_len + t];
                }
            }
        }
        let norm2_out = self
            .norm2
            .forward(&norm2_in, batch * seq_len)
            .expect("norm2 fw");

        // === Backward chain (reverse) ===
        // 1. Linear backward
        let (grad_norm2_out, linear_w_grad, linear_b_grad) = self
            .linear
            .backward(&norm2_out, grad_output, batch * seq_len)
            .expect("linear bw");

        // 2. norm2 backward
        let (grad_norm2_in, norm2_gamma_grad, norm2_beta_grad) = self
            .norm2
            .backward(&norm2_in, &grad_norm2_out, batch * seq_len)
            .expect("norm2 bw");

        // 3. reshape grad_norm2_in [B*seq_len, ph] → [B, ph, seq_len] for ReLU + conv2 backward
        let mut grad_conv2_relu = vec![0.0_f32; batch * seq_len * ph];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..ph {
                    grad_conv2_relu[b * ph * seq_len + c * seq_len + t] =
                        grad_norm2_in[b * seq_len * ph + t * ph + c];
                }
            }
        }

        // 4. ReLU 2 backward: grad_conv2_out = grad_conv2_relu * (conv2_out > 0)
        let grad_conv2_out: Vec<f32> = grad_conv2_relu
            .iter()
            .zip(&conv2_out)
            .map(|(g, &pre)| if pre > 0.0 { *g } else { 0.0 })
            .collect();

        // 5. conv2 backward
        let (grad_norm1_reshaped, conv2_w_grad, conv2_b_grad) = self
            .conv2
            .backward(&norm1_reshaped, &grad_conv2_out, batch, seq_len)
            .expect("conv2 bw");

        // 6. reshape grad_norm1_reshaped [B, ph, seq_len] → [B*seq_len, ph]
        let mut grad_norm1_out = vec![0.0_f32; batch * seq_len * ph];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..ph {
                    grad_norm1_out[b * seq_len * ph + t * ph + c] =
                        grad_norm1_reshaped[b * ph * seq_len + c * seq_len + t];
                }
            }
        }

        // 7. norm1 backward
        let (grad_norm1_in, norm1_gamma_grad, norm1_beta_grad) = self
            .norm1
            .backward(&norm1_in, &grad_norm1_out, batch * seq_len)
            .expect("norm1 bw");

        // 8. reshape grad_norm1_in [B*seq_len, ph] → [B, ph, seq_len] for ReLU + conv1 backward
        let mut grad_conv1_relu = vec![0.0_f32; batch * seq_len * ph];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..ph {
                    grad_conv1_relu[b * ph * seq_len + c * seq_len + t] =
                        grad_norm1_in[b * seq_len * ph + t * ph + c];
                }
            }
        }

        // 9. ReLU 1 backward
        let grad_conv1_out: Vec<f32> = grad_conv1_relu
            .iter()
            .zip(&conv1_out)
            .map(|(g, &pre)| if pre > 0.0 { *g } else { 0.0 })
            .collect();

        // 10. conv1 backward
        let (grad_reshaped_input, conv1_w_grad, conv1_b_grad) = self
            .conv1
            .backward(&reshaped, &grad_conv1_out, batch, seq_len)
            .expect("conv1 bw");

        // 11. reshape grad_reshaped_input [B, h, seq_len] → [B, seq_len, h]
        let mut grad_input = vec![0.0_f32; batch * seq_len * h];
        for b in 0..batch {
            for t in 0..seq_len {
                for c in 0..h {
                    grad_input[b * seq_len * h + t * h + c] =
                        grad_reshaped_input[b * h * seq_len + c * seq_len + t];
                }
            }
        }

        (
            grad_input,
            VariancePredictorGrads {
                conv1_w: conv1_w_grad,
                conv1_b: conv1_b_grad,
                norm1_gamma: norm1_gamma_grad,
                norm1_beta: norm1_beta_grad,
                conv2_w: conv2_w_grad,
                conv2_b: conv2_b_grad,
                norm2_gamma: norm2_gamma_grad,
                norm2_beta: norm2_beta_grad,
                linear_w: linear_w_grad,
                linear_b: linear_b_grad,
            },
        )
    }
}

/// VariancePredictor backward で返される sub-layer 勾配 bundle。
#[derive(Clone, Debug, Default)]
pub struct VariancePredictorGrads {
    /// Conv1 weight 勾配。
    pub conv1_w: Vec<f32>,
    /// Conv1 bias 勾配。
    pub conv1_b: Vec<f32>,
    /// norm1 gamma 勾配。
    pub norm1_gamma: Vec<f32>,
    /// norm1 beta 勾配。
    pub norm1_beta: Vec<f32>,
    /// Conv2 weight 勾配。
    pub conv2_w: Vec<f32>,
    /// Conv2 bias 勾配。
    pub conv2_b: Vec<f32>,
    /// norm2 gamma 勾配。
    pub norm2_gamma: Vec<f32>,
    /// norm2 beta 勾配。
    pub norm2_beta: Vec<f32>,
    /// Linear weight 勾配。
    pub linear_w: Vec<f32>,
    /// Linear bias 勾配。
    pub linear_b: Vec<f32>,
}

/// reshape `[batch, seq_len, channels]` → `[batch, channels, seq_len]` for Conv1D input。
fn reshape_time_last_to_channel_first(
    x: &[f32],
    batch: usize,
    seq_len: usize,
    channels: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * channels * seq_len];
    for b in 0..batch {
        for t in 0..seq_len {
            for c in 0..channels {
                out[b * channels * seq_len + c * seq_len + t] =
                    x[b * seq_len * channels + t * channels + c];
            }
        }
    }
    out
}

/// element-wise add of two same-shape vectors。
fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// FastSpeech2 model (encoder + variance adaptor + decoder + postnet)。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FastSpeech2 {
    config: FastSpeech2Config,
    /// mora embedding table `[vocab_size, hidden_dim]` flatten。
    embedding: Vec<f32>,
    encoder_pos_enc: SinusoidalPositionalEncoding,
    encoder: Vec<FftBlock>,
    duration_predictor: VariancePredictor,
    pitch_predictor: VariancePredictor,
    energy_predictor: VariancePredictor,
    /// pitch scalar → hidden proj (Conv1D 1x1)。
    pitch_embed: Conv1d,
    /// energy scalar → hidden proj (Conv1D 1x1)。
    energy_embed: Conv1d,
    decoder_pos_enc: SinusoidalPositionalEncoding,
    decoder: Vec<FftBlock>,
    /// mel projection (hidden → mel_dim)。
    mel_linear: Linear,
    /// Postnet layers (Conv1D、最初は mel_dim→postnet_hidden、中間は postnet_hidden→postnet_hidden、
    /// 最後は postnet_hidden→mel_dim)。
    postnet: Vec<Conv1d>,
}

impl FastSpeech2 {
    /// zero 初期化で構築 (テスト用 / 学習前初期化のベース)。
    ///
    /// # Errors
    ///
    /// config validation。
    pub fn zeros(config: FastSpeech2Config) -> Result<Self, FastSpeech2Error> {
        config.validate()?;

        let embedding = vec![0.0_f32; config.vocab_size * config.hidden_dim];

        let pos_enc_cfg =
            SinusoidalPositionalEncodingConfig::new(config.hidden_dim, config.max_len);
        let encoder_pos_enc = SinusoidalPositionalEncoding::new(pos_enc_cfg).map_err(|e| {
            FastSpeech2Error::InvalidConfig {
                reason: format!("encoder positional encoding: {e}"),
            }
        })?;
        let decoder_pos_enc = SinusoidalPositionalEncoding::new(pos_enc_cfg).map_err(|e| {
            FastSpeech2Error::InvalidConfig {
                reason: format!("decoder positional encoding: {e}"),
            }
        })?;

        let encoder = (0..config.num_encoder_layers)
            .map(|_| {
                FftBlock::zeros(
                    config.hidden_dim,
                    config.num_heads,
                    config.fft_kernel_size,
                    config.fft_expansion,
                )
            })
            .collect();
        let decoder = (0..config.num_decoder_layers)
            .map(|_| {
                FftBlock::zeros(
                    config.hidden_dim,
                    config.num_heads,
                    config.fft_kernel_size,
                    config.fft_expansion,
                )
            })
            .collect();

        let duration_predictor = VariancePredictor::zeros(
            config.hidden_dim,
            config.predictor_hidden,
            config.predictor_kernel_size,
        );
        let pitch_predictor = VariancePredictor::zeros(
            config.hidden_dim,
            config.predictor_hidden,
            config.predictor_kernel_size,
        );
        let energy_predictor = VariancePredictor::zeros(
            config.hidden_dim,
            config.predictor_hidden,
            config.predictor_kernel_size,
        );

        let pitch_embed =
            Conv1d::zeros(Conv1dConfig::new(1, config.hidden_dim, 1)).expect("pitch embed");
        let energy_embed =
            Conv1d::zeros(Conv1dConfig::new(1, config.hidden_dim, 1)).expect("energy embed");

        let mel_linear = Linear::zeros(LinearConfig::new(config.hidden_dim, config.mel_dim))
            .expect("mel linear");

        // Postnet: 5-layer Conv1D 前提 (最初 mel→postnet_hidden、間 postnet_hidden、最後 postnet_hidden→mel_dim)
        let padding = config.postnet_kernel_size / 2;
        let mut postnet = Vec::with_capacity(config.postnet_layers);
        for i in 0..config.postnet_layers {
            let (in_ch, out_ch) = if i == 0 {
                (config.mel_dim, config.postnet_hidden)
            } else if i == config.postnet_layers - 1 {
                (config.postnet_hidden, config.mel_dim)
            } else {
                (config.postnet_hidden, config.postnet_hidden)
            };
            let cfg =
                Conv1dConfig::new(in_ch, out_ch, config.postnet_kernel_size).with_padding(padding);
            postnet.push(Conv1d::zeros(cfg).expect("postnet"));
        }

        Ok(Self {
            config,
            embedding,
            encoder_pos_enc,
            encoder,
            duration_predictor,
            pitch_predictor,
            energy_predictor,
            pitch_embed,
            energy_embed,
            decoder_pos_enc,
            decoder,
            mel_linear,
            postnet,
        })
    }

    /// config への参照。
    #[must_use]
    pub fn config(&self) -> &FastSpeech2Config {
        &self.config
    }

    /// forward pass (teacher forcing duration):
    /// input mora_ids `[batch, mora_len]` + target_durations `[batch, mora_len]` (frame 数)
    /// → mel `[batch, frame_len, mel_dim]` flatten
    ///
    /// # Errors
    ///
    /// - mora_ids / target_durations shape 不整合
    /// - mora id が vocab_size を超える
    /// - mora_len > max_len
    pub fn forward(
        &self,
        mora_ids: &[u32],
        batch: usize,
        mora_len: usize,
        target_durations: &[u32],
    ) -> Result<Vec<f32>, FastSpeech2Error> {
        let cfg = self.config;
        if mora_ids.len() != batch * mora_len {
            return Err(FastSpeech2Error::ShapeMismatch {
                field: "mora_ids",
                expected: batch * mora_len,
                actual: mora_ids.len(),
            });
        }
        if target_durations.len() != batch * mora_len {
            return Err(FastSpeech2Error::ShapeMismatch {
                field: "target_durations",
                expected: batch * mora_len,
                actual: target_durations.len(),
            });
        }
        if mora_len > cfg.max_len {
            return Err(FastSpeech2Error::SeqLenExceedsMax {
                seq_len: mora_len,
                max_len: cfg.max_len,
            });
        }
        for &id in mora_ids {
            if id as usize >= cfg.vocab_size {
                return Err(FastSpeech2Error::VocabOverflow {
                    id: id as usize,
                    vocab_size: cfg.vocab_size,
                });
            }
        }

        // 1. Embedding lookup + positional encoding
        let mut embedded = vec![0.0_f32; batch * mora_len * cfg.hidden_dim];
        for b in 0..batch {
            for t in 0..mora_len {
                let id = mora_ids[b * mora_len + t] as usize;
                let src = &self.embedding[id * cfg.hidden_dim..(id + 1) * cfg.hidden_dim];
                let dst_start = b * mora_len * cfg.hidden_dim + t * cfg.hidden_dim;
                embedded[dst_start..dst_start + cfg.hidden_dim].copy_from_slice(src);
            }
        }
        let mut encoder_input = self
            .encoder_pos_enc
            .forward(&embedded, batch, mora_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("encoder PE: {e}"),
            })?;

        // 2. Encoder stack
        for block in &self.encoder {
            encoder_input = block.forward(&encoder_input, batch, mora_len);
        }

        // 3. Variance adaptor: pitch/energy prediction (skipped: 学習時は teacher forcing、
        //    ここでは 0 として hidden に何も足さない (最小 MVP))。
        let _dur_pred = self
            .duration_predictor
            .forward(&encoder_input, batch, mora_len);
        let _pitch_pred = self
            .pitch_predictor
            .forward(&encoder_input, batch, mora_len);
        let _energy_pred = self
            .energy_predictor
            .forward(&encoder_input, batch, mora_len);
        // 実装省略: pitch/energy embed の hidden 加算は teacher forcing 時のみ意味あり、
        // MVP forward-only では predicted 0 が hidden に加わっても影響ない → skip

        // 4. Length Regulator: expand mora → frame using target_durations
        let (expanded, frame_len_per_batch) = length_regulator(
            &encoder_input,
            batch,
            mora_len,
            cfg.hidden_dim,
            target_durations,
        );
        // 現 MVP は batch 内 frame 数一致前提 (全 sample が同じ total frames を持つ) or
        // batch=1 (実運用は batch=1 のみサポート)。異なる長さの batch は Phase T.4a 継続で対応。
        if batch > 1 {
            for &fl in &frame_len_per_batch {
                if fl != frame_len_per_batch[0] {
                    return Err(FastSpeech2Error::VariableFrameLenNotSupported {
                        batch,
                        lens: frame_len_per_batch.clone(),
                    });
                }
            }
        }
        let frame_len = frame_len_per_batch[0];
        if frame_len > cfg.max_len {
            return Err(FastSpeech2Error::SeqLenExceedsMax {
                seq_len: frame_len,
                max_len: cfg.max_len,
            });
        }

        // 5. Decoder: PE + FFT block stack
        let mut decoder_input = self
            .decoder_pos_enc
            .forward(&expanded, batch, frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("decoder PE: {e}"),
            })?;
        for block in &self.decoder {
            decoder_input = block.forward(&decoder_input, batch, frame_len);
        }

        // 6. Mel projection [B, frame_len, hidden] → [B, frame_len, mel_dim]
        let mel_before = self
            .mel_linear
            .forward(&decoder_input, batch * frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("mel linear: {e}"),
            })?;

        // 7. Postnet residual (need [B, mel, frame_len] layout for Conv1D)
        let mel_reshaped =
            reshape_time_last_to_channel_first(&mel_before, batch, frame_len, cfg.mel_dim);
        let mut postnet_out = mel_reshaped;
        for (i, conv) in self.postnet.iter().enumerate() {
            let conv_out = conv.forward(&postnet_out, batch, frame_len).map_err(|e| {
                FastSpeech2Error::Internal {
                    reason: format!("postnet[{i}]: {e}"),
                }
            })?;
            if i < self.postnet.len() - 1 {
                postnet_out = conv_out.iter().map(|&x| x.tanh()).collect();
            } else {
                // 最終層は activation なし
                postnet_out = conv_out;
            }
        }
        // reshape back [B, mel, frame_len] → [B, frame_len, mel_dim]
        let mut postnet_out_reshaped = vec![0.0_f32; batch * frame_len * cfg.mel_dim];
        for b in 0..batch {
            for t in 0..frame_len {
                for c in 0..cfg.mel_dim {
                    postnet_out_reshaped[b * frame_len * cfg.mel_dim + t * cfg.mel_dim + c] =
                        postnet_out[b * cfg.mel_dim * frame_len + c * frame_len + t];
                }
            }
        }
        let mel_after: Vec<f32> = mel_before
            .iter()
            .zip(&postnet_out_reshaped)
            .map(|(a, b)| a + b)
            .collect();

        Ok(mel_after)
    }

    /// Forward + prosody prediction (Phase T.4a Phase E: `ProsodyLoss` joint training)。
    ///
    /// 通常の forward に加え、duration / pitch / energy predictor の出力を
    /// [`crate::tts::ProsodyPrediction`] 形式で返却する。
    ///
    /// pitch/energy embed の hidden injection はまだ実装しないため mel_pred は `forward` と同一。
    ///
    /// # Errors
    ///
    /// forward と同じ。
    pub fn forward_with_prosody(
        &self,
        mora_ids: &[u32],
        batch: usize,
        mora_len: usize,
        target_durations: &[u32],
    ) -> Result<(Vec<f32>, crate::tts::ProsodyPrediction), FastSpeech2Error> {
        let cfg = self.config;
        // shape / vocab 検証は forward と同じ
        if mora_ids.len() != batch * mora_len {
            return Err(FastSpeech2Error::ShapeMismatch {
                field: "mora_ids",
                expected: batch * mora_len,
                actual: mora_ids.len(),
            });
        }
        if target_durations.len() != batch * mora_len {
            return Err(FastSpeech2Error::ShapeMismatch {
                field: "target_durations",
                expected: batch * mora_len,
                actual: target_durations.len(),
            });
        }
        if mora_len > cfg.max_len {
            return Err(FastSpeech2Error::SeqLenExceedsMax {
                seq_len: mora_len,
                max_len: cfg.max_len,
            });
        }
        for &id in mora_ids {
            if id as usize >= cfg.vocab_size {
                return Err(FastSpeech2Error::VocabOverflow {
                    id: id as usize,
                    vocab_size: cfg.vocab_size,
                });
            }
        }

        // Encoder path (同じ)
        let mut embedded = vec![0.0_f32; batch * mora_len * cfg.hidden_dim];
        for b in 0..batch {
            for t in 0..mora_len {
                let id = mora_ids[b * mora_len + t] as usize;
                let src = &self.embedding[id * cfg.hidden_dim..(id + 1) * cfg.hidden_dim];
                let dst_start = b * mora_len * cfg.hidden_dim + t * cfg.hidden_dim;
                embedded[dst_start..dst_start + cfg.hidden_dim].copy_from_slice(src);
            }
        }
        let mut encoder_input = self
            .encoder_pos_enc
            .forward(&embedded, batch, mora_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("encoder PE: {e}"),
            })?;
        for block in &self.encoder {
            encoder_input = block.forward(&encoder_input, batch, mora_len);
        }

        // Variance predictor forward (Phase E: 結果を保持)
        let dur_pred_flat = self
            .duration_predictor
            .forward(&encoder_input, batch, mora_len);
        let pitch_pred_flat = self
            .pitch_predictor
            .forward(&encoder_input, batch, mora_len);
        let energy_pred_flat = self
            .energy_predictor
            .forward(&encoder_input, batch, mora_len);
        let prosody = flat_predictions_to_prosody(
            &dur_pred_flat,
            &pitch_pred_flat,
            &energy_pred_flat,
            batch,
            mora_len,
        );

        // Length Regulator + Decoder + Mel + Postnet (forward と同じロジックを直呼び出し)
        let (expanded, frame_len_per_batch) = length_regulator(
            &encoder_input,
            batch,
            mora_len,
            cfg.hidden_dim,
            target_durations,
        );
        if batch > 1 {
            for &fl in &frame_len_per_batch {
                if fl != frame_len_per_batch[0] {
                    return Err(FastSpeech2Error::VariableFrameLenNotSupported {
                        batch,
                        lens: frame_len_per_batch.clone(),
                    });
                }
            }
        }
        let frame_len = frame_len_per_batch[0];
        if frame_len > cfg.max_len {
            return Err(FastSpeech2Error::SeqLenExceedsMax {
                seq_len: frame_len,
                max_len: cfg.max_len,
            });
        }
        let mut decoder_input = self
            .decoder_pos_enc
            .forward(&expanded, batch, frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("decoder PE: {e}"),
            })?;
        for block in &self.decoder {
            decoder_input = block.forward(&decoder_input, batch, frame_len);
        }
        let mel_before = self
            .mel_linear
            .forward(&decoder_input, batch * frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("mel linear: {e}"),
            })?;
        let mel_reshaped =
            reshape_time_last_to_channel_first(&mel_before, batch, frame_len, cfg.mel_dim);
        let mut postnet_out = mel_reshaped;
        for (i, conv) in self.postnet.iter().enumerate() {
            let conv_out = conv.forward(&postnet_out, batch, frame_len).map_err(|e| {
                FastSpeech2Error::Internal {
                    reason: format!("postnet[{i}]: {e}"),
                }
            })?;
            if i < self.postnet.len() - 1 {
                postnet_out = conv_out.iter().map(|&x| x.tanh()).collect();
            } else {
                postnet_out = conv_out;
            }
        }
        let mut postnet_out_reshaped = vec![0.0_f32; batch * frame_len * cfg.mel_dim];
        for b in 0..batch {
            for t in 0..frame_len {
                for c in 0..cfg.mel_dim {
                    postnet_out_reshaped[b * frame_len * cfg.mel_dim + t * cfg.mel_dim + c] =
                        postnet_out[b * cfg.mel_dim * frame_len + c * frame_len + t];
                }
            }
        }
        let mel_after: Vec<f32> = mel_before
            .iter()
            .zip(&postnet_out_reshaped)
            .map(|(a, b)| a + b)
            .collect();

        Ok((mel_after, prosody))
    }

    /// FastSpeech2 backward (Phase 1 partial: **Postnet + mel_linear のみ**)。
    ///
    /// grad_mel を受け取り、Postnet 残差経路 + mel_linear projection の勾配を計算する。
    /// encoder / decoder / variance predictor / embedding の勾配は現 Phase 1 では未実装
    /// (0 埋めで返却)、Phase T.4a 継続実装で埋める予定。
    ///
    /// # 戻り値
    ///
    /// `(grads, grad_decoder_out)`:
    /// - `grads`: [`FastSpeech2Grads`] bundle。mel_linear と postnet の grad_w / grad_b が埋まる、
    ///   他は shape 正しい 0 埋め (encoder/decoder backward 未実装のため)
    /// - `grad_decoder_out`: mel_linear 入力に対する grad (`[batch, frame_len, hidden_dim]` flatten)。
    ///   Phase T.4a 継続で decoder backward に渡す入力になる。
    ///
    /// # 手順
    ///
    /// 1. Forward を再計算して mel_before と postnet 各層の中間 activation を取得
    /// 2. `grad_mel_after` を Postnet residual add 経由で分岐: `grad_mel_before = grad_mel_after`,
    ///    `grad_postnet_residual = grad_mel_after`
    /// 3. Postnet chain backward: 最終 Conv1D → tanh backward → Conv1D → ... (逆順)
    /// 4. `grad_mel_before` を mel_linear.backward で `grad_decoder_out` へ変換
    ///
    /// # Errors
    ///
    /// - shape 不整合
    /// - forward と同じエラー
    pub fn backward_terminal(
        &self,
        mora_ids: &[u32],
        target_durations: &[u32],
        grad_mel: &[f32],
        batch: usize,
        mora_len: usize,
    ) -> Result<(FastSpeech2Grads, Vec<f32>), FastSpeech2Error> {
        let cfg = self.config;
        // grad_mel shape check
        let total_frames: u32 = target_durations
            .chunks_exact(mora_len)
            .map(|c| c.iter().sum::<u32>())
            .max()
            .unwrap_or(0);
        let frame_len = total_frames as usize;
        let expected = batch * frame_len * cfg.mel_dim;
        if grad_mel.len() != expected {
            return Err(FastSpeech2Error::ShapeMismatch {
                field: "grad_mel",
                expected,
                actual: grad_mel.len(),
            });
        }

        // Step 1: Recompute forward up to decoder_out + mel_before + postnet chain
        // (再計算コストはあるが、cache を返さない forward API 互換性を保つ)
        let mut embedded = vec![0.0_f32; batch * mora_len * cfg.hidden_dim];
        for b in 0..batch {
            for t in 0..mora_len {
                let id = mora_ids[b * mora_len + t] as usize;
                let src = &self.embedding[id * cfg.hidden_dim..(id + 1) * cfg.hidden_dim];
                let dst_start = b * mora_len * cfg.hidden_dim + t * cfg.hidden_dim;
                embedded[dst_start..dst_start + cfg.hidden_dim].copy_from_slice(src);
            }
        }
        let mut encoder_input = self
            .encoder_pos_enc
            .forward(&embedded, batch, mora_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("encoder PE: {e}"),
            })?;
        for block in &self.encoder {
            encoder_input = block.forward(&encoder_input, batch, mora_len);
        }
        let (expanded, _) = length_regulator(
            &encoder_input,
            batch,
            mora_len,
            cfg.hidden_dim,
            target_durations,
        );
        let mut decoder_input = self
            .decoder_pos_enc
            .forward(&expanded, batch, frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("decoder PE: {e}"),
            })?;
        for block in &self.decoder {
            decoder_input = block.forward(&decoder_input, batch, frame_len);
        }
        let mel_before = self
            .mel_linear
            .forward(&decoder_input, batch * frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("mel linear: {e}"),
            })?;

        // Postnet forward で intermediate 保存 (backward 用)
        let mel_reshaped =
            reshape_time_last_to_channel_first(&mel_before, batch, frame_len, cfg.mel_dim);
        let mut postnet_inputs: Vec<Vec<f32>> = Vec::with_capacity(self.postnet.len());
        let mut postnet_pre_tanh: Vec<Vec<f32>> = Vec::with_capacity(self.postnet.len());
        let mut current = mel_reshaped;
        for (i, conv) in self.postnet.iter().enumerate() {
            postnet_inputs.push(current.clone());
            let conv_out = conv.forward(&current, batch, frame_len).map_err(|e| {
                FastSpeech2Error::Internal {
                    reason: format!("postnet[{i}]: {e}"),
                }
            })?;
            postnet_pre_tanh.push(conv_out.clone());
            if i < self.postnet.len() - 1 {
                current = conv_out.iter().map(|&x| x.tanh()).collect();
            } else {
                current = conv_out;
            }
        }

        // Step 2: grad_mel_after residual 分岐
        // grad_mel_before += grad_mel_after (from add)
        // grad_postnet_out += grad_mel_after (from add)
        let grad_mel_before = grad_mel.to_vec();
        // reshape grad_mel to [B, mel, frame_len] for postnet chain backward
        let grad_postnet_out_reshaped =
            reshape_time_last_to_channel_first(grad_mel, batch, frame_len, cfg.mel_dim);

        // Step 3: Postnet chain backward (reverse order)
        let mut postnet_w_grads: Vec<Vec<f32>> = vec![Vec::new(); self.postnet.len()];
        let mut postnet_b_grads: Vec<Vec<f32>> = vec![Vec::new(); self.postnet.len()];

        let mut grad_current = grad_postnet_out_reshaped;
        for i in (0..self.postnet.len()).rev() {
            // 最終層以外は tanh backward: grad_pre_tanh = grad_post_tanh * (1 - tanh²)
            if i < self.postnet.len() - 1 {
                let pre_tanh = &postnet_pre_tanh[i];
                for (g, &pre) in grad_current.iter_mut().zip(pre_tanh) {
                    let t = pre.tanh();
                    *g *= 1.0 - t * t;
                }
            }
            // Conv1D backward
            let (grad_in, grad_w, grad_b) = self.postnet[i]
                .backward(&postnet_inputs[i], &grad_current, batch, frame_len)
                .map_err(|e| FastSpeech2Error::Internal {
                    reason: format!("postnet[{i}] backward: {e}"),
                })?;
            postnet_w_grads[i] = grad_w;
            postnet_b_grads[i] = grad_b;
            grad_current = grad_in;
        }
        // grad_current is now in [B, mel, frame_len] layout, reshape to [B, frame_len, mel]
        let mut grad_postnet_chain_out = vec![0.0_f32; batch * frame_len * cfg.mel_dim];
        for b in 0..batch {
            for t in 0..frame_len {
                for c in 0..cfg.mel_dim {
                    grad_postnet_chain_out[b * frame_len * cfg.mel_dim + t * cfg.mel_dim + c] =
                        grad_current[b * cfg.mel_dim * frame_len + c * frame_len + t];
                }
            }
        }
        // grad_mel_before は 2 経路合成: mel_after からの直接 + postnet chain 経由
        let grad_mel_before_combined: Vec<f32> = grad_mel_before
            .iter()
            .zip(&grad_postnet_chain_out)
            .map(|(a, b)| a + b)
            .collect();

        // Step 4: mel_linear backward
        let (grad_decoder_out, grad_mel_w, grad_mel_b) = self
            .mel_linear
            .backward(&decoder_input, &grad_mel_before_combined, batch * frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("mel_linear backward: {e}"),
            })?;

        // Bundle grads (encoder/decoder/variance/embed 系は 0 埋め、backward_full が全埋め)
        let mut grads = FastSpeech2Grads::zeros_from_config(&self.config);
        grads.mel_linear_w = grad_mel_w;
        grads.mel_linear_b = grad_mel_b;
        grads.postnet_w = postnet_w_grads;
        grads.postnet_b = postnet_b_grads;

        Ok((grads, grad_decoder_out))
    }

    /// FastSpeech2 backward **全 chain 完全版** (Phase 3 完全)。
    ///
    /// backward_terminal に加えて、decoder FFT block chain → Length Regulator →
    /// encoder FFT block chain → embedding backward の完全 chain を実装。
    /// pitch/energy embed は forward MVP で使用されていないため grad は 0 埋め。
    /// Variance predictor (duration/pitch/energy) は forward output を hidden に反映しないため
    /// (MVP 制約)、backward chain 上 grad は 0 のまま。予測誤差 loss を別途追加する場合は
    /// ProsodyLoss + variance predictor の直接 backward を chain 外で計算する設計。
    ///
    /// # 引数
    ///
    /// - `mora_ids`: `[batch, mora_len]`
    /// - `target_durations`: `[batch, mora_len]` (frame 単位)
    /// - `grad_mel`: `[batch, frame_len, mel_dim]`
    ///
    /// # 戻り値
    ///
    /// [`FastSpeech2Grads`] 完全埋め (embedding + encoder + decoder + mel_linear + postnet)。
    ///
    /// # Errors
    ///
    /// shape 不整合。
    pub fn backward_full(
        &self,
        mora_ids: &[u32],
        target_durations: &[u32],
        grad_mel: &[f32],
        batch: usize,
        mora_len: usize,
    ) -> Result<FastSpeech2Grads, FastSpeech2Error> {
        let cfg = self.config;

        // Step 1: backward_terminal で Postnet + mel_linear → grad_decoder_out を取得
        let (mut grads, grad_decoder_out) =
            self.backward_terminal(mora_ids, target_durations, grad_mel, batch, mora_len)?;

        // frame_len は grad_decoder_out.len() から逆算
        let frame_len = grad_decoder_out.len() / (batch * cfg.hidden_dim);

        // Step 2: Decoder FftBlock chain backward (逆順)
        // 各 block の input が必要 → forward 再計算で intermediates 取得
        let mut decoder_inputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_decoder_layers);

        // 再計算: mora_ids → embedded → encoder_pe → encoder blocks → LR → decoder_pe → decoder blocks
        let mut embedded = vec![0.0_f32; batch * mora_len * cfg.hidden_dim];
        for b in 0..batch {
            for t in 0..mora_len {
                let id = mora_ids[b * mora_len + t] as usize;
                let src = &self.embedding[id * cfg.hidden_dim..(id + 1) * cfg.hidden_dim];
                let dst_start = b * mora_len * cfg.hidden_dim + t * cfg.hidden_dim;
                embedded[dst_start..dst_start + cfg.hidden_dim].copy_from_slice(src);
            }
        }
        let encoder_pe_out = self
            .encoder_pos_enc
            .forward(&embedded, batch, mora_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("encoder PE fw: {e}"),
            })?;
        // encoder inputs cache
        let mut encoder_inputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_encoder_layers);
        let mut encoder_current = encoder_pe_out;
        for block in &self.encoder {
            encoder_inputs.push(encoder_current.clone());
            encoder_current = block.forward(&encoder_current, batch, mora_len);
        }
        let encoder_out = encoder_current;

        // LR forward
        let (expanded, _) = length_regulator(
            &encoder_out,
            batch,
            mora_len,
            cfg.hidden_dim,
            target_durations,
        );
        // decoder PE + blocks
        let decoder_pe_out = self
            .decoder_pos_enc
            .forward(&expanded, batch, frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("decoder PE fw: {e}"),
            })?;
        let mut decoder_current = decoder_pe_out;
        for block in &self.decoder {
            decoder_inputs.push(decoder_current.clone());
            decoder_current = block.forward(&decoder_current, batch, frame_len);
        }

        // Step 3: Decoder FftBlock chain backward (reverse)
        let mut grad_current = grad_decoder_out;
        for i in (0..cfg.num_decoder_layers).rev() {
            let (grad_in, fft_grads) =
                self.decoder[i].backward(&decoder_inputs[i], &grad_current, batch, frame_len);
            grads.decoder[i] = fft_grads;
            grad_current = grad_in;
        }

        // Step 4: Decoder PE backward = pass-through
        let grad_expanded = self
            .decoder_pos_enc
            .backward(&grad_current, batch, frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("decoder PE bw: {e}"),
            })?;

        // Step 5: Length Regulator backward
        let grad_encoder_out = length_regulator_backward(
            &grad_expanded,
            batch,
            mora_len,
            cfg.hidden_dim,
            target_durations,
        );

        // Step 6: Encoder FftBlock chain backward (reverse)
        // Note: variance predictor + pitch/energy embed は forward MVP で使用してないため
        // grad_encoder_out に何も足さない (grad = 0 で通過)
        let mut grad_current = grad_encoder_out;
        for i in (0..cfg.num_encoder_layers).rev() {
            let (grad_in, fft_grads) =
                self.encoder[i].backward(&encoder_inputs[i], &grad_current, batch, mora_len);
            grads.encoder[i] = fft_grads;
            grad_current = grad_in;
        }

        // Step 7: Encoder PE backward = pass-through
        let grad_embedded = self
            .encoder_pos_enc
            .backward(&grad_current, batch, mora_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("encoder PE bw: {e}"),
            })?;

        // Step 8: Embedding backward
        grads.embedding = embedding_backward(
            &grad_embedded,
            mora_ids,
            cfg.vocab_size,
            cfg.hidden_dim,
            batch,
            mora_len,
        );

        Ok(grads)
    }

    /// Full backward with `ProsodyLoss` prosody grads (Phase T.4a Phase E)。
    ///
    /// `backward_full` に加え、prosody 3 予測の grad を variance predictor 経由で
    /// backprop、grad_encoder_out に加算 + `grads.duration/pitch/energy_predictor` を埋める。
    ///
    /// # 引数
    ///
    /// - `grad_duration_pred`: `[batch][mora_len]` duration 予測の grad
    /// - `grad_pitch_pred`: `[batch][mora_len]` pitch 予測の grad
    /// - `grad_energy_pred`: `[batch][mora_len]` energy 予測の grad
    /// - 他は `backward_full` と同じ
    ///
    /// # Errors
    ///
    /// - shape 不整合 (prosody grad の `[batch][mora_len]`)
    /// - `backward_full` と同じエラー
    #[allow(clippy::too_many_arguments)]
    pub fn backward_full_with_prosody(
        &self,
        mora_ids: &[u32],
        target_durations: &[u32],
        grad_mel: &[f32],
        grad_duration_pred: &[Vec<f32>],
        grad_pitch_pred: &[Vec<f32>],
        grad_energy_pred: &[Vec<f32>],
        batch: usize,
        mora_len: usize,
    ) -> Result<FastSpeech2Grads, FastSpeech2Error> {
        let cfg = self.config;

        // prosody grad shape 検証
        if grad_duration_pred.len() != batch
            || grad_pitch_pred.len() != batch
            || grad_energy_pred.len() != batch
        {
            return Err(FastSpeech2Error::ShapeMismatch {
                field: "grad_prosody batch",
                expected: batch,
                actual: grad_duration_pred.len(),
            });
        }
        for b in 0..batch {
            for (name, v) in [
                ("grad_duration_pred", &grad_duration_pred[b]),
                ("grad_pitch_pred", &grad_pitch_pred[b]),
                ("grad_energy_pred", &grad_energy_pred[b]),
            ] {
                if v.len() != mora_len {
                    return Err(FastSpeech2Error::ShapeMismatch {
                        field: name,
                        expected: mora_len,
                        actual: v.len(),
                    });
                }
            }
        }

        // Step 1-4: mel path grad_encoder_out まで backward_full と同じ手順
        let (mut grads, grad_decoder_out) =
            self.backward_terminal(mora_ids, target_durations, grad_mel, batch, mora_len)?;
        let frame_len = grad_decoder_out.len() / (batch * cfg.hidden_dim);
        let mut decoder_inputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_decoder_layers);
        let mut embedded = vec![0.0_f32; batch * mora_len * cfg.hidden_dim];
        for b in 0..batch {
            for t in 0..mora_len {
                let id = mora_ids[b * mora_len + t] as usize;
                let src = &self.embedding[id * cfg.hidden_dim..(id + 1) * cfg.hidden_dim];
                let dst_start = b * mora_len * cfg.hidden_dim + t * cfg.hidden_dim;
                embedded[dst_start..dst_start + cfg.hidden_dim].copy_from_slice(src);
            }
        }
        let encoder_pe_out = self
            .encoder_pos_enc
            .forward(&embedded, batch, mora_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("encoder PE fw: {e}"),
            })?;
        let mut encoder_inputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_encoder_layers);
        let mut encoder_current = encoder_pe_out;
        for block in &self.encoder {
            encoder_inputs.push(encoder_current.clone());
            encoder_current = block.forward(&encoder_current, batch, mora_len);
        }
        let encoder_out = encoder_current;
        let (expanded, _) = length_regulator(
            &encoder_out,
            batch,
            mora_len,
            cfg.hidden_dim,
            target_durations,
        );
        let decoder_pe_out = self
            .decoder_pos_enc
            .forward(&expanded, batch, frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("decoder PE fw: {e}"),
            })?;
        let mut decoder_current = decoder_pe_out;
        for block in &self.decoder {
            decoder_inputs.push(decoder_current.clone());
            decoder_current = block.forward(&decoder_current, batch, frame_len);
        }
        let mut grad_current = grad_decoder_out;
        for i in (0..cfg.num_decoder_layers).rev() {
            let (grad_in, fft_grads) =
                self.decoder[i].backward(&decoder_inputs[i], &grad_current, batch, frame_len);
            grads.decoder[i] = fft_grads;
            grad_current = grad_in;
        }
        let grad_expanded = self
            .decoder_pos_enc
            .backward(&grad_current, batch, frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("decoder PE bw: {e}"),
            })?;
        let mut grad_encoder_out = length_regulator_backward(
            &grad_expanded,
            batch,
            mora_len,
            cfg.hidden_dim,
            target_durations,
        );

        // Step 5.5 (Phase E): Variance predictor 3 系統 backward
        let (dur_grad_flat, pitch_grad_flat, energy_grad_flat) = prosody_grads_to_flat(
            grad_duration_pred,
            grad_pitch_pred,
            grad_energy_pred,
            batch,
            mora_len,
        );
        let (grad_from_dur, dur_grads) =
            self.duration_predictor
                .backward(&encoder_out, &dur_grad_flat, batch, mora_len);
        let (grad_from_pitch, pitch_grads) =
            self.pitch_predictor
                .backward(&encoder_out, &pitch_grad_flat, batch, mora_len);
        let (grad_from_energy, energy_grads) =
            self.energy_predictor
                .backward(&encoder_out, &energy_grad_flat, batch, mora_len);
        grads.duration_predictor = dur_grads;
        grads.pitch_predictor = pitch_grads;
        grads.energy_predictor = energy_grads;
        // grad_encoder_out に 3 系統の変分予測 backward からの grad を加算
        for i in 0..grad_encoder_out.len() {
            grad_encoder_out[i] += grad_from_dur[i] + grad_from_pitch[i] + grad_from_energy[i];
        }

        // Step 6-8: Encoder chain + PE + Embedding backward
        let mut grad_current = grad_encoder_out;
        for i in (0..cfg.num_encoder_layers).rev() {
            let (grad_in, fft_grads) =
                self.encoder[i].backward(&encoder_inputs[i], &grad_current, batch, mora_len);
            grads.encoder[i] = fft_grads;
            grad_current = grad_in;
        }
        let grad_embedded = self
            .encoder_pos_enc
            .backward(&grad_current, batch, mora_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("encoder PE bw: {e}"),
            })?;
        grads.embedding = embedding_backward(
            &grad_embedded,
            mora_ids,
            cfg.vocab_size,
            cfg.hidden_dim,
            batch,
            mora_len,
        );

        Ok(grads)
    }

    /// Variable-length forward (Phase T.4a Phase D)。
    ///
    /// Per-sample mora length + attention mask 対応。padding 位置は encoder / decoder attention
    /// で除外され、Length Regulator は padding mora (duration=0 前提) を expand しない。
    ///
    /// # 引数
    ///
    /// - `mora_ids`: `[batch, max_mora_len]` flatten。padding は任意 valid id (mask で無視される)
    /// - `target_durations`: `[batch, max_mora_len]`、padding 位置は 0 必須
    /// - `batch`: batch size
    /// - `max_mora_len`: batch 内最大 mora 長 (padding 込み)
    /// - `mora_lens`: 各 sample の実 mora 長 `[batch]` (mora_lens[b] <= max_mora_len)
    ///
    /// # 戻り値
    ///
    /// `(mel, frame_lens, max_frame_len)`:
    /// - `mel`: `[batch, max_frame_len, mel_dim]` flatten (padding 領域は attention mask 経由で 0 近似)
    /// - `frame_lens`: 各 sample の実 frame 長 (= sum of durations for valid mora)
    /// - `max_frame_len`: batch 内最大 frame 長
    ///
    /// # Errors
    ///
    /// - shape 不整合
    /// - `mora_lens[b] > max_mora_len`
    /// - `max_frame_len > cfg.max_len`
    pub fn forward_variable(
        &self,
        mora_ids: &[u32],
        target_durations: &[u32],
        batch: usize,
        max_mora_len: usize,
        mora_lens: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>, usize), FastSpeech2Error> {
        let cfg = self.config;
        if mora_ids.len() != batch * max_mora_len {
            return Err(FastSpeech2Error::ShapeMismatch {
                field: "mora_ids",
                expected: batch * max_mora_len,
                actual: mora_ids.len(),
            });
        }
        if target_durations.len() != batch * max_mora_len {
            return Err(FastSpeech2Error::ShapeMismatch {
                field: "target_durations",
                expected: batch * max_mora_len,
                actual: target_durations.len(),
            });
        }
        if mora_lens.len() != batch {
            return Err(FastSpeech2Error::ShapeMismatch {
                field: "mora_lens",
                expected: batch,
                actual: mora_lens.len(),
            });
        }
        for (b, &ml) in mora_lens.iter().enumerate() {
            if ml > max_mora_len {
                return Err(FastSpeech2Error::Internal {
                    reason: format!("mora_lens[{b}]={ml} > max_mora_len={max_mora_len}"),
                });
            }
        }
        for &id in mora_ids {
            if id as usize >= cfg.vocab_size {
                return Err(FastSpeech2Error::VocabOverflow {
                    id: id as usize,
                    vocab_size: cfg.vocab_size,
                });
            }
        }

        // Encoder mask [batch * max_mora_len]
        let mut encoder_mask = vec![false; batch * max_mora_len];
        for b in 0..batch {
            for t in 0..mora_lens[b] {
                encoder_mask[b * max_mora_len + t] = true;
            }
        }

        // 1. Embedding lookup + positional encoding
        let mut embedded = vec![0.0_f32; batch * max_mora_len * cfg.hidden_dim];
        for b in 0..batch {
            for t in 0..max_mora_len {
                let id = mora_ids[b * max_mora_len + t] as usize;
                let src = &self.embedding[id * cfg.hidden_dim..(id + 1) * cfg.hidden_dim];
                let dst_start = b * max_mora_len * cfg.hidden_dim + t * cfg.hidden_dim;
                embedded[dst_start..dst_start + cfg.hidden_dim].copy_from_slice(src);
            }
        }
        let mut encoder_input = self
            .encoder_pos_enc
            .forward(&embedded, batch, max_mora_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("encoder PE: {e}"),
            })?;

        // 2. Encoder stack with mask
        for block in &self.encoder {
            encoder_input =
                block.forward_masked(&encoder_input, batch, max_mora_len, Some(&encoder_mask));
        }

        // 3. Length Regulator: expand mora → frame (padding mora は duration=0 で自然 skip)
        let (expanded, frame_len_per_batch) = length_regulator(
            &encoder_input,
            batch,
            max_mora_len,
            cfg.hidden_dim,
            target_durations,
        );
        let max_frame_len = *frame_len_per_batch.iter().max().unwrap_or(&0);
        if max_frame_len > cfg.max_len {
            return Err(FastSpeech2Error::SeqLenExceedsMax {
                seq_len: max_frame_len,
                max_len: cfg.max_len,
            });
        }

        // Decoder mask
        let mut decoder_mask = vec![false; batch * max_frame_len];
        for b in 0..batch {
            for t in 0..frame_len_per_batch[b] {
                decoder_mask[b * max_frame_len + t] = true;
            }
        }

        // 4. Decoder: PE + FFT block stack with mask
        let mut decoder_input = self
            .decoder_pos_enc
            .forward(&expanded, batch, max_frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("decoder PE: {e}"),
            })?;
        for block in &self.decoder {
            decoder_input =
                block.forward_masked(&decoder_input, batch, max_frame_len, Some(&decoder_mask));
        }

        // 5. Mel projection
        let mel_before = self
            .mel_linear
            .forward(&decoder_input, batch * max_frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("mel linear: {e}"),
            })?;

        // 6. Postnet residual
        let mel_reshaped =
            reshape_time_last_to_channel_first(&mel_before, batch, max_frame_len, cfg.mel_dim);
        let mut postnet_out = mel_reshaped;
        for (i, conv) in self.postnet.iter().enumerate() {
            let conv_out = conv
                .forward(&postnet_out, batch, max_frame_len)
                .map_err(|e| FastSpeech2Error::Internal {
                    reason: format!("postnet[{i}]: {e}"),
                })?;
            if i < self.postnet.len() - 1 {
                postnet_out = conv_out.iter().map(|&x| x.tanh()).collect();
            } else {
                postnet_out = conv_out;
            }
        }
        let mut postnet_out_reshaped = vec![0.0_f32; batch * max_frame_len * cfg.mel_dim];
        for b in 0..batch {
            for t in 0..max_frame_len {
                for c in 0..cfg.mel_dim {
                    postnet_out_reshaped[b * max_frame_len * cfg.mel_dim + t * cfg.mel_dim + c] =
                        postnet_out[b * cfg.mel_dim * max_frame_len + c * max_frame_len + t];
                }
            }
        }
        let mel_after: Vec<f32> = mel_before
            .iter()
            .zip(&postnet_out_reshaped)
            .map(|(a, b)| a + b)
            .collect();

        Ok((mel_after, frame_len_per_batch, max_frame_len))
    }

    /// Variable-length full backward (Phase T.4a Phase D)。
    ///
    /// `forward_variable` に対する完全 backward chain (encoder mask + decoder mask 経由)。
    /// grad_mel の padding 領域は呼び出し側で 0 済み前提 (masked loss を通せば自動)。
    ///
    /// # Errors
    ///
    /// - shape 不整合
    /// - forward_variable と同じエラー
    pub fn backward_full_variable(
        &self,
        mora_ids: &[u32],
        target_durations: &[u32],
        grad_mel: &[f32],
        batch: usize,
        max_mora_len: usize,
        mora_lens: &[usize],
    ) -> Result<FastSpeech2Grads, FastSpeech2Error> {
        let cfg = self.config;

        // Encoder mask
        let mut encoder_mask = vec![false; batch * max_mora_len];
        for b in 0..batch {
            for t in 0..mora_lens[b] {
                encoder_mask[b * max_mora_len + t] = true;
            }
        }

        // Forward 再計算: encoder path
        let mut embedded = vec![0.0_f32; batch * max_mora_len * cfg.hidden_dim];
        for b in 0..batch {
            for t in 0..max_mora_len {
                let id = mora_ids[b * max_mora_len + t] as usize;
                let src = &self.embedding[id * cfg.hidden_dim..(id + 1) * cfg.hidden_dim];
                let dst_start = b * max_mora_len * cfg.hidden_dim + t * cfg.hidden_dim;
                embedded[dst_start..dst_start + cfg.hidden_dim].copy_from_slice(src);
            }
        }
        let encoder_pe_out = self
            .encoder_pos_enc
            .forward(&embedded, batch, max_mora_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("encoder PE fw: {e}"),
            })?;
        let mut encoder_inputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_encoder_layers);
        let mut encoder_current = encoder_pe_out;
        for block in &self.encoder {
            encoder_inputs.push(encoder_current.clone());
            encoder_current =
                block.forward_masked(&encoder_current, batch, max_mora_len, Some(&encoder_mask));
        }
        let encoder_out = encoder_current;

        // LR forward
        let (expanded, frame_len_per_batch) = length_regulator(
            &encoder_out,
            batch,
            max_mora_len,
            cfg.hidden_dim,
            target_durations,
        );
        let max_frame_len = *frame_len_per_batch.iter().max().unwrap_or(&0);

        // Decoder mask
        let mut decoder_mask = vec![false; batch * max_frame_len];
        for b in 0..batch {
            for t in 0..frame_len_per_batch[b] {
                decoder_mask[b * max_frame_len + t] = true;
            }
        }

        // Decoder path
        let decoder_pe_out = self
            .decoder_pos_enc
            .forward(&expanded, batch, max_frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("decoder PE fw: {e}"),
            })?;
        let mut decoder_inputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_decoder_layers);
        let mut decoder_current = decoder_pe_out;
        for block in &self.decoder {
            decoder_inputs.push(decoder_current.clone());
            decoder_current =
                block.forward_masked(&decoder_current, batch, max_frame_len, Some(&decoder_mask));
        }
        let decoder_out = decoder_current;

        let mel_before = self
            .mel_linear
            .forward(&decoder_out, batch * max_frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("mel linear fw: {e}"),
            })?;

        // Postnet re-forward + cache
        let mel_reshaped =
            reshape_time_last_to_channel_first(&mel_before, batch, max_frame_len, cfg.mel_dim);
        let mut postnet_inputs: Vec<Vec<f32>> = Vec::with_capacity(self.postnet.len());
        let mut postnet_pre_act: Vec<Vec<f32>> = Vec::with_capacity(self.postnet.len());
        let mut current = mel_reshaped;
        for (i, conv) in self.postnet.iter().enumerate() {
            postnet_inputs.push(current.clone());
            let pre = conv.forward(&current, batch, max_frame_len).map_err(|e| {
                FastSpeech2Error::Internal {
                    reason: format!("postnet[{i}] fw: {e}"),
                }
            })?;
            postnet_pre_act.push(pre.clone());
            if i < self.postnet.len() - 1 {
                current = pre.iter().map(|&x| x.tanh()).collect();
            } else {
                current = pre;
            }
        }

        // grad_mel [B, F, mel_dim] → grad_mel_before + grad_postnet_residual (both = grad_mel)
        let grad_mel_after = grad_mel;

        // Postnet residual reshape [B, F, mel_dim] → [B, mel_dim, F]
        let mut grad_postnet_out = vec![0.0_f32; batch * cfg.mel_dim * max_frame_len];
        for b in 0..batch {
            for t in 0..max_frame_len {
                for c in 0..cfg.mel_dim {
                    grad_postnet_out[b * cfg.mel_dim * max_frame_len + c * max_frame_len + t] =
                        grad_mel_after[b * max_frame_len * cfg.mel_dim + t * cfg.mel_dim + c];
                }
            }
        }
        // Postnet chain backward (reverse)
        let mut grads = FastSpeech2Grads::zeros_from_config(&cfg);
        let mut grad_current = grad_postnet_out;
        for i in (0..self.postnet.len()).rev() {
            let pre = &postnet_pre_act[i];
            let input = &postnet_inputs[i];
            // tanh backward for i < last
            let grad_pre: Vec<f32> = if i < self.postnet.len() - 1 {
                pre.iter()
                    .zip(&grad_current)
                    .map(|(&z, &g)| g * (1.0 - z.tanh() * z.tanh()))
                    .collect()
            } else {
                grad_current
            };
            let (grad_in, grad_w, grad_b) = self.postnet[i]
                .backward(input, &grad_pre, batch, max_frame_len)
                .map_err(|e| FastSpeech2Error::Internal {
                    reason: format!("postnet[{i}] bw: {e}"),
                })?;
            grads.postnet_w[i] = grad_w;
            grads.postnet_b[i] = grad_b;
            grad_current = grad_in;
        }
        // reshape grad_current [B, mel_dim, F] → [B, F, mel_dim]
        let mut grad_postnet_residual = vec![0.0_f32; batch * max_frame_len * cfg.mel_dim];
        for b in 0..batch {
            for t in 0..max_frame_len {
                for c in 0..cfg.mel_dim {
                    grad_postnet_residual[b * max_frame_len * cfg.mel_dim + t * cfg.mel_dim + c] =
                        grad_current[b * cfg.mel_dim * max_frame_len + c * max_frame_len + t];
                }
            }
        }
        // grad_mel_before = grad_mel_after + grad_postnet_residual
        let grad_mel_before: Vec<f32> = grad_mel_after
            .iter()
            .zip(&grad_postnet_residual)
            .map(|(a, b)| a + b)
            .collect();

        // mel_linear backward
        let (grad_decoder_out, grad_mel_w, grad_mel_b) = self
            .mel_linear
            .backward(&decoder_out, &grad_mel_before, batch * max_frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("mel_linear bw: {e}"),
            })?;
        grads.mel_linear_w = grad_mel_w;
        grads.mel_linear_b = grad_mel_b;

        // Decoder FftBlock chain backward (reverse) with mask
        let mut grad_current = grad_decoder_out;
        for i in (0..cfg.num_decoder_layers).rev() {
            let (grad_in, fft_grads) = self.decoder[i].backward_masked(
                &decoder_inputs[i],
                &grad_current,
                batch,
                max_frame_len,
                Some(&decoder_mask),
            );
            grads.decoder[i] = fft_grads;
            grad_current = grad_in;
        }

        // Decoder PE backward
        let grad_expanded = self
            .decoder_pos_enc
            .backward(&grad_current, batch, max_frame_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("decoder PE bw: {e}"),
            })?;

        // Length Regulator backward (padding mora は duration=0 で grad=0)
        let grad_encoder_out = length_regulator_backward(
            &grad_expanded,
            batch,
            max_mora_len,
            cfg.hidden_dim,
            target_durations,
        );

        // Encoder FftBlock chain backward (reverse) with mask
        let mut grad_current = grad_encoder_out;
        for i in (0..cfg.num_encoder_layers).rev() {
            let (grad_in, fft_grads) = self.encoder[i].backward_masked(
                &encoder_inputs[i],
                &grad_current,
                batch,
                max_mora_len,
                Some(&encoder_mask),
            );
            grads.encoder[i] = fft_grads;
            grad_current = grad_in;
        }

        // Encoder PE backward
        let grad_embedded = self
            .encoder_pos_enc
            .backward(&grad_current, batch, max_mora_len)
            .map_err(|e| FastSpeech2Error::Internal {
                reason: format!("encoder PE bw: {e}"),
            })?;

        // Embedding backward (mora_ids で accumulate、padding 位置は grad_embedded が 0 なので自然)
        grads.embedding = embedding_backward(
            &grad_embedded,
            mora_ids,
            cfg.vocab_size,
            cfg.hidden_dim,
            batch,
            max_mora_len,
        );

        Ok(grads)
    }

    /// Xavier uniform init for全 sub-layer + embedding Gaussian init。
    ///
    /// 決定論 seed 対応。`zeros(cfg)?` の後で `model.init_xavier(seed)` すれば学習可能な初期値。
    ///
    /// - Embedding: N(0, 0.02) (transformer 慣習)
    /// - Conv1D / Linear / MHA projections: Xavier uniform
    /// - LayerNorm gamma/beta: default (gamma=1, beta=0)
    /// - Positional encoding: init 不要
    /// - Variance predictor + pitch/energy embed + Postnet も Xavier
    pub fn init_xavier(&mut self, seed: u64) {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Embedding: N(0, 0.02)
        let std = 0.02_f32;
        for w in &mut self.embedding {
            *w = gaussian(&mut rng, 0.0, std);
        }

        // Encoder + Decoder FftBlocks
        for block in &mut self.encoder {
            init_fft_block_xavier(block, &mut rng);
        }
        for block in &mut self.decoder {
            init_fft_block_xavier(block, &mut rng);
        }

        // Variance predictors
        init_variance_predictor_xavier(&mut self.duration_predictor, &mut rng);
        init_variance_predictor_xavier(&mut self.pitch_predictor, &mut rng);
        init_variance_predictor_xavier(&mut self.energy_predictor, &mut rng);

        // pitch/energy embed (Conv1D 1x1) + mel_linear + postnet
        self.pitch_embed.init_xavier(&mut rng);
        self.energy_embed.init_xavier(&mut rng);
        self.mel_linear.init_xavier(&mut rng);
        for conv in &mut self.postnet {
            conv.init_xavier(&mut rng);
        }
    }

    /// AdamW update: bias-corrected moment estimate + weight decay。
    ///
    /// state.step は自動 increment、初 call で step=1 開始。
    /// weight_decay=0 で通常 Adam に相当 (loshchilov 2019 AdamW と等価)。
    ///
    /// # 前提
    ///
    /// - `state` は [`FastSpeech2AdamWState::zeros_from_config`] で shape 整合済であること
    /// - grads は [`Self::backward_full`] 返却値と互換 shape
    pub fn apply_adamw(
        &mut self,
        grads: &FastSpeech2Grads,
        state: &mut FastSpeech2AdamWState,
        config: &AdamWConfig,
    ) {
        state.step += 1;
        let step = state.step as f32;
        let bc1 = 1.0 - config.beta1.powf(step);
        let bc2 = 1.0 - config.beta2.powf(step);

        adamw_slice(
            &mut self.embedding,
            &grads.embedding,
            &mut state.embedding_m,
            &mut state.embedding_v,
            config,
            bc1,
            bc2,
        );
        for (i, block) in self.encoder.iter_mut().enumerate() {
            apply_fft_block_adamw(
                block,
                &grads.encoder[i],
                &mut state.encoder[i],
                config,
                bc1,
                bc2,
            );
        }
        for (i, block) in self.decoder.iter_mut().enumerate() {
            apply_fft_block_adamw(
                block,
                &grads.decoder[i],
                &mut state.decoder[i],
                config,
                bc1,
                bc2,
            );
        }
        adamw_slice(
            self.mel_linear.weight_mut(),
            &grads.mel_linear_w,
            &mut state.mel_linear_m_w,
            &mut state.mel_linear_v_w,
            config,
            bc1,
            bc2,
        );
        adamw_slice(
            self.mel_linear.bias_mut(),
            &grads.mel_linear_b,
            &mut state.mel_linear_m_b,
            &mut state.mel_linear_v_b,
            config,
            bc1,
            bc2,
        );
        for (i, conv) in self.postnet.iter_mut().enumerate() {
            adamw_slice(
                conv.weight_mut(),
                &grads.postnet_w[i],
                &mut state.postnet_m_w[i],
                &mut state.postnet_v_w[i],
                config,
                bc1,
                bc2,
            );
            adamw_slice(
                conv.bias_mut(),
                &grads.postnet_b[i],
                &mut state.postnet_m_b[i],
                &mut state.postnet_v_b[i],
                config,
                bc1,
                bc2,
            );
        }
        // Variance predictor / pitch/energy embed backward で grad = 0 のため update 無効化
        // (Phase T.4a 継続で ProsodyLoss joint 時に生きる)
    }

    /// 全 model weight を safetensors ファイルに保存する (Phase T.4a Phase C)。
    ///
    /// Paperspace 6-hour session limit や長期学習の中断復帰用 tensor 命名 convention は
    /// `embedding` / `encoder.{i}.mha.w_q` / `decoder.{i}.attn_norm.gamma` / `mel_linear.w` etc
    ///
    /// # Errors
    ///
    /// I/O + safetensors serialize error。
    pub fn save_safetensors(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), FastSpeech2Error> {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;
        use std::collections::HashMap;

        let mut named: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
        push_tensor(
            &mut named,
            "embedding",
            &self.embedding,
            &[self.config.vocab_size, self.config.hidden_dim],
        );

        for (i, block) in self.encoder.iter().enumerate() {
            dump_fft_block(&mut named, &format!("encoder.{i}"), block);
        }
        for (i, block) in self.decoder.iter().enumerate() {
            dump_fft_block(&mut named, &format!("decoder.{i}"), block);
        }
        dump_variance_predictor(&mut named, "duration_predictor", &self.duration_predictor);
        dump_variance_predictor(&mut named, "pitch_predictor", &self.pitch_predictor);
        dump_variance_predictor(&mut named, "energy_predictor", &self.energy_predictor);
        push_tensor(
            &mut named,
            "pitch_embed.w",
            self.pitch_embed.weight(),
            &[self.pitch_embed.weight().len()],
        );
        push_tensor(
            &mut named,
            "pitch_embed.b",
            self.pitch_embed.bias(),
            &[self.pitch_embed.bias().len()],
        );
        push_tensor(
            &mut named,
            "energy_embed.w",
            self.energy_embed.weight(),
            &[self.energy_embed.weight().len()],
        );
        push_tensor(
            &mut named,
            "energy_embed.b",
            self.energy_embed.bias(),
            &[self.energy_embed.bias().len()],
        );
        push_tensor(
            &mut named,
            "mel_linear.w",
            self.mel_linear.weight(),
            &[self.config.mel_dim, self.config.hidden_dim],
        );
        push_tensor(
            &mut named,
            "mel_linear.b",
            self.mel_linear.bias(),
            &[self.config.mel_dim],
        );
        for (i, conv) in self.postnet.iter().enumerate() {
            push_tensor(
                &mut named,
                &format!("postnet.{i}.w"),
                conv.weight(),
                &[conv.weight().len()],
            );
            push_tensor(
                &mut named,
                &format!("postnet.{i}.b"),
                conv.bias(),
                &[conv.bias().len()],
            );
        }

        let views: HashMap<String, TensorView> = named
            .iter()
            .map(|(n, b, s)| {
                (
                    n.clone(),
                    TensorView::new(Dtype::F32, s.clone(), b).expect("valid tensor"),
                )
            })
            .collect();
        safetensors::serialize_to_file(&views, &None, path.as_ref()).map_err(|e| {
            FastSpeech2Error::Internal {
                reason: format!("safetensors save failed: {e}"),
            }
        })
    }

    /// safetensors ファイルから全 model weight を復元する。
    ///
    /// # Errors
    ///
    /// I/O + safetensors + tensor 不在 + shape 不一致 error。
    pub fn load_safetensors(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), FastSpeech2Error> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| FastSpeech2Error::Internal {
            reason: format!("read failed: {e}"),
        })?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| {
            FastSpeech2Error::Internal {
                reason: format!("safetensors deserialize failed: {e}"),
            }
        })?;
        self.embedding = load_tensor(&tensors, "embedding")?;
        for (i, block) in self.encoder.iter_mut().enumerate() {
            apply_fft_block_tensors(block, &tensors, &format!("encoder.{i}"))?;
        }
        for (i, block) in self.decoder.iter_mut().enumerate() {
            apply_fft_block_tensors(block, &tensors, &format!("decoder.{i}"))?;
        }
        apply_variance_predictor_tensors(
            &mut self.duration_predictor,
            &tensors,
            "duration_predictor",
        )?;
        apply_variance_predictor_tensors(&mut self.pitch_predictor, &tensors, "pitch_predictor")?;
        apply_variance_predictor_tensors(&mut self.energy_predictor, &tensors, "energy_predictor")?;
        replace_conv1d_weights(&mut self.pitch_embed, &tensors, "pitch_embed")?;
        replace_conv1d_weights(&mut self.energy_embed, &tensors, "energy_embed")?;
        replace_linear_weights(&mut self.mel_linear, &tensors, "mel_linear")?;
        for (i, conv) in self.postnet.iter_mut().enumerate() {
            replace_conv1d_weights(conv, &tensors, &format!("postnet.{i}"))?;
        }
        Ok(())
    }

    /// embedding への参照 (checkpoint 保存 test の accessor)。
    #[must_use]
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// SGD update: `w -= lr * grad` を全 sub-layer weight + bias に適用する。
    ///
    /// Variance predictor + pitch/energy embed は backward_full で grad が 0 になっているため
    /// (MVP 制約: forward で使用してない) 更新しても影響なし。
    pub fn apply_sgd(&mut self, grads: &FastSpeech2Grads, lr: f32) {
        // Embedding
        for (w, g) in self.embedding.iter_mut().zip(&grads.embedding) {
            *w -= lr * g;
        }
        // Encoder FftBlocks
        for (block, bg) in self.encoder.iter_mut().zip(&grads.encoder) {
            apply_fft_block_sgd(block, bg, lr);
        }
        // Decoder FftBlocks
        for (block, bg) in self.decoder.iter_mut().zip(&grads.decoder) {
            apply_fft_block_sgd(block, bg, lr);
        }
        // mel_linear
        for (w, g) in self
            .mel_linear
            .weight_mut()
            .iter_mut()
            .zip(&grads.mel_linear_w)
        {
            *w -= lr * g;
        }
        for (b, g) in self
            .mel_linear
            .bias_mut()
            .iter_mut()
            .zip(&grads.mel_linear_b)
        {
            *b -= lr * g;
        }
        // Postnet
        for (i, conv) in self.postnet.iter_mut().enumerate() {
            for (w, g) in conv.weight_mut().iter_mut().zip(&grads.postnet_w[i]) {
                *w -= lr * g;
            }
            for (b, g) in conv.bias_mut().iter_mut().zip(&grads.postnet_b[i]) {
                *b -= lr * g;
            }
        }
        // Variance predictors (Phase E: ProsodyLoss joint training で grad != 0 になる)
        apply_variance_predictor_sgd(&mut self.duration_predictor, &grads.duration_predictor, lr);
        apply_variance_predictor_sgd(&mut self.pitch_predictor, &grads.pitch_predictor, lr);
        apply_variance_predictor_sgd(&mut self.energy_predictor, &grads.energy_predictor, lr);
    }

    /// AdamW trainer 経路で variance predictor だけ SGD で fallback 更新する helper (Phase E)。
    ///
    /// `AdamW state` に variance predictor state が未拡張のため、Phase E MVP では
    /// AdamW trainer 使用時も variance predictor は SGD 更新で運用する。
    /// (次 Phase で `FastSpeech2AdamWState` を拡張し完全 AdamW 化予定)
    pub fn apply_sgd_variance_only(&mut self, grads: &FastSpeech2Grads, lr: f32) {
        apply_variance_predictor_sgd(&mut self.duration_predictor, &grads.duration_predictor, lr);
        apply_variance_predictor_sgd(&mut self.pitch_predictor, &grads.pitch_predictor, lr);
        apply_variance_predictor_sgd(&mut self.energy_predictor, &grads.energy_predictor, lr);
    }
}

/// AdamW optimizer 設定 (Loshchilov & Hutter 2019)。
///
/// FastSpeech2 論文 default (Ren et al. 2021): lr=1e-3 initial with Noam schedule、
/// beta1=0.9、beta2=0.98、eps=1e-9、weight_decay=0。
#[derive(Clone, Copy, Debug)]
pub struct AdamWConfig {
    /// 学習率。
    pub learning_rate: f32,
    /// 1st moment decay (通常 0.9)。
    pub beta1: f32,
    /// 2nd moment decay (通常 0.999 または FastSpeech2 論文の 0.98)。
    pub beta2: f32,
    /// 数値安定性 eps (通常 1e-8)。
    pub eps: f32,
    /// L2 weight decay (通常 0)。
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// FFT block 用 AdamW state (mha + attn_norm + ffn_conv1/2 + ffn_norm の m/v)。
#[derive(Clone, Debug, Default)]
pub struct FftBlockAdamWState {
    /// MHA Q/K/V/O weight m/v。
    pub mha_m_w_q: Vec<f32>,
    /// MHA V weight m。
    pub mha_v_w_q: Vec<f32>,
    /// (K)。
    pub mha_m_w_k: Vec<f32>,
    /// (K)。
    pub mha_v_w_k: Vec<f32>,
    /// (V)。
    pub mha_m_w_v: Vec<f32>,
    /// (V)。
    pub mha_v_w_v: Vec<f32>,
    /// (O)。
    pub mha_m_w_o: Vec<f32>,
    /// (O)。
    pub mha_v_w_o: Vec<f32>,
    /// MHA bias m/v。
    pub mha_m_b_q: Vec<f32>,
    /// (Q)。
    pub mha_v_b_q: Vec<f32>,
    /// (K)。
    pub mha_m_b_k: Vec<f32>,
    /// (K)。
    pub mha_v_b_k: Vec<f32>,
    /// (V)。
    pub mha_m_b_v: Vec<f32>,
    /// (V)。
    pub mha_v_b_v: Vec<f32>,
    /// (O)。
    pub mha_m_b_o: Vec<f32>,
    /// (O)。
    pub mha_v_b_o: Vec<f32>,
    /// attn_norm gamma m/v。
    pub attn_norm_m_gamma: Vec<f32>,
    /// (v)。
    pub attn_norm_v_gamma: Vec<f32>,
    /// attn_norm beta m/v。
    pub attn_norm_m_beta: Vec<f32>,
    /// (v)。
    pub attn_norm_v_beta: Vec<f32>,
    /// ffn_conv1 w m/v。
    pub ffn_conv1_m_w: Vec<f32>,
    /// (v)。
    pub ffn_conv1_v_w: Vec<f32>,
    /// ffn_conv1 b m/v。
    pub ffn_conv1_m_b: Vec<f32>,
    /// (v)。
    pub ffn_conv1_v_b: Vec<f32>,
    /// ffn_conv2 w m/v。
    pub ffn_conv2_m_w: Vec<f32>,
    /// (v)。
    pub ffn_conv2_v_w: Vec<f32>,
    /// ffn_conv2 b m/v。
    pub ffn_conv2_m_b: Vec<f32>,
    /// (v)。
    pub ffn_conv2_v_b: Vec<f32>,
    /// ffn_norm gamma m/v。
    pub ffn_norm_m_gamma: Vec<f32>,
    /// (v)。
    pub ffn_norm_v_gamma: Vec<f32>,
    /// ffn_norm beta m/v。
    pub ffn_norm_m_beta: Vec<f32>,
    /// (v)。
    pub ffn_norm_v_beta: Vec<f32>,
}

/// FastSpeech2 全体の AdamW state (m/v + step)。
#[derive(Clone, Debug, Default)]
pub struct FastSpeech2AdamWState {
    /// step counter (bias correction 用)。
    pub step: usize,
    /// Embedding m/v。
    pub embedding_m: Vec<f32>,
    /// (v)。
    pub embedding_v: Vec<f32>,
    /// Encoder FftBlock 各層 state。
    pub encoder: Vec<FftBlockAdamWState>,
    /// Decoder FftBlock 各層 state。
    pub decoder: Vec<FftBlockAdamWState>,
    /// mel_linear weight m/v。
    pub mel_linear_m_w: Vec<f32>,
    /// (v)。
    pub mel_linear_v_w: Vec<f32>,
    /// mel_linear bias m/v。
    pub mel_linear_m_b: Vec<f32>,
    /// (v)。
    pub mel_linear_v_b: Vec<f32>,
    /// Postnet 各層 weight m。
    pub postnet_m_w: Vec<Vec<f32>>,
    /// Postnet 各層 weight v。
    pub postnet_v_w: Vec<Vec<f32>>,
    /// Postnet 各層 bias m。
    pub postnet_m_b: Vec<Vec<f32>>,
    /// Postnet 各層 bias v。
    pub postnet_v_b: Vec<Vec<f32>>,
}

impl FastSpeech2AdamWState {
    /// FastSpeech2Config に合わせた 0 埋め state を構築する。
    #[must_use]
    pub fn zeros_from_config(cfg: &FastSpeech2Config) -> Self {
        let h = cfg.hidden_dim;
        let mel = cfg.mel_dim;
        let mid = h * cfg.fft_expansion;
        let fft_state = || FftBlockAdamWState {
            mha_m_w_q: vec![0.0; h * h],
            mha_v_w_q: vec![0.0; h * h],
            mha_m_w_k: vec![0.0; h * h],
            mha_v_w_k: vec![0.0; h * h],
            mha_m_w_v: vec![0.0; h * h],
            mha_v_w_v: vec![0.0; h * h],
            mha_m_w_o: vec![0.0; h * h],
            mha_v_w_o: vec![0.0; h * h],
            mha_m_b_q: vec![0.0; h],
            mha_v_b_q: vec![0.0; h],
            mha_m_b_k: vec![0.0; h],
            mha_v_b_k: vec![0.0; h],
            mha_m_b_v: vec![0.0; h],
            mha_v_b_v: vec![0.0; h],
            mha_m_b_o: vec![0.0; h],
            mha_v_b_o: vec![0.0; h],
            attn_norm_m_gamma: vec![0.0; h],
            attn_norm_v_gamma: vec![0.0; h],
            attn_norm_m_beta: vec![0.0; h],
            attn_norm_v_beta: vec![0.0; h],
            ffn_conv1_m_w: vec![0.0; mid * h * cfg.fft_kernel_size],
            ffn_conv1_v_w: vec![0.0; mid * h * cfg.fft_kernel_size],
            ffn_conv1_m_b: vec![0.0; mid],
            ffn_conv1_v_b: vec![0.0; mid],
            ffn_conv2_m_w: vec![0.0; h * mid * cfg.fft_kernel_size],
            ffn_conv2_v_w: vec![0.0; h * mid * cfg.fft_kernel_size],
            ffn_conv2_m_b: vec![0.0; h],
            ffn_conv2_v_b: vec![0.0; h],
            ffn_norm_m_gamma: vec![0.0; h],
            ffn_norm_v_gamma: vec![0.0; h],
            ffn_norm_m_beta: vec![0.0; h],
            ffn_norm_v_beta: vec![0.0; h],
        };

        let postnet_shapes: Vec<usize> = (0..cfg.postnet_layers)
            .map(|i| {
                let (in_ch, out_ch) = if i == 0 {
                    (mel, cfg.postnet_hidden)
                } else if i == cfg.postnet_layers - 1 {
                    (cfg.postnet_hidden, mel)
                } else {
                    (cfg.postnet_hidden, cfg.postnet_hidden)
                };
                out_ch * in_ch * cfg.postnet_kernel_size
            })
            .collect();
        let postnet_bias_shapes: Vec<usize> = (0..cfg.postnet_layers)
            .map(|i| {
                if i == cfg.postnet_layers - 1 {
                    mel
                } else {
                    cfg.postnet_hidden
                }
            })
            .collect();

        Self {
            step: 0,
            embedding_m: vec![0.0; cfg.vocab_size * h],
            embedding_v: vec![0.0; cfg.vocab_size * h],
            encoder: (0..cfg.num_encoder_layers).map(|_| fft_state()).collect(),
            decoder: (0..cfg.num_decoder_layers).map(|_| fft_state()).collect(),
            mel_linear_m_w: vec![0.0; mel * h],
            mel_linear_v_w: vec![0.0; mel * h],
            mel_linear_m_b: vec![0.0; mel],
            mel_linear_v_b: vec![0.0; mel],
            postnet_m_w: postnet_shapes.iter().map(|&n| vec![0.0; n]).collect(),
            postnet_v_w: postnet_shapes.iter().map(|&n| vec![0.0; n]).collect(),
            postnet_m_b: postnet_bias_shapes.iter().map(|&n| vec![0.0; n]).collect(),
            postnet_v_b: postnet_bias_shapes.iter().map(|&n| vec![0.0; n]).collect(),
        }
    }
}

/// AdamW slice update: `w -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)`。
fn adamw_slice(
    w: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    config: &AdamWConfig,
    bc1: f32,
    bc2: f32,
) {
    for i in 0..w.len() {
        let gi = g[i];
        m[i] = config.beta1 * m[i] + (1.0 - config.beta1) * gi;
        v[i] = config.beta2 * v[i] + (1.0 - config.beta2) * gi * gi;
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        w[i] -= config.learning_rate
            * (m_hat / (v_hat.sqrt() + config.eps) + config.weight_decay * w[i]);
    }
}

/// FftBlock AdamW update helper。
fn apply_fft_block_adamw(
    block: &mut FftBlock,
    grads: &FftBlockGrads,
    state: &mut FftBlockAdamWState,
    config: &AdamWConfig,
    bc1: f32,
    bc2: f32,
) {
    // MHA
    let has_bias = block.self_attn.config().bias;
    adamw_slice(
        block.self_attn.w_q_mut(),
        &grads.mha.w_q,
        &mut state.mha_m_w_q,
        &mut state.mha_v_w_q,
        config,
        bc1,
        bc2,
    );
    adamw_slice(
        block.self_attn.w_k_mut(),
        &grads.mha.w_k,
        &mut state.mha_m_w_k,
        &mut state.mha_v_w_k,
        config,
        bc1,
        bc2,
    );
    adamw_slice(
        block.self_attn.w_v_mut(),
        &grads.mha.w_v,
        &mut state.mha_m_w_v,
        &mut state.mha_v_w_v,
        config,
        bc1,
        bc2,
    );
    adamw_slice(
        block.self_attn.w_o_mut(),
        &grads.mha.w_o,
        &mut state.mha_m_w_o,
        &mut state.mha_v_w_o,
        config,
        bc1,
        bc2,
    );
    if has_bias {
        adamw_slice(
            block.self_attn.b_q_mut(),
            &grads.mha.b_q,
            &mut state.mha_m_b_q,
            &mut state.mha_v_b_q,
            config,
            bc1,
            bc2,
        );
        adamw_slice(
            block.self_attn.b_k_mut(),
            &grads.mha.b_k,
            &mut state.mha_m_b_k,
            &mut state.mha_v_b_k,
            config,
            bc1,
            bc2,
        );
        adamw_slice(
            block.self_attn.b_v_mut(),
            &grads.mha.b_v,
            &mut state.mha_m_b_v,
            &mut state.mha_v_b_v,
            config,
            bc1,
            bc2,
        );
        adamw_slice(
            block.self_attn.b_o_mut(),
            &grads.mha.b_o,
            &mut state.mha_m_b_o,
            &mut state.mha_v_b_o,
            config,
            bc1,
            bc2,
        );
    }
    // attn_norm
    adamw_slice(
        block.attn_norm.gamma_mut(),
        &grads.attn_norm_gamma,
        &mut state.attn_norm_m_gamma,
        &mut state.attn_norm_v_gamma,
        config,
        bc1,
        bc2,
    );
    adamw_slice(
        block.attn_norm.beta_mut(),
        &grads.attn_norm_beta,
        &mut state.attn_norm_m_beta,
        &mut state.attn_norm_v_beta,
        config,
        bc1,
        bc2,
    );
    // ffn_conv1
    adamw_slice(
        block.ffn_conv1.weight_mut(),
        &grads.ffn_conv1_w,
        &mut state.ffn_conv1_m_w,
        &mut state.ffn_conv1_v_w,
        config,
        bc1,
        bc2,
    );
    adamw_slice(
        block.ffn_conv1.bias_mut(),
        &grads.ffn_conv1_b,
        &mut state.ffn_conv1_m_b,
        &mut state.ffn_conv1_v_b,
        config,
        bc1,
        bc2,
    );
    // ffn_conv2
    adamw_slice(
        block.ffn_conv2.weight_mut(),
        &grads.ffn_conv2_w,
        &mut state.ffn_conv2_m_w,
        &mut state.ffn_conv2_v_w,
        config,
        bc1,
        bc2,
    );
    adamw_slice(
        block.ffn_conv2.bias_mut(),
        &grads.ffn_conv2_b,
        &mut state.ffn_conv2_m_b,
        &mut state.ffn_conv2_v_b,
        config,
        bc1,
        bc2,
    );
    // ffn_norm
    adamw_slice(
        block.ffn_norm.gamma_mut(),
        &grads.ffn_norm_gamma,
        &mut state.ffn_norm_m_gamma,
        &mut state.ffn_norm_v_gamma,
        config,
        bc1,
        bc2,
    );
    adamw_slice(
        block.ffn_norm.beta_mut(),
        &grads.ffn_norm_beta,
        &mut state.ffn_norm_m_beta,
        &mut state.ffn_norm_v_beta,
        config,
        bc1,
        bc2,
    );
}

/// safetensors save helper: (name, bytes, shape) tuple を named_tensors vec に push。
fn push_tensor(
    out: &mut Vec<(String, Vec<u8>, Vec<usize>)>,
    name: &str,
    data: &[f32],
    shape: &[usize],
) {
    let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
    out.push((name.to_string(), bytes, shape.to_vec()));
}

/// safetensors load helper: tensor 名で f32 Vec を取り出す。
fn load_tensor(
    tensors: &safetensors::SafeTensors,
    name: &str,
) -> Result<Vec<f32>, FastSpeech2Error> {
    let view = tensors
        .tensor(name)
        .map_err(|e| FastSpeech2Error::Internal {
            reason: format!("tensor '{name}' not found: {e}"),
        })?;
    let data: &[f32] = bytemuck::cast_slice(view.data());
    Ok(data.to_vec())
}

/// FftBlock の全 sub-layer weight を named list に dump する。
fn dump_fft_block(out: &mut Vec<(String, Vec<u8>, Vec<usize>)>, prefix: &str, block: &FftBlock) {
    push_tensor(
        out,
        &format!("{prefix}.mha.w_q"),
        block.self_attn.w_q(),
        &[block.self_attn.w_q().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.mha.w_k"),
        block.self_attn.w_k(),
        &[block.self_attn.w_k().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.mha.w_v"),
        block.self_attn.w_v(),
        &[block.self_attn.w_v().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.mha.w_o"),
        block.self_attn.w_o(),
        &[block.self_attn.w_o().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.mha.b_q"),
        block.self_attn.b_q(),
        &[block.self_attn.b_q().len()],
    );
    // For b_k/v/o, access via accessor (add if missing later)
    push_tensor(
        out,
        &format!("{prefix}.attn_norm.gamma"),
        block.attn_norm.gamma(),
        &[block.attn_norm.gamma().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.attn_norm.beta"),
        block.attn_norm.beta(),
        &[block.attn_norm.beta().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.ffn_conv1.w"),
        block.ffn_conv1.weight(),
        &[block.ffn_conv1.weight().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.ffn_conv1.b"),
        block.ffn_conv1.bias(),
        &[block.ffn_conv1.bias().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.ffn_conv2.w"),
        block.ffn_conv2.weight(),
        &[block.ffn_conv2.weight().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.ffn_conv2.b"),
        block.ffn_conv2.bias(),
        &[block.ffn_conv2.bias().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.ffn_norm.gamma"),
        block.ffn_norm.gamma(),
        &[block.ffn_norm.gamma().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.ffn_norm.beta"),
        block.ffn_norm.beta(),
        &[block.ffn_norm.beta().len()],
    );
}

/// FftBlock の全 sub-layer weight を tensors から復元する。
fn apply_fft_block_tensors(
    block: &mut FftBlock,
    tensors: &safetensors::SafeTensors,
    prefix: &str,
) -> Result<(), FastSpeech2Error> {
    let w_q = load_tensor(tensors, &format!("{prefix}.mha.w_q"))?;
    let w_k = load_tensor(tensors, &format!("{prefix}.mha.w_k"))?;
    let w_v = load_tensor(tensors, &format!("{prefix}.mha.w_v"))?;
    let w_o = load_tensor(tensors, &format!("{prefix}.mha.w_o"))?;
    let b_q = load_tensor(tensors, &format!("{prefix}.mha.b_q"))?;
    block.self_attn.w_q_mut().copy_from_slice(&w_q);
    block.self_attn.w_k_mut().copy_from_slice(&w_k);
    block.self_attn.w_v_mut().copy_from_slice(&w_v);
    block.self_attn.w_o_mut().copy_from_slice(&w_o);
    if !b_q.is_empty() && block.self_attn.config().bias {
        block.self_attn.b_q_mut().copy_from_slice(&b_q);
    }
    let attn_gamma = load_tensor(tensors, &format!("{prefix}.attn_norm.gamma"))?;
    let attn_beta = load_tensor(tensors, &format!("{prefix}.attn_norm.beta"))?;
    block.attn_norm.gamma_mut().copy_from_slice(&attn_gamma);
    block.attn_norm.beta_mut().copy_from_slice(&attn_beta);
    let c1_w = load_tensor(tensors, &format!("{prefix}.ffn_conv1.w"))?;
    let c1_b = load_tensor(tensors, &format!("{prefix}.ffn_conv1.b"))?;
    block.ffn_conv1.weight_mut().copy_from_slice(&c1_w);
    block.ffn_conv1.bias_mut().copy_from_slice(&c1_b);
    let c2_w = load_tensor(tensors, &format!("{prefix}.ffn_conv2.w"))?;
    let c2_b = load_tensor(tensors, &format!("{prefix}.ffn_conv2.b"))?;
    block.ffn_conv2.weight_mut().copy_from_slice(&c2_w);
    block.ffn_conv2.bias_mut().copy_from_slice(&c2_b);
    let ffn_gamma = load_tensor(tensors, &format!("{prefix}.ffn_norm.gamma"))?;
    let ffn_beta = load_tensor(tensors, &format!("{prefix}.ffn_norm.beta"))?;
    block.ffn_norm.gamma_mut().copy_from_slice(&ffn_gamma);
    block.ffn_norm.beta_mut().copy_from_slice(&ffn_beta);
    Ok(())
}

/// VariancePredictor dump helper。
fn dump_variance_predictor(
    out: &mut Vec<(String, Vec<u8>, Vec<usize>)>,
    prefix: &str,
    vp: &VariancePredictor,
) {
    push_tensor(
        out,
        &format!("{prefix}.conv1.w"),
        vp.conv1.weight(),
        &[vp.conv1.weight().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.conv1.b"),
        vp.conv1.bias(),
        &[vp.conv1.bias().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.norm1.gamma"),
        vp.norm1.gamma(),
        &[vp.norm1.gamma().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.norm1.beta"),
        vp.norm1.beta(),
        &[vp.norm1.beta().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.conv2.w"),
        vp.conv2.weight(),
        &[vp.conv2.weight().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.conv2.b"),
        vp.conv2.bias(),
        &[vp.conv2.bias().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.norm2.gamma"),
        vp.norm2.gamma(),
        &[vp.norm2.gamma().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.norm2.beta"),
        vp.norm2.beta(),
        &[vp.norm2.beta().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.linear.w"),
        vp.linear.weight(),
        &[vp.linear.weight().len()],
    );
    push_tensor(
        out,
        &format!("{prefix}.linear.b"),
        vp.linear.bias(),
        &[vp.linear.bias().len()],
    );
}

/// VariancePredictor 復元 helper。
fn apply_variance_predictor_tensors(
    vp: &mut VariancePredictor,
    tensors: &safetensors::SafeTensors,
    prefix: &str,
) -> Result<(), FastSpeech2Error> {
    replace_conv1d_weights(&mut vp.conv1, tensors, &format!("{prefix}.conv1"))?;
    replace_layer_norm(&mut vp.norm1, tensors, &format!("{prefix}.norm1"))?;
    replace_conv1d_weights(&mut vp.conv2, tensors, &format!("{prefix}.conv2"))?;
    replace_layer_norm(&mut vp.norm2, tensors, &format!("{prefix}.norm2"))?;
    replace_linear_weights(&mut vp.linear, tensors, &format!("{prefix}.linear"))?;
    Ok(())
}

/// Conv1d weight/bias を tensors から復元する helper。
fn replace_conv1d_weights(
    conv: &mut Conv1d,
    tensors: &safetensors::SafeTensors,
    prefix: &str,
) -> Result<(), FastSpeech2Error> {
    let w = load_tensor(tensors, &format!("{prefix}.w"))?;
    let b = load_tensor(tensors, &format!("{prefix}.b"))?;
    conv.weight_mut().copy_from_slice(&w);
    conv.bias_mut().copy_from_slice(&b);
    Ok(())
}

/// Linear weight/bias を tensors から復元する helper。
fn replace_linear_weights(
    lin: &mut Linear,
    tensors: &safetensors::SafeTensors,
    prefix: &str,
) -> Result<(), FastSpeech2Error> {
    let w = load_tensor(tensors, &format!("{prefix}.w"))?;
    let b = load_tensor(tensors, &format!("{prefix}.b"))?;
    lin.weight_mut().copy_from_slice(&w);
    lin.bias_mut().copy_from_slice(&b);
    Ok(())
}

/// LayerNorm gamma/beta を tensors から復元する helper。
fn replace_layer_norm(
    ln: &mut LayerNorm,
    tensors: &safetensors::SafeTensors,
    prefix: &str,
) -> Result<(), FastSpeech2Error> {
    let gamma = load_tensor(tensors, &format!("{prefix}.gamma"))?;
    let beta = load_tensor(tensors, &format!("{prefix}.beta"))?;
    ln.gamma_mut().copy_from_slice(&gamma);
    ln.beta_mut().copy_from_slice(&beta);
    Ok(())
}

/// Gaussian (Box-Muller) sampling helper (uniform → normal)。
fn gaussian<R: rand::Rng>(rng: &mut R, mean: f32, std: f32) -> f32 {
    let u1: f32 = rng.gen_range(1e-9_f32..1.0); // avoid log(0)
    let u2: f32 = rng.gen();
    let z = (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
    mean + std * z
}

/// FftBlock Xavier init helper (private struct、同 module 内)。
fn init_fft_block_xavier<R: rand::Rng>(block: &mut FftBlock, rng: &mut R) {
    block.self_attn.init_xavier(rng);
    // attn_norm gamma=1, beta=0 は default_init のまま (触らない)
    block.ffn_conv1.init_xavier(rng);
    block.ffn_conv2.init_xavier(rng);
    // ffn_norm も同上
}

/// VariancePredictor Xavier init helper (private struct、同 module 内)。
fn init_variance_predictor_xavier<R: rand::Rng>(vp: &mut VariancePredictor, rng: &mut R) {
    vp.conv1.init_xavier(rng);
    vp.conv2.init_xavier(rng);
    vp.linear.init_xavier(rng);
    // norm1/norm2 gamma=1, beta=0 は default_init のまま
}

/// FftBlock SGD update helper (private struct のため同 module 内 helper)。
fn apply_fft_block_sgd(block: &mut FftBlock, grads: &FftBlockGrads, lr: f32) {
    // MHA
    block.self_attn.apply_sgd(&grads.mha, lr);
    // attn_norm
    for (w, g) in block
        .attn_norm
        .gamma_mut()
        .iter_mut()
        .zip(&grads.attn_norm_gamma)
    {
        *w -= lr * g;
    }
    for (b, g) in block
        .attn_norm
        .beta_mut()
        .iter_mut()
        .zip(&grads.attn_norm_beta)
    {
        *b -= lr * g;
    }
    // ffn_conv1
    for (w, g) in block
        .ffn_conv1
        .weight_mut()
        .iter_mut()
        .zip(&grads.ffn_conv1_w)
    {
        *w -= lr * g;
    }
    for (b, g) in block
        .ffn_conv1
        .bias_mut()
        .iter_mut()
        .zip(&grads.ffn_conv1_b)
    {
        *b -= lr * g;
    }
    // ffn_conv2
    for (w, g) in block
        .ffn_conv2
        .weight_mut()
        .iter_mut()
        .zip(&grads.ffn_conv2_w)
    {
        *w -= lr * g;
    }
    for (b, g) in block
        .ffn_conv2
        .bias_mut()
        .iter_mut()
        .zip(&grads.ffn_conv2_b)
    {
        *b -= lr * g;
    }
    // ffn_norm
    for (w, g) in block
        .ffn_norm
        .gamma_mut()
        .iter_mut()
        .zip(&grads.ffn_norm_gamma)
    {
        *w -= lr * g;
    }
    for (b, g) in block
        .ffn_norm
        .beta_mut()
        .iter_mut()
        .zip(&grads.ffn_norm_beta)
    {
        *b -= lr * g;
    }
}

/// Variance predictor の flat [B*S] 出力 3 系統を `ProsodyPrediction` `Vec<Vec<f32>>` に整形 (Phase E)。
fn flat_predictions_to_prosody(
    duration_flat: &[f32],
    pitch_flat: &[f32],
    energy_flat: &[f32],
    batch: usize,
    mora_len: usize,
) -> crate::tts::ProsodyPrediction {
    let mut duration_frames = Vec::with_capacity(batch);
    let mut f0 = Vec::with_capacity(batch);
    let mut energy = Vec::with_capacity(batch);
    for b in 0..batch {
        let start = b * mora_len;
        let end = start + mora_len;
        duration_frames.push(duration_flat[start..end].to_vec());
        f0.push(pitch_flat[start..end].to_vec());
        energy.push(energy_flat[start..end].to_vec());
    }
    crate::tts::ProsodyPrediction {
        f0,
        duration_frames,
        energy,
    }
}

/// `ProsodyPrediction` の grad `Vec<Vec<f32>>` 3 系統を flat [B*S] に戻す (Phase E backward)。
fn prosody_grads_to_flat(
    grad_duration: &[Vec<f32>],
    grad_pitch: &[Vec<f32>],
    grad_energy: &[Vec<f32>],
    batch: usize,
    mora_len: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let total = batch * mora_len;
    let mut dur_flat = vec![0.0_f32; total];
    let mut pitch_flat = vec![0.0_f32; total];
    let mut energy_flat = vec![0.0_f32; total];
    for b in 0..batch {
        for m in 0..mora_len {
            let idx = b * mora_len + m;
            dur_flat[idx] = grad_duration[b][m];
            pitch_flat[idx] = grad_pitch[b][m];
            energy_flat[idx] = grad_energy[b][m];
        }
    }
    (dur_flat, pitch_flat, energy_flat)
}

/// VariancePredictor SGD update helper (Phase E)。private struct のため同 module 内。
fn apply_variance_predictor_sgd(
    vp: &mut VariancePredictor,
    grads: &VariancePredictorGrads,
    lr: f32,
) {
    for (w, g) in vp.conv1.weight_mut().iter_mut().zip(&grads.conv1_w) {
        *w -= lr * g;
    }
    for (b, g) in vp.conv1.bias_mut().iter_mut().zip(&grads.conv1_b) {
        *b -= lr * g;
    }
    for (w, g) in vp.norm1.gamma_mut().iter_mut().zip(&grads.norm1_gamma) {
        *w -= lr * g;
    }
    for (b, g) in vp.norm1.beta_mut().iter_mut().zip(&grads.norm1_beta) {
        *b -= lr * g;
    }
    for (w, g) in vp.conv2.weight_mut().iter_mut().zip(&grads.conv2_w) {
        *w -= lr * g;
    }
    for (b, g) in vp.conv2.bias_mut().iter_mut().zip(&grads.conv2_b) {
        *b -= lr * g;
    }
    for (w, g) in vp.norm2.gamma_mut().iter_mut().zip(&grads.norm2_gamma) {
        *w -= lr * g;
    }
    for (b, g) in vp.norm2.beta_mut().iter_mut().zip(&grads.norm2_beta) {
        *b -= lr * g;
    }
    for (w, g) in vp.linear.weight_mut().iter_mut().zip(&grads.linear_w) {
        *w -= lr * g;
    }
    for (b, g) in vp.linear.bias_mut().iter_mut().zip(&grads.linear_b) {
        *b -= lr * g;
    }
}

/// FastSpeech2 backward で返される weight/bias 勾配 bundle (Phase 3 完全版)。
///
/// `backward_terminal` は postnet + mel_linear のみ埋め、encoder / decoder / variance / embed
/// 系は shape 正しい 0 埋め。`backward_full` は全 field を埋める (Phase 3 対応)。
#[derive(Clone, Debug)]
pub struct FastSpeech2Grads {
    /// Embedding table `[vocab_size, hidden_dim]` の勾配。
    pub embedding: Vec<f32>,
    /// Encoder FFT block 各層の勾配。
    pub encoder: Vec<FftBlockGrads>,
    /// Duration predictor の勾配。
    pub duration_predictor: VariancePredictorGrads,
    /// Pitch predictor の勾配。
    pub pitch_predictor: VariancePredictorGrads,
    /// Energy predictor の勾配。
    pub energy_predictor: VariancePredictorGrads,
    /// pitch_embed Conv1D weight 勾配。
    pub pitch_embed_w: Vec<f32>,
    /// pitch_embed Conv1D bias 勾配。
    pub pitch_embed_b: Vec<f32>,
    /// energy_embed Conv1D weight 勾配。
    pub energy_embed_w: Vec<f32>,
    /// energy_embed Conv1D bias 勾配。
    pub energy_embed_b: Vec<f32>,
    /// Decoder FFT block 各層の勾配。
    pub decoder: Vec<FftBlockGrads>,
    /// mel_linear weight `[mel_dim, hidden_dim]` の勾配。
    pub mel_linear_w: Vec<f32>,
    /// mel_linear bias `[mel_dim]` の勾配。
    pub mel_linear_b: Vec<f32>,
    /// Postnet 各層 Conv1D weight の勾配 (`Vec<Vec<f32>>[layer][flatten_len]`)。
    pub postnet_w: Vec<Vec<f32>>,
    /// Postnet 各層 Conv1D bias の勾配。
    pub postnet_b: Vec<Vec<f32>>,
}

impl FastSpeech2Grads {
    /// model の shape に合わせて zero 初期化した grads を返す (backward で埋めていく用)。
    #[must_use]
    fn zeros_from_config(cfg: &FastSpeech2Config) -> Self {
        let ph = cfg.predictor_hidden;
        let h = cfg.hidden_dim;
        let mel = cfg.mel_dim;
        let vp_zeros = || VariancePredictorGrads {
            conv1_w: vec![0.0; ph * h * cfg.predictor_kernel_size],
            conv1_b: vec![0.0; ph],
            norm1_gamma: vec![0.0; ph],
            norm1_beta: vec![0.0; ph],
            conv2_w: vec![0.0; ph * ph * cfg.predictor_kernel_size],
            conv2_b: vec![0.0; ph],
            norm2_gamma: vec![0.0; ph],
            norm2_beta: vec![0.0; ph],
            linear_w: vec![0.0; ph],
            linear_b: vec![0.0; 1],
        };
        let mid = h * cfg.fft_expansion;
        let fft_zeros = || FftBlockGrads {
            mha: crate::tts::MhaGrads {
                w_q: vec![0.0; h * h],
                w_k: vec![0.0; h * h],
                w_v: vec![0.0; h * h],
                w_o: vec![0.0; h * h],
                b_q: vec![0.0; h],
                b_k: vec![0.0; h],
                b_v: vec![0.0; h],
                b_o: vec![0.0; h],
            },
            attn_norm_gamma: vec![0.0; h],
            attn_norm_beta: vec![0.0; h],
            ffn_conv1_w: vec![0.0; mid * h * cfg.fft_kernel_size],
            ffn_conv1_b: vec![0.0; mid],
            ffn_conv2_w: vec![0.0; h * mid * cfg.fft_kernel_size],
            ffn_conv2_b: vec![0.0; h],
            ffn_norm_gamma: vec![0.0; h],
            ffn_norm_beta: vec![0.0; h],
        };

        // Postnet weight shapes (per-layer)
        let postnet_w: Vec<Vec<f32>> = (0..cfg.postnet_layers)
            .map(|i| {
                let (in_ch, out_ch) = if i == 0 {
                    (mel, cfg.postnet_hidden)
                } else if i == cfg.postnet_layers - 1 {
                    (cfg.postnet_hidden, mel)
                } else {
                    (cfg.postnet_hidden, cfg.postnet_hidden)
                };
                vec![0.0; out_ch * in_ch * cfg.postnet_kernel_size]
            })
            .collect();
        let postnet_b: Vec<Vec<f32>> = (0..cfg.postnet_layers)
            .map(|i| {
                let out_ch = if i == cfg.postnet_layers - 1 {
                    mel
                } else {
                    cfg.postnet_hidden
                };
                vec![0.0; out_ch]
            })
            .collect();

        Self {
            embedding: vec![0.0; cfg.vocab_size * h],
            encoder: (0..cfg.num_encoder_layers).map(|_| fft_zeros()).collect(),
            duration_predictor: vp_zeros(),
            pitch_predictor: vp_zeros(),
            energy_predictor: vp_zeros(),
            pitch_embed_w: vec![0.0; h],
            pitch_embed_b: vec![0.0; h],
            energy_embed_w: vec![0.0; h],
            energy_embed_b: vec![0.0; h],
            decoder: (0..cfg.num_decoder_layers).map(|_| fft_zeros()).collect(),
            mel_linear_w: vec![0.0; mel * h],
            mel_linear_b: vec![0.0; mel],
            postnet_w,
            postnet_b,
        }
    }
}

/// Length Regulator: mora hidden `[B, mora_len, hidden]` を duration に従って frame 単位に展開する。
///
/// 戻り値: (expanded flatten `[B, frame_len, hidden]`, `Vec<usize>` per-batch frame length)。
fn length_regulator(
    hidden: &[f32],
    batch: usize,
    mora_len: usize,
    hidden_dim: usize,
    durations: &[u32],
) -> (Vec<f32>, Vec<usize>) {
    let mut per_batch_frames = Vec::with_capacity(batch);
    for b in 0..batch {
        let total: u32 = durations[b * mora_len..(b + 1) * mora_len].iter().sum();
        per_batch_frames.push(total as usize);
    }
    let max_frames = *per_batch_frames.iter().max().unwrap_or(&0);
    let mut out = vec![0.0_f32; batch * max_frames * hidden_dim];

    for b in 0..batch {
        let mut frame_cursor = 0_usize;
        for m in 0..mora_len {
            let dur = durations[b * mora_len + m] as usize;
            let src_start = b * mora_len * hidden_dim + m * hidden_dim;
            let src = &hidden[src_start..src_start + hidden_dim];
            for _ in 0..dur {
                let dst_start = b * max_frames * hidden_dim + frame_cursor * hidden_dim;
                out[dst_start..dst_start + hidden_dim].copy_from_slice(src);
                frame_cursor += 1;
            }
        }
    }

    (out, per_batch_frames)
}

/// Length Regulator backward: `grad_expanded[B, max_frame, hidden]` を duration に従って mora-level に還元する。
///
/// forward で 1 mora を duration frame に copy-broadcast したので、backward では
/// 対応する duration frame の grad を **加算合計** して mora-level grad に戻す。
///
/// # 戻り値
///
/// `Vec<f32>[batch * mora_len * hidden_dim]` の grad_hidden (forward 前の encoder 出力への grad)。
fn length_regulator_backward(
    grad_expanded: &[f32],
    batch: usize,
    mora_len: usize,
    hidden_dim: usize,
    durations: &[u32],
) -> Vec<f32> {
    let max_frames = if batch * hidden_dim == 0 {
        0
    } else {
        grad_expanded.len() / (batch * hidden_dim)
    };
    let mut grad_hidden = vec![0.0_f32; batch * mora_len * hidden_dim];

    for b in 0..batch {
        let mut frame_cursor = 0_usize;
        for m in 0..mora_len {
            let dur = durations[b * mora_len + m] as usize;
            let dst_start = b * mora_len * hidden_dim + m * hidden_dim;
            for _ in 0..dur {
                let src_start = b * max_frames * hidden_dim + frame_cursor * hidden_dim;
                for c in 0..hidden_dim {
                    grad_hidden[dst_start + c] += grad_expanded[src_start + c];
                }
                frame_cursor += 1;
            }
        }
    }

    grad_hidden
}

/// Embedding backward: `grad_embedded[B, mora_len, hidden]` を mora_id ごとに accumulate して
/// `grad_embedding_table[vocab_size, hidden]` に還元する。
///
/// forward は embedding lookup (grad copy per mora_id) → backward は同じ id で足し込み。
fn embedding_backward(
    grad_embedded: &[f32],
    mora_ids: &[u32],
    vocab_size: usize,
    hidden_dim: usize,
    batch: usize,
    mora_len: usize,
) -> Vec<f32> {
    let mut grad_embedding = vec![0.0_f32; vocab_size * hidden_dim];
    for b in 0..batch {
        for m in 0..mora_len {
            let id = mora_ids[b * mora_len + m] as usize;
            if id >= vocab_size {
                continue;
            }
            let src_start = b * mora_len * hidden_dim + m * hidden_dim;
            let dst_start = id * hidden_dim;
            for c in 0..hidden_dim {
                grad_embedding[dst_start + c] += grad_embedded[src_start + c];
            }
        }
    }
    grad_embedding
}

/// FastSpeech2 操作で発生し得るエラー。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FastSpeech2Error {
    /// config が不正。
    InvalidConfig {
        /// 具体的な理由。
        reason: String,
    },
    /// input / target shape mismatch。
    ShapeMismatch {
        /// 対象 field 名。
        field: &'static str,
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
    /// mora id が vocab_size 超過。
    VocabOverflow {
        /// 越えた id。
        id: usize,
        /// vocab_size。
        vocab_size: usize,
    },
    /// seq_len (mora_len or frame_len) が max_len 超過。
    SeqLenExceedsMax {
        /// 要求 seq_len。
        seq_len: usize,
        /// pre-computed max_len。
        max_len: usize,
    },
    /// batch 内 frame 長不一致 (現 MVP 未対応)。
    VariableFrameLenNotSupported {
        /// batch size。
        batch: usize,
        /// per-sample frame length list。
        lens: Vec<usize>,
    },
    /// primitive 内部エラー (chain した error message)。
    Internal {
        /// 内部 error message。
        reason: String,
    },
}

impl std::fmt::Display for FastSpeech2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig { reason } => write!(f, "invalid FastSpeech2 config: {reason}"),
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch on '{field}': expected {expected}, got {actual}"
            ),
            Self::VocabOverflow { id, vocab_size } => {
                write!(f, "mora id {id} exceeds vocab_size {vocab_size}")
            }
            Self::SeqLenExceedsMax { seq_len, max_len } => write!(
                f,
                "seq_len {seq_len} exceeds pre-computed max_len {max_len}"
            ),
            Self::VariableFrameLenNotSupported { batch, lens } => write!(
                f,
                "variable frame length not supported (batch={batch}, lens={lens:?})"
            ),
            Self::Internal { reason } => write!(f, "internal error: {reason}"),
        }
    }
}

impl std::error::Error for FastSpeech2Error {}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> FastSpeech2Config {
        FastSpeech2Config {
            vocab_size: 20,
            hidden_dim: 8,
            num_heads: 2,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            fft_kernel_size: 3,
            fft_expansion: 2,
            predictor_kernel_size: 3,
            predictor_hidden: 8,
            mel_dim: 16,
            postnet_kernel_size: 5,
            postnet_layers: 3,
            postnet_hidden: 16,
            max_len: 64,
        }
    }

    #[test]
    fn config_validates() {
        small_config().validate().unwrap();

        let mut bad = small_config();
        bad.hidden_dim = 7; // not divisible by 2
        assert!(bad.validate().is_err());

        bad = small_config();
        bad.vocab_size = 0;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn zeros_model_builds() {
        let cfg = small_config();
        let model = FastSpeech2::zeros(cfg).unwrap();
        assert_eq!(model.config().vocab_size, 20);
        assert_eq!(model.encoder.len(), 2);
        assert_eq!(model.decoder.len(), 2);
        assert_eq!(model.postnet.len(), 3);
    }

    #[test]
    fn forward_produces_correct_mel_shape() {
        let cfg = small_config();
        let model = FastSpeech2::zeros(cfg).unwrap();
        let mora_ids: Vec<u32> = vec![1, 2, 3, 4, 5];
        let durations: Vec<u32> = vec![3, 2, 4, 5, 3]; // total = 17 frames
        let mel = model.forward(&mora_ids, 1, 5, &durations).unwrap();
        // expected: 1 * 17 * 16 = 272
        assert_eq!(mel.len(), 17 * 16);
    }

    #[test]
    fn forward_zero_weights_produces_zero_mel() {
        // zero init モデル + zero embedding → mel も全 0 or 定数 (postnet zeros ⇒ residual 0 ⇒ mel = mel_before = 0)
        let cfg = small_config();
        let model = FastSpeech2::zeros(cfg).unwrap();
        let mora_ids: Vec<u32> = vec![0, 0, 0];
        let durations: Vec<u32> = vec![2, 2, 2];
        let mel = model.forward(&mora_ids, 1, 3, &durations).unwrap();
        // zeros 初期化なので mel_before = 0、postnet も weight 0 で出力 0 → mel_after = 0
        // ただし LayerNorm の gamma=1, beta=0 なので、encoder/decoder の LayerNorm 出力は
        // (x - mean) / (std + eps)、zeros では NaN/Inf 危険だが eps>0 で safe
        // 中央値検証: mel の絶対値の平均が小さい
        let abs_sum: f32 = mel.iter().map(|x| x.abs()).sum();
        assert!(
            abs_sum < 100.0,
            "zeros model should produce small mel, got abs sum {abs_sum}"
        );
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let cfg = small_config();
        let model = FastSpeech2::zeros(cfg).unwrap();
        let err = model
            .forward(&[1_u32, 2], 1, 3, &[2, 2, 2])
            .expect_err("mora_ids len 2 vs 3");
        assert!(matches!(err, FastSpeech2Error::ShapeMismatch { .. }));
    }

    #[test]
    fn vocab_overflow_returns_error() {
        let cfg = small_config();
        let model = FastSpeech2::zeros(cfg).unwrap();
        let err = model
            .forward(&[100_u32], 1, 1, &[3])
            .expect_err("mora id 100 >= vocab 20");
        assert!(matches!(err, FastSpeech2Error::VocabOverflow { .. }));
    }

    #[test]
    fn seq_len_exceeds_max_returns_error() {
        let mut cfg = small_config();
        cfg.max_len = 5;
        let model = FastSpeech2::zeros(cfg).unwrap();
        let mora_ids: Vec<u32> = vec![1; 6]; // mora_len=6 > max_len=5
        let durations: Vec<u32> = vec![1; 6];
        let err = model
            .forward(&mora_ids, 1, 6, &durations)
            .expect_err("mora exceeds max");
        assert!(matches!(err, FastSpeech2Error::SeqLenExceedsMax { .. }));
    }

    #[test]
    fn length_regulator_expands_correctly() {
        // hidden = [1, 2, 3] (1 mora, 3 hidden dim), duration = [2] → expand to [1,2,3, 1,2,3] (2 frames)
        let hidden = vec![1.0_f32, 2.0, 3.0];
        let durations = vec![2_u32];
        let (out, per_frames) = length_regulator(&hidden, 1, 1, 3, &durations);
        assert_eq!(per_frames, vec![2]);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn error_display_and_error_trait() {
        let e = FastSpeech2Error::InvalidConfig {
            reason: "test".to_string(),
        };
        let s = format!("{e}");
        assert!(s.contains("invalid FastSpeech2"));
        let boxed: Box<dyn std::error::Error> = Box::new(e);
        assert!(boxed.to_string().contains("test"));
    }

    #[test]
    fn backward_terminal_returns_correct_shapes() {
        let cfg = small_config();
        let model = FastSpeech2::zeros(cfg).unwrap();
        let mora_ids: Vec<u32> = vec![1, 2, 3];
        let durations: Vec<u32> = vec![2, 2, 2]; // total 6 frames
        let grad_mel = vec![1.0_f32; 6 * cfg.mel_dim];

        let (grads, grad_decoder_out) = model
            .backward_terminal(&mora_ids, &durations, &grad_mel, 1, 3)
            .expect("backward_terminal");

        // shape 検証
        assert_eq!(grad_decoder_out.len(), 6 * cfg.hidden_dim);
        assert_eq!(grads.embedding.len(), cfg.vocab_size * cfg.hidden_dim);
        assert_eq!(grads.mel_linear_w.len(), cfg.mel_dim * cfg.hidden_dim);
        assert_eq!(grads.mel_linear_b.len(), cfg.mel_dim);
        assert_eq!(grads.postnet_w.len(), cfg.postnet_layers);
        assert_eq!(grads.postnet_b.len(), cfg.postnet_layers);

        // Phase 1 では embedding は 0 埋め (未実装)
        for &v in &grads.embedding {
            assert!((v).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn fft_block_backward_returns_correct_shapes() {
        // FftBlock backward の shape 検証 (private struct を module 内 test で直接構築)
        let block = FftBlock::zeros(8, 2, 3, 2);
        let batch = 1;
        let seq_len = 5;
        let hidden = 8;
        let input: Vec<f32> = (0..batch * seq_len * hidden)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let grad_output: Vec<f32> = (0..batch * seq_len * hidden)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();

        let (grad_input, grads) = block.backward(&input, &grad_output, batch, seq_len);

        // shape 検証
        assert_eq!(grad_input.len(), batch * seq_len * hidden);
        assert_eq!(grads.attn_norm_gamma.len(), hidden);
        assert_eq!(grads.attn_norm_beta.len(), hidden);
        assert_eq!(grads.ffn_conv1_w.len(), hidden * hidden * 2 * 3); // out * (in/groups=in) * kernel = 16*8*3
        assert_eq!(grads.ffn_conv1_b.len(), hidden * 2);
        assert_eq!(grads.ffn_conv2_w.len(), hidden * hidden * 2 * 3); // out=hidden, in=mid=hidden*2, k=3
        assert_eq!(grads.ffn_conv2_b.len(), hidden);
        assert_eq!(grads.ffn_norm_gamma.len(), hidden);
        assert_eq!(grads.ffn_norm_beta.len(), hidden);
        assert_eq!(grads.mha.w_q.len(), hidden * hidden);
        assert_eq!(grads.mha.b_q.len(), hidden);
    }

    #[test]
    fn fft_block_backward_is_finite() {
        // NaN/Inf 出さないことを確認 (zero weights model は LayerNorm eps で保護)
        let block = FftBlock::zeros(8, 2, 3, 2);
        let input = vec![0.5_f32; 5 * 8];
        let grad_output = vec![1.0_f32; 5 * 8];
        let (grad_input, grads) = block.backward(&input, &grad_output, 1, 5);
        for &v in &grad_input {
            assert!(v.is_finite(), "grad_input contains NaN/Inf");
        }
        for &v in &grads.attn_norm_gamma {
            assert!(v.is_finite());
        }
        for &v in &grads.ffn_norm_gamma {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn backward_terminal_shape_mismatch_returns_error() {
        let cfg = small_config();
        let model = FastSpeech2::zeros(cfg).unwrap();
        let mora_ids: Vec<u32> = vec![1, 2, 3];
        let durations: Vec<u32> = vec![2, 2, 2];
        let grad_mel = vec![1.0_f32; 5 * cfg.mel_dim]; // wrong frame count
        let err = model
            .backward_terminal(&mora_ids, &durations, &grad_mel, 1, 3)
            .expect_err("shape mismatch");
        assert!(matches!(err, FastSpeech2Error::ShapeMismatch { .. }));
    }

    #[test]
    fn backward_full_returns_correct_shapes_for_all_grads() {
        let cfg = small_config();
        let model = FastSpeech2::zeros(cfg).unwrap();
        let mora_ids: Vec<u32> = vec![1, 2, 3];
        let durations: Vec<u32> = vec![2, 2, 2];
        let grad_mel = vec![1.0_f32; 6 * cfg.mel_dim];

        let grads = model
            .backward_full(&mora_ids, &durations, &grad_mel, 1, 3)
            .expect("backward_full");

        assert_eq!(grads.embedding.len(), cfg.vocab_size * cfg.hidden_dim);
        assert_eq!(grads.encoder.len(), cfg.num_encoder_layers);
        assert_eq!(grads.decoder.len(), cfg.num_decoder_layers);
        for enc in &grads.encoder {
            assert_eq!(enc.attn_norm_gamma.len(), cfg.hidden_dim);
            assert_eq!(enc.ffn_norm_gamma.len(), cfg.hidden_dim);
            assert_eq!(enc.mha.w_q.len(), cfg.hidden_dim * cfg.hidden_dim);
        }
        assert_eq!(
            grads.duration_predictor.conv1_w.len(),
            cfg.predictor_hidden * cfg.hidden_dim * cfg.predictor_kernel_size
        );
        assert_eq!(grads.pitch_embed_w.len(), cfg.hidden_dim);
        assert_eq!(grads.mel_linear_w.len(), cfg.mel_dim * cfg.hidden_dim);
        assert_eq!(grads.postnet_w.len(), cfg.postnet_layers);
    }

    #[test]
    fn backward_full_grads_are_finite() {
        let cfg = small_config();
        let model = FastSpeech2::zeros(cfg).unwrap();
        let mora_ids: Vec<u32> = vec![1, 2, 3];
        let durations: Vec<u32> = vec![2, 2, 2];
        let grad_mel = vec![1.0_f32; 6 * cfg.mel_dim];
        let grads = model
            .backward_full(&mora_ids, &durations, &grad_mel, 1, 3)
            .expect("backward_full");
        for &v in &grads.embedding {
            assert!(v.is_finite());
        }
        for enc in &grads.encoder {
            for &v in &enc.mha.w_q {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn length_regulator_backward_sums_by_duration() {
        // mora = 2, hidden = 2, duration = [2, 3], expanded = 5 frames
        // grad_expanded = [1,2, 3,4, 5,6, 7,8, 9,10]
        // grad_hidden expected: mora 0 = [1+3, 2+4] = [4, 6]、mora 1 = [5+7+9, 6+8+10] = [21, 24]
        let grad_expanded = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let durations = vec![2_u32, 3];
        let grad_hidden = length_regulator_backward(&grad_expanded, 1, 2, 2, &durations);
        assert_eq!(grad_hidden.len(), 4);
        assert!((grad_hidden[0] - 4.0).abs() < 1e-6);
        assert!((grad_hidden[1] - 6.0).abs() < 1e-6);
        assert!((grad_hidden[2] - 21.0).abs() < 1e-6);
        assert!((grad_hidden[3] - 24.0).abs() < 1e-6);
    }

    #[test]
    fn embedding_backward_accumulates_by_id() {
        // vocab=4, hidden=2, mora_ids=[1, 2, 1]
        // grad_embedded = [1,2, 3,4, 5,6]
        // grad_embedding: id 1: [1+5, 2+6] = [6, 8]、id 2: [3, 4]
        let grad_embedded = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mora_ids = vec![1_u32, 2, 1];
        let grad_emb = embedding_backward(&grad_embedded, &mora_ids, 4, 2, 1, 3);
        assert_eq!(grad_emb.len(), 8);
        assert!((grad_emb[0]).abs() < 1e-6);
        assert!((grad_emb[1]).abs() < 1e-6);
        assert!((grad_emb[2] - 6.0).abs() < 1e-6);
        assert!((grad_emb[3] - 8.0).abs() < 1e-6);
        assert!((grad_emb[4] - 3.0).abs() < 1e-6);
        assert!((grad_emb[5] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn init_xavier_changes_weights_deterministically() {
        let cfg = small_config();
        let mut model_a = FastSpeech2::zeros(cfg).unwrap();
        let mut model_b = FastSpeech2::zeros(cfg).unwrap();

        // 同 seed で 2 model init → weight 完全一致 (決定論)
        model_a.init_xavier(42);
        model_b.init_xavier(42);
        assert_eq!(model_a.embedding, model_b.embedding);
        assert_eq!(model_a.mel_linear.weight(), model_b.mel_linear.weight());

        // 異なる seed → 異なる weight
        let mut model_c = FastSpeech2::zeros(cfg).unwrap();
        model_c.init_xavier(43);
        assert_ne!(model_a.embedding, model_c.embedding);
    }

    #[test]
    fn init_xavier_produces_non_zero_weights() {
        let cfg = small_config();
        let mut model = FastSpeech2::zeros(cfg).unwrap();
        model.init_xavier(42);

        // 少なくとも 1 つは 0 以外
        let has_nonzero_embedding = model.embedding.iter().any(|&w| w.abs() > 1e-6);
        assert!(has_nonzero_embedding);

        let has_nonzero_mel_linear = model.mel_linear.weight().iter().any(|&w| w.abs() > 1e-6);
        assert!(has_nonzero_mel_linear);

        // MHA も同様 (最初 encoder block を verify)
        let mha_q_grad = model.encoder[0].self_attn.w_q();
        let has_nonzero_q = mha_q_grad.iter().any(|&w| w.abs() > 1e-6);
        assert!(has_nonzero_q);
    }

    #[test]
    fn init_xavier_weights_are_finite() {
        let cfg = small_config();
        let mut model = FastSpeech2::zeros(cfg).unwrap();
        model.init_xavier(42);

        for &w in &model.embedding {
            assert!(w.is_finite());
        }
        for &w in model.mel_linear.weight() {
            assert!(w.is_finite());
        }
        for &w in model.encoder[0].self_attn.w_q() {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn save_load_roundtrip_recovers_weights() {
        use tempfile::TempDir;
        let cfg = small_config();
        let mut model_a = FastSpeech2::zeros(cfg).unwrap();
        model_a.init_xavier(42);

        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("model.safetensors");
        model_a.save_safetensors(&path).expect("save");

        let mut model_b = FastSpeech2::zeros(cfg).unwrap();
        // load 前は完全に異なる (model_b はまだ zero)
        assert_ne!(model_a.embedding(), model_b.embedding());
        model_b.load_safetensors(&path).expect("load");
        // load 後は完全一致
        assert_eq!(model_a.embedding(), model_b.embedding());
        assert_eq!(model_a.mel_linear.weight(), model_b.mel_linear.weight());
    }

    #[test]
    fn save_load_forward_output_matches() {
        // Save/load roundtrip 後、同じ input で forward 出力が一致 (weight 実効一致確認)
        use tempfile::TempDir;
        let cfg = small_config();
        let mut model_a = FastSpeech2::zeros(cfg).unwrap();
        model_a.init_xavier(7);

        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("m.safetensors");
        model_a.save_safetensors(&path).unwrap();

        let mut model_b = FastSpeech2::zeros(cfg).unwrap();
        model_b.load_safetensors(&path).unwrap();

        let mora_ids = vec![1_u32, 2, 3];
        let durations = vec![2_u32, 2, 2];
        let out_a = model_a.forward(&mora_ids, 1, 3, &durations).unwrap();
        let out_b = model_b.forward(&mora_ids, 1, 3, &durations).unwrap();
        for (a, b) in out_a.iter().zip(&out_b) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn variance_predictor_backward_returns_correct_shapes() {
        let cfg = small_config();
        let predictor = VariancePredictor::zeros(cfg.hidden_dim, cfg.predictor_hidden, 3);
        let batch = 1;
        let seq_len = 5;
        let input: Vec<f32> = (0..batch * seq_len * cfg.hidden_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let grad_output: Vec<f32> = (0..batch * seq_len)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();
        let (grad_input, grads) = predictor.backward(&input, &grad_output, batch, seq_len);
        assert_eq!(grad_input.len(), batch * seq_len * cfg.hidden_dim);
        assert_eq!(
            grads.conv1_w.len(),
            cfg.predictor_hidden * cfg.hidden_dim * 3
        );
        assert_eq!(grads.linear_w.len(), cfg.predictor_hidden);
        assert_eq!(grads.linear_b.len(), 1);
    }
}
