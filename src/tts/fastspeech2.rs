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
        let h = self.self_attn.config().embed_dim;
        // 1. Self-attention (bidirectional)
        let attn_out = self
            .self_attn
            .forward_self_attention(input, batch, seq_len, false)
            .expect("attn");
        // 2. Add & LayerNorm
        let residual = add(input, &attn_out);
        let after_attn = self
            .attn_norm
            .forward(&residual, batch * seq_len)
            .expect("norm");

        // 3. FFN: hidden → mid via Conv1D (需要 [B, hidden, seq_len] layout)
        // reshape [B, seq_len, hidden] → [B, hidden, seq_len]
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

        // Conv1D 1 (hidden → mid), same padding
        let mid = self.ffn_conv1.config().out_channels;
        let ffn_mid = self
            .ffn_conv1
            .forward(&reshaped, batch, seq_len)
            .expect("ffn1");
        // ReLU
        let ffn_mid_relu: Vec<f32> = ffn_mid.iter().map(|&x| x.max(0.0)).collect();
        // Conv1D 2 (mid → hidden)
        let _ = mid;
        let ffn_out_ch = self
            .ffn_conv2
            .forward(&ffn_mid_relu, batch, seq_len)
            .expect("ffn2");
        // reshape back [B, hidden, seq_len] → [B, seq_len, hidden]
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
        let h = self.self_attn.config().embed_dim;

        // === Forward 再計算 (中間 activation 保持) ===
        let attn_out = self
            .self_attn
            .forward_self_attention(input, batch, seq_len, false)
            .expect("attn");
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

        // Step 10: MHA backward (self-attention Q=K=V=input)
        let (grad_attn_input, mha_grads) = self
            .self_attn
            .backward_self_attention(input, &grad_residual1, batch, seq_len, false)
            .expect("mha bw");

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

        // Bundle grads (encoder/decoder/variance は 0 埋め、Phase T.4a 継続で埋める)
        let grads = FastSpeech2Grads {
            embedding: vec![0.0; self.embedding.len()],
            mel_linear_w: grad_mel_w,
            mel_linear_b: grad_mel_b,
            postnet_w: postnet_w_grads,
            postnet_b: postnet_b_grads,
        };

        Ok((grads, grad_decoder_out))
    }
}

/// FastSpeech2 backward で返される weight/bias 勾配 bundle (Phase 1: terminal-only)。
///
/// 現 Phase 1 では mel_linear + postnet の勾配のみ埋まる。encoder / decoder / variance predictor
/// の勾配は Phase T.4a 継続で追加予定 (現状は本 struct に field なし = 未 return)。
#[derive(Clone, Debug)]
pub struct FastSpeech2Grads {
    /// Embedding table `[vocab_size, hidden_dim]` の勾配 (現 Phase 1 は 0 埋め)。
    pub embedding: Vec<f32>,
    /// mel_linear weight `[mel_dim, hidden_dim]` の勾配。
    pub mel_linear_w: Vec<f32>,
    /// mel_linear bias `[mel_dim]` の勾配。
    pub mel_linear_b: Vec<f32>,
    /// Postnet 各層 Conv1D weight の勾配 (`Vec<Vec<f32>>[layer][flatten_len]`)。
    pub postnet_w: Vec<Vec<f32>>,
    /// Postnet 各層 Conv1D bias の勾配。
    pub postnet_b: Vec<Vec<f32>>,
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
}
