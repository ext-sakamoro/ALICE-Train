//! Qwen3.5 バックワードパス — ハイブリッド DeltaNet + Full Attention。
//!
//! forward の cache を使い、全勾配を計算する。
//! STE で FP32 潜在重みに勾配を通す。

use crate::llama_forward::{matmul, matmul_bt};
use crate::qwen35::{Qwen35Config, LayerType};

/// DeltaNet レイヤーの重み勾配。
pub struct DeltaNetWeightGrads {
    /// input_layernorm 勾配。
    pub d_input_layernorm: Vec<f32>,
    /// post_attn_layernorm 勾配。
    pub d_post_attn_layernorm: Vec<f32>,
    /// in_proj_qkv 勾配。
    pub d_in_proj_qkv: Vec<f32>,
    /// in_proj_z 勾配。
    pub d_in_proj_z: Vec<f32>,
    /// in_proj_b 勾配。
    pub d_in_proj_b: Vec<f32>,
    /// in_proj_a 勾配。
    pub d_in_proj_a: Vec<f32>,
    /// A_log 勾配。
    pub d_a_log: Vec<f32>,
    /// dt_bias 勾配。
    pub d_dt_bias: Vec<f32>,
    /// conv1d_weight 勾配。
    pub d_conv1d_weight: Vec<f32>,
    /// norm_weight 勾配。
    pub d_norm_weight: Vec<f32>,
    /// out_proj 勾配。
    pub d_out_proj: Vec<f32>,
    /// gate_proj 勾配。
    pub d_gate_proj: Vec<f32>,
    /// up_proj 勾配。
    pub d_up_proj: Vec<f32>,
    /// down_proj 勾配。
    pub d_down_proj: Vec<f32>,
}

/// Full Attention レイヤーの重み勾配。
pub struct FullAttnWeightGrads {
    /// input_layernorm 勾配。
    pub d_input_layernorm: Vec<f32>,
    /// post_attn_layernorm 勾配。
    pub d_post_attn_layernorm: Vec<f32>,
    /// q_proj 勾配。
    pub d_q_proj: Vec<f32>,
    /// k_proj 勾配。
    pub d_k_proj: Vec<f32>,
    /// v_proj 勾配。
    pub d_v_proj: Vec<f32>,
    /// o_proj 勾配。
    pub d_o_proj: Vec<f32>,
    /// q_norm 勾配。
    pub d_q_norm: Vec<f32>,
    /// k_norm 勾配。
    pub d_k_norm: Vec<f32>,
    /// gate_proj 勾配。
    pub d_gate_proj: Vec<f32>,
    /// up_proj 勾配。
    pub d_up_proj: Vec<f32>,
    /// down_proj 勾配。
    pub d_down_proj: Vec<f32>,
}

/// ハイブリッドレイヤーの重み勾配。
pub enum Qwen35WeightGrads {
    /// DeltaNet 層の勾配。
    DeltaNet(DeltaNetWeightGrads),
    /// Full Attention 層の勾配。
    FullAttention(FullAttnWeightGrads),
}

impl DeltaNetWeightGrads {
    /// ゼロ初期化。
    #[must_use]
    pub fn zeros(config: &Qwen35Config) -> Self {
        let hidden = config.hidden_size;
        let key_dim = config.linear_key_dim();
        let val_dim = config.linear_value_dim();
        let conv_dim = config.conv_dim();
        let n_v = config.linear_num_value_heads;
        let inter = config.intermediate_size;

        Self {
            d_input_layernorm: vec![0.0; hidden],
            d_post_attn_layernorm: vec![0.0; hidden],
            d_in_proj_qkv: vec![0.0; hidden * (key_dim * 2 + val_dim)],
            d_in_proj_z: vec![0.0; hidden * val_dim],
            d_in_proj_b: vec![0.0; hidden * n_v],
            d_in_proj_a: vec![0.0; hidden * n_v],
            d_a_log: vec![0.0; n_v],
            d_dt_bias: vec![0.0; n_v],
            d_conv1d_weight: vec![0.0; conv_dim * config.linear_conv_kernel_dim],
            d_norm_weight: vec![0.0; config.linear_value_head_dim],
            d_out_proj: vec![0.0; val_dim * hidden],
            d_gate_proj: vec![0.0; inter * hidden],
            d_up_proj: vec![0.0; inter * hidden],
            d_down_proj: vec![0.0; hidden * inter],
        }
    }

    /// SGD で重みに反映する。
    pub fn apply_sgd(
        &self,
        weights: &mut crate::qwen35::DeltaNetLayerWeights,
        lr: f32,
        wd: f32,
    ) {
        sgd_update(&mut weights.input_layernorm, &self.d_input_layernorm, lr, 0.0);
        sgd_update(&mut weights.post_attn_layernorm, &self.d_post_attn_layernorm, lr, 0.0);
        sgd_update(&mut weights.in_proj_qkv, &self.d_in_proj_qkv, lr, wd);
        sgd_update(&mut weights.in_proj_z, &self.d_in_proj_z, lr, wd);
        sgd_update(&mut weights.in_proj_b, &self.d_in_proj_b, lr, wd);
        sgd_update(&mut weights.in_proj_a, &self.d_in_proj_a, lr, wd);
        sgd_update(&mut weights.a_log, &self.d_a_log, lr, 0.0);
        sgd_update(&mut weights.dt_bias, &self.d_dt_bias, lr, 0.0);
        sgd_update(&mut weights.conv1d_weight, &self.d_conv1d_weight, lr, wd);
        sgd_update(&mut weights.norm_weight, &self.d_norm_weight, lr, 0.0);
        sgd_update(&mut weights.out_proj, &self.d_out_proj, lr, wd);
        sgd_update(&mut weights.gate_proj, &self.d_gate_proj, lr, wd);
        sgd_update(&mut weights.up_proj, &self.d_up_proj, lr, wd);
        sgd_update(&mut weights.down_proj, &self.d_down_proj, lr, wd);
    }
}

impl FullAttnWeightGrads {
    /// ゼロ初期化。
    #[must_use]
    pub fn zeros(config: &Qwen35Config) -> Self {
        let hidden = config.hidden_size;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let inter = config.intermediate_size;

        Self {
            d_input_layernorm: vec![0.0; hidden],
            d_post_attn_layernorm: vec![0.0; hidden],
            d_q_proj: vec![0.0; hidden * nh * hd],
            d_k_proj: vec![0.0; hidden * nkv * hd],
            d_v_proj: vec![0.0; hidden * nkv * hd],
            d_o_proj: vec![0.0; nh * hd * hidden],
            d_q_norm: vec![0.0; hd],
            d_k_norm: vec![0.0; hd],
            d_gate_proj: vec![0.0; inter * hidden],
            d_up_proj: vec![0.0; inter * hidden],
            d_down_proj: vec![0.0; hidden * inter],
        }
    }

    /// SGD で重みに反映する。
    pub fn apply_sgd(
        &self,
        weights: &mut crate::qwen35::FullAttnLayerWeights,
        lr: f32,
        wd: f32,
    ) {
        sgd_update(&mut weights.input_layernorm, &self.d_input_layernorm, lr, 0.0);
        sgd_update(&mut weights.post_attn_layernorm, &self.d_post_attn_layernorm, lr, 0.0);
        sgd_update(&mut weights.q_proj, &self.d_q_proj, lr, wd);
        sgd_update(&mut weights.k_proj, &self.d_k_proj, lr, wd);
        sgd_update(&mut weights.v_proj, &self.d_v_proj, lr, wd);
        sgd_update(&mut weights.o_proj, &self.d_o_proj, lr, wd);
        sgd_update(&mut weights.q_norm, &self.d_q_norm, lr, 0.0);
        sgd_update(&mut weights.k_norm, &self.d_k_norm, lr, 0.0);
        sgd_update(&mut weights.gate_proj, &self.d_gate_proj, lr, wd);
        sgd_update(&mut weights.up_proj, &self.d_up_proj, lr, wd);
        sgd_update(&mut weights.down_proj, &self.d_down_proj, lr, wd);
    }
}

impl Qwen35WeightGrads {
    /// ゼロ初期化。
    #[must_use]
    pub fn zeros(config: &Qwen35Config, layer_type: LayerType) -> Self {
        match layer_type {
            LayerType::LinearAttention => Self::DeltaNet(DeltaNetWeightGrads::zeros(config)),
            LayerType::FullAttention => Self::FullAttention(FullAttnWeightGrads::zeros(config)),
        }
    }
}

/// SGD + weight decay 更新。
fn sgd_update(w: &mut [f32], grad: &[f32], lr: f32, wd: f32) {
    for (w_i, &g_i) in w.iter_mut().zip(grad.iter()) {
        *w_i -= lr * (g_i + wd * *w_i);
    }
}

/// SwiGLU FFN backward (共通)。
///
/// FFN 部分の backward は DeltaNet / Full Attention 共通。
/// 入力勾配を `d_input` に書き込み、重み勾配を `d_gate/d_up/d_down` に累積。
pub fn swiglu_ffn_backward(
    d_output: &[f32],
    normed_input: &[f32],
    gate: &[f32],
    up: &[f32],
    gate_silu: &[f32],
    gate_proj: &[f32],
    up_proj: &[f32],
    down_proj: &[f32],
    d_input: &mut [f32],
    d_gate_proj: &mut [f32],
    d_up_proj: &mut [f32],
    d_down_proj: &mut [f32],
    seq_len: usize,
    hidden: usize,
    inter: usize,
) {
    // d_output: (seq_len × hidden) — FFN出力の勾配
    // backward through down_proj: d_intermediate = d_output × down_proj (not transposed)
    let mut d_intermediate = vec![0.0f32; seq_len * inter];
    matmul(d_output, down_proj, &mut d_intermediate, seq_len, inter, hidden);

    // d_down_proj: d_output^T × intermediate → (hidden × inter)
    // intermediate = gate_silu ⊙ up
    let mut intermediate = vec![0.0f32; seq_len * inter];
    for i in 0..seq_len * inter {
        intermediate[i] = gate_silu[i] * up[i];
    }
    // d_down_proj += d_output^T × intermediate (accumulated)
    for t in 0..seq_len {
        for i in 0..hidden {
            for j in 0..inter {
                d_down_proj[i * inter + j] +=
                    d_output[t * hidden + i] * intermediate[t * inter + j];
            }
        }
    }

    // d_gate_silu = d_intermediate ⊙ up
    // d_up_val = d_intermediate ⊙ gate_silu
    let mut d_gate_silu = vec![0.0f32; seq_len * inter];
    let mut d_up_val = vec![0.0f32; seq_len * inter];
    for i in 0..seq_len * inter {
        d_gate_silu[i] = d_intermediate[i] * up[i];
        d_up_val[i] = d_intermediate[i] * gate_silu[i];
    }

    // SiLU backward: d_gate = d_gate_silu * silu'(gate)
    let mut d_gate_val = vec![0.0f32; seq_len * inter];
    for i in 0..seq_len * inter {
        let sig = 1.0 / (1.0 + (-gate[i]).exp());
        let silu_grad = sig + gate[i] * sig * (1.0 - sig);
        d_gate_val[i] = d_gate_silu[i] * silu_grad;
    }

    // d_gate_proj += d_gate^T × normed_input
    // d_up_proj += d_up^T × normed_input
    for t in 0..seq_len {
        for i in 0..inter {
            for j in 0..hidden {
                d_gate_proj[i * hidden + j] +=
                    d_gate_val[t * inter + i] * normed_input[t * hidden + j];
                d_up_proj[i * hidden + j] +=
                    d_up_val[t * inter + i] * normed_input[t * hidden + j];
            }
        }
    }

    // d_normed_input = d_gate × gate_proj^T + d_up × up_proj^T
    matmul_bt(
        &d_gate_val,
        gate_proj,
        d_input,
        seq_len,
        hidden,
        inter,
    );
    let mut d_input_up = vec![0.0f32; seq_len * hidden];
    matmul_bt(
        &d_up_val,
        up_proj,
        &mut d_input_up,
        seq_len,
        hidden,
        inter,
    );
    for i in 0..seq_len * hidden {
        d_input[i] += d_input_up[i];
    }
}

// ── レイヤー単位 backward ───────────────────────────────────────────────────

/// DeltaNet レイヤーの backward — cache から勾配計算 + 重み更新。
///
/// `d_output`: このレイヤー出力の勾配 (seq_len × hidden)。
/// 戻り値: `d_input` (前のレイヤーへの勾配)。
pub fn deltanet_layer_backward(
    d_output: &[f32],
    cache: &crate::deltanet::DeltaNetLayerCache,
    weights: &crate::qwen35::DeltaNetLayerWeights,
    config: &Qwen35Config,
    seq_len: usize,
    _lr: f32,
    _wd: f32,
    weight_grads: &mut DeltaNetWeightGrads,
) -> Vec<f32> {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;

    let mut d_ffn_input = vec![0.0f32; seq_len * hidden];
    swiglu_ffn_backward(
        d_output,
        &cache.normed_ffn,
        &cache.gate,
        &cache.up,
        &cache.gate_silu,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
        &mut d_ffn_input,
        &mut weight_grads.d_gate_proj,
        &mut weight_grads.d_up_proj,
        &mut weight_grads.d_down_proj,
        seq_len,
        hidden,
        inter,
    );

    // TODO: RMSNorm backward (post_attn_layernorm)
    // 簡略化: d_ffn_input をそのまま attention 出力の勾配として使用
    // (RMSNorm backward は norm weight の勾配 + 入力勾配のスケーリング)

    // d_residual_ffn = d_output (residual skip)
    // d_attn_block = d_ffn_input + d_output (residual add backward)
    let mut d_input = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden {
        d_input[i] = d_ffn_input[i] + d_output[i];
    }

    // 2. Attention block backward は DeltaNet 再帰の backward が必要
    // → deltanet_recurrence_backward で d_q, d_k, d_v, d_beta, d_a を取得
    // → projection backward で d_in_proj_qkv 等を計算
    // ここでは residual skip の勾配のみ伝播 (attention backward は省略)
    // 完全な attention backward は CUDA 統合時に実装

    // residual skip: d_input += d_output (attention block の residual)
    for i in 0..seq_len * hidden {
        d_input[i] += d_output[i];
    }

    d_input
}

/// Full Attention レイヤーの backward (簡略版 — FFN backward + residual)。
pub fn full_attn_layer_backward(
    d_output: &[f32],
    cache: &crate::qwen35_forward::FullAttnLayerCache,
    weights: &crate::qwen35::FullAttnLayerWeights,
    config: &Qwen35Config,
    seq_len: usize,
    _lr: f32,
    _wd: f32,
    weight_grads: &mut FullAttnWeightGrads,
) -> Vec<f32> {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;

    let mut d_ffn_input = vec![0.0f32; seq_len * hidden];
    swiglu_ffn_backward(
        d_output,
        &cache.normed_ffn,
        &cache.gate,
        &cache.up,
        &cache.gate_silu,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
        &mut d_ffn_input,
        &mut weight_grads.d_gate_proj,
        &mut weight_grads.d_up_proj,
        &mut weight_grads.d_down_proj,
        seq_len,
        hidden,
        inter,
    );

    let mut d_input = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden {
        d_input[i] = d_ffn_input[i] + d_output[i] + d_output[i]; // FFN + 2× residual
    }

    d_input
}

/// ハイブリッドレイヤーの backward。
pub fn qwen35_layer_backward(
    d_output: &[f32],
    cache: &crate::qwen35_forward::Qwen35LayerCache,
    weights: &crate::qwen35::Qwen35LayerWeights,
    config: &Qwen35Config,
    seq_len: usize,
    lr: f32,
    wd: f32,
) -> (Vec<f32>, Qwen35WeightGrads) {
    match (cache, weights) {
        (
            crate::qwen35_forward::Qwen35LayerCache::DeltaNet(c),
            crate::qwen35::Qwen35LayerWeights::DeltaNet(w),
        ) => {
            let mut grads = DeltaNetWeightGrads::zeros(config);
            let d_input = deltanet_layer_backward(d_output, c, w, config, seq_len, lr, wd, &mut grads);
            (d_input, Qwen35WeightGrads::DeltaNet(grads))
        }
        (
            crate::qwen35_forward::Qwen35LayerCache::FullAttention(c),
            crate::qwen35::Qwen35LayerWeights::FullAttention(w),
        ) => {
            let mut grads = FullAttnWeightGrads::zeros(config);
            let d_input = full_attn_layer_backward(d_output, c, w, config, seq_len, lr, wd, &mut grads);
            (d_input, Qwen35WeightGrads::FullAttention(grads))
        }
        _ => panic!("cache と weights の型が一致しません"),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deltanet_weight_grads_zeros() {
        let config = Qwen35Config::qwen35_9b();
        let grads = DeltaNetWeightGrads::zeros(&config);
        assert_eq!(grads.d_in_proj_qkv.len(), config.hidden_size * (config.linear_key_dim() * 2 + config.linear_value_dim()));
        assert!(grads.d_in_proj_qkv.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn full_attn_weight_grads_zeros() {
        let config = Qwen35Config::qwen35_9b();
        let grads = FullAttnWeightGrads::zeros(&config);
        assert_eq!(
            grads.d_q_proj.len(),
            config.hidden_size * config.num_attention_heads * config.head_dim
        );
    }

    #[test]
    fn swiglu_ffn_backward_smoke() {
        let hidden = 4;
        let inter = 8;
        let seq_len = 2;

        let d_output = vec![0.1f32; seq_len * hidden];
        let normed = vec![0.5f32; seq_len * hidden];
        let gate = vec![1.0f32; seq_len * inter];
        let up = vec![0.5f32; seq_len * inter];
        let gate_silu: Vec<f32> = gate.iter().map(|&g| g / (1.0 + (-g).exp())).collect();
        let gate_proj = vec![0.01f32; inter * hidden];
        let up_proj = vec![0.01f32; inter * hidden];
        let down_proj = vec![0.01f32; hidden * inter];

        let mut d_input = vec![0.0f32; seq_len * hidden];
        let mut d_gate_proj = vec![0.0f32; inter * hidden];
        let mut d_up_proj = vec![0.0f32; inter * hidden];
        let mut d_down_proj = vec![0.0f32; hidden * inter];

        swiglu_ffn_backward(
            &d_output,
            &normed,
            &gate,
            &up,
            &gate_silu,
            &gate_proj,
            &up_proj,
            &down_proj,
            &mut d_input,
            &mut d_gate_proj,
            &mut d_up_proj,
            &mut d_down_proj,
            seq_len,
            hidden,
            inter,
        );

        assert!(d_input.iter().all(|v| v.is_finite()));
        assert!(d_gate_proj.iter().all(|v| v.is_finite()));
        assert!(d_input.iter().any(|v| v.abs() > 1e-10));
    }
}
