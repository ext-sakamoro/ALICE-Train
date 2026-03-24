//! Llama-3 バックワードパス — STE 付き QAT 学習用。
//!
//! `llama_forward` の `LayerCache` を使い、全勾配を計算する。
//! weight は FP32 潜在重みに対して STE で勾配を通す。

use crate::llama::{LlamaConfig, LlamaLayerWeights};
use crate::llama_forward::LayerCache;
use rayon::prelude::*;

/// 1レイヤーの重み勾配。
pub struct LayerWeightGrads {
    /// attention norm 勾配。
    pub d_attn_norm: Vec<f32>,
    /// Q projection 勾配。
    pub d_q_proj: Vec<f32>,
    /// K projection 勾配。
    pub d_k_proj: Vec<f32>,
    /// V projection 勾配。
    pub d_v_proj: Vec<f32>,
    /// O projection 勾配。
    pub d_o_proj: Vec<f32>,
    /// Q bias 勾配 (attention_bias=true の場合のみ Some)。
    pub d_q_bias: Option<Vec<f32>>,
    /// K bias 勾配。
    pub d_k_bias: Option<Vec<f32>>,
    /// V bias 勾配。
    pub d_v_bias: Option<Vec<f32>>,
    /// FFN norm 勾配。
    pub d_ffn_norm: Vec<f32>,
    /// gate projection 勾配。
    pub d_gate_proj: Vec<f32>,
    /// up projection 勾配。
    pub d_up_proj: Vec<f32>,
    /// down projection 勾配。
    pub d_down_proj: Vec<f32>,
}

impl LayerWeightGrads {
    /// ゼロ初期化。
    #[must_use]
    pub fn zeros(config: &LlamaConfig) -> Self {
        let hidden = config.hidden_dim;
        let inter = config.intermediate_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let (d_q_bias, d_k_bias, d_v_bias) = if config.attention_bias {
            (
                Some(vec![0.0; config.num_heads * config.head_dim]),
                Some(vec![0.0; kv_dim]),
                Some(vec![0.0; kv_dim]),
            )
        } else {
            (None, None, None)
        };
        Self {
            d_attn_norm: vec![0.0; hidden],
            d_q_proj: vec![0.0; hidden * hidden],
            d_k_proj: vec![0.0; kv_dim * hidden],
            d_v_proj: vec![0.0; kv_dim * hidden],
            d_o_proj: vec![0.0; hidden * hidden],
            d_q_bias,
            d_k_bias,
            d_v_bias,
            d_ffn_norm: vec![0.0; hidden],
            d_gate_proj: vec![0.0; inter * hidden],
            d_up_proj: vec![0.0; inter * hidden],
            d_down_proj: vec![0.0; hidden * inter],
        }
    }

    /// 勾配を SGD で重みに反映する。
    pub fn apply_sgd(&self, weights: &mut LlamaLayerWeights, lr: f32, weight_decay: f32) {
        sgd_update(&mut weights.attn_norm, &self.d_attn_norm, lr, weight_decay);
        sgd_update(&mut weights.q_proj, &self.d_q_proj, lr, weight_decay);
        sgd_update(&mut weights.k_proj, &self.d_k_proj, lr, weight_decay);
        sgd_update(&mut weights.v_proj, &self.d_v_proj, lr, weight_decay);
        sgd_update(&mut weights.o_proj, &self.d_o_proj, lr, weight_decay);
        sgd_update(&mut weights.ffn_norm, &self.d_ffn_norm, lr, weight_decay);
        sgd_update(&mut weights.gate_proj, &self.d_gate_proj, lr, weight_decay);
        sgd_update(&mut weights.up_proj, &self.d_up_proj, lr, weight_decay);
        sgd_update(&mut weights.down_proj, &self.d_down_proj, lr, weight_decay);
        // Bias は weight_decay なしで更新（bias に正則化は不適用）
        if let (Some(ref mut wb), Some(ref db)) = (&mut weights.q_bias, &self.d_q_bias) {
            sgd_update(wb, db, lr, 0.0);
        }
        if let (Some(ref mut wb), Some(ref db)) = (&mut weights.k_bias, &self.d_k_bias) {
            sgd_update(wb, db, lr, 0.0);
        }
        if let (Some(ref mut wb), Some(ref db)) = (&mut weights.v_bias, &self.d_v_bias) {
            sgd_update(wb, db, lr, 0.0);
        }
    }
}

/// SGD + weight decay 更新。
fn sgd_update(w: &mut [f32], grad: &[f32], lr: f32, wd: f32) {
    for (w_i, &g_i) in w.iter_mut().zip(grad.iter()) {
        *w_i -= lr * (g_i + wd * *w_i);
    }
}

// ── 行列 backward ──────────────────────────────────────────────────────────

/// matmul_bt backward: C = A × B^T の勾配。
///
/// dA = dC × B     (m × k)
/// dB = dC^T × A   (n × k)  ← dC^T は (n × m), A は (m × k)
pub fn matmul_bt_backward(
    d_output: &[f32], // (m × n)
    a: &[f32],        // (m × k)
    b: &[f32],        // (n × k)
    d_a: &mut [f32],  // (m × k)
    d_b: &mut [f32],  // (n × k)
    m: usize,
    n: usize,
    k: usize,
) {
    // dA = dC × B  (m×n) × (n×k) → (m×k)
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0f32;
            for h in 0..n {
                sum = d_output[i * n + h].mul_add(b[h * k + j], sum);
            }
            d_a[i * k + j] += sum;
        }
    }

    // dB = dC^T × A  (n×m) × (m×k) → (n×k)
    for i in 0..n {
        for j in 0..k {
            let mut sum = 0.0f32;
            for h in 0..m {
                sum = d_output[h * n + i].mul_add(a[h * k + j], sum);
            }
            d_b[i * k + j] += sum;
        }
    }
}

// ── RMSNorm backward ───────────────────────────────────────────────────────

/// RMSNorm backward。
///
/// `d_output`: 出力勾配 (seq_len × dim)
/// `input`: forward 前の入力 (seq_len × dim)
/// `weight`: norm weight (dim)
/// `d_input`: 入力勾配 (seq_len × dim) — 累積
/// `d_weight`: 重み勾配 (dim) — 累積
pub fn rmsnorm_backward(
    d_output: &[f32],
    input: &[f32],
    weight: &[f32],
    d_input: &mut [f32],
    d_weight: &mut [f32],
    dim: usize,
    eps: f32,
) {
    let seq_len = input.len() / dim;

    // d_input: トークンごとに独立 → 並列化
    d_input
        .par_chunks_exact_mut(dim)
        .enumerate()
        .for_each(|(t, d_in_row)| {
            let off = t * dim;
            let x = &input[off..off + dim];
            let dy = &d_output[off..off + dim];

            let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let inv_rms = (1.0 / (ss / dim as f64 + eps as f64).sqrt()) as f32;
            let inv_rms3 = inv_rms * inv_rms * inv_rms;

            let d_rms_sum: f32 = (0..dim).map(|d| dy[d] * weight[d] * x[d]).sum();

            for d in 0..dim {
                let d_norm = dy[d] * weight[d];
                d_in_row[d] += d_norm * inv_rms - d_rms_sum * x[d] * inv_rms3 / dim as f32;
            }
        });

    // d_weight: 全トークンから累積 — par_chunks → thread-local 集約
    let partial: Vec<f32> = (0..seq_len)
        .into_par_iter()
        .fold(
            || vec![0.0f32; dim],
            |mut acc, t| {
                let off = t * dim;
                let x = &input[off..off + dim];
                let dy = &d_output[off..off + dim];
                let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let inv_rms = (1.0 / (ss / dim as f64 + eps as f64).sqrt()) as f32;
                for d in 0..dim {
                    acc[d] += dy[d] * x[d] * inv_rms;
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f32; dim],
            |mut a, b| {
                for d in 0..dim {
                    a[d] += b[d];
                }
                a
            },
        );
    for d in 0..dim {
        d_weight[d] += partial[d];
    }
}

// ── RoPE backward ──────────────────────────────────────────────────────────

/// RoPE backward — 逆回転（転置回転）を適用。
pub fn rope_backward(
    d_x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    _seq_len: usize,
    theta: f32,
) {
    let stride = n_heads * head_dim;
    let half = head_dim / 2;
    d_x.par_chunks_exact_mut(stride)
        .enumerate()
        .for_each(|(t, token)| {
            for h in 0..n_heads {
                let base = h * head_dim;
                for d in 0..half {
                    let freq = 1.0 / theta.powf((2 * d) as f32 / head_dim as f32);
                    let angle = t as f32 * freq;
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    let d0 = token[base + d];
                    let d1 = token[base + d + half];
                    token[base + d] = d0.mul_add(cos_a, d1 * sin_a);
                    token[base + d + half] = (-d0).mul_add(sin_a, d1 * cos_a);
                }
            }
        });
}

// ── Attention backward ─────────────────────────────────────────────────────

/// GQA Attention backward。
///
/// `d_attn_out`: attention 出力の勾配 (seq_len × num_heads × head_dim)
/// → dQ, dK, dV を計算。
#[allow(clippy::needless_range_loop)]
pub fn gqa_attention_backward(
    d_attn_out: &[f32],
    cache: &LayerCache,
    config: &LlamaConfig,
    seq_len: usize,
    d_q: &mut [f32],
    d_k: &mut [f32],
    d_v: &mut [f32],
) {
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let kv_group_size = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let kv_h = h / kv_group_size;

        for t in 0..seq_len {
            let out_offset = t * num_heads * head_dim + h * head_dim;
            let aw_base = h * seq_len * seq_len + t * seq_len;

            // d_attn_weights: dw[s] = sum_d(d_out[d] * V[s, kv_h, d])
            let mut d_weights = vec![0.0f32; seq_len];
            for s in 0..=t {
                let v_offset = s * num_kv_heads * head_dim + kv_h * head_dim;
                let mut dw = 0.0f32;
                for d in 0..head_dim {
                    dw += d_attn_out[out_offset + d] * cache.v[v_offset + d];
                }
                d_weights[s] = dw;
            }

            // dV: d_v[s, kv_h, d] += attn_weights[h, t, s] * d_out[t, h, d]
            for s in 0..=t {
                let v_offset = s * num_kv_heads * head_dim + kv_h * head_dim;
                let w = cache.attn_weights[aw_base + s];
                for d in 0..head_dim {
                    d_v[v_offset + d] += w * d_attn_out[out_offset + d];
                }
            }

            // Softmax backward: d_score = attn * (d_w - dot(attn, d_w))
            let mut dot_aw_dw = 0.0f32;
            for s in 0..=t {
                dot_aw_dw += cache.attn_weights[aw_base + s] * d_weights[s];
            }

            // dQ, dK from d_score
            for s in 0..=t {
                let d_score = cache.attn_weights[aw_base + s] * (d_weights[s] - dot_aw_dw) * scale;
                let q_offset = t * num_heads * head_dim + h * head_dim;
                let k_offset = s * num_kv_heads * head_dim + kv_h * head_dim;
                for d in 0..head_dim {
                    d_q[q_offset + d] += d_score * cache.k[k_offset + d];
                    d_k[k_offset + d] += d_score * cache.q[q_offset + d];
                }
            }
        }
    }
}

// ── SwiGLU backward ────────────────────────────────────────────────────────

/// SiLU backward: d_silu/dx = silu(x) + sigmoid(x) * (1 - silu(x))
fn silu_grad(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    sig * (1.0 + x * (1.0 - sig))
}

/// SwiGLU FFN backward。
///
/// forward: intermediate = SiLU(gate) ⊙ up, output = intermediate × down^T
pub fn swiglu_ffn_backward(
    d_output: &[f32], // (seq_len × hidden_dim)
    cache: &LayerCache,
    weights: &LlamaLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
    d_input: &mut [f32], // (seq_len × hidden_dim) — 累積
    grads: &mut LayerWeightGrads,
) {
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let total_inter = seq_len * intermediate_dim;

    // 1. d_intermediate = d_output × down_proj  (reverse of matmul_bt with down)
    // forward: output = intermediate × down_proj^T
    // → d_intermediate = d_output × down_proj  (not transposed)
    let mut d_intermediate = vec![0.0f32; total_inter];

    // intermediate (reconstruct)
    let intermediate: Vec<f32> = cache
        .gate_silu
        .iter()
        .zip(cache.up.iter())
        .map(|(&g, &u)| g * u)
        .collect();

    // d_down_proj backward
    matmul_bt_backward(
        d_output,
        &intermediate,
        &weights.down_proj,
        &mut d_intermediate,
        &mut grads.d_down_proj,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );

    // 2. d_gate_silu = d_intermediate ⊙ up
    // 3. d_up = d_intermediate ⊙ gate_silu
    let mut d_gate_silu = vec![0.0f32; total_inter];
    let mut d_up = vec![0.0f32; total_inter];
    for i in 0..total_inter {
        d_gate_silu[i] = d_intermediate[i] * cache.up[i];
        d_up[i] = d_intermediate[i] * cache.gate_silu[i];
    }

    // 4. d_gate = d_gate_silu * silu'(gate)
    let mut d_gate = vec![0.0f32; total_inter];
    for i in 0..total_inter {
        d_gate[i] = d_gate_silu[i] * silu_grad(cache.gate[i]);
    }

    // 5. d_input from gate_proj and up_proj
    let mut d_input_gate = vec![0.0f32; seq_len * hidden_dim];
    matmul_bt_backward(
        &d_gate,
        &cache.normed_ffn,
        &weights.gate_proj,
        &mut d_input_gate,
        &mut grads.d_gate_proj,
        seq_len,
        intermediate_dim,
        hidden_dim,
    );

    let mut d_input_up = vec![0.0f32; seq_len * hidden_dim];
    matmul_bt_backward(
        &d_up,
        &cache.normed_ffn,
        &weights.up_proj,
        &mut d_input_up,
        &mut grads.d_up_proj,
        seq_len,
        intermediate_dim,
        hidden_dim,
    );

    for i in 0..seq_len * hidden_dim {
        d_input[i] += d_input_gate[i] + d_input_up[i];
    }
}

// ── レイヤー backward ──────────────────────────────────────────────────────

/// 1 Transformer レイヤーの backward パス。
///
/// 入力: `d_output` — 出力勾配 (seq_len × hidden_dim)
/// 出力: `d_input` — 入力勾配 (seq_len × hidden_dim), `LayerWeightGrads`
#[must_use]
pub fn layer_backward(
    d_output: &[f32],
    cache: &LayerCache,
    weights: &LlamaLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
) -> (Vec<f32>, LayerWeightGrads) {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let total = seq_len * hidden_dim;

    let mut grads = LayerWeightGrads::zeros(config);

    // ─── FFN backward ───
    // residual connection: output = residual_ffn + ffn_out
    // → d_residual_ffn = d_output, d_ffn_out = d_output
    let d_ffn_out = d_output;

    // FFN norm backward
    let mut d_normed_ffn = vec![0.0f32; total];
    swiglu_ffn_backward(
        d_ffn_out,
        cache,
        weights,
        config,
        seq_len,
        &mut d_normed_ffn,
        &mut grads,
    );

    // RMSNorm backward (FFN)
    let mut d_pre_ffn = vec![0.0f32; total];
    rmsnorm_backward(
        &d_normed_ffn,
        &cache.residual_ffn,
        &weights.ffn_norm,
        &mut d_pre_ffn,
        &mut grads.d_ffn_norm,
        hidden_dim,
        config.norm_eps,
    );

    // Residual: d_residual_ffn += d_pre_ffn  (← d_output already)
    let mut d_attn_residual = vec![0.0f32; total];
    for i in 0..total {
        d_attn_residual[i] = d_output[i] + d_pre_ffn[i];
    }

    // ─── Attention backward ───
    // O projection backward
    // attn_out = attn_out_raw × o_proj^T
    let mut d_attn_out_raw = vec![0.0f32; seq_len * num_heads * head_dim];
    matmul_bt_backward(
        &d_attn_residual,
        &cache.attn_out, // attn_out_raw (reshaped) — we stored o_proj output
        &weights.o_proj,
        &mut d_attn_out_raw,
        &mut grads.d_o_proj,
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );

    // GQA attention backward
    let mut d_q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut d_k = vec![0.0f32; seq_len * num_kv_heads * head_dim];
    let mut d_v = vec![0.0f32; seq_len * num_kv_heads * head_dim];

    gqa_attention_backward(
        &d_attn_out_raw,
        cache,
        config,
        seq_len,
        &mut d_q,
        &mut d_k,
        &mut d_v,
    );

    // RoPE backward
    rope_backward(&mut d_q, num_heads, head_dim, seq_len, config.rope_theta);
    rope_backward(&mut d_k, num_kv_heads, head_dim, seq_len, config.rope_theta);

    // QKV projection backward
    let mut d_normed_attn = vec![0.0f32; total];

    // Q proj backward
    matmul_bt_backward(
        &d_q,
        &cache.normed_attn,
        &weights.q_proj,
        &mut d_normed_attn,
        &mut grads.d_q_proj,
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );

    // K proj backward
    let mut d_normed_k = vec![0.0f32; total];
    matmul_bt_backward(
        &d_k,
        &cache.normed_attn,
        &weights.k_proj,
        &mut d_normed_k,
        &mut grads.d_k_proj,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );
    for i in 0..total {
        d_normed_attn[i] += d_normed_k[i];
    }

    // V proj backward
    let mut d_normed_v = vec![0.0f32; total];
    matmul_bt_backward(
        &d_v,
        &cache.normed_attn,
        &weights.v_proj,
        &mut d_normed_v,
        &mut grads.d_v_proj,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );
    for i in 0..total {
        d_normed_attn[i] += d_normed_v[i];
    }

    // Attention RMSNorm backward
    let mut d_input = vec![0.0f32; total];
    rmsnorm_backward(
        &d_normed_attn,
        &cache.residual_attn,
        &weights.attn_norm,
        &mut d_input,
        &mut grads.d_attn_norm,
        hidden_dim,
        config.norm_eps,
    );

    // Residual: d_input += d_attn_residual
    for i in 0..total {
        d_input[i] += d_attn_residual[i];
    }

    (d_input, grads)
}

/// RMSNorm backward (出力 norm 用 — weight 勾配のみ)。
pub fn rmsnorm_backward_output(
    d_output: &[f32],
    input: &[f32],
    weight: &[f32],
    d_input: &mut [f32],
    d_weight: &mut [f32],
    dim: usize,
    eps: f32,
) {
    rmsnorm_backward(d_output, input, weight, d_input, d_weight, dim, eps);
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_update() {
        let mut w = vec![1.0, 2.0, 3.0];
        let grad = vec![0.1, 0.2, 0.3];
        sgd_update(&mut w, &grad, 0.01, 0.0);
        assert!((w[0] - 0.999).abs() < 1e-6);
        assert!((w[1] - 1.998).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_with_weight_decay() {
        let mut w = vec![1.0];
        let grad = vec![0.0];
        sgd_update(&mut w, &grad, 0.1, 0.01);
        // w -= 0.1 * (0.0 + 0.01 * 1.0) = 1.0 - 0.001 = 0.999
        assert!((w[0] - 0.999).abs() < 1e-6);
    }

    #[test]
    fn test_silu_grad_at_zero() {
        // silu'(0) = sigmoid(0) = 0.5
        let g = silu_grad(0.0);
        assert!((g - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_bt_backward_shapes() {
        // C = A × B^T: A(2×3), B(4×3) → C(2×4)
        let d_c = vec![1.0f32; 8];
        let a = vec![1.0f32; 6];
        let b = vec![1.0f32; 12];
        let mut d_a = vec![0.0f32; 6];
        let mut d_b = vec![0.0f32; 12];
        matmul_bt_backward(&d_c, &a, &b, &mut d_a, &mut d_b, 2, 4, 3);
        // d_a should be non-zero
        assert!(d_a.iter().any(|&v| v.abs() > 1e-6));
        assert!(d_b.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn test_rope_backward_inverse() {
        // RoPE forward then backward should recover original gradient
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut grad = original.clone();
        // forward rotation
        crate::llama_forward::apply_rope(&mut grad, 1, 4, 1, 10000.0);
        // backward (inverse rotation)
        rope_backward(&mut grad, 1, 4, 1, 10000.0);
        for (a, b) in original.iter().zip(grad.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "RoPE backward should invert forward: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_layer_weight_grads_zeros() {
        let config = LlamaConfig {
            vocab_size: 32,
            hidden_dim: 8,
            intermediate_dim: 16,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 1,
            max_seq_len: 64,
            head_dim: 4,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            attention_bias: false,
        };
        let grads = LayerWeightGrads::zeros(&config);
        assert_eq!(grads.d_q_proj.len(), 8 * 8);
        assert_eq!(grads.d_gate_proj.len(), 16 * 8);
        assert!(grads.d_attn_norm.iter().all(|&v| v == 0.0));
    }
}
