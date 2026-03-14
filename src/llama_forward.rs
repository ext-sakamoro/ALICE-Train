//! Llama-3 フォワードパス — QAT 学習用。
//!
//! matmul, RMSNorm, RoPE, GQA Attention, SwiGLU FFN を CPU FP32 で実行。
//! backward 用に中間活性化を `LayerCache` に保存する。

use crate::llama::{LlamaConfig, LlamaLayerWeights};

/// 1レイヤーの forward 中間活性化（backward 用）。
pub struct LayerCache {
    /// RMSNorm 前の入力 (hidden_dim × seq_len)。
    pub residual_attn: Vec<f32>,
    /// attention norm 後 (hidden_dim × seq_len)。
    pub normed_attn: Vec<f32>,
    /// Q (num_heads × head_dim × seq_len)。
    pub q: Vec<f32>,
    /// K (num_kv_heads × head_dim × seq_len)。
    pub k: Vec<f32>,
    /// V (num_kv_heads × head_dim × seq_len)。
    pub v: Vec<f32>,
    /// Attention weights (num_heads × seq_len × seq_len)。
    pub attn_weights: Vec<f32>,
    /// Attention output (hidden_dim × seq_len)。
    pub attn_out: Vec<f32>,
    /// FFN norm 前の入力 (hidden_dim × seq_len)。
    pub residual_ffn: Vec<f32>,
    /// FFN norm 後 (hidden_dim × seq_len)。
    pub normed_ffn: Vec<f32>,
    /// SwiGLU gate (intermediate_dim × seq_len)。
    pub gate: Vec<f32>,
    /// SwiGLU up (intermediate_dim × seq_len)。
    pub up: Vec<f32>,
    /// SiLU(gate) (intermediate_dim × seq_len)。
    pub gate_silu: Vec<f32>,
}

// ── 行列演算 ────────────────────────────────────────────────────────────────

/// 行列-行列積: C = A × B^T (行優先)。
///
/// A: (m × k), B: (n × k) → C: (m × n)
/// B は転置して掛ける（weight が [out_features × in_features] 格納のため）。
pub fn matmul_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut sum = 0.0f32;
            for h in 0..k {
                sum = a_row[h].mul_add(b_row[h], sum);
            }
            c[i * n + j] = sum;
        }
    }
}

/// 行列-行列積: C = A × B (行優先)。
///
/// A: (m × k), B: (k × n) → C: (m × n)
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for h in 0..k {
                sum = a[i * k + h].mul_add(b[h * n + j], sum);
            }
            c[i * n + j] = sum;
        }
    }
}

// ── RMSNorm ────────────────────────────────────────────────────────────────

/// RMSNorm: x_norm = x / rms(x) * weight。
///
/// `x`: (seq_len × dim) in-place 更新。
/// `weight`: (dim,)
pub fn rmsnorm(x: &mut [f32], weight: &[f32], dim: usize, eps: f32) {
    let seq_len = x.len() / dim;
    for t in 0..seq_len {
        let row = &mut x[t * dim..(t + 1) * dim];
        let mut ss = 0.0f32;
        for &v in row.iter() {
            ss = v.mul_add(v, ss);
        }
        let rms = (ss / dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v *= inv_rms * w;
        }
    }
}

/// Bias 加算: x (seq_len × dim) の各行に bias (dim,) を加算。
pub fn add_bias(x: &mut [f32], bias: &[f32], seq_len: usize, dim: usize) {
    for t in 0..seq_len {
        let row = &mut x[t * dim..(t + 1) * dim];
        for (v, &b) in row.iter_mut().zip(bias.iter()) {
            *v += b;
        }
    }
}

// ── RoPE ────────────────────────────────────────────────────────────────────

/// Rotary Position Embeddings を適用する。
///
/// `x`: (seq_len × n_heads × head_dim) — Q or K テンソル。
/// `head_dim` の偶数ペアに cos/sin 回転を適用する。
pub fn apply_rope(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    seq_len: usize,
    theta: f32,
) {
    for t in 0..seq_len {
        for h in 0..n_heads {
            let base = t * n_heads * head_dim + h * head_dim;
            for d in (0..head_dim).step_by(2) {
                let freq = 1.0 / theta.powf(d as f32 / head_dim as f32);
                let angle = t as f32 * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = x[base + d];
                let x1 = x[base + d + 1];
                x[base + d] = x0.mul_add(cos_a, -(x1 * sin_a));
                x[base + d + 1] = x0.mul_add(sin_a, x1 * cos_a);
            }
        }
    }
}

// ── GQA Attention ──────────────────────────────────────────────────────────

/// Grouped Query Attention — causal mask 付き。
///
/// Q: (seq_len × num_heads × head_dim)
/// K, V: (seq_len × num_kv_heads × head_dim)
/// 出力: (seq_len × hidden_dim)
///
/// GQA: 各 Q ヘッドは `num_heads / num_kv_heads` ごとに KV ヘッドを共有。
pub fn gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    attn_weights_out: &mut [f32],
    config: &LlamaConfig,
    seq_len: usize,
) {
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let kv_group_size = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let kv_h = h / kv_group_size; // 対応する KV ヘッド

        for t in 0..seq_len {
            // Q[t, h] · K[s, kv_h] for s <= t (causal)
            let q_offset = t * num_heads * head_dim + h * head_dim;

            // Score 計算
            let aw_base = h * seq_len * seq_len + t * seq_len;
            let mut max_score = f32::NEG_INFINITY;

            for s in 0..=t {
                let k_offset = s * num_kv_heads * head_dim + kv_h * head_dim;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score = q[q_offset + d].mul_add(k[k_offset + d], score);
                }
                score *= scale;
                attn_weights_out[aw_base + s] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            // Causal mask: s > t → -inf
            for s in (t + 1)..seq_len {
                attn_weights_out[aw_base + s] = f32::NEG_INFINITY;
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for s in 0..=t {
                let e = (attn_weights_out[aw_base + s] - max_score).exp();
                attn_weights_out[aw_base + s] = e;
                sum_exp += e;
            }
            let inv_sum = 1.0 / sum_exp.max(1e-10);
            for s in 0..=t {
                attn_weights_out[aw_base + s] *= inv_sum;
            }
            for s in (t + 1)..seq_len {
                attn_weights_out[aw_base + s] = 0.0;
            }

            // Weighted sum of V
            let out_offset = t * num_heads * head_dim + h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for s in 0..=t {
                    let v_offset = s * num_kv_heads * head_dim + kv_h * head_dim;
                    val = attn_weights_out[aw_base + s].mul_add(v[v_offset + d], val);
                }
                output[out_offset + d] = val;
            }
        }
    }
}

// ── SwiGLU FFN ─────────────────────────────────────────────────────────────

/// SiLU (Swish) 活性化: x * sigmoid(x)。
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU FFN forward。
///
/// gate = input × gate_proj^T
/// up   = input × up_proj^T
/// ffn_out = (SiLU(gate) ⊙ up) × down_proj^T
pub fn swiglu_ffn(
    input: &[f32],
    gate_proj: &[f32],
    up_proj: &[f32],
    down_proj: &[f32],
    output: &mut [f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    gate_silu_buf: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    // gate = input × gate_proj^T  (seq_len × intermediate_dim)
    matmul_bt(input, gate_proj, gate_buf, seq_len, intermediate_dim, hidden_dim);

    // up = input × up_proj^T  (seq_len × intermediate_dim)
    matmul_bt(input, up_proj, up_buf, seq_len, intermediate_dim, hidden_dim);

    // SiLU(gate) ⊙ up → intermediate
    let total = seq_len * intermediate_dim;
    for i in 0..total {
        gate_silu_buf[i] = silu(gate_buf[i]);
    }
    let mut intermediate = vec![0.0f32; total];
    for i in 0..total {
        intermediate[i] = gate_silu_buf[i] * up_buf[i];
    }

    // output = intermediate × down_proj^T  (seq_len × hidden_dim)
    matmul_bt(&intermediate, down_proj, output, seq_len, hidden_dim, intermediate_dim);
}

// ── レイヤー Forward ───────────────────────────────────────────────────────

/// 1 Transformer レイヤーの forward パス。
///
/// 入力: (seq_len × hidden_dim)
/// 出力: (seq_len × hidden_dim) + `LayerCache`
pub fn layer_forward(
    input: &mut Vec<f32>,
    weights: &LlamaLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
) -> LayerCache {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_dim = config.intermediate_dim;

    // 1. 残差保存 + Attention RMSNorm
    let residual_attn = input.clone();
    let mut normed = input.clone();
    rmsnorm(&mut normed, &weights.attn_norm, hidden_dim, config.norm_eps);
    let normed_attn = normed.clone();

    // 2. QKV projection
    let mut q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k = vec![0.0f32; seq_len * num_kv_heads * head_dim];
    let mut v = vec![0.0f32; seq_len * num_kv_heads * head_dim];

    matmul_bt(&normed, &weights.q_proj, &mut q, seq_len, num_heads * head_dim, hidden_dim);
    matmul_bt(&normed, &weights.k_proj, &mut k, seq_len, num_kv_heads * head_dim, hidden_dim);
    matmul_bt(&normed, &weights.v_proj, &mut v, seq_len, num_kv_heads * head_dim, hidden_dim);

    // 2b. Attention bias (Qwen2.5 等)
    if let Some(ref b) = weights.q_bias {
        add_bias(&mut q, b, seq_len, num_heads * head_dim);
    }
    if let Some(ref b) = weights.k_bias {
        add_bias(&mut k, b, seq_len, num_kv_heads * head_dim);
    }
    if let Some(ref b) = weights.v_bias {
        add_bias(&mut v, b, seq_len, num_kv_heads * head_dim);
    }

    // 3. RoPE
    apply_rope(&mut q, num_heads, head_dim, seq_len, config.rope_theta);
    apply_rope(&mut k, num_kv_heads, head_dim, seq_len, config.rope_theta);

    // 4. GQA Attention
    let mut attn_out_raw = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut attn_weights = vec![0.0f32; num_heads * seq_len * seq_len];
    gqa_attention(&q, &k, &v, &mut attn_out_raw, &mut attn_weights, config, seq_len);

    // 5. O projection
    let mut attn_out = vec![0.0f32; seq_len * hidden_dim];
    matmul_bt(&attn_out_raw, &weights.o_proj, &mut attn_out, seq_len, hidden_dim, num_heads * head_dim);

    // 6. Residual add
    for i in 0..input.len() {
        input[i] = residual_attn[i] + attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm
    let residual_ffn = input.clone();
    let mut normed_ffn_buf = input.clone();
    rmsnorm(&mut normed_ffn_buf, &weights.ffn_norm, hidden_dim, config.norm_eps);
    let normed_ffn = normed_ffn_buf.clone();

    // 8. SwiGLU FFN
    let mut ffn_out = vec![0.0f32; seq_len * hidden_dim];
    let mut gate_buf = vec![0.0f32; seq_len * intermediate_dim];
    let mut up_buf = vec![0.0f32; seq_len * intermediate_dim];
    let mut gate_silu_buf = vec![0.0f32; seq_len * intermediate_dim];

    swiglu_ffn(
        &normed_ffn_buf,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
        &mut ffn_out,
        &mut gate_buf,
        &mut up_buf,
        &mut gate_silu_buf,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );

    // 9. Residual add
    for i in 0..input.len() {
        input[i] = residual_ffn[i] + ffn_out[i];
    }

    LayerCache {
        residual_attn,
        normed_attn,
        q,
        k,
        v,
        attn_weights,
        attn_out,
        residual_ffn,
        normed_ffn,
        gate: gate_buf,
        up: up_buf,
        gate_silu: gate_silu_buf,
    }
}

/// 全レイヤーの forward パス。
///
/// 入力: token IDs (seq_len)
/// 出力: logits (seq_len × vocab_size), Vec<LayerCache>
pub fn model_forward(
    token_ids: &[u32],
    embedding_table: &[f32],
    layers: &[LlamaLayerWeights],
    output_norm: &[f32],
    output_proj: &[f32],
    config: &LlamaConfig,
) -> (Vec<f32>, Vec<LayerCache>) {
    let seq_len = token_ids.len();
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;

    // Embedding lookup
    let mut hidden = vec![0.0f32; seq_len * hidden_dim];
    for (t, &tok) in token_ids.iter().enumerate() {
        let tok = tok as usize;
        if tok < vocab_size {
            let src = &embedding_table[tok * hidden_dim..(tok + 1) * hidden_dim];
            hidden[t * hidden_dim..(t + 1) * hidden_dim].copy_from_slice(src);
        }
    }

    // Transformer layers
    let mut caches = Vec::with_capacity(layers.len());
    for layer_weights in layers {
        let cache = layer_forward(&mut hidden, layer_weights, config, seq_len);
        caches.push(cache);
    }

    // Output RMSNorm
    rmsnorm(&mut hidden, output_norm, hidden_dim, config.norm_eps);

    // Output projection → logits
    let mut logits = vec![0.0f32; seq_len * vocab_size];
    matmul_bt(&hidden, output_proj, &mut logits, seq_len, vocab_size, hidden_dim);

    (logits, caches)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_identity() {
        let weight = vec![1.0; 4];
        let mut x = vec![1.0, 0.0, 1.0, 0.0];
        rmsnorm(&mut x, &weight, 4, 1e-5);
        // rms = sqrt((1+0+1+0)/4) = sqrt(0.5) ≈ 0.7071
        // x[0] = 1.0 / 0.7071 ≈ 1.4142
        assert!((x[0] - 1.4142).abs() < 0.01);
        assert!(x[1].abs() < 1e-6);
    }

    #[test]
    fn test_matmul_bt_simple() {
        // A = [[1, 2], [3, 4]], B = [[1, 0], [0, 1]]
        // C = A × B^T = A × B = [[1, 2], [3, 4]]
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [1.0, 0.0, 0.0, 1.0f32];
        let mut c = [0.0f32; 4];
        matmul_bt(&a, &b, &mut c, 2, 2, 2);
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
        assert!((c[2] - 3.0).abs() < 1e-6);
        assert!((c[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu() {
        assert!(silu(0.0).abs() < 1e-6);
        // silu(1) = 1 / (1 + e^-1) ≈ 0.7311
        assert!((silu(1.0) - 0.7311).abs() < 0.01);
    }

    #[test]
    fn test_rope_preserves_norm() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0];
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        apply_rope(&mut x, 1, 4, 1, 10000.0);
        let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-5, "RoPE should preserve norm");
    }
}
