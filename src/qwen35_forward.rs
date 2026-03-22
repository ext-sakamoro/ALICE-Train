//! Qwen3.5 フォワードパス — カリカリ最適化ハイブリッド DeltaNet + Full Attention。
//!
//! # 最適化
//!
//! - L1: Rayon ヘッド並列 (deltanet.rs)
//! - L2: ゼロアロケーション再帰 (deltanet.rs)
//! - L3: transpose排除 — conv1d を行優先で直接実行
//! - L4: GQA遅延展開 — L2 norm + repeat を融合
//! - L5: gate融合 — sigmoid/softplus/exp をワンパスで計算
//! - L6: conv1d Rayon並列 (deltanet.rs)

use crate::deltanet::{
    apply_partial_rope, causal_conv1d_silu_row_major, compute_gates_fused,
    deltanet_recurrence_forward, deltanet_recurrence_forward_eval, gated_rmsnorm,
    l2norm_and_gqa_expand, qk_norm, DeltaNetLayerCache,
};
use crate::blas::{blas_matmul_bt, blas_rmsnorm, blas_swiglu_ffn, blas_swiglu_ffn_training};
use crate::llama_forward::gqa_attention;
use crate::qwen35::{
    DeltaNetLayerWeights, FullAttnLayerWeights, Qwen35Config, Qwen35LayerWeights,
};

/// Full Attention レイヤーの forward cache。
pub struct FullAttnLayerCache {
    /// 残差 (attention 前)。
    pub residual_attn: Vec<f32>,
    /// Attention norm 後。
    pub normed_attn: Vec<f32>,
    /// Q (QK-norm + partial RoPE 後)。
    pub q: Vec<f32>,
    /// K (QK-norm + partial RoPE 後)。
    pub k: Vec<f32>,
    /// V。
    pub v: Vec<f32>,
    /// Attention weights。
    pub attn_weights: Vec<f32>,
    /// Attention output (O projection 後)。
    pub attn_out: Vec<f32>,
    /// 残差 (FFN 前)。
    pub residual_ffn: Vec<f32>,
    /// FFN norm 後。
    pub normed_ffn: Vec<f32>,
    /// SwiGLU gate。
    pub gate: Vec<f32>,
    /// SwiGLU up。
    pub up: Vec<f32>,
    /// SiLU(gate)。
    pub gate_silu: Vec<f32>,
}

/// ハイブリッドレイヤーの cache。
pub enum Qwen35LayerCache {
    /// DeltaNet 層の cache。
    DeltaNet(DeltaNetLayerCache),
    /// Full Attention 層の cache。
    FullAttention(FullAttnLayerCache),
}

// ── DeltaNet Layer Forward (L3: transpose排除, L4: 融合展開, L5: gate融合) ──

/// DeltaNet レイヤーの forward — カリカリ最適化版。
fn deltanet_layer_forward(
    input: &mut Vec<f32>,
    weights: &DeltaNetLayerWeights,
    config: &Qwen35Config,
    seq_len: usize,
) -> DeltaNetLayerCache {
    let hidden = config.hidden_size;
    let key_dim = config.linear_key_dim();
    let val_dim = config.linear_value_dim();
    let n_k_heads = config.linear_num_key_heads;
    let n_v_heads = config.linear_num_value_heads;
    let dk = config.linear_key_head_dim;
    let dv = config.linear_value_head_dim;
    let kernel_size = config.linear_conv_kernel_dim;
    let qkv_dim = key_dim * 2 + val_dim;

    // 1. 残差保存 + Input RMSNorm
    let residual_attn = input.clone();
    let mut normed = input.clone();
    blas_rmsnorm(&mut normed, &weights.input_layernorm, hidden, config.rms_norm_eps);

    // 2. Projections — 4つの matmul
    let mut qkv = vec![0.0f32; seq_len * qkv_dim];
    blas_matmul_bt(&normed, &weights.in_proj_qkv, &mut qkv, seq_len, qkv_dim, hidden);

    let mut z = vec![0.0f32; seq_len * val_dim];
    blas_matmul_bt(&normed, &weights.in_proj_z, &mut z, seq_len, val_dim, hidden);

    let mut b_raw = vec![0.0f32; seq_len * n_v_heads];
    blas_matmul_bt(&normed, &weights.in_proj_b, &mut b_raw, seq_len, n_v_heads, hidden);

    let mut a_raw = vec![0.0f32; seq_len * n_v_heads];
    blas_matmul_bt(&normed, &weights.in_proj_a, &mut a_raw, seq_len, n_v_heads, hidden);

    // 3. L3: Causal Conv1d — 行優先のまま処理 (transpose排除)
    let pre_conv_qkv = qkv.clone();
    let mut qkv_conv = vec![0.0f32; seq_len * qkv_dim];
    causal_conv1d_silu_row_major(
        &qkv, &weights.conv1d_weight, &mut qkv_conv, qkv_dim, seq_len, kernel_size,
    );

    // 4. Split Q, K, V — 行優先から直接スライス
    let mut q_raw = vec![0.0f32; seq_len * n_k_heads * dk];
    let mut k_raw = vec![0.0f32; seq_len * n_k_heads * dk];
    let mut v_all = vec![0.0f32; seq_len * n_v_heads * dv];
    for t in 0..seq_len {
        let row = &qkv_conv[t * qkv_dim..];
        q_raw[t * key_dim..(t + 1) * key_dim].copy_from_slice(&row[..key_dim]);
        k_raw[t * key_dim..(t + 1) * key_dim].copy_from_slice(&row[key_dim..key_dim * 2]);
        v_all[t * val_dim..(t + 1) * val_dim].copy_from_slice(&row[key_dim * 2..qkv_dim]);
    }

    // 5. L4: L2 norm + GQA expansion 融合
    let mut q_expanded = vec![0.0f32; seq_len * n_v_heads * dk];
    let mut k_expanded = vec![0.0f32; seq_len * n_v_heads * dk];
    l2norm_and_gqa_expand(
        &q_raw, &k_raw, &mut q_expanded, &mut k_expanded,
        seq_len, n_k_heads, n_v_heads, dk, 1e-6,
    );

    // 6. L5: Gate融合 — sigmoid + softplus + exp をワンパス
    let mut beta = vec![0.0f32; seq_len * n_v_heads];
    let mut g = vec![0.0f32; seq_len * n_v_heads];
    compute_gates_fused(
        &b_raw, &a_raw, &weights.a_log, &weights.dt_bias,
        &mut beta, &mut g, seq_len, n_v_heads,
    );

    // 7. L1+L2: Gated DeltaNet recurrence — ヘッド並列 + ゼロアロケーション
    let mut attn_out_raw = vec![0.0f32; seq_len * n_v_heads * dv];
    let (step_caches, final_states) = deltanet_recurrence_forward(
        &q_expanded, &k_expanded, &v_all, &beta, &g, &b_raw, &a_raw,
        &mut attn_out_raw, n_v_heads, dk, dv, seq_len,
    );

    // 8. Gated RMSNorm — Rayon並列
    let attn_out_pre_norm = attn_out_raw.clone();
    let mut attn_normed = vec![0.0f32; seq_len * val_dim];
    gated_rmsnorm(&attn_out_raw, &z, &weights.norm_weight, &mut attn_normed, dv, config.rms_norm_eps);

    // 9. Output projection
    let mut attn_out = vec![0.0f32; seq_len * hidden];
    blas_matmul_bt(&attn_normed, &weights.out_proj, &mut attn_out, seq_len, hidden, val_dim);

    // 10. Residual add
    for i in 0..input.len() {
        input[i] = residual_attn[i] + attn_out[i];
    }

    // 11. FFN
    let residual_ffn = input.clone();
    let mut normed_ffn = input.clone();
    blas_rmsnorm(&mut normed_ffn, &weights.post_attn_layernorm, hidden, config.rms_norm_eps);
    let normed_ffn_saved = normed_ffn.clone();

    let inter = config.intermediate_size;
    let mut ffn_out = vec![0.0f32; seq_len * hidden];
    let mut gate_buf = vec![0.0f32; seq_len * inter];
    let mut up_buf = vec![0.0f32; seq_len * inter];
    let mut gate_silu_buf = vec![0.0f32; seq_len * inter];

    blas_swiglu_ffn_training(
        &normed_ffn, &weights.gate_proj, &weights.up_proj, &weights.down_proj,
        &mut ffn_out, &mut gate_buf, &mut up_buf, &mut gate_silu_buf,
        seq_len, hidden, inter,
    );

    // 12. Residual add
    for i in 0..input.len() {
        input[i] = residual_ffn[i] + ffn_out[i];
    }

    DeltaNetLayerCache {
        step_caches,
        final_states,
        pre_conv_qkv,
        z,
        attn_out_pre_norm,
        residual_attn,
        residual_ffn,
        normed_ffn: normed_ffn_saved,
        gate: gate_buf,
        up: up_buf,
        gate_silu: gate_silu_buf,
    }
}

// ── Full Attention Layer Forward ────────────────────────────────────────────

/// Full Attention レイヤーの forward (QK-norm + partial RoPE)。
fn full_attn_layer_forward(
    input: &mut Vec<f32>,
    weights: &FullAttnLayerWeights,
    config: &Qwen35Config,
    seq_len: usize,
) -> FullAttnLayerCache {
    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let rotary_dim = config.rotary_dim();
    let inter = config.intermediate_size;

    let residual_attn = input.clone();
    let mut normed = input.clone();
    blas_rmsnorm(&mut normed, &weights.input_layernorm, hidden, config.rms_norm_eps);
    let normed_attn = normed.clone();

    let mut q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k = vec![0.0f32; seq_len * num_kv_heads * head_dim];
    let mut v = vec![0.0f32; seq_len * num_kv_heads * head_dim];

    blas_matmul_bt(&normed, &weights.q_proj, &mut q, seq_len, num_heads * head_dim, hidden);
    blas_matmul_bt(&normed, &weights.k_proj, &mut k, seq_len, num_kv_heads * head_dim, hidden);
    blas_matmul_bt(&normed, &weights.v_proj, &mut v, seq_len, num_kv_heads * head_dim, hidden);

    qk_norm(&mut q, &weights.q_norm, num_heads, head_dim, config.rms_norm_eps);
    qk_norm(&mut k, &weights.k_norm, num_kv_heads, head_dim, config.rms_norm_eps);

    apply_partial_rope(&mut q, num_heads, head_dim, seq_len, rotary_dim, config.rope_theta);
    apply_partial_rope(&mut k, num_kv_heads, head_dim, seq_len, rotary_dim, config.rope_theta);

    let llama_compat = crate::llama::LlamaConfig {
        vocab_size: config.vocab_size,
        hidden_dim: hidden,
        intermediate_dim: inter,
        num_heads,
        num_kv_heads,
        num_layers: config.num_hidden_layers,
        max_seq_len: 262_144,
        head_dim,
        rope_theta: config.rope_theta,
        norm_eps: config.rms_norm_eps,
        attention_bias: false,
    };

    let mut attn_out_raw = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut attn_weights = vec![0.0f32; num_heads * seq_len * seq_len];
    gqa_attention(&q, &k, &v, &mut attn_out_raw, &mut attn_weights, &llama_compat, seq_len);

    let mut attn_out = vec![0.0f32; seq_len * hidden];
    blas_matmul_bt(&attn_out_raw, &weights.o_proj, &mut attn_out, seq_len, hidden, num_heads * head_dim);

    for i in 0..input.len() {
        input[i] = residual_attn[i] + attn_out[i];
    }

    let residual_ffn = input.clone();
    let mut normed_ffn = input.clone();
    blas_rmsnorm(&mut normed_ffn, &weights.post_attn_layernorm, hidden, config.rms_norm_eps);
    let normed_ffn_saved = normed_ffn.clone();

    let mut ffn_out = vec![0.0f32; seq_len * hidden];
    let mut gate_buf = vec![0.0f32; seq_len * inter];
    let mut up_buf = vec![0.0f32; seq_len * inter];
    let mut gate_silu_buf = vec![0.0f32; seq_len * inter];

    blas_swiglu_ffn_training(
        &normed_ffn, &weights.gate_proj, &weights.up_proj, &weights.down_proj,
        &mut ffn_out, &mut gate_buf, &mut up_buf, &mut gate_silu_buf,
        seq_len, hidden, inter,
    );

    for i in 0..input.len() {
        input[i] = residual_ffn[i] + ffn_out[i];
    }

    FullAttnLayerCache {
        residual_attn,
        normed_attn,
        q, k, v,
        attn_weights,
        attn_out,
        residual_ffn,
        normed_ffn: normed_ffn_saved,
        gate: gate_buf,
        up: up_buf,
        gate_silu: gate_silu_buf,
    }
}

// ── Model Forward ───────────────────────────────────────────────────────────

/// 1レイヤーの forward (ハイブリッドルーティング)。
pub fn qwen35_layer_forward(
    input: &mut Vec<f32>,
    weights: &Qwen35LayerWeights,
    config: &Qwen35Config,
    seq_len: usize,
) -> Qwen35LayerCache {
    match weights {
        Qwen35LayerWeights::DeltaNet(w) => {
            Qwen35LayerCache::DeltaNet(deltanet_layer_forward(input, w, config, seq_len))
        }
        Qwen35LayerWeights::FullAttention(w) => {
            Qwen35LayerCache::FullAttention(full_attn_layer_forward(input, w, config, seq_len))
        }
    }
}

/// 全レイヤーの forward パス。
#[must_use]
pub fn qwen35_model_forward(
    token_ids: &[u32],
    embedding_table: &[f32],
    layers: &[Qwen35LayerWeights],
    output_norm: &[f32],
    lm_head: &[f32],
    config: &Qwen35Config,
) -> (Vec<f32>, Vec<Qwen35LayerCache>) {
    let seq_len = token_ids.len();
    let hidden = config.hidden_size;
    let vocab_size = config.vocab_size;

    let mut hidden_states = vec![0.0f32; seq_len * hidden];
    for (t, &tok) in token_ids.iter().enumerate() {
        let tok = (tok as usize) % vocab_size;
        hidden_states[t * hidden..(t + 1) * hidden]
            .copy_from_slice(&embedding_table[tok * hidden..(tok + 1) * hidden]);
    }

    let mut caches = Vec::with_capacity(layers.len());
    for layer_weights in layers {
        let cache = qwen35_layer_forward(&mut hidden_states, layer_weights, config, seq_len);
        caches.push(cache);
    }

    blas_rmsnorm(&mut hidden_states, output_norm, hidden, config.rms_norm_eps);

    let mut logits = vec![0.0f32; seq_len * vocab_size];
    blas_matmul_bt(&hidden_states, lm_head, &mut logits, seq_len, vocab_size, hidden);

    (logits, caches)
}

// ── L8: Eval 専用 Model Forward ─────────────────────────────────────────────

/// DeltaNet レイヤーの eval forward — cache 不要。
fn deltanet_layer_forward_eval(
    input: &mut Vec<f32>,
    weights: &DeltaNetLayerWeights,
    config: &Qwen35Config,
    seq_len: usize,
) {
    let hidden = config.hidden_size;
    let key_dim = config.linear_key_dim();
    let val_dim = config.linear_value_dim();
    let n_k_heads = config.linear_num_key_heads;
    let n_v_heads = config.linear_num_value_heads;
    let dk = config.linear_key_head_dim;
    let dv = config.linear_value_head_dim;
    let kernel_size = config.linear_conv_kernel_dim;
    let qkv_dim = key_dim * 2 + val_dim;

    let residual_attn = input.clone();
    let mut normed = input.clone();
    blas_rmsnorm(&mut normed, &weights.input_layernorm, hidden, config.rms_norm_eps);

    let mut qkv = vec![0.0f32; seq_len * qkv_dim];
    blas_matmul_bt(&normed, &weights.in_proj_qkv, &mut qkv, seq_len, qkv_dim, hidden);

    let mut z = vec![0.0f32; seq_len * val_dim];
    blas_matmul_bt(&normed, &weights.in_proj_z, &mut z, seq_len, val_dim, hidden);

    let mut b_raw = vec![0.0f32; seq_len * n_v_heads];
    blas_matmul_bt(&normed, &weights.in_proj_b, &mut b_raw, seq_len, n_v_heads, hidden);

    let mut a_raw = vec![0.0f32; seq_len * n_v_heads];
    blas_matmul_bt(&normed, &weights.in_proj_a, &mut a_raw, seq_len, n_v_heads, hidden);

    let mut qkv_conv = vec![0.0f32; seq_len * qkv_dim];
    causal_conv1d_silu_row_major(&qkv, &weights.conv1d_weight, &mut qkv_conv, qkv_dim, seq_len, kernel_size);

    let mut q_raw = vec![0.0f32; seq_len * key_dim];
    let mut k_raw = vec![0.0f32; seq_len * key_dim];
    let mut v_all = vec![0.0f32; seq_len * val_dim];
    for t in 0..seq_len {
        let row = &qkv_conv[t * qkv_dim..];
        q_raw[t * key_dim..(t + 1) * key_dim].copy_from_slice(&row[..key_dim]);
        k_raw[t * key_dim..(t + 1) * key_dim].copy_from_slice(&row[key_dim..key_dim * 2]);
        v_all[t * val_dim..(t + 1) * val_dim].copy_from_slice(&row[key_dim * 2..qkv_dim]);
    }

    let mut q_expanded = vec![0.0f32; seq_len * n_v_heads * dk];
    let mut k_expanded = vec![0.0f32; seq_len * n_v_heads * dk];
    l2norm_and_gqa_expand(&q_raw, &k_raw, &mut q_expanded, &mut k_expanded, seq_len, n_k_heads, n_v_heads, dk, 1e-6);

    let mut beta = vec![0.0f32; seq_len * n_v_heads];
    let mut g = vec![0.0f32; seq_len * n_v_heads];
    compute_gates_fused(&b_raw, &a_raw, &weights.a_log, &weights.dt_bias, &mut beta, &mut g, seq_len, n_v_heads);

    let mut attn_out_raw = vec![0.0f32; seq_len * n_v_heads * dv];
    deltanet_recurrence_forward_eval(
        &q_expanded, &k_expanded, &v_all, &beta, &g,
        &mut attn_out_raw, n_v_heads, dk, dv, seq_len,
    );

    let mut attn_normed = vec![0.0f32; seq_len * val_dim];
    gated_rmsnorm(&attn_out_raw, &z, &weights.norm_weight, &mut attn_normed, dv, config.rms_norm_eps);

    let mut attn_out = vec![0.0f32; seq_len * hidden];
    blas_matmul_bt(&attn_normed, &weights.out_proj, &mut attn_out, seq_len, hidden, val_dim);

    for i in 0..input.len() {
        input[i] = residual_attn[i] + attn_out[i];
    }

    // FFN
    let residual_ffn = input.clone();
    let mut normed_ffn = input.clone();
    blas_rmsnorm(&mut normed_ffn, &weights.post_attn_layernorm, hidden, config.rms_norm_eps);

    let inter = config.intermediate_size;
    let mut ffn_out = vec![0.0f32; seq_len * hidden];
    let mut gate_buf = vec![0.0f32; seq_len * inter];
    let mut up_buf = vec![0.0f32; seq_len * inter];
    let mut gate_silu_buf = vec![0.0f32; seq_len * inter];
    blas_swiglu_ffn(&normed_ffn, &weights.gate_proj, &weights.up_proj, &weights.down_proj,
        &mut ffn_out, &mut gate_buf, &mut up_buf, &mut gate_silu_buf, seq_len, hidden, inter);

    for i in 0..input.len() {
        input[i] = residual_ffn[i] + ffn_out[i];
    }
}

/// Full Attention レイヤーの eval forward — cache 不要。
fn full_attn_layer_forward_eval(
    input: &mut Vec<f32>,
    weights: &FullAttnLayerWeights,
    config: &Qwen35Config,
    seq_len: usize,
) {
    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let rotary_dim = config.rotary_dim();
    let inter = config.intermediate_size;

    let residual_attn = input.clone();
    let mut normed = input.clone();
    blas_rmsnorm(&mut normed, &weights.input_layernorm, hidden, config.rms_norm_eps);

    let mut q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut k = vec![0.0f32; seq_len * num_kv_heads * head_dim];
    let mut v = vec![0.0f32; seq_len * num_kv_heads * head_dim];
    blas_matmul_bt(&normed, &weights.q_proj, &mut q, seq_len, num_heads * head_dim, hidden);
    blas_matmul_bt(&normed, &weights.k_proj, &mut k, seq_len, num_kv_heads * head_dim, hidden);
    blas_matmul_bt(&normed, &weights.v_proj, &mut v, seq_len, num_kv_heads * head_dim, hidden);

    qk_norm(&mut q, &weights.q_norm, num_heads, head_dim, config.rms_norm_eps);
    qk_norm(&mut k, &weights.k_norm, num_kv_heads, head_dim, config.rms_norm_eps);
    apply_partial_rope(&mut q, num_heads, head_dim, seq_len, rotary_dim, config.rope_theta);
    apply_partial_rope(&mut k, num_kv_heads, head_dim, seq_len, rotary_dim, config.rope_theta);

    let llama_compat = crate::llama::LlamaConfig {
        vocab_size: config.vocab_size, hidden_dim: hidden, intermediate_dim: inter,
        num_heads, num_kv_heads, num_layers: config.num_hidden_layers,
        max_seq_len: 262_144, head_dim, rope_theta: config.rope_theta,
        norm_eps: config.rms_norm_eps, attention_bias: false,
    };

    let mut attn_out_raw = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut attn_weights = vec![0.0f32; num_heads * seq_len * seq_len];
    gqa_attention(&q, &k, &v, &mut attn_out_raw, &mut attn_weights, &llama_compat, seq_len);

    let mut attn_out = vec![0.0f32; seq_len * hidden];
    blas_matmul_bt(&attn_out_raw, &weights.o_proj, &mut attn_out, seq_len, hidden, num_heads * head_dim);

    for i in 0..input.len() { input[i] = residual_attn[i] + attn_out[i]; }

    let residual_ffn = input.clone();
    let mut normed_ffn = input.clone();
    blas_rmsnorm(&mut normed_ffn, &weights.post_attn_layernorm, hidden, config.rms_norm_eps);

    let mut ffn_out = vec![0.0f32; seq_len * hidden];
    let mut gate_buf = vec![0.0f32; seq_len * inter];
    let mut up_buf = vec![0.0f32; seq_len * inter];
    let mut gate_silu_buf = vec![0.0f32; seq_len * inter];
    blas_swiglu_ffn(&normed_ffn, &weights.gate_proj, &weights.up_proj, &weights.down_proj,
        &mut ffn_out, &mut gate_buf, &mut up_buf, &mut gate_silu_buf, seq_len, hidden, inter);

    for i in 0..input.len() { input[i] = residual_ffn[i] + ffn_out[i]; }
}

/// L8: Eval 専用 model forward — cache 不保存、メモリ最小。
///
/// 学習時の `qwen35_model_forward` と同じ logits を返すが、
/// backward 用の cache を一切保存しない。eval/推論用。
#[must_use]
pub fn qwen35_model_forward_eval(
    token_ids: &[u32],
    embedding_table: &[f32],
    layers: &[Qwen35LayerWeights],
    output_norm: &[f32],
    lm_head: &[f32],
    config: &Qwen35Config,
) -> Vec<f32> {
    let seq_len = token_ids.len();
    let hidden = config.hidden_size;
    let vocab_size = config.vocab_size;

    let mut hidden_states = vec![0.0f32; seq_len * hidden];
    for (t, &tok) in token_ids.iter().enumerate() {
        let tok = tok as usize;
        if tok < vocab_size {
            hidden_states[t * hidden..(t + 1) * hidden]
                .copy_from_slice(&embedding_table[tok * hidden..tok * hidden + hidden]);
        }
    }

    for layer_weights in layers {
        match layer_weights {
            Qwen35LayerWeights::DeltaNet(w) => {
                deltanet_layer_forward_eval(&mut hidden_states, w, config, seq_len);
            }
            Qwen35LayerWeights::FullAttention(w) => {
                full_attn_layer_forward_eval(&mut hidden_states, w, config, seq_len);
            }
        }
    }

    blas_rmsnorm(&mut hidden_states, output_norm, hidden, config.rms_norm_eps);

    let mut logits = vec![0.0f32; seq_len * vocab_size];
    blas_matmul_bt(&hidden_states, lm_head, &mut logits, seq_len, vocab_size, hidden);

    logits
}

// ── L10: ストリーミング Eval Forward ─────────────────────────────────────────

/// L10+L12: レイヤーストリーミング eval — FP32キャッシュ対応。
///
/// 初回: safetensors (BF16) → FP32変換 → キャッシュ保存。
/// 2回目以降: FP32キャッシュから直接読み込み (BF16デコード不要)。
/// `cache_dir`: FP32キャッシュ保存先。None の場合は毎回safetensorsから読む。
#[must_use]
pub fn qwen35_model_forward_eval_streaming(
    token_ids: &[u32],
    embedding_table: &[f32],
    get_tensor: &dyn Fn(&str) -> Option<Vec<f32>>,
    weight_prefix: &str,
    output_norm: &[f32],
    lm_head: &[f32],
    config: &Qwen35Config,
    cache_dir: Option<&str>,
) -> Vec<f32> {
    let seq_len = token_ids.len();
    let hidden = config.hidden_size;
    let vocab_size = config.vocab_size;

    let mut hidden_states = vec![0.0f32; seq_len * hidden];
    for (t, &tok) in token_ids.iter().enumerate() {
        let tok = tok as usize;
        if tok < vocab_size {
            hidden_states[t * hidden..(t + 1) * hidden]
                .copy_from_slice(&embedding_table[tok * hidden..tok * hidden + hidden]);
        }
    }

    // L12: FP32 キャッシュ有無で分岐
    let use_cache = cache_dir
        .map(|d| crate::fp32_cache::cache_exists(d, config))
        .unwrap_or(false);

    for i in 0..config.num_hidden_layers {
        let layer = if use_cache {
            // FP32 キャッシュから読み込み (BF16デコード不要)
            crate::fp32_cache::load_layer_from_cache(cache_dir.unwrap(), i, config)
                .unwrap_or_else(|e| {
                    eprintln!("[fp32_cache] layer {i} 読み込み失敗: {e}");
                    std::process::exit(1);
                })
        } else {
            // safetensors から読み込み (BF16→FP32変換)
            let layer_prefix = format!("{weight_prefix}.layers.{i}");
            let lt = config.layer_type(i);
            match lt {
                crate::qwen35::LayerType::LinearAttention => {
                    let w = DeltaNetLayerWeights::from_tensors(&layer_prefix, get_tensor)
                        .unwrap_or_else(|| {
                            eprintln!("[streaming] DeltaNet layer {i} 読み込み失敗");
                            std::process::exit(1);
                        });
                    Qwen35LayerWeights::DeltaNet(w)
                }
                crate::qwen35::LayerType::FullAttention => {
                    let w = FullAttnLayerWeights::from_tensors(&layer_prefix, get_tensor)
                        .unwrap_or_else(|| {
                            eprintln!("[streaming] FullAttn layer {i} 読み込み失敗");
                            std::process::exit(1);
                        });
                    Qwen35LayerWeights::FullAttention(w)
                }
            }
        };

        match &layer {
            Qwen35LayerWeights::DeltaNet(w) => {
                deltanet_layer_forward_eval(&mut hidden_states, w, config, seq_len);
            }
            Qwen35LayerWeights::FullAttention(w) => {
                full_attn_layer_forward_eval(&mut hidden_states, w, config, seq_len);
            }
        }
        // layer は即 drop
    }

    blas_rmsnorm(&mut hidden_states, output_norm, hidden, config.rms_norm_eps);

    let mut logits = vec![0.0f32; seq_len * vocab_size];
    blas_matmul_bt(&hidden_states, lm_head, &mut logits, seq_len, vocab_size, hidden);

    logits
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::LayerType;

    #[test]
    fn test_qwen35_layer_forward_deltanet_smoke() {
        let config = Qwen35Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            rms_norm_eps: 1e-6,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            linear_key_head_dim: 4,
            linear_num_key_heads: 2,
            linear_value_head_dim: 4,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 2,
            full_attention_interval: 4,
            layer_types: vec![LayerType::LinearAttention],
        };

        let key_dim = config.linear_key_dim();
        let val_dim = config.linear_value_dim();
        let qkv_dim = key_dim * 2 + val_dim;
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let n_v = config.linear_num_value_heads;

        let weights = DeltaNetLayerWeights {
            input_layernorm: vec![1.0; hidden],
            post_attn_layernorm: vec![1.0; hidden],
            in_proj_qkv: vec![0.01; hidden * qkv_dim],
            in_proj_z: vec![0.01; hidden * val_dim],
            in_proj_b: vec![0.01; hidden * n_v],
            in_proj_a: vec![0.01; hidden * n_v],
            a_log: vec![0.1; n_v],
            dt_bias: vec![1.0; n_v],
            conv1d_weight: vec![0.5; qkv_dim * config.linear_conv_kernel_dim],
            norm_weight: vec![1.0; config.linear_value_head_dim],
            out_proj: vec![0.01; val_dim * hidden],
            gate_proj: vec![0.01; inter * hidden],
            up_proj: vec![0.01; inter * hidden],
            down_proj: vec![0.01; hidden * inter],
        };

        let layer = Qwen35LayerWeights::DeltaNet(weights);
        let seq_len = 4;
        let mut input = vec![0.1f32; seq_len * hidden];

        let _cache = qwen35_layer_forward(&mut input, &layer, &config, seq_len);

        assert!(input.iter().all(|v| v.is_finite()), "output contains NaN/Inf");
        assert!(input.iter().any(|v| v.abs() > 1e-10), "output is all zeros");
    }

    #[test]
    fn test_qwen35_layer_forward_full_attn_smoke() {
        let config = Qwen35Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            rms_norm_eps: 1e-6,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            linear_key_head_dim: 4,
            linear_num_key_heads: 2,
            linear_value_head_dim: 4,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 2,
            full_attention_interval: 4,
            layer_types: vec![LayerType::FullAttention],
        };

        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;

        let weights = FullAttnLayerWeights {
            input_layernorm: vec![1.0; hidden],
            post_attn_layernorm: vec![1.0; hidden],
            q_proj: vec![0.01; hidden * nh * hd],
            k_proj: vec![0.01; hidden * nkv * hd],
            v_proj: vec![0.01; hidden * nkv * hd],
            o_proj: vec![0.01; nh * hd * hidden],
            q_norm: vec![1.0; hd],
            k_norm: vec![1.0; hd],
            gate_proj: vec![0.01; inter * hidden],
            up_proj: vec![0.01; inter * hidden],
            down_proj: vec![0.01; hidden * inter],
        };

        let layer = Qwen35LayerWeights::FullAttention(weights);
        let seq_len = 4;
        let mut input = vec![0.1f32; seq_len * hidden];

        let _cache = qwen35_layer_forward(&mut input, &layer, &config, seq_len);

        assert!(input.iter().all(|v| v.is_finite()), "output contains NaN/Inf");
    }
}
