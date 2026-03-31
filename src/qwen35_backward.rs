//! Qwen3.5 バックワードパス — ハイブリッド DeltaNet + Full Attention。
//!
//! forward の cache を使い、全勾配を計算する。
//! STE で FP32 潜在重みに勾配を通す。

use crate::blas::{blas_matmul_bt, blas_matmul_nn, blas_rmsnorm};
use crate::deltanet::{causal_conv1d_backward, deltanet_recurrence_backward};
use crate::qwen35::{LayerType, Qwen35Config};

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

impl FullAttnWeightGrads {
    /// 全勾配テンソルの L2 ノルムを計算。
    pub fn grad_norm(&self) -> f32 {
        let mut sum = 0.0f64;
        for v in [
            &self.d_q_proj,
            &self.d_k_proj,
            &self.d_v_proj,
            &self.d_o_proj,
            &self.d_gate_proj,
            &self.d_up_proj,
            &self.d_down_proj,
        ] {
            for &x in v.iter() {
                sum += (x as f64) * (x as f64);
            }
        }
        (sum as f32).sqrt()
    }

    /// 勾配ノルムが max_norm を超える場合、全勾配をスケーリング。
    pub fn clip_grad_norm(&mut self, max_norm: f32) {
        let norm = self.grad_norm();
        if norm > max_norm && norm > 1e-10 {
            let scale = max_norm / norm;
            for v in [
                &mut self.d_q_proj,
                &mut self.d_k_proj,
                &mut self.d_v_proj,
                &mut self.d_o_proj,
                &mut self.d_gate_proj,
                &mut self.d_up_proj,
                &mut self.d_down_proj,
            ] {
                for x in v.iter_mut() {
                    *x *= scale;
                }
            }
        }
    }
}

impl DeltaNetWeightGrads {
    /// 全勾配テンソルの L2 ノルムを計算。
    pub fn grad_norm(&self) -> f32 {
        let mut sum = 0.0f64;
        for v in [
            &self.d_in_proj_qkv,
            &self.d_in_proj_z,
            &self.d_in_proj_b,
            &self.d_in_proj_a,
            &self.d_out_proj,
            &self.d_gate_proj,
            &self.d_up_proj,
            &self.d_down_proj,
        ] {
            for &x in v.iter() {
                sum += (x as f64) * (x as f64);
            }
        }
        (sum as f32).sqrt()
    }

    /// 勾配ノルムが max_norm を超える場合、全勾配をスケーリング。
    pub fn clip_grad_norm(&mut self, max_norm: f32) {
        let norm = self.grad_norm();
        if norm > max_norm && norm > 1e-10 {
            let scale = max_norm / norm;
            for v in [
                &mut self.d_in_proj_qkv,
                &mut self.d_in_proj_z,
                &mut self.d_in_proj_b,
                &mut self.d_in_proj_a,
                &mut self.d_out_proj,
                &mut self.d_gate_proj,
                &mut self.d_up_proj,
                &mut self.d_down_proj,
            ] {
                for x in v.iter_mut() {
                    *x *= scale;
                }
            }
        }
    }
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
    pub fn apply_sgd(&self, weights: &mut crate::qwen35::DeltaNetLayerWeights, lr: f32, wd: f32) {
        sgd_update(
            &mut weights.input_layernorm,
            &self.d_input_layernorm,
            lr,
            0.0,
        );
        sgd_update(
            &mut weights.post_attn_layernorm,
            &self.d_post_attn_layernorm,
            lr,
            0.0,
        );
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
    pub fn apply_sgd(&self, weights: &mut crate::qwen35::FullAttnLayerWeights, lr: f32, wd: f32) {
        sgd_update(
            &mut weights.input_layernorm,
            &self.d_input_layernorm,
            lr,
            0.0,
        );
        sgd_update(
            &mut weights.post_attn_layernorm,
            &self.d_post_attn_layernorm,
            lr,
            0.0,
        );
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
    // down_proj: [hidden × inter], d_output: [seq × hidden] → d_intermediate: [seq × inter]
    // d_intermediate = d_output × down_proj^T だが down_proj は [hidden × inter] なので
    // C[seq×inter] = A[seq×hidden] × B[hidden×inter] → matmul_nn
    let mut d_intermediate = vec![0.0f32; seq_len * inter];
    blas_matmul_nn(
        d_output,
        down_proj,
        &mut d_intermediate,
        seq_len,
        inter,
        hidden,
    );

    // d_down_proj: d_output^T × intermediate → (hidden × inter)
    // intermediate = gate_silu ⊙ up
    let mut intermediate = vec![0.0f32; seq_len * inter];
    for i in 0..seq_len * inter {
        intermediate[i] = gate_silu[i] * up[i];
    }
    // d_down_proj[hidden×inter] += d_output[seq×hidden]^T × intermediate[seq×inter]
    // = matmul_bt(d_output^T, intermediate) だが転置が必要
    // d_output: [seq×hidden], intermediate: [seq×inter]
    // d_down_proj[h][j] = Σ_t d_output[t][h] * intermediate[t][j]
    // これは d_output を転置して掛ける: d_output^T[hidden×seq] × intermediate[seq×inter]
    // BLAS: C[h×inter] = A^T[h×seq] × B[seq×inter]
    // matmul_bt は C=A×B^T なので直接使えない。d_output^T を明示的に構築するか、
    // 対称性を利用: d_down_proj^T = intermediate^T × d_output
    // d_down_proj_t[inter×hidden] = intermediate[inter×seq]^T ... これも複雑
    // → BLAS matmul_nn で: tmp[hidden×inter] = d_output_t[hidden×seq] × intermediate[seq×inter]
    {
        // d_output を転置: [seq×hidden] → [hidden×seq]
        let mut d_output_t = vec![0.0f32; hidden * seq_len];
        for t in 0..seq_len {
            for h in 0..hidden {
                d_output_t[h * seq_len + t] = d_output[t * hidden + h];
            }
        }
        let mut tmp = vec![0.0f32; hidden * inter];
        blas_matmul_nn(&d_output_t, &intermediate, &mut tmp, hidden, inter, seq_len);
        for i in 0..hidden * inter {
            d_down_proj[i] += tmp[i];
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

    // d_gate_proj[inter×hidden] += d_gate_val[seq×inter]^T × normed_input[seq×hidden]
    // d_up_proj[inter×hidden] += d_up_val[seq×inter]^T × normed_input[seq×hidden]
    {
        let mut d_gate_t = vec![0.0f32; inter * seq_len];
        let mut d_up_t = vec![0.0f32; inter * seq_len];
        for t in 0..seq_len {
            for i in 0..inter {
                d_gate_t[i * seq_len + t] = d_gate_val[t * inter + i];
                d_up_t[i * seq_len + t] = d_up_val[t * inter + i];
            }
        }
        let mut tmp_gate = vec![0.0f32; inter * hidden];
        let mut tmp_up = vec![0.0f32; inter * hidden];
        blas_matmul_nn(
            &d_gate_t,
            normed_input,
            &mut tmp_gate,
            inter,
            hidden,
            seq_len,
        );
        blas_matmul_nn(&d_up_t, normed_input, &mut tmp_up, inter, hidden, seq_len);
        for i in 0..inter * hidden {
            d_gate_proj[i] += tmp_gate[i];
            d_up_proj[i] += tmp_up[i];
        }
    }

    // d_normed_input = d_gate × gate_proj^T + d_up × up_proj^T
    blas_matmul_bt(&d_gate_val, gate_proj, d_input, seq_len, hidden, inter);
    let mut d_input_up = vec![0.0f32; seq_len * hidden];
    blas_matmul_bt(&d_up_val, up_proj, &mut d_input_up, seq_len, hidden, inter);
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
    let key_dim = config.linear_key_dim();
    let val_dim = config.linear_value_dim();
    let n_k_heads = config.linear_num_key_heads;
    let n_v_heads = config.linear_num_value_heads;
    let dk = config.linear_key_head_dim;
    let dv = config.linear_value_head_dim;
    let kernel_size = config.linear_conv_kernel_dim;
    let qkv_dim = key_dim * 2 + val_dim;

    // ── 1. FFN backward ──
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

    // ── 2. RMSNorm backward (post_attn_layernorm) ──
    // d_pre_norm = d_ffn_input * rmsnorm_grad
    // 簡略化: RMSNorm の勾配スケーリングは diag 近似 (weight * inv_rms)
    // norm weight 勾配は d_ffn_input ⊙ normalized_input で計算
    let mut d_attn_block = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = &cache.residual_ffn[t * hidden..(t + 1) * hidden]; // attention output (pre-norm input)
        let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv_rms = 1.0 / (ss / hidden as f64 + config.rms_norm_eps as f64).sqrt() as f32;
        for h in 0..hidden {
            // post_attn_layernorm backward
            let normed = row[h] * inv_rms;
            weight_grads.d_post_attn_layernorm[h] += d_ffn_input[t * hidden + h] * normed;
            d_attn_block[t * hidden + h] =
                d_ffn_input[t * hidden + h] * weights.post_attn_layernorm[h] * inv_rms;
        }
    }

    // residual: d_attn_out = d_attn_block + d_output
    for i in 0..seq_len * hidden {
        d_attn_block[i] += d_output[i];
    }

    // ── 3. out_proj backward ──
    // attn_out = attn_normed × out_proj^T → d_attn_normed = d_attn_block × out_proj
    let mut d_attn_normed = vec![0.0f32; seq_len * val_dim];
    blas_matmul_nn(
        &d_attn_block,
        &weights.out_proj,
        &mut d_attn_normed,
        seq_len,
        val_dim,
        hidden,
    );
    // d_out_proj += d_attn_block^T × attn_normed (accumulated)
    {
        let mut d_block_t = vec![0.0f32; hidden * seq_len];
        for t in 0..seq_len {
            for h in 0..hidden {
                d_block_t[h * seq_len + t] = d_attn_block[t * hidden + h];
            }
        }
        // Need the attn_normed values — recompute from cache
        let mut attn_normed = vec![0.0f32; seq_len * val_dim];
        crate::deltanet::gated_rmsnorm(
            &cache.attn_out_pre_norm,
            &cache.z,
            &weights.norm_weight,
            &mut attn_normed,
            dv,
            config.rms_norm_eps,
        );
        let mut tmp = vec![0.0f32; hidden * val_dim];
        blas_matmul_nn(&d_block_t, &attn_normed, &mut tmp, hidden, val_dim, seq_len);
        for i in 0..hidden * val_dim {
            weight_grads.d_out_proj[i] += tmp[i];
        }
    }

    // ── 4. Gated RMSNorm backward (簡略化) ──
    // gated_rmsnorm: out = RMSNorm(x) * SiLU(z)
    // 簡略化: d_x ≈ d_attn_normed * SiLU(z) * weight * inv_rms
    let mut d_recurrence_out = vec![0.0f32; seq_len * val_dim];
    for t in 0..seq_len {
        let row_x = &cache.attn_out_pre_norm[t * val_dim..(t + 1) * val_dim];
        let row_z = &cache.z[t * val_dim..(t + 1) * val_dim];
        let ss: f64 = row_x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv_rms = 1.0 / (ss / dv as f64 + config.rms_norm_eps as f64).sqrt() as f32;
        for j in 0..dv {
            let sig_z = 1.0 / (1.0 + (-row_z[j]).exp());
            let silu_z = row_z[j] * sig_z;
            d_recurrence_out[t * val_dim + j] =
                d_attn_normed[t * val_dim + j] * silu_z * weights.norm_weight[j % dv] * inv_rms;
            // norm weight 勾配
            let normed_x = row_x[j] * inv_rms;
            weight_grads.d_norm_weight[j % dv] +=
                d_attn_normed[t * val_dim + j] * normed_x * silu_z;
        }
    }

    // ── 5. DeltaNet recurrence backward ──
    let rec_grads = deltanet_recurrence_backward(
        &d_recurrence_out,
        &cache.step_caches,
        &weights.a_log,
        &weights.dt_bias,
        n_v_heads,
        dk,
        dv,
        seq_len,
    );
    for h in 0..n_v_heads {
        weight_grads.d_a_log[h] += rec_grads.d_a_log[h];
        weight_grads.d_dt_bias[h] += rec_grads.d_dt_bias[h];
    }

    // ── 6. Projection backward (through conv1d + projections) ──
    // rec_grads.d_q/d_k: [seq × n_v_heads × dk] (expanded), d_v: [seq × n_v_heads × dv]
    // conv1d operates on qkv_dim = key_dim*2 + val_dim = n_k_heads*dk*2 + n_v_heads*dv
    // GQA: n_v_heads=32, n_k_heads=16 → 2:1 ratio. Contract by averaging pairs.
    // 簡略化: expanded d_q/d_k (n_v_heads=32) → contract to n_k_heads=16 by sum
    let mut d_qkv_conv = vec![0.0f32; seq_len * qkv_dim];
    let gqa_ratio = n_v_heads / n_k_heads;
    for t in 0..seq_len {
        // Q: contract n_v_heads → n_k_heads
        for kh in 0..n_k_heads {
            for d in 0..dk {
                let mut sum = 0.0f32;
                for g in 0..gqa_ratio {
                    let vh = kh * gqa_ratio + g;
                    sum += rec_grads.d_q[t * n_v_heads * dk + vh * dk + d];
                }
                d_qkv_conv[t * qkv_dim + kh * dk + d] = sum;
            }
        }
        // K: contract n_v_heads → n_k_heads
        for kh in 0..n_k_heads {
            for d in 0..dk {
                let mut sum = 0.0f32;
                for g in 0..gqa_ratio {
                    let vh = kh * gqa_ratio + g;
                    sum += rec_grads.d_k[t * n_v_heads * dk + vh * dk + d];
                }
                d_qkv_conv[t * qkv_dim + key_dim + kh * dk + d] = sum;
            }
        }
        // V: direct copy
        for j in 0..val_dim {
            d_qkv_conv[t * qkv_dim + key_dim * 2 + j] = rec_grads.d_v[t * n_v_heads * dv + j];
        }
    }
    // conv1d backward
    let mut d_qkv_pre_conv = vec![0.0f32; seq_len * qkv_dim];
    causal_conv1d_backward(
        &d_qkv_conv,
        &cache.pre_conv_qkv,
        &weights.conv1d_weight,
        &mut d_qkv_pre_conv,
        &mut weight_grads.d_conv1d_weight,
        qkv_dim,
        seq_len,
        kernel_size,
    );

    // in_proj_qkv backward: d_normed = d_qkv_pre_conv × in_proj_qkv
    let mut d_normed = vec![0.0f32; seq_len * hidden];
    blas_matmul_nn(
        &d_qkv_pre_conv,
        &weights.in_proj_qkv,
        &mut d_normed,
        seq_len,
        hidden,
        qkv_dim,
    );
    // d_in_proj_qkv += d_qkv_pre_conv^T × normed_input (recompute normed)
    {
        let mut normed_input = cache.residual_attn.clone();
        blas_rmsnorm(
            &mut normed_input,
            &weights.input_layernorm,
            hidden,
            config.rms_norm_eps,
        );
        let mut d_qkv_t = vec![0.0f32; qkv_dim * seq_len];
        for t in 0..seq_len {
            for j in 0..qkv_dim {
                d_qkv_t[j * seq_len + t] = d_qkv_pre_conv[t * qkv_dim + j];
            }
        }
        let mut tmp = vec![0.0f32; qkv_dim * hidden];
        blas_matmul_nn(&d_qkv_t, &normed_input, &mut tmp, qkv_dim, hidden, seq_len);
        for i in 0..qkv_dim * hidden {
            weight_grads.d_in_proj_qkv[i] += tmp[i];
        }
    }

    // in_proj_b, in_proj_a backward (from rec_grads.d_b_logit, d_a_logit)
    {
        let mut normed_input = cache.residual_attn.clone();
        blas_rmsnorm(
            &mut normed_input,
            &weights.input_layernorm,
            hidden,
            config.rms_norm_eps,
        );
        // d_in_proj_b += d_b_logit^T × normed
        let mut d_b_t = vec![0.0f32; n_v_heads * seq_len];
        let mut d_a_t = vec![0.0f32; n_v_heads * seq_len];
        for t in 0..seq_len {
            for h in 0..n_v_heads {
                d_b_t[h * seq_len + t] = rec_grads.d_b_logit[t * n_v_heads + h];
                d_a_t[h * seq_len + t] = rec_grads.d_a_logit[t * n_v_heads + h];
            }
        }
        let mut tmp_b = vec![0.0f32; n_v_heads * hidden];
        let mut tmp_a = vec![0.0f32; n_v_heads * hidden];
        blas_matmul_nn(
            &d_b_t,
            &normed_input,
            &mut tmp_b,
            n_v_heads,
            hidden,
            seq_len,
        );
        blas_matmul_nn(
            &d_a_t,
            &normed_input,
            &mut tmp_a,
            n_v_heads,
            hidden,
            seq_len,
        );
        for i in 0..n_v_heads * hidden {
            weight_grads.d_in_proj_b[i] += tmp_b[i];
            weight_grads.d_in_proj_a[i] += tmp_a[i];
        }
    }

    // ── 7. Input RMSNorm backward ──
    let mut d_input = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = &cache.residual_attn[t * hidden..(t + 1) * hidden];
        let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv_rms = 1.0 / (ss / hidden as f64 + config.rms_norm_eps as f64).sqrt() as f32;
        for h in 0..hidden {
            let normed = row[h] * inv_rms;
            weight_grads.d_input_layernorm[h] += d_normed[t * hidden + h] * normed;
            d_input[t * hidden + h] =
                d_normed[t * hidden + h] * weights.input_layernorm[h] * inv_rms;
        }
    }

    // residual skip: d_input += d_attn_block (from FFN path) + d_output
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
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;

    // ── 1. FFN backward ──
    // d_output は FFN 出力の勾配 + residual
    // FFN residual: d_ffn_residual = d_output
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

    // FFN residual add backward: d_pre_ffn = d_ffn_input + d_output
    let mut d_pre_ffn = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden {
        d_pre_ffn[i] = d_ffn_input[i] + d_output[i];
    }

    // ── 2. O projection backward ──
    // attn_out = attn_out_raw × o_proj^T → d_attn_out_raw = d_pre_ffn × o_proj
    let mut d_attn_out_raw = vec![0.0f32; seq_len * num_heads * head_dim];
    blas_matmul_nn(
        &d_pre_ffn,
        &weights.o_proj,
        &mut d_attn_out_raw,
        seq_len,
        num_heads * head_dim,
        hidden,
    );

    // d_o_proj: d_pre_ffn^T × attn_out_raw (cache.attn_out is post-o_proj, need pre-o_proj)
    // We don't have pre-o_proj attn_out in cache directly.
    // For weight grad accumulation: d_o_proj[h×(nh*hd)] += d_pre_ffn[seq×h]^T × attn_out_raw[seq×(nh*hd)]
    // Skip o_proj weight grads for now (SGD updates projections in-place during forward)

    // ── 3. GQA Attention backward ──
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

    let mut d_q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut d_k = vec![0.0f32; seq_len * num_kv_heads * head_dim];
    let mut d_v = vec![0.0f32; seq_len * num_kv_heads * head_dim];

    #[cfg(feature = "cuda")]
    let used_cuda = {
        if crate::blas::cuda_blas_available() {
            let cuda_mtx = crate::blas::CUDA_MATMUL.get().unwrap();
            let cuda = cuda_mtx.lock().unwrap();
            crate::cuda_matmul::cuda_gqa_attention_backward(
                &cuda,
                &d_attn_out_raw,
                &cache.attn_weights,
                &cache.q,
                &cache.k,
                &cache.v,
                &mut d_q,
                &mut d_k,
                &mut d_v,
                &llama_compat,
                seq_len,
            );
            true
        } else {
            false
        }
    };
    #[cfg(not(feature = "cuda"))]
    let used_cuda = false;

    if !used_cuda {
        // CPU fallback: per-head attention backward
        let kv_group_size = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let extract = |data: &[f32], head: usize, nh: usize, hd: usize, sl: usize| -> Vec<f32> {
            let stride = nh * hd;
            let mut out = vec![0.0f32; sl * hd];
            for t in 0..sl {
                out[t * hd..(t + 1) * hd]
                    .copy_from_slice(&data[t * stride + head * hd..t * stride + (head + 1) * hd]);
            }
            out
        };
        let scatter =
            |src: &[f32], dst: &mut [f32], head: usize, nh: usize, hd: usize, sl: usize| {
                let stride = nh * hd;
                for t in 0..sl {
                    dst[t * stride + head * hd..t * stride + (head + 1) * hd]
                        .copy_from_slice(&src[t * hd..(t + 1) * hd]);
                }
            };
        let accumulate =
            |src: &[f32], dst: &mut [f32], head: usize, nh: usize, hd: usize, sl: usize| {
                let stride = nh * hd;
                for t in 0..sl {
                    for d in 0..hd {
                        dst[t * stride + head * hd + d] += src[t * hd + d];
                    }
                }
            };
        let softmax_bwd = |d_aw: &[f32], aw: &[f32], sl: usize| -> Vec<f32> {
            let mut ds = vec![0.0f32; sl * sl];
            for t in 0..sl {
                let a_row = &aw[t * sl..(t + 1) * sl];
                let d_row = &d_aw[t * sl..(t + 1) * sl];
                let dot: f32 = a_row.iter().zip(d_row.iter()).map(|(a, d)| a * d).sum();
                for s in 0..sl {
                    ds[t * sl + s] = a_row[s] * (d_row[s] - dot);
                }
            }
            ds
        };
        let matmul_tn =
            |a: &[f32], b: &[f32], rows_a: usize, cols_b: usize, inner: usize| -> Vec<f32> {
                let mut out = vec![0.0f32; rows_a * cols_b];
                for r in 0..rows_a {
                    for c in 0..cols_b {
                        let mut s = 0.0f32;
                        for t in 0..inner {
                            s += a[t * rows_a + r] * b[t * cols_b + c];
                        }
                        out[r * cols_b + c] = s;
                    }
                }
                out
            };

        for h in 0..num_heads {
            let kv_h = h / kv_group_size;
            let d_out_h = extract(&d_attn_out_raw, h, num_heads, head_dim, seq_len);
            let v_h = extract(&cache.v, kv_h, num_kv_heads, head_dim, seq_len);
            let aw_h = &cache.attn_weights[h * seq_len * seq_len..(h + 1) * seq_len * seq_len];

            let mut d_attn_w = vec![0.0f32; seq_len * seq_len];
            blas_matmul_bt(&d_out_h, &v_h, &mut d_attn_w, seq_len, seq_len, head_dim);

            let d_v_h = matmul_tn(aw_h, &d_out_h, seq_len, head_dim, seq_len);
            accumulate(&d_v_h, &mut d_v, kv_h, num_kv_heads, head_dim, seq_len);

            let mut d_scores = softmax_bwd(&d_attn_w, aw_h, seq_len);
            for s in &mut d_scores {
                *s *= scale;
            }

            let k_h = extract(&cache.k, kv_h, num_kv_heads, head_dim, seq_len);
            let mut d_q_h = vec![0.0f32; seq_len * head_dim];
            blas_matmul_nn(&d_scores, &k_h, &mut d_q_h, seq_len, head_dim, seq_len);
            scatter(&d_q_h, &mut d_q, h, num_heads, head_dim, seq_len);

            let q_h = extract(&cache.q, h, num_heads, head_dim, seq_len);
            let d_k_h = matmul_tn(&d_scores, &q_h, seq_len, head_dim, seq_len);
            accumulate(&d_k_h, &mut d_k, kv_h, num_kv_heads, head_dim, seq_len);
        }
    }

    // ── 4. QKV projection backward → d_normed_attn ──
    // d_normed = d_q × q_proj + d_k × k_proj + d_v × v_proj
    let mut d_normed = vec![0.0f32; seq_len * hidden];
    {
        let mut tmp = vec![0.0f32; seq_len * hidden];
        blas_matmul_nn(
            &d_q,
            &weights.q_proj,
            &mut tmp,
            seq_len,
            hidden,
            num_heads * head_dim,
        );
        for i in 0..seq_len * hidden {
            d_normed[i] += tmp[i];
        }
        blas_matmul_nn(
            &d_k,
            &weights.k_proj,
            &mut tmp,
            seq_len,
            hidden,
            num_kv_heads * head_dim,
        );
        for i in 0..seq_len * hidden {
            d_normed[i] += tmp[i];
        }
        blas_matmul_nn(
            &d_v,
            &weights.v_proj,
            &mut tmp,
            seq_len,
            hidden,
            num_kv_heads * head_dim,
        );
        for i in 0..seq_len * hidden {
            d_normed[i] += tmp[i];
        }
    }

    // ── 5. Attention residual backward ──
    // d_input = d_normed (through layernorm is identity for eval) + d_pre_ffn (residual)
    let mut d_input = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden {
        d_input[i] = d_normed[i] + d_pre_ffn[i];
    }

    d_input
}

/// ハイブリッドレイヤーの backward。
#[must_use]
/// DeltaNet層のGradient Checkpointing Fused — GPU forward(VRAM state保持) + GPU backward
/// step_caches を一切生成しない。VRAM上の保存済みS_{t-1}で backward を直接実行。
#[cfg(feature = "cuda")]
pub fn deltanet_layer_gc_fused(
    saved_input: &[f32],
    d_output: &[f32],
    weights: &crate::qwen35::DeltaNetLayerWeights,
    config: &Qwen35Config,
    seq_len: usize,
    weight_grads: &mut DeltaNetWeightGrads,
) -> Vec<f32> {
    use crate::blas::blas_swiglu_ffn_training;
    use crate::deltanet::causal_conv1d_silu_row_major;

    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let key_dim = config.linear_key_dim();
    let val_dim = config.linear_value_dim();
    let n_k_heads = config.linear_num_key_heads;
    let n_v_heads = config.linear_num_value_heads;
    let dk = config.linear_key_head_dim;
    let dv = config.linear_value_head_dim;
    let kernel_size = config.linear_conv_kernel_dim;
    let qkv_dim = key_dim * 2 + val_dim;

    // ═══ Phase 1: Pre-recurrence forward (projections, conv1d, gates) — GPU ═══
    let residual_attn = saved_input.to_vec();
    let mut normed = saved_input.to_vec();
    blas_rmsnorm(
        &mut normed,
        &weights.input_layernorm,
        hidden,
        config.rms_norm_eps,
    );

    let mut qkv = vec![0.0f32; seq_len * qkv_dim];
    blas_matmul_bt(
        &normed,
        &weights.in_proj_qkv,
        &mut qkv,
        seq_len,
        qkv_dim,
        hidden,
    );
    let mut z = vec![0.0f32; seq_len * val_dim];
    blas_matmul_bt(
        &normed,
        &weights.in_proj_z,
        &mut z,
        seq_len,
        val_dim,
        hidden,
    );
    let mut b_raw = vec![0.0f32; seq_len * n_v_heads];
    blas_matmul_bt(
        &normed,
        &weights.in_proj_b,
        &mut b_raw,
        seq_len,
        n_v_heads,
        hidden,
    );
    let mut a_raw = vec![0.0f32; seq_len * n_v_heads];
    blas_matmul_bt(
        &normed,
        &weights.in_proj_a,
        &mut a_raw,
        seq_len,
        n_v_heads,
        hidden,
    );

    let pre_conv_qkv = qkv.clone();
    let mut qkv_conv = vec![0.0f32; seq_len * qkv_dim];
    causal_conv1d_silu_row_major(
        &qkv,
        &weights.conv1d_weight,
        &mut qkv_conv,
        qkv_dim,
        seq_len,
        kernel_size,
    );

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
    {
        use crate::blas::CUDA_MATMUL;
        let cuda_mtx = CUDA_MATMUL.get().expect("CUDA 未初期化");
        let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
        crate::cuda_matmul::cuda_dn_l2norm_gqa(
            &cuda,
            &q_raw,
            &k_raw,
            &mut q_expanded,
            &mut k_expanded,
            seq_len,
            n_k_heads,
            n_v_heads,
            dk,
            1e-6,
        );
    }

    let mut beta = vec![0.0f32; seq_len * n_v_heads];
    let mut g = vec![0.0f32; seq_len * n_v_heads];
    {
        use crate::blas::CUDA_MATMUL;
        let cuda_mtx = CUDA_MATMUL.get().expect("CUDA 未初期化");
        let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
        crate::cuda_matmul::cuda_dn_gate_compute(
            &cuda,
            &b_raw,
            &a_raw,
            &weights.a_log,
            &weights.dt_bias,
            &mut beta,
            &mut g,
            seq_len,
            n_v_heads,
        );
    }

    // ═══ Phase 2: GPU Forward (VRAM state保持) ═══
    let total_qk = n_v_heads * seq_len * dk;
    let total_v = n_v_heads * seq_len * dv;
    let total_bg = n_v_heads * seq_len;

    // Layout: [seq, heads, dim] → [heads, seq, dim]
    let mut q_gpu = vec![0.0f32; total_qk];
    let mut k_gpu = vec![0.0f32; total_qk];
    let mut v_gpu = vec![0.0f32; total_v];
    let mut beta_gpu = vec![0.0f32; total_bg];
    let mut g_gpu = vec![0.0f32; total_bg];

    for t in 0..seq_len {
        for h in 0..n_v_heads {
            let sq = t * n_v_heads * dk + h * dk;
            let dq = h * seq_len * dk + t * dk;
            q_gpu[dq..dq + dk].copy_from_slice(&q_expanded[sq..sq + dk]);
            k_gpu[dq..dq + dk].copy_from_slice(&k_expanded[sq..sq + dk]);
            let sv = t * n_v_heads * dv + h * dv;
            let dvv = h * seq_len * dv + t * dv;
            v_gpu[dvv..dvv + dv].copy_from_slice(&v_all[sv..sv + dv]);
            beta_gpu[h * seq_len + t] = beta[t * n_v_heads + h];
            g_gpu[h * seq_len + t] = g[t * n_v_heads + h];
        }
    }

    let all_s_vram;
    let mut attn_out_raw = vec![0.0f32; seq_len * val_dim];
    {
        let mut out_gpu = vec![0.0f32; total_v];
        {
            use crate::blas::CUDA_MATMUL;
            let cuda_mtx = CUDA_MATMUL.get().expect("CUDA 未初期化");
            let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
            all_s_vram = crate::cuda_matmul::cuda_deltanet_forward_store(
                &cuda,
                &q_gpu,
                &k_gpu,
                &v_gpu,
                &beta_gpu,
                &g_gpu,
                &mut out_gpu,
                n_v_heads,
                seq_len,
                dk,
                dv,
            );
        }
        // Layout back: [heads, seq, dv] → [seq, heads, dv]
        for t in 0..seq_len {
            for h in 0..n_v_heads {
                let src = h * seq_len * dv + t * dv;
                let dst = t * n_v_heads * dv + h * dv;
                attn_out_raw[dst..dst + dv].copy_from_slice(&out_gpu[src..src + dv]);
            }
        }
    }

    // ═══ Phase 3: Post-recurrence forward (gated rmsnorm, out_proj, FFN) ═══
    let attn_out_pre_norm = attn_out_raw.clone();
    let mut attn_normed = vec![0.0f32; seq_len * val_dim];
    {
        use crate::blas::CUDA_MATMUL;
        let cuda_mtx = CUDA_MATMUL.get().expect("CUDA 未初期化");
        let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
        crate::cuda_matmul::cuda_dn_gated_rmsnorm(
            &cuda,
            &attn_out_raw,
            &z,
            &weights.norm_weight,
            &mut attn_normed,
            dv,
            config.rms_norm_eps,
        );
    }

    let mut attn_out = vec![0.0f32; seq_len * hidden];
    blas_matmul_bt(
        &attn_normed,
        &weights.out_proj,
        &mut attn_out,
        seq_len,
        hidden,
        val_dim,
    );

    let mut layer_out = saved_input.to_vec();
    for i in 0..layer_out.len() {
        layer_out[i] = residual_attn[i] + attn_out[i];
    }

    let residual_ffn = layer_out.clone();
    let mut normed_ffn = layer_out.clone();
    blas_rmsnorm(
        &mut normed_ffn,
        &weights.post_attn_layernorm,
        hidden,
        config.rms_norm_eps,
    );
    let normed_ffn_saved = normed_ffn.clone();

    let mut ffn_out = vec![0.0f32; seq_len * hidden];
    let mut gate_buf = vec![0.0f32; seq_len * inter];
    let mut up_buf = vec![0.0f32; seq_len * inter];
    let mut gate_silu_buf = vec![0.0f32; seq_len * inter];
    blas_swiglu_ffn_training(
        &normed_ffn,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
        &mut ffn_out,
        &mut gate_buf,
        &mut up_buf,
        &mut gate_silu_buf,
        seq_len,
        hidden,
        inter,
    );

    // ═══ Phase 4: Post-recurrence backward (FFN → gated rmsnorm → out_proj) ═══
    let mut d_ffn_input = vec![0.0f32; seq_len * hidden];
    swiglu_ffn_backward(
        d_output,
        &normed_ffn_saved,
        &gate_buf,
        &up_buf,
        &gate_silu_buf,
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

    let mut d_attn_block = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = &residual_ffn[t * hidden..(t + 1) * hidden];
        let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv_rms = 1.0 / (ss / hidden as f64 + config.rms_norm_eps as f64).sqrt() as f32;
        for h in 0..hidden {
            let n = row[h] * inv_rms;
            weight_grads.d_post_attn_layernorm[h] += d_ffn_input[t * hidden + h] * n;
            d_attn_block[t * hidden + h] =
                d_ffn_input[t * hidden + h] * weights.post_attn_layernorm[h] * inv_rms;
        }
    }
    for i in 0..seq_len * hidden {
        d_attn_block[i] += d_output[i];
    }

    let mut d_attn_normed = vec![0.0f32; seq_len * val_dim];
    blas_matmul_nn(
        &d_attn_block,
        &weights.out_proj,
        &mut d_attn_normed,
        seq_len,
        val_dim,
        hidden,
    );
    {
        let mut d_block_t = vec![0.0f32; hidden * seq_len];
        for t in 0..seq_len {
            for h in 0..hidden {
                d_block_t[h * seq_len + t] = d_attn_block[t * hidden + h];
            }
        }
        let mut tmp = vec![0.0f32; hidden * val_dim];
        blas_matmul_nn(&d_block_t, &attn_normed, &mut tmp, hidden, val_dim, seq_len);
        for i in 0..hidden * val_dim {
            weight_grads.d_out_proj[i] += tmp[i];
        }
    }

    let mut d_recurrence_out = vec![0.0f32; seq_len * val_dim];
    for t in 0..seq_len {
        let row_x = &attn_out_pre_norm[t * val_dim..(t + 1) * val_dim];
        let row_z = &z[t * val_dim..(t + 1) * val_dim];
        let ss: f64 = row_x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv_rms = 1.0 / (ss / dv as f64 + config.rms_norm_eps as f64).sqrt() as f32;
        for j in 0..dv {
            let sig_z = 1.0 / (1.0 + (-row_z[j]).exp());
            let silu_z = row_z[j] * sig_z;
            d_recurrence_out[t * val_dim + j] =
                d_attn_normed[t * val_dim + j] * silu_z * weights.norm_weight[j % dv] * inv_rms;
            let normed_x = row_x[j] * inv_rms;
            weight_grads.d_norm_weight[j % dv] +=
                d_attn_normed[t * val_dim + j] * normed_x * silu_z;
        }
    }

    // ═══ Phase 5: GPU Backward (VRAM state使用) ═══
    let mut do_gpu = vec![0.0f32; total_v];
    for t in 0..seq_len {
        for h in 0..n_v_heads {
            let src = t * n_v_heads * dv + h * dv;
            let dst = h * seq_len * dv + t * dv;
            do_gpu[dst..dst + dv].copy_from_slice(&d_recurrence_out[src..src + dv]);
        }
    }

    let mut dq_gpu = vec![0.0f32; total_qk];
    let mut dk_gpu_out = vec![0.0f32; total_qk];
    let mut dv_gpu_out = vec![0.0f32; total_v];
    let mut dbeta_gpu = vec![0.0f32; total_bg];
    let mut dg_gpu = vec![0.0f32; total_bg];

    {
        use crate::blas::CUDA_MATMUL;
        let cuda_mtx = CUDA_MATMUL.get().expect("CUDA 未初期化");
        let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
        crate::cuda_matmul::cuda_deltanet_backward_from_vram(
            &cuda,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &beta_gpu,
            &g_gpu,
            &do_gpu,
            &all_s_vram,
            &mut dq_gpu,
            &mut dk_gpu_out,
            &mut dv_gpu_out,
            &mut dbeta_gpu,
            &mut dg_gpu,
            n_v_heads,
            seq_len,
            dk,
            dv,
        );
    }
    drop(all_s_vram); // VRAM解放

    // Layout back: [heads, seq, dim] → [seq, heads, dim]
    let mut d_q = vec![0.0f32; total_qk];
    let mut d_k = vec![0.0f32; total_qk];
    let mut d_v = vec![0.0f32; total_v];
    let mut d_b_logit = vec![0.0f32; total_bg];
    let mut d_a_logit = vec![0.0f32; total_bg];
    for t in 0..seq_len {
        for h in 0..n_v_heads {
            let sq = h * seq_len * dk + t * dk;
            let dq = t * n_v_heads * dk + h * dk;
            d_q[dq..dq + dk].copy_from_slice(&dq_gpu[sq..sq + dk]);
            d_k[dq..dq + dk].copy_from_slice(&dk_gpu_out[sq..sq + dk]);
            let sv = h * seq_len * dv + t * dv;
            let dvv = t * n_v_heads * dv + h * dv;
            d_v[dvv..dvv + dv].copy_from_slice(&dv_gpu_out[sv..sv + dv]);
            d_b_logit[t * n_v_heads + h] = dbeta_gpu[h * seq_len + t];
            d_a_logit[t * n_v_heads + h] = dg_gpu[h * seq_len + t];
        }
    }

    // ═══ Phase 6: Projection backward ═══
    let gqa_ratio = n_v_heads / n_k_heads;
    let mut d_qkv_conv = vec![0.0f32; seq_len * qkv_dim];
    for t in 0..seq_len {
        for kh in 0..n_k_heads {
            for d in 0..dk {
                let mut sum = 0.0f32;
                for gg in 0..gqa_ratio {
                    sum += d_q[t * n_v_heads * dk + (kh * gqa_ratio + gg) * dk + d];
                }
                d_qkv_conv[t * qkv_dim + kh * dk + d] = sum;
            }
        }
        for kh in 0..n_k_heads {
            for d in 0..dk {
                let mut sum = 0.0f32;
                for gg in 0..gqa_ratio {
                    sum += d_k[t * n_v_heads * dk + (kh * gqa_ratio + gg) * dk + d];
                }
                d_qkv_conv[t * qkv_dim + key_dim + kh * dk + d] = sum;
            }
        }
        for j in 0..val_dim {
            d_qkv_conv[t * qkv_dim + key_dim * 2 + j] = d_v[t * n_v_heads * dv + j];
        }
    }

    let mut d_qkv_pre_conv = vec![0.0f32; seq_len * qkv_dim];
    causal_conv1d_backward(
        &d_qkv_conv,
        &pre_conv_qkv,
        &weights.conv1d_weight,
        &mut d_qkv_pre_conv,
        &mut weight_grads.d_conv1d_weight,
        qkv_dim,
        seq_len,
        kernel_size,
    );

    let mut d_normed = vec![0.0f32; seq_len * hidden];
    blas_matmul_nn(
        &d_qkv_pre_conv,
        &weights.in_proj_qkv,
        &mut d_normed,
        seq_len,
        hidden,
        qkv_dim,
    );
    {
        let mut normed_input = residual_attn.clone();
        blas_rmsnorm(
            &mut normed_input,
            &weights.input_layernorm,
            hidden,
            config.rms_norm_eps,
        );
        let mut d_qkv_t = vec![0.0f32; qkv_dim * seq_len];
        for t in 0..seq_len {
            for j in 0..qkv_dim {
                d_qkv_t[j * seq_len + t] = d_qkv_pre_conv[t * qkv_dim + j];
            }
        }
        let mut tmp = vec![0.0f32; qkv_dim * hidden];
        blas_matmul_nn(&d_qkv_t, &normed_input, &mut tmp, qkv_dim, hidden, seq_len);
        for i in 0..qkv_dim * hidden {
            weight_grads.d_in_proj_qkv[i] += tmp[i];
        }
    }
    {
        let mut normed_input = residual_attn.clone();
        blas_rmsnorm(
            &mut normed_input,
            &weights.input_layernorm,
            hidden,
            config.rms_norm_eps,
        );
        let mut d_b_t = vec![0.0f32; n_v_heads * seq_len];
        let mut d_a_t = vec![0.0f32; n_v_heads * seq_len];
        for t in 0..seq_len {
            for h in 0..n_v_heads {
                d_b_t[h * seq_len + t] = d_b_logit[t * n_v_heads + h];
                d_a_t[h * seq_len + t] = d_a_logit[t * n_v_heads + h];
            }
        }
        let mut tmp_b = vec![0.0f32; n_v_heads * hidden];
        let mut tmp_a = vec![0.0f32; n_v_heads * hidden];
        blas_matmul_nn(
            &d_b_t,
            &normed_input,
            &mut tmp_b,
            n_v_heads,
            hidden,
            seq_len,
        );
        blas_matmul_nn(
            &d_a_t,
            &normed_input,
            &mut tmp_a,
            n_v_heads,
            hidden,
            seq_len,
        );
        for i in 0..n_v_heads * hidden {
            weight_grads.d_in_proj_b[i] += tmp_b[i];
            weight_grads.d_in_proj_a[i] += tmp_a[i];
        }
    }

    let mut d_input = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = &residual_attn[t * hidden..(t + 1) * hidden];
        let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv_rms = 1.0 / (ss / hidden as f64 + config.rms_norm_eps as f64).sqrt() as f32;
        for h in 0..hidden {
            let n = row[h] * inv_rms;
            weight_grads.d_input_layernorm[h] += d_normed[t * hidden + h] * n;
            d_input[t * hidden + h] =
                d_normed[t * hidden + h] * weights.input_layernorm[h] * inv_rms;
        }
    }
    for i in 0..seq_len * hidden {
        d_input[i] += d_output[i];
    }

    d_input
}

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
            let d_input =
                deltanet_layer_backward(d_output, c, w, config, seq_len, lr, wd, &mut grads);
            (d_input, Qwen35WeightGrads::DeltaNet(grads))
        }
        (
            crate::qwen35_forward::Qwen35LayerCache::FullAttention(c),
            crate::qwen35::Qwen35LayerWeights::FullAttention(w),
        ) => {
            let mut grads = FullAttnWeightGrads::zeros(config);
            let d_input =
                full_attn_layer_backward(d_output, c, w, config, seq_len, lr, wd, &mut grads);
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
        assert_eq!(
            grads.d_in_proj_qkv.len(),
            config.hidden_size * (config.linear_key_dim() * 2 + config.linear_value_dim())
        );
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
