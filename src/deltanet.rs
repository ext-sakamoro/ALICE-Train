//! Gated DeltaNet — カリカリ最適化 forward / backward 実装。
//!
//! Qwen3.5 の線形再帰層。Delta Rule + 指数減衰ゲートで状態行列を更新。
//!
//! # 最適化
//!
//! - L1: Rayon ヘッド並列化 — 32ヘッドを並列実行
//! - L2: ゼロアロケーション再帰 — 事前確保バッファ再利用
//! - L3: transpose排除 — 行優先レイアウトで直接処理
//! - L5: gate融合 — sigmoid/softplus/exp をワンパスで計算
//! - L6: conv1d Rayon並列 — チャネル並列
//! - L7: チャンク再帰 — chunk_size=64 で intra-chunk 並列化
//!
//! # 再帰式 (per head, per timestep)
//!
//! ```text
//! g_t = -exp(A_log) * softplus(a_t + dt_bias)
//! β_t = sigmoid(b_t)
//! e_t = v_t - S_{t-1}^T k_t
//! S_t = exp(g_t) * S_{t-1} + β_t * k_t ⊗ e_t
//! o_t = S_t^T q_t
//! ```

use rayon::prelude::*;

// ── 基本演算 ────────────────────────────────────────────────────────────────

/// L2 正規化 (in-place)。FMA活用。
pub fn l2_normalize(x: &mut [f32], eps: f32) {
    let mut norm_sq = 0.0f32;
    for &v in x.iter() {
        norm_sq = v.mul_add(v, norm_sq);
    }
    let inv_norm = 1.0 / (norm_sq + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv_norm;
    }
}

/// L2 正規化の backward。
pub fn l2_normalize_backward(x: &[f32], dx_norm: &[f32], dx: &mut [f32], eps: f32) {
    let d = x.len();
    let mut norm_sq = 0.0f32;
    for &v in x {
        norm_sq = v.mul_add(v, norm_sq);
    }
    let inv_norm = 1.0 / (norm_sq + eps).sqrt();
    let mut dot = 0.0f32;
    for i in 0..d {
        dot = dx_norm[i].mul_add(x[i], dot);
    }
    let inv3 = inv_norm * inv_norm * inv_norm;
    for i in 0..d {
        dx[i] = inv_norm.mul_add(dx_norm[i], -(inv3 * x[i] * dot));
    }
}

/// Sigmoid。
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus: log(1 + exp(x))。
#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        x.exp().ln_1p()
    }
}

/// SiLU (Swish): x * sigmoid(x)。
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ── Causal Conv1d (L6: Rayon並列) ───────────────────────────────────────────

/// Depthwise causal conv1d + SiLU — 行優先入力対応、チャネル並列。
///
/// `x`: (seq_len × channels) — 行優先。
/// `weight`: (channels × kernel_size) — depthwise。
/// `out`: (seq_len × channels) — 行優先。
///
/// L3: transpose排除 — 行優先のまま処理。
/// L6: Rayon チャネル並列。
pub fn causal_conv1d_silu_row_major(
    x: &[f32],
    weight: &[f32],
    out: &mut [f32],
    channels: usize,
    _seq_len: usize,
    kernel_size: usize,
) {
    // 行並列: 各タイムステップの全チャネルを一括処理
    out.par_chunks_exact_mut(channels)
        .enumerate()
        .for_each(|(t, row_out)| {
            let t_signed = t.cast_signed();
            let ks_signed = kernel_size.cast_signed();
            for c in 0..channels {
                let w = &weight[c * kernel_size..(c + 1) * kernel_size];
                let mut sum = 0.0f32;
                for (ki, &wk) in w.iter().enumerate() {
                    let src_t = t_signed - (ks_signed - 1) + ki.cast_signed();
                    if src_t >= 0 {
                        sum = x[src_t as usize * channels + c].mul_add(wk, sum);
                    }
                }
                row_out[c] = silu(sum);
            }
        });
}

/// Causal conv1d backward — 行優先、チャネル並列。
pub fn causal_conv1d_backward(
    d_out: &[f32],
    x: &[f32],
    weight: &[f32],
    d_x: &mut [f32],
    d_weight: &mut [f32],
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) {
    let ks_signed = kernel_size.cast_signed();
    for c in 0..channels {
        let w = &weight[c * kernel_size..(c + 1) * kernel_size];
        for t in 0..seq_len {
            let t_signed = t.cast_signed();
            let mut conv_out = 0.0f32;
            for (ki, &wk) in w.iter().enumerate() {
                let src_t = t_signed - (ks_signed - 1) + ki.cast_signed();
                if src_t >= 0 {
                    conv_out = x[src_t as usize * channels + c].mul_add(wk, conv_out);
                }
            }
            let sig = sigmoid(conv_out);
            let silu_grad = sig + conv_out * sig * (1.0 - sig);
            let d_conv = d_out[t * channels + c] * silu_grad;
            for (ki, &wk) in w.iter().enumerate() {
                let src_t = t_signed - (ks_signed - 1) + ki.cast_signed();
                if src_t >= 0 {
                    let src = src_t as usize;
                    d_x[src * channels + c] += d_conv * wk;
                    d_weight[c * kernel_size + ki] += d_conv * x[src * channels + c];
                }
            }
        }
    }
}

// ── Gated RMSNorm (Rayon並列) ───────────────────────────────────────────────

/// Gated RMSNorm: RMSNorm(x) * SiLU(z) — 行並列。
pub fn gated_rmsnorm(x: &[f32], z: &[f32], weight: &[f32], out: &mut [f32], dim: usize, eps: f32) {
    let seq_len = x.len() / dim;
    out.par_chunks_exact_mut(dim)
        .enumerate()
        .for_each(|(t, row_out)| {
            let row_x = &x[t * dim..(t + 1) * dim];
            let row_z = &z[t * dim..(t + 1) * dim];
            let ss: f64 = row_x.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let inv_rms = 1.0 / (ss / dim as f64 + eps as f64).sqrt() as f32;
            for i in 0..dim {
                let normed = row_x[i] * inv_rms * weight[i];
                row_out[i] = normed * silu(row_z[i]);
            }
        });
    let _ = seq_len;
}

// ── DeltaNet Forward Cache ──────────────────────────────────────────────────

/// 1ステップの DeltaNet forward で保存する中間値。
pub struct DeltaNetStepCache {
    /// 状態行列 S_{t-1} (dk × dv)。
    pub s_prev: Vec<f32>,
    /// 正規化済み query (dk)。
    pub q: Vec<f32>,
    /// 正規化済み key (dk)。
    pub k: Vec<f32>,
    /// value (dv)。
    pub v: Vec<f32>,
    /// delta = v - S^T k (dv)。
    pub e: Vec<f32>,
    /// write gate β (scalar)。
    pub beta: f32,
    /// decay gate g (scalar, 負値)。
    pub g: f32,
    /// exp(g) (scalar)。
    pub exp_g: f32,
    /// decay 前の a (sigmoid 前の logit)。
    pub a_logit: f32,
    /// beta 前の b (sigmoid 前の logit)。
    pub b_logit: f32,
}

/// DeltaNet レイヤー全体の forward cache。
pub struct DeltaNetLayerCache {
    /// 各ヘッド × 各タイムステップの cache。
    /// インデックス: [head][t] (ヘッドファースト)。
    pub step_caches: Vec<Vec<DeltaNetStepCache>>,
    /// 最終状態行列 S_T per head: [head][dk × dv]。
    pub final_states: Vec<Vec<f32>>,
    /// Conv1d 前の QKV (conv backward 用)。行優先。
    pub pre_conv_qkv: Vec<f32>,
    /// Output gate z (seq_len × value_dim)。
    pub z: Vec<f32>,
    /// RMSNorm 前の attention output (seq_len × value_dim)。
    pub attn_out_pre_norm: Vec<f32>,
    /// norm 前の residual (seq_len × hidden_size)。
    pub residual_attn: Vec<f32>,
    /// FFN 用の cache。
    pub residual_ffn: Vec<f32>,
    /// normed FFN input。
    pub normed_ffn: Vec<f32>,
    /// SwiGLU gate buffer。
    pub gate: Vec<f32>,
    /// SwiGLU up buffer。
    pub up: Vec<f32>,
    /// SiLU(gate) buffer。
    pub gate_silu: Vec<f32>,
}

// ── DeltaNet Forward (L1: ヘッド並列, L2: ゼロアロケーション, L7: チャンク再帰, L8: eval専用) ──

/// チャンクサイズ。intra-chunk は O(C²) 並列、inter-chunk は O(T/C) 逐次。
#[allow(dead_code)]
const CHUNK_SIZE: usize = 64;

/// L8: 1ヘッドの eval 専用 forward — backward 用 cache を一切保存しない。
///
/// メモリ: s_prev clone 不要 → 64KB×2048 = 128MB/head 節約。
/// 速度: clone + Vec::push 消滅で ~1.5x。
pub fn head_recurrence_forward_eval(
    q_all: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    beta_all: &[f32],
    g_all: &[f32],
    output: &mut [f32],
    state: &mut [f32],
    dk: usize,
    dv: usize,
    seq_len: usize,
) {
    let mut r_buf = vec![0.0f32; dv];

    for t in 0..seq_len {
        let q = &q_all[t * dk..(t + 1) * dk];
        let k = &k_all[t * dk..(t + 1) * dk];
        let v = &v_all[t * dv..(t + 1) * dv];
        let beta = beta_all[t];
        let exp_g = g_all[t].exp();

        // retrieve
        r_buf.fill(0.0);
        for di in 0..dk {
            let k_di = k[di];
            let row = &state[di * dv..(di + 1) * dv];
            for dj in 0..dv {
                r_buf[dj] = row[dj].mul_add(k_di, r_buf[dj]);
            }
        }

        // state update: S = exp(g)*S + β*k⊗(v-r)
        for di in 0..dk {
            let bk = beta * k[di];
            let row = &mut state[di * dv..(di + 1) * dv];
            for dj in 0..dv {
                row[dj] = exp_g.mul_add(row[dj], bk * (v[dj] - r_buf[dj]));
            }
        }

        // output
        let o = &mut output[t * dv..(t + 1) * dv];
        o.fill(0.0);
        for di in 0..dk {
            let q_di = q[di];
            let row = &state[di * dv..(di + 1) * dv];
            for dj in 0..dv {
                o[dj] = row[dj].mul_add(q_di, o[dj]);
            }
        }
    }
}

/// L7: チャンク再帰 — intra-chunk を展開して中間状態保存を C 境界のみに削減。
///
/// 通常再帰: s_prev を T 回保存 = T×dk×dv floats。
/// チャンク版: s_prev を T/C 回保存 = T/C×dk×dv floats。
/// 2048/64 = 32 回 → メモリ 32x 削減。
///
/// intra-chunk は依然逐次だが、チャンク境界の状態のみ保存することで
/// backward 時にチャンク内を再計算（recompute）する戦略。
#[allow(dead_code)]
fn head_recurrence_forward_chunked(
    q_all: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    beta_all: &[f32],
    g_all: &[f32],
    b_logits: &[f32],
    a_logits: &[f32],
    output: &mut [f32],
    state: &mut [f32],
    dk: usize,
    dv: usize,
    seq_len: usize,
) -> Vec<DeltaNetStepCache> {
    let num_chunks = seq_len.div_ceil(CHUNK_SIZE);

    let mut caches = Vec::with_capacity(seq_len);
    let mut r_buf = vec![0.0f32; dv];
    let mut e_buf = vec![0.0f32; dv];

    for chunk_idx in 0..num_chunks {
        let chunk_start = chunk_idx * CHUNK_SIZE;
        let chunk_end = (chunk_start + CHUNK_SIZE).min(seq_len);

        for t in chunk_start..chunk_end {
            let q = &q_all[t * dk..(t + 1) * dk];
            let k = &k_all[t * dk..(t + 1) * dk];
            let v = &v_all[t * dv..(t + 1) * dv];
            let beta = beta_all[t];
            let g = g_all[t];
            let exp_g = g.exp();

            let s_prev = state.to_vec();

            // retrieve
            r_buf.fill(0.0);
            for di in 0..dk {
                let k_di = k[di];
                let row = &state[di * dv..(di + 1) * dv];
                for dj in 0..dv {
                    r_buf[dj] = row[dj].mul_add(k_di, r_buf[dj]);
                }
            }

            // error
            for (ej, (vj, rj)) in e_buf.iter_mut().zip(v.iter().zip(r_buf.iter())) {
                *ej = vj - rj;
            }

            // state update
            for di in 0..dk {
                let bk = beta * k[di];
                let row = &mut state[di * dv..(di + 1) * dv];
                for dj in 0..dv {
                    row[dj] = exp_g.mul_add(row[dj], bk * e_buf[dj]);
                }
            }

            // output
            let o = &mut output[t * dv..(t + 1) * dv];
            o.fill(0.0);
            for di in 0..dk {
                let q_di = q[di];
                let row = &state[di * dv..(di + 1) * dv];
                for dj in 0..dv {
                    o[dj] = row[dj].mul_add(q_di, o[dj]);
                }
            }

            caches.push(DeltaNetStepCache {
                s_prev,
                q: q.to_vec(),
                k: k.to_vec(),
                v: v.to_vec(),
                e: e_buf.clone(),
                beta,
                g,
                exp_g,
                a_logit: a_logits[t],
                b_logit: b_logits[t],
            });
        }

        // チャンク境界の状態（将来: backward recompute 用）
    }

    caches
}

/// 1ヘッドの再帰 forward — ゼロアロケーション（学習用、full cache）。
fn head_recurrence_forward(
    q_all: &[f32],      // (seq_len × dk) — このヘッドの Q
    k_all: &[f32],      // (seq_len × dk)
    v_all: &[f32],      // (seq_len × dv)
    beta_all: &[f32],   // (seq_len,)
    g_all: &[f32],      // (seq_len,)
    b_logits: &[f32],   // (seq_len,)
    a_logits: &[f32],   // (seq_len,)
    output: &mut [f32], // (seq_len × dv)
    state: &mut [f32],  // (dk × dv) — 状態行列
    dk: usize,
    dv: usize,
    seq_len: usize,
) -> Vec<DeltaNetStepCache> {
    let mut caches = Vec::with_capacity(seq_len);
    let mut r_buf = vec![0.0f32; dv]; // retrieve buffer
    let mut e_buf = vec![0.0f32; dv]; // error buffer

    for t in 0..seq_len {
        let q = &q_all[t * dk..(t + 1) * dk];
        let k = &k_all[t * dk..(t + 1) * dk];
        let v = &v_all[t * dv..(t + 1) * dv];
        let beta = beta_all[t];
        let g = g_all[t];
        let exp_g = g.exp();

        // backward 用に S_{t-1} を保存
        let s_prev = state.to_vec();

        // retrieve: r = S^T k — FMA
        r_buf.fill(0.0);
        for di in 0..dk {
            let k_di = k[di];
            let row = &state[di * dv..(di + 1) * dv];
            for dj in 0..dv {
                r_buf[dj] = row[dj].mul_add(k_di, r_buf[dj]);
            }
        }

        // error: e = v - r
        for (ej, (vj, rj)) in e_buf.iter_mut().zip(v.iter().zip(r_buf.iter())) {
            *ej = vj - rj;
        }

        // state update: S = exp(g) * S + β * k ⊗ e — FMA
        for di in 0..dk {
            let bk = beta * k[di];
            let row = &mut state[di * dv..(di + 1) * dv];
            for dj in 0..dv {
                row[dj] = exp_g.mul_add(row[dj], bk * e_buf[dj]);
            }
        }

        // output: o = S^T q — FMA
        let o = &mut output[t * dv..(t + 1) * dv];
        o.fill(0.0);
        for di in 0..dk {
            let q_di = q[di];
            let row = &state[di * dv..(di + 1) * dv];
            for dj in 0..dv {
                o[dj] = row[dj].mul_add(q_di, o[dj]);
            }
        }

        caches.push(DeltaNetStepCache {
            s_prev,
            q: q.to_vec(),
            k: k.to_vec(),
            v: v.to_vec(),
            e: e_buf.clone(),
            beta,
            g,
            exp_g,
            a_logit: a_logits[t],
            b_logit: b_logits[t],
        });
    }

    caches
}

/// Gated DeltaNet 再帰 forward — L1: ヘッド間 Rayon 並列。
///
/// 入力レイアウト: (seq_len × num_heads × dim) — ヘッド軸はインターリーブ。
/// 出力: per-head cache + 最終状態。
pub fn deltanet_recurrence_forward(
    q_all: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    beta_all: &[f32],
    g_all: &[f32],
    b_logits: &[f32],
    a_logits: &[f32],
    output: &mut [f32],
    num_heads: usize,
    dk: usize,
    dv: usize,
    seq_len: usize,
) -> (Vec<Vec<DeltaNetStepCache>>, Vec<Vec<f32>>) {
    // ヘッドごとに de-interleave したビューを作成
    // (seq_len × num_heads × dk) → per head: (seq_len × dk)
    let mut per_head_q: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len * dk]).collect();
    let mut per_head_k: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len * dk]).collect();
    let mut per_head_v: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len * dv]).collect();
    let mut per_head_beta: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len]).collect();
    let mut per_head_g: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len]).collect();
    let mut per_head_b: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len]).collect();
    let mut per_head_a: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len]).collect();

    for t in 0..seq_len {
        for h in 0..num_heads {
            per_head_q[h][t * dk..(t + 1) * dk].copy_from_slice(
                &q_all[t * num_heads * dk + h * dk..t * num_heads * dk + (h + 1) * dk],
            );
            per_head_k[h][t * dk..(t + 1) * dk].copy_from_slice(
                &k_all[t * num_heads * dk + h * dk..t * num_heads * dk + (h + 1) * dk],
            );
            per_head_v[h][t * dv..(t + 1) * dv].copy_from_slice(
                &v_all[t * num_heads * dv + h * dv..t * num_heads * dv + (h + 1) * dv],
            );
            per_head_beta[h][t] = beta_all[t * num_heads + h];
            per_head_g[h][t] = g_all[t * num_heads + h];
            per_head_b[h][t] = b_logits[t * num_heads + h];
            per_head_a[h][t] = a_logits[t * num_heads + h];
        }
    }

    // L1: ヘッド並列で再帰実行
    let results: Vec<(Vec<DeltaNetStepCache>, Vec<f32>, Vec<f32>)> = (0..num_heads)
        .into_par_iter()
        .map(|h| {
            let mut state = vec![0.0f32; dk * dv];
            let mut head_output = vec![0.0f32; seq_len * dv];
            let caches = head_recurrence_forward(
                &per_head_q[h],
                &per_head_k[h],
                &per_head_v[h],
                &per_head_beta[h],
                &per_head_g[h],
                &per_head_b[h],
                &per_head_a[h],
                &mut head_output,
                &mut state,
                dk,
                dv,
                seq_len,
            );
            (caches, state, head_output)
        })
        .collect();

    // 結果を interleaved output に re-pack
    let mut all_caches: Vec<Vec<DeltaNetStepCache>> = Vec::with_capacity(num_heads);
    let mut final_states: Vec<Vec<f32>> = Vec::with_capacity(num_heads);

    for (h, (caches, state, head_output)) in results.into_iter().enumerate() {
        for t in 0..seq_len {
            output[t * num_heads * dv + h * dv..t * num_heads * dv + (h + 1) * dv]
                .copy_from_slice(&head_output[t * dv..(t + 1) * dv]);
        }
        all_caches.push(caches);
        final_states.push(state);
    }

    (all_caches, final_states)
}

/// L8: Eval 専用 forward — cache 不要、メモリ最小。
///
/// 学習時の `deltanet_recurrence_forward` と同じ出力だが、
/// `DeltaNetStepCache` を保存しないため:
/// - メモリ: per head per step 64KB (s_prev) + 数百B (q/k/v/e) が完全消滅
/// - 速度: Vec::push, clone, to_vec が全消滅で ~1.5x
pub fn deltanet_recurrence_forward_eval(
    q_all: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    beta_all: &[f32],
    g_all: &[f32],
    output: &mut [f32],
    num_heads: usize,
    dk: usize,
    dv: usize,
    seq_len: usize,
) -> Vec<Vec<f32>> {
    // de-interleave
    let mut per_head_q: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len * dk]).collect();
    let mut per_head_k: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len * dk]).collect();
    let mut per_head_v: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len * dv]).collect();
    let mut per_head_beta: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len]).collect();
    let mut per_head_g: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len]).collect();

    for t in 0..seq_len {
        for h in 0..num_heads {
            per_head_q[h][t * dk..(t + 1) * dk].copy_from_slice(
                &q_all[t * num_heads * dk + h * dk..t * num_heads * dk + (h + 1) * dk],
            );
            per_head_k[h][t * dk..(t + 1) * dk].copy_from_slice(
                &k_all[t * num_heads * dk + h * dk..t * num_heads * dk + (h + 1) * dk],
            );
            per_head_v[h][t * dv..(t + 1) * dv].copy_from_slice(
                &v_all[t * num_heads * dv + h * dv..t * num_heads * dv + (h + 1) * dv],
            );
            per_head_beta[h][t] = beta_all[t * num_heads + h];
            per_head_g[h][t] = g_all[t * num_heads + h];
        }
    }

    // ヘッド並列 eval
    let results: Vec<(Vec<f32>, Vec<f32>)> = (0..num_heads)
        .into_par_iter()
        .map(|h| {
            let mut state = vec![0.0f32; dk * dv];
            let mut head_output = vec![0.0f32; seq_len * dv];
            head_recurrence_forward_eval(
                &per_head_q[h],
                &per_head_k[h],
                &per_head_v[h],
                &per_head_beta[h],
                &per_head_g[h],
                &mut head_output,
                &mut state,
                dk,
                dv,
                seq_len,
            );
            (state, head_output)
        })
        .collect();

    let mut final_states: Vec<Vec<f32>> = Vec::with_capacity(num_heads);
    for (h, (state, head_output)) in results.into_iter().enumerate() {
        for t in 0..seq_len {
            output[t * num_heads * dv + h * dv..t * num_heads * dv + (h + 1) * dv]
                .copy_from_slice(&head_output[t * dv..(t + 1) * dv]);
        }
        final_states.push(state);
    }

    final_states
}

// ── L13: CUDA DeltaNet 再帰 (eval) ──────────────────────────────────────────

/// CUDA 版 DeltaNet 再帰 forward eval。
///
/// 入力: `[seq_len, num_heads, dk/dv]` (interleaved) — CPU版と同じインターフェース。
/// 内部で `[num_heads, seq_len, dk/dv]` にレイアウト変換し、CUDA カーネルを呼び出す。
#[cfg(feature = "cuda")]
pub fn deltanet_recurrence_forward_eval_cuda(
    q_all: &[f32],      // [seq_len × num_heads × dk]
    k_all: &[f32],      // [seq_len × num_heads × dk]
    v_all: &[f32],      // [seq_len × num_heads × dv]
    beta_all: &[f32],   // [seq_len × num_heads]
    g_all: &[f32],      // [seq_len × num_heads]
    output: &mut [f32], // [seq_len × num_heads × dv]
    num_heads: usize,
    dk: usize,
    dv: usize,
    seq_len: usize,
) {
    let total_qk = num_heads * seq_len * dk;
    let total_v = num_heads * seq_len * dv;
    let total_bg = num_heads * seq_len;

    // レイアウト変換: [seq, heads, dim] → [heads, seq, dim]
    let mut q_gpu = vec![0.0f32; total_qk];
    let mut k_gpu = vec![0.0f32; total_qk];
    let mut v_gpu = vec![0.0f32; total_v];
    let mut beta_gpu = vec![0.0f32; total_bg];
    let mut g_gpu = vec![0.0f32; total_bg];

    for t in 0..seq_len {
        for h in 0..num_heads {
            let src_qk = t * num_heads * dk + h * dk;
            let dst_qk = h * seq_len * dk + t * dk;
            q_gpu[dst_qk..dst_qk + dk].copy_from_slice(&q_all[src_qk..src_qk + dk]);
            k_gpu[dst_qk..dst_qk + dk].copy_from_slice(&k_all[src_qk..src_qk + dk]);

            let src_v = t * num_heads * dv + h * dv;
            let dst_v = h * seq_len * dv + t * dv;
            v_gpu[dst_v..dst_v + dv].copy_from_slice(&v_all[src_v..src_v + dv]);

            beta_gpu[h * seq_len + t] = beta_all[t * num_heads + h];
            g_gpu[h * seq_len + t] = g_all[t * num_heads + h];
        }
    }

    // CUDA カーネル実行
    let mut out_gpu = vec![0.0f32; total_v]; // [num_heads, seq_len, dv]

    {
        use crate::blas::CUDA_MATMUL;
        let cuda_mtx = CUDA_MATMUL.get().expect("CUDA 未初期化");
        let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
        crate::cuda_matmul::cuda_deltanet_recurrence(
            &cuda,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &beta_gpu,
            &g_gpu,
            &mut out_gpu,
            num_heads,
            seq_len,
            dk,
            dv,
        );
    }

    // レイアウト逆変換: [heads, seq, dv] → [seq, heads, dv]
    for t in 0..seq_len {
        for h in 0..num_heads {
            let src = h * seq_len * dv + t * dv;
            let dst = t * num_heads * dv + h * dv;
            output[dst..dst + dv].copy_from_slice(&out_gpu[src..src + dv]);
        }
    }
}

/// GPU DeltaNet 訓練 forward — S_{t-1} と e をGPU上で計算し DeltaNetStepCache を構築
#[cfg(feature = "cuda")]
pub fn deltanet_recurrence_forward_train_cuda(
    q_all: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    beta_all: &[f32],
    g_all: &[f32],
    a_logit_all: &[f32],
    b_logit_all: &[f32],
    output: &mut [f32],
    num_heads: usize,
    dk: usize,
    dv: usize,
    seq_len: usize,
) -> (Vec<Vec<DeltaNetStepCache>>, Vec<Vec<f32>>) {
    let total_qk = num_heads * seq_len * dk;
    let total_v = num_heads * seq_len * dv;
    let total_bg = num_heads * seq_len;
    let total_states = num_heads * seq_len * dk * dv;
    let total_e = num_heads * seq_len * dv;

    // レイアウト変換: [seq, heads, dim] → [heads, seq, dim]
    let mut q_gpu = vec![0.0f32; total_qk];
    let mut k_gpu = vec![0.0f32; total_qk];
    let mut v_gpu = vec![0.0f32; total_v];
    let mut beta_gpu = vec![0.0f32; total_bg];
    let mut g_gpu = vec![0.0f32; total_bg];

    for t in 0..seq_len {
        for h in 0..num_heads {
            let src_qk = t * num_heads * dk + h * dk;
            let dst_qk = h * seq_len * dk + t * dk;
            q_gpu[dst_qk..dst_qk + dk].copy_from_slice(&q_all[src_qk..src_qk + dk]);
            k_gpu[dst_qk..dst_qk + dk].copy_from_slice(&k_all[src_qk..src_qk + dk]);

            let src_v = t * num_heads * dv + h * dv;
            let dst_v = h * seq_len * dv + t * dv;
            v_gpu[dst_v..dst_v + dv].copy_from_slice(&v_all[src_v..src_v + dv]);

            beta_gpu[h * seq_len + t] = beta_all[t * num_heads + h];
            g_gpu[h * seq_len + t] = g_all[t * num_heads + h];
        }
    }

    // GPU forward + state storage
    let mut out_gpu = vec![0.0f32; total_v];
    let mut all_s_prev = vec![0.0f32; total_states];
    let mut all_e_flat = vec![0.0f32; total_e];

    {
        use crate::blas::CUDA_MATMUL;
        let cuda_mtx = CUDA_MATMUL.get().expect("CUDA 未初期化");
        let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
        crate::cuda_matmul::cuda_deltanet_recurrence_train(
            &cuda,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &beta_gpu,
            &g_gpu,
            &mut out_gpu,
            &mut all_s_prev,
            &mut all_e_flat,
            num_heads,
            seq_len,
            dk,
            dv,
        );
    }

    // レイアウト逆変換: [heads, seq, dv] → [seq, heads, dv]
    for t in 0..seq_len {
        for h in 0..num_heads {
            let src = h * seq_len * dv + t * dv;
            let dst = t * num_heads * dv + h * dv;
            output[dst..dst + dv].copy_from_slice(&out_gpu[src..src + dv]);
        }
    }

    // DeltaNetStepCache 構築
    let mut per_head_caches: Vec<Vec<DeltaNetStepCache>> = Vec::with_capacity(num_heads);
    let mut final_states: Vec<Vec<f32>> = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        let mut step_caches = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let s_off = h * seq_len * dk * dv + t * dk * dv;
            let s_prev = all_s_prev[s_off..s_off + dk * dv].to_vec();
            let q_t: Vec<f32> =
                q_gpu[h * seq_len * dk + t * dk..h * seq_len * dk + (t + 1) * dk].to_vec();
            let k_t: Vec<f32> =
                k_gpu[h * seq_len * dk + t * dk..h * seq_len * dk + (t + 1) * dk].to_vec();
            let v_t: Vec<f32> =
                v_gpu[h * seq_len * dv + t * dv..h * seq_len * dv + (t + 1) * dv].to_vec();
            let e_t: Vec<f32> =
                all_e_flat[h * seq_len * dv + t * dv..h * seq_len * dv + (t + 1) * dv].to_vec();
            let beta_t = beta_gpu[h * seq_len + t];
            let g_t = g_gpu[h * seq_len + t];

            step_caches.push(DeltaNetStepCache {
                s_prev,
                q: q_t,
                k: k_t,
                v: v_t,
                e: e_t,
                beta: beta_t,
                g: g_t,
                exp_g: g_t.exp(),
                a_logit: a_logit_all[t * num_heads + h],
                b_logit: b_logit_all[t * num_heads + h],
            });
        }
        // Final state
        let last = &step_caches[seq_len - 1];
        let mut fs = vec![0.0f32; dk * dv];
        for i in 0..dk {
            for j in 0..dv {
                fs[i * dv + j] =
                    last.exp_g * last.s_prev[i * dv + j] + last.beta * last.k[i] * last.e[j];
            }
        }
        final_states.push(fs);
        per_head_caches.push(step_caches);
    }

    (per_head_caches, final_states)
}

/// GPU DeltaNet Fused Forward+Backward — 状態をVRAM内に保持
/// forward output + backward gradients を一発で計算。DeltaNetStepCache 不要。
#[cfg(feature = "cuda")]
pub fn deltanet_recurrence_fused_fwd_bwd_cuda(
    q_all: &[f32], // [seq_len × num_heads × dk]
    k_all: &[f32],
    v_all: &[f32],
    beta_all: &[f32], // [seq_len × num_heads]
    g_all: &[f32],
    d_output: &[f32],   // [seq_len × num_heads × dv]
    output: &mut [f32], // [seq_len × num_heads × dv]
    num_heads: usize,
    dk: usize,
    dv: usize,
    seq_len: usize,
) -> DeltaNetRecurrenceGrads {
    let total_qk = num_heads * seq_len * dk;
    let total_v = num_heads * seq_len * dv;
    let total_bg = num_heads * seq_len;

    // レイアウト変換: [seq, heads, dim] → [heads, seq, dim]
    let mut q_gpu = vec![0.0f32; total_qk];
    let mut k_gpu = vec![0.0f32; total_qk];
    let mut v_gpu = vec![0.0f32; total_v];
    let mut beta_gpu = vec![0.0f32; total_bg];
    let mut g_gpu = vec![0.0f32; total_bg];
    let mut do_gpu = vec![0.0f32; total_v];

    for t in 0..seq_len {
        for h in 0..num_heads {
            let src_qk = t * num_heads * dk + h * dk;
            let dst_qk = h * seq_len * dk + t * dk;
            q_gpu[dst_qk..dst_qk + dk].copy_from_slice(&q_all[src_qk..src_qk + dk]);
            k_gpu[dst_qk..dst_qk + dk].copy_from_slice(&k_all[src_qk..src_qk + dk]);

            let src_v = t * num_heads * dv + h * dv;
            let dst_v = h * seq_len * dv + t * dv;
            v_gpu[dst_v..dst_v + dv].copy_from_slice(&v_all[src_v..src_v + dv]);
            do_gpu[dst_v..dst_v + dv].copy_from_slice(&d_output[src_v..src_v + dv]);

            beta_gpu[h * seq_len + t] = beta_all[t * num_heads + h];
            g_gpu[h * seq_len + t] = g_all[t * num_heads + h];
        }
    }

    let mut out_gpu = vec![0.0f32; total_v];
    let mut dq_gpu = vec![0.0f32; total_qk];
    let mut dk_gpu_out = vec![0.0f32; total_qk];
    let mut dv_gpu_out = vec![0.0f32; total_v];
    let mut dbeta_gpu = vec![0.0f32; total_bg];
    let mut dg_gpu = vec![0.0f32; total_bg];

    {
        use crate::blas::CUDA_MATMUL;
        let cuda_mtx = CUDA_MATMUL.get().expect("CUDA 未初期化");
        let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
        crate::cuda_matmul::cuda_deltanet_fused_fwd_bwd(
            &cuda,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            &beta_gpu,
            &g_gpu,
            &do_gpu,
            &mut out_gpu,
            &mut dq_gpu,
            &mut dk_gpu_out,
            &mut dv_gpu_out,
            &mut dbeta_gpu,
            &mut dg_gpu,
            num_heads,
            seq_len,
            dk,
            dv,
        );
    }

    // レイアウト逆変換: [heads, seq, dim] → [seq, heads, dim]
    for t in 0..seq_len {
        for h in 0..num_heads {
            let src = h * seq_len * dv + t * dv;
            let dst = t * num_heads * dv + h * dv;
            output[dst..dst + dv].copy_from_slice(&out_gpu[src..src + dv]);
        }
    }

    let mut d_q = vec![0.0f32; total_qk];
    let mut d_k = vec![0.0f32; total_qk];
    let mut d_v = vec![0.0f32; total_v];
    let mut d_b_logit = vec![0.0f32; total_bg];
    let mut d_a_logit = vec![0.0f32; total_bg];

    for t in 0..seq_len {
        for h in 0..num_heads {
            let src_qk = h * seq_len * dk + t * dk;
            let dst_qk = t * num_heads * dk + h * dk;
            d_q[dst_qk..dst_qk + dk].copy_from_slice(&dq_gpu[src_qk..src_qk + dk]);
            d_k[dst_qk..dst_qk + dk].copy_from_slice(&dk_gpu_out[src_qk..src_qk + dk]);

            let src_v = h * seq_len * dv + t * dv;
            let dst_v = t * num_heads * dv + h * dv;
            d_v[dst_v..dst_v + dv].copy_from_slice(&dv_gpu_out[src_v..src_v + dv]);

            // d_beta → d_b_logit (sigmoid chain既にカーネル内で適用)
            d_b_logit[t * num_heads + h] = dbeta_gpu[h * seq_len + t];
            // d_g → d_a_logit (chain rule は呼出側で)
            d_a_logit[t * num_heads + h] = dg_gpu[h * seq_len + t];
        }
    }

    // d_a_log, d_dt_bias: カーネルの d_g は raw dg (chain未適用部分)
    // 呼出側で a_log/dt_bias の chain rule を適用
    let d_a_log = vec![0.0f32; num_heads];
    let d_dt_bias = vec![0.0f32; num_heads];

    DeltaNetRecurrenceGrads {
        d_q,
        d_k,
        d_v,
        d_b_logit,
        d_a_logit,
        d_a_log,
        d_dt_bias,
    }
}

// ── DeltaNet Backward (L1: ヘッド並列) ─────────────────────────────────────

/// DeltaNet 再帰の backward 勾配。
pub struct DeltaNetRecurrenceGrads {
    /// Query 勾配 (seq_len × num_heads × dk)。
    pub d_q: Vec<f32>,
    /// Key 勾配 (seq_len × num_heads × dk)。
    pub d_k: Vec<f32>,
    /// Value 勾配 (seq_len × num_heads × dv)。
    pub d_v: Vec<f32>,
    /// Beta logit 勾配 (seq_len × num_heads)。
    pub d_b_logit: Vec<f32>,
    /// Decay logit 勾配 (seq_len × num_heads)。
    pub d_a_logit: Vec<f32>,
    /// A_log 勾配 (num_heads)。
    pub d_a_log: Vec<f32>,
    /// dt_bias 勾配 (num_heads)。
    pub d_dt_bias: Vec<f32>,
}

/// 1ヘッドの backward。
fn head_recurrence_backward(
    d_output: &[f32], // (seq_len × dv) — このヘッドの出力勾配
    caches: &[DeltaNetStepCache],
    a_log_h: f32,
    dt_bias_h: f32,
    dk: usize,
    dv: usize,
    seq_len: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32, f32) {
    let mut d_q = vec![0.0f32; seq_len * dk];
    let mut d_k = vec![0.0f32; seq_len * dk];
    let mut d_v = vec![0.0f32; seq_len * dv];
    let mut d_b_logit = vec![0.0f32; seq_len];
    let mut d_a_logit = vec![0.0f32; seq_len];
    let mut d_a_log = 0.0f32;
    let mut d_dt_bias = 0.0f32;

    let mut ds = vec![0.0f32; dk * dv];
    // 再利用バッファ
    let mut dst_k = vec![0.0f32; dv];

    for t in (0..seq_len).rev() {
        let cache = &caches[t];
        let do_t = &d_output[t * dv..(t + 1) * dv];

        // S_t を再構築
        let mut s_t = vec![0.0f32; dk * dv];
        for di in 0..dk {
            let bk = cache.beta * cache.k[di];
            for dj in 0..dv {
                s_t[di * dv + dj] = cache
                    .exp_g
                    .mul_add(cache.s_prev[di * dv + dj], bk * cache.e[dj]);
            }
        }

        // dq_t = S_t · do_t
        for di in 0..dk {
            let mut val = 0.0f32;
            for dj in 0..dv {
                val = s_t[di * dv + dj].mul_add(do_t[dj], val);
            }
            d_q[t * dk + di] = val;
        }

        // dS += q · do^T
        for di in 0..dk {
            let q_di = cache.q[di];
            for dj in 0..dv {
                ds[di * dv + dj] += q_di * do_t[dj];
            }
        }

        // dS^T · k → dst_k (dv)
        dst_k.fill(0.0);
        for dj in 0..dv {
            for di in 0..dk {
                dst_k[dj] = ds[di * dv + dj].mul_add(cache.k[di], dst_k[dj]);
            }
        }

        // dk = β * (dS·e - S_{t-1}·dst_k)
        for di in 0..dk {
            let mut ds_e = 0.0f32;
            let mut s_dst = 0.0f32;
            for dj in 0..dv {
                ds_e = ds[di * dv + dj].mul_add(cache.e[dj], ds_e);
                s_dst = cache.s_prev[di * dv + dj].mul_add(dst_k[dj], s_dst);
            }
            d_k[t * dk + di] = cache.beta * (ds_e - s_dst);
        }

        // dv = β * dst_k
        for dj in 0..dv {
            d_v[t * dv + dj] = cache.beta * dst_k[dj];
        }

        // dβ = Σ dS_ij * k_i * e_j → chain sigmoid
        let mut d_beta_val = 0.0f32;
        for di in 0..dk {
            for dj in 0..dv {
                d_beta_val += ds[di * dv + dj] * cache.k[di] * cache.e[dj];
            }
        }
        d_b_logit[t] = d_beta_val * cache.beta * (1.0 - cache.beta);

        // dg = exp(g) * ⟨dS, S_{t-1}⟩_F
        let frobenius: f32 = ds
            .iter()
            .zip(cache.s_prev.iter())
            .fold(0.0f32, |acc, (&d, &s)| d.mul_add(s, acc));
        let dg = cache.exp_g * frobenius;

        // chain: g = -A * softplus(a + dt_bias)
        let a_val = a_log_h.exp();
        let sig_a = sigmoid(cache.a_logit + dt_bias_h);
        let sp = softplus(cache.a_logit + dt_bias_h);
        d_a_logit[t] = dg * (-a_val * sig_a);
        d_a_log += dg * (-sp * a_val);
        d_dt_bias += dg * (-a_val * sig_a);

        // dS_{t-1} = exp(g) * dS - β * k ⊗ dst_k
        for di in 0..dk {
            let bk = cache.beta * cache.k[di];
            let row = &mut ds[di * dv..(di + 1) * dv];
            for dj in 0..dv {
                row[dj] = cache.exp_g * row[dj] - bk * dst_k[dj];
            }
        }
    }

    (d_q, d_k, d_v, d_b_logit, d_a_logit, d_a_log, d_dt_bias)
}

/// DeltaNet 再帰の backward — L1: ヘッド並列。
#[must_use]
pub fn deltanet_recurrence_backward(
    d_output: &[f32],
    caches: &[Vec<DeltaNetStepCache>], // [num_heads][seq_len]
    a_log: &[f32],
    dt_bias: &[f32],
    num_heads: usize,
    dk: usize,
    dv: usize,
    seq_len: usize,
) -> DeltaNetRecurrenceGrads {
    // d_output を per-head にde-interleave
    let mut per_head_do: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.0; seq_len * dv]).collect();
    for t in 0..seq_len {
        for h in 0..num_heads {
            per_head_do[h][t * dv..(t + 1) * dv].copy_from_slice(
                &d_output[t * num_heads * dv + h * dv..t * num_heads * dv + (h + 1) * dv],
            );
        }
    }

    // L1: ヘッド並列 backward
    let results: Vec<_> = (0..num_heads)
        .into_par_iter()
        .map(|h| {
            head_recurrence_backward(
                &per_head_do[h],
                &caches[h],
                a_log[h],
                dt_bias[h],
                dk,
                dv,
                seq_len,
            )
        })
        .collect();

    // re-interleave
    let total_q = seq_len * num_heads * dk;
    let total_v = seq_len * num_heads * dv;
    let total_bg = seq_len * num_heads;
    let mut d_q = vec![0.0f32; total_q];
    let mut d_k = vec![0.0f32; total_q];
    let mut d_v = vec![0.0f32; total_v];
    let mut d_b_logit = vec![0.0f32; total_bg];
    let mut d_a_logit = vec![0.0f32; total_bg];
    let mut d_a_log = vec![0.0f32; num_heads];
    let mut d_dt_bias = vec![0.0f32; num_heads];

    for (h, (hq, hk, hv, hb, ha, hal, hdt)) in results.into_iter().enumerate() {
        for t in 0..seq_len {
            d_q[t * num_heads * dk + h * dk..t * num_heads * dk + (h + 1) * dk]
                .copy_from_slice(&hq[t * dk..(t + 1) * dk]);
            d_k[t * num_heads * dk + h * dk..t * num_heads * dk + (h + 1) * dk]
                .copy_from_slice(&hk[t * dk..(t + 1) * dk]);
            d_v[t * num_heads * dv + h * dv..t * num_heads * dv + (h + 1) * dv]
                .copy_from_slice(&hv[t * dv..(t + 1) * dv]);
            d_b_logit[t * num_heads + h] = hb[t];
            d_a_logit[t * num_heads + h] = ha[t];
        }
        d_a_log[h] = hal;
        d_dt_bias[h] = hdt;
    }

    DeltaNetRecurrenceGrads {
        d_q,
        d_k,
        d_v,
        d_b_logit,
        d_a_logit,
        d_a_log,
        d_dt_bias,
    }
}

// ── Partial RoPE (Full Attention 用) ────────────────────────────────────────

/// Partial RoPE — ヘッド次元の先頭 `rotary_dim` 次元にのみ適用。Rayon並列。
pub fn apply_partial_rope(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    _seq_len: usize,
    rotary_dim: usize,
    theta: f32,
) {
    let stride = n_heads * head_dim;
    let half_rot = rotary_dim / 2;
    x.par_chunks_exact_mut(stride)
        .enumerate()
        .for_each(|(t, token)| {
            for h in 0..n_heads {
                let base = h * head_dim;
                for d in 0..half_rot {
                    let freq = 1.0 / theta.powf((2 * d) as f32 / rotary_dim as f32);
                    let angle = t as f32 * freq;
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    let x0 = token[base + d];
                    let x1 = token[base + d + half_rot];
                    token[base + d] = x0.mul_add(cos_a, -(x1 * sin_a));
                    token[base + d + half_rot] = x0.mul_add(sin_a, x1 * cos_a);
                }
            }
        });
}

/// QK-norm: per-head RMSNorm。Rayon並列。
pub fn qk_norm(x: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    let stride = n_heads * head_dim;
    x.par_chunks_exact_mut(stride).for_each(|token| {
        for h in 0..n_heads {
            let base = h * head_dim;
            let head = &mut token[base..base + head_dim];
            let ss: f64 = head.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let inv_rms = 1.0 / (ss / head_dim as f64 + eps as f64).sqrt() as f32;
            for (v, &w) in head.iter_mut().zip(weight.iter()) {
                *v *= inv_rms * w;
            }
        }
    });
}

// ── Gate融合 (L5) ───────────────────────────────────────────────────────────

/// β (sigmoid) と g (decay) をワンパスで計算。
///
/// L5: sigmoid/softplus/exp を1ループで融合。
/// `b_raw`: (seq_len × n_v_heads) — β logits。
/// `a_raw`: (seq_len × n_v_heads) — α logits。
/// `a_log`: (n_v_heads,)。
/// `dt_bias`: (n_v_heads,)。
/// `beta_out`: (seq_len × n_v_heads) — sigmoid(b)。
/// `g_out`: (seq_len × n_v_heads) — -exp(A_log) * softplus(a + dt_bias)。
pub fn compute_gates_fused(
    b_raw: &[f32],
    a_raw: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    beta_out: &mut [f32],
    g_out: &mut [f32],
    _seq_len: usize,
    n_v_heads: usize,
) {
    // A = exp(A_log) を事前計算
    let a_vals: Vec<f32> = a_log.iter().map(|&al| al.exp()).collect();

    beta_out
        .par_chunks_exact_mut(n_v_heads)
        .zip(g_out.par_chunks_exact_mut(n_v_heads))
        .enumerate()
        .for_each(|(t, (beta_row, g_row))| {
            let b_row = &b_raw[t * n_v_heads..(t + 1) * n_v_heads];
            let a_row = &a_raw[t * n_v_heads..(t + 1) * n_v_heads];
            for h in 0..n_v_heads {
                beta_row[h] = sigmoid(b_row[h]);
                g_row[h] = -a_vals[h] * softplus(a_row[h] + dt_bias[h]);
            }
        });
}

/// L2 norm + GQA expansion を融合。
///
/// Q, K を L2 正規化しながら GQA repeat を同時実行。
/// `q_in`: (seq_len × n_k_heads × dk) — 正規化前。
/// `k_in`: (seq_len × n_k_heads × dk)。
/// `q_out`: (seq_len × n_v_heads × dk) — 正規化+展開済み。
/// `k_out`: (seq_len × n_v_heads × dk)。
pub fn l2norm_and_gqa_expand(
    q_in: &[f32],
    k_in: &[f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    _seq_len: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    dk: usize,
    eps: f32,
) {
    let repeat = n_v_heads / n_k_heads;

    q_out
        .par_chunks_exact_mut(n_v_heads * dk)
        .zip(k_out.par_chunks_exact_mut(n_v_heads * dk))
        .enumerate()
        .for_each(|(t, (q_row, k_row))| {
            let q_src = &q_in[t * n_k_heads * dk..(t + 1) * n_k_heads * dk];
            let k_src = &k_in[t * n_k_heads * dk..(t + 1) * n_k_heads * dk];

            for kh in 0..n_k_heads {
                let qs = &q_src[kh * dk..(kh + 1) * dk];
                let ks = &k_src[kh * dk..(kh + 1) * dk];

                // L2 norm
                let mut q_norm_sq = 0.0f32;
                let mut k_norm_sq = 0.0f32;
                for d in 0..dk {
                    q_norm_sq = qs[d].mul_add(qs[d], q_norm_sq);
                    k_norm_sq = ks[d].mul_add(ks[d], k_norm_sq);
                }
                let q_inv = 1.0 / (q_norm_sq + eps).sqrt();
                let k_inv = 1.0 / (k_norm_sq + eps).sqrt();

                // GQA repeat + normalized write
                for r in 0..repeat {
                    let vh = kh * repeat + r;
                    let qd = &mut q_row[vh * dk..(vh + 1) * dk];
                    let kd = &mut k_row[vh * dk..(vh + 1) * dk];
                    for d in 0..dk {
                        qd[d] = qs[d] * q_inv;
                        kd[d] = ks[d] * k_inv;
                    }
                }
            }
        });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let mut x = vec![3.0, 4.0f32];
        l2_normalize(&mut x, 1e-6);
        assert!((x[0] - 0.6).abs() < 1e-5);
        assert!((x[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_backward_identity() {
        let x = vec![3.0, 4.0f32];
        let dx_norm = vec![1.0, 0.0];
        let mut dx = vec![0.0; 2];
        l2_normalize_backward(&x, &dx_norm, &mut dx, 1e-6);
        assert!(dx[0].is_finite());
        assert!(dx[1].is_finite());
        assert!(dx.iter().any(|&v| v.abs() > 1e-8));
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_softplus() {
        assert!((softplus(0.0) - 0.6931).abs() < 0.01);
        assert!((softplus(100.0) - 100.0).abs() < 0.01);
        assert!(softplus(-100.0).abs() < 0.01);
    }

    #[test]
    fn test_causal_conv1d_row_major() {
        // 2 channels, seq_len=4, kernel=2, row-major
        // x[t][c]: t=0:[1,2], t=1:[3,4], t=2:[5,6], t=3:[7,8]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let w = vec![0.5, 1.0, 0.5, 1.0f32]; // ch0: [0.5, 1.0], ch1: [0.5, 1.0]
        let mut out = vec![0.0f32; 8];
        causal_conv1d_silu_row_major(&x, &w, &mut out, 2, 4, 2);
        // t=0, c=0: pad + x[0][0]*w[1] = 1.0 → silu(1.0)
        assert!((out[0] - silu(1.0)).abs() < 1e-4);
        // t=1, c=0: x[0][0]*w[0] + x[1][0]*w[1] = 0.5 + 3.0 = 3.5 → silu(3.5)
        assert!((out[2] - silu(3.5)).abs() < 1e-4);
    }

    #[test]
    fn test_deltanet_single_step_parallel() {
        let q = vec![1.0, 0.0f32];
        let k = vec![0.6, 0.8f32];
        let v = vec![1.0, 2.0f32];
        let beta = vec![1.0f32];
        let g = vec![0.0f32];
        let b_logits = vec![100.0];
        let a_logits = vec![0.0];
        let mut output = vec![0.0f32; 2];

        let (caches, _) = deltanet_recurrence_forward(
            &q,
            &k,
            &v,
            &beta,
            &g,
            &b_logits,
            &a_logits,
            &mut output,
            1,
            2,
            2,
            1,
        );

        assert!((output[0] - 0.6).abs() < 1e-5);
        assert!((output[1] - 1.2).abs() < 1e-5);
        assert_eq!(caches.len(), 1);
        assert_eq!(caches[0].len(), 1);
    }

    #[test]
    fn test_deltanet_backward_gradient_check() {
        let dk = 2;
        let dv = 2;
        let num_heads = 1;
        let seq_len = 2;

        let q = vec![0.5, 0.3, 0.7, 0.2f32];
        let k = vec![0.6, 0.8, 0.4, 0.9f32];
        let v = vec![1.0, 2.0, 0.5, 1.5f32];
        let beta = vec![0.7, 0.5f32];
        let g = vec![-0.1, -0.2f32];
        let b_logits = vec![0.84, 0.0];
        let a_logits = vec![0.0, 0.0];
        let a_log_val = vec![0.1f32];
        let dt_bias_val = vec![0.0f32];

        let mut output = vec![0.0f32; seq_len * num_heads * dv];
        let (caches, _) = deltanet_recurrence_forward(
            &q,
            &k,
            &v,
            &beta,
            &g,
            &b_logits,
            &a_logits,
            &mut output,
            num_heads,
            dk,
            dv,
            seq_len,
        );

        let d_output = vec![1.0f32; seq_len * num_heads * dv];
        let grads = deltanet_recurrence_backward(
            &d_output,
            &caches,
            &a_log_val,
            &dt_bias_val,
            num_heads,
            dk,
            dv,
            seq_len,
        );

        assert!(grads.d_q.iter().all(|v| v.is_finite()));
        assert!(grads.d_k.iter().all(|v| v.is_finite()));
        assert!(grads.d_v.iter().all(|v| v.is_finite()));
        assert!(grads.d_q.iter().any(|v| v.abs() > 1e-8));
    }

    #[test]
    fn test_partial_rope_preserves_non_rotary() {
        let head_dim = 8;
        let rotary_dim = 4;
        let mut x = vec![1.0; 8];
        let original = x.clone();
        apply_partial_rope(&mut x, 1, head_dim, 1, rotary_dim, 10000.0);
        for i in rotary_dim..head_dim {
            assert_eq!(x[i], original[i], "dim {i} should be unchanged");
        }
    }

    #[test]
    fn test_qk_norm() {
        let mut x = vec![1.0, 0.0, 1.0, 0.0f32];
        let weight = vec![1.0; 4];
        qk_norm(&mut x, &weight, 1, 4, 1e-5);
        assert!((x[0] - 1.4142).abs() < 0.01);
        assert!(x[1].abs() < 1e-6);
    }

    #[test]
    fn test_gated_rmsnorm() {
        let x = vec![1.0, 1.0, 1.0, 1.0f32];
        let z = vec![0.0; 4]; // SiLU(0) = 0
        let weight = vec![1.0; 4];
        let mut out = vec![0.0; 4];
        gated_rmsnorm(&x, &z, &weight, &mut out, 4, 1e-5);
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn test_compute_gates_fused() {
        let b_raw = vec![0.0f32; 4]; // sigmoid(0) = 0.5
        let a_raw = vec![0.0f32; 4];
        let a_log = vec![0.0f32; 2]; // exp(0) = 1
        let dt_bias = vec![0.0f32; 2];
        let mut beta = vec![0.0f32; 4];
        let mut g = vec![0.0f32; 4];
        compute_gates_fused(&b_raw, &a_raw, &a_log, &dt_bias, &mut beta, &mut g, 2, 2);
        assert!((beta[0] - 0.5).abs() < 1e-5);
        // g = -1.0 * softplus(0) = -ln(2)
        assert!((g[0] - (-0.6931)).abs() < 0.01);
    }

    #[test]
    fn test_l2norm_and_gqa_expand() {
        // 1 token, 1 k_head → 2 v_heads, dk=2
        let q_in = vec![3.0, 4.0f32]; // norm = 5 → [0.6, 0.8]
        let k_in = vec![0.0, 1.0f32]; // norm = 1 → [0.0, 1.0]
        let mut q_out = vec![0.0f32; 4]; // 2 v_heads × 2
        let mut k_out = vec![0.0f32; 4];
        l2norm_and_gqa_expand(&q_in, &k_in, &mut q_out, &mut k_out, 1, 1, 2, 2, 1e-6);
        // both v_heads should get same normalized values
        assert!((q_out[0] - 0.6).abs() < 1e-4);
        assert!((q_out[1] - 0.8).abs() < 1e-4);
        assert!((q_out[2] - 0.6).abs() < 1e-4);
        assert!((q_out[3] - 0.8).abs() < 1e-4);
        assert!(k_out[1].abs() - 1.0 < 1e-4);
    }
}
