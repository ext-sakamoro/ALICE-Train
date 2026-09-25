//! `llama_backward::layer_backward` の数値微分 oracle。
//!
//! # なぜ要るか
//!
//! CPU path も CUDA path も **backward の独立した検証を持っていなかった**。
//! 2026-09-25 に両 path を突合したら勾配が符号レベルで食い違い、しかも
//! 「どちらが正しいか」を決める根拠がどこにも無かった (CPU path で学習が
//! 進んで見えたのは「それらしく動いた」だけで、正しさの証跡ではない)。
//!
//! # oracle の作り方
//!
//! スカラー損失 `L = Σ M ⊙ layer_forward(x)` を使う (M は固定のランダム行列)。
//! このとき `dL/d(出力) = M` なので、`layer_backward(M, …)` の戻りが
//! 解析勾配になる。これを **中心差分** と突き合わせる。
//! 期待値は微分の定義そのものから来ており、実装の出力を見て書いていない。
//!
//! f32 の桁で中心差分を取るので、h と許容は実測で決めた値 (下記 const)。

use alice_train::llama::{LlamaConfig, LlamaLayerWeights};
use alice_train::llama_backward::layer_backward;
use alice_train::llama_forward::layer_forward;

/// 中心差分の刻み。f32 の丸めと打ち切り誤差の折り合いで実測した値。
const H: f32 = 2e-3;
/// 相対許容 (f32 中心差分の実力)。
const RTOL: f64 = 3e-2;
/// 絶対許容 (勾配がゼロ近傍の要素用)。
const ATOL: f64 = 2e-3;

/// 決定論的な擬似乱数 (xorshift64*)。
fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
            (bits as f32 / 16_777_216.0).mul_add(0.4, -0.2)
        })
        .collect()
}

/// 小さい構成 (GQA / SwiGLU / RoPE / 2 段 RMSNorm を全部通る最小形)。
fn tiny_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 16,
        hidden_dim: 32,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 2,
        num_layers: 1,
        max_seq_len: 8,
        head_dim: 8,
        rope_theta: 10_000.0,
        norm_eps: 1e-6,
        attention_bias: false,
    }
}

fn tiny_weights(config: &LlamaConfig) -> LlamaLayerWeights {
    let h = config.hidden_dim;
    let kv = config.num_kv_heads * config.head_dim;
    let q = config.num_heads * config.head_dim;
    let i = config.intermediate_dim;
    LlamaLayerWeights {
        // norm weight は 1 近傍 (RMSNorm の実運用に近い値)
        attn_norm: fill(11, h).iter().map(|v| 1.0 + v).collect(),
        q_proj: fill(12, q * h),
        k_proj: fill(13, kv * h),
        v_proj: fill(14, kv * h),
        o_proj: fill(15, h * q),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        ffn_norm: fill(16, h).iter().map(|v| 1.0 + v).collect(),
        gate_proj: fill(17, i * h),
        up_proj: fill(18, i * h),
        down_proj: fill(19, h * i),
    }
}

/// `L = Σ M ⊙ layer_forward(x)` を f64 で足して返す。
fn loss(input: &[f32], w: &LlamaLayerWeights, m: &[f32], config: &LlamaConfig, seq: usize) -> f64 {
    let mut h = input.to_vec();
    let _cache = layer_forward(&mut h, w, config, seq);
    h.iter()
        .zip(m.iter())
        .map(|(o, mm)| f64::from(*o) * f64::from(*mm))
        .sum()
}

/// 解析勾配と中心差分を突合する。`pick` は検査する index。刻みは [`H`]。
fn check(
    label: &str,
    analytic: &[f32],
    pick: &[usize],
    perturb: impl FnMut(usize, f32) -> (f64, f64),
) {
    check_with_step(label, analytic, pick, H, perturb);
}

/// [`check`] の刻み可変版。実 shape では桁落ち対策で H を大きく取る。
///
/// 中心差分の分母は **実際に動かした量** でなければならない (固定 H で割ると
/// 刻みを変えた瞬間に期待値が定数倍ずれる)。
fn check_with_step(
    label: &str,
    analytic: &[f32],
    pick: &[usize],
    step: f32,
    mut perturb: impl FnMut(usize, f32) -> (f64, f64),
) {
    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    let mut worst_pair = (0.0f64, 0.0f64);
    for &idx in pick {
        let (plus, minus) = perturb(idx, step);
        let num = (plus - minus) / f64::from(2.0 * step);
        let ana = f64::from(analytic[idx]);
        let err = (num - ana).abs();
        let allow = ATOL + RTOL * ana.abs().max(num.abs());
        if err / allow > worst {
            worst = err / allow;
            worst_at = idx;
            worst_pair = (num, ana);
        }
    }
    assert!(
        worst <= 1.0,
        "{label}: 中心差分と解析勾配が合わない idx={worst_at} 数値={:.6e} 解析={:.6e} (許容比 {worst:.2})",
        worst_pair.0,
        worst_pair.1
    );
    println!(
        "  ok {label:<12} 最悪許容比 {worst:.3} ({} 点検査)",
        pick.len()
    );
}

/// 検査 index を決定論的に散らす。
fn picks(n: usize, count: usize, seed: usize) -> Vec<usize> {
    (0..count)
        .map(|i| (i * 7919 + seed * 104_729) % n)
        .collect()
}

#[test]
fn layer_backward_matches_central_difference() {
    let config = tiny_config();
    let w = tiny_weights(&config);
    let seq = 5usize;
    let input = fill(21, seq * config.hidden_dim);
    let m = fill(22, seq * config.hidden_dim);

    // 解析勾配: dL/d(出力) = M なので d_output に M を入れる
    let mut h = input.clone();
    let cache = layer_forward(&mut h, &w, &config, seq);
    let (d_input, grads) = layer_backward(&m, &cache, &w, &config, seq);

    println!(
        "layer_backward 数値微分 oracle (hidden {} seq {seq})",
        config.hidden_dim
    );

    // ── 入力勾配 ──
    check(
        "d_input",
        &d_input,
        &picks(input.len(), 12, 1),
        |idx, h_step| {
            let mut p = input.clone();
            p[idx] += h_step;
            let plus = loss(&p, &w, &m, &config, seq);
            p[idx] -= 2.0 * h_step;
            let minus = loss(&p, &w, &m, &config, seq);
            (plus, minus)
        },
    );

    // ── 重み勾配 ──
    macro_rules! check_w {
        ($label:literal, $field:ident, $grad:ident, $seed:expr) => {{
            let n = w.$field.len();
            let g = &grads.$grad;
            check($label, g, &picks(n, 10, $seed), |idx, h_step| {
                let mut wp = w.clone();
                wp.$field[idx] += h_step;
                let plus = loss(&input, &wp, &m, &config, seq);
                wp.$field[idx] -= 2.0 * h_step;
                let minus = loss(&input, &wp, &m, &config, seq);
                (plus, minus)
            });
        }};
    }
    check_w!("d_q_proj", q_proj, d_q_proj, 2);
    check_w!("d_k_proj", k_proj, d_k_proj, 3);
    check_w!("d_v_proj", v_proj, d_v_proj, 4);
    check_w!("d_o_proj", o_proj, d_o_proj, 5);
    check_w!("d_gate_proj", gate_proj, d_gate_proj, 6);
    check_w!("d_up_proj", up_proj, d_up_proj, 7);
    check_w!("d_down_proj", down_proj, d_down_proj, 8);
    check_w!("d_attn_norm", attn_norm, d_attn_norm, 9);
    check_w!("d_ffn_norm", ffn_norm, d_ffn_norm, 10);
}

// ── 実 shape 版 ───────────────────────────────────────────────────────────────
//
// 小 shape (hidden 32 / seq 5) で CPU / CUDA 双方の oracle が green なのに、実
// model (MiniCPM5-2B / seq 466) の `--verify-cuda-parity` では attention 側勾配
// (`d_input` / `d_q/k/v/o_proj` / `d_attn_norm`) だけが食い違い FFN 側は一致する。
// parity は「どちらが正しいか」を決めないので、**両 path に実寸の独立 oracle** を
// 当てて canonical を確定させる (CUDA 版は `tests/cuda_layer_oracle.rs`)。

/// 実 model と同じ shape (MiniCPM5-2B の 1 層)。CUDA 版と同一値。
fn real_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 130_560,
        hidden_dim: 2048,
        intermediate_dim: 6144,
        num_heads: 16,
        num_kv_heads: 2,
        num_layers: 1,
        max_seq_len: 512,
        head_dim: 128,
        rope_theta: 5_000_000.0,
        norm_eps: 1e-6,
        attention_bias: false,
    }
}

/// 実 shape 用の重み。振幅を 1/sqrt(fan_in) 級に落として activation の発散を防ぐ。
/// CUDA 版 `real_weights` と同一 (同じ seed / 同じスケール)。
fn real_weights(config: &LlamaConfig) -> LlamaLayerWeights {
    let h = config.hidden_dim;
    let kv = config.num_kv_heads * config.head_dim;
    let q = config.num_heads * config.head_dim;
    let i = config.intermediate_dim;
    #[allow(clippy::cast_precision_loss)]
    let s = 1.0 / (h as f32).sqrt();
    let scaled =
        |seed: u64, n: usize| -> Vec<f32> { fill(seed, n).iter().map(|v| v * s).collect() };
    LlamaLayerWeights {
        attn_norm: fill(11, h).iter().map(|v| 1.0 + v).collect(),
        q_proj: scaled(12, q * h),
        k_proj: scaled(13, kv * h),
        v_proj: scaled(14, kv * h),
        o_proj: scaled(15, h * q),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        ffn_norm: fill(16, h).iter().map(|v| 1.0 + v).collect(),
        gate_proj: scaled(17, i * h),
        up_proj: scaled(18, i * h),
        down_proj: scaled(19, h * i),
    }
}

/// 解析勾配の絶対値が大きい index を上から `count` 個返す (信号 > f32 桁落ち)。
fn top_indices(grad: &[f32], count: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..grad.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        grad[b]
            .abs()
            .partial_cmp(&grad[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(count);
    idx
}

/// 実 shape の CPU backward 中心差分 oracle。
///
/// CPU forward が 1 回 2 秒級なので既定 run からは外す:
/// `cargo test --release --test layer_backward_oracle -- --ignored --nocapture`
#[test]
#[ignore = "実 shape は CPU forward 80 回で数分かかる (GPU 版と対で手動実行する)"]
fn layer_backward_matches_central_difference_at_real_shape() {
    const H_REAL: f32 = 5e-3;
    const PROBES: usize = 4;

    let config = real_config();
    let w = real_weights(&config);
    let seq = 466usize;
    let hd = config.hidden_dim;
    let input: Vec<f32> = fill(21, seq * hd).iter().map(|v| v * 0.5).collect();

    // M は疎 (CUDA 版と同一): 全要素に重みを乗せると L が 10^4 級になり
    // f32 の桁落ちが中心差分の分子を潰す。
    let mut m = vec![0.0f32; seq * hd];
    for (row, &t) in [7usize, 113, 229, 351, 465].iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let vals = fill(300 + row as u64, hd);
        m[t * hd..(t + 1) * hd].copy_from_slice(&vals);
    }

    let mut h = input.clone();
    let cache = layer_forward(&mut h, &w, &config, seq);
    let (d_input, grads) = layer_backward(&m, &cache, &w, &config, seq);

    println!("layer_backward 数値微分 oracle 実 shape (hidden {hd} seq {seq})");

    check_with_step(
        "d_input",
        &d_input,
        &top_indices(&d_input, PROBES),
        H_REAL,
        |idx, step| {
            let mut p = input.clone();
            p[idx] += step;
            let plus = loss(&p, &w, &m, &config, seq);
            p[idx] -= 2.0 * step;
            let minus = loss(&p, &w, &m, &config, seq);
            (plus, minus)
        },
    );

    macro_rules! check_real {
        ($label:literal, $field:ident, $grad:ident) => {{
            let g = &grads.$grad;
            check_with_step($label, g, &top_indices(g, PROBES), H_REAL, |idx, step| {
                let mut wp = w.clone();
                wp.$field[idx] += step;
                let plus = loss(&input, &wp, &m, &config, seq);
                wp.$field[idx] -= 2.0 * step;
                let minus = loss(&input, &wp, &m, &config, seq);
                (plus, minus)
            });
        }};
    }
    check_real!("d_q_proj", q_proj, d_q_proj);
    check_real!("d_k_proj", k_proj, d_k_proj);
    check_real!("d_v_proj", v_proj, d_v_proj);
    check_real!("d_o_proj", o_proj, d_o_proj);
    check_real!("d_gate_proj", gate_proj, d_gate_proj);
    check_real!("d_up_proj", up_proj, d_up_proj);
    check_real!("d_down_proj", down_proj, d_down_proj);
    check_real!("d_attn_norm", attn_norm, d_attn_norm);
    check_real!("d_ffn_norm", ffn_norm, d_ffn_norm);
}
