//! CUDA layer path の数値微分 oracle (GPU 必須)。
//!
//! `cuda_layer_forward_ws_vram` / `cuda_layer_backward_ws_vram` を、CPU path と
//! 同じ中心差分で検証する。CPU path との parity (`--verify-cuda-parity`) だけでは
//! 「どちらが正しいか」が決まらないので、**CUDA 側にも独立 oracle を当てる**。
//!
//! 実行には実 GPU が要る (driver 不在では cudarc が内部 panic する)。
//! `feature = "cuda"` でのみ compile されるので、default features の CI には乗らない。
//!
//! feature は `cuda` 単独で指定する。`qat-cuda` を指定すると
//! `required-features = ["qat-cuda"]` の `train-qat-70b` bin まで build 対象に入り、
//! そちらの compile error (47 件) で test が丸ごと落ちる (罠
//! `cargo-test-blocked-by-broken-bin`)。
//!
//! ```bash
//! cargo test --release --features cuda --test cuda_layer_oracle -- --nocapture
//! ```

#![cfg(feature = "cuda")]

use alice_train::blas;
use alice_train::cuda_matmul::{
    cuda_layer_backward_ws_vram, cuda_layer_forward_ws_vram, CudaLayerWorkspace, VramLayerWeights,
};
use alice_train::llama::{LlamaConfig, LlamaLayerWeights};

const H: f32 = 2e-3;
const RTOL: f64 = 5e-2; // TF32 の cuBLAS を通るので CPU 版より緩める
const ATOL: f64 = 5e-3;

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

fn picks(n: usize, count: usize, seed: usize) -> Vec<usize> {
    (0..count)
        .map(|i| (i * 7919 + seed * 104_729) % n)
        .collect()
}

/// 実 model と同じ shape (MiniCPM5-2B の 1 層)。
///
/// 小 shape (hidden 32 / seq 5) で green でも実 shape で落ちる経路がある
/// (tile / kv broadcast / GPU softmax の seq 依存) ので、**実寸で独立 oracle を当てる**。
/// `--verify-cuda-parity` は CPU と CUDA のどちらが正しいかを決めないため、
/// ここで canonical を確定させる。
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
fn real_weights(config: &LlamaConfig) -> LlamaLayerWeights {
    real_weights_seeded(config, 0)
}

/// [`real_weights`] の seed をずらした版 (別 layer 相当の重みを作るため)。
fn real_weights_seeded(config: &LlamaConfig, off: u64) -> LlamaLayerWeights {
    let h = config.hidden_dim;
    let kv = config.num_kv_heads * config.head_dim;
    let q = config.num_heads * config.head_dim;
    let i = config.intermediate_dim;
    #[allow(clippy::cast_precision_loss)]
    let s = 1.0 / (h as f32).sqrt(); // ≈ 0.022
    let scaled =
        |seed: u64, n: usize| -> Vec<f32> { fill(seed + off, n).iter().map(|v| v * s).collect() };
    LlamaLayerWeights {
        attn_norm: fill(11 + off, h).iter().map(|v| 1.0 + v).collect(),
        q_proj: scaled(12, q * h),
        k_proj: scaled(13, kv * h),
        v_proj: scaled(14, kv * h),
        o_proj: scaled(15, h * q),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        ffn_norm: fill(16 + off, h).iter().map(|v| 1.0 + v).collect(),
        gate_proj: scaled(17, i * h),
        up_proj: scaled(18, i * h),
        down_proj: scaled(19, h * i),
    }
}

/// 解析勾配の絶対値が大きい index を上から `count` 個返す。
///
/// 実 shape では `L` が 10^2 級になり f32 forward の桁落ちが中心差分の分子に
/// 効くので、**信号の大きい要素で判定する** (ゼロ近傍要素を引くと数値微分の
/// 誤差だけを見ることになる)。期待値そのものは中心差分から来ており、
/// 実装の出力を見て書いていない。
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

#[test]
fn cuda_layer_backward_matches_central_difference() {
    blas::init_cuda_blas();
    assert!(
        blas::cuda_blas_available(),
        "CUDA を初期化できない (GPU 必須)"
    );

    let config = tiny_config();
    let w = tiny_weights(&config);
    let seq = 5usize;
    let hd = config.hidden_dim;
    let input = fill(21, seq * hd);
    let m = fill(22, seq * hd);
    let mut ws = CudaLayerWorkspace::new(&config, config.max_seq_len);

    // 解析勾配
    let (d_input, grads) = {
        let cuda_mtx = blas::CUDA_MATMUL.get().expect("初期化済");
        let cuda = cuda_mtx.lock().expect("mutex");
        let vram = VramLayerWeights::upload(&cuda, &w);
        let mut h = input.clone();
        let cache = cuda_layer_forward_ws_vram(&cuda, &mut h, &w, &vram, &config, seq, &mut ws);
        let (d_in, g) =
            cuda_layer_backward_ws_vram(&cuda, &m, &cache, &w, &vram, &config, seq, &mut ws);
        let g: alice_train::llama_backward::LayerWeightGrads = g.into();
        (d_in, g)
    };

    println!("CUDA layer backward 数値微分 oracle (hidden {hd} seq {seq})");
    let mut failures: Vec<String> = Vec::new();

    // ── d_input ──
    {
        let mut worst = 0.0f64;
        let mut pair = (0.0f64, 0.0f64);
        let mut at = 0usize;
        for idx in picks(input.len(), 6, 1) {
            let mut p = input.clone();
            p[idx] += H;
            let plus = loss(&w, &p, &m, &config, seq, &mut ws);
            p[idx] -= 2.0 * H;
            let minus = loss(&w, &p, &m, &config, seq, &mut ws);
            let num = (plus - minus) / f64::from(2.0 * H);
            let ana = f64::from(d_input[idx]);
            let ratio = (num - ana).abs() / (ATOL + RTOL * ana.abs().max(num.abs()));
            if ratio > worst {
                worst = ratio;
                pair = (num, ana);
                at = idx;
            }
        }
        record("d_input", worst, pair, at, &mut failures);
    }

    // ── 重み勾配 ──
    macro_rules! check_w {
        ($label:literal, $field:ident, $grad:ident, $seed:expr) => {{
            let mut worst = 0.0f64;
            let mut pair = (0.0f64, 0.0f64);
            let mut at = 0usize;
            for idx in picks(w.$field.len(), 6, $seed) {
                let mut wp = w.clone();
                wp.$field[idx] += H;
                let plus = loss(&wp, &input, &m, &config, seq, &mut ws);
                wp.$field[idx] -= 2.0 * H;
                let minus = loss(&wp, &input, &m, &config, seq, &mut ws);
                let num = (plus - minus) / f64::from(2.0 * H);
                let ana = f64::from(grads.$grad[idx]);
                let ratio = (num - ana).abs() / (ATOL + RTOL * ana.abs().max(num.abs()));
                if ratio > worst {
                    worst = ratio;
                    pair = (num, ana);
                    at = idx;
                }
            }
            record($label, worst, pair, at, &mut failures);
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

    assert!(
        failures.is_empty(),
        "CUDA backward が中心差分と合わない: {}",
        failures.join(", ")
    );
}

/// 実 shape (hidden 2048 / heads 16 / kv 2 / head_dim 128 / seq 466) の中心差分 oracle。
///
/// 小 shape 版が green なのに `--verify-cuda-parity` は layer 40 以降で attention 側
/// 勾配 (`d_input` / `d_q/k/v/o_proj` / `d_attn_norm`) が食い違い、FFN 側は一致する。
/// parity は「どちらが正しいか」を決めないので、実寸で独立 oracle を当てて canonical を
/// 確定させる (同じ shape の CPU 版は `tests/layer_backward_oracle.rs` の実 shape test)。
#[test]
fn cuda_layer_backward_matches_central_difference_at_real_shape() {
    const H_REAL: f32 = 5e-3;
    const PROBES: usize = 4;

    blas::init_cuda_blas();
    assert!(
        blas::cuda_blas_available(),
        "CUDA を初期化できない (GPU 必須)"
    );

    let config = real_config();
    let w = real_weights(&config);
    let seq = 466usize; // A3 の sample[0] と同じ長さ
    let hd = config.hidden_dim;
    #[allow(clippy::cast_precision_loss)]
    let input: Vec<f32> = fill(21, seq * hd)
        .iter()
        .map(|v| v * 0.5) // embedding 相当のスケール
        .collect();

    // M は疎にする: 全 seq×hidden に重みを乗せると L が 10^4 級になり、
    // f32 forward の桁落ちが中心差分の分子 (差が 10^-3 級) を潰す。
    let mut m = vec![0.0f32; seq * hd];
    for (row, &t) in [7usize, 113, 229, 351, 465].iter().enumerate() {
        let vals = fill(300 + row as u64, hd);
        m[t * hd..(t + 1) * hd].copy_from_slice(&vals);
    }
    let mut ws = CudaLayerWorkspace::new(&config, config.max_seq_len);

    let (d_input, grads) = {
        let cuda_mtx = blas::CUDA_MATMUL.get().expect("初期化済");
        let cuda = cuda_mtx.lock().expect("mutex");
        let vram = VramLayerWeights::upload(&cuda, &w);
        let mut h = input.clone();
        let cache = cuda_layer_forward_ws_vram(&cuda, &mut h, &w, &vram, &config, seq, &mut ws);
        let (d_in, g) =
            cuda_layer_backward_ws_vram(&cuda, &m, &cache, &w, &vram, &config, seq, &mut ws);
        let g: alice_train::llama_backward::LayerWeightGrads = g.into();
        (d_in, g)
    };

    println!("CUDA layer backward 数値微分 oracle 実 shape (hidden {hd} seq {seq})");
    let mut failures: Vec<String> = Vec::new();

    {
        let mut worst = 0.0f64;
        let mut pair = (0.0f64, 0.0f64);
        let mut at = 0usize;
        for idx in top_indices(&d_input, PROBES) {
            let mut p = input.clone();
            p[idx] += H_REAL;
            let plus = loss(&w, &p, &m, &config, seq, &mut ws);
            p[idx] -= 2.0 * H_REAL;
            let minus = loss(&w, &p, &m, &config, seq, &mut ws);
            let num = (plus - minus) / f64::from(2.0 * H_REAL);
            let ana = f64::from(d_input[idx]);
            let ratio = (num - ana).abs() / (ATOL + RTOL * ana.abs().max(num.abs()));
            if ratio > worst {
                worst = ratio;
                pair = (num, ana);
                at = idx;
            }
        }
        record("d_input", worst, pair, at, &mut failures);
    }

    macro_rules! check_real {
        ($label:literal, $field:ident, $grad:ident) => {{
            let mut worst = 0.0f64;
            let mut pair = (0.0f64, 0.0f64);
            let mut at = 0usize;
            for idx in top_indices(&grads.$grad, PROBES) {
                let mut wp = w.clone();
                wp.$field[idx] += H_REAL;
                let plus = loss(&wp, &input, &m, &config, seq, &mut ws);
                wp.$field[idx] -= 2.0 * H_REAL;
                let minus = loss(&wp, &input, &m, &config, seq, &mut ws);
                let num = (plus - minus) / f64::from(2.0 * H_REAL);
                let ana = f64::from(grads.$grad[idx]);
                let ratio = (num - ana).abs() / (ATOL + RTOL * ana.abs().max(num.abs()));
                if ratio > worst {
                    worst = ratio;
                    pair = (num, ana);
                    at = idx;
                }
            }
            record($label, worst, pair, at, &mut failures);
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

    assert!(
        failures.is_empty(),
        "実 shape の CUDA backward が中心差分と合わない: {}",
        failures.join(", ")
    );
}

/// 同じ `CudaLayerWorkspace` で forward→backward を **2 組連続** 回した時、
/// 2 組目の勾配が中心差分と合うかを見る (順序依存の検出)。
///
/// 実 model の `--verify-cuda-parity` では backward を layer 41 → 0 と回すと
/// **1 本目 (layer 41) だけ CPU と一致し、2 本目以降が無相関になる**。1 組だけ回す
/// oracle は green なので、「2 組目」を明示的に検査する test を置く。
/// 重みは 1 組目と 2 組目で変える (別 layer を回すのと同じ状況にするため)。
#[test]
fn cuda_layer_backward_is_correct_on_the_second_consecutive_pair() {
    const H_REAL: f32 = 5e-3;
    const PROBES: usize = 3;

    blas::init_cuda_blas();
    assert!(
        blas::cuda_blas_available(),
        "CUDA を初期化できない (GPU 必須)"
    );

    let config = real_config();
    let wa = real_weights_seeded(&config, 0);
    let wb = real_weights_seeded(&config, 50);
    let seq = 466usize;
    let hd = config.hidden_dim;
    let input: Vec<f32> = fill(21, seq * hd).iter().map(|v| v * 0.5).collect();
    let mut m = vec![0.0f32; seq * hd];
    for (row, &t) in [7usize, 113, 229, 351, 465].iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let vals = fill(300 + row as u64, hd);
        m[t * hd..(t + 1) * hd].copy_from_slice(&vals);
    }
    let mut ws = CudaLayerWorkspace::new(&config, config.max_seq_len);

    // 1 組目 (wa) を回して捨て、そのあと 2 組目 (wb) の勾配を取る
    let (d_input, grads) = {
        let cuda_mtx = blas::CUDA_MATMUL.get().expect("初期化済");
        let cuda = cuda_mtx.lock().expect("mutex");

        let vram_a = VramLayerWeights::upload(&cuda, &wa);
        let mut ha = input.clone();
        let cache_a =
            cuda_layer_forward_ws_vram(&cuda, &mut ha, &wa, &vram_a, &config, seq, &mut ws);
        let _ =
            cuda_layer_backward_ws_vram(&cuda, &m, &cache_a, &wa, &vram_a, &config, seq, &mut ws);

        let vram_b = VramLayerWeights::upload(&cuda, &wb);
        let mut hb = input.clone();
        let cache_b =
            cuda_layer_forward_ws_vram(&cuda, &mut hb, &wb, &vram_b, &config, seq, &mut ws);
        let (d_in, g) =
            cuda_layer_backward_ws_vram(&cuda, &m, &cache_b, &wb, &vram_b, &config, seq, &mut ws);
        let g: alice_train::llama_backward::LayerWeightGrads = g.into();
        (d_in, g)
    };

    println!("CUDA layer backward 2 組目の数値微分 oracle (hidden {hd} seq {seq})");
    let mut failures: Vec<String> = Vec::new();

    {
        let mut worst = 0.0f64;
        let mut pair = (0.0f64, 0.0f64);
        let mut at = 0usize;
        for idx in top_indices(&d_input, PROBES) {
            let mut p = input.clone();
            p[idx] += H_REAL;
            let plus = loss(&wb, &p, &m, &config, seq, &mut ws);
            p[idx] -= 2.0 * H_REAL;
            let minus = loss(&wb, &p, &m, &config, seq, &mut ws);
            let num = (plus - minus) / f64::from(2.0 * H_REAL);
            let ana = f64::from(d_input[idx]);
            let ratio = (num - ana).abs() / (ATOL + RTOL * ana.abs().max(num.abs()));
            if ratio > worst {
                worst = ratio;
                pair = (num, ana);
                at = idx;
            }
        }
        record("d_input", worst, pair, at, &mut failures);
    }

    macro_rules! check_second {
        ($label:literal, $field:ident, $grad:ident) => {{
            let mut worst = 0.0f64;
            let mut pair = (0.0f64, 0.0f64);
            let mut at = 0usize;
            for idx in top_indices(&grads.$grad, PROBES) {
                let mut wp = wb.clone();
                wp.$field[idx] += H_REAL;
                let plus = loss(&wp, &input, &m, &config, seq, &mut ws);
                wp.$field[idx] -= 2.0 * H_REAL;
                let minus = loss(&wp, &input, &m, &config, seq, &mut ws);
                let num = (plus - minus) / f64::from(2.0 * H_REAL);
                let ana = f64::from(grads.$grad[idx]);
                let ratio = (num - ana).abs() / (ATOL + RTOL * ana.abs().max(num.abs()));
                if ratio > worst {
                    worst = ratio;
                    pair = (num, ana);
                    at = idx;
                }
            }
            record($label, worst, pair, at, &mut failures);
        }};
    }
    check_second!("d_q_proj", q_proj, d_q_proj);
    check_second!("d_k_proj", k_proj, d_k_proj);
    check_second!("d_v_proj", v_proj, d_v_proj);
    check_second!("d_o_proj", o_proj, d_o_proj);
    check_second!("d_gate_proj", gate_proj, d_gate_proj);
    check_second!("d_up_proj", up_proj, d_up_proj);
    check_second!("d_down_proj", down_proj, d_down_proj);
    check_second!("d_attn_norm", attn_norm, d_attn_norm);
    check_second!("d_ffn_norm", ffn_norm, d_ffn_norm);

    assert!(
        failures.is_empty(),
        "2 組目の CUDA backward が中心差分と合わない: {}",
        failures.join(", ")
    );
}

/// `L = Σ M ⊙ cuda_forward(x)` を f64 で返す。
fn loss(
    w: &LlamaLayerWeights,
    x: &[f32],
    m: &[f32],
    config: &LlamaConfig,
    seq: usize,
    ws: &mut CudaLayerWorkspace,
) -> f64 {
    let cuda_mtx = blas::CUDA_MATMUL.get().expect("初期化済");
    let cuda = cuda_mtx.lock().expect("mutex");
    let vram = VramLayerWeights::upload(&cuda, w);
    let mut h = x.to_vec();
    let _c = cuda_layer_forward_ws_vram(&cuda, &mut h, w, &vram, config, seq, ws);
    h.iter()
        .zip(m.iter())
        .map(|(o, mm)| f64::from(*o) * f64::from(*mm))
        .sum()
}

/// 突合結果を 1 行出して、閾値超えなら failures に積む。
fn record(label: &str, worst: f64, pair: (f64, f64), at: usize, failures: &mut Vec<String>) {
    let mark = if worst <= 1.0 { "ok" } else { "!!" };
    println!(
        "  {mark} {label:<12} 最悪許容比 {worst:>8.2}  idx={at} 数値={:+.4e} 解析={:+.4e}",
        pair.0, pair.1
    );
    if worst > 1.0 {
        failures.push(format!("{label} (許容比 {worst:.1})"));
    }
}
