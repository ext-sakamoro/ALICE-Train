//! CUDA layer path の数値微分 oracle (GPU 必須)。
//!
//! `cuda_layer_forward_ws_vram` / `cuda_layer_backward_ws_vram` を、CPU path と
//! 同じ中心差分で検証する。CPU path との parity (`--verify-cuda-parity`) だけでは
//! 「どちらが正しいか」が決まらないので、**CUDA 側にも独立 oracle を当てる**。
//!
//! 実行には実 GPU が要る (driver 不在では cudarc が内部 panic する)。
//! `feature = "cuda"` でのみ compile されるので、default features の CI には乗らない。
//!
//! ```bash
//! cargo test --release --features qat-cuda --test cuda_layer_oracle -- --nocapture
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
