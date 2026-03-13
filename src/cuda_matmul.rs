//! CUDA FP32 行列演算 — cuBLAS sgemm による高速 matmul。
//!
//! wgpu/Vulkan が使えない環境（Paperspace 等）で CUDA 直接呼び出しにより
//! GPU matmul を実行する。cuBLAS は NVIDIA 最適化済みで理論性能に近い。
//!
//! # Row-major → cuBLAS (col-major) 変換公式
//!
//! Row-major `X[a×b]` は cuBLAS に `X_cm[b×a]` (ld=b) として見える = X^T。
//! 従って row-major `C = A × B` は cuBLAS で `C^T = B^T × A^T` として計算。

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;

/// CUDA matmul エンジン。
pub struct CudaMatmul {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
}

impl CudaMatmul {
    /// CUDA デバイス 0 で初期化。
    #[must_use]
    pub fn new() -> Self {
        let ctx = CudaContext::new(0).expect("CUDA device 0 の初期化に失敗");
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).expect("cuBLAS 初期化に失敗");
        Self { stream, blas }
    }

    /// C\[m×n\] = A\[m×k\] × B\[n×k\]^T  (row-major)
    ///
    /// cuBLAS: C^T\[n×m\] = op(B_cm) × op(A_cm)
    /// B_rm\[n×k\] → B_cm\[k×n\] (ld=k). Transpose → B\[n×k\]. transa=T, lda=k.
    /// A_rm\[m×k\] → A_cm\[k×m\] (ld=k). No transpose → A^T\[k×m\]. transb=N, ldb=k.
    pub fn matmul_bt(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let d_a = self.stream.clone_htod(a).expect("A→GPU 転送失敗");
        let d_b = self.stream.clone_htod(b).expect("B→GPU 転送失敗");
        let mut d_c: CudaSlice<f32> = self.stream.alloc_zeros(m * n).expect("C 確保失敗");

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: k as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };

        unsafe {
            self.blas
                .gemm(cfg, &d_b, &d_a, &mut d_c)
                .expect("cuBLAS sgemm (matmul_bt) 失敗");
        }

        let mut result = vec![0.0f32; m * n];
        self.stream
            .memcpy_dtoh(&d_c, &mut result)
            .expect("GPU→CPU 転送失敗");
        result
    }

    /// C\[m×n\] = A\[m×k\] × B\[k×n\]  (row-major, 標準 matmul)
    ///
    /// cuBLAS: C^T\[n×m\] = B^T\[n×k\] × A^T\[k×m\]
    /// B_rm\[k×n\] → B_cm\[n×k\] (ld=n). No transpose. transa=N, lda=n.
    /// A_rm\[m×k\] → A_cm\[k×m\] (ld=k). No transpose. transb=N, ldb=k.
    pub fn matmul_nn(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let d_a = self.stream.clone_htod(a).expect("A→GPU 転送失敗");
        let d_b = self.stream.clone_htod(b).expect("B→GPU 転送失敗");
        let mut d_c: CudaSlice<f32> = self.stream.alloc_zeros(m * n).expect("C 確保失敗");

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };

        unsafe {
            self.blas
                .gemm(cfg, &d_b, &d_a, &mut d_c)
                .expect("cuBLAS sgemm (matmul_nn) 失敗");
        }

        let mut result = vec![0.0f32; m * n];
        self.stream
            .memcpy_dtoh(&d_c, &mut result)
            .expect("GPU→CPU 転送失敗");
        result
    }

    /// SwiGLU FFN を CUDA で実行。
    ///
    /// gate = input × gate_proj^T, up = input × up_proj^T
    /// intermediate = SiLU(gate) ⊙ up
    /// output = intermediate × down_proj^T
    ///
    /// 戻り値: (output, gate, up, gate_silu) — backward 用に中間値も返す
    pub fn swiglu_ffn(
        &self,
        input: &[f32],
        gate_proj: &[f32],
        up_proj: &[f32],
        down_proj: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let gate = self.matmul_bt(input, gate_proj, seq_len, intermediate_dim, hidden_dim);
        let up = self.matmul_bt(input, up_proj, seq_len, intermediate_dim, hidden_dim);

        // SiLU(gate) ⊙ up (CPU)
        let total = seq_len * intermediate_dim;
        let mut gate_silu = vec![0.0f32; total];
        let mut intermediate = vec![0.0f32; total];
        for i in 0..total {
            let x = gate[i];
            gate_silu[i] = x / (1.0 + (-x).exp());
            intermediate[i] = gate_silu[i] * up[i];
        }

        let output = self.matmul_bt(&intermediate, down_proj, seq_len, hidden_dim, intermediate_dim);
        (output, gate, up, gate_silu)
    }

    /// SwiGLU FFN backward (CUDA 加速)。
    ///
    /// d_input を返す。weight grads は現時点では計算しない（ストリーミングモード）。
    pub fn swiglu_ffn_backward(
        &self,
        d_output: &[f32],
        gate: &[f32],
        up: &[f32],
        gate_silu: &[f32],
        gate_proj: &[f32],
        up_proj: &[f32],
        down_proj: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Vec<f32> {
        // 1. down_proj backward: d_intermediate = d_output × down_proj
        //    down_proj is [hidden_dim × intermediate_dim]
        let d_intermediate =
            self.matmul_nn(d_output, down_proj, seq_len, intermediate_dim, hidden_dim);

        // 2. Element-wise SwiGLU backward (CPU)
        let total = seq_len * intermediate_dim;
        let mut d_gate = vec![0.0f32; total];
        let mut d_up = vec![0.0f32; total];
        for i in 0..total {
            // d_gate_silu = d_intermediate ⊙ up
            let d_gate_silu = d_intermediate[i] * up[i];
            // d_up = d_intermediate ⊙ gate_silu
            d_up[i] = d_intermediate[i] * gate_silu[i];
            // SiLU derivative: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            let x = gate[i];
            let sig = 1.0 / (1.0 + (-x).exp());
            let silu_grad = sig * (1.0 + x * (1.0 - sig));
            d_gate[i] = d_gate_silu * silu_grad;
        }

        // 3. gate_proj backward: d_input_gate = d_gate × gate_proj
        //    gate_proj is [intermediate_dim × hidden_dim]
        let d_input_gate =
            self.matmul_nn(&d_gate, gate_proj, seq_len, hidden_dim, intermediate_dim);

        // 4. up_proj backward: d_input_up = d_up × up_proj
        let d_input_up = self.matmul_nn(&d_up, up_proj, seq_len, hidden_dim, intermediate_dim);

        // 5. d_input = d_input_gate + d_input_up
        let mut d_input = d_input_gate;
        for i in 0..d_input.len() {
            d_input[i] += d_input_up[i];
        }
        d_input
    }
}

// ── CUDA レイヤー Forward / Backward ──────────────────────────────────────

use crate::llama::{LlamaConfig, LlamaLayerWeights};
use crate::llama_forward::{apply_rope, gqa_attention, rmsnorm, LayerCache};

/// CUDA 加速版 Transformer レイヤー forward。
///
/// QKV/O projection と SwiGLU FFN を cuBLAS sgemm で実行。
/// RMSNorm, RoPE, GQA Attention は CPU（小規模なので十分高速）。
pub fn cuda_layer_forward(
    cuda: &CudaMatmul,
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

    // 1. 残差保存 + Attention RMSNorm (CPU)
    let residual_attn = input.clone();
    let mut normed = input.clone();
    rmsnorm(&mut normed, &weights.attn_norm, hidden_dim, config.norm_eps);
    let normed_attn = normed.clone();

    // 2. QKV projection (CUDA)
    let mut q = cuda.matmul_bt(&normed, &weights.q_proj, seq_len, num_heads * head_dim, hidden_dim);
    let mut k = cuda.matmul_bt(&normed, &weights.k_proj, seq_len, num_kv_heads * head_dim, hidden_dim);
    let v = cuda.matmul_bt(&normed, &weights.v_proj, seq_len, num_kv_heads * head_dim, hidden_dim);

    // 3. RoPE (CPU)
    apply_rope(&mut q, num_heads, head_dim, seq_len, config.rope_theta);
    apply_rope(&mut k, num_kv_heads, head_dim, seq_len, config.rope_theta);

    // 4. GQA Attention (CPU)
    let mut attn_out_raw = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut attn_weights = vec![0.0f32; num_heads * seq_len * seq_len];
    gqa_attention(&q, &k, &v, &mut attn_out_raw, &mut attn_weights, config, seq_len);

    // 5. O projection (CUDA)
    let attn_out = cuda.matmul_bt(&attn_out_raw, &weights.o_proj, seq_len, hidden_dim, num_heads * head_dim);

    // 6. Residual add
    for i in 0..input.len() {
        input[i] = residual_attn[i] + attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm (CPU)
    let residual_ffn = input.clone();
    let mut normed_ffn_buf = input.clone();
    rmsnorm(&mut normed_ffn_buf, &weights.ffn_norm, hidden_dim, config.norm_eps);
    let normed_ffn = normed_ffn_buf.clone();

    // 8. SwiGLU FFN (CUDA) — 中間値も返す
    let (ffn_out, gate, up, gate_silu) = cuda.swiglu_ffn(
        &normed_ffn_buf,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
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
        gate,
        up,
        gate_silu,
    }
}

/// CUDA 加速版 Transformer レイヤー backward。
///
/// d_input のみ返す（weight grads は現時点ではスキップ — ストリーミングモード）。
/// Projection backward の matmul を cuBLAS で実行。
/// RMSNorm, RoPE, Attention backward は CPU。
pub fn cuda_layer_backward(
    cuda: &CudaMatmul,
    d_output: &[f32],
    cache: &LayerCache,
    weights: &LlamaLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
) -> Vec<f32> {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_dim = config.intermediate_dim;

    // ── FFN Backward ──

    // 1. SwiGLU FFN backward (CUDA matmul + CPU elementwise)
    let d_pre_ffn = cuda.swiglu_ffn_backward(
        d_output,
        &cache.gate,
        &cache.up,
        &cache.gate_silu,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );

    // 2. FFN RMSNorm backward (CPU)
    let mut d_pre_ffn_residual = vec![0.0f32; seq_len * hidden_dim];
    let mut _d_ffn_norm_w = vec![0.0f32; hidden_dim];
    crate::llama_backward::rmsnorm_backward(
        &d_pre_ffn,
        &cache.residual_ffn,
        &weights.ffn_norm,
        &mut d_pre_ffn_residual,
        &mut _d_ffn_norm_w,
        hidden_dim,
        config.norm_eps,
    );

    // 3. Residual: d_attn_output = d_output + d_pre_ffn_residual
    let mut d_attn_output = vec![0.0f32; seq_len * hidden_dim];
    for i in 0..d_attn_output.len() {
        d_attn_output[i] = d_output[i] + d_pre_ffn_residual[i];
    }

    // ── Attention Backward ──

    // 4. O projection backward: d_attn_raw = d_attn_output × o_proj
    //    o_proj is [hidden_dim × (num_heads * head_dim)]
    let d_attn_raw = cuda.matmul_nn(
        &d_attn_output,
        &weights.o_proj,
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );

    // 5. GQA Attention backward (CPU)
    let kv_group_size = num_heads / num_kv_heads;
    let mut d_q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut d_k = vec![0.0f32; seq_len * num_kv_heads * head_dim];
    let mut d_v = vec![0.0f32; seq_len * num_kv_heads * head_dim];

    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..num_heads {
        let kv_h = h / kv_group_size;
        for t in 0..seq_len {
            // d_attn_weights from d_attn_raw and V
            let mut d_attn_w = vec![0.0f32; seq_len];
            for s in 0..=t {
                let mut d_w = 0.0f32;
                for d in 0..head_dim {
                    let v_idx = s * num_kv_heads * head_dim + kv_h * head_dim + d;
                    let do_idx = t * num_heads * head_dim + h * head_dim + d;
                    d_w += d_attn_raw[do_idx] * cache.v[v_idx];
                }
                d_attn_w[s] = d_w;
            }

            // dV
            for s in 0..=t {
                let attn_w = cache.attn_weights[h * seq_len * seq_len + t * seq_len + s];
                for d in 0..head_dim {
                    let do_idx = t * num_heads * head_dim + h * head_dim + d;
                    let v_idx = s * num_kv_heads * head_dim + kv_h * head_dim + d;
                    d_v[v_idx] += attn_w * d_attn_raw[do_idx];
                }
            }

            // Softmax backward: d_score = attn * (d_w - dot(attn, d_w))
            let aw_off = h * seq_len * seq_len + t * seq_len;
            let mut dot = 0.0f32;
            for s in 0..=t {
                dot += cache.attn_weights[aw_off + s] * d_attn_w[s];
            }
            for s in 0..=t {
                let attn_w = cache.attn_weights[aw_off + s];
                let d_score = attn_w * (d_attn_w[s] - dot) * scale;

                // dQ += d_score * K[s]
                for d in 0..head_dim {
                    let q_idx = t * num_heads * head_dim + h * head_dim + d;
                    let k_idx = s * num_kv_heads * head_dim + kv_h * head_dim + d;
                    d_q[q_idx] += d_score * cache.k[k_idx];
                    d_k[k_idx] += d_score * cache.q[q_idx];
                }
            }
        }
    }

    // 6. RoPE backward (CPU — inverse rotation)
    crate::llama_backward::rope_backward(&mut d_q, num_heads, head_dim, seq_len, config.rope_theta);
    crate::llama_backward::rope_backward(
        &mut d_k,
        num_kv_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );

    // 7. QKV projection backward (CUDA)
    //    d_normed = d_q × q_proj + d_k × k_proj + d_v × v_proj
    let d_normed_q = cuda.matmul_nn(
        &d_q,
        &weights.q_proj,
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );
    let d_normed_k = cuda.matmul_nn(
        &d_k,
        &weights.k_proj,
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
    );
    let d_normed_v = cuda.matmul_nn(
        &d_v,
        &weights.v_proj,
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
    );

    let mut d_normed = d_normed_q;
    for i in 0..d_normed.len() {
        d_normed[i] += d_normed_k[i] + d_normed_v[i];
    }

    // 8. Attention RMSNorm backward (CPU)
    let mut d_input = vec![0.0f32; seq_len * hidden_dim];
    let mut _d_attn_norm_w = vec![0.0f32; hidden_dim];
    crate::llama_backward::rmsnorm_backward(
        &d_normed,
        &cache.residual_attn,
        &weights.attn_norm,
        &mut d_input,
        &mut _d_attn_norm_w,
        hidden_dim,
        config.norm_eps,
    );

    // 9. Residual: d_input += d_attn_output
    for i in 0..d_input.len() {
        d_input[i] += d_attn_output[i];
    }

    d_input
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_matmul_bt_identity() {
        let cuda = CudaMatmul::new();
        let a = vec![1.0, 2.0, 3.0, 4.0f32];
        let b = vec![1.0, 0.0, 0.0, 1.0f32];
        let c = cuda.matmul_bt(&a, &b, 2, 2, 2);
        assert!((c[0] - 1.0).abs() < 1e-4, "c[0]={}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-4, "c[1]={}", c[1]);
        assert!((c[2] - 3.0).abs() < 1e-4, "c[2]={}", c[2]);
        assert!((c[3] - 4.0).abs() < 1e-4, "c[3]={}", c[3]);
    }

    #[test]
    fn test_cuda_matmul_bt_rectangular() {
        let cuda = CudaMatmul::new();
        // A[2×3] × B[4×3]^T → C[2×4]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let b = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0f32,
        ];
        let c = cuda.matmul_bt(&a, &b, 2, 4, 3);
        // Row 0: [1*1+2*0+3*0, 1*0+2*1+3*0, 1*0+2*0+3*1, 1*1+2*1+3*1] = [1, 2, 3, 6]
        // Row 1: [4*1+5*0+6*0, 4*0+5*1+6*0, 4*0+5*0+6*1, 4*1+5*1+6*1] = [4, 5, 6, 15]
        assert!((c[0] - 1.0).abs() < 1e-4);
        assert!((c[1] - 2.0).abs() < 1e-4);
        assert!((c[2] - 3.0).abs() < 1e-4);
        assert!((c[3] - 6.0).abs() < 1e-4);
        assert!((c[4] - 4.0).abs() < 1e-4);
        assert!((c[5] - 5.0).abs() < 1e-4);
        assert!((c[6] - 6.0).abs() < 1e-4);
        assert!((c[7] - 15.0).abs() < 1e-4);
    }

    #[test]
    fn test_cuda_matmul_nn() {
        let cuda = CudaMatmul::new();
        // A[2×3] × B[3×2] → C[2×2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let c = cuda.matmul_nn(&a, &b, 2, 2, 3);
        // C[0,0] = 1*1 + 2*3 + 3*5 = 22
        // C[0,1] = 1*2 + 2*4 + 3*6 = 28
        // C[1,0] = 4*1 + 5*3 + 6*5 = 49
        // C[1,1] = 4*2 + 5*4 + 6*6 = 64
        assert!((c[0] - 22.0).abs() < 1e-3, "c[0]={}", c[0]);
        assert!((c[1] - 28.0).abs() < 1e-3, "c[1]={}", c[1]);
        assert!((c[2] - 49.0).abs() < 1e-3, "c[2]={}", c[2]);
        assert!((c[3] - 64.0).abs() < 1e-3, "c[3]={}", c[3]);
    }

    #[test]
    fn test_cuda_matmul_bt_large() {
        let cuda = CudaMatmul::new();
        let m = 64;
        let k = 128;
        let n = 64;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.1).collect();

        let gpu_result = cuda.matmul_bt(&a, &b, m, n, k);

        let mut cpu_result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for h in 0..k {
                    sum += a[i * k + h] * b[j * k + h];
                }
                cpu_result[i * n + j] = sum;
            }
        }

        for idx in 0..m * n {
            assert!(
                (gpu_result[idx] - cpu_result[idx]).abs() < 0.1,
                "mismatch at {idx}: gpu={} cpu={}",
                gpu_result[idx],
                cpu_result[idx]
            );
        }
    }
}
