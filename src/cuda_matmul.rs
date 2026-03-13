//! CUDA FP32 行列演算 — cuBLAS sgemm による高速 matmul。
//!
//! wgpu/Vulkan が使えない環境（Paperspace 等）で CUDA 直接呼び出しにより
//! GPU matmul を実行する。cuBLAS は NVIDIA 最適化済みで理論性能に近い。
//!
//! # 設計
//!
//! - `cudarc` クレートで CUDA Driver API + cuBLAS を呼び出す
//! - `matmul_bt(A, B, m, n, k)` → C = A × B^T を cuBLAS sgemm で実行
//! - `swiglu_ffn` → 3 回の sgemm + SiLU (CPU fallback) で FFN 実行
//! - バッファ管理は都度 alloc/free（ストリーミング方式と相性良い）

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;

/// CUDA matmul エンジン。
pub struct CudaMatmul {
    /// CUDA ストリーム
    stream: Arc<CudaStream>,
    /// cuBLAS ハンドル
    blas: CudaBlas,
}

impl CudaMatmul {
    /// CUDA デバイス 0 で初期化。
    ///
    /// # Panics
    ///
    /// CUDA デバイスが見つからない場合。
    #[must_use]
    pub fn new() -> Self {
        let ctx = CudaContext::new(0).expect("CUDA device 0 の初期化に失敗");
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).expect("cuBLAS 初期化に失敗");
        Self { stream, blas }
    }

    /// C = A × B^T を cuBLAS sgemm で計算。
    ///
    /// A: (m × k) row-major, B: (n × k) row-major, C: (m × n) row-major
    ///
    /// cuBLAS は column-major なので、row-major の A×B^T を column-major で表現:
    /// C^T = B × A^T (column-major) → cublas(N, T, n, m, k, B, A) → C^T in col-major = C in row-major
    pub fn matmul_bt(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let d_a = self.stream.clone_htod(a).expect("A の GPU 転送に失敗");
        let d_b = self.stream.clone_htod(b).expect("B の GPU 転送に失敗");
        let mut d_c: CudaSlice<f32> = self.stream.alloc_zeros(m * n).expect("C の確保に失敗");

        // Row-major C = A × B^T
        // → Column-major: C^T = B × A^T
        // cublasSgemm(N, T, n, m, k, 1.0, B, n, A, k, 0.0, C, n)
        //
        // transa=N → op(B) = B,    B は (n×k) col-major → leading dim = n
        // transb=T → op(A^T) = A,  A は (m×k) col-major as (k×m) → leading dim = k
        // 結果 C は (n×m) col-major = (m×n) row-major
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: n as i32,  // B is (n×k) row-major, viewed as col-major ldb=n
            beta: 0.0f32,
            ldc: n as i32,
        };

        unsafe {
            self.blas.gemm(cfg, &d_b, &d_a, &mut d_c)
                .expect("cuBLAS sgemm 実行に失敗");
        }

        let mut result = vec![0.0f32; m * n];
        self.stream.memcpy_dtoh(&d_c, &mut result).expect("結果の GPU 転送に失敗");
        result
    }

    /// SwiGLU FFN を CUDA で実行。
    ///
    /// gate = input × gate_proj^T, up = input × up_proj^T
    /// output = (SiLU(gate) ⊙ up) × down_proj^T
    ///
    /// SiLU と elementwise mul は CPU で実行（データ量が小さいため十分高速）。
    pub fn swiglu_ffn(
        &self,
        input: &[f32],
        gate_proj: &[f32],
        up_proj: &[f32],
        down_proj: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Vec<f32> {
        // gate = input × gate_proj^T  (seq_len × intermediate_dim)
        let mut gate = self.matmul_bt(input, gate_proj, seq_len, intermediate_dim, hidden_dim);

        // up = input × up_proj^T  (seq_len × intermediate_dim)
        let up = self.matmul_bt(input, up_proj, seq_len, intermediate_dim, hidden_dim);

        // SiLU(gate) ⊙ up (CPU — seq_len × intermediate_dim ≈ 3.7M 要素、数ms)
        let total = seq_len * intermediate_dim;
        for i in 0..total {
            let x = gate[i];
            gate[i] = (x / (1.0 + (-x).exp())) * up[i];
        }

        // output = intermediate × down_proj^T  (seq_len × hidden_dim)
        self.matmul_bt(&gate, down_proj, seq_len, hidden_dim, intermediate_dim)
    }
}

// ── CUDA レイヤー Forward ──────────────────────────────────────────────────

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

    // 2. QKV projection (CUDA cuBLAS sgemm)
    let q = cuda.matmul_bt(&normed, &weights.q_proj, seq_len, num_heads * head_dim, hidden_dim);
    let k = cuda.matmul_bt(&normed, &weights.k_proj, seq_len, num_kv_heads * head_dim, hidden_dim);
    let v = cuda.matmul_bt(&normed, &weights.v_proj, seq_len, num_kv_heads * head_dim, hidden_dim);

    let mut q = q;
    let mut k = k;

    // 3. RoPE (CPU)
    apply_rope(&mut q, num_heads, head_dim, seq_len, config.rope_theta);
    apply_rope(&mut k, num_kv_heads, head_dim, seq_len, config.rope_theta);

    // 4. GQA Attention (CPU — seq_len^2 × head_dim, 小規模)
    let mut attn_out_raw = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut attn_weights = vec![0.0f32; num_heads * seq_len * seq_len];
    gqa_attention(&q, &k, &v, &mut attn_out_raw, &mut attn_weights, config, seq_len);

    // 5. O projection (CUDA)
    let attn_out = cuda.matmul_bt(&attn_out_raw, &weights.o_proj, seq_len, hidden_dim, num_heads * head_dim);

    // 6. Residual add (CPU)
    for i in 0..input.len() {
        input[i] = residual_attn[i] + attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm (CPU)
    let residual_ffn = input.clone();
    let mut normed_ffn_buf = input.clone();
    rmsnorm(&mut normed_ffn_buf, &weights.ffn_norm, hidden_dim, config.norm_eps);
    let normed_ffn = normed_ffn_buf.clone();

    // 8. SwiGLU FFN (CUDA — 3 sgemm + CPU SiLU)
    let ffn_out = cuda.swiglu_ffn(
        &normed_ffn_buf,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );

    // gate/up/gate_silu を backward 用に再計算
    let gate = cuda.matmul_bt(&normed_ffn_buf, &weights.gate_proj, seq_len, intermediate_dim, hidden_dim);
    let up = cuda.matmul_bt(&normed_ffn_buf, &weights.up_proj, seq_len, intermediate_dim, hidden_dim);
    let gate_silu: Vec<f32> = gate.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    // 9. Residual add (CPU)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_matmul_bt_identity() {
        let cuda = CudaMatmul::new();
        // A = [[1, 2], [3, 4]], B = I (identity)
        let a = vec![1.0, 2.0, 3.0, 4.0f32];
        let b = vec![1.0, 0.0, 0.0, 1.0f32];
        let c = cuda.matmul_bt(&a, &b, 2, 2, 2);
        assert!((c[0] - 1.0).abs() < 1e-4, "c[0]={}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-4, "c[1]={}", c[1]);
        assert!((c[2] - 3.0).abs() < 1e-4, "c[2]={}", c[2]);
        assert!((c[3] - 4.0).abs() < 1e-4, "c[3]={}", c[3]);
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

        // CPU reference
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
