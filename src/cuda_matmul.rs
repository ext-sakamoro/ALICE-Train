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
use rayon::prelude::*;
use std::sync::Arc;

use std::cell::RefCell;

/// GPU バッファ（1つ分）。自動拡張対応。
struct GpuBuf {
    buf: CudaSlice<f32>,
    cap: usize,
}

/// CUDA matmul エンジン。
pub struct CudaMatmul {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    buf_a: RefCell<GpuBuf>,
    buf_b: RefCell<GpuBuf>,
    buf_c: RefCell<GpuBuf>,
}

impl CudaMatmul {
    /// CUDA デバイス 0 で初期化。
    #[must_use]
    pub fn new() -> Self {
        let ctx = CudaContext::new(0).expect("CUDA device 0 の初期化に失敗");
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).expect("cuBLAS 初期化に失敗");
        // 初期バッファ: 16M 要素 (~64MB) — 必要に応じて自動拡張
        let init_cap = 16 * 1024 * 1024;
        let ba = GpuBuf { buf: stream.alloc_zeros(init_cap).expect("GPU buf_a 確保失敗"), cap: init_cap };
        let bb = GpuBuf { buf: stream.alloc_zeros(init_cap).expect("GPU buf_b 確保失敗"), cap: init_cap };
        let bc = GpuBuf { buf: stream.alloc_zeros(init_cap).expect("GPU buf_c 確保失敗"), cap: init_cap };
        Self {
            stream,
            blas,
            buf_a: RefCell::new(ba),
            buf_b: RefCell::new(bb),
            buf_c: RefCell::new(bc),
        }
    }

    /// バッファを必要サイズに拡張。
    fn ensure_buf(stream: &Arc<CudaStream>, buf: &RefCell<GpuBuf>, need: usize) {
        let mut b = buf.borrow_mut();
        if need > b.cap {
            let new_cap = need.next_power_of_two();
            b.buf = stream.alloc_zeros(new_cap).expect("GPU バッファ拡張失敗");
            b.cap = new_cap;
        }
    }

    /// C\[m×n\] = A\[m×k\] × B\[n×k\]^T  (row-major)
    ///
    /// cuBLAS: C^T\[n×m\] = op(B_cm) × op(A_cm)
    /// B_rm\[n×k\] → B_cm\[k×n\] (ld=k). Transpose → B\[n×k\]. transa=T, lda=k.
    /// A_rm\[m×k\] → A_cm\[k×m\] (ld=k). No transpose → A^T\[k×m\]. transb=N, ldb=k.
    pub fn matmul_bt(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_b, n * k);
        Self::ensure_buf(&self.stream, &self.buf_c, m * n);

        // H2D 転送
        {
            let mut ba = self.buf_a.borrow_mut();
            self.stream.memcpy_htod(a, &mut ba.buf.slice_mut(..m * k)).expect("A→GPU 転送失敗");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream.memcpy_htod(b, &mut bb.buf.slice_mut(..n * k)).expect("B→GPU 転送失敗");
        }

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

        // gemm: 3つの RefCell を同時に borrow
        {
            let ba = self.buf_a.borrow();
            let bb = self.buf_b.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas
                    .gemm(cfg, &bb.buf.slice(..n * k), &ba.buf.slice(..m * k), &mut bc.buf.slice_mut(..m * n))
                    .expect("cuBLAS sgemm (matmul_bt) 失敗");
            }
        }

        let mut result = vec![0.0f32; m * n];
        let bc = self.buf_c.borrow();
        self.stream
            .memcpy_dtoh(&bc.buf.slice(..m * n), &mut result)
            .expect("GPU→CPU 転送失敗");
        result
    }

    /// C\[m×n\] = A\[m×k\] × B\[k×n\]  (row-major, 標準 matmul)
    ///
    /// cuBLAS: C^T\[n×m\] = B^T\[n×k\] × A^T\[k×m\]
    /// B_rm\[k×n\] → B_cm\[n×k\] (ld=n). No transpose. transa=N, lda=n.
    /// A_rm\[m×k\] → A_cm\[k×m\] (ld=k). No transpose. transb=N, ldb=k.
    pub fn matmul_nn(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_b, k * n);
        Self::ensure_buf(&self.stream, &self.buf_c, m * n);

        {
            let mut ba = self.buf_a.borrow_mut();
            self.stream.memcpy_htod(a, &mut ba.buf.slice_mut(..m * k)).expect("A→GPU 転送失敗");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream.memcpy_htod(b, &mut bb.buf.slice_mut(..k * n)).expect("B→GPU 転送失敗");
        }

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

        {
            let ba = self.buf_a.borrow();
            let bb = self.buf_b.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas
                    .gemm(cfg, &bb.buf.slice(..k * n), &ba.buf.slice(..m * k), &mut bc.buf.slice_mut(..m * n))
                    .expect("cuBLAS sgemm (matmul_nn) 失敗");
            }
        }

        let mut result = vec![0.0f32; m * n];
        let bc = self.buf_c.borrow();
        self.stream
            .memcpy_dtoh(&bc.buf.slice(..m * n), &mut result)
            .expect("GPU→CPU 転送失敗");
        result
    }

    /// C\[m×n\] = A\[m×k\] × B\[n×k\]^T — アロケーションレス版。
    /// 出力バッファ `c_out` を呼び出し側から受け取る。
    pub fn matmul_bt_inplace(&self, a: &[f32], b: &[f32], c_out: &mut [f32], m: usize, n: usize, k: usize) {
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_b, n * k);
        Self::ensure_buf(&self.stream, &self.buf_c, m * n);

        {
            let mut ba = self.buf_a.borrow_mut();
            self.stream.memcpy_htod(a, &mut ba.buf.slice_mut(..m * k)).expect("H2D a");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream.memcpy_htod(b, &mut bb.buf.slice_mut(..n * k)).expect("H2D b");
        }

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32, n: m as i32, k: k as i32,
            alpha: 1.0f32, lda: k as i32, ldb: k as i32, beta: 0.0f32, ldc: n as i32,
        };

        {
            let ba = self.buf_a.borrow();
            let bb = self.buf_b.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas.gemm(cfg, &bb.buf.slice(..n * k), &ba.buf.slice(..m * k), &mut bc.buf.slice_mut(..m * n)).unwrap();
            }
        }

        let bc = self.buf_c.borrow();
        self.stream.memcpy_dtoh(&bc.buf.slice(..m * n), c_out).expect("D2H c");
    }

    /// C\[m×n\] = A\[m×k\] × B\[k×n\] — アロケーションレス版。
    pub fn matmul_nn_inplace(&self, a: &[f32], b: &[f32], c_out: &mut [f32], m: usize, n: usize, k: usize) {
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_b, k * n);
        Self::ensure_buf(&self.stream, &self.buf_c, m * n);

        {
            let mut ba = self.buf_a.borrow_mut();
            self.stream.memcpy_htod(a, &mut ba.buf.slice_mut(..m * k)).expect("H2D a");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream.memcpy_htod(b, &mut bb.buf.slice_mut(..k * n)).expect("H2D b");
        }

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32, n: m as i32, k: k as i32,
            alpha: 1.0f32, lda: n as i32, ldb: k as i32, beta: 0.0f32, ldc: n as i32,
        };

        {
            let ba = self.buf_a.borrow();
            let bb = self.buf_b.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas.gemm(cfg, &bb.buf.slice(..k * n), &ba.buf.slice(..m * k), &mut bc.buf.slice_mut(..m * n)).unwrap();
            }
        }

        let bc = self.buf_c.borrow();
        self.stream.memcpy_dtoh(&bc.buf.slice(..m * n), c_out).expect("D2H c");
    }

    /// C\[k×n\] = A\[m×k\]^T × B\[m×n\] — アロケーションレス版。
    pub fn matmul_tn_inplace(&self, a: &[f32], b: &[f32], c_out: &mut [f32], m: usize, n: usize, k: usize) {
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_b, m * n);
        Self::ensure_buf(&self.stream, &self.buf_c, k * n);

        {
            let mut ba = self.buf_a.borrow_mut();
            self.stream.memcpy_htod(a, &mut ba.buf.slice_mut(..m * k)).expect("H2D a");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream.memcpy_htod(b, &mut bb.buf.slice_mut(..m * n)).expect("H2D b");
        }

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: n as i32, n: k as i32, k: m as i32,
            alpha: 1.0f32, lda: n as i32, ldb: k as i32, beta: 0.0f32, ldc: n as i32,
        };

        {
            let ba = self.buf_a.borrow();
            let bb = self.buf_b.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas.gemm(cfg, &bb.buf.slice(..m * n), &ba.buf.slice(..m * k), &mut bc.buf.slice_mut(..k * n)).unwrap();
            }
        }

        let bc = self.buf_c.borrow();
        self.stream.memcpy_dtoh(&bc.buf.slice(..k * n), c_out).expect("D2H c");
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

        // SiLU(gate) ⊙ up (CPU — rayon 並列)
        let total = seq_len * intermediate_dim;
        let mut gate_silu = vec![0.0f32; total];
        let mut intermediate = vec![0.0f32; total];
        gate_silu.par_iter_mut()
            .zip(intermediate.par_iter_mut())
            .zip(gate.par_iter().zip(up.par_iter()))
            .for_each(|((gs, im), (&g, &u))| {
                let s = g / (1.0 + (-g).exp());
                *gs = s;
                *im = s * u;
            });

        let output = self.matmul_bt(&intermediate, down_proj, seq_len, hidden_dim, intermediate_dim);
        (output, gate, up, gate_silu)
    }

    /// C[m×n] = A[m×k]^T × B[m×n]  (row-major)
    ///
    /// 実際は C[k×n] = A^T[k×m] × B[m×n] — 重み勾配計算用。
    /// cuBLAS: C^T[n×k] = B_cm[n×m] × trans(A_cm[k×m])
    /// B_rm[m×n] → B_cm[n×m] (ld=n). transa=N, lda=n.
    /// A_rm[m×k] → A_cm[k×m] (ld=k). transb=T → [m×k], ldb=k.
    pub fn matmul_tn(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        // A is [m×k], B is [m×n], output C is [k×n]
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_b, m * n);
        Self::ensure_buf(&self.stream, &self.buf_c, k * n);

        {
            let mut ba = self.buf_a.borrow_mut();
            self.stream.memcpy_htod(a, &mut ba.buf.slice_mut(..m * k)).expect("A→GPU 転送失敗");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream.memcpy_htod(b, &mut bb.buf.slice_mut(..m * n)).expect("B→GPU 転送失敗");
        }

        // C_cm[n×k] = B_cm[n×m] × trans(A_cm[k×m])
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: n as i32,
            n: k as i32,
            k: m as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };

        {
            let ba = self.buf_a.borrow();
            let bb = self.buf_b.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas
                    .gemm(cfg, &bb.buf.slice(..m * n), &ba.buf.slice(..m * k), &mut bc.buf.slice_mut(..k * n))
                    .expect("cuBLAS sgemm (matmul_tn) 失敗");
            }
        }

        // Result is C^T in col-major = C[k×n] in row-major
        let mut result = vec![0.0f32; k * n];
        let bc = self.buf_c.borrow();
        self.stream
            .memcpy_dtoh(&bc.buf.slice(..k * n), &mut result)
            .expect("GPU→CPU 転送失敗");
        result
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
use crate::llama_forward::{apply_rope, rmsnorm, LayerCache};

// ── CUDA GQA Attention ──────────────────────────────────────────────────

/// ヘッドデータ抽出: interleaved [seq_len, num_heads * head_dim] → [seq_len, head_dim]
fn extract_head(data: &[f32], head: usize, num_heads: usize, head_dim: usize, seq_len: usize) -> Vec<f32> {
    let stride = num_heads * head_dim;
    let mut out = vec![0.0f32; seq_len * head_dim];
    for t in 0..seq_len {
        let src = &data[t * stride + head * head_dim..t * stride + (head + 1) * head_dim];
        out[t * head_dim..(t + 1) * head_dim].copy_from_slice(src);
    }
    out
}

/// ヘッドデータ書き戻し: [seq_len, head_dim] → interleaved [seq_len, num_heads * head_dim]
fn scatter_head(src: &[f32], dst: &mut [f32], head: usize, num_heads: usize, head_dim: usize, seq_len: usize) {
    let stride = num_heads * head_dim;
    for t in 0..seq_len {
        dst[t * stride + head * head_dim..t * stride + (head + 1) * head_dim]
            .copy_from_slice(&src[t * head_dim..(t + 1) * head_dim]);
    }
}

/// ヘッドデータ加算: [seq_len, head_dim] を interleaved dst に加算（GQA 用）
fn accumulate_head(src: &[f32], dst: &mut [f32], head: usize, num_heads: usize, head_dim: usize, seq_len: usize) {
    let stride = num_heads * head_dim;
    for t in 0..seq_len {
        for d in 0..head_dim {
            dst[t * stride + head * head_dim + d] += src[t * head_dim + d];
        }
    }
}

/// Causal softmax (in-place): scores [seq_len, seq_len] に causal mask + softmax 適用
fn causal_softmax(scores: &mut [f32], seq_len: usize) {
    for t in 0..seq_len {
        let row = &mut scores[t * seq_len..(t + 1) * seq_len];
        // Causal mask: s > t → -inf
        for s in (t + 1)..seq_len {
            row[s] = f32::NEG_INFINITY;
        }
        // Stable softmax
        let max_val = row[..=t].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in 0..=t {
            let e = (row[s] - max_val).exp();
            row[s] = e;
            sum += e;
        }
        let inv = 1.0 / sum.max(1e-10);
        for s in 0..=t {
            row[s] *= inv;
        }
        for s in (t + 1)..seq_len {
            row[s] = 0.0;
        }
    }
}

/// Softmax backward (in-place): d_scores = attn * (d_attn_w - row_dot)
fn softmax_backward(d_attn_w: &[f32], attn: &[f32], seq_len: usize) -> Vec<f32> {
    let mut d_scores = vec![0.0f32; seq_len * seq_len];
    for t in 0..seq_len {
        let aw = &attn[t * seq_len..(t + 1) * seq_len];
        let dw = &d_attn_w[t * seq_len..(t + 1) * seq_len];
        // dot = sum(attn[t,s] * d_attn_w[t,s]) for s=0..=t
        let mut dot = 0.0f32;
        for s in 0..=t {
            dot += aw[s] * dw[s];
        }
        for s in 0..=t {
            d_scores[t * seq_len + s] = aw[s] * (dw[s] - dot);
        }
    }
    d_scores
}

/// CUDA GQA Attention forward。
///
/// 各ヘッドの QK^T と attn×V を cuBLAS で実行。
/// Causal softmax は CPU（要素演算のため軽量）。
pub fn cuda_gqa_attention(
    cuda: &CudaMatmul,
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
        let kv_h = h / kv_group_size;

        let q_h = extract_head(q, h, num_heads, head_dim, seq_len);
        let k_h = extract_head(k, kv_h, num_kv_heads, head_dim, seq_len);
        let v_h = extract_head(v, kv_h, num_kv_heads, head_dim, seq_len);

        // scores = Q_h × K_h^T: [seq_len, seq_len] — GPU matmul_bt
        let mut scores = cuda.matmul_bt(&q_h, &k_h, seq_len, seq_len, head_dim);

        // Scale
        for s in &mut scores {
            *s *= scale;
        }

        // Causal mask + softmax (CPU)
        causal_softmax(&mut scores, seq_len);

        // Store attn_weights
        attn_weights_out[h * seq_len * seq_len..(h + 1) * seq_len * seq_len]
            .copy_from_slice(&scores);

        // output_h = scores × V_h: [seq_len, head_dim] — GPU matmul_nn
        let out_h = cuda.matmul_nn(&scores, &v_h, seq_len, head_dim, seq_len);

        // Scatter into output
        scatter_head(&out_h, output, h, num_heads, head_dim, seq_len);
    }
}

/// CUDA GQA Attention backward。
///
/// 各ヘッドの勾配計算を cuBLAS で実行。
pub fn cuda_gqa_attention_backward(
    cuda: &CudaMatmul,
    d_attn_raw: &[f32],
    attn_weights: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_q: &mut [f32],
    d_k: &mut [f32],
    d_v: &mut [f32],
    config: &LlamaConfig,
    seq_len: usize,
) {
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let kv_group_size = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let kv_h = h / kv_group_size;

        let d_out_h = extract_head(d_attn_raw, h, num_heads, head_dim, seq_len);
        let v_h = extract_head(v, kv_h, num_kv_heads, head_dim, seq_len);
        let aw_h = &attn_weights[h * seq_len * seq_len..(h + 1) * seq_len * seq_len];

        // d_attn_w = d_out_h × V_h^T: [seq_len, seq_len] — GPU matmul_bt
        let d_attn_w = cuda.matmul_bt(&d_out_h, &v_h, seq_len, seq_len, head_dim);

        // d_V_h = attn^T × d_out_h: [seq_len, head_dim] — GPU matmul_tn
        let d_v_h = cuda.matmul_tn(aw_h, &d_out_h, seq_len, head_dim, seq_len);
        accumulate_head(&d_v_h, d_v, kv_h, num_kv_heads, head_dim, seq_len);

        // Softmax backward (CPU)
        let mut d_scores = softmax_backward(&d_attn_w, aw_h, seq_len);
        // Apply scale
        for s in &mut d_scores {
            *s *= scale;
        }

        // d_Q_h = d_scores × K_h: [seq_len, head_dim] — GPU matmul_nn
        let k_h = extract_head(k, kv_h, num_kv_heads, head_dim, seq_len);
        let d_q_h = cuda.matmul_nn(&d_scores, &k_h, seq_len, head_dim, seq_len);
        scatter_head(&d_q_h, d_q, h, num_heads, head_dim, seq_len);

        // d_K_h = d_scores^T × Q_h: [seq_len, head_dim] — GPU matmul_tn
        let q_h = extract_head(q, h, num_heads, head_dim, seq_len);
        let d_k_h = cuda.matmul_tn(&d_scores, &q_h, seq_len, head_dim, seq_len);
        accumulate_head(&d_k_h, d_k, kv_h, num_kv_heads, head_dim, seq_len);
    }
}

/// CUDA 加速版 Transformer レイヤー forward。
///
/// QKV/O projection, GQA Attention, SwiGLU FFN を cuBLAS sgemm で実行。
/// RMSNorm, RoPE は CPU。
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
    let mut v = cuda.matmul_bt(&normed, &weights.v_proj, seq_len, num_kv_heads * head_dim, hidden_dim);

    // 2b. Attention bias (Qwen2.5 等)
    if let Some(ref b) = weights.q_bias {
        crate::llama_forward::add_bias(&mut q, b, seq_len, num_heads * head_dim);
    }
    if let Some(ref b) = weights.k_bias {
        crate::llama_forward::add_bias(&mut k, b, seq_len, num_kv_heads * head_dim);
    }
    if let Some(ref b) = weights.v_bias {
        crate::llama_forward::add_bias(&mut v, b, seq_len, num_kv_heads * head_dim);
    }

    // 3. RoPE (CPU)
    apply_rope(&mut q, num_heads, head_dim, seq_len, config.rope_theta);
    apply_rope(&mut k, num_kv_heads, head_dim, seq_len, config.rope_theta);

    // 4. GQA Attention (CUDA — per-head matmul)
    let mut attn_out_raw = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut attn_weights = vec![0.0f32; num_heads * seq_len * seq_len];
    cuda_gqa_attention(cuda, &q, &k, &v, &mut attn_out_raw, &mut attn_weights, config, seq_len);

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

/// レイヤー重みの勾配。
pub struct LayerWeightGrads {
    /// Attention RMSNorm 勾配
    pub attn_norm: Vec<f32>,
    /// Q projection 勾配
    pub q_proj: Vec<f32>,
    /// K projection 勾配
    pub k_proj: Vec<f32>,
    /// V projection 勾配
    pub v_proj: Vec<f32>,
    /// O projection 勾配
    pub o_proj: Vec<f32>,
    /// Q bias 勾配 (attention_bias=true の場合のみ Some)
    pub q_bias: Option<Vec<f32>>,
    /// K bias 勾配
    pub k_bias: Option<Vec<f32>>,
    /// V bias 勾配
    pub v_bias: Option<Vec<f32>>,
    /// FFN RMSNorm 勾配
    pub ffn_norm: Vec<f32>,
    /// Gate projection 勾配
    pub gate_proj: Vec<f32>,
    /// Up projection 勾配
    pub up_proj: Vec<f32>,
    /// Down projection 勾配
    pub down_proj: Vec<f32>,
}

/// CUDA 加速版 Transformer レイヤー backward。
///
/// (d_input, LayerWeightGrads) を返す。
/// Projection backward の matmul を cuBLAS で実行。
/// RMSNorm, RoPE, Attention backward は CPU。
pub fn cuda_layer_backward(
    cuda: &CudaMatmul,
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
    let intermediate_dim = config.intermediate_dim;

    // ── FFN Backward ──

    // 1. SwiGLU FFN backward (CUDA matmul + CPU elementwise)
    //    down_proj backward: d_intermediate = d_output × down_proj (matmul_nn)
    let d_intermediate =
        cuda.matmul_nn(d_output, &weights.down_proj, seq_len, intermediate_dim, hidden_dim);

    // SwiGLU elementwise backward (CPU — rayon 並列)
    let total = seq_len * intermediate_dim;
    let mut d_gate = vec![0.0f32; total];
    let mut d_up = vec![0.0f32; total];
    d_gate.par_iter_mut()
        .zip(d_up.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (dg, du))| {
            let d_gate_silu = d_intermediate[idx] * cache.up[idx];
            *du = d_intermediate[idx] * cache.gate_silu[idx];
            let x = cache.gate[idx];
            let sig = 1.0 / (1.0 + (-x).exp());
            let silu_grad = sig * (1.0 + x * (1.0 - sig));
            *dg = d_gate_silu * silu_grad;
        });

    // gate/up_proj backward: d_input
    let d_input_gate =
        cuda.matmul_nn(&d_gate, &weights.gate_proj, seq_len, hidden_dim, intermediate_dim);
    let d_input_up = cuda.matmul_nn(&d_up, &weights.up_proj, seq_len, hidden_dim, intermediate_dim);
    let mut d_pre_ffn = d_input_gate;
    for idx in 0..d_pre_ffn.len() {
        d_pre_ffn[idx] += d_input_up[idx];
    }

    // SwiGLU intermediate for down_proj grad: silu(gate) ⊙ up
    let mut swiglu_out = vec![0.0f32; total];
    for idx in 0..total {
        swiglu_out[idx] = cache.gate_silu[idx] * cache.up[idx];
    }

    // FFN weight grads (matmul_tn: X^T × dY)
    // down_proj: [hidden×intermediate] ← d_output^T × swiglu_out
    let d_down_proj = cuda.matmul_tn(d_output, &swiglu_out, seq_len, intermediate_dim, hidden_dim);
    // gate_proj: [intermediate×hidden] ← d_gate^T × normed_ffn
    let d_gate_proj = cuda.matmul_tn(&d_gate, &cache.normed_ffn, seq_len, hidden_dim, intermediate_dim);
    // up_proj: [intermediate×hidden] ← d_up^T × normed_ffn
    let d_up_proj = cuda.matmul_tn(&d_up, &cache.normed_ffn, seq_len, hidden_dim, intermediate_dim);

    // 2. FFN RMSNorm backward (CPU)
    let mut d_pre_ffn_residual = vec![0.0f32; seq_len * hidden_dim];
    let mut d_ffn_norm_w = vec![0.0f32; hidden_dim];
    crate::llama_backward::rmsnorm_backward(
        &d_pre_ffn,
        &cache.residual_ffn,
        &weights.ffn_norm,
        &mut d_pre_ffn_residual,
        &mut d_ffn_norm_w,
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

    // 5. GQA Attention backward (CUDA — per-head matmul)
    let mut d_q = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut d_k = vec![0.0f32; seq_len * num_kv_heads * head_dim];
    let mut d_v = vec![0.0f32; seq_len * num_kv_heads * head_dim];

    cuda_gqa_attention_backward(
        cuda,
        &d_attn_raw,
        &cache.attn_weights,
        &cache.q,
        &cache.k,
        &cache.v,
        &mut d_q,
        &mut d_k,
        &mut d_v,
        config,
        seq_len,
    );

    // 6. RoPE backward (CPU — inverse rotation)
    crate::llama_backward::rope_backward(&mut d_q, num_heads, head_dim, seq_len, config.rope_theta);
    crate::llama_backward::rope_backward(
        &mut d_k,
        num_kv_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );

    // 7. QKV projection backward (CUDA) — d_input + weight grads
    let d_normed_q = cuda.matmul_nn(
        &d_q, &weights.q_proj, seq_len, hidden_dim, num_heads * head_dim,
    );
    let d_normed_k = cuda.matmul_nn(
        &d_k, &weights.k_proj, seq_len, hidden_dim, num_kv_heads * head_dim,
    );
    let d_normed_v = cuda.matmul_nn(
        &d_v, &weights.v_proj, seq_len, hidden_dim, num_kv_heads * head_dim,
    );

    let mut d_normed = d_normed_q;
    for i in 0..d_normed.len() {
        d_normed[i] += d_normed_k[i] + d_normed_v[i];
    }

    // O projection weight grad: o_proj[hidden×(heads*head_dim)] ← d_attn_output^T × attn_out_raw
    // attn_out_raw を attn_weights × V から再計算 (CUDA)
    let mut attn_out_raw = vec![0.0f32; seq_len * num_heads * head_dim];
    {
        let kv_group_size2 = num_heads / num_kv_heads;
        for h in 0..num_heads {
            let kv_h = h / kv_group_size2;
            let aw_h = &cache.attn_weights[h * seq_len * seq_len..(h + 1) * seq_len * seq_len];
            let v_h = extract_head(&cache.v, kv_h, num_kv_heads, head_dim, seq_len);
            let out_h = cuda.matmul_nn(aw_h, &v_h, seq_len, head_dim, seq_len);
            scatter_head(&out_h, &mut attn_out_raw, h, num_heads, head_dim, seq_len);
        }
    }
    let d_o_proj = cuda.matmul_tn(
        &d_attn_output, &attn_out_raw, seq_len, num_heads * head_dim, hidden_dim,
    );

    // QKV weight grads: W[out×in] ← d_out^T × normed_attn
    let d_q_proj = cuda.matmul_tn(
        &d_q, &cache.normed_attn, seq_len, hidden_dim, num_heads * head_dim,
    );
    let d_k_proj = cuda.matmul_tn(
        &d_k, &cache.normed_attn, seq_len, hidden_dim, num_kv_heads * head_dim,
    );
    let d_v_proj = cuda.matmul_tn(
        &d_v, &cache.normed_attn, seq_len, hidden_dim, num_kv_heads * head_dim,
    );

    // 8. Attention RMSNorm backward (CPU)
    let mut d_input = vec![0.0f32; seq_len * hidden_dim];
    let mut d_attn_norm_w = vec![0.0f32; hidden_dim];
    crate::llama_backward::rmsnorm_backward(
        &d_normed,
        &cache.residual_attn,
        &weights.attn_norm,
        &mut d_input,
        &mut d_attn_norm_w,
        hidden_dim,
        config.norm_eps,
    );

    // 9. Residual: d_input += d_attn_output
    for i in 0..d_input.len() {
        d_input[i] += d_attn_output[i];
    }

    // Bias 勾配: d_bias = sum(d_output, axis=0) — seq_len 軸で合計
    let compute_bias_grad = |d_out: &[f32], dim: usize| -> Vec<f32> {
        let mut d_bias = vec![0.0f32; dim];
        for t in 0..seq_len {
            for d in 0..dim {
                d_bias[d] += d_out[t * dim + d];
            }
        }
        d_bias
    };
    let d_q_bias = if weights.q_bias.is_some() {
        Some(compute_bias_grad(&d_q, num_heads * head_dim))
    } else {
        None
    };
    let d_k_bias = if weights.k_bias.is_some() {
        Some(compute_bias_grad(&d_k, num_kv_heads * head_dim))
    } else {
        None
    };
    let d_v_bias = if weights.v_bias.is_some() {
        Some(compute_bias_grad(&d_v, num_kv_heads * head_dim))
    } else {
        None
    };

    let grads = LayerWeightGrads {
        attn_norm: d_attn_norm_w,
        q_proj: d_q_proj,
        k_proj: d_k_proj,
        v_proj: d_v_proj,
        o_proj: d_o_proj,
        q_bias: d_q_bias,
        k_bias: d_k_bias,
        v_bias: d_v_bias,
        ffn_norm: d_ffn_norm_w,
        gate_proj: d_gate_proj,
        up_proj: d_up_proj,
        down_proj: d_down_proj,
    };

    (d_input, grads)
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
