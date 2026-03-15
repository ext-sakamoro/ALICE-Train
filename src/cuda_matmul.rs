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
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use rayon::prelude::*;
use std::sync::Arc;

use std::cell::RefCell;

/// GPU Causal Softmax + Scale カーネル (NVRTC)。
/// 256スレッド/ブロック、共有メモリリダクションで max/sum を並列計算。
const CAUSAL_SOFTMAX_CU: &str = r#"
#define BLOCK_SIZE 256
extern "C" __global__ void causal_softmax_scaled(
    float* scores,
    int seq_len,
    float scale)
{
    int row = blockIdx.x;
    if (row >= seq_len) return;

    int tid = threadIdx.x;
    int valid_len = row + 1;
    float* row_ptr = scores + row * seq_len;

    __shared__ float sdata[BLOCK_SIZE];

    /* --- Scale + Max --- */
    float local_max = -1e30f;
    for (int s = tid; s < valid_len; s += BLOCK_SIZE) {
        row_ptr[s] *= scale;
        float v = row_ptr[s];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = sdata[tid + stride];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    /* --- Exp + Sum --- */
    float local_sum = 0.0f;
    for (int s = tid; s < valid_len; s += BLOCK_SIZE) {
        float e = expf(row_ptr[s] - max_val);
        row_ptr[s] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];
    __syncthreads();

    /* --- Normalize --- */
    float inv_sum = 1.0f / (sum_val + 1e-10f);
    for (int s = tid; s < valid_len; s += BLOCK_SIZE) {
        row_ptr[s] *= inv_sum;
    }

    /* --- Causal mask: zero upper triangle --- */
    for (int s = valid_len + tid; s < seq_len; s += BLOCK_SIZE) {
        row_ptr[s] = 0.0f;
    }
}
"#;

/// GPU バッファ（1つ分）。自動拡張対応。
struct GpuBuf {
    buf: CudaSlice<f32>,
    cap: usize,
}

/// CUDA matmul エンジン。
pub struct CudaMatmul {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    softmax_func: CudaFunction,
    buf_a: RefCell<GpuBuf>,
    buf_b: RefCell<GpuBuf>,
    buf_c: RefCell<GpuBuf>,
    /// GPU 上の scores バッファ（Attention 用、D2H なしで保持）。
    buf_scores: RefCell<GpuBuf>,
}

impl CudaMatmul {
    /// CUDA デバイス 0 で初期化。
    #[must_use]
    pub fn new() -> Self {
        let ctx = CudaContext::new(0).expect("CUDA device 0 の初期化に失敗");
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).expect("cuBLAS 初期化に失敗");

        // 【究極チューン4】TF32 Tensor Cores 覚醒
        // Pascal (P5000) ではフォールバック、Ampere以降で自動的にTensor Coreが起動
        unsafe {
            cudarc::cublas::sys::cublasSetMathMode(
                *blas.handle(),
                cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
            );
        }

        // 【究極チューン5】GPU Causal Softmax カーネルを NVRTC でコンパイル
        let ptx = compile_ptx(CAUSAL_SOFTMAX_CU).expect("NVRTC: causal_softmax カーネルのコンパイルに失敗");
        let module = ctx.load_module(ptx).expect("CUDA module ロードに失敗");
        let softmax_func = module.load_function("causal_softmax_scaled").expect("causal_softmax_scaled 関数ロードに失敗");

        // 初期バッファ: 16M 要素 (~64MB) — 必要に応じて自動拡張
        let init_cap = 16 * 1024 * 1024;
        let ba = GpuBuf { buf: stream.alloc_zeros(init_cap).expect("GPU buf_a 確保失敗"), cap: init_cap };
        let bb = GpuBuf { buf: stream.alloc_zeros(init_cap).expect("GPU buf_b 確保失敗"), cap: init_cap };
        let bc = GpuBuf { buf: stream.alloc_zeros(init_cap).expect("GPU buf_c 確保失敗"), cap: init_cap };
        let bs = GpuBuf { buf: stream.alloc_zeros(init_cap).expect("GPU buf_scores 確保失敗"), cap: init_cap };
        Self {
            stream,
            blas,
            softmax_func,
            buf_a: RefCell::new(ba),
            buf_b: RefCell::new(bb),
            buf_c: RefCell::new(bc),
            buf_scores: RefCell::new(bs),
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

    /// CPU 配列を VRAM にアップロードし、GPU 上のスライスを返す。
    /// レイヤー単位で1回だけ呼び、以降はバッチ間で使い回す。
    pub fn upload_weight(&self, cpu: &[f32]) -> CudaSlice<f32> {
        let mut gpu: CudaSlice<f32> = self.stream.alloc_zeros(cpu.len()).expect("VRAM weight 確保失敗");
        self.stream.memcpy_htod(cpu, &mut gpu).expect("weight H2D 転送失敗");
        gpu
    }

    /// C\[m×n\] = A\[m×k\] × B\[n×k\]^T — B が既に VRAM 上にあるバージョン。
    /// 重みの H2D 転送をスキップし、A(hidden) のみ転送する。
    pub fn matmul_bt_with_gpu_b(
        &self,
        a: &[f32],
        b_gpu: &CudaSlice<f32>,
        c_out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_c, m * n);

        {
            let mut ba = self.buf_a.borrow_mut();
            self.stream.memcpy_htod(a, &mut ba.buf.slice_mut(..m * k)).expect("H2D a");
        }

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32, n: m as i32, k: k as i32,
            alpha: 1.0f32, lda: k as i32, ldb: k as i32, beta: 0.0f32, ldc: n as i32,
        };

        {
            let ba = self.buf_a.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas.gemm(
                    cfg,
                    &b_gpu.slice(..n * k),
                    &ba.buf.slice(..m * k),
                    &mut bc.buf.slice_mut(..m * n),
                ).expect("cuBLAS sgemm (matmul_bt_with_gpu_b) 失敗");
            }
        }

        let bc = self.buf_c.borrow();
        self.stream.memcpy_dtoh(&bc.buf.slice(..m * n), c_out).expect("D2H c");
    }

    /// C\[m×n\] = A\[m×k\] × B\[k×n\] — B が既に VRAM 上にあるバージョン。
    /// backward の d_input 計算（重み×勾配）で使用。
    pub fn matmul_nn_with_gpu_b(
        &self,
        a: &[f32],
        b_gpu: &CudaSlice<f32>,
        c_out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_c, m * n);

        {
            let mut ba = self.buf_a.borrow_mut();
            self.stream.memcpy_htod(a, &mut ba.buf.slice_mut(..m * k)).expect("H2D a");
        }

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32, n: m as i32, k: k as i32,
            alpha: 1.0f32, lda: n as i32, ldb: k as i32, beta: 0.0f32, ldc: n as i32,
        };

        {
            let ba = self.buf_a.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas.gemm(
                    cfg,
                    &b_gpu.slice(..k * n),
                    &ba.buf.slice(..m * k),
                    &mut bc.buf.slice_mut(..m * n),
                ).expect("cuBLAS sgemm (matmul_nn_with_gpu_b) 失敗");
            }
        }

        let bc = self.buf_c.borrow();
        self.stream.memcpy_dtoh(&bc.buf.slice(..m * n), c_out).expect("D2H c");
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

    /// C\[m×n\] = A\[m×k\] × B\[n×k\]^T — 結果を GPU 上 `buf_scores` に保持（D2H なし）。
    /// Attention の scores 計算用: Q×K^T の結果を GPU に留めたまま softmax → matmul_nn に流す。
    pub fn matmul_bt_to_scores(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_b, n * k);
        Self::ensure_buf(&self.stream, &self.buf_scores, m * n);

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
            let mut bs = self.buf_scores.borrow_mut();
            unsafe {
                self.blas
                    .gemm(cfg, &bb.buf.slice(..n * k), &ba.buf.slice(..m * k), &mut bs.buf.slice_mut(..m * n))
                    .expect("cuBLAS sgemm (matmul_bt_to_scores) 失敗");
            }
        }
    }

    /// GPU 上の `buf_scores` に対して scale + causal mask + softmax を実行。
    /// PCIe 転送ゼロ — 全て GPU 上で完結。
    pub fn gpu_scale_causal_softmax_scores(&self, seq_len: usize, scale: f32) {
        let seq_len_i32 = seq_len as i32;
        let cfg = LaunchConfig {
            grid_dim: (seq_len as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bs = self.buf_scores.borrow_mut();
        unsafe {
            self.stream.launch_builder(&self.softmax_func)
                .arg(&mut bs.buf)
                .arg(&seq_len_i32)
                .arg(&scale)
                .launch(cfg)
                .expect("causal_softmax GPU カーネル起動失敗");
        }
    }

    /// GPU 上の `buf_scores` から CPU にダウンロード。
    /// Attention backward 用に attn_weights を保存するための D2H。
    pub fn dtoh_scores(&self, dst: &mut [f32], size: usize) {
        let bs = self.buf_scores.borrow();
        self.stream
            .memcpy_dtoh(&bs.buf.slice(..size), dst)
            .expect("D2H scores");
    }

    /// C\[m×n\] = buf_scores\[m×k\] × B\[k×n\] — A(scores) は GPU 常駐、B のみ H2D。
    /// Attention の output = softmax(scores) × V の計算用。
    pub fn matmul_nn_from_scores(&self, b: &[f32], c_out: &mut [f32], m: usize, n: usize, k: usize) {
        Self::ensure_buf(&self.stream, &self.buf_b, k * n);
        Self::ensure_buf(&self.stream, &self.buf_c, m * n);

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
            let bb = self.buf_b.borrow();
            let bs = self.buf_scores.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas
                    .gemm(cfg, &bb.buf.slice(..k * n), &bs.buf.slice(..m * k), &mut bc.buf.slice_mut(..m * n))
                    .expect("cuBLAS sgemm (matmul_nn_from_scores) 失敗");
            }
        }

        let bc = self.buf_c.borrow();
        self.stream.memcpy_dtoh(&bc.buf.slice(..m * n), c_out).expect("D2H c");
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
                let s = g * crate::fast_math::fast_sigmoid(g);
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
            let sig = crate::fast_math::fast_sigmoid(x);
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

use crate::fast_math::fast_exp;
use crate::llama::{LlamaConfig, LlamaLayerWeights};
use crate::llama_forward::{apply_rope, rmsnorm, LayerCache};

// ── グローバルワークスペース ─────────────────────────────────────────────

/// 全レイヤーで使い回す、アロケーションゼロ化のためのワークスペース。
/// 起動時に1回だけ確保し、以降は `&mut` で使い回す。
/// Forward / Backward 両方の一時バッファを含む。
pub struct CudaLayerWorkspace {
    // --- Forward 中間バッファ ---
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out_raw: Vec<f32>,
    pub attn_weights: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub normed_ffn: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub gate_silu: Vec<f32>,
    pub intermediate: Vec<f32>,
    pub ffn_out: Vec<f32>,

    // --- Backward 中間バッファ ---
    pub d_intermediate: Vec<f32>,
    pub d_gate: Vec<f32>,
    pub d_up: Vec<f32>,
    pub d_input_gate: Vec<f32>,
    pub d_input_up: Vec<f32>,
    pub swiglu_out: Vec<f32>,
    pub d_pre_ffn: Vec<f32>,
    pub d_pre_ffn_residual: Vec<f32>,
    pub d_attn_output: Vec<f32>,
    pub d_attn_raw: Vec<f32>,
    pub d_q: Vec<f32>,
    pub d_k: Vec<f32>,
    pub d_v: Vec<f32>,
    pub d_normed_q: Vec<f32>,
    pub d_normed_k: Vec<f32>,
    pub d_normed_v: Vec<f32>,
    pub d_normed: Vec<f32>,
    pub attn_out_raw_recompute: Vec<f32>,
    pub d_input: Vec<f32>,
    pub d_attn_norm_w: Vec<f32>,
    pub d_ffn_norm_w: Vec<f32>,
}

impl CudaLayerWorkspace {
    /// 起動時に最大シーケンス長に合わせて事前確保する。
    #[must_use]
    pub fn new(config: &LlamaConfig, max_seq_len: usize) -> Self {
        let hidden = config.hidden_dim;
        let heads = config.num_heads;
        let kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let inter = config.intermediate_dim;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;

        let ws = Self {
            // Forward
            normed: vec![0.0; max_seq_len * hidden],
            q: vec![0.0; max_seq_len * q_dim],
            k: vec![0.0; max_seq_len * kv_dim],
            v: vec![0.0; max_seq_len * kv_dim],
            attn_out_raw: vec![0.0; max_seq_len * q_dim],
            attn_weights: vec![0.0; heads * max_seq_len * max_seq_len],
            attn_out: vec![0.0; max_seq_len * hidden],
            normed_ffn: vec![0.0; max_seq_len * hidden],
            gate: vec![0.0; max_seq_len * inter],
            up: vec![0.0; max_seq_len * inter],
            gate_silu: vec![0.0; max_seq_len * inter],
            intermediate: vec![0.0; max_seq_len * inter],
            ffn_out: vec![0.0; max_seq_len * hidden],
            // Backward
            d_intermediate: vec![0.0; max_seq_len * inter],
            d_gate: vec![0.0; max_seq_len * inter],
            d_up: vec![0.0; max_seq_len * inter],
            d_input_gate: vec![0.0; max_seq_len * hidden],
            d_input_up: vec![0.0; max_seq_len * hidden],
            swiglu_out: vec![0.0; max_seq_len * inter],
            d_pre_ffn: vec![0.0; max_seq_len * hidden],
            d_pre_ffn_residual: vec![0.0; max_seq_len * hidden],
            d_attn_output: vec![0.0; max_seq_len * hidden],
            d_attn_raw: vec![0.0; max_seq_len * q_dim],
            d_q: vec![0.0; max_seq_len * q_dim],
            d_k: vec![0.0; max_seq_len * kv_dim],
            d_v: vec![0.0; max_seq_len * kv_dim],
            d_normed_q: vec![0.0; max_seq_len * hidden],
            d_normed_k: vec![0.0; max_seq_len * hidden],
            d_normed_v: vec![0.0; max_seq_len * hidden],
            d_normed: vec![0.0; max_seq_len * hidden],
            attn_out_raw_recompute: vec![0.0; max_seq_len * q_dim],
            d_input: vec![0.0; max_seq_len * hidden],
            d_attn_norm_w: vec![0.0; hidden],
            d_ffn_norm_w: vec![0.0; hidden],
        };

        // Pinned Memory 化: mlock で全バッファをページロック。
        // PCIe DMA 直結になり、ドライバの隠れた CPU コピーが消滅。
        #[cfg(unix)]
        {
            let lock = |buf: &[f32]| {
                unsafe {
                    libc::mlock(buf.as_ptr().cast::<libc::c_void>(), buf.len() * 4);
                }
            };
            lock(&ws.normed);
            lock(&ws.q);
            lock(&ws.k);
            lock(&ws.v);
            lock(&ws.attn_out_raw);
            lock(&ws.attn_out);
            lock(&ws.normed_ffn);
            lock(&ws.gate);
            lock(&ws.up);
            lock(&ws.intermediate);
            lock(&ws.ffn_out);
            lock(&ws.d_intermediate);
            lock(&ws.d_gate);
            lock(&ws.d_up);
            lock(&ws.d_input_gate);
            lock(&ws.d_input_up);
            lock(&ws.d_pre_ffn);
            lock(&ws.d_attn_output);
            lock(&ws.d_attn_raw);
            lock(&ws.d_q);
            lock(&ws.d_k);
            lock(&ws.d_v);
            lock(&ws.d_normed);
            lock(&ws.d_input);
        }

        ws
    }

    /// ワークスペースの事前確保サイズ（バイト）を返す。
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let count = self.normed.len() + self.q.len() + self.k.len() + self.v.len()
            + self.attn_out_raw.len() + self.attn_weights.len() + self.attn_out.len()
            + self.normed_ffn.len() + self.gate.len() + self.up.len()
            + self.gate_silu.len() + self.intermediate.len() + self.ffn_out.len()
            + self.d_intermediate.len() + self.d_gate.len() + self.d_up.len()
            + self.d_input_gate.len() + self.d_input_up.len() + self.swiglu_out.len()
            + self.d_pre_ffn.len() + self.d_pre_ffn_residual.len()
            + self.d_attn_output.len() + self.d_attn_raw.len()
            + self.d_q.len() + self.d_k.len() + self.d_v.len()
            + self.d_normed_q.len() + self.d_normed_k.len() + self.d_normed_v.len()
            + self.d_normed.len() + self.attn_out_raw_recompute.len()
            + self.d_input.len() + self.d_attn_norm_w.len() + self.d_ffn_norm_w.len();
        count * 4
    }
}

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
    scores.par_chunks_exact_mut(seq_len).enumerate().for_each(|(t, row)| {
        // Causal mask: s > t → -inf
        for s in (t + 1)..seq_len {
            row[s] = f32::NEG_INFINITY;
        }
        // Stable softmax with fast_exp
        let max_val = row[..=t].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in 0..=t {
            let e = fast_exp(row[s] - max_val);
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
    });
}

/// Softmax backward (in-place): d_scores = attn * (d_attn_w - row_dot)
fn softmax_backward(d_attn_w: &[f32], attn: &[f32], seq_len: usize) -> Vec<f32> {
    let mut d_scores = vec![0.0f32; seq_len * seq_len];
    d_scores.par_chunks_exact_mut(seq_len).enumerate().for_each(|(t, score_row)| {
        let aw = &attn[t * seq_len..(t + 1) * seq_len];
        let dw = &d_attn_w[t * seq_len..(t + 1) * seq_len];
        let mut dot = 0.0f32;
        for s in 0..=t {
            dot += aw[s] * dw[s];
        }
        for s in 0..=t {
            score_row[s] = aw[s] * (dw[s] - dot);
        }
    });
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

    let scores_size = seq_len * seq_len;

    for h in 0..num_heads {
        let kv_h = h / kv_group_size;

        let q_h = extract_head(q, h, num_heads, head_dim, seq_len);
        let k_h = extract_head(k, kv_h, num_kv_heads, head_dim, seq_len);
        let v_h = extract_head(v, kv_h, num_kv_heads, head_dim, seq_len);

        // 【究極チューン5】scores を GPU 上に留めたまま softmax → matmul を完結
        // Step 1: Q_h × K_h^T → buf_scores (GPU 常駐、D2H なし)
        cuda.matmul_bt_to_scores(&q_h, &k_h, seq_len, seq_len, head_dim);

        // Step 2: Scale + Causal Mask + Softmax — 全て GPU カーネルで実行
        cuda.gpu_scale_causal_softmax_scores(seq_len, scale);

        // Step 3: attn_weights_out に D2H (backward 用に保存 — 唯一の必要な転送)
        cuda.dtoh_scores(
            &mut attn_weights_out[h * scores_size..(h + 1) * scores_size],
            scores_size,
        );

        // Step 4: output_h = scores × V_h — scores は GPU 上、V_h のみ H2D
        let mut out_h = vec![0.0f32; seq_len * head_dim];
        cuda.matmul_nn_from_scores(&v_h, &mut out_h, seq_len, head_dim, seq_len);

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

/// Eval 専用 forward — `LayerCache` を生成せず `input` を in-place 更新するだけ。
/// backward 不要なので中間値を保持しない。メモリ使用量: ~1層分の matmul バッファのみ。
pub fn cuda_layer_forward_eval(
    cuda: &CudaMatmul,
    input: &mut Vec<f32>,
    weights: &LlamaLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
) {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_dim = config.intermediate_dim;

    // 1. 残差保存 + Attention RMSNorm (CPU)
    let residual_attn = input.clone();
    let mut normed = input.clone();
    rmsnorm(&mut normed, &weights.attn_norm, hidden_dim, config.norm_eps);

    // 2. QKV projection (CUDA)
    let mut q = cuda.matmul_bt(&normed, &weights.q_proj, seq_len, num_heads * head_dim, hidden_dim);
    let mut k = cuda.matmul_bt(&normed, &weights.k_proj, seq_len, num_kv_heads * head_dim, hidden_dim);
    let mut v = cuda.matmul_bt(&normed, &weights.v_proj, seq_len, num_kv_heads * head_dim, hidden_dim);
    drop(normed);

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
    let mut attn_weights_buf = vec![0.0f32; num_heads * seq_len * seq_len];
    cuda_gqa_attention(cuda, &q, &k, &v, &mut attn_out_raw, &mut attn_weights_buf, config, seq_len);
    drop(q);
    drop(k);
    drop(v);
    drop(attn_weights_buf);

    // 5. O projection (CUDA)
    let attn_out = cuda.matmul_bt(&attn_out_raw, &weights.o_proj, seq_len, hidden_dim, num_heads * head_dim);
    drop(attn_out_raw);

    // 6. Residual add
    for i in 0..input.len() {
        input[i] = residual_attn[i] + attn_out[i];
    }
    drop(residual_attn);
    drop(attn_out);

    // 7. FFN: 残差保存 + RMSNorm (CPU)
    let residual_ffn = input.clone();
    let mut normed_ffn_buf = input.clone();
    rmsnorm(&mut normed_ffn_buf, &weights.ffn_norm, hidden_dim, config.norm_eps);

    // 8. SwiGLU FFN (CUDA) — eval では中間値不要
    let gate = cuda.matmul_bt(&normed_ffn_buf, &weights.gate_proj, seq_len, intermediate_dim, hidden_dim);
    let up = cuda.matmul_bt(&normed_ffn_buf, &weights.up_proj, seq_len, intermediate_dim, hidden_dim);
    drop(normed_ffn_buf);

    let total = seq_len * intermediate_dim;
    let mut intermediate = vec![0.0f32; total];
    intermediate.par_iter_mut()
        .zip(gate.par_iter().zip(up.par_iter()))
        .for_each(|(im, (&g, &u))| {
            let s = g * crate::fast_math::fast_sigmoid(g);
            *im = s * u;
        });
    drop(gate);
    drop(up);

    let ffn_out = cuda.matmul_bt(&intermediate, &weights.down_proj, seq_len, hidden_dim, intermediate_dim);
    drop(intermediate);

    // 9. Residual add
    for i in 0..input.len() {
        input[i] = residual_ffn[i] + ffn_out[i];
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

impl LayerWeightGrads {
    /// ゼロ初期化の勾配バッファを生成。勾配累積用。
    #[must_use]
    pub fn zeros(config: &LlamaConfig) -> Self {
        let hidden = config.hidden_dim;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let inter = config.intermediate_dim;
        Self {
            attn_norm: vec![0.0; hidden],
            q_proj: vec![0.0; q_dim * hidden],
            k_proj: vec![0.0; kv_dim * hidden],
            v_proj: vec![0.0; kv_dim * hidden],
            o_proj: vec![0.0; hidden * q_dim],
            q_bias: if config.attention_bias { Some(vec![0.0; q_dim]) } else { None },
            k_bias: if config.attention_bias { Some(vec![0.0; kv_dim]) } else { None },
            v_bias: if config.attention_bias { Some(vec![0.0; kv_dim]) } else { None },
            ffn_norm: vec![0.0; hidden],
            gate_proj: vec![0.0; inter * hidden],
            up_proj: vec![0.0; inter * hidden],
            down_proj: vec![0.0; hidden * inter],
        }
    }

    /// 別の勾配を要素ごとに加算（勾配累積）。Rayon並列。
    pub fn accumulate(&mut self, other: &Self) {
        let add = |dst: &mut [f32], src: &[f32]| {
            dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d += *s);
        };
        add(&mut self.attn_norm, &other.attn_norm);
        add(&mut self.q_proj, &other.q_proj);
        add(&mut self.k_proj, &other.k_proj);
        add(&mut self.v_proj, &other.v_proj);
        add(&mut self.o_proj, &other.o_proj);
        add(&mut self.ffn_norm, &other.ffn_norm);
        add(&mut self.gate_proj, &other.gate_proj);
        add(&mut self.up_proj, &other.up_proj);
        add(&mut self.down_proj, &other.down_proj);
        if let (Some(ref mut d), Some(ref s)) = (&mut self.q_bias, &other.q_bias) {
            add(d, s);
        }
        if let (Some(ref mut d), Some(ref s)) = (&mut self.k_bias, &other.k_bias) {
            add(d, s);
        }
        if let (Some(ref mut d), Some(ref s)) = (&mut self.v_bias, &other.v_bias) {
            add(d, s);
        }
    }

    /// 全要素をゼロにリセット。Rayon並列。
    pub fn zero_out(&mut self) {
        self.attn_norm.par_iter_mut().for_each(|x| *x = 0.0);
        self.q_proj.par_iter_mut().for_each(|x| *x = 0.0);
        self.k_proj.par_iter_mut().for_each(|x| *x = 0.0);
        self.v_proj.par_iter_mut().for_each(|x| *x = 0.0);
        self.o_proj.par_iter_mut().for_each(|x| *x = 0.0);
        self.ffn_norm.par_iter_mut().for_each(|x| *x = 0.0);
        self.gate_proj.par_iter_mut().for_each(|x| *x = 0.0);
        self.up_proj.par_iter_mut().for_each(|x| *x = 0.0);
        self.down_proj.par_iter_mut().for_each(|x| *x = 0.0);
        if let Some(ref mut b) = self.q_bias { b.par_iter_mut().for_each(|x| *x = 0.0); }
        if let Some(ref mut b) = self.k_bias { b.par_iter_mut().for_each(|x| *x = 0.0); }
        if let Some(ref mut b) = self.v_bias { b.par_iter_mut().for_each(|x| *x = 0.0); }
    }
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
            let sig = crate::fast_math::fast_sigmoid(x);
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

// ── Workspace 版 Forward / Backward（アロケーションゼロ） ─────────────

/// Workspace 版 eval forward — 動的アロケーション完全排除。
/// `CudaLayerWorkspace` のバッファのみ使用し、`input` を in-place 更新。
pub fn cuda_layer_forward_eval_ws(
    cuda: &CudaMatmul,
    input: &mut [f32],
    weights: &LlamaLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
    ws: &mut CudaLayerWorkspace,
) {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_dim = config.intermediate_dim;
    let hid_len = seq_len * hidden_dim;

    // 1. 残差保存(input→ws.ffn_out を一時residualとして流用) + RMSNorm
    ws.ffn_out[..hid_len].copy_from_slice(&input[..hid_len]); // residual_attn
    ws.normed[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(&mut ws.normed[..hid_len], &weights.attn_norm, hidden_dim, config.norm_eps);

    // 2. QKV projection (CUDA inplace)
    cuda.matmul_bt_inplace(&ws.normed[..hid_len], &weights.q_proj, &mut ws.q, seq_len, num_heads * head_dim, hidden_dim);
    cuda.matmul_bt_inplace(&ws.normed[..hid_len], &weights.k_proj, &mut ws.k, seq_len, num_kv_heads * head_dim, hidden_dim);
    cuda.matmul_bt_inplace(&ws.normed[..hid_len], &weights.v_proj, &mut ws.v, seq_len, num_kv_heads * head_dim, hidden_dim);

    // 2b. Attention bias
    if let Some(ref b) = weights.q_bias {
        crate::llama_forward::add_bias(&mut ws.q, b, seq_len, num_heads * head_dim);
    }
    if let Some(ref b) = weights.k_bias {
        crate::llama_forward::add_bias(&mut ws.k, b, seq_len, num_kv_heads * head_dim);
    }
    if let Some(ref b) = weights.v_bias {
        crate::llama_forward::add_bias(&mut ws.v, b, seq_len, num_kv_heads * head_dim);
    }

    // 3. RoPE (CPU)
    apply_rope(&mut ws.q, num_heads, head_dim, seq_len, config.rope_theta);
    apply_rope(&mut ws.k, num_kv_heads, head_dim, seq_len, config.rope_theta);

    // 4. GQA Attention (CUDA)
    cuda_gqa_attention(cuda, &ws.q, &ws.k, &ws.v, &mut ws.attn_out_raw, &mut ws.attn_weights, config, seq_len);

    // 5. O projection (CUDA inplace)
    cuda.matmul_bt_inplace(&ws.attn_out_raw, &weights.o_proj, &mut ws.attn_out, seq_len, hidden_dim, num_heads * head_dim);

    // 6. Residual add: input = residual_attn(ws.ffn_out) + attn_out
    for i in 0..hid_len {
        input[i] = ws.ffn_out[i] + ws.attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm
    ws.ffn_out[..hid_len].copy_from_slice(&input[..hid_len]); // residual_ffn
    ws.normed_ffn[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(&mut ws.normed_ffn[..hid_len], &weights.ffn_norm, hidden_dim, config.norm_eps);

    // 8. SwiGLU FFN (CUDA inplace)
    let total = seq_len * intermediate_dim;
    cuda.matmul_bt_inplace(&ws.normed_ffn[..hid_len], &weights.gate_proj, &mut ws.gate[..total], seq_len, intermediate_dim, hidden_dim);
    cuda.matmul_bt_inplace(&ws.normed_ffn[..hid_len], &weights.up_proj, &mut ws.up[..total], seq_len, intermediate_dim, hidden_dim);

    ws.intermediate[..total].par_iter_mut()
        .zip(ws.gate[..total].par_iter().zip(ws.up[..total].par_iter()))
        .for_each(|(im, (&g, &u))| {
            let s = g * crate::fast_math::fast_sigmoid(g);
            *im = s * u;
        });

    cuda.matmul_bt_inplace(&ws.intermediate[..total], &weights.down_proj, &mut ws.attn_out[..hid_len], seq_len, hidden_dim, intermediate_dim);
    // ws.attn_out をFFN出力として流用

    // 9. Residual add: input = residual_ffn(ws.ffn_out) + ffn_out(ws.attn_out)
    for i in 0..hid_len {
        input[i] = ws.ffn_out[i] + ws.attn_out[i];
    }
}

/// レイヤー重みの VRAM 常駐版。レイヤー処理の先頭で1回だけ転送し、
/// 全バッチで GPU 上の重みを直接参照する。PCIe 転送を 1/N_batch に削減。
pub struct VramLayerWeights {
    /// Q projection (VRAM)
    pub q_proj: CudaSlice<f32>,
    /// K projection (VRAM)
    pub k_proj: CudaSlice<f32>,
    /// V projection (VRAM)
    pub v_proj: CudaSlice<f32>,
    /// O projection (VRAM)
    pub o_proj: CudaSlice<f32>,
    /// Gate projection (VRAM)
    pub gate_proj: CudaSlice<f32>,
    /// Up projection (VRAM)
    pub up_proj: CudaSlice<f32>,
    /// Down projection (VRAM)
    pub down_proj: CudaSlice<f32>,
}

impl VramLayerWeights {
    /// CPU の `LlamaLayerWeights` から projection 重みを VRAM にアップロード。
    /// RMSNorm weight / bias は CPU 側で使うため転送しない。
    pub fn upload(cuda: &CudaMatmul, w: &LlamaLayerWeights) -> Self {
        Self {
            q_proj: cuda.upload_weight(&w.q_proj),
            k_proj: cuda.upload_weight(&w.k_proj),
            v_proj: cuda.upload_weight(&w.v_proj),
            o_proj: cuda.upload_weight(&w.o_proj),
            gate_proj: cuda.upload_weight(&w.gate_proj),
            up_proj: cuda.upload_weight(&w.up_proj),
            down_proj: cuda.upload_weight(&w.down_proj),
        }
    }

    /// VRAM 使用量 (バイト) を返す。
    #[must_use]
    pub fn vram_bytes(&self) -> usize {
        (self.q_proj.len() + self.k_proj.len() + self.v_proj.len()
            + self.o_proj.len() + self.gate_proj.len()
            + self.up_proj.len() + self.down_proj.len()) * 4
    }
}

/// VRAM 常駐重み版 Eval forward。重みの H2D 転送をスキップし、
/// hidden 状態のみ CPU↔GPU 間で転送する。
pub fn cuda_layer_forward_eval_ws_vram(
    cuda: &CudaMatmul,
    input: &mut [f32],
    cpu_weights: &LlamaLayerWeights,
    vram: &VramLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
    ws: &mut CudaLayerWorkspace,
) {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_dim = config.intermediate_dim;
    let hid_len = seq_len * hidden_dim;

    // 1. 残差保存 + RMSNorm (CPU — attn_norm は小さいので CPU 側で十分)
    ws.ffn_out[..hid_len].copy_from_slice(&input[..hid_len]);
    ws.normed[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(&mut ws.normed[..hid_len], &cpu_weights.attn_norm, hidden_dim, config.norm_eps);

    // 2. QKV projection — VRAM 上の重みを直接使用 (H2D 重み転送ゼロ)
    cuda.matmul_bt_with_gpu_b(&ws.normed[..hid_len], &vram.q_proj, &mut ws.q, seq_len, num_heads * head_dim, hidden_dim);
    cuda.matmul_bt_with_gpu_b(&ws.normed[..hid_len], &vram.k_proj, &mut ws.k, seq_len, num_kv_heads * head_dim, hidden_dim);
    cuda.matmul_bt_with_gpu_b(&ws.normed[..hid_len], &vram.v_proj, &mut ws.v, seq_len, num_kv_heads * head_dim, hidden_dim);

    // 2b. Attention bias (CPU — bias は小さい)
    if let Some(ref b) = cpu_weights.q_bias {
        crate::llama_forward::add_bias(&mut ws.q, b, seq_len, num_heads * head_dim);
    }
    if let Some(ref b) = cpu_weights.k_bias {
        crate::llama_forward::add_bias(&mut ws.k, b, seq_len, num_kv_heads * head_dim);
    }
    if let Some(ref b) = cpu_weights.v_bias {
        crate::llama_forward::add_bias(&mut ws.v, b, seq_len, num_kv_heads * head_dim);
    }

    // 3. RoPE (CPU)
    apply_rope(&mut ws.q, num_heads, head_dim, seq_len, config.rope_theta);
    apply_rope(&mut ws.k, num_kv_heads, head_dim, seq_len, config.rope_theta);

    // 4. GQA Attention (CUDA — per-head matmul は小行列なので buf_b 経由で十分)
    cuda_gqa_attention(cuda, &ws.q, &ws.k, &ws.v, &mut ws.attn_out_raw, &mut ws.attn_weights, config, seq_len);

    // 5. O projection — VRAM 常駐
    cuda.matmul_bt_with_gpu_b(&ws.attn_out_raw, &vram.o_proj, &mut ws.attn_out, seq_len, hidden_dim, num_heads * head_dim);

    // 6. Residual add
    for i in 0..hid_len {
        input[i] = ws.ffn_out[i] + ws.attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm
    ws.ffn_out[..hid_len].copy_from_slice(&input[..hid_len]);
    ws.normed_ffn[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(&mut ws.normed_ffn[..hid_len], &cpu_weights.ffn_norm, hidden_dim, config.norm_eps);

    // 8. SwiGLU FFN — VRAM 常駐重み
    let total = seq_len * intermediate_dim;
    cuda.matmul_bt_with_gpu_b(&ws.normed_ffn[..hid_len], &vram.gate_proj, &mut ws.gate[..total], seq_len, intermediate_dim, hidden_dim);
    cuda.matmul_bt_with_gpu_b(&ws.normed_ffn[..hid_len], &vram.up_proj, &mut ws.up[..total], seq_len, intermediate_dim, hidden_dim);

    ws.intermediate[..total].par_iter_mut()
        .zip(ws.gate[..total].par_iter().zip(ws.up[..total].par_iter()))
        .for_each(|(im, (&g, &u))| {
            let s = g * crate::fast_math::fast_sigmoid(g);
            *im = s * u;
        });

    cuda.matmul_bt_with_gpu_b(&ws.intermediate[..total], &vram.down_proj, &mut ws.attn_out[..hid_len], seq_len, hidden_dim, intermediate_dim);

    // 9. Residual add
    for i in 0..hid_len {
        input[i] = ws.ffn_out[i] + ws.attn_out[i];
    }
}

/// Workspace 版学習 forward — `LayerCache` を返すが、一時バッファは workspace から使い回す。
/// residual/normed/q/k/v 等の backward 必須データは `LayerCache` にコピーして返す。
pub fn cuda_layer_forward_ws(
    cuda: &CudaMatmul,
    input: &mut Vec<f32>,
    weights: &LlamaLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
    ws: &mut CudaLayerWorkspace,
) -> LayerCache {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_dim = config.intermediate_dim;
    let hid_len = seq_len * hidden_dim;
    let q_len = seq_len * num_heads * head_dim;
    let kv_len = seq_len * num_kv_heads * head_dim;
    let inter_len = seq_len * intermediate_dim;
    let aw_len = num_heads * seq_len * seq_len;

    // 1. 残差保存 + Attention RMSNorm
    let residual_attn = input[..hid_len].to_vec();
    ws.normed[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(&mut ws.normed[..hid_len], &weights.attn_norm, hidden_dim, config.norm_eps);
    let normed_attn = ws.normed[..hid_len].to_vec();

    // 2. QKV projection (CUDA inplace into workspace)
    cuda.matmul_bt_inplace(&ws.normed[..hid_len], &weights.q_proj, &mut ws.q[..q_len], seq_len, num_heads * head_dim, hidden_dim);
    cuda.matmul_bt_inplace(&ws.normed[..hid_len], &weights.k_proj, &mut ws.k[..kv_len], seq_len, num_kv_heads * head_dim, hidden_dim);
    cuda.matmul_bt_inplace(&ws.normed[..hid_len], &weights.v_proj, &mut ws.v[..kv_len], seq_len, num_kv_heads * head_dim, hidden_dim);

    // 2b. Attention bias
    if let Some(ref b) = weights.q_bias {
        crate::llama_forward::add_bias(&mut ws.q[..q_len], b, seq_len, num_heads * head_dim);
    }
    if let Some(ref b) = weights.k_bias {
        crate::llama_forward::add_bias(&mut ws.k[..kv_len], b, seq_len, num_kv_heads * head_dim);
    }
    if let Some(ref b) = weights.v_bias {
        crate::llama_forward::add_bias(&mut ws.v[..kv_len], b, seq_len, num_kv_heads * head_dim);
    }

    // 3. RoPE
    apply_rope(&mut ws.q[..q_len], num_heads, head_dim, seq_len, config.rope_theta);
    apply_rope(&mut ws.k[..kv_len], num_kv_heads, head_dim, seq_len, config.rope_theta);

    // LayerCache 用にコピー（backward で必要）
    let q_cache = ws.q[..q_len].to_vec();
    let k_cache = ws.k[..kv_len].to_vec();
    let v_cache = ws.v[..kv_len].to_vec();

    // 4. GQA Attention
    cuda_gqa_attention(cuda, &ws.q[..q_len], &ws.k[..kv_len], &ws.v[..kv_len],
        &mut ws.attn_out_raw[..q_len], &mut ws.attn_weights[..aw_len], config, seq_len);
    let attn_weights_cache = ws.attn_weights[..aw_len].to_vec();

    // 5. O projection
    cuda.matmul_bt_inplace(&ws.attn_out_raw[..q_len], &weights.o_proj, &mut ws.attn_out[..hid_len], seq_len, hidden_dim, num_heads * head_dim);
    let attn_out_cache = ws.attn_out[..hid_len].to_vec();

    // 6. Residual add
    for i in 0..hid_len {
        input[i] = residual_attn[i] + ws.attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm
    let residual_ffn = input[..hid_len].to_vec();
    ws.normed_ffn[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(&mut ws.normed_ffn[..hid_len], &weights.ffn_norm, hidden_dim, config.norm_eps);
    let normed_ffn_cache = ws.normed_ffn[..hid_len].to_vec();

    // 8. SwiGLU FFN
    cuda.matmul_bt_inplace(&ws.normed_ffn[..hid_len], &weights.gate_proj, &mut ws.gate[..inter_len], seq_len, intermediate_dim, hidden_dim);
    cuda.matmul_bt_inplace(&ws.normed_ffn[..hid_len], &weights.up_proj, &mut ws.up[..inter_len], seq_len, intermediate_dim, hidden_dim);

    // SiLU(gate) ⊙ up
    ws.gate_silu[..inter_len].par_iter_mut()
        .zip(ws.intermediate[..inter_len].par_iter_mut())
        .zip(ws.gate[..inter_len].par_iter().zip(ws.up[..inter_len].par_iter()))
        .for_each(|((gs, im), (&g, &u))| {
            let s = g * crate::fast_math::fast_sigmoid(g);
            *gs = s;
            *im = s * u;
        });

    let gate_cache = ws.gate[..inter_len].to_vec();
    let up_cache = ws.up[..inter_len].to_vec();
    let gate_silu_cache = ws.gate_silu[..inter_len].to_vec();

    cuda.matmul_bt_inplace(&ws.intermediate[..inter_len], &weights.down_proj, &mut ws.ffn_out[..hid_len], seq_len, hidden_dim, intermediate_dim);

    // 9. Residual add
    for i in 0..hid_len {
        input[i] = residual_ffn[i] + ws.ffn_out[i];
    }

    LayerCache {
        residual_attn,
        normed_attn,
        q: q_cache,
        k: k_cache,
        v: v_cache,
        attn_weights: attn_weights_cache,
        attn_out: attn_out_cache,
        residual_ffn,
        normed_ffn: normed_ffn_cache,
        gate: gate_cache,
        up: up_cache,
        gate_silu: gate_silu_cache,
    }
}

/// VRAM 常駐重み版 学習 forward — 重みの H2D 転送を完全排除。
/// Forward + Backward で1回のアップロードを共有するため、学習時のPCIe転送を 1/14 に削減。
pub fn cuda_layer_forward_ws_vram(
    cuda: &CudaMatmul,
    input: &mut Vec<f32>,
    cpu_weights: &LlamaLayerWeights,
    vram: &VramLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
    ws: &mut CudaLayerWorkspace,
) -> LayerCache {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_dim = config.intermediate_dim;
    let hid_len = seq_len * hidden_dim;
    let q_len = seq_len * num_heads * head_dim;
    let kv_len = seq_len * num_kv_heads * head_dim;
    let inter_len = seq_len * intermediate_dim;
    let aw_len = num_heads * seq_len * seq_len;

    // 1. 残差保存 + Attention RMSNorm
    let residual_attn = input[..hid_len].to_vec();
    ws.normed[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(&mut ws.normed[..hid_len], &cpu_weights.attn_norm, hidden_dim, config.norm_eps);
    let normed_attn = ws.normed[..hid_len].to_vec();

    // 2. QKV projection — VRAM 常駐
    cuda.matmul_bt_with_gpu_b(&ws.normed[..hid_len], &vram.q_proj, &mut ws.q[..q_len], seq_len, num_heads * head_dim, hidden_dim);
    cuda.matmul_bt_with_gpu_b(&ws.normed[..hid_len], &vram.k_proj, &mut ws.k[..kv_len], seq_len, num_kv_heads * head_dim, hidden_dim);
    cuda.matmul_bt_with_gpu_b(&ws.normed[..hid_len], &vram.v_proj, &mut ws.v[..kv_len], seq_len, num_kv_heads * head_dim, hidden_dim);

    // 2b. Attention bias
    if let Some(ref b) = cpu_weights.q_bias {
        crate::llama_forward::add_bias(&mut ws.q[..q_len], b, seq_len, num_heads * head_dim);
    }
    if let Some(ref b) = cpu_weights.k_bias {
        crate::llama_forward::add_bias(&mut ws.k[..kv_len], b, seq_len, num_kv_heads * head_dim);
    }
    if let Some(ref b) = cpu_weights.v_bias {
        crate::llama_forward::add_bias(&mut ws.v[..kv_len], b, seq_len, num_kv_heads * head_dim);
    }

    // 3. RoPE
    apply_rope(&mut ws.q[..q_len], num_heads, head_dim, seq_len, config.rope_theta);
    apply_rope(&mut ws.k[..kv_len], num_kv_heads, head_dim, seq_len, config.rope_theta);

    let q_cache = ws.q[..q_len].to_vec();
    let k_cache = ws.k[..kv_len].to_vec();
    let v_cache = ws.v[..kv_len].to_vec();

    // 4. GQA Attention
    cuda_gqa_attention(cuda, &ws.q[..q_len], &ws.k[..kv_len], &ws.v[..kv_len],
        &mut ws.attn_out_raw[..q_len], &mut ws.attn_weights[..aw_len], config, seq_len);
    let attn_weights_cache = ws.attn_weights[..aw_len].to_vec();

    // 5. O projection — VRAM 常駐
    cuda.matmul_bt_with_gpu_b(&ws.attn_out_raw[..q_len], &vram.o_proj, &mut ws.attn_out[..hid_len], seq_len, hidden_dim, num_heads * head_dim);
    let attn_out_cache = ws.attn_out[..hid_len].to_vec();

    // 6. Residual add
    for i in 0..hid_len {
        input[i] = residual_attn[i] + ws.attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm
    let residual_ffn = input[..hid_len].to_vec();
    ws.normed_ffn[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(&mut ws.normed_ffn[..hid_len], &cpu_weights.ffn_norm, hidden_dim, config.norm_eps);
    let normed_ffn_cache = ws.normed_ffn[..hid_len].to_vec();

    // 8. SwiGLU FFN — VRAM 常駐
    cuda.matmul_bt_with_gpu_b(&ws.normed_ffn[..hid_len], &vram.gate_proj, &mut ws.gate[..inter_len], seq_len, intermediate_dim, hidden_dim);
    cuda.matmul_bt_with_gpu_b(&ws.normed_ffn[..hid_len], &vram.up_proj, &mut ws.up[..inter_len], seq_len, intermediate_dim, hidden_dim);

    ws.gate_silu[..inter_len].par_iter_mut()
        .zip(ws.intermediate[..inter_len].par_iter_mut())
        .zip(ws.gate[..inter_len].par_iter().zip(ws.up[..inter_len].par_iter()))
        .for_each(|((gs, im), (&g, &u))| {
            let s = g * crate::fast_math::fast_sigmoid(g);
            *gs = s;
            *im = s * u;
        });

    let gate_cache = ws.gate[..inter_len].to_vec();
    let up_cache = ws.up[..inter_len].to_vec();
    let gate_silu_cache = ws.gate_silu[..inter_len].to_vec();

    cuda.matmul_bt_with_gpu_b(&ws.intermediate[..inter_len], &vram.down_proj, &mut ws.ffn_out[..hid_len], seq_len, hidden_dim, intermediate_dim);

    // 9. Residual add
    for i in 0..hid_len {
        input[i] = residual_ffn[i] + ws.ffn_out[i];
    }

    LayerCache {
        residual_attn,
        normed_attn,
        q: q_cache,
        k: k_cache,
        v: v_cache,
        attn_weights: attn_weights_cache,
        attn_out: attn_out_cache,
        residual_ffn,
        normed_ffn: normed_ffn_cache,
        gate: gate_cache,
        up: up_cache,
        gate_silu: gate_silu_cache,
    }
}

/// Workspace 版 backward — 動的アロケーションを最小化。
/// `ws` の backward バッファを使い回し、`d_input` は `ws.d_input` に書き込む。
/// 戻り値の `Vec<f32>` は `ws.d_input` のコピー（呼び出し側で次レイヤーへ渡すため）。
pub fn cuda_layer_backward_ws(
    cuda: &CudaMatmul,
    d_output: &[f32],
    cache: &LayerCache,
    weights: &LlamaLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
    ws: &mut CudaLayerWorkspace,
) -> (Vec<f32>, LayerWeightGrads) {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_dim = config.intermediate_dim;
    let hid_len = seq_len * hidden_dim;
    let inter_len = seq_len * intermediate_dim;
    let q_len = seq_len * num_heads * head_dim;
    let kv_len = seq_len * num_kv_heads * head_dim;

    // ── FFN Backward ──

    // 1. down_proj backward: d_intermediate = d_output × down_proj
    cuda.matmul_nn_inplace(d_output, &weights.down_proj, &mut ws.d_intermediate[..inter_len], seq_len, intermediate_dim, hidden_dim);

    // SwiGLU elementwise backward (CPU — rayon 並列)
    ws.d_gate[..inter_len].par_iter_mut()
        .zip(ws.d_up[..inter_len].par_iter_mut())
        .enumerate()
        .for_each(|(idx, (dg, du))| {
            let d_gate_silu = ws.d_intermediate[idx] * cache.up[idx];
            *du = ws.d_intermediate[idx] * cache.gate_silu[idx];
            let x = cache.gate[idx];
            let sig = crate::fast_math::fast_sigmoid(x);
            let silu_grad = sig * (1.0 + x * (1.0 - sig));
            *dg = d_gate_silu * silu_grad;
        });

    // gate/up_proj backward
    cuda.matmul_nn_inplace(&ws.d_gate[..inter_len], &weights.gate_proj, &mut ws.d_input_gate[..hid_len], seq_len, hidden_dim, intermediate_dim);
    cuda.matmul_nn_inplace(&ws.d_up[..inter_len], &weights.up_proj, &mut ws.d_input_up[..hid_len], seq_len, hidden_dim, intermediate_dim);

    // d_pre_ffn = d_input_gate + d_input_up
    for i in 0..hid_len {
        ws.d_pre_ffn[i] = ws.d_input_gate[i] + ws.d_input_up[i];
    }

    // SwiGLU intermediate for down_proj grad
    for idx in 0..inter_len {
        ws.swiglu_out[idx] = cache.gate_silu[idx] * cache.up[idx];
    }

    // FFN weight grads (matmul_tn)
    let d_down_proj = cuda.matmul_tn(d_output, &ws.swiglu_out[..inter_len], seq_len, intermediate_dim, hidden_dim);
    let d_gate_proj = cuda.matmul_tn(&ws.d_gate[..inter_len], &cache.normed_ffn, seq_len, hidden_dim, intermediate_dim);
    let d_up_proj = cuda.matmul_tn(&ws.d_up[..inter_len], &cache.normed_ffn, seq_len, hidden_dim, intermediate_dim);

    // 2. FFN RMSNorm backward
    ws.d_ffn_norm_w.iter_mut().for_each(|x| *x = 0.0);
    crate::llama_backward::rmsnorm_backward(
        &ws.d_pre_ffn[..hid_len],
        &cache.residual_ffn,
        &weights.ffn_norm,
        &mut ws.d_pre_ffn_residual[..hid_len],
        &mut ws.d_ffn_norm_w,
        hidden_dim,
        config.norm_eps,
    );

    // 3. Residual: d_attn_output = d_output + d_pre_ffn_residual
    for i in 0..hid_len {
        ws.d_attn_output[i] = d_output[i] + ws.d_pre_ffn_residual[i];
    }

    // ── Attention Backward ──

    // 4. O projection backward
    cuda.matmul_nn_inplace(&ws.d_attn_output[..hid_len], &weights.o_proj, &mut ws.d_attn_raw[..q_len], seq_len, num_heads * head_dim, hidden_dim);

    // 5. GQA Attention backward
    ws.d_q[..q_len].iter_mut().for_each(|x| *x = 0.0);
    ws.d_k[..kv_len].iter_mut().for_each(|x| *x = 0.0);
    ws.d_v[..kv_len].iter_mut().for_each(|x| *x = 0.0);
    cuda_gqa_attention_backward(
        cuda, &ws.d_attn_raw[..q_len], &cache.attn_weights,
        &cache.q, &cache.k, &cache.v,
        &mut ws.d_q[..q_len], &mut ws.d_k[..kv_len], &mut ws.d_v[..kv_len],
        config, seq_len,
    );

    // 6. RoPE backward
    crate::llama_backward::rope_backward(&mut ws.d_q[..q_len], num_heads, head_dim, seq_len, config.rope_theta);
    crate::llama_backward::rope_backward(&mut ws.d_k[..kv_len], num_kv_heads, head_dim, seq_len, config.rope_theta);

    // 7. QKV projection backward
    cuda.matmul_nn_inplace(&ws.d_q[..q_len], &weights.q_proj, &mut ws.d_normed_q[..hid_len], seq_len, hidden_dim, num_heads * head_dim);
    cuda.matmul_nn_inplace(&ws.d_k[..kv_len], &weights.k_proj, &mut ws.d_normed_k[..hid_len], seq_len, hidden_dim, num_kv_heads * head_dim);
    cuda.matmul_nn_inplace(&ws.d_v[..kv_len], &weights.v_proj, &mut ws.d_normed_v[..hid_len], seq_len, hidden_dim, num_kv_heads * head_dim);

    for i in 0..hid_len {
        ws.d_normed[i] = ws.d_normed_q[i] + ws.d_normed_k[i] + ws.d_normed_v[i];
    }

    // O projection weight grad: attn_out_raw を再計算
    {
        let kv_group_size = num_heads / num_kv_heads;
        ws.attn_out_raw_recompute[..q_len].iter_mut().for_each(|x| *x = 0.0);
        for h in 0..num_heads {
            let kv_h = h / kv_group_size;
            let aw_h = &cache.attn_weights[h * seq_len * seq_len..(h + 1) * seq_len * seq_len];
            let v_h = extract_head(&cache.v, kv_h, num_kv_heads, head_dim, seq_len);
            let out_h = cuda.matmul_nn(aw_h, &v_h, seq_len, head_dim, seq_len);
            scatter_head(&out_h, &mut ws.attn_out_raw_recompute[..q_len], h, num_heads, head_dim, seq_len);
        }
    }
    let d_o_proj = cuda.matmul_tn(&ws.d_attn_output[..hid_len], &ws.attn_out_raw_recompute[..q_len], seq_len, num_heads * head_dim, hidden_dim);

    // QKV weight grads
    let d_q_proj = cuda.matmul_tn(&ws.d_q[..q_len], &cache.normed_attn, seq_len, hidden_dim, num_heads * head_dim);
    let d_k_proj = cuda.matmul_tn(&ws.d_k[..kv_len], &cache.normed_attn, seq_len, hidden_dim, num_kv_heads * head_dim);
    let d_v_proj = cuda.matmul_tn(&ws.d_v[..kv_len], &cache.normed_attn, seq_len, hidden_dim, num_kv_heads * head_dim);

    // 8. Attention RMSNorm backward
    ws.d_attn_norm_w.iter_mut().for_each(|x| *x = 0.0);
    crate::llama_backward::rmsnorm_backward(
        &ws.d_normed[..hid_len],
        &cache.residual_attn,
        &weights.attn_norm,
        &mut ws.d_input[..hid_len],
        &mut ws.d_attn_norm_w,
        hidden_dim,
        config.norm_eps,
    );

    // 9. Residual: d_input += d_attn_output
    for i in 0..hid_len {
        ws.d_input[i] += ws.d_attn_output[i];
    }

    // Bias 勾配
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
        Some(compute_bias_grad(&ws.d_q[..q_len], num_heads * head_dim))
    } else { None };
    let d_k_bias = if weights.k_bias.is_some() {
        Some(compute_bias_grad(&ws.d_k[..kv_len], num_kv_heads * head_dim))
    } else { None };
    let d_v_bias = if weights.v_bias.is_some() {
        Some(compute_bias_grad(&ws.d_v[..kv_len], num_kv_heads * head_dim))
    } else { None };

    let grads = LayerWeightGrads {
        attn_norm: ws.d_attn_norm_w.clone(),
        q_proj: d_q_proj,
        k_proj: d_k_proj,
        v_proj: d_v_proj,
        o_proj: d_o_proj,
        q_bias: d_q_bias,
        k_bias: d_k_bias,
        v_bias: d_v_bias,
        ffn_norm: ws.d_ffn_norm_w.clone(),
        gate_proj: d_gate_proj,
        up_proj: d_up_proj,
        down_proj: d_down_proj,
    };

    (ws.d_input[..hid_len].to_vec(), grads)
}

/// VRAM 常駐重み版 backward。重みの H2D 転送を完全排除。
/// `matmul_nn_inplace` → `matmul_nn_with_gpu_b` で backward の d_input 計算を高速化。
pub fn cuda_layer_backward_ws_vram(
    cuda: &CudaMatmul,
    d_output: &[f32],
    cache: &LayerCache,
    cpu_weights: &LlamaLayerWeights,
    vram: &VramLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
    ws: &mut CudaLayerWorkspace,
) -> (Vec<f32>, LayerWeightGrads) {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_dim = config.intermediate_dim;
    let hid_len = seq_len * hidden_dim;
    let inter_len = seq_len * intermediate_dim;
    let q_len = seq_len * num_heads * head_dim;
    let kv_len = seq_len * num_kv_heads * head_dim;

    // ── FFN Backward ──

    // 1. down_proj backward — VRAM 常駐
    cuda.matmul_nn_with_gpu_b(d_output, &vram.down_proj, &mut ws.d_intermediate[..inter_len], seq_len, intermediate_dim, hidden_dim);

    // SwiGLU elementwise backward (CPU — rayon 並列)
    ws.d_gate[..inter_len].par_iter_mut()
        .zip(ws.d_up[..inter_len].par_iter_mut())
        .enumerate()
        .for_each(|(idx, (dg, du))| {
            let d_gate_silu = ws.d_intermediate[idx] * cache.up[idx];
            *du = ws.d_intermediate[idx] * cache.gate_silu[idx];
            let x = cache.gate[idx];
            let sig = crate::fast_math::fast_sigmoid(x);
            let silu_grad = sig * (1.0 + x * (1.0 - sig));
            *dg = d_gate_silu * silu_grad;
        });

    // gate/up_proj backward — VRAM 常駐
    cuda.matmul_nn_with_gpu_b(&ws.d_gate[..inter_len], &vram.gate_proj, &mut ws.d_input_gate[..hid_len], seq_len, hidden_dim, intermediate_dim);
    cuda.matmul_nn_with_gpu_b(&ws.d_up[..inter_len], &vram.up_proj, &mut ws.d_input_up[..hid_len], seq_len, hidden_dim, intermediate_dim);

    for i in 0..hid_len {
        ws.d_pre_ffn[i] = ws.d_input_gate[i] + ws.d_input_up[i];
    }

    // SwiGLU intermediate for down_proj grad
    for idx in 0..inter_len {
        ws.swiglu_out[idx] = cache.gate_silu[idx] * cache.up[idx];
    }

    // FFN weight grads (matmul_tn — activations 使用、VRAM 不要)
    let d_down_proj = cuda.matmul_tn(d_output, &ws.swiglu_out[..inter_len], seq_len, intermediate_dim, hidden_dim);
    let d_gate_proj = cuda.matmul_tn(&ws.d_gate[..inter_len], &cache.normed_ffn, seq_len, hidden_dim, intermediate_dim);
    let d_up_proj = cuda.matmul_tn(&ws.d_up[..inter_len], &cache.normed_ffn, seq_len, hidden_dim, intermediate_dim);

    // 2. FFN RMSNorm backward
    ws.d_ffn_norm_w.iter_mut().for_each(|x| *x = 0.0);
    crate::llama_backward::rmsnorm_backward(
        &ws.d_pre_ffn[..hid_len],
        &cache.residual_ffn,
        &cpu_weights.ffn_norm,
        &mut ws.d_pre_ffn_residual[..hid_len],
        &mut ws.d_ffn_norm_w,
        hidden_dim,
        config.norm_eps,
    );

    // 3. Residual
    for i in 0..hid_len {
        ws.d_attn_output[i] = d_output[i] + ws.d_pre_ffn_residual[i];
    }

    // ── Attention Backward ──

    // 4. O projection backward — VRAM 常駐
    cuda.matmul_nn_with_gpu_b(&ws.d_attn_output[..hid_len], &vram.o_proj, &mut ws.d_attn_raw[..q_len], seq_len, num_heads * head_dim, hidden_dim);

    // 5. GQA Attention backward
    ws.d_q[..q_len].iter_mut().for_each(|x| *x = 0.0);
    ws.d_k[..kv_len].iter_mut().for_each(|x| *x = 0.0);
    ws.d_v[..kv_len].iter_mut().for_each(|x| *x = 0.0);
    cuda_gqa_attention_backward(
        cuda, &ws.d_attn_raw[..q_len], &cache.attn_weights,
        &cache.q, &cache.k, &cache.v,
        &mut ws.d_q[..q_len], &mut ws.d_k[..kv_len], &mut ws.d_v[..kv_len],
        config, seq_len,
    );

    // 6. RoPE backward
    crate::llama_backward::rope_backward(&mut ws.d_q[..q_len], num_heads, head_dim, seq_len, config.rope_theta);
    crate::llama_backward::rope_backward(&mut ws.d_k[..kv_len], num_kv_heads, head_dim, seq_len, config.rope_theta);

    // 7. QKV projection backward — VRAM 常駐
    cuda.matmul_nn_with_gpu_b(&ws.d_q[..q_len], &vram.q_proj, &mut ws.d_normed_q[..hid_len], seq_len, hidden_dim, num_heads * head_dim);
    cuda.matmul_nn_with_gpu_b(&ws.d_k[..kv_len], &vram.k_proj, &mut ws.d_normed_k[..hid_len], seq_len, hidden_dim, num_kv_heads * head_dim);
    cuda.matmul_nn_with_gpu_b(&ws.d_v[..kv_len], &vram.v_proj, &mut ws.d_normed_v[..hid_len], seq_len, hidden_dim, num_kv_heads * head_dim);

    for i in 0..hid_len {
        ws.d_normed[i] = ws.d_normed_q[i] + ws.d_normed_k[i] + ws.d_normed_v[i];
    }

    // O projection weight grad: attn_out_raw 再計算
    {
        let kv_group_size = num_heads / num_kv_heads;
        ws.attn_out_raw_recompute[..q_len].iter_mut().for_each(|x| *x = 0.0);
        for h in 0..num_heads {
            let kv_h = h / kv_group_size;
            let aw_h = &cache.attn_weights[h * seq_len * seq_len..(h + 1) * seq_len * seq_len];
            let v_h = extract_head(&cache.v, kv_h, num_kv_heads, head_dim, seq_len);
            let out_h = cuda.matmul_nn(aw_h, &v_h, seq_len, head_dim, seq_len);
            scatter_head(&out_h, &mut ws.attn_out_raw_recompute[..q_len], h, num_heads, head_dim, seq_len);
        }
    }
    let d_o_proj = cuda.matmul_tn(&ws.d_attn_output[..hid_len], &ws.attn_out_raw_recompute[..q_len], seq_len, num_heads * head_dim, hidden_dim);

    // QKV weight grads
    let d_q_proj = cuda.matmul_tn(&ws.d_q[..q_len], &cache.normed_attn, seq_len, hidden_dim, num_heads * head_dim);
    let d_k_proj = cuda.matmul_tn(&ws.d_k[..kv_len], &cache.normed_attn, seq_len, hidden_dim, num_kv_heads * head_dim);
    let d_v_proj = cuda.matmul_tn(&ws.d_v[..kv_len], &cache.normed_attn, seq_len, hidden_dim, num_kv_heads * head_dim);

    // 8. Attention RMSNorm backward
    ws.d_attn_norm_w.iter_mut().for_each(|x| *x = 0.0);
    crate::llama_backward::rmsnorm_backward(
        &ws.d_normed[..hid_len],
        &cache.residual_attn,
        &cpu_weights.attn_norm,
        &mut ws.d_input[..hid_len],
        &mut ws.d_attn_norm_w,
        hidden_dim,
        config.norm_eps,
    );

    // 9. Residual
    for i in 0..hid_len {
        ws.d_input[i] += ws.d_attn_output[i];
    }

    // Bias 勾配
    let compute_bias_grad = |d_out: &[f32], dim: usize| -> Vec<f32> {
        let mut d_bias = vec![0.0f32; dim];
        for t in 0..seq_len {
            for d in 0..dim {
                d_bias[d] += d_out[t * dim + d];
            }
        }
        d_bias
    };
    let d_q_bias = if cpu_weights.q_bias.is_some() {
        Some(compute_bias_grad(&ws.d_q[..q_len], num_heads * head_dim))
    } else { None };
    let d_k_bias = if cpu_weights.k_bias.is_some() {
        Some(compute_bias_grad(&ws.d_k[..kv_len], num_kv_heads * head_dim))
    } else { None };
    let d_v_bias = if cpu_weights.v_bias.is_some() {
        Some(compute_bias_grad(&ws.d_v[..kv_len], num_kv_heads * head_dim))
    } else { None };

    let grads = LayerWeightGrads {
        attn_norm: ws.d_attn_norm_w.clone(),
        q_proj: d_q_proj,
        k_proj: d_k_proj,
        v_proj: d_v_proj,
        o_proj: d_o_proj,
        q_bias: d_q_bias,
        k_bias: d_k_bias,
        v_bias: d_v_bias,
        ffn_norm: ws.d_ffn_norm_w.clone(),
        gate_proj: d_gate_proj,
        up_proj: d_up_proj,
        down_proj: d_down_proj,
    };

    (ws.d_input[..hid_len].to_vec(), grads)
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
