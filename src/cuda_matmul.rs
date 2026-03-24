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
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use rayon::prelude::*;
use std::sync::Arc;

use std::cell::RefCell;

/// GPU DeltaNet 再帰カーネル (NVRTC)。
///
/// 1ブロック = 1ヘッド。blockDim.x = dv。
/// 各スレッドは状態行列 S の1列 (dk 要素) を担当。
/// 共有メモリに S (dk×dv), k_buf (dk), q_buf (dk) を保持。
///
/// A100: 164KB shared memory → dk=dv=128 で 64KB+1KB = 65KB → OK。
const DELTANET_RECURRENCE_CU: &str = r#"
extern "C" __global__ void deltanet_recurrence(
    const float* __restrict__ q,       // [num_heads, seq_len, dk]
    const float* __restrict__ k,       // [num_heads, seq_len, dk]
    const float* __restrict__ v,       // [num_heads, seq_len, dv]
    const float* __restrict__ beta,    // [num_heads, seq_len]
    const float* __restrict__ g,       // [num_heads, seq_len]
    float* __restrict__ output,        // [num_heads, seq_len, dv]
    float* __restrict__ state_buf,     // [num_heads, dk, dv] — global memory state
    int seq_len,
    int dk,
    int dv)
{
    int head = blockIdx.x;
    int j = threadIdx.x;  // column index in S (0..dv-1)

    // State in global memory (per-head)
    float* S = state_buf + head * dk * dv;

    // Shared memory for k, q vectors only (small: 2*dk floats)
    extern __shared__ float smem[];
    float* k_buf = smem;           // [dk]
    float* q_buf = smem + dk;      // [dk]

    // Zero-initialize S column
    for (int i = 0; i < dk; i++) {
        S[i * dv + j] = 0.0f;
    }
    __syncthreads();

    // Per-head offsets
    int off_qk = head * seq_len * dk;
    int off_v  = head * seq_len * dv;
    int off_bg = head * seq_len;
    int off_o  = head * seq_len * dv;

    for (int t = 0; t < seq_len; t++) {
        // Collaborative load of k, q into shared memory
        if (j < dk) {
            k_buf[j] = k[off_qk + t * dk + j];
            q_buf[j] = q[off_qk + t * dk + j];
        }
        __syncthreads();

        float beta_t = beta[off_bg + t];
        float exp_g_t = expf(g[off_bg + t]);
        float v_j = v[off_v + t * dv + j];

        // Step 1: retrieve r[j] = sum_i S[i][j] * k[i]
        float r_j = 0.0f;
        for (int i = 0; i < dk; i++) {
            r_j = __fmaf_rn(S[i * dv + j], k_buf[i], r_j);
        }

        // Step 2: error
        float e_j = v_j - r_j;

        // Step 3: state update S[i][j] = exp_g * S[i][j] + beta * k[i] * e_j
        float be = beta_t * e_j;
        for (int i = 0; i < dk; i++) {
            S[i * dv + j] = __fmaf_rn(exp_g_t, S[i * dv + j], k_buf[i] * be);
        }
        __syncthreads();

        // Step 4: output o[j] = sum_i S[i][j] * q[i]
        float o_j = 0.0f;
        for (int i = 0; i < dk; i++) {
            o_j = __fmaf_rn(S[i * dv + j], q_buf[i], o_j);
        }

        output[off_o + t * dv + j] = o_j;
        __syncthreads();
    }
}
"#;

/// GPU DeltaNet 訓練用カーネル — 全timestepのS_{t-1}とe_tを保存
const DELTANET_RECURRENCE_TRAIN_CU: &str = r#"
extern "C" __global__ void deltanet_recurrence_train(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    float* __restrict__ output,
    float* __restrict__ state_buf,
    float* __restrict__ all_s_prev,
    float* __restrict__ all_e,
    int seq_len, int dk, int dv)
{
    int head = blockIdx.x;
    int j = threadIdx.x;

    float* S = state_buf + head * dk * dv;
    extern __shared__ float smem[];
    float* k_buf = smem;
    float* q_buf = smem + dk;

    for (int i = 0; i < dk; i++) {
        S[i * dv + j] = 0.0f;
    }
    __syncthreads();

    int off_qk = head * seq_len * dk;
    int off_v  = head * seq_len * dv;
    int off_bg = head * seq_len;
    int off_o  = head * seq_len * dv;
    long long off_st = (long long)head * seq_len * dk * dv;
    int off_e  = head * seq_len * dv;

    for (int t = 0; t < seq_len; t++) {
        if (j < dk) {
            k_buf[j] = k[off_qk + t * dk + j];
            q_buf[j] = q[off_qk + t * dk + j];
        }
        __syncthreads();

        float beta_t = beta[off_bg + t];
        float exp_g_t = expf(g[off_bg + t]);
        float v_j = v[off_v + t * dv + j];

        // Store S_{t-1}
        for (int i = 0; i < dk; i++) {
            all_s_prev[off_st + (long long)t * dk * dv + i * dv + j] = S[i * dv + j];
        }

        // r[j] = sum_i S[i][j] * k[i]
        float r_j = 0.0f;
        for (int i = 0; i < dk; i++) {
            r_j = __fmaf_rn(S[i * dv + j], k_buf[i], r_j);
        }

        // e[j] = v[j] - r[j]
        float e_j = v_j - r_j;
        all_e[off_e + t * dv + j] = e_j;

        // State update
        float be = beta_t * e_j;
        for (int i = 0; i < dk; i++) {
            S[i * dv + j] = __fmaf_rn(exp_g_t, S[i * dv + j], k_buf[i] * be);
        }
        __syncthreads();

        // Output
        float o_j = 0.0f;
        for (int i = 0; i < dk; i++) {
            o_j = __fmaf_rn(S[i * dv + j], q_buf[i], o_j);
        }
        output[off_o + t * dv + j] = o_j;
        __syncthreads();
    }
}
"#;

/// GPU DeltaNet Fused Forward+Backward — 状態をVRAM内に保持、D2H転送ゼロ
const DELTANET_FUSED_FWD_BWD_CU: &str = r#"
extern "C" __global__ void deltanet_fused_fwd_bwd(
    const float* __restrict__ q,       // [H, T, dk]
    const float* __restrict__ k,       // [H, T, dk]
    const float* __restrict__ v,       // [H, T, dv]
    const float* __restrict__ beta,    // [H, T]
    const float* __restrict__ g,       // [H, T]
    const float* __restrict__ d_out,   // [H, T, dv]
    float* __restrict__ output,        // [H, T, dv]
    float* __restrict__ d_q_out,       // [H, T, dk]
    float* __restrict__ d_k_out,       // [H, T, dk]
    float* __restrict__ d_v_out,       // [H, T, dv]
    float* __restrict__ d_beta_out,    // [H, T]
    float* __restrict__ d_g_out,       // [H, T]
    float* __restrict__ state_buf,     // [H, dk, dv] current state
    float* __restrict__ all_s,         // [H, T, dk, dv] workspace
    int seq_len, int dk, int dv)
{
    int h = blockIdx.x;
    int j = threadIdx.x;  // 0..dv-1

    float* S = state_buf + h * dk * dv;
    extern __shared__ float smem[];
    float* k_buf = smem;
    float* q_buf = smem + dk;
    float* do_buf = smem + 2 * dk;  // reuse in backward

    int off_qk = h * seq_len * dk;
    int off_v  = h * seq_len * dv;
    int off_bg = h * seq_len;
    long long off_st = (long long)h * seq_len * dk * dv;

    // ═══ Phase 1: Forward ═══
    for (int i = 0; i < dk; i++) S[i * dv + j] = 0.0f;
    __syncthreads();

    for (int t = 0; t < seq_len; t++) {
        if (j < dk) {
            k_buf[j] = k[off_qk + t * dk + j];
            q_buf[j] = q[off_qk + t * dk + j];
        }
        __syncthreads();

        float beta_t = beta[off_bg + t];
        float exp_g_t = expf(g[off_bg + t]);
        float v_j = v[off_v + t * dv + j];

        // Store S_{t-1}
        for (int i = 0; i < dk; i++)
            all_s[off_st + (long long)t * dk * dv + i * dv + j] = S[i * dv + j];

        // r = S^T k
        float r_j = 0.0f;
        for (int i = 0; i < dk; i++)
            r_j = __fmaf_rn(S[i * dv + j], k_buf[i], r_j);

        // e = v - r (stored implicitly via output)
        float e_j = v_j - r_j;

        // State update: S = exp_g * S + beta * k * e
        float be = beta_t * e_j;
        for (int i = 0; i < dk; i++)
            S[i * dv + j] = __fmaf_rn(exp_g_t, S[i * dv + j], k_buf[i] * be);
        __syncthreads();

        // Output: o = S q
        float o_j = 0.0f;
        for (int i = 0; i < dk; i++)
            o_j = __fmaf_rn(S[i * dv + j], q_buf[i], o_j);
        output[off_v + t * dv + j] = o_j;
        __syncthreads();
    }

    // ═══ Phase 2: Backward ═══
    // dS accumulator in state_buf (reuse S memory)
    for (int i = 0; i < dk; i++) S[i * dv + j] = 0.0f;
    __syncthreads();

    for (int t = seq_len - 1; t >= 0; t--) {
        if (j < dk) {
            k_buf[j] = k[off_qk + t * dk + j];
            q_buf[j] = q[off_qk + t * dk + j];
        }
        __syncthreads();

        float beta_t = beta[off_bg + t];
        float g_t = g[off_bg + t];
        float exp_g_t = expf(g_t);
        float v_j = v[off_v + t * dv + j];
        float do_j = d_out[off_v + t * dv + j];

        // Load S_{t-1} from saved states
        // Reconstruct S_t = exp_g * S_{t-1} + beta * k * e
        float s_prev_col[128];  // Stack alloc per thread (max dk=128)
        for (int i = 0; i < dk; i++)
            s_prev_col[i] = all_s[off_st + (long long)t * dk * dv + i * dv + j];

        // r_j = sum_i S_{t-1}[i][j] * k[i]
        float r_j = 0.0f;
        for (int i = 0; i < dk; i++)
            r_j = __fmaf_rn(s_prev_col[i], k_buf[i], r_j);
        float e_j = v_j - r_j;

        // S_t[i][j] = exp_g * S_{t-1}[i][j] + beta * k[i] * e_j
        float be = beta_t * e_j;
        float s_t_col[128];
        for (int i = 0; i < dk; i++)
            s_t_col[i] = __fmaf_rn(exp_g_t, s_prev_col[i], k_buf[i] * be);

        // dq_t[i] = sum_j S_t[i][j] * do[j]  — need warp reduce across j
        // Thread j holds s_t_col[i] and do_j
        // dq_t[i] = S_t[i][j] * do_j accumulated across all j
        // Use shared memory for reduction
        if (j < dk) {
            float dq_i = 0.0f;
            // Each thread j contributes s_t[i=j_loop][j] * do_j, but we need sum over j
            // This requires a different threading model. Use atomicAdd.
        }

        // Simpler approach: each thread j computes its contribution
        // dS[i][j] += q[i] * do[j]
        for (int i = 0; i < dk; i++)
            S[i * dv + j] += q_buf[i] * do_j;  // dS accumulation

        // dst_k[j] = sum_i dS[i][j] * k[i]
        float dst_k_j = 0.0f;
        for (int i = 0; i < dk; i++)
            dst_k_j = __fmaf_rn(S[i * dv + j], k_buf[i], dst_k_j);

        // dv[t][j] = beta * dst_k[j]
        d_v_out[off_v + t * dv + j] = beta_t * dst_k_j;

        // dq[t][i]: need reduction across j dimension
        // Thread j holds: s_t_col[i] * do_j for each i
        // dq[t][i] = sum_j (s_t_col[i] * do_j)  — need cross-thread sum
        // Use atomicAdd to global d_q_out
        for (int i = 0; i < dk; i++)
            atomicAdd(&d_q_out[off_qk + t * dk + i], s_t_col[i] * do_j);

        // dk[t][i] = beta * (sum_j dS[i][j]*e[j] - sum_j S_{t-1}[i][j]*dst_k[j])
        for (int i = 0; i < dk; i++) {
            float ds_e = S[i * dv + j] * e_j;
            float s_dst = s_prev_col[i] * dst_k_j;
            atomicAdd(&d_k_out[off_qk + t * dk + i], beta_t * (ds_e - s_dst));
        }

        // d_beta[t] = sum_{i,j} dS[i][j] * k[i] * e[j]
        float d_beta_contrib = 0.0f;
        for (int i = 0; i < dk; i++)
            d_beta_contrib += S[i * dv + j] * k_buf[i] * e_j;
        // Chain sigmoid: d_beta_logit = d_beta * beta * (1 - beta)
        atomicAdd(&d_beta_out[off_bg + t], d_beta_contrib * beta_t * (1.0f - beta_t));

        // d_g[t] = exp_g * frobenius(dS, S_{t-1})
        float frob_contrib = 0.0f;
        for (int i = 0; i < dk; i++)
            frob_contrib += S[i * dv + j] * s_prev_col[i];
        atomicAdd(&d_g_out[off_bg + t], exp_g_t * frob_contrib);

        // dS update: dS = exp_g * dS - beta * k ⊗ dst_k
        for (int i = 0; i < dk; i++)
            S[i * dv + j] = exp_g_t * S[i * dv + j] - beta_t * k_buf[i] * dst_k_j;
        __syncthreads();
    }
}
"#;

/// GPU DeltaNet Backward — VRAM上の保存済み状態S_{t-1}を使用
/// atomicAdd排除: warp shuffle + shared memory reduction
const DELTANET_BACKWARD_CU: &str = r#"

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

extern "C" __global__ void deltanet_backward(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    const float* __restrict__ d_out,
    const float* __restrict__ all_s,
    float* __restrict__ d_q_out,
    float* __restrict__ d_k_out,
    float* __restrict__ d_v_out,
    float* __restrict__ d_beta_out,
    float* __restrict__ d_g_out,
    int seq_len, int dk, int dv)
{
    int h = blockIdx.x;
    int j = threadIdx.x;
    int lane = j & 31;
    int warp_id = j >> 5;
    int num_warps = (dv + 31) >> 5;

    extern __shared__ float smem[];
    float* k_buf = smem;
    float* q_buf = smem + dk;
    float* warp_buf = smem + 2 * dk;  // [num_warps] for cross-warp reduce

    float dS_col[128];
    for (int i = 0; i < dk; i++) dS_col[i] = 0.0f;

    int off_qk = h * seq_len * dk;
    int off_v  = h * seq_len * dv;
    int off_bg = h * seq_len;
    long long off_st = (long long)h * seq_len * dk * dv;

    for (int t = seq_len - 1; t >= 0; t--) {
        if (j < dk) {
            k_buf[j] = k[off_qk + t * dk + j];
            q_buf[j] = q[off_qk + t * dk + j];
        }
        __syncthreads();

        float beta_t = beta[off_bg + t];
        float exp_g_t = expf(g[off_bg + t]);
        float v_j = v[off_v + t * dv + j];
        float do_j = d_out[off_v + t * dv + j];

        float s_prev_col[128];
        for (int i = 0; i < dk; i++)
            s_prev_col[i] = all_s[off_st + (long long)t * dk * dv + i * dv + j];

        float r_j = 0.0f;
        for (int i = 0; i < dk; i++)
            r_j = __fmaf_rn(s_prev_col[i], k_buf[i], r_j);
        float e_j = v_j - r_j;

        float be = beta_t * e_j;
        float s_t_col[128];
        for (int i = 0; i < dk; i++)
            s_t_col[i] = __fmaf_rn(exp_g_t, s_prev_col[i], k_buf[i] * be);

        for (int i = 0; i < dk; i++)
            dS_col[i] += q_buf[i] * do_j;

        // d_q[t][i] = sum_j(S_t[i][j] * do[j]) — warp reduce
        for (int i = 0; i < dk; i++) {
            float val = warp_reduce_sum(s_t_col[i] * do_j);
            if (lane == 0) warp_buf[warp_id] = val;
            __syncthreads();
            if (j == 0) {
                float total = 0.0f;
                for (int w = 0; w < num_warps; w++) total += warp_buf[w];
                d_q_out[off_qk + t * dk + i] = total;
            }
            __syncthreads();
        }

        float dst_k_j = 0.0f;
        for (int i = 0; i < dk; i++)
            dst_k_j = __fmaf_rn(dS_col[i], k_buf[i], dst_k_j);

        // d_k[t][i] = beta * sum_j(dS[i][j]*e[j] - S_{t-1}[i][j]*dst_k[j])
        for (int i = 0; i < dk; i++) {
            float contrib = beta_t * (dS_col[i] * e_j - s_prev_col[i] * dst_k_j);
            float val = warp_reduce_sum(contrib);
            if (lane == 0) warp_buf[warp_id] = val;
            __syncthreads();
            if (j == 0) {
                float total = 0.0f;
                for (int w = 0; w < num_warps; w++) total += warp_buf[w];
                d_k_out[off_qk + t * dk + i] = total;
            }
            __syncthreads();
        }

        d_v_out[off_v + t * dv + j] = beta_t * dst_k_j;

        // d_beta[t] = sum_{i,j}(dS[i][j]*k[i]*e[j]) * beta*(1-beta)
        {
            float db = 0.0f;
            for (int i = 0; i < dk; i++)
                db += dS_col[i] * k_buf[i] * e_j;
            float val = warp_reduce_sum(db);
            if (lane == 0) warp_buf[warp_id] = val;
            __syncthreads();
            if (j == 0) {
                float total = 0.0f;
                for (int w = 0; w < num_warps; w++) total += warp_buf[w];
                d_beta_out[off_bg + t] = total * beta_t * (1.0f - beta_t);
            }
            __syncthreads();
        }

        // d_g[t] = exp_g * sum_{i,j}(dS[i][j]*S_{t-1}[i][j])
        {
            float fr = 0.0f;
            for (int i = 0; i < dk; i++)
                fr += dS_col[i] * s_prev_col[i];
            float val = warp_reduce_sum(fr);
            if (lane == 0) warp_buf[warp_id] = val;
            __syncthreads();
            if (j == 0) {
                float total = 0.0f;
                for (int w = 0; w < num_warps; w++) total += warp_buf[w];
                d_g_out[off_bg + t] = total * exp_g_t;
            }
            __syncthreads();
        }

        for (int i = 0; i < dk; i++)
            dS_col[i] = exp_g_t * dS_col[i] - beta_t * k_buf[i] * dst_k_j;
        __syncthreads();
    }
}
"#;

/// GPU DeltaNet 補助カーネル群 (NVRTC)。
/// RMSNorm, Conv1d+SiLU, L2Norm+GQA, Gate計算, Gated RMSNorm, Residual Add。
/// 1回のNVRTCコンパイルで全カーネルを生成。
const DELTANET_OPS_CU: &str = r#"
// RMSNorm: x[i] = x[i] / rms * weight[i]
// 1ブロック = 1トークン、blockDim.x = dim
extern "C" __global__ void rmsnorm_kernel(
    float* x,              // [seq_len, dim] in-place
    const float* weight,   // [dim]
    int dim,
    float eps)
{
    int t = blockIdx.x;
    float* row = x + t * dim;
    int tid = threadIdx.x;

    // 二乗和 (shared memory reduction)
    extern __shared__ float sdata[];
    float val = (tid < dim) ? row[tid] * row[tid] : 0.0f;
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)dim + eps);
    __syncthreads();

    if (tid < dim) {
        row[tid] = row[tid] * inv_rms * weight[tid];
    }
}

// Gated RMSNorm: out = RMSNorm(x) * SiLU(z)
extern "C" __global__ void gated_rmsnorm_kernel(
    const float* x,        // [n, dim]
    const float* z,        // [n, dim]
    const float* weight,   // [dim]
    float* out,            // [n, dim]
    int dim,
    float eps)
{
    int t = blockIdx.x;
    const float* row_x = x + t * dim;
    const float* row_z = z + t * dim;
    float* row_out = out + t * dim;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float val = (tid < dim) ? row_x[tid] * row_x[tid] : 0.0f;
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)dim + eps);
    __syncthreads();

    if (tid < dim) {
        float normed = row_x[tid] * inv_rms * weight[tid];
        float zv = row_z[tid];
        float silu_z = zv / (1.0f + expf(-zv));
        row_out[tid] = normed * silu_z;
    }
}

// Depthwise Causal Conv1d + SiLU (行優先入力)
// 1ブロック = 1タイムステップ、blockDim.x = channels
extern "C" __global__ void conv1d_silu_kernel(
    const float* x,        // [seq_len, channels]
    const float* weight,   // [channels, kernel_size]
    float* out,            // [seq_len, channels]
    int channels,
    int seq_len,
    int kernel_size)
{
    int t = blockIdx.x;
    int c = threadIdx.x;
    if (c >= channels) return;

    const float* w = weight + c * kernel_size;
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int src_t = t - (kernel_size - 1) + k;
        if (src_t >= 0) {
            sum = __fmaf_rn(x[src_t * channels + c], w[k], sum);
        }
    }
    // SiLU
    out[t * channels + c] = sum / (1.0f + expf(-sum));
}

// Gate計算: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)
// 1ブロック = 1タイムステップ、blockDim.x = n_heads
extern "C" __global__ void gate_compute_kernel(
    const float* b_raw,    // [seq_len, n_heads]
    const float* a_raw,    // [seq_len, n_heads]
    const float* a_log,    // [n_heads]
    const float* dt_bias,  // [n_heads]
    float* beta_out,       // [seq_len, n_heads]
    float* g_out,          // [seq_len, n_heads]
    int n_heads)
{
    int t = blockIdx.x;
    int h = threadIdx.x;
    if (h >= n_heads) return;

    int idx = t * n_heads + h;
    beta_out[idx] = 1.0f / (1.0f + expf(-b_raw[idx]));

    float a_val = expf(a_log[h]);
    float x = a_raw[idx] + dt_bias[h];
    float sp = (x > 20.0f) ? x : ((x < -20.0f) ? 0.0f : logf(1.0f + expf(x)));
    g_out[idx] = -a_val * sp;
}

// L2 Norm + GQA Expand: Q,K を正規化しながら GQA repeat
// 1ブロック = 1タイムステップ × 1 k_head, blockDim.x = dk
extern "C" __global__ void l2norm_gqa_kernel(
    const float* q_in,     // [seq_len, n_k_heads, dk]
    const float* k_in,     // [seq_len, n_k_heads, dk]
    float* q_out,          // [seq_len, n_v_heads, dk]
    float* k_out,          // [seq_len, n_v_heads, dk]
    int n_k_heads,
    int n_v_heads,
    int dk,
    float eps)
{
    int t = blockIdx.x / n_k_heads;
    int kh = blockIdx.x % n_k_heads;
    int d = threadIdx.x;
    if (d >= dk) return;

    int repeat = n_v_heads / n_k_heads;
    int q_off = (t * n_k_heads + kh) * dk;
    int k_off = q_off;

    float qv = q_in[q_off + d];
    float kv = k_in[k_off + d];

    // Shared memory for norm computation
    extern __shared__ float sdata[];
    float* q_sq = sdata;
    float* k_sq = sdata + dk;

    q_sq[d] = qv * qv;
    k_sq[d] = kv * kv;
    __syncthreads();

    // Reduction for norm
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (d < s) {
            q_sq[d] += q_sq[d + s];
            k_sq[d] += k_sq[d + s];
        }
        __syncthreads();
    }

    float q_inv = rsqrtf(q_sq[0] + eps);
    float k_inv = rsqrtf(k_sq[0] + eps);

    float q_normed = qv * q_inv;
    float k_normed = kv * k_inv;

    // GQA repeat write
    for (int r = 0; r < repeat; r++) {
        int vh = kh * repeat + r;
        int out_off = (t * n_v_heads + vh) * dk + d;
        q_out[out_off] = q_normed;
        k_out[out_off] = k_normed;
    }
}

// Residual Add: x[i] += residual[i]
extern "C" __global__ void residual_add_kernel(
    float* x,
    const float* residual,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += residual[idx];
}

// SiLU(gate) * up → out (SwiGLU element-wise)
extern "C" __global__ void swiglu_elementwise_kernel(
    const float* gate,
    const float* up,
    float* out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));
        out[idx] = silu_g * up[idx];
    }
}
"#;

/// GPU BF16→FP32 展開カーネル (NVRTC)。
/// BF16 は FP32 の上位16ビット。展開は (u16 as u32) << 16 → reinterpret as f32。
const BF16_EXPAND_CU: &str = r#"
extern "C" __global__ void bf16_expand(
    const unsigned short* src,
    float* dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int bits = ((unsigned int)src[idx]) << 16;
    dst[idx] = __uint_as_float(bits);
}
"#;

/// GPU Fused Merge + Ternary Quantize + Alpha Blend カーネル (NVRTC)。
/// Pass 1: base+delta → workspace, |w| の部分和（shared memory reduction）
/// Pass 2: quantize + alpha blend（scale は CPU から渡す）
const FUSED_QUANTIZE_CU: &str = r#"
extern "C" __global__ void fused_merge_quantize(
    const float* base,
    const float* delta,
    float* workspace,
    int n,
    float inv_scale,
    float inv_temp,
    float scale,
    float alpha,
    float one_minus_alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = base[idx] + delta[idx];
    float scaled = val * inv_scale * inv_temp;
    // round + clamp to {-1, 0, +1}
    float qv_hard = fminf(fmaxf(rintf(scaled), -1.0f), 1.0f) * scale;
    workspace[idx] = val * one_minus_alpha + qv_hard * alpha;
}

// scale計算用: |base+delta| の合計をブロックリダクションで算出
extern "C" __global__ void sum_abs_kernel(
    const float* base,
    const float* delta,
    float* workspace,
    float* block_sums,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sdata[256];
    float val = 0.0f;
    if (idx < n) {
        float w = base[idx] + delta[idx];
        workspace[idx] = w;
        val = fabsf(w);
    }
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sums[blockIdx.x] = sdata[0];
}
"#;

/// GPU Fused AdamW カーネル (NVRTC)。
/// delta, grad, m, v を GPU 上で一括更新。
const FUSED_ADAMW_CU: &str = r#"
extern "C" __global__ void fused_adamw(
    float* delta,
    const float* grad,
    float* m,
    float* v,
    int n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2,
    float weight_decay)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = grad[idx];
    float mi = beta1 * m[idx] + (1.0f - beta1) * g;
    float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    delta[idx] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * delta[idx]);
}
"#;

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
    bf16_expand_func: CudaFunction,
    fused_quantize_func: CudaFunction,
    sum_abs_func: CudaFunction,
    fused_adamw_func: CudaFunction,
    deltanet_func: CudaFunction,
    deltanet_train_func: CudaFunction,
    deltanet_fused_func: CudaFunction,
    deltanet_bwd_func: CudaFunction,
    // DeltaNet 補助カーネル群
    dn_rmsnorm_func: CudaFunction,
    dn_gated_rmsnorm_func: CudaFunction,
    dn_conv1d_silu_func: CudaFunction,
    dn_gate_compute_func: CudaFunction,
    dn_l2norm_gqa_func: CudaFunction,
    dn_residual_add_func: CudaFunction,
    dn_swiglu_elem_func: CudaFunction,
    buf_a: RefCell<GpuBuf>,
    buf_b: RefCell<GpuBuf>,
    buf_c: RefCell<GpuBuf>,
    /// GPU 上の scores バッファ（Attention 用、D2H なしで保持）。
    buf_scores: RefCell<GpuBuf>,
    /// GPU 上の汎用バッファ（expand/quantize/adamw 用）。
    buf_d: RefCell<GpuBuf>,
    buf_e: RefCell<GpuBuf>,
}

impl CudaMatmul {
    /// CUDA デバイス 0 で初期化。
    #[must_use]
    pub fn new() -> Self {
        let ctx = CudaContext::new(0).expect("CUDA device 0 の初期化に失敗");
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).expect("cuBLAS 初期化に失敗");

        // FP32 精度モード（TF32は仮数部10bitで精度不足 → 28層蓄積で発散）
        unsafe {
            cudarc::cublas::sys::cublasSetMathMode(
                *blas.handle(),
                cudarc::cublas::sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
            );
        }

        // NVRTC カーネルコンパイル
        let ptx_softmax =
            compile_ptx(CAUSAL_SOFTMAX_CU).expect("NVRTC: causal_softmax コンパイル失敗");
        let mod_softmax = ctx
            .load_module(ptx_softmax)
            .expect("CUDA module ロード失敗");
        let softmax_func = mod_softmax
            .load_function("causal_softmax_scaled")
            .expect("causal_softmax_scaled ロード失敗");

        let ptx_expand = compile_ptx(BF16_EXPAND_CU).expect("NVRTC: bf16_expand コンパイル失敗");
        let mod_expand = ctx
            .load_module(ptx_expand)
            .expect("bf16_expand module ロード失敗");
        let bf16_expand_func = mod_expand
            .load_function("bf16_expand")
            .expect("bf16_expand ロード失敗");

        let ptx_quant =
            compile_ptx(FUSED_QUANTIZE_CU).expect("NVRTC: fused_quantize コンパイル失敗");
        let mod_quant = ctx
            .load_module(ptx_quant)
            .expect("fused_quantize module ロード失敗");
        let fused_quantize_func = mod_quant
            .load_function("fused_merge_quantize")
            .expect("fused_merge_quantize ロード失敗");
        let sum_abs_func = mod_quant
            .load_function("sum_abs_kernel")
            .expect("sum_abs_kernel ロード失敗");

        let ptx_adam = compile_ptx(FUSED_ADAMW_CU).expect("NVRTC: fused_adamw コンパイル失敗");
        let mod_adam = ctx
            .load_module(ptx_adam)
            .expect("fused_adamw module ロード失敗");
        let fused_adamw_func = mod_adam
            .load_function("fused_adamw")
            .expect("fused_adamw ロード失敗");

        let ptx_dn =
            compile_ptx(DELTANET_RECURRENCE_CU).expect("NVRTC: deltanet_recurrence コンパイル失敗");
        let mod_dn = ctx
            .load_module(ptx_dn)
            .expect("deltanet_recurrence module ロード失敗");
        let deltanet_func = mod_dn
            .load_function("deltanet_recurrence")
            .expect("deltanet_recurrence ロード失敗");

        let ptx_dn_train = compile_ptx(DELTANET_RECURRENCE_TRAIN_CU)
            .expect("NVRTC: deltanet_recurrence_train コンパイル失敗");
        let mod_dn_train = ctx
            .load_module(ptx_dn_train)
            .expect("deltanet_recurrence_train module ロード失敗");
        let deltanet_train_func = mod_dn_train
            .load_function("deltanet_recurrence_train")
            .expect("deltanet_recurrence_train ロード失敗");

        let ptx_dn_fused = compile_ptx(DELTANET_FUSED_FWD_BWD_CU)
            .expect("NVRTC: deltanet_fused_fwd_bwd コンパイル失敗");
        let mod_dn_fused = ctx
            .load_module(ptx_dn_fused)
            .expect("deltanet_fused module ロード失敗");
        let deltanet_fused_func = mod_dn_fused
            .load_function("deltanet_fused_fwd_bwd")
            .expect("deltanet_fused_fwd_bwd ロード失敗");

        let ptx_dn_bwd = compile_ptx(DELTANET_BACKWARD_CU)
            .expect("NVRTC: deltanet_backward コンパイル失敗");
        let mod_dn_bwd = ctx.load_module(ptx_dn_bwd)
            .expect("deltanet_backward module ロード失敗");
        let deltanet_bwd_func = mod_dn_bwd
            .load_function("deltanet_backward")
            .expect("deltanet_backward ロード失敗");

        let ptx_ops = compile_ptx(DELTANET_OPS_CU).expect("NVRTC: deltanet_ops コンパイル失敗");
        let mod_ops = ctx
            .load_module(ptx_ops)
            .expect("deltanet_ops module ロード失敗");
        let dn_rmsnorm_func = mod_ops
            .load_function("rmsnorm_kernel")
            .expect("rmsnorm_kernel ロード失敗");
        let dn_gated_rmsnorm_func = mod_ops
            .load_function("gated_rmsnorm_kernel")
            .expect("gated_rmsnorm_kernel ロード失敗");
        let dn_conv1d_silu_func = mod_ops
            .load_function("conv1d_silu_kernel")
            .expect("conv1d_silu_kernel ロード失敗");
        let dn_gate_compute_func = mod_ops
            .load_function("gate_compute_kernel")
            .expect("gate_compute_kernel ロード失敗");
        let dn_l2norm_gqa_func = mod_ops
            .load_function("l2norm_gqa_kernel")
            .expect("l2norm_gqa_kernel ロード失敗");
        let dn_residual_add_func = mod_ops
            .load_function("residual_add_kernel")
            .expect("residual_add_kernel ロード失敗");
        let dn_swiglu_elem_func = mod_ops
            .load_function("swiglu_elementwise_kernel")
            .expect("swiglu_elementwise_kernel ロード失敗");

        // 初期バッファ: 16M 要素 (~64MB) — 必要に応じて自動拡張
        let init_cap = 16 * 1024 * 1024;
        let ba = GpuBuf {
            buf: stream.alloc_zeros(init_cap).expect("GPU buf 確保失敗"),
            cap: init_cap,
        };
        let bb = GpuBuf {
            buf: stream.alloc_zeros(init_cap).expect("GPU buf 確保失敗"),
            cap: init_cap,
        };
        let bc = GpuBuf {
            buf: stream.alloc_zeros(init_cap).expect("GPU buf 確保失敗"),
            cap: init_cap,
        };
        let bs = GpuBuf {
            buf: stream.alloc_zeros(init_cap).expect("GPU buf 確保失敗"),
            cap: init_cap,
        };
        let bd = GpuBuf {
            buf: stream.alloc_zeros(init_cap).expect("GPU buf 確保失敗"),
            cap: init_cap,
        };
        let be = GpuBuf {
            buf: stream.alloc_zeros(init_cap).expect("GPU buf 確保失敗"),
            cap: init_cap,
        };
        Self {
            stream,
            blas,
            softmax_func,
            bf16_expand_func,
            fused_quantize_func,
            sum_abs_func,
            fused_adamw_func,
            deltanet_func,
            deltanet_train_func,
            deltanet_fused_func,
            deltanet_bwd_func,
            dn_rmsnorm_func,
            dn_gated_rmsnorm_func,
            dn_conv1d_silu_func,
            dn_gate_compute_func,
            dn_l2norm_gqa_func,
            dn_residual_add_func,
            dn_swiglu_elem_func,
            buf_a: RefCell::new(ba),
            buf_b: RefCell::new(bb),
            buf_c: RefCell::new(bc),
            buf_scores: RefCell::new(bs),
            buf_d: RefCell::new(bd),
            buf_e: RefCell::new(be),
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
        let mut gpu: CudaSlice<f32> = self
            .stream
            .alloc_zeros(cpu.len())
            .expect("VRAM weight 確保失敗");
        self.stream
            .memcpy_htod(cpu, &mut gpu)
            .expect("weight H2D 転送失敗");
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
            self.stream
                .memcpy_htod(a, &mut ba.buf.slice_mut(..m * k))
                .expect("H2D a");
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

        {
            let ba = self.buf_a.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas
                    .gemm(
                        cfg,
                        &b_gpu.slice(..n * k),
                        &ba.buf.slice(..m * k),
                        &mut bc.buf.slice_mut(..m * n),
                    )
                    .expect("cuBLAS sgemm (matmul_bt_with_gpu_b) 失敗");
            }
        }

        let bc = self.buf_c.borrow();
        self.stream
            .memcpy_dtoh(&bc.buf.slice(..m * n), c_out)
            .expect("D2H c");
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
            self.stream
                .memcpy_htod(a, &mut ba.buf.slice_mut(..m * k))
                .expect("H2D a");
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
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas
                    .gemm(
                        cfg,
                        &b_gpu.slice(..k * n),
                        &ba.buf.slice(..m * k),
                        &mut bc.buf.slice_mut(..m * n),
                    )
                    .expect("cuBLAS sgemm (matmul_nn_with_gpu_b) 失敗");
            }
        }

        let bc = self.buf_c.borrow();
        self.stream
            .memcpy_dtoh(&bc.buf.slice(..m * n), c_out)
            .expect("D2H c");
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
            self.stream
                .memcpy_htod(a, &mut ba.buf.slice_mut(..m * k))
                .expect("A→GPU 転送失敗");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream
                .memcpy_htod(b, &mut bb.buf.slice_mut(..n * k))
                .expect("B→GPU 転送失敗");
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
                    .gemm(
                        cfg,
                        &bb.buf.slice(..n * k),
                        &ba.buf.slice(..m * k),
                        &mut bc.buf.slice_mut(..m * n),
                    )
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
            self.stream
                .memcpy_htod(a, &mut ba.buf.slice_mut(..m * k))
                .expect("H2D a");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream
                .memcpy_htod(b, &mut bb.buf.slice_mut(..n * k))
                .expect("H2D b");
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

        {
            let ba = self.buf_a.borrow();
            let bb = self.buf_b.borrow();
            let mut bs = self.buf_scores.borrow_mut();
            unsafe {
                self.blas
                    .gemm(
                        cfg,
                        &bb.buf.slice(..n * k),
                        &ba.buf.slice(..m * k),
                        &mut bs.buf.slice_mut(..m * n),
                    )
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
            self.stream
                .launch_builder(&self.softmax_func)
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
    pub fn matmul_nn_from_scores(
        &self,
        b: &[f32],
        c_out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        Self::ensure_buf(&self.stream, &self.buf_b, k * n);
        Self::ensure_buf(&self.stream, &self.buf_c, m * n);

        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream
                .memcpy_htod(b, &mut bb.buf.slice_mut(..k * n))
                .expect("H2D b");
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
            let bb = self.buf_b.borrow();
            let bs = self.buf_scores.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas
                    .gemm(
                        cfg,
                        &bb.buf.slice(..k * n),
                        &bs.buf.slice(..m * k),
                        &mut bc.buf.slice_mut(..m * n),
                    )
                    .expect("cuBLAS sgemm (matmul_nn_from_scores) 失敗");
            }
        }

        let bc = self.buf_c.borrow();
        self.stream
            .memcpy_dtoh(&bc.buf.slice(..m * n), c_out)
            .expect("D2H c");
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
            self.stream
                .memcpy_htod(a, &mut ba.buf.slice_mut(..m * k))
                .expect("A→GPU 転送失敗");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream
                .memcpy_htod(b, &mut bb.buf.slice_mut(..k * n))
                .expect("B→GPU 転送失敗");
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
                    .gemm(
                        cfg,
                        &bb.buf.slice(..k * n),
                        &ba.buf.slice(..m * k),
                        &mut bc.buf.slice_mut(..m * n),
                    )
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
    pub fn matmul_bt_inplace(
        &self,
        a: &[f32],
        b: &[f32],
        c_out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_b, n * k);
        Self::ensure_buf(&self.stream, &self.buf_c, m * n);

        {
            let mut ba = self.buf_a.borrow_mut();
            self.stream
                .memcpy_htod(&a[..m * k], &mut ba.buf.slice_mut(..m * k))
                .expect("H2D a");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream
                .memcpy_htod(&b[..n * k], &mut bb.buf.slice_mut(..n * k))
                .expect("H2D b");
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

        {
            let ba = self.buf_a.borrow();
            let bb = self.buf_b.borrow();
            let mut bc = self.buf_c.borrow_mut();
            unsafe {
                self.blas
                    .gemm(
                        cfg,
                        &bb.buf.slice(..n * k),
                        &ba.buf.slice(..m * k),
                        &mut bc.buf.slice_mut(..m * n),
                    )
                    .unwrap();
            }
        }

        let bc = self.buf_c.borrow();
        self.stream
            .memcpy_dtoh(&bc.buf.slice(..m * n), c_out)
            .expect("D2H c");
    }

    /// C\[m×n\] = A\[m×k\] × B\[k×n\] — アロケーションレス版。
    pub fn matmul_nn_inplace(
        &self,
        a: &[f32],
        b: &[f32],
        c_out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_b, k * n);
        Self::ensure_buf(&self.stream, &self.buf_c, m * n);

        {
            let mut ba = self.buf_a.borrow_mut();
            self.stream
                .memcpy_htod(&a[..m * k], &mut ba.buf.slice_mut(..m * k))
                .expect("H2D a");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream
                .memcpy_htod(&b[..k * n], &mut bb.buf.slice_mut(..k * n))
                .expect("H2D b");
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
                    .gemm(
                        cfg,
                        &bb.buf.slice(..k * n),
                        &ba.buf.slice(..m * k),
                        &mut bc.buf.slice_mut(..m * n),
                    )
                    .unwrap();
            }
        }

        let bc = self.buf_c.borrow();
        self.stream
            .memcpy_dtoh(&bc.buf.slice(..m * n), c_out)
            .expect("D2H c");
    }

    /// C\[k×n\] = A\[m×k\]^T × B\[m×n\] — アロケーションレス版。
    pub fn matmul_tn_inplace(
        &self,
        a: &[f32],
        b: &[f32],
        c_out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        Self::ensure_buf(&self.stream, &self.buf_a, m * k);
        Self::ensure_buf(&self.stream, &self.buf_b, m * n);
        Self::ensure_buf(&self.stream, &self.buf_c, k * n);

        {
            let mut ba = self.buf_a.borrow_mut();
            self.stream
                .memcpy_htod(a, &mut ba.buf.slice_mut(..m * k))
                .expect("H2D a");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream
                .memcpy_htod(b, &mut bb.buf.slice_mut(..m * n))
                .expect("H2D b");
        }

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
                    .gemm(
                        cfg,
                        &bb.buf.slice(..m * n),
                        &ba.buf.slice(..m * k),
                        &mut bc.buf.slice_mut(..k * n),
                    )
                    .unwrap();
            }
        }

        let bc = self.buf_c.borrow();
        self.stream
            .memcpy_dtoh(&bc.buf.slice(..k * n), c_out)
            .expect("D2H c");
    }

    // ── GPU 高速化カーネル ──────────────────────────────────────────────

    /// BF16 配列を FP32 に展開。CPU で bit shift → GPU 不要（bit操作のみで十分高速）。
    /// Rayon 版より高速: SIMD-friendly な単純ループ。
    pub fn cpu_bf16_expand_fast(src_u16: &[u16], dst_f32: &mut [f32]) {
        assert_eq!(src_u16.len(), dst_f32.len());
        // BF16→FP32 は単純な bit shift: (u16 as u32) << 16
        // メモリバウンドなのでGPU転送のオーバーヘッドより CPU SIMD が有利
        dst_f32.iter_mut().zip(src_u16.iter()).for_each(|(d, &s)| {
            *d = f32::from_bits((s as u32) << 16);
        });
    }

    /// Fused AdamW 更新を GPU で実行。delta, grad, m, v を一括更新。
    pub fn gpu_adamw_update(
        &self,
        delta: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        bc1: f32,
        bc2: f32,
        weight_decay: f32,
    ) {
        let n = delta.len();
        assert_eq!(n, grad.len());
        assert_eq!(n, m.len());
        assert_eq!(n, v.len());

        // 4つのバッファを GPU に転送 (buf_a=delta, buf_b=grad, buf_c=m, buf_d=v)
        Self::ensure_buf(&self.stream, &self.buf_a, n);
        Self::ensure_buf(&self.stream, &self.buf_b, n);
        Self::ensure_buf(&self.stream, &self.buf_c, n);
        Self::ensure_buf(&self.stream, &self.buf_d, n);

        {
            let mut ba = self.buf_a.borrow_mut();
            let mut bb = self.buf_b.borrow_mut();
            let mut bc = self.buf_c.borrow_mut();
            let mut bd = self.buf_d.borrow_mut();
            self.stream
                .memcpy_htod(delta, &mut ba.buf.slice_mut(..n))
                .expect("H2D delta");
            self.stream
                .memcpy_htod(grad, &mut bb.buf.slice_mut(..n))
                .expect("H2D grad");
            self.stream
                .memcpy_htod(m, &mut bc.buf.slice_mut(..n))
                .expect("H2D m");
            self.stream
                .memcpy_htod(v, &mut bd.buf.slice_mut(..n))
                .expect("H2D v");
        }

        let n_i32 = n as i32;
        let block = 256u32;
        let grid = ((n as u32 + block - 1) / block, 1, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut ba = self.buf_a.borrow_mut();
            let bb = self.buf_b.borrow();
            let mut bc = self.buf_c.borrow_mut();
            let mut bd = self.buf_d.borrow_mut();
            unsafe {
                self.stream
                    .launch_builder(&self.fused_adamw_func)
                    .arg(&mut ba.buf)
                    .arg(&bb.buf)
                    .arg(&mut bc.buf)
                    .arg(&mut bd.buf)
                    .arg(&n_i32)
                    .arg(&lr)
                    .arg(&beta1)
                    .arg(&beta2)
                    .arg(&eps)
                    .arg(&bc1)
                    .arg(&bc2)
                    .arg(&weight_decay)
                    .launch(cfg)
                    .expect("fused_adamw カーネル起動失敗");
            }
        }

        // D2H: delta, m, v を書き戻し（grad は読み取り専用）
        {
            let ba = self.buf_a.borrow();
            let bc = self.buf_c.borrow();
            let bd = self.buf_d.borrow();
            self.stream
                .memcpy_dtoh(&ba.buf.slice(..n), delta)
                .expect("D2H delta");
            self.stream
                .memcpy_dtoh(&bc.buf.slice(..n), m)
                .expect("D2H m");
            self.stream
                .memcpy_dtoh(&bd.buf.slice(..n), v)
                .expect("D2H v");
        }
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
        gate_silu
            .par_iter_mut()
            .zip(intermediate.par_iter_mut())
            .zip(gate.par_iter().zip(up.par_iter()))
            .for_each(|((gs, im), (&g, &u))| {
                let s = g * crate::fast_math::fast_sigmoid(g);
                *gs = s;
                *im = s * u;
            });

        let output = self.matmul_bt(
            &intermediate,
            down_proj,
            seq_len,
            hidden_dim,
            intermediate_dim,
        );
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
            self.stream
                .memcpy_htod(a, &mut ba.buf.slice_mut(..m * k))
                .expect("A→GPU 転送失敗");
        }
        {
            let mut bb = self.buf_b.borrow_mut();
            self.stream
                .memcpy_htod(b, &mut bb.buf.slice_mut(..m * n))
                .expect("B→GPU 転送失敗");
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
                    .gemm(
                        cfg,
                        &bb.buf.slice(..m * n),
                        &ba.buf.slice(..m * k),
                        &mut bc.buf.slice_mut(..k * n),
                    )
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
    // --- Weight grad バッファ（matmul_tn alloc 排除） ---
    pub wg_q_proj: Vec<f32>,
    pub wg_k_proj: Vec<f32>,
    pub wg_v_proj: Vec<f32>,
    pub wg_o_proj: Vec<f32>,
    pub wg_gate_proj: Vec<f32>,
    pub wg_up_proj: Vec<f32>,
    pub wg_down_proj: Vec<f32>,
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
            // Weight grad バッファ（事前確保で matmul_tn の alloc を排除）
            wg_q_proj: vec![0.0; hidden * q_dim],
            wg_k_proj: vec![0.0; kv_dim * hidden],
            wg_v_proj: vec![0.0; kv_dim * hidden],
            wg_o_proj: vec![0.0; q_dim * hidden],
            wg_gate_proj: vec![0.0; inter * hidden],
            wg_up_proj: vec![0.0; inter * hidden],
            wg_down_proj: vec![0.0; hidden * inter],
        };

        // Pinned Memory 化: mlock で全バッファをページロック。
        // PCIe DMA 直結になり、ドライバの隠れた CPU コピーが消滅。
        #[cfg(unix)]
        {
            let lock = |buf: &[f32]| unsafe {
                libc::mlock(buf.as_ptr().cast::<libc::c_void>(), buf.len() * 4);
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
        let count = self.normed.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.attn_out_raw.len()
            + self.attn_weights.len()
            + self.attn_out.len()
            + self.normed_ffn.len()
            + self.gate.len()
            + self.up.len()
            + self.gate_silu.len()
            + self.intermediate.len()
            + self.ffn_out.len()
            + self.d_intermediate.len()
            + self.d_gate.len()
            + self.d_up.len()
            + self.d_input_gate.len()
            + self.d_input_up.len()
            + self.swiglu_out.len()
            + self.d_pre_ffn.len()
            + self.d_pre_ffn_residual.len()
            + self.d_attn_output.len()
            + self.d_attn_raw.len()
            + self.d_q.len()
            + self.d_k.len()
            + self.d_v.len()
            + self.d_normed_q.len()
            + self.d_normed_k.len()
            + self.d_normed_v.len()
            + self.d_normed.len()
            + self.attn_out_raw_recompute.len()
            + self.d_input.len()
            + self.d_attn_norm_w.len()
            + self.d_ffn_norm_w.len();
        count * 4
    }
}

// ── CUDA GQA Attention ──────────────────────────────────────────────────

/// ヘッドデータ抽出: interleaved [seq_len, num_heads * head_dim] → [seq_len, head_dim]
fn extract_head(
    data: &[f32],
    head: usize,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    let stride = num_heads * head_dim;
    let mut out = vec![0.0f32; seq_len * head_dim];
    for t in 0..seq_len {
        let src = &data[t * stride + head * head_dim..t * stride + (head + 1) * head_dim];
        out[t * head_dim..(t + 1) * head_dim].copy_from_slice(src);
    }
    out
}

/// ヘッドデータ書き戻し: [seq_len, head_dim] → interleaved [seq_len, num_heads * head_dim]
fn scatter_head(
    src: &[f32],
    dst: &mut [f32],
    head: usize,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) {
    let stride = num_heads * head_dim;
    for t in 0..seq_len {
        dst[t * stride + head * head_dim..t * stride + (head + 1) * head_dim]
            .copy_from_slice(&src[t * head_dim..(t + 1) * head_dim]);
    }
}

/// ヘッドデータ加算: [seq_len, head_dim] を interleaved dst に加算（GQA 用）
fn accumulate_head(
    src: &[f32],
    dst: &mut [f32],
    head: usize,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) {
    let stride = num_heads * head_dim;
    for t in 0..seq_len {
        for d in 0..head_dim {
            dst[t * stride + head * head_dim + d] += src[t * head_dim + d];
        }
    }
}

/// Causal softmax (in-place): scores [seq_len, seq_len] に causal mask + softmax 適用
fn causal_softmax(scores: &mut [f32], seq_len: usize) {
    scores
        .par_chunks_exact_mut(seq_len)
        .enumerate()
        .for_each(|(t, row)| {
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
    d_scores
        .par_chunks_exact_mut(seq_len)
        .enumerate()
        .for_each(|(t, score_row)| {
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
    let mut q = cuda.matmul_bt(
        &normed,
        &weights.q_proj,
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );
    let mut k = cuda.matmul_bt(
        &normed,
        &weights.k_proj,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );
    let mut v = cuda.matmul_bt(
        &normed,
        &weights.v_proj,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );

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
    cuda_gqa_attention(
        cuda,
        &q,
        &k,
        &v,
        &mut attn_out_raw,
        &mut attn_weights,
        config,
        seq_len,
    );

    // 5. O projection (CUDA)
    let attn_out = cuda.matmul_bt(
        &attn_out_raw,
        &weights.o_proj,
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );

    // 6. Residual add
    for i in 0..input.len() {
        input[i] = residual_attn[i] + attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm (CPU)
    let residual_ffn = input.clone();
    let mut normed_ffn_buf = input.clone();
    rmsnorm(
        &mut normed_ffn_buf,
        &weights.ffn_norm,
        hidden_dim,
        config.norm_eps,
    );
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
    let mut q = cuda.matmul_bt(
        &normed,
        &weights.q_proj,
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );
    let mut k = cuda.matmul_bt(
        &normed,
        &weights.k_proj,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );
    let mut v = cuda.matmul_bt(
        &normed,
        &weights.v_proj,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );
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
    cuda_gqa_attention(
        cuda,
        &q,
        &k,
        &v,
        &mut attn_out_raw,
        &mut attn_weights_buf,
        config,
        seq_len,
    );
    drop(q);
    drop(k);
    drop(v);
    drop(attn_weights_buf);

    // 5. O projection (CUDA)
    let attn_out = cuda.matmul_bt(
        &attn_out_raw,
        &weights.o_proj,
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );
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
    rmsnorm(
        &mut normed_ffn_buf,
        &weights.ffn_norm,
        hidden_dim,
        config.norm_eps,
    );

    // 8. SwiGLU FFN (CUDA) — eval では中間値不要
    let gate = cuda.matmul_bt(
        &normed_ffn_buf,
        &weights.gate_proj,
        seq_len,
        intermediate_dim,
        hidden_dim,
    );
    let up = cuda.matmul_bt(
        &normed_ffn_buf,
        &weights.up_proj,
        seq_len,
        intermediate_dim,
        hidden_dim,
    );
    drop(normed_ffn_buf);

    let total = seq_len * intermediate_dim;
    let mut intermediate = vec![0.0f32; total];
    intermediate
        .par_iter_mut()
        .zip(gate.par_iter().zip(up.par_iter()))
        .for_each(|(im, (&g, &u))| {
            let s = g * crate::fast_math::fast_sigmoid(g);
            *im = s * u;
        });
    drop(gate);
    drop(up);

    let ffn_out = cuda.matmul_bt(
        &intermediate,
        &weights.down_proj,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );
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
            q_bias: if config.attention_bias {
                Some(vec![0.0; q_dim])
            } else {
                None
            },
            k_bias: if config.attention_bias {
                Some(vec![0.0; kv_dim])
            } else {
                None
            },
            v_bias: if config.attention_bias {
                Some(vec![0.0; kv_dim])
            } else {
                None
            },
            ffn_norm: vec![0.0; hidden],
            gate_proj: vec![0.0; inter * hidden],
            up_proj: vec![0.0; inter * hidden],
            down_proj: vec![0.0; hidden * inter],
        }
    }

    /// 別の勾配を要素ごとに加算（勾配累積）。Rayon並列。
    pub fn accumulate(&mut self, other: &Self) {
        let add = |dst: &mut [f32], src: &[f32]| {
            dst.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(d, s)| *d += *s);
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
        if let Some(ref mut b) = self.q_bias {
            b.par_iter_mut().for_each(|x| *x = 0.0);
        }
        if let Some(ref mut b) = self.k_bias {
            b.par_iter_mut().for_each(|x| *x = 0.0);
        }
        if let Some(ref mut b) = self.v_bias {
            b.par_iter_mut().for_each(|x| *x = 0.0);
        }
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
    let d_intermediate = cuda.matmul_nn(
        d_output,
        &weights.down_proj,
        seq_len,
        intermediate_dim,
        hidden_dim,
    );

    // SwiGLU elementwise backward (CPU — rayon 並列)
    let total = seq_len * intermediate_dim;
    let mut d_gate = vec![0.0f32; total];
    let mut d_up = vec![0.0f32; total];
    d_gate
        .par_iter_mut()
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
    let d_input_gate = cuda.matmul_nn(
        &d_gate,
        &weights.gate_proj,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );
    let d_input_up = cuda.matmul_nn(
        &d_up,
        &weights.up_proj,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );
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
    let d_gate_proj = cuda.matmul_tn(
        &d_gate,
        &cache.normed_ffn,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );
    // up_proj: [intermediate×hidden] ← d_up^T × normed_ffn
    let d_up_proj = cuda.matmul_tn(
        &d_up,
        &cache.normed_ffn,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );

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
        &d_attn_output,
        &attn_out_raw,
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );

    // QKV weight grads: W[out×in] ← d_out^T × normed_attn
    let d_q_proj = cuda.matmul_tn(
        &d_q,
        &cache.normed_attn,
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );
    let d_k_proj = cuda.matmul_tn(
        &d_k,
        &cache.normed_attn,
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
    );
    let d_v_proj = cuda.matmul_tn(
        &d_v,
        &cache.normed_attn,
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
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
    rmsnorm(
        &mut ws.normed[..hid_len],
        &weights.attn_norm,
        hidden_dim,
        config.norm_eps,
    );

    // 2. QKV projection (CUDA inplace)
    cuda.matmul_bt_inplace(
        &ws.normed[..hid_len],
        &weights.q_proj,
        &mut ws.q,
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );
    cuda.matmul_bt_inplace(
        &ws.normed[..hid_len],
        &weights.k_proj,
        &mut ws.k,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );
    cuda.matmul_bt_inplace(
        &ws.normed[..hid_len],
        &weights.v_proj,
        &mut ws.v,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );

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
    apply_rope(
        &mut ws.k,
        num_kv_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );

    // 4. GQA Attention (CUDA)
    cuda_gqa_attention(
        cuda,
        &ws.q,
        &ws.k,
        &ws.v,
        &mut ws.attn_out_raw,
        &mut ws.attn_weights,
        config,
        seq_len,
    );

    // 5. O projection (CUDA inplace)
    cuda.matmul_bt_inplace(
        &ws.attn_out_raw,
        &weights.o_proj,
        &mut ws.attn_out,
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );

    // 6. Residual add: input = residual_attn(ws.ffn_out) + attn_out
    for i in 0..hid_len {
        input[i] = ws.ffn_out[i] + ws.attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm
    ws.ffn_out[..hid_len].copy_from_slice(&input[..hid_len]); // residual_ffn
    ws.normed_ffn[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(
        &mut ws.normed_ffn[..hid_len],
        &weights.ffn_norm,
        hidden_dim,
        config.norm_eps,
    );

    // 8. SwiGLU FFN (CUDA inplace)
    let total = seq_len * intermediate_dim;
    cuda.matmul_bt_inplace(
        &ws.normed_ffn[..hid_len],
        &weights.gate_proj,
        &mut ws.gate[..total],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );
    cuda.matmul_bt_inplace(
        &ws.normed_ffn[..hid_len],
        &weights.up_proj,
        &mut ws.up[..total],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );

    ws.intermediate[..total]
        .par_iter_mut()
        .zip(ws.gate[..total].par_iter().zip(ws.up[..total].par_iter()))
        .for_each(|(im, (&g, &u))| {
            let s = g * crate::fast_math::fast_sigmoid(g);
            *im = s * u;
        });

    cuda.matmul_bt_inplace(
        &ws.intermediate[..total],
        &weights.down_proj,
        &mut ws.attn_out[..hid_len],
        seq_len,
        hidden_dim,
        intermediate_dim,
    );
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
        (self.q_proj.len()
            + self.k_proj.len()
            + self.v_proj.len()
            + self.o_proj.len()
            + self.gate_proj.len()
            + self.up_proj.len()
            + self.down_proj.len())
            * 4
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
    rmsnorm(
        &mut ws.normed[..hid_len],
        &cpu_weights.attn_norm,
        hidden_dim,
        config.norm_eps,
    );

    // 2. QKV projection — VRAM 上の重みを直接使用 (H2D 重み転送ゼロ)
    cuda.matmul_bt_with_gpu_b(
        &ws.normed[..hid_len],
        &vram.q_proj,
        &mut ws.q,
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );
    cuda.matmul_bt_with_gpu_b(
        &ws.normed[..hid_len],
        &vram.k_proj,
        &mut ws.k,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );
    cuda.matmul_bt_with_gpu_b(
        &ws.normed[..hid_len],
        &vram.v_proj,
        &mut ws.v,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );

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
    apply_rope(
        &mut ws.k,
        num_kv_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );

    // 4. GQA Attention (CUDA — per-head matmul は小行列なので buf_b 経由で十分)
    cuda_gqa_attention(
        cuda,
        &ws.q,
        &ws.k,
        &ws.v,
        &mut ws.attn_out_raw,
        &mut ws.attn_weights,
        config,
        seq_len,
    );

    // 5. O projection — VRAM 常駐
    cuda.matmul_bt_with_gpu_b(
        &ws.attn_out_raw,
        &vram.o_proj,
        &mut ws.attn_out,
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );

    // 6. Residual add
    for i in 0..hid_len {
        input[i] = ws.ffn_out[i] + ws.attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm
    ws.ffn_out[..hid_len].copy_from_slice(&input[..hid_len]);
    ws.normed_ffn[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(
        &mut ws.normed_ffn[..hid_len],
        &cpu_weights.ffn_norm,
        hidden_dim,
        config.norm_eps,
    );

    // 8. SwiGLU FFN — VRAM 常駐重み
    let total = seq_len * intermediate_dim;
    cuda.matmul_bt_with_gpu_b(
        &ws.normed_ffn[..hid_len],
        &vram.gate_proj,
        &mut ws.gate[..total],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );
    cuda.matmul_bt_with_gpu_b(
        &ws.normed_ffn[..hid_len],
        &vram.up_proj,
        &mut ws.up[..total],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );

    ws.intermediate[..total]
        .par_iter_mut()
        .zip(ws.gate[..total].par_iter().zip(ws.up[..total].par_iter()))
        .for_each(|(im, (&g, &u))| {
            let s = g * crate::fast_math::fast_sigmoid(g);
            *im = s * u;
        });

    cuda.matmul_bt_with_gpu_b(
        &ws.intermediate[..total],
        &vram.down_proj,
        &mut ws.attn_out[..hid_len],
        seq_len,
        hidden_dim,
        intermediate_dim,
    );

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
    rmsnorm(
        &mut ws.normed[..hid_len],
        &weights.attn_norm,
        hidden_dim,
        config.norm_eps,
    );
    let normed_attn = ws.normed[..hid_len].to_vec();

    // 2. QKV projection (CUDA inplace into workspace)
    cuda.matmul_bt_inplace(
        &ws.normed[..hid_len],
        &weights.q_proj,
        &mut ws.q[..q_len],
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );
    cuda.matmul_bt_inplace(
        &ws.normed[..hid_len],
        &weights.k_proj,
        &mut ws.k[..kv_len],
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );
    cuda.matmul_bt_inplace(
        &ws.normed[..hid_len],
        &weights.v_proj,
        &mut ws.v[..kv_len],
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );

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
    apply_rope(
        &mut ws.q[..q_len],
        num_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );
    apply_rope(
        &mut ws.k[..kv_len],
        num_kv_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );

    // LayerCache 用にコピー（backward で必要）
    let q_cache = ws.q[..q_len].to_vec();
    let k_cache = ws.k[..kv_len].to_vec();
    let v_cache = ws.v[..kv_len].to_vec();

    // 4. GQA Attention
    cuda_gqa_attention(
        cuda,
        &ws.q[..q_len],
        &ws.k[..kv_len],
        &ws.v[..kv_len],
        &mut ws.attn_out_raw[..q_len],
        &mut ws.attn_weights[..aw_len],
        config,
        seq_len,
    );
    let attn_weights_cache = ws.attn_weights[..aw_len].to_vec();

    // 5. O projection
    cuda.matmul_bt_inplace(
        &ws.attn_out_raw[..q_len],
        &weights.o_proj,
        &mut ws.attn_out[..hid_len],
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );
    let attn_out_cache = ws.attn_out[..hid_len].to_vec();

    // 6. Residual add
    for i in 0..hid_len {
        input[i] = residual_attn[i] + ws.attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm
    let residual_ffn = input[..hid_len].to_vec();
    ws.normed_ffn[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(
        &mut ws.normed_ffn[..hid_len],
        &weights.ffn_norm,
        hidden_dim,
        config.norm_eps,
    );
    let normed_ffn_cache = ws.normed_ffn[..hid_len].to_vec();

    // 8. SwiGLU FFN
    cuda.matmul_bt_inplace(
        &ws.normed_ffn[..hid_len],
        &weights.gate_proj,
        &mut ws.gate[..inter_len],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );
    cuda.matmul_bt_inplace(
        &ws.normed_ffn[..hid_len],
        &weights.up_proj,
        &mut ws.up[..inter_len],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );

    // SiLU(gate) ⊙ up
    ws.gate_silu[..inter_len]
        .par_iter_mut()
        .zip(ws.intermediate[..inter_len].par_iter_mut())
        .zip(
            ws.gate[..inter_len]
                .par_iter()
                .zip(ws.up[..inter_len].par_iter()),
        )
        .for_each(|((gs, im), (&g, &u))| {
            let s = g * crate::fast_math::fast_sigmoid(g);
            *gs = s;
            *im = s * u;
        });

    let gate_cache = ws.gate[..inter_len].to_vec();
    let up_cache = ws.up[..inter_len].to_vec();
    let gate_silu_cache = ws.gate_silu[..inter_len].to_vec();

    cuda.matmul_bt_inplace(
        &ws.intermediate[..inter_len],
        &weights.down_proj,
        &mut ws.ffn_out[..hid_len],
        seq_len,
        hidden_dim,
        intermediate_dim,
    );

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
    rmsnorm(
        &mut ws.normed[..hid_len],
        &cpu_weights.attn_norm,
        hidden_dim,
        config.norm_eps,
    );
    let normed_attn = ws.normed[..hid_len].to_vec();

    // 2. QKV projection — VRAM 常駐
    cuda.matmul_bt_with_gpu_b(
        &ws.normed[..hid_len],
        &vram.q_proj,
        &mut ws.q[..q_len],
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );
    cuda.matmul_bt_with_gpu_b(
        &ws.normed[..hid_len],
        &vram.k_proj,
        &mut ws.k[..kv_len],
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );
    cuda.matmul_bt_with_gpu_b(
        &ws.normed[..hid_len],
        &vram.v_proj,
        &mut ws.v[..kv_len],
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );

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
    apply_rope(
        &mut ws.q[..q_len],
        num_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );
    apply_rope(
        &mut ws.k[..kv_len],
        num_kv_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );

    let q_cache = ws.q[..q_len].to_vec();
    let k_cache = ws.k[..kv_len].to_vec();
    let v_cache = ws.v[..kv_len].to_vec();

    // 4. GQA Attention
    cuda_gqa_attention(
        cuda,
        &ws.q[..q_len],
        &ws.k[..kv_len],
        &ws.v[..kv_len],
        &mut ws.attn_out_raw[..q_len],
        &mut ws.attn_weights[..aw_len],
        config,
        seq_len,
    );
    let attn_weights_cache = ws.attn_weights[..aw_len].to_vec();

    // 5. O projection — VRAM 常駐
    cuda.matmul_bt_with_gpu_b(
        &ws.attn_out_raw[..q_len],
        &vram.o_proj,
        &mut ws.attn_out[..hid_len],
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );
    let attn_out_cache = ws.attn_out[..hid_len].to_vec();

    // 6. Residual add
    for i in 0..hid_len {
        input[i] = residual_attn[i] + ws.attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm
    let residual_ffn = input[..hid_len].to_vec();
    ws.normed_ffn[..hid_len].copy_from_slice(&input[..hid_len]);
    rmsnorm(
        &mut ws.normed_ffn[..hid_len],
        &cpu_weights.ffn_norm,
        hidden_dim,
        config.norm_eps,
    );
    let normed_ffn_cache = ws.normed_ffn[..hid_len].to_vec();

    // 8. SwiGLU FFN — VRAM 常駐
    cuda.matmul_bt_with_gpu_b(
        &ws.normed_ffn[..hid_len],
        &vram.gate_proj,
        &mut ws.gate[..inter_len],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );
    cuda.matmul_bt_with_gpu_b(
        &ws.normed_ffn[..hid_len],
        &vram.up_proj,
        &mut ws.up[..inter_len],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );

    ws.gate_silu[..inter_len]
        .par_iter_mut()
        .zip(ws.intermediate[..inter_len].par_iter_mut())
        .zip(
            ws.gate[..inter_len]
                .par_iter()
                .zip(ws.up[..inter_len].par_iter()),
        )
        .for_each(|((gs, im), (&g, &u))| {
            let s = g * crate::fast_math::fast_sigmoid(g);
            *gs = s;
            *im = s * u;
        });

    let gate_cache = ws.gate[..inter_len].to_vec();
    let up_cache = ws.up[..inter_len].to_vec();
    let gate_silu_cache = ws.gate_silu[..inter_len].to_vec();

    cuda.matmul_bt_with_gpu_b(
        &ws.intermediate[..inter_len],
        &vram.down_proj,
        &mut ws.ffn_out[..hid_len],
        seq_len,
        hidden_dim,
        intermediate_dim,
    );

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
    cuda.matmul_nn_inplace(
        d_output,
        &weights.down_proj,
        &mut ws.d_intermediate[..inter_len],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );

    // SwiGLU elementwise backward (CPU — rayon 並列)
    ws.d_gate[..inter_len]
        .par_iter_mut()
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
    cuda.matmul_nn_inplace(
        &ws.d_gate[..inter_len],
        &weights.gate_proj,
        &mut ws.d_input_gate[..hid_len],
        seq_len,
        hidden_dim,
        intermediate_dim,
    );
    cuda.matmul_nn_inplace(
        &ws.d_up[..inter_len],
        &weights.up_proj,
        &mut ws.d_input_up[..hid_len],
        seq_len,
        hidden_dim,
        intermediate_dim,
    );

    // d_pre_ffn = d_input_gate + d_input_up
    for i in 0..hid_len {
        ws.d_pre_ffn[i] = ws.d_input_gate[i] + ws.d_input_up[i];
    }

    // SwiGLU intermediate for down_proj grad
    for idx in 0..inter_len {
        ws.swiglu_out[idx] = cache.gate_silu[idx] * cache.up[idx];
    }

    // FFN weight grads (matmul_tn)
    let d_down_proj = cuda.matmul_tn(
        d_output,
        &ws.swiglu_out[..inter_len],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );
    let d_gate_proj = cuda.matmul_tn(
        &ws.d_gate[..inter_len],
        &cache.normed_ffn,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );
    let d_up_proj = cuda.matmul_tn(
        &ws.d_up[..inter_len],
        &cache.normed_ffn,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );

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
    cuda.matmul_nn_inplace(
        &ws.d_attn_output[..hid_len],
        &weights.o_proj,
        &mut ws.d_attn_raw[..q_len],
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );

    // 5. GQA Attention backward
    ws.d_q[..q_len].iter_mut().for_each(|x| *x = 0.0);
    ws.d_k[..kv_len].iter_mut().for_each(|x| *x = 0.0);
    ws.d_v[..kv_len].iter_mut().for_each(|x| *x = 0.0);
    cuda_gqa_attention_backward(
        cuda,
        &ws.d_attn_raw[..q_len],
        &cache.attn_weights,
        &cache.q,
        &cache.k,
        &cache.v,
        &mut ws.d_q[..q_len],
        &mut ws.d_k[..kv_len],
        &mut ws.d_v[..kv_len],
        config,
        seq_len,
    );

    // 6. RoPE backward
    crate::llama_backward::rope_backward(
        &mut ws.d_q[..q_len],
        num_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );
    crate::llama_backward::rope_backward(
        &mut ws.d_k[..kv_len],
        num_kv_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );

    // 7. QKV projection backward
    cuda.matmul_nn_inplace(
        &ws.d_q[..q_len],
        &weights.q_proj,
        &mut ws.d_normed_q[..hid_len],
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );
    cuda.matmul_nn_inplace(
        &ws.d_k[..kv_len],
        &weights.k_proj,
        &mut ws.d_normed_k[..hid_len],
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
    );
    cuda.matmul_nn_inplace(
        &ws.d_v[..kv_len],
        &weights.v_proj,
        &mut ws.d_normed_v[..hid_len],
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
    );

    for i in 0..hid_len {
        ws.d_normed[i] = ws.d_normed_q[i] + ws.d_normed_k[i] + ws.d_normed_v[i];
    }

    // O projection weight grad: attn_out_raw を再計算
    {
        let kv_group_size = num_heads / num_kv_heads;
        ws.attn_out_raw_recompute[..q_len]
            .iter_mut()
            .for_each(|x| *x = 0.0);
        for h in 0..num_heads {
            let kv_h = h / kv_group_size;
            let aw_h = &cache.attn_weights[h * seq_len * seq_len..(h + 1) * seq_len * seq_len];
            let v_h = extract_head(&cache.v, kv_h, num_kv_heads, head_dim, seq_len);
            let out_h = cuda.matmul_nn(aw_h, &v_h, seq_len, head_dim, seq_len);
            scatter_head(
                &out_h,
                &mut ws.attn_out_raw_recompute[..q_len],
                h,
                num_heads,
                head_dim,
                seq_len,
            );
        }
    }
    let d_o_proj = cuda.matmul_tn(
        &ws.d_attn_output[..hid_len],
        &ws.attn_out_raw_recompute[..q_len],
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );

    // QKV weight grads
    let d_q_proj = cuda.matmul_tn(
        &ws.d_q[..q_len],
        &cache.normed_attn,
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );
    let d_k_proj = cuda.matmul_tn(
        &ws.d_k[..kv_len],
        &cache.normed_attn,
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
    );
    let d_v_proj = cuda.matmul_tn(
        &ws.d_v[..kv_len],
        &cache.normed_attn,
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
    );

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
    } else {
        None
    };
    let d_k_bias = if weights.k_bias.is_some() {
        Some(compute_bias_grad(
            &ws.d_k[..kv_len],
            num_kv_heads * head_dim,
        ))
    } else {
        None
    };
    let d_v_bias = if weights.v_bias.is_some() {
        Some(compute_bias_grad(
            &ws.d_v[..kv_len],
            num_kv_heads * head_dim,
        ))
    } else {
        None
    };

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
    let _t0 = std::time::Instant::now();

    // 1. down_proj backward — VRAM 常駐
    cuda.matmul_nn_with_gpu_b(
        d_output,
        &vram.down_proj,
        &mut ws.d_intermediate[..inter_len],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );
    let _t1 = std::time::Instant::now();

    // SwiGLU elementwise backward (CPU — rayon 並列)
    ws.d_gate[..inter_len]
        .par_iter_mut()
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
    let _t2 = std::time::Instant::now();

    // gate/up_proj backward — VRAM 常駐
    cuda.matmul_nn_with_gpu_b(
        &ws.d_gate[..inter_len],
        &vram.gate_proj,
        &mut ws.d_input_gate[..hid_len],
        seq_len,
        hidden_dim,
        intermediate_dim,
    );
    cuda.matmul_nn_with_gpu_b(
        &ws.d_up[..inter_len],
        &vram.up_proj,
        &mut ws.d_input_up[..hid_len],
        seq_len,
        hidden_dim,
        intermediate_dim,
    );
    let _t3 = std::time::Instant::now();

    for i in 0..hid_len {
        ws.d_pre_ffn[i] = ws.d_input_gate[i] + ws.d_input_up[i];
    }

    // SwiGLU intermediate for down_proj grad
    for idx in 0..inter_len {
        ws.swiglu_out[idx] = cache.gate_silu[idx] * cache.up[idx];
    }
    let _t4 = std::time::Instant::now();

    // FFN weight grads (matmul_tn — activations 使用、VRAM 不要)
    let d_down_proj = cuda.matmul_tn(
        d_output,
        &ws.swiglu_out[..inter_len],
        seq_len,
        intermediate_dim,
        hidden_dim,
    );
    let d_gate_proj = cuda.matmul_tn(
        &ws.d_gate[..inter_len],
        &cache.normed_ffn,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );
    let d_up_proj = cuda.matmul_tn(
        &ws.d_up[..inter_len],
        &cache.normed_ffn,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );
    let _t5 = std::time::Instant::now();

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
    let _t6 = std::time::Instant::now();

    // 3. Residual
    for i in 0..hid_len {
        ws.d_attn_output[i] = d_output[i] + ws.d_pre_ffn_residual[i];
    }

    // ── Attention Backward ──

    // 4. O projection backward — VRAM 常駐
    cuda.matmul_nn_with_gpu_b(
        &ws.d_attn_output[..hid_len],
        &vram.o_proj,
        &mut ws.d_attn_raw[..q_len],
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );
    let _t7 = std::time::Instant::now();

    // 5. GQA Attention backward
    ws.d_q[..q_len].iter_mut().for_each(|x| *x = 0.0);
    ws.d_k[..kv_len].iter_mut().for_each(|x| *x = 0.0);
    ws.d_v[..kv_len].iter_mut().for_each(|x| *x = 0.0);
    cuda_gqa_attention_backward(
        cuda,
        &ws.d_attn_raw[..q_len],
        &cache.attn_weights,
        &cache.q,
        &cache.k,
        &cache.v,
        &mut ws.d_q[..q_len],
        &mut ws.d_k[..kv_len],
        &mut ws.d_v[..kv_len],
        config,
        seq_len,
    );
    let _t8 = std::time::Instant::now();

    // 6. RoPE backward
    crate::llama_backward::rope_backward(
        &mut ws.d_q[..q_len],
        num_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );
    crate::llama_backward::rope_backward(
        &mut ws.d_k[..kv_len],
        num_kv_heads,
        head_dim,
        seq_len,
        config.rope_theta,
    );
    let _t9 = std::time::Instant::now();

    // 7. QKV projection backward — VRAM 常駐
    cuda.matmul_nn_with_gpu_b(
        &ws.d_q[..q_len],
        &vram.q_proj,
        &mut ws.d_normed_q[..hid_len],
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );
    cuda.matmul_nn_with_gpu_b(
        &ws.d_k[..kv_len],
        &vram.k_proj,
        &mut ws.d_normed_k[..hid_len],
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
    );
    cuda.matmul_nn_with_gpu_b(
        &ws.d_v[..kv_len],
        &vram.v_proj,
        &mut ws.d_normed_v[..hid_len],
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
    );
    let _t10 = std::time::Instant::now();

    for i in 0..hid_len {
        ws.d_normed[i] = ws.d_normed_q[i] + ws.d_normed_k[i] + ws.d_normed_v[i];
    }

    // O projection weight grad: attn_out_raw 再計算
    {
        let kv_group_size = num_heads / num_kv_heads;
        ws.attn_out_raw_recompute[..q_len]
            .iter_mut()
            .for_each(|x| *x = 0.0);
        for h in 0..num_heads {
            let kv_h = h / kv_group_size;
            let aw_h = &cache.attn_weights[h * seq_len * seq_len..(h + 1) * seq_len * seq_len];
            let v_h = extract_head(&cache.v, kv_h, num_kv_heads, head_dim, seq_len);
            let out_h = cuda.matmul_nn(aw_h, &v_h, seq_len, head_dim, seq_len);
            scatter_head(
                &out_h,
                &mut ws.attn_out_raw_recompute[..q_len],
                h,
                num_heads,
                head_dim,
                seq_len,
            );
        }
    }
    let _t11 = std::time::Instant::now();

    let d_o_proj = cuda.matmul_tn(
        &ws.d_attn_output[..hid_len],
        &ws.attn_out_raw_recompute[..q_len],
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );

    // QKV weight grads
    let d_q_proj = cuda.matmul_tn(
        &ws.d_q[..q_len],
        &cache.normed_attn,
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );
    let d_k_proj = cuda.matmul_tn(
        &ws.d_k[..kv_len],
        &cache.normed_attn,
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
    );
    let d_v_proj = cuda.matmul_tn(
        &ws.d_v[..kv_len],
        &cache.normed_attn,
        seq_len,
        hidden_dim,
        num_kv_heads * head_dim,
    );
    let _t12 = std::time::Instant::now();

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
    let _t13 = std::time::Instant::now();

    // 9. Residual
    for i in 0..hid_len {
        ws.d_input[i] += ws.d_attn_output[i];
    }

    // --- Backward profiling (1レイヤー分、最初の数ステップのみ出力) ---
    let ms =
        |a: std::time::Instant, b: std::time::Instant| b.duration_since(a).as_secs_f64() * 1000.0;
    eprintln!("    [BWD-PROFILE] down={:.1} swiglu={:.1} gate_up={:.1} elem={:.1} ffn_wg={:.1} rmsn1={:.1} o_bwd={:.1} gqa={:.1} rope={:.1} qkv_bwd={:.1} recomp={:.1} attn_wg={:.1} rmsn2={:.1} total={:.1}ms",
        ms(_t0,_t1), ms(_t1,_t2), ms(_t2,_t3), ms(_t3,_t4), ms(_t4,_t5), ms(_t5,_t6),
        ms(_t6,_t7), ms(_t7,_t8), ms(_t8,_t9), ms(_t9,_t10), ms(_t10,_t11), ms(_t11,_t12),
        ms(_t12,_t13), ms(_t0,_t13));

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
    } else {
        None
    };
    let d_k_bias = if cpu_weights.k_bias.is_some() {
        Some(compute_bias_grad(
            &ws.d_k[..kv_len],
            num_kv_heads * head_dim,
        ))
    } else {
        None
    };
    let d_v_bias = if cpu_weights.v_bias.is_some() {
        Some(compute_bias_grad(
            &ws.d_v[..kv_len],
            num_kv_heads * head_dim,
        ))
    } else {
        None
    };

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

// ============================================================================
// Qwen3.5 DeltaNet CUDA Forward (projection 部分のみ GPU 化)
// ============================================================================

/// Qwen3.5 DeltaNet レイヤーの projection 部分を CUDA で高速化する forward。
///
/// 再帰部分 (状態行列更新) は CPU で実行し、projection (matmul) のみ GPU。
/// 9B モデルの projection は hidden=4096 × key/value_dim=2048-4096 なので
/// cuBLAS sgemm の恩恵が大きい。
///
/// # 引数
///
/// - `cuda`: CudaMatmul インスタンス
/// - `input`: (seq_len × hidden_size) — 正規化済み入力
/// - `weights`: DeltaNet レイヤーの重み
/// - `config`: Qwen3.5 設定
/// - `seq_len`: シーケンス長
/// - `qkv_out`: (seq_len × (key_dim*2 + value_dim)) — QKV projection 出力
/// - `z_out`: (seq_len × value_dim) — output gate 出力
/// - `b_out`: (seq_len × num_v_heads) — beta logits 出力
/// - `a_out`: (seq_len × num_v_heads) — alpha logits 出力
/// - `attn_normed`: (seq_len × value_dim) — norm後の出力
/// - `layer_out`: (seq_len × hidden_size) — output projection 後
pub fn cuda_deltanet_projections(
    cuda: &CudaMatmul,
    input: &[f32],
    in_proj_qkv: &[f32],
    in_proj_z: &[f32],
    in_proj_b: &[f32],
    in_proj_a: &[f32],
    out_proj: &[f32],
    qkv_out: &mut [f32],
    z_out: &mut [f32],
    b_out: &mut [f32],
    a_out: &mut [f32],
    attn_normed: &[f32],
    layer_out: &mut [f32],
    seq_len: usize,
    hidden: usize,
    qkv_dim: usize,
    value_dim: usize,
    n_v_heads: usize,
) {
    // in_proj_qkv: (qkv_dim × hidden) — QKV projection
    cuda.matmul_bt_inplace(input, in_proj_qkv, qkv_out, seq_len, qkv_dim, hidden);

    // in_proj_z: (value_dim × hidden) — output gate
    cuda.matmul_bt_inplace(input, in_proj_z, z_out, seq_len, value_dim, hidden);

    // in_proj_b: (n_v_heads × hidden) — beta logits
    cuda.matmul_bt_inplace(input, in_proj_b, b_out, seq_len, n_v_heads, hidden);

    // in_proj_a: (n_v_heads × hidden) — alpha logits
    cuda.matmul_bt_inplace(input, in_proj_a, a_out, seq_len, n_v_heads, hidden);

    // out_proj: (hidden × value_dim) — output projection (attn_normed → layer_out)
    cuda.matmul_bt_inplace(attn_normed, out_proj, layer_out, seq_len, hidden, value_dim);
}

/// Qwen3.5 Full Attention レイヤーの QKV + O projection を CUDA で実行。
pub fn cuda_full_attn_projections(
    cuda: &CudaMatmul,
    input: &[f32],
    q_proj: &[f32],
    k_proj: &[f32],
    v_proj: &[f32],
    o_proj: &[f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    attn_out_raw: &[f32],
    o_out: &mut [f32],
    seq_len: usize,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
) {
    cuda.matmul_bt_inplace(input, q_proj, q_out, seq_len, q_dim, hidden);
    cuda.matmul_bt_inplace(input, k_proj, k_out, seq_len, kv_dim, hidden);
    cuda.matmul_bt_inplace(input, v_proj, v_out, seq_len, kv_dim, hidden);
    cuda.matmul_bt_inplace(attn_out_raw, o_proj, o_out, seq_len, hidden, q_dim);
}

/// Qwen3.5 SwiGLU FFN の projection を CUDA で実行。
pub fn cuda_qwen35_swiglu_fwd(
    cuda: &CudaMatmul,
    input: &[f32],
    gate_proj: &[f32],
    up_proj: &[f32],
    down_proj: &[f32],
    gate_out: &mut [f32],
    up_out: &mut [f32],
    intermediate: &mut [f32],
    ffn_out: &mut [f32],
    seq_len: usize,
    hidden: usize,
    inter: usize,
) {
    cuda.matmul_bt_inplace(input, gate_proj, gate_out, seq_len, inter, hidden);
    cuda.matmul_bt_inplace(input, up_proj, up_out, seq_len, inter, hidden);

    // SiLU(gate) ⊙ up → intermediate (CPU — 要素wise演算はGPU転送不要)
    let total = seq_len * inter;
    for i in 0..total {
        let g = gate_out[i];
        let silu_g = g / (1.0 + (-g).exp());
        intermediate[i] = silu_g * up_out[i];
    }

    cuda.matmul_bt_inplace(intermediate, down_proj, ffn_out, seq_len, hidden, inter);
}

/// DeltaNet 再帰を GPU で実行。
///
/// 全ヘッドを同時に並列処理。1ブロック=1ヘッド、blockDim.x=dv。
/// 状態行列 S (dk×dv) を shared memory に保持し、seq_len ステップの
/// 再帰を GPU 上で完結させる。PCIe 転送は入出力データのみ。
///
/// # 入力レイアウト (per-head contiguous)
///
/// - `q`: [num_heads × seq_len × dk]
/// - `k`: [num_heads × seq_len × dk]
/// - `v`: [num_heads × seq_len × dv]
/// - `beta`: [num_heads × seq_len]
/// - `g`: [num_heads × seq_len]
///
/// # 出力
///
/// - `output`: [num_heads × seq_len × dv]
pub fn cuda_deltanet_recurrence(
    cuda: &CudaMatmul,
    q: &[f32],          // [num_heads * seq_len * dk]
    k: &[f32],          // [num_heads * seq_len * dk]
    v: &[f32],          // [num_heads * seq_len * dv]
    beta: &[f32],       // [num_heads * seq_len]
    g: &[f32],          // [num_heads * seq_len]
    output: &mut [f32], // [num_heads * seq_len * dv]
    num_heads: usize,
    seq_len: usize,
    dk: usize,
    dv: usize,
) {
    let total_qk = num_heads * seq_len * dk;
    let total_v = num_heads * seq_len * dv;
    let total_bg = num_heads * seq_len;

    // H2D 転送
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_a, total_qk);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_b, total_qk);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_c, total_v);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_d, total_bg);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_e, total_bg);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_scores, total_v);

    {
        let mut ba = cuda.buf_a.borrow_mut();
        cuda.stream
            .memcpy_htod(&q[..total_qk], &mut ba.buf.slice_mut(..total_qk))
            .expect("q H2D 失敗");
    }
    {
        let mut bb = cuda.buf_b.borrow_mut();
        cuda.stream
            .memcpy_htod(&k[..total_qk], &mut bb.buf.slice_mut(..total_qk))
            .expect("k H2D 失敗");
    }
    {
        let mut bc = cuda.buf_c.borrow_mut();
        cuda.stream
            .memcpy_htod(&v[..total_v], &mut bc.buf.slice_mut(..total_v))
            .expect("v H2D 失敗");
    }
    {
        let mut bd = cuda.buf_d.borrow_mut();
        cuda.stream
            .memcpy_htod(&beta[..total_bg], &mut bd.buf.slice_mut(..total_bg))
            .expect("beta H2D 失敗");
    }
    {
        let mut be = cuda.buf_e.borrow_mut();
        cuda.stream
            .memcpy_htod(&g[..total_bg], &mut be.buf.slice_mut(..total_bg))
            .expect("g H2D 失敗");
    }

    // State buffer in global memory: [num_heads, dk, dv]
    let state_size = num_heads * dk * dv;
    let mut state_gpu: CudaSlice<f32> = cuda
        .stream
        .alloc_zeros(state_size)
        .expect("state_buf 確保失敗");

    // Shared memory: k_buf (dk) + q_buf (dk) のみ
    let shared_bytes = dk * 2 * 4;

    // カーネル起動
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (dv as u32, 1, 1),
        shared_mem_bytes: shared_bytes as u32,
    };

    {
        let ba = cuda.buf_a.borrow();
        let bb = cuda.buf_b.borrow();
        let bc = cuda.buf_c.borrow();
        let bd = cuda.buf_d.borrow();
        let be = cuda.buf_e.borrow();
        let mut bs = cuda.buf_scores.borrow_mut();

        let mut builder = cuda.stream.launch_builder(&cuda.deltanet_func);
        builder.arg(&ba.buf); // q
        builder.arg(&bb.buf); // k
        builder.arg(&bc.buf); // v
        builder.arg(&bd.buf); // beta
        builder.arg(&be.buf); // g
        builder.arg(&mut bs.buf); // output
        builder.arg(&mut state_gpu); // state_buf (global memory)
        let seq_len_i32 = seq_len as i32;
        let dk_i32 = dk as i32;
        let dv_i32 = dv as i32;
        builder.arg(&seq_len_i32); // seq_len
        builder.arg(&dk_i32); // dk
        builder.arg(&dv_i32); // dv

        unsafe {
            builder
                .launch(cfg)
                .expect("deltanet_recurrence カーネル起動失敗");
        }
    }

    // D2H 転送
    {
        let bs = cuda.buf_scores.borrow();
        cuda.stream
            .memcpy_dtoh(&bs.buf.slice(..total_v), &mut output[..total_v])
            .expect("output D2H 失敗");
    }
}

/// GPU DeltaNet forward (訓練用) — 全timestepの S_{t-1} と e を保存して返す
pub fn cuda_deltanet_recurrence_train(
    cuda: &CudaMatmul,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    output: &mut [f32],
    all_s_prev: &mut [f32],
    all_e: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    dk: usize,
    dv: usize,
) {
    let total_qk = num_heads * seq_len * dk;
    let total_v = num_heads * seq_len * dv;
    let total_bg = num_heads * seq_len;
    let total_states = num_heads * seq_len * dk * dv;
    let total_e = num_heads * seq_len * dv;

    // H2D: q, k, v, beta, g
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_a, total_qk);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_b, total_qk);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_c, total_v);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_d, total_bg);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_e, total_bg);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_scores, total_v);

    {
        let mut ba = cuda.buf_a.borrow_mut();
        cuda.stream.memcpy_htod(&q[..total_qk], &mut ba.buf.slice_mut(..total_qk)).expect("q H2D");
    }
    {
        let mut bb = cuda.buf_b.borrow_mut();
        cuda.stream.memcpy_htod(&k[..total_qk], &mut bb.buf.slice_mut(..total_qk)).expect("k H2D");
    }
    {
        let mut bc = cuda.buf_c.borrow_mut();
        cuda.stream.memcpy_htod(&v[..total_v], &mut bc.buf.slice_mut(..total_v)).expect("v H2D");
    }
    {
        let mut bd = cuda.buf_d.borrow_mut();
        cuda.stream.memcpy_htod(&beta[..total_bg], &mut bd.buf.slice_mut(..total_bg)).expect("beta H2D");
    }
    {
        let mut be_buf = cuda.buf_e.borrow_mut();
        cuda.stream.memcpy_htod(&g[..total_bg], &mut be_buf.buf.slice_mut(..total_bg)).expect("g H2D");
    }

    // GPU バッファ: state, all_s_prev, all_e
    let state_size = num_heads * dk * dv;
    let mut state_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(state_size).expect("state_buf 確保失敗");
    let mut all_s_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_states).expect("all_s_prev 確保失敗");
    let mut all_e_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_e).expect("all_e 確保失敗");

    let shared_bytes = dk * 2 * 4;
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (dv as u32, 1, 1),
        shared_mem_bytes: shared_bytes as u32,
    };

    {
        let ba = cuda.buf_a.borrow();
        let bb = cuda.buf_b.borrow();
        let bc = cuda.buf_c.borrow();
        let bd = cuda.buf_d.borrow();
        let be_buf = cuda.buf_e.borrow();
        let mut bs = cuda.buf_scores.borrow_mut();

        let mut builder = cuda.stream.launch_builder(&cuda.deltanet_train_func);
        builder.arg(&ba.buf);
        builder.arg(&bb.buf);
        builder.arg(&bc.buf);
        builder.arg(&bd.buf);
        builder.arg(&be_buf.buf);
        builder.arg(&mut bs.buf);
        builder.arg(&mut state_gpu);
        builder.arg(&mut all_s_gpu);
        builder.arg(&mut all_e_gpu);
        let seq_len_i32 = seq_len as i32;
        let dk_i32 = dk as i32;
        let dv_i32 = dv as i32;
        builder.arg(&seq_len_i32);
        builder.arg(&dk_i32);
        builder.arg(&dv_i32);

        unsafe {
            builder.launch(cfg).expect("deltanet_recurrence_train カーネル起動失敗");
        }
    }

    // D2H: output, all_s_prev, all_e
    {
        let bs = cuda.buf_scores.borrow();
        cuda.stream.memcpy_dtoh(&bs.buf.slice(..total_v), &mut output[..total_v]).expect("output D2H");
    }
    cuda.stream.memcpy_dtoh(&all_s_gpu.slice(..total_states), &mut all_s_prev[..total_states]).expect("all_s_prev D2H");
    cuda.stream.memcpy_dtoh(&all_e_gpu.slice(..total_e), &mut all_e[..total_e]).expect("all_e D2H");
}

/// GPU DeltaNet Forward — 状態をVRAMに保持して返す（D2Hなし）
/// 返り値の CudaSlice は backward で使う
pub fn cuda_deltanet_forward_store(
    cuda: &CudaMatmul,
    q: &[f32], k: &[f32], v: &[f32], beta: &[f32], g: &[f32],
    output: &mut [f32],
    num_heads: usize, seq_len: usize, dk: usize, dv: usize,
) -> CudaSlice<f32> {
    let total_qk = num_heads * seq_len * dk;
    let total_v = num_heads * seq_len * dv;
    let total_bg = num_heads * seq_len;
    let total_states = num_heads * seq_len * dk * dv;

    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_a, total_qk);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_b, total_qk);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_c, total_v);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_d, total_bg);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_e, total_bg);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_scores, total_v);

    { let mut ba = cuda.buf_a.borrow_mut(); cuda.stream.memcpy_htod(&q[..total_qk], &mut ba.buf.slice_mut(..total_qk)).expect("q"); }
    { let mut bb = cuda.buf_b.borrow_mut(); cuda.stream.memcpy_htod(&k[..total_qk], &mut bb.buf.slice_mut(..total_qk)).expect("k"); }
    { let mut bc = cuda.buf_c.borrow_mut(); cuda.stream.memcpy_htod(&v[..total_v], &mut bc.buf.slice_mut(..total_v)).expect("v"); }
    { let mut bd = cuda.buf_d.borrow_mut(); cuda.stream.memcpy_htod(&beta[..total_bg], &mut bd.buf.slice_mut(..total_bg)).expect("beta"); }
    { let mut be_buf = cuda.buf_e.borrow_mut(); cuda.stream.memcpy_htod(&g[..total_bg], &mut be_buf.buf.slice_mut(..total_bg)).expect("g"); }

    let state_size = num_heads * dk * dv;
    let mut state_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(state_size).expect("state");
    let mut all_s_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_states).expect("all_s");
    let mut all_e_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_v).expect("all_e");

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (dv as u32, 1, 1),
        shared_mem_bytes: (dk * 2 * 4) as u32,
    };

    {
        let ba = cuda.buf_a.borrow();
        let bb = cuda.buf_b.borrow();
        let bc = cuda.buf_c.borrow();
        let bd = cuda.buf_d.borrow();
        let be_buf = cuda.buf_e.borrow();
        let mut bs = cuda.buf_scores.borrow_mut();

        let mut builder = cuda.stream.launch_builder(&cuda.deltanet_train_func);
        builder.arg(&ba.buf); builder.arg(&bb.buf); builder.arg(&bc.buf);
        builder.arg(&bd.buf); builder.arg(&be_buf.buf);
        builder.arg(&mut bs.buf); builder.arg(&mut state_gpu);
        builder.arg(&mut all_s_gpu); builder.arg(&mut all_e_gpu);
        let sl = seq_len as i32; let dki = dk as i32; let dvi = dv as i32;
        builder.arg(&sl); builder.arg(&dki); builder.arg(&dvi);
        unsafe { builder.launch(cfg).expect("forward_store launch"); }
    }

    // D2H: output のみ (all_s はVRAMに残す)
    {
        let bs = cuda.buf_scores.borrow();
        cuda.stream.memcpy_dtoh(&bs.buf.slice(..total_v), &mut output[..total_v]).expect("output D2H");
    }

    all_s_gpu // VRAM handle を返す
}

/// GPU DeltaNet Backward — VRAM上の保存済み状態を使用
pub fn cuda_deltanet_backward_from_vram(
    cuda: &CudaMatmul,
    q: &[f32], k: &[f32], v: &[f32], beta: &[f32], g: &[f32],
    d_out: &[f32],
    all_s_vram: &CudaSlice<f32>,
    d_q: &mut [f32], d_k: &mut [f32], d_v: &mut [f32],
    d_beta: &mut [f32], d_g: &mut [f32],
    num_heads: usize, seq_len: usize, dk: usize, dv: usize,
) {
    let total_qk = num_heads * seq_len * dk;
    let total_v = num_heads * seq_len * dv;
    let total_bg = num_heads * seq_len;

    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_a, total_qk);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_b, total_qk);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_c, total_v);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_d, total_bg);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_e, total_bg);

    { let mut ba = cuda.buf_a.borrow_mut(); cuda.stream.memcpy_htod(&q[..total_qk], &mut ba.buf.slice_mut(..total_qk)).expect("q"); }
    { let mut bb = cuda.buf_b.borrow_mut(); cuda.stream.memcpy_htod(&k[..total_qk], &mut bb.buf.slice_mut(..total_qk)).expect("k"); }
    { let mut bc = cuda.buf_c.borrow_mut(); cuda.stream.memcpy_htod(&v[..total_v], &mut bc.buf.slice_mut(..total_v)).expect("v"); }
    { let mut bd = cuda.buf_d.borrow_mut(); cuda.stream.memcpy_htod(&beta[..total_bg], &mut bd.buf.slice_mut(..total_bg)).expect("beta"); }
    { let mut be_buf = cuda.buf_e.borrow_mut(); cuda.stream.memcpy_htod(&g[..total_bg], &mut be_buf.buf.slice_mut(..total_bg)).expect("g"); }

    let mut d_out_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_v).expect("d_out");
    cuda.stream.memcpy_htod(&d_out[..total_v], &mut d_out_gpu).expect("d_out H2D");

    let mut dq_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_qk).expect("dq");
    let mut dk_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_qk).expect("dk");
    let mut dv_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_v).expect("dv");
    let mut dbeta_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_bg).expect("dbeta");
    let mut dg_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_bg).expect("dg");

    let num_warps = (dv + 31) / 32;
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (dv as u32, 1, 1),
        shared_mem_bytes: ((dk * 2 + num_warps) * 4) as u32,
    };

    {
        let ba = cuda.buf_a.borrow();
        let bb = cuda.buf_b.borrow();
        let bc = cuda.buf_c.borrow();
        let bd = cuda.buf_d.borrow();
        let be_buf = cuda.buf_e.borrow();

        let mut builder = cuda.stream.launch_builder(&cuda.deltanet_bwd_func);
        builder.arg(&ba.buf); builder.arg(&bb.buf); builder.arg(&bc.buf);
        builder.arg(&bd.buf); builder.arg(&be_buf.buf);
        builder.arg(&d_out_gpu); builder.arg(all_s_vram);
        builder.arg(&mut dq_gpu); builder.arg(&mut dk_gpu); builder.arg(&mut dv_gpu);
        builder.arg(&mut dbeta_gpu); builder.arg(&mut dg_gpu);
        let sl = seq_len as i32; let dki = dk as i32; let dvi = dv as i32;
        builder.arg(&sl); builder.arg(&dki); builder.arg(&dvi);
        unsafe { builder.launch(cfg).expect("backward launch"); }
    }

    cuda.stream.memcpy_dtoh(&dq_gpu.slice(..total_qk), &mut d_q[..total_qk]).expect("dq D2H");
    cuda.stream.memcpy_dtoh(&dk_gpu.slice(..total_qk), &mut d_k[..total_qk]).expect("dk D2H");
    cuda.stream.memcpy_dtoh(&dv_gpu.slice(..total_v), &mut d_v[..total_v]).expect("dv D2H");
    cuda.stream.memcpy_dtoh(&dbeta_gpu.slice(..total_bg), &mut d_beta[..total_bg]).expect("dbeta D2H");
    cuda.stream.memcpy_dtoh(&dg_gpu.slice(..total_bg), &mut d_g[..total_bg]).expect("dg D2H");
}

/// GPU DeltaNet Fused Forward+Backward — 状態をVRAM内に保持、D2H転送ゼロ
/// forward output + backward gradients を一発で計算
pub fn cuda_deltanet_fused_fwd_bwd(
    cuda: &CudaMatmul,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    d_out: &[f32],
    output: &mut [f32],
    d_q: &mut [f32],
    d_k: &mut [f32],
    d_v: &mut [f32],
    d_beta: &mut [f32],
    d_g: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    dk: usize,
    dv: usize,
) {
    let total_qk = num_heads * seq_len * dk;
    let total_v = num_heads * seq_len * dv;
    let total_bg = num_heads * seq_len;
    let total_states = num_heads * seq_len * dk * dv;

    // H2D: q, k, v, beta, g, d_out
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_a, total_qk);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_b, total_qk);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_c, total_v);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_d, total_bg);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_e, total_bg);
    CudaMatmul::ensure_buf(&cuda.stream, &cuda.buf_scores, total_v);

    {
        let mut ba = cuda.buf_a.borrow_mut();
        cuda.stream.memcpy_htod(&q[..total_qk], &mut ba.buf.slice_mut(..total_qk)).expect("q H2D");
    }
    {
        let mut bb = cuda.buf_b.borrow_mut();
        cuda.stream.memcpy_htod(&k[..total_qk], &mut bb.buf.slice_mut(..total_qk)).expect("k H2D");
    }
    {
        let mut bc = cuda.buf_c.borrow_mut();
        cuda.stream.memcpy_htod(&v[..total_v], &mut bc.buf.slice_mut(..total_v)).expect("v H2D");
    }
    {
        let mut bd = cuda.buf_d.borrow_mut();
        cuda.stream.memcpy_htod(&beta[..total_bg], &mut bd.buf.slice_mut(..total_bg)).expect("beta H2D");
    }
    {
        let mut be_buf = cuda.buf_e.borrow_mut();
        cuda.stream.memcpy_htod(&g[..total_bg], &mut be_buf.buf.slice_mut(..total_bg)).expect("g H2D");
    }

    // d_out H2D — 専用バッファ
    let mut d_out_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_v).expect("d_out 確保");
    cuda.stream.memcpy_htod(&d_out[..total_v], &mut d_out_gpu).expect("d_out H2D");

    // Output + gradient バッファ
    let mut out_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_v).expect("output 確保");
    let mut dq_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_qk).expect("dq 確保");
    let mut dk_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_qk).expect("dk 確保");
    let mut dv_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_v).expect("dv 確保");
    let mut dbeta_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_bg).expect("dbeta 確保");
    let mut dg_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_bg).expect("dg 確保");

    // State + workspace (VRAM内完結)
    let state_size = num_heads * dk * dv;
    let mut state_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(state_size).expect("state 確保");
    let mut all_s_gpu: CudaSlice<f32> = cuda.stream.alloc_zeros(total_states).expect("all_s 確保");

    let shared_bytes = (dk * 2 + dv) * 4; // k_buf + q_buf + do_buf
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (dv as u32, 1, 1),
        shared_mem_bytes: shared_bytes as u32,
    };

    {
        let ba = cuda.buf_a.borrow();
        let bb = cuda.buf_b.borrow();
        let bc = cuda.buf_c.borrow();
        let bd = cuda.buf_d.borrow();
        let be_buf = cuda.buf_e.borrow();

        let mut builder = cuda.stream.launch_builder(&cuda.deltanet_fused_func);
        builder.arg(&ba.buf);       // q
        builder.arg(&bb.buf);       // k
        builder.arg(&bc.buf);       // v
        builder.arg(&bd.buf);       // beta
        builder.arg(&be_buf.buf);   // g
        builder.arg(&d_out_gpu);    // d_out
        builder.arg(&mut out_gpu);  // output
        builder.arg(&mut dq_gpu);   // d_q
        builder.arg(&mut dk_gpu);   // d_k
        builder.arg(&mut dv_gpu);   // d_v
        builder.arg(&mut dbeta_gpu);// d_beta
        builder.arg(&mut dg_gpu);   // d_g
        builder.arg(&mut state_gpu);// state_buf
        builder.arg(&mut all_s_gpu);// all_s workspace
        let sl = seq_len as i32;
        let dki = dk as i32;
        let dvi = dv as i32;
        builder.arg(&sl);
        builder.arg(&dki);
        builder.arg(&dvi);

        unsafe { builder.launch(cfg).expect("deltanet_fused_fwd_bwd 起動失敗"); }
    }

    // D2H: output + gradients のみ (状態はGPU内で完結)
    cuda.stream.memcpy_dtoh(&out_gpu.slice(..total_v), &mut output[..total_v]).expect("output D2H");
    cuda.stream.memcpy_dtoh(&dq_gpu.slice(..total_qk), &mut d_q[..total_qk]).expect("dq D2H");
    cuda.stream.memcpy_dtoh(&dk_gpu.slice(..total_qk), &mut d_k[..total_qk]).expect("dk D2H");
    cuda.stream.memcpy_dtoh(&dv_gpu.slice(..total_v), &mut d_v[..total_v]).expect("dv D2H");
    cuda.stream.memcpy_dtoh(&dbeta_gpu.slice(..total_bg), &mut d_beta[..total_bg]).expect("dbeta D2H");
    cuda.stream.memcpy_dtoh(&dg_gpu.slice(..total_bg), &mut d_g[..total_bg]).expect("dg D2H");
}
