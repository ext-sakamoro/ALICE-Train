//! BLAS matmul — macOS Accelerate / フォールバック タイル matmul。
//!
//! L9: projection matmul を 50-100x 高速化。
//!
//! - macOS: Accelerate framework の `cblas_sgemm` (ゼロ依存 FFI)
//! - Linux/その他: タイル matmul (キャッシュライン最適化、ナイーブ比 5-10x)
//! - CUDA 環境: `cuda_matmul.rs` の cuBLAS を使用 (本モジュール不使用)

// ── macOS Accelerate FFI ────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(target_os = "macos")]
const CBLAS_TRANS: i32 = 112;

// ── 公開 API ────────────────────────────────────────────────────────────────

/// C = A × B^T — BLAS 最適化版。
///
/// A: (m × k), B: (n × k) → C: (m × n)。
/// B は転置して掛ける (weight が [out_features × in_features] 格納のため)。
///
/// macOS: Accelerate cblas_sgemm。
/// その他: タイル matmul。
pub fn blas_matmul_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(target_os = "macos")]
    {
        // C = A × B^T → cblas: C = A(NoTrans) × B(Trans)
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                k as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        return;
    }

    #[cfg(not(target_os = "macos"))]
    {
        tiled_matmul_bt(a, b, c, m, n, k);
    }
}

/// C = A × B — BLAS 最適化版。
///
/// A: (m × k), B: (k × n) → C: (m × n)。
pub fn blas_matmul_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(target_os = "macos")]
    {
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                n as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        return;
    }

    #[cfg(not(target_os = "macos"))]
    {
        tiled_matmul_nn(a, b, c, m, n, k);
    }
}

// ── タイル matmul (非macOS フォールバック) ───────────────────────────────────

/// タイルサイズ。L1 キャッシュ (32-64KB) に収まるサイズ。
#[cfg(not(target_os = "macos"))]
const TILE: usize = 64;

/// C = A × B^T — タイル matmul (キャッシュ最適化)。
#[cfg(not(target_os = "macos"))]
fn tiled_matmul_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // zero
    for v in c.iter_mut() {
        *v = 0.0;
    }

    // タイルループ: k 方向を外側にしてキャッシュヒット率を最大化
    let mut kk = 0;
    while kk < k {
        let k_end = (kk + TILE).min(k);
        let mut ii = 0;
        while ii < m {
            let i_end = (ii + TILE).min(m);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + TILE).min(n);

                // マイクロカーネル
                for i in ii..i_end {
                    let a_row = &a[i * k..];
                    let c_row = &mut c[i * n..];
                    for j in jj..j_end {
                        let b_row = &b[j * k..];
                        let mut sum = c_row[j];
                        for h in kk..k_end {
                            sum = a_row[h].mul_add(b_row[h], sum);
                        }
                        c_row[j] = sum;
                    }
                }

                jj += TILE;
            }
            ii += TILE;
        }
        kk += TILE;
    }
}

/// C = A × B — タイル matmul。
#[cfg(not(target_os = "macos"))]
fn tiled_matmul_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for v in c.iter_mut() {
        *v = 0.0;
    }

    let mut kk = 0;
    while kk < k {
        let k_end = (kk + TILE).min(k);
        let mut ii = 0;
        while ii < m {
            let i_end = (ii + TILE).min(m);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + TILE).min(n);
                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = c[i * n + j];
                        for h in kk..k_end {
                            sum = a[i * k + h].mul_add(b[h * n + j], sum);
                        }
                        c[i * n + j] = sum;
                    }
                }
                jj += TILE;
            }
            ii += TILE;
        }
        kk += TILE;
    }
}

// ── L11: SwiGLU FFN BLAS化 ──────────────────────────────────────────────────

/// SiLU (Swish): x * sigmoid(x)。
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU FFN forward — BLAS matmul 版。
///
/// gate = input × gate_proj^T   (BLAS)
/// up   = input × up_proj^T     (BLAS)
/// ffn_out = (SiLU(gate) ⊙ up) × down_proj^T  (BLAS)
///
/// ナイーブ版 (`llama_forward::swiglu_ffn`) の 50-100x 高速化。
pub fn blas_swiglu_ffn(
    input: &[f32],
    gate_proj: &[f32],
    up_proj: &[f32],
    down_proj: &[f32],
    output: &mut [f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    gate_silu_buf: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    // gate = input × gate_proj^T
    blas_matmul_bt(input, gate_proj, gate_buf, seq_len, intermediate_dim, hidden_dim);

    // up = input × up_proj^T
    blas_matmul_bt(input, up_proj, up_buf, seq_len, intermediate_dim, hidden_dim);

    // SiLU(gate) ⊙ up → intermediate (in-place on gate_silu_buf)
    let total = seq_len * intermediate_dim;
    for i in 0..total {
        let g = gate_buf[i];
        gate_silu_buf[i] = silu(g);
    }
    // intermediate = gate_silu ⊙ up (reuse gate_buf as temp)
    for i in 0..total {
        gate_buf[i] = gate_silu_buf[i] * up_buf[i];
    }

    // output = intermediate × down_proj^T
    blas_matmul_bt(gate_buf, down_proj, output, seq_len, hidden_dim, intermediate_dim);

    // gate_buf を復元 (元の gate 値に戻す — backward cache 用)
    // Note: gate_buf は呼び出し元で cache に保存される場合がある。
    // eval 用途では不要だが、学習 forward では必要。
    // ここでは gate_buf を intermediate で上書きしてしまったので、
    // 呼び出し元が cache に保存する場合は元の gate を再計算する必要がある。
    // → eval 専用の場合は問題なし（cache 不保存）。
    // → 学習用は blas_swiglu_ffn_training を使う。
}

/// SwiGLU FFN forward — BLAS matmul 版（学習用、gate/up を保存）。
///
/// `gate_buf`, `up_buf`, `gate_silu_buf` を正しく保存する。
pub fn blas_swiglu_ffn_training(
    input: &[f32],
    gate_proj: &[f32],
    up_proj: &[f32],
    down_proj: &[f32],
    output: &mut [f32],
    gate_buf: &mut [f32],
    up_buf: &mut [f32],
    gate_silu_buf: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    blas_matmul_bt(input, gate_proj, gate_buf, seq_len, intermediate_dim, hidden_dim);
    blas_matmul_bt(input, up_proj, up_buf, seq_len, intermediate_dim, hidden_dim);

    let total = seq_len * intermediate_dim;
    let mut intermediate = vec![0.0f32; total];
    for i in 0..total {
        gate_silu_buf[i] = silu(gate_buf[i]);
        intermediate[i] = gate_silu_buf[i] * up_buf[i];
    }

    blas_matmul_bt(&intermediate, down_proj, output, seq_len, hidden_dim, intermediate_dim);
}

/// RMSNorm — Rayon 並列 (llama_forward::rmsnorm と同等だが blas モジュールに統合)。
pub fn blas_rmsnorm(x: &mut [f32], weight: &[f32], dim: usize, eps: f32) {
    use rayon::prelude::*;
    x.par_chunks_exact_mut(dim).for_each(|row| {
        let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv_rms = 1.0 / (ss / dim as f64 + eps as f64).sqrt() as f32;
        row.iter_mut().zip(weight.iter()).for_each(|(v, &w)| {
            *v *= inv_rms * w;
        });
    });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blas_matmul_bt_identity() {
        // A = [[1,2],[3,4]], B = I = [[1,0],[0,1]]
        // C = A × I^T = A
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [1.0, 0.0, 0.0, 1.0f32];
        let mut c = [0.0f32; 4];
        blas_matmul_bt(&a, &b, &mut c, 2, 2, 2);
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 2.0).abs() < 1e-5);
        assert!((c[2] - 3.0).abs() < 1e-5);
        assert!((c[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_blas_matmul_bt_large() {
        // 128×256 × 64×256 → 128×64
        let m = 128;
        let n = 64;
        let k = 256;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i % 11) as f32 * 0.01).collect();
        let mut c_blas = vec![0.0f32; m * n];
        let mut c_naive = vec![0.0f32; m * n];

        blas_matmul_bt(&a, &b, &mut c_blas, m, n, k);

        // naive 参照実装
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for h in 0..k {
                    sum += a[i * k + h] * b[j * k + h];
                }
                c_naive[i * n + j] = sum;
            }
        }

        for idx in 0..m * n {
            assert!(
                (c_blas[idx] - c_naive[idx]).abs() < 0.01,
                "mismatch at {idx}: blas={} naive={}",
                c_blas[idx],
                c_naive[idx]
            );
        }
    }

    #[test]
    fn test_blas_matmul_nn() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [1.0, 0.0, 0.0, 1.0f32];
        let mut c = [0.0f32; 4];
        blas_matmul_nn(&a, &b, &mut c, 2, 2, 2);
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 2.0).abs() < 1e-5);
        assert!((c[2] - 3.0).abs() < 1e-5);
        assert!((c[3] - 4.0).abs() < 1e-5);
    }
}
