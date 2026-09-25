//! llama path の行列積が `blas.rs` に委譲されていることの gate。
//!
//! # なぜこの test が要るか
//!
//! `blas.rs` (cuBLAS TF32 / Accelerate / tiled CPU に dispatch する層) は
//! `llama_forward.rs` / `qwen35_forward.rs` と同じ commit (`eae33c9`) で入ったが、
//! 当時 Llama-3 70B 戦略が凍結された影響で **llama path だけ移行されず**、
//! 素朴な三重ループが残り続けた。その結果 2.5B の 1 sample に 440 秒以上かかり
//! (8 core の 1 core だけ使用、GPU 使用率 0%)、誰も気付かないまま放置されていた。
//!
//! # なぜ「速度」でも「bit 一致」でもなく呼び出し回数で測るのか
//!
//! - **速度**: CI (ubuntu, OpenBLAS feature なし, CUDA なし) では `blas.rs` の
//!   fallback が tiled matmul になり、素朴実装との差が小さく flaky になる。
//! - **bit 一致**: 実測で不成立。macOS の Accelerate は k=128 程度だと素朴実装と
//!   **同じ値を返す**ため、手書きループに戻しても気付けなかった (この gate を
//!   最初 bit 一致で書いて、意図的に naive へ戻す自己検査をして判明した)。
//! - **呼び出し回数**: `blas::gemm_call_count()` の増分は環境に依らず決定論的で、
//!   委譲をやめた瞬間に増分が落ちる。
//!
//! カウンタは process 全体で共有なので、増分の厳密比較をするため
//! **1 つの test 関数に集約**してある (同一 binary 内の並行実行を避ける)。

use alice_train::blas;
use alice_train::llama_backward::matmul_bt_backward;
use alice_train::llama_forward::{matmul, matmul_bt};

/// 決定論的な擬似乱数 (xorshift64*)。
fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
            (bits as f32 / 16_777_216.0).mul_add(2.0, -1.0)
        })
        .collect()
}

#[test]
fn llama_matmuls_delegate_to_blas() {
    // ── matmul_bt は gemm を 1 回使う ──
    let (m, n, k) = (37usize, 53usize, 128usize);
    let a = fill(1, m * k);
    let b = fill(2, n * k);
    let mut got = vec![0.0f32; m * n];
    let before = blas::gemm_call_count();
    matmul_bt(&a, &b, &mut got, m, n, k);
    assert_eq!(
        blas::gemm_call_count() - before,
        1,
        "llama_forward::matmul_bt が blas.rs を呼んでいない \
         (手書きループに戻っていないか確認する)"
    );
    // 値も blas 直呼びと一致すること
    let mut want = vec![0.0f32; m * n];
    blas::blas_matmul_bt(&a, &b, &mut want, m, n, k);
    assert_eq!(got, want, "matmul_bt の結果が blas 直呼びと違う");

    // ── matmul (nn) も 1 回 ──
    let (m2, n2, k2) = (29usize, 41usize, 96usize);
    let a2 = fill(3, m2 * k2);
    let b2 = fill(4, k2 * n2);
    let mut got2 = vec![0.0f32; m2 * n2];
    let before = blas::gemm_call_count();
    matmul(&a2, &b2, &mut got2, m2, n2, k2);
    assert_eq!(
        blas::gemm_call_count() - before,
        1,
        "llama_forward::matmul が blas.rs を呼んでいない"
    );
    let mut want2 = vec![0.0f32; m2 * n2];
    blas::blas_matmul_nn(&a2, &b2, &mut want2, m2, n2, k2);
    assert_eq!(got2, want2, "matmul の結果が blas 直呼びと違う");

    // ── matmul_bt_backward は dA / dB で 2 回、かつ既存値に加算する契約 ──
    let (m3, n3, k3) = (23usize, 31usize, 64usize);
    let d_out = fill(5, m3 * n3);
    let a3 = fill(6, m3 * k3);
    let b3 = fill(7, n3 * k3);
    let init_a = fill(8, m3 * k3);
    let init_b = fill(9, n3 * k3);
    let mut d_a = init_a.clone();
    let mut d_b = init_b.clone();
    let before = blas::gemm_call_count();
    matmul_bt_backward(&d_out, &a3, &b3, &mut d_a, &mut d_b, m3, n3, k3);
    assert_eq!(
        blas::gemm_call_count() - before,
        2,
        "matmul_bt_backward が blas.rs を 2 回 (dA: nn / dB: tn) 呼んでいない"
    );

    // oracle: dA = dC × B (nn), dB = dCᵀ × A (tn、共有次元は m)
    let mut want_a = vec![0.0f32; m3 * k3];
    blas::blas_matmul_nn(&d_out, &b3, &mut want_a, m3, k3, n3);
    let mut want_b = vec![0.0f32; n3 * k3];
    blas::blas_matmul_tn(&d_out, &a3, &mut want_b, n3, k3, m3);
    for (i, ((got, base), delta)) in d_a.iter().zip(init_a.iter()).zip(want_a.iter()).enumerate() {
        assert_eq!(
            *got,
            base + delta,
            "dA[{i}] が blas の結果の加算になっていない"
        );
    }
    for (i, ((got, base), delta)) in d_b.iter().zip(init_b.iter()).zip(want_b.iter()).enumerate() {
        assert_eq!(
            *got,
            base + delta,
            "dB[{i}] が blas の結果の加算になっていない"
        );
    }
}
