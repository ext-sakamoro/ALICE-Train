//! LoRA の解析解突合 oracle。
//!
//! 検証対象は `W_eff = W + s·B·A` の合成と、full 勾配 `dL/dW` からの射影
//!
//! ```text
//! dL/dB = s · (dL/dW) · Aᵀ
//! dL/dA = s · Bᵀ · (dL/dW)
//! ```
//!
//! これらは近似ではなく連鎖律からの厳密式なので、
//! (1) 手計算した閉形式、(2) 中心差分による数値微分 の 2 系統で突合する。
//!
//! oracle の出所は連鎖律そのもの (外部実装を呼んで期待値を作っていない)。

use alice_train::lora::{LoraAdapter, LoraConfig};

/// 行優先 (rows × cols) の素朴 matmul: `c = a × b`。
/// 期待値生成専用なので実装側の matmul は使わない。
fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += a[i * k + t] * b[t * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// 決定論的なテスト用アダプタ (A/B とも非ゼロ、scale も 1 以外)。
fn fixture() -> (LoraAdapter, usize, usize, usize) {
    let (in_dim, out_dim, rank) = (4usize, 3usize, 2usize);
    let cfg = LoraConfig::try_new(rank, 4.0).expect("valid cfg"); // scale = 2.0
    let mut ad = LoraAdapter::new(in_dim, out_dim, cfg, 12345);
    // A/B を決め打ちして手計算可能にする
    ad.a = vec![
        0.5, -0.25, 1.0, 0.125, // rank 0
        -1.5, 0.75, 0.0, 2.0, // rank 1
    ];
    ad.b = vec![
        1.0, -2.0, // out 0
        0.5, 0.25, // out 1
        -1.0, 3.0, // out 2
    ];
    (ad, in_dim, out_dim, rank)
}

#[test]
fn merge_matches_closed_form_low_rank_product() {
    let (ad, in_dim, out_dim, rank) = fixture();
    let base: Vec<f32> = (0..out_dim * in_dim).map(|i| i as f32 * 0.1).collect();

    let mut got = vec![0.0f32; out_dim * in_dim];
    ad.merge_into(&base, &mut got);

    // oracle: W + s·(B×A) を素朴 matmul で独立に構成
    let ba = naive_matmul(&ad.b, &ad.a, out_dim, rank, in_dim);
    let want: Vec<f32> = base
        .iter()
        .zip(ba.iter())
        .map(|(w, d)| w + ad.scale * d)
        .collect();

    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-6,
            "merge mismatch at {i}: got {g}, want {w}"
        );
    }
}

#[test]
fn merge_with_zero_b_is_identity() {
    // 標準 LoRA 初期化 (B = 0) では W_eff == W でなければならない
    let cfg = LoraConfig::preset_r16_a32();
    let ad = LoraAdapter::new(8, 5, cfg, 999);
    assert!(ad.b.iter().all(|v| *v == 0.0), "B は 0 初期化のはず");

    let base: Vec<f32> = (0..5 * 8).map(|i| (i as f32).sin()).collect();
    let mut got = vec![0.0f32; 5 * 8];
    ad.merge_into(&base, &mut got);
    assert_eq!(got, base, "B=0 で W_eff != W になっている");
}

#[test]
fn project_grad_matches_closed_form() {
    let (ad, in_dim, out_dim, rank) = fixture();
    // 任意の full 勾配 (L = Σ M⊙W_eff のとき dL/dW = M)
    let d_w: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3)
        .collect();

    let mut d_a = vec![0.0f32; rank * in_dim];
    let mut d_b = vec![0.0f32; out_dim * rank];
    ad.project_grad(&d_w, &mut d_a, &mut d_b);

    // oracle: dL/dB = s · dW × Aᵀ, dL/dA = s · Bᵀ × dW
    let mut a_t = vec![0.0f32; in_dim * rank];
    for r in 0..rank {
        for c in 0..in_dim {
            a_t[c * rank + r] = ad.a[r * in_dim + c];
        }
    }
    let mut b_t = vec![0.0f32; rank * out_dim];
    for o in 0..out_dim {
        for r in 0..rank {
            b_t[r * out_dim + o] = ad.b[o * rank + r];
        }
    }
    let want_b = naive_matmul(&d_w, &a_t, out_dim, in_dim, rank);
    let want_a = naive_matmul(&b_t, &d_w, rank, out_dim, in_dim);

    for (i, g) in d_b.iter().enumerate() {
        let w = ad.scale * want_b[i];
        assert!(
            (g - w).abs() <= 1e-5,
            "dB mismatch at {i}: got {g}, want {w}"
        );
    }
    for (i, g) in d_a.iter().enumerate() {
        let w = ad.scale * want_a[i];
        assert!(
            (g - w).abs() <= 1e-5,
            "dA mismatch at {i}: got {g}, want {w}"
        );
    }
}

#[test]
fn project_grad_accumulates_rather_than_overwrites() {
    let (ad, in_dim, out_dim, rank) = fixture();
    let d_w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32) * 0.05).collect();

    let mut a1 = vec![0.0f32; rank * in_dim];
    let mut b1 = vec![0.0f32; out_dim * rank];
    ad.project_grad(&d_w, &mut a1, &mut b1);

    let mut a2 = vec![0.0f32; rank * in_dim];
    let mut b2 = vec![0.0f32; out_dim * rank];
    ad.project_grad(&d_w, &mut a2, &mut b2);
    ad.project_grad(&d_w, &mut a2, &mut b2);

    for (i, (one, two)) in a1.iter().zip(a2.iter()).enumerate() {
        assert!(
            (two - one * 2.0).abs() <= 1e-5,
            "dA が累積していない at {i}: 1回 {one}, 2回 {two}"
        );
    }
    for (i, (one, two)) in b1.iter().zip(b2.iter()).enumerate() {
        assert!(
            (two - one * 2.0).abs() <= 1e-5,
            "dB が累積していない at {i}: 1回 {one}, 2回 {two}"
        );
    }
}

/// 数値微分 oracle。
///
/// `f(A,B) = 0.5 · ‖W_eff·x‖²` (x 固定) を使う。
/// このとき `dL/dW = (W_eff·x)·xᵀ` で、`dW` は A/B に依存する (定数ではない) ので
/// 射影式が定数勾配のときだけ合う実装になっていないかを検出できる。
#[test]
fn project_grad_matches_central_difference() {
    let (mut ad, in_dim, out_dim, rank) = fixture();
    let x: Vec<f32> = (0..in_dim).map(|i| 0.3 + i as f32 * 0.17).collect();
    let base: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.4)
        .collect();

    // f(A,B) を A/B から直接計算する (実装の merge_into のみ使用)
    let eval = |ad: &LoraAdapter| -> f64 {
        let mut w = vec![0.0f32; out_dim * in_dim];
        ad.merge_into(&base, &mut w);
        let y = naive_matmul(&w, &x, out_dim, in_dim, 1);
        0.5 * y.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>()
    };

    // 解析: dL/dW = (W_eff x) xᵀ → それを射影
    let mut w = vec![0.0f32; out_dim * in_dim];
    ad.merge_into(&base, &mut w);
    let y = naive_matmul(&w, &x, out_dim, in_dim, 1);
    let mut d_w = vec![0.0f32; out_dim * in_dim];
    for o in 0..out_dim {
        for i in 0..in_dim {
            d_w[o * in_dim + i] = y[o] * x[i];
        }
    }
    let mut d_a = vec![0.0f32; rank * in_dim];
    let mut d_b = vec![0.0f32; out_dim * rank];
    ad.project_grad(&d_w, &mut d_a, &mut d_b);

    // 中心差分
    let h = 1e-3f32;
    for idx in 0..ad.a.len() {
        let orig = ad.a[idx];
        ad.a[idx] = orig + h;
        let plus = eval(&ad);
        ad.a[idx] = orig - h;
        let minus = eval(&ad);
        ad.a[idx] = orig;
        let num = (plus - minus) / f64::from(2.0 * h);
        let ana = f64::from(d_a[idx]);
        assert!(
            (num - ana).abs() <= 1e-2 * (1.0 + ana.abs()),
            "dA[{idx}] 数値 {num} vs 解析 {ana}"
        );
    }
    for idx in 0..ad.b.len() {
        let orig = ad.b[idx];
        ad.b[idx] = orig + h;
        let plus = eval(&ad);
        ad.b[idx] = orig - h;
        let minus = eval(&ad);
        ad.b[idx] = orig;
        let num = (plus - minus) / f64::from(2.0 * h);
        let ana = f64::from(d_b[idx]);
        assert!(
            (num - ana).abs() <= 1e-2 * (1.0 + ana.abs()),
            "dB[{idx}] 数値 {num} vs 解析 {ana}"
        );
    }
}

#[test]
fn config_rejects_invalid_values() {
    use alice_train::lora::LoraConfigError;
    assert_eq!(LoraConfig::try_new(0, 32.0), Err(LoraConfigError::ZeroRank));
    assert_eq!(
        LoraConfig::try_new(4096, 32.0),
        Err(LoraConfigError::RankTooLarge(4096))
    );
    assert_eq!(
        LoraConfig::try_new(16, 0.0),
        Err(LoraConfigError::NonPositiveAlpha)
    );
    assert_eq!(
        LoraConfig::try_new(16, f32::NAN),
        Err(LoraConfigError::NonPositiveAlpha)
    );
    let ok = LoraConfig::try_new(16, 32.0).expect("valid");
    assert!((ok.scale() - 2.0).abs() < 1e-9);
    assert!((LoraConfig::preset_r8_a16().scale() - 2.0).abs() < 1e-9);
}
