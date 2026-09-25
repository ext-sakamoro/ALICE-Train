//! LoRA (Low-Rank Adaptation) アダプタ。
//!
//! base 重み `W` (out_dim × in_dim) を凍結したまま、低ランク差分
//! `ΔW = s · B · A` (B: out_dim × rank, A: rank × in_dim, s = alpha / rank)
//! だけを学習する。
//!
//! # 設計判断: full 勾配からの射影
//!
//! backward の数式を新規実装せず、既存 [`crate::llama_backward::layer_backward`]
//! が返す **full の `dL/dW`** から LoRA 勾配を射影する。
//! `W_eff = W + s·B·A` に対する連鎖律から、
//!
//! ```text
//! dL/dB = s · (dL/dW) · Aᵀ      (out × in)·(in × rank) = out × rank
//! dL/dA = s · Bᵀ · (dL/dW)      (rank × out)·(out × in) = rank × in
//! ```
//!
//! が **厳密に** 成り立つ (近似ではない)。これにより backward 経路の再実装リスクを
//! 負わずに LoRA を得る。代償は full `dW` を一度作る計算量だが、
//! `layer_backward` は元々それを計算している。
//!
//! oracle: `tests/lora_oracle.rs` が上式を (1) 閉形式 (2) 数値微分 の 2 系統で検証。
//!
//! # メモリ
//!
//! base 重みは凍結なので optimizer state を持たない。学習対象は A/B のみ。
//! MiniCPM5-2B (42 層, hidden 2048, intermediate 6144, kv_dim 256) を
//! rank 16 で 7 projection すべてに付けた場合、adapter は約 25.1M param
//! (f32 で約 100 MB) — base 10.07 GB に対して約 1%。

use crate::llama::{LlamaConfig, LlamaLayerWeights};
use crate::llama_backward::LayerWeightGrads;
use core::fmt;
use rayon::prelude::*;

/// [`LoraConfig`] 構築時の検証エラー。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraConfigError {
    /// rank が 0 (低ランク差分が消滅する)。
    ZeroRank,
    /// rank が in_dim / out_dim を超えうる非現実値。
    RankTooLarge(usize),
    /// alpha が非正 (スケールが 0 以下になる)。
    NonPositiveAlpha,
}

impl fmt::Display for LoraConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroRank => write!(f, "LoRA rank must be > 0"),
            Self::RankTooLarge(r) => write!(f, "LoRA rank {r} exceeds the supported maximum 1024"),
            Self::NonPositiveAlpha => write!(f, "LoRA alpha must be > 0"),
        }
    }
}

impl std::error::Error for LoraConfigError {}

/// LoRA のハイパーパラメータ。
///
/// `Default` は実装しない。`rank` / `alpha` は学習結果を左右する domain 制約なので、
/// 呼び出し側に [`LoraConfig::try_new`] か preset の明示選択を強制する
/// (silent load-bearing 化の予防、`rust-config-struct-guard` §3 anti-pattern 2)。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoraConfig {
    rank: usize,
    alpha: f32,
}

impl LoraConfig {
    /// 検証付き構築。
    ///
    /// # Errors
    /// - `rank == 0` → [`LoraConfigError::ZeroRank`]
    /// - `rank > 1024` → [`LoraConfigError::RankTooLarge`]
    /// - `alpha <= 0` または非有限 → [`LoraConfigError::NonPositiveAlpha`]
    pub fn try_new(rank: usize, alpha: f32) -> Result<Self, LoraConfigError> {
        if rank == 0 {
            return Err(LoraConfigError::ZeroRank);
        }
        if rank > 1024 {
            return Err(LoraConfigError::RankTooLarge(rank));
        }
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(LoraConfigError::NonPositiveAlpha);
        }
        Ok(Self { rank, alpha })
    }

    /// SFT の一般的な既定 (rank 16 / alpha 32 → scale 2.0)。
    ///
    /// MiniCPM5-2B 全 42 層 × 7 projection で約 25.1M param。
    #[must_use]
    pub const fn preset_r16_a32() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
        }
    }

    /// 軽量版 (rank 8 / alpha 16 → scale 2.0)。param は preset_r16_a32 の半分。
    #[must_use]
    pub const fn preset_r8_a16() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
        }
    }

    /// 低ランク次元 r。
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// スケール係数 α。
    #[must_use]
    pub const fn alpha(&self) -> f32 {
        self.alpha
    }

    /// 実効スケール s = α / rank。
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

/// 1 つの projection 行列に付く LoRA アダプタ。
///
/// 行優先で `a` は (rank × in_dim)、`b` は (out_dim × rank)。
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// down-projection A (rank × in_dim)。
    pub a: Vec<f32>,
    /// up-projection B (out_dim × rank)。
    pub b: Vec<f32>,
    /// 入力次元。
    pub in_dim: usize,
    /// 出力次元。
    pub out_dim: usize,
    /// 低ランク次元。
    pub rank: usize,
    /// 実効スケール s = α / rank。
    pub scale: f32,
}

impl LoraAdapter {
    /// 標準 LoRA 初期化: A は決定論的擬似乱数、B はゼロ。
    ///
    /// B = 0 なので初期状態では `W_eff == W` (base の出力を壊さない)。
    /// A は `[-1/√in_dim, 1/√in_dim]` の一様分布で、`seed` により決定論的
    /// (外部 rng に依存しない = run 間で再現する)。
    #[must_use]
    pub fn new(in_dim: usize, out_dim: usize, cfg: LoraConfig, seed: u64) -> Self {
        let rank = cfg.rank();
        let mut a = vec![0.0f32; rank * in_dim];
        let mut state = seed | 1;
        let bound = 1.0 / (in_dim as f32).sqrt();
        for v in &mut a {
            // xorshift64*
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40; // 上位 24 bit
            let unit = bits as f32 / f32::from(u16::MAX) / 256.0; // [0, 1)
            *v = (unit * 2.0 - 1.0) * bound;
        }
        Self {
            a,
            b: vec![0.0f32; out_dim * rank],
            in_dim,
            out_dim,
            rank,
            scale: cfg.scale(),
        }
    }

    /// 学習対象パラメータ数 (A + B)。
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.a.len() + self.b.len()
    }

    /// `dst = base + s · B · A` を書き込む。
    ///
    /// `base` / `dst` は行優先 (out_dim × in_dim)。
    ///
    /// # Panics
    /// `base` / `dst` の長さが `out_dim * in_dim` でない場合。
    pub fn merge_into(&self, base: &[f32], dst: &mut [f32]) {
        let (out_dim, in_dim, rank) = (self.out_dim, self.in_dim, self.rank);
        assert_eq!(base.len(), out_dim * in_dim, "base の形が合わない");
        assert_eq!(dst.len(), out_dim * in_dim, "dst の形が合わない");
        dst.copy_from_slice(base);
        // dst[o, i] += s · Σ_r B[o, r] · A[r, i]
        // 学習初期は B = 0 なので係数 0 の行を飛ばすと実質ゼロコストになる
        dst.par_chunks_exact_mut(in_dim)
            .enumerate()
            .for_each(|(o, drow)| {
                for r in 0..rank {
                    let coef = self.scale * self.b[o * rank + r];
                    if coef == 0.0 {
                        continue;
                    }
                    let arow = &self.a[r * in_dim..(r + 1) * in_dim];
                    for (d, a) in drow.iter_mut().zip(arow.iter()) {
                        *d += coef * a;
                    }
                }
            });
    }

    /// full の `d_w` (out_dim × in_dim) から A/B の勾配を射影して **加算** する。
    ///
    /// `d_a` は (rank × in_dim)、`d_b` は (out_dim × rank)。
    /// 勾配累積のため上書きせず加算する。
    ///
    /// # Panics
    /// 各バッファの長さが期待と異なる場合。
    pub fn project_grad(&self, d_w: &[f32], d_a: &mut [f32], d_b: &mut [f32]) {
        let (out_dim, in_dim, rank) = (self.out_dim, self.in_dim, self.rank);
        assert_eq!(d_w.len(), out_dim * in_dim, "d_w の形が合わない");
        assert_eq!(d_a.len(), rank * in_dim, "d_a の形が合わない");
        assert_eq!(d_b.len(), out_dim * rank, "d_b の形が合わない");

        // dL/dB = s · dW × Aᵀ  →  d_b[o, r] += s · Σ_i dW[o, i] · A[r, i]
        d_b.par_chunks_exact_mut(rank)
            .enumerate()
            .for_each(|(o, dbrow)| {
                let dwrow = &d_w[o * in_dim..(o + 1) * in_dim];
                for (r, db) in dbrow.iter_mut().enumerate() {
                    let arow = &self.a[r * in_dim..(r + 1) * in_dim];
                    let acc: f32 = dwrow.iter().zip(arow.iter()).map(|(w, a)| w * a).sum();
                    *db += self.scale * acc;
                }
            });

        // dL/dA = s · Bᵀ × dW  →  d_a[r, i] += s · Σ_o B[o, r] · dW[o, i]
        d_a.par_chunks_exact_mut(in_dim)
            .enumerate()
            .for_each(|(r, darow)| {
                for o in 0..out_dim {
                    let coef = self.scale * self.b[o * rank + r];
                    let dwrow = &d_w[o * in_dim..(o + 1) * in_dim];
                    for (d, w) in darow.iter_mut().zip(dwrow.iter()) {
                        *d += coef * w;
                    }
                }
            });
    }
}

/// 1 Transformer レイヤー分の LoRA アダプタ (7 projection)。
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// Q projection。
    pub q: LoraAdapter,
    /// K projection。
    pub k: LoraAdapter,
    /// V projection。
    pub v: LoraAdapter,
    /// O projection。
    pub o: LoraAdapter,
    /// Gate projection。
    pub gate: LoraAdapter,
    /// Up projection。
    pub up: LoraAdapter,
    /// Down projection。
    pub down: LoraAdapter,
}

impl LoraLayer {
    /// レイヤー構成から 7 アダプタを初期化。
    #[must_use]
    pub fn new(config: &LlamaConfig, cfg: LoraConfig, layer_idx: usize) -> Self {
        let h = config.hidden_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let q_dim = config.num_heads * config.head_dim;
        let i = config.intermediate_dim;
        let seed = |slot: u64| ((layer_idx as u64) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ slot;
        Self {
            q: LoraAdapter::new(h, q_dim, cfg, seed(0)),
            k: LoraAdapter::new(h, kv, cfg, seed(1)),
            v: LoraAdapter::new(h, kv, cfg, seed(2)),
            o: LoraAdapter::new(q_dim, h, cfg, seed(3)),
            gate: LoraAdapter::new(h, i, cfg, seed(4)),
            up: LoraAdapter::new(h, i, cfg, seed(5)),
            down: LoraAdapter::new(i, h, cfg, seed(6)),
        }
    }

    /// 学習対象パラメータ数。
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.q.num_params()
            + self.k.num_params()
            + self.v.num_params()
            + self.o.num_params()
            + self.gate.num_params()
            + self.up.num_params()
            + self.down.num_params()
    }

    /// base 重みに adapter を合成した重みを `dst` に書き込む。
    ///
    /// norm / bias は LoRA 対象外なので base をそのまま複製する。
    pub fn merge_into(&self, base: &LlamaLayerWeights, dst: &mut LlamaLayerWeights) {
        dst.attn_norm.copy_from_slice(&base.attn_norm);
        dst.ffn_norm.copy_from_slice(&base.ffn_norm);
        dst.q_bias.clone_from(&base.q_bias);
        dst.k_bias.clone_from(&base.k_bias);
        dst.v_bias.clone_from(&base.v_bias);
        self.q.merge_into(&base.q_proj, &mut dst.q_proj);
        self.k.merge_into(&base.k_proj, &mut dst.k_proj);
        self.v.merge_into(&base.v_proj, &mut dst.v_proj);
        self.o.merge_into(&base.o_proj, &mut dst.o_proj);
        self.gate.merge_into(&base.gate_proj, &mut dst.gate_proj);
        self.up.merge_into(&base.up_proj, &mut dst.up_proj);
        self.down.merge_into(&base.down_proj, &mut dst.down_proj);
    }

    /// full の [`LayerWeightGrads`] から LoRA 勾配へ射影して累積する。
    ///
    /// norm / bias の勾配は捨てる (LoRA では base を凍結するため)。
    pub fn project_grads(&self, grads: &LayerWeightGrads, out: &mut LoraLayerGrads) {
        self.q
            .project_grad(&grads.d_q_proj, &mut out.q.d_a, &mut out.q.d_b);
        self.k
            .project_grad(&grads.d_k_proj, &mut out.k.d_a, &mut out.k.d_b);
        self.v
            .project_grad(&grads.d_v_proj, &mut out.v.d_a, &mut out.v.d_b);
        self.o
            .project_grad(&grads.d_o_proj, &mut out.o.d_a, &mut out.o.d_b);
        self.gate
            .project_grad(&grads.d_gate_proj, &mut out.gate.d_a, &mut out.gate.d_b);
        self.up
            .project_grad(&grads.d_up_proj, &mut out.up.d_a, &mut out.up.d_b);
        self.down
            .project_grad(&grads.d_down_proj, &mut out.down.d_a, &mut out.down.d_b);
    }
}

/// 1 アダプタ分の勾配バッファ。
#[derive(Debug, Clone)]
pub struct LoraAdapterGrads {
    /// A の勾配 (rank × in_dim)。
    pub d_a: Vec<f32>,
    /// B の勾配 (out_dim × rank)。
    pub d_b: Vec<f32>,
}

impl LoraAdapterGrads {
    /// ゼロ初期化。
    #[must_use]
    pub fn zeros(adapter: &LoraAdapter) -> Self {
        Self {
            d_a: vec![0.0; adapter.a.len()],
            d_b: vec![0.0; adapter.b.len()],
        }
    }

    /// 勾配をゼロに戻す。
    pub fn zero_out(&mut self) {
        self.d_a.fill(0.0);
        self.d_b.fill(0.0);
    }
}

/// 1 レイヤー分の LoRA 勾配バッファ。
#[derive(Debug, Clone)]
pub struct LoraLayerGrads {
    /// Q。
    pub q: LoraAdapterGrads,
    /// K。
    pub k: LoraAdapterGrads,
    /// V。
    pub v: LoraAdapterGrads,
    /// O。
    pub o: LoraAdapterGrads,
    /// Gate。
    pub gate: LoraAdapterGrads,
    /// Up。
    pub up: LoraAdapterGrads,
    /// Down。
    pub down: LoraAdapterGrads,
}

impl LoraLayerGrads {
    /// ゼロ初期化。
    #[must_use]
    pub fn zeros(layer: &LoraLayer) -> Self {
        Self {
            q: LoraAdapterGrads::zeros(&layer.q),
            k: LoraAdapterGrads::zeros(&layer.k),
            v: LoraAdapterGrads::zeros(&layer.v),
            o: LoraAdapterGrads::zeros(&layer.o),
            gate: LoraAdapterGrads::zeros(&layer.gate),
            up: LoraAdapterGrads::zeros(&layer.up),
            down: LoraAdapterGrads::zeros(&layer.down),
        }
    }

    /// 全勾配をゼロに戻す。
    pub fn zero_out(&mut self) {
        self.q.zero_out();
        self.k.zero_out();
        self.v.zero_out();
        self.o.zero_out();
        self.gate.zero_out();
        self.up.zero_out();
        self.down.zero_out();
    }
}
