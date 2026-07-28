//! Positional Encoding — sinusoidal (Vaswani 2017) + Rotary (RoPE, Su 2021)。
//!
//! FastSpeech2 encoder/decoder は sinusoidal PE を使用、Qwen 系や LLaMA 系は RoPE を使用。
//! いずれも学習パラメータなし (deterministic function of position and dim)、backward は
//! sinusoidal は pass-through、rotary は rotation matrix の transpose 適用。
//!
//! # Sinusoidal
//!
//! ```text
//! PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
//! PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
//! y = x + PE
//! ```
//!
//! # Rotary (RoPE)
//!
//! Q / K を pair-wise 回転させる:
//!
//! ```text
//! θ_i = pos / 10000^(2i / d_head)
//! q_rot[2i]   = q[2i] * cos(θ_i) - q[2i+1] * sin(θ_i)
//! q_rot[2i+1] = q[2i] * sin(θ_i) + q[2i+1] * cos(θ_i)
//! ```
//!
//! K も同様、V は回転しない。

use serde::{Deserialize, Serialize};

/// Sinusoidal positional encoding config。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SinusoidalPositionalEncodingConfig {
    /// embedding 次元 (通常 hidden_dim)。
    pub embed_dim: usize,
    /// 最大 sequence length (pre-compute 上限)。
    pub max_len: usize,
    /// スケーリング base (通常 10000.0)。
    pub base: f32,
}

impl SinusoidalPositionalEncodingConfig {
    /// 最も一般的な config (base=10000.0)。
    #[must_use]
    pub fn new(embed_dim: usize, max_len: usize) -> Self {
        Self {
            embed_dim,
            max_len,
            base: 10_000.0,
        }
    }

    /// base を明示指定。
    #[must_use]
    pub fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    /// config validity 検証。
    ///
    /// # Errors
    ///
    /// - `embed_dim == 0` / `max_len == 0`
    /// - `embed_dim % 2 != 0` (sin/cos ペアで消費するため偶数必須)
    /// - `base <= 0`
    pub fn validate(&self) -> Result<(), PosEncError> {
        if self.embed_dim == 0 {
            return Err(PosEncError::InvalidConfig {
                reason: "embed_dim must be > 0".to_string(),
            });
        }
        if self.max_len == 0 {
            return Err(PosEncError::InvalidConfig {
                reason: "max_len must be > 0".to_string(),
            });
        }
        if !self.embed_dim.is_multiple_of(2) {
            return Err(PosEncError::InvalidConfig {
                reason: format!(
                    "embed_dim {} must be even (sin/cos pair consumption)",
                    self.embed_dim
                ),
            });
        }
        if self.base <= 0.0 {
            return Err(PosEncError::InvalidConfig {
                reason: format!("base must be > 0, got {}", self.base),
            });
        }
        Ok(())
    }
}

/// Sinusoidal positional encoding (Vaswani 2017)。
///
/// 学習パラメータなし、pre-compute した `[max_len, embed_dim]` の PE table を保持。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SinusoidalPositionalEncoding {
    config: SinusoidalPositionalEncodingConfig,
    pe_table: Vec<f32>,
}

impl SinusoidalPositionalEncoding {
    /// config から PE table を pre-compute して構築する。
    ///
    /// # Errors
    ///
    /// config validation。
    pub fn new(config: SinusoidalPositionalEncodingConfig) -> Result<Self, PosEncError> {
        config.validate()?;
        let mut pe_table = vec![0.0_f32; config.max_len * config.embed_dim];
        for pos in 0..config.max_len {
            for i in 0..(config.embed_dim / 2) {
                let exp = (2 * i) as f32 / config.embed_dim as f32;
                let div_term = config.base.powf(exp);
                let angle = pos as f32 / div_term;
                let sin_val = angle.sin();
                let cos_val = angle.cos();
                pe_table[pos * config.embed_dim + 2 * i] = sin_val;
                pe_table[pos * config.embed_dim + 2 * i + 1] = cos_val;
            }
        }
        Ok(Self { config, pe_table })
    }

    /// config への参照。
    #[must_use]
    pub fn config(&self) -> &SinusoidalPositionalEncodingConfig {
        &self.config
    }

    /// PE table `[max_len, embed_dim]` flatten への参照。
    #[must_use]
    pub fn pe_table(&self) -> &[f32] {
        &self.pe_table
    }

    /// 入力 `[batch, seq_len, embed_dim]` に PE を加算する。
    ///
    /// # Errors
    ///
    /// - input shape 不整合 (`input.len() != batch * seq_len * embed_dim`)
    /// - `seq_len > max_len`
    pub fn forward(
        &self,
        input: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, PosEncError> {
        let e = self.config.embed_dim;
        if input.len() != batch * seq_len * e {
            return Err(PosEncError::ShapeMismatch {
                field: "input",
                expected: batch * seq_len * e,
                actual: input.len(),
            });
        }
        if seq_len > self.config.max_len {
            return Err(PosEncError::SeqLenExceedsMax {
                seq_len,
                max_len: self.config.max_len,
            });
        }

        let mut out = input.to_vec();
        for b in 0..batch {
            for t in 0..seq_len {
                for i in 0..e {
                    out[b * seq_len * e + t * e + i] += self.pe_table[t * e + i];
                }
            }
        }
        Ok(out)
    }

    /// backward: PE は加算のみで学習パラメータなし → grad_output は grad_input としてそのまま伝播。
    ///
    /// 便宜上 API を用意 (通常は forward 呼び出し側が add 後に chain rule で
    /// `grad_input = grad_output` として済ませられるため、必須ではない)。
    ///
    /// # Errors
    ///
    /// grad_output shape 不整合。
    pub fn backward(
        &self,
        grad_output: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, PosEncError> {
        let e = self.config.embed_dim;
        if grad_output.len() != batch * seq_len * e {
            return Err(PosEncError::ShapeMismatch {
                field: "grad_output",
                expected: batch * seq_len * e,
                actual: grad_output.len(),
            });
        }
        Ok(grad_output.to_vec())
    }
}

/// Rotary embedding (RoPE, Su et al. 2021) config。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RotaryEmbeddingConfig {
    /// 回転を適用する dim (通常 head_dim)。偶数必須。
    pub head_dim: usize,
    /// 最大 sequence length (pre-compute 上限)。
    pub max_len: usize,
    /// スケーリング base (通常 10000.0)。
    pub base: f32,
}

impl RotaryEmbeddingConfig {
    /// 最も一般的な config (base=10000.0)。
    #[must_use]
    pub fn new(head_dim: usize, max_len: usize) -> Self {
        Self {
            head_dim,
            max_len,
            base: 10_000.0,
        }
    }

    /// base を明示指定。
    #[must_use]
    pub fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    /// config validity 検証。
    ///
    /// # Errors
    ///
    /// - `head_dim == 0` / `max_len == 0`
    /// - `head_dim % 2 != 0`
    /// - `base <= 0`
    pub fn validate(&self) -> Result<(), PosEncError> {
        if self.head_dim == 0 {
            return Err(PosEncError::InvalidConfig {
                reason: "head_dim must be > 0".to_string(),
            });
        }
        if self.max_len == 0 {
            return Err(PosEncError::InvalidConfig {
                reason: "max_len must be > 0".to_string(),
            });
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(PosEncError::InvalidConfig {
                reason: format!(
                    "head_dim {} must be even (pair-wise rotation)",
                    self.head_dim
                ),
            });
        }
        if self.base <= 0.0 {
            return Err(PosEncError::InvalidConfig {
                reason: format!("base must be > 0, got {}", self.base),
            });
        }
        Ok(())
    }
}

/// Rotary embedding (RoPE)。
///
/// 学習パラメータなし、pre-compute した cos/sin table `[max_len, head_dim/2]` を保持。
/// Q / K に適用、V には適用しない (RoPE 慣習)。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RotaryEmbedding {
    config: RotaryEmbeddingConfig,
    /// `[max_len, head_dim / 2]` flatten の cos table。
    cos_table: Vec<f32>,
    /// `[max_len, head_dim / 2]` flatten の sin table。
    sin_table: Vec<f32>,
}

impl RotaryEmbedding {
    /// config から cos/sin table を pre-compute して構築する。
    ///
    /// # Errors
    ///
    /// config validation。
    pub fn new(config: RotaryEmbeddingConfig) -> Result<Self, PosEncError> {
        config.validate()?;
        let half = config.head_dim / 2;
        let mut cos_table = vec![0.0_f32; config.max_len * half];
        let mut sin_table = vec![0.0_f32; config.max_len * half];
        for pos in 0..config.max_len {
            for i in 0..half {
                let exp = (2 * i) as f32 / config.head_dim as f32;
                let inv_freq = 1.0 / config.base.powf(exp);
                let angle = pos as f32 * inv_freq;
                cos_table[pos * half + i] = angle.cos();
                sin_table[pos * half + i] = angle.sin();
            }
        }
        Ok(Self {
            config,
            cos_table,
            sin_table,
        })
    }

    /// config への参照。
    #[must_use]
    pub fn config(&self) -> &RotaryEmbeddingConfig {
        &self.config
    }

    /// cos table `[max_len, head_dim/2]` flatten への参照。
    #[must_use]
    pub fn cos_table(&self) -> &[f32] {
        &self.cos_table
    }

    /// sin table `[max_len, head_dim/2]` flatten への参照。
    #[must_use]
    pub fn sin_table(&self) -> &[f32] {
        &self.sin_table
    }

    /// Q または K に rotary 回転を適用する。
    ///
    /// 入力 shape: `[batch, num_heads, seq_len, head_dim]` flatten
    /// 出力 shape: 同じ (in-place ではなく新規 `Vec`)
    ///
    /// # 引数
    ///
    /// - `input`: `[batch * num_heads * seq_len * head_dim]` flatten
    /// - `batch`, `num_heads`, `seq_len`: shape 情報
    ///
    /// # Errors
    ///
    /// - input shape 不整合
    /// - `seq_len > max_len`
    pub fn apply(
        &self,
        input: &[f32],
        batch: usize,
        num_heads: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, PosEncError> {
        let d = self.config.head_dim;
        if input.len() != batch * num_heads * seq_len * d {
            return Err(PosEncError::ShapeMismatch {
                field: "input",
                expected: batch * num_heads * seq_len * d,
                actual: input.len(),
            });
        }
        if seq_len > self.config.max_len {
            return Err(PosEncError::SeqLenExceedsMax {
                seq_len,
                max_len: self.config.max_len,
            });
        }

        let half = d / 2;
        let mut out = vec![0.0_f32; input.len()];

        for b in 0..batch {
            for h in 0..num_heads {
                for t in 0..seq_len {
                    let base_idx = b * num_heads * seq_len * d + h * seq_len * d + t * d;
                    for i in 0..half {
                        let c = self.cos_table[t * half + i];
                        let s = self.sin_table[t * half + i];
                        let x0 = input[base_idx + 2 * i];
                        let x1 = input[base_idx + 2 * i + 1];
                        // (x0, x1) → (x0 cos - x1 sin, x0 sin + x1 cos)
                        out[base_idx + 2 * i] = x0 * c - x1 * s;
                        out[base_idx + 2 * i + 1] = x0 * s + x1 * c;
                    }
                }
            }
        }

        Ok(out)
    }

    /// rotary 適用の backward: rotation matrix R の transpose R^T = 逆回転を適用する。
    ///
    /// `[x0', x1'] = [x0*cos - x1*sin, x0*sin + x1*cos]` の backward は、
    /// grad_x0 = grad_x0' * cos + grad_x1' * sin
    /// grad_x1 = -grad_x0' * sin + grad_x1' * cos
    ///
    /// (これは同じ形の rotation を **sin を負** にして適用したものに等しい)
    ///
    /// # Errors
    ///
    /// - grad_output shape 不整合
    /// - `seq_len > max_len`
    pub fn backward(
        &self,
        grad_output: &[f32],
        batch: usize,
        num_heads: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>, PosEncError> {
        let d = self.config.head_dim;
        if grad_output.len() != batch * num_heads * seq_len * d {
            return Err(PosEncError::ShapeMismatch {
                field: "grad_output",
                expected: batch * num_heads * seq_len * d,
                actual: grad_output.len(),
            });
        }
        if seq_len > self.config.max_len {
            return Err(PosEncError::SeqLenExceedsMax {
                seq_len,
                max_len: self.config.max_len,
            });
        }

        let half = d / 2;
        let mut grad_input = vec![0.0_f32; grad_output.len()];

        for b in 0..batch {
            for h in 0..num_heads {
                for t in 0..seq_len {
                    let base_idx = b * num_heads * seq_len * d + h * seq_len * d + t * d;
                    for i in 0..half {
                        let c = self.cos_table[t * half + i];
                        let s = self.sin_table[t * half + i];
                        let g0 = grad_output[base_idx + 2 * i];
                        let g1 = grad_output[base_idx + 2 * i + 1];
                        // grad_x0 = g0 * cos + g1 * sin
                        // grad_x1 = -g0 * sin + g1 * cos
                        grad_input[base_idx + 2 * i] = g0 * c + g1 * s;
                        grad_input[base_idx + 2 * i + 1] = -g0 * s + g1 * c;
                    }
                }
            }
        }

        Ok(grad_input)
    }
}

/// PositionalEncoding 操作で発生し得るエラー。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PosEncError {
    /// config が不正 (dim=0、奇数 dim、負 base 等)。
    InvalidConfig {
        /// 具体的な理由。
        reason: String,
    },
    /// input / grad_output の shape 不整合。
    ShapeMismatch {
        /// 対象 field 名。
        field: &'static str,
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
    /// `seq_len` が `max_len` を超えた (pre-compute table 範囲外)。
    SeqLenExceedsMax {
        /// 要求 seq_len。
        seq_len: usize,
        /// 最大許容 seq_len (config の max_len)。
        max_len: usize,
    },
}

impl std::fmt::Display for PosEncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig { reason } => {
                write!(f, "invalid PositionalEncoding config: {reason}")
            }
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch on '{field}': expected {expected}, got {actual}"
            ),
            Self::SeqLenExceedsMax { seq_len, max_len } => write!(
                f,
                "seq_len {seq_len} exceeds pre-computed max_len {max_len}"
            ),
        }
    }
}

impl std::error::Error for PosEncError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_pe_pos_zero_is_zero_and_one_pattern() {
        // pos=0: PE(0, 2i) = sin(0) = 0, PE(0, 2i+1) = cos(0) = 1
        let cfg = SinusoidalPositionalEncodingConfig::new(4, 10);
        let pe = SinusoidalPositionalEncoding::new(cfg).unwrap();
        let table = pe.pe_table();
        // pos=0 の 4 dim
        assert!((table[0] - 0.0).abs() < 1e-6); // sin(0) = 0
        assert!((table[1] - 1.0).abs() < 1e-6); // cos(0) = 1
        assert!((table[2] - 0.0).abs() < 1e-6);
        assert!((table[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sinusoidal_pe_pos_1_dim_0_matches_sin_1() {
        // pos=1, dim=0: PE = sin(1 / 10000^0) = sin(1)
        let cfg = SinusoidalPositionalEncodingConfig::new(4, 10);
        let pe = SinusoidalPositionalEncoding::new(cfg).unwrap();
        let table = pe.pe_table();
        let expected = (1.0_f32).sin();
        assert!(
            (table[4] - expected).abs() < 1e-5,
            "expected sin(1)={expected}, got {}",
            table[4]
        );
    }

    #[test]
    fn sinusoidal_forward_adds_pe_to_input() {
        let cfg = SinusoidalPositionalEncodingConfig::new(4, 10);
        let pe = SinusoidalPositionalEncoding::new(cfg).unwrap();
        let input = vec![0.0_f32; 3 * 4]; // batch=1, seq=3, embed=4
        let out = pe.forward(&input, 1, 3).unwrap();
        // input=0 なので out = PE table の [seq=3, embed=4] portion
        assert_eq!(out.len(), 12);
        for t in 0..3 {
            for i in 0..4 {
                assert!(
                    (out[t * 4 + i] - pe.pe_table()[t * 4 + i]).abs() < 1e-6,
                    "out[{t},{i}] != PE table"
                );
            }
        }
    }

    #[test]
    fn sinusoidal_backward_passes_gradient_through() {
        let cfg = SinusoidalPositionalEncodingConfig::new(4, 10);
        let pe = SinusoidalPositionalEncoding::new(cfg).unwrap();
        let grad_output = vec![1.0_f32, 2.0, 3.0, 4.0];
        let grad_input = pe.backward(&grad_output, 1, 1).unwrap();
        assert_eq!(grad_input, grad_output);
    }

    #[test]
    fn sinusoidal_seq_exceeds_max_returns_error() {
        let cfg = SinusoidalPositionalEncodingConfig::new(4, 5);
        let pe = SinusoidalPositionalEncoding::new(cfg).unwrap();
        let input = vec![0.0_f32; 10 * 4]; // seq=10 > max=5
        let err = pe.forward(&input, 1, 10).expect_err("seq exceeds max");
        assert!(matches!(err, PosEncError::SeqLenExceedsMax { .. }));
    }

    #[test]
    fn sinusoidal_odd_dim_returns_error() {
        let cfg = SinusoidalPositionalEncodingConfig::new(3, 10);
        let err = cfg.validate().expect_err("odd dim");
        assert!(matches!(err, PosEncError::InvalidConfig { .. }));
    }

    #[test]
    fn rotary_pos_zero_is_identity() {
        // pos=0: cos=1, sin=0 → 回転なし = identity
        let cfg = RotaryEmbeddingConfig::new(4, 10);
        let rope = RotaryEmbedding::new(cfg).unwrap();
        let input = vec![1.0_f32, 2.0, 3.0, 4.0]; // batch=1, heads=1, seq=1, dim=4
        let out = rope.apply(&input, 1, 1, 1).unwrap();
        for (a, b) in out.iter().zip(&input) {
            assert!((a - b).abs() < 1e-6, "pos=0 should be identity");
        }
    }

    #[test]
    fn rotary_preserves_pair_norm() {
        // 回転は norm 保存: |x0'|² + |x1'|² = |x0|² + |x1|²
        let cfg = RotaryEmbeddingConfig::new(4, 10);
        let rope = RotaryEmbedding::new(cfg).unwrap();
        // batch=1, heads=1, seq=3, dim=4
        let input: Vec<f32> = (0..12).map(|i| ((i as f32 + 1.0) * 0.3).sin()).collect();
        let out = rope.apply(&input, 1, 1, 3).unwrap();

        // 各 (t, pair) について norm 一致
        for t in 0..3 {
            for i in 0..2 {
                let base = t * 4;
                let n_in = input[base + 2 * i].powi(2) + input[base + 2 * i + 1].powi(2);
                let n_out = out[base + 2 * i].powi(2) + out[base + 2 * i + 1].powi(2);
                assert!(
                    (n_in - n_out).abs() < 1e-5,
                    "pair norm not preserved at t={t}, i={i}: in={n_in}, out={n_out}"
                );
            }
        }
    }

    #[test]
    fn rotary_forward_backward_roundtrip_recovers_input() {
        // 回転 → 逆回転 = 恒等 → forward(input) を apply、backward(rotated) → input と一致
        // (これは grad_input を rotated=1.0 で計算した場合の実質 rotation inverse)
        // 手動確認: apply(input) 後、rotation の inverse で戻す (backward で sin 負適用)
        let cfg = RotaryEmbeddingConfig::new(4, 10);
        let rope = RotaryEmbedding::new(cfg).unwrap();
        let input: Vec<f32> = (0..8).map(|i| (i as f32 * 0.5).cos()).collect(); // batch=1, heads=1, seq=2, dim=4
        let rotated = rope.apply(&input, 1, 1, 2).unwrap();
        // rotated を backward すると (実質) rotation inverse を適用
        let recovered = rope.backward(&rotated, 1, 1, 2).unwrap();
        // recovered ≈ input (回転と逆回転で戻る)
        for (a, b) in recovered.iter().zip(&input) {
            assert!(
                (a - b).abs() < 1e-5,
                "roundtrip mismatch: recovered={a}, input={b}"
            );
        }
    }

    #[test]
    fn rotary_backward_matches_finite_difference() {
        // Numerical check: apply の Jacobian と backward が整合するか
        let cfg = RotaryEmbeddingConfig::new(4, 10);
        let rope = RotaryEmbedding::new(cfg).unwrap();
        let batch = 1;
        let heads = 1;
        let seq = 2;
        let d = 4;
        let input: Vec<f32> = (0..batch * heads * seq * d)
            .map(|i| (i as f32 * 0.3).sin() * 0.5)
            .collect();
        let grad_output: Vec<f32> = (0..batch * heads * seq * d)
            .map(|i| (i as f32 * 0.17).cos() * 0.3)
            .collect();

        let analytical = rope.backward(&grad_output, batch, heads, seq).unwrap();

        let h = 1e-3_f32;
        for i in 0..input.len() {
            let mut ip = input.clone();
            ip[i] += h;
            let out_p = rope.apply(&ip, batch, heads, seq).unwrap();
            let mut im = input.clone();
            im[i] -= h;
            let out_m = rope.apply(&im, batch, heads, seq).unwrap();
            let loss_p: f32 = out_p.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let loss_m: f32 = out_m.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let ana = analytical[i];
            let diff = (ana - num).abs();
            let scale = ana.abs().max(num.abs()).max(1e-4);
            assert!(
                diff / scale < 1e-2,
                "input[{i}] analytical={ana}, numerical={num}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn rotary_odd_dim_returns_error() {
        let cfg = RotaryEmbeddingConfig::new(5, 10);
        let err = cfg.validate().expect_err("odd head_dim");
        assert!(matches!(err, PosEncError::InvalidConfig { .. }));
    }

    #[test]
    fn rotary_seq_exceeds_max_returns_error() {
        let cfg = RotaryEmbeddingConfig::new(4, 5);
        let rope = RotaryEmbedding::new(cfg).unwrap();
        let input = vec![0.0_f32; 10 * 4]; // seq=10 > max=5
        let err = rope.apply(&input, 1, 1, 10).expect_err("seq exceeds max");
        assert!(matches!(err, PosEncError::SeqLenExceedsMax { .. }));
    }

    #[test]
    fn rotary_multi_head_multi_batch() {
        // batch=2, heads=3, seq=4, dim=4 で shape check + non-zero
        let cfg = RotaryEmbeddingConfig::new(4, 10);
        let rope = RotaryEmbedding::new(cfg).unwrap();
        let input: Vec<f32> = (0..2 * 3 * 4 * 4).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = rope.apply(&input, 2, 3, 4).unwrap();
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn error_display_and_error_trait() {
        let e = PosEncError::InvalidConfig {
            reason: "test".to_string(),
        };
        let s = format!("{e}");
        assert!(s.contains("invalid PositionalEncoding"));
        let boxed: Box<dyn std::error::Error> = Box::new(e);
        assert!(boxed.to_string().contains("test"));

        let e = PosEncError::SeqLenExceedsMax {
            seq_len: 100,
            max_len: 50,
        };
        assert!(format!("{e}").contains("100"));
        assert!(format!("{e}").contains("50"));
    }

    #[test]
    fn config_builders_compose() {
        let cfg = SinusoidalPositionalEncodingConfig::new(8, 64).with_base(500.0);
        cfg.validate().unwrap();
        assert!((cfg.base - 500.0).abs() < f32::EPSILON);

        let cfg = RotaryEmbeddingConfig::new(64, 128).with_base(1_000_000.0);
        cfg.validate().unwrap();
        assert!((cfg.base - 1_000_000.0).abs() < 1.0);
    }
}
