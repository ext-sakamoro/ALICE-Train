//! Multi-Head Attention — forward + 手書き backward (self-attention + cross-attention 両対応)。
//!
//! FastSpeech2 FFT block encoder/decoder / VITS2 の attention 層で使用される
//! Vaswani et al. (2017) の scaled dot-product multi-head attention 実装。
//! PyTorch `nn.MultiheadAttention(batch_first=True)` と数値互換を目指す。
//!
//! # 演算定義
//!
//! ```text
//! Q_proj = q @ W_q^T + b_q     # [B, Sq, E]
//! K_proj = k @ W_k^T + b_k     # [B, Sk, E]
//! V_proj = v @ W_v^T + b_v     # [B, Sk, E]
//!
//! # split heads: [B, S, E] -> [B, H, S, D]  (E = H * D)
//! Q_h, K_h, V_h = split_heads(...)
//!
//! scores = Q_h @ K_h^T / sqrt(D)   # [B, H, Sq, Sk]
//! if causal_mask: scores[i, j] = -inf for j > i
//! attn = softmax(scores, dim=-1)   # [B, H, Sq, Sk]
//! out_h = attn @ V_h              # [B, H, Sq, D]
//!
//! # concat heads: [B, H, Sq, D] -> [B, Sq, E]
//! out = concat_heads(out_h)
//! output = out @ W_o^T + b_o       # [B, Sq, E]
//! ```
//!
//! # backward
//!
//! - Output projection (W_o / b_o) → grad_out_pre
//! - attn @ V → grad_attn (via V_h), grad_V_h (via attn)
//! - softmax backward: `grad_scores[i] = attn[i] * (grad_attn[i] - Σ attn[j] * grad_attn[j])`
//! - Scaled dot-product → grad_Q_h (via K_h), grad_K_h (via Q_h)
//! - Concat / split reshape (inverse)
//! - Q_proj, K_proj, V_proj backward → grad_q, grad_k, grad_v + grad_W_{q,k,v}, grad_b_{q,k,v}
//!
//! # レイアウト
//!
//! 全 tensor を flatten `Vec<f32>` として扱う (2D→4D は index 計算で対応):
//!
//! - input: `[batch, seq_len, embed_dim]` flatten
//! - weight: `[embed_dim, embed_dim]` (Q/K/V/O 各 projection、row-major)
//! - bias: `[embed_dim]`

use serde::{Deserialize, Serialize};

/// Multi-Head Attention config。
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MultiHeadAttentionConfig {
    /// 入出力 embedding 次元 (`num_heads * head_dim` と一致必須)。
    pub embed_dim: usize,
    /// head 数 (embed_dim を等分割)。
    pub num_heads: usize,
    /// bias を持つか (Q/K/V/O projection 全てに共通適用)。
    pub bias: bool,
}

impl MultiHeadAttentionConfig {
    /// 最も一般的な config (bias=true)。
    #[must_use]
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self {
            embed_dim,
            num_heads,
            bias: true,
        }
    }

    /// bias 無効化。
    #[must_use]
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// head_dim = `embed_dim / num_heads` を返す。
    #[must_use]
    pub fn head_dim(&self) -> usize {
        if self.num_heads == 0 {
            0
        } else {
            self.embed_dim / self.num_heads
        }
    }

    /// config validity 検証。
    ///
    /// # Errors
    ///
    /// - `embed_dim == 0` / `num_heads == 0`
    /// - `embed_dim % num_heads != 0`
    pub fn validate(&self) -> Result<(), MhaError> {
        if self.embed_dim == 0 {
            return Err(MhaError::InvalidConfig {
                reason: "embed_dim must be > 0".to_string(),
            });
        }
        if self.num_heads == 0 {
            return Err(MhaError::InvalidConfig {
                reason: "num_heads must be > 0".to_string(),
            });
        }
        if !self.embed_dim.is_multiple_of(self.num_heads) {
            return Err(MhaError::InvalidConfig {
                reason: format!(
                    "embed_dim {} not divisible by num_heads {}",
                    self.embed_dim, self.num_heads
                ),
            });
        }
        Ok(())
    }
}

/// Multi-Head Attention レイヤー (Q/K/V/O projection の 4 weight + 4 bias を保持)。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiHeadAttention {
    config: MultiHeadAttentionConfig,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    b_q: Vec<f32>,
    b_k: Vec<f32>,
    b_v: Vec<f32>,
    b_o: Vec<f32>,
}

impl MultiHeadAttention {
    /// weights / biases を指定して構築する。
    ///
    /// 全 weight は `[embed_dim, embed_dim]` (row-major)、全 bias は `[embed_dim]` or 空 (bias=false)。
    ///
    /// # Errors
    ///
    /// - config validation
    /// - weight/bias shape 不整合
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: MultiHeadAttentionConfig,
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        w_o: Vec<f32>,
        b_q: Vec<f32>,
        b_k: Vec<f32>,
        b_v: Vec<f32>,
        b_o: Vec<f32>,
    ) -> Result<Self, MhaError> {
        config.validate()?;
        let e = config.embed_dim;
        let expected_w = e * e;
        for (name, w) in [("w_q", &w_q), ("w_k", &w_k), ("w_v", &w_v), ("w_o", &w_o)] {
            if w.len() != expected_w {
                return Err(MhaError::ShapeMismatch {
                    field: name,
                    expected: expected_w,
                    actual: w.len(),
                });
            }
        }
        let expected_b = if config.bias { e } else { 0 };
        for (name, b) in [("b_q", &b_q), ("b_k", &b_k), ("b_v", &b_v), ("b_o", &b_o)] {
            if b.len() != expected_b {
                return Err(MhaError::ShapeMismatch {
                    field: name,
                    expected: expected_b,
                    actual: b.len(),
                });
            }
        }
        Ok(Self {
            config,
            w_q,
            w_k,
            w_v,
            w_o,
            b_q,
            b_k,
            b_v,
            b_o,
        })
    }

    /// zero 初期化で構築 (テスト用)。
    ///
    /// # Errors
    ///
    /// config validation。
    pub fn zeros(config: MultiHeadAttentionConfig) -> Result<Self, MhaError> {
        config.validate()?;
        let e = config.embed_dim;
        let n_w = e * e;
        let n_b = if config.bias { e } else { 0 };
        Ok(Self {
            config,
            w_q: vec![0.0; n_w],
            w_k: vec![0.0; n_w],
            w_v: vec![0.0; n_w],
            w_o: vec![0.0; n_w],
            b_q: vec![0.0; n_b],
            b_k: vec![0.0; n_b],
            b_v: vec![0.0; n_b],
            b_o: vec![0.0; n_b],
        })
    }

    /// config への参照。
    #[must_use]
    pub fn config(&self) -> &MultiHeadAttentionConfig {
        &self.config
    }

    /// Q projection weight への参照。
    #[must_use]
    pub fn w_q(&self) -> &[f32] {
        &self.w_q
    }

    /// K projection weight への参照。
    #[must_use]
    pub fn w_k(&self) -> &[f32] {
        &self.w_k
    }

    /// V projection weight への参照。
    #[must_use]
    pub fn w_v(&self) -> &[f32] {
        &self.w_v
    }

    /// O projection weight への参照。
    #[must_use]
    pub fn w_o(&self) -> &[f32] {
        &self.w_o
    }

    /// Q projection bias への参照。
    #[must_use]
    pub fn b_q(&self) -> &[f32] {
        &self.b_q
    }

    /// Xavier uniform init for Q/K/V/O projections + zero bias。
    ///
    /// 各 projection weight `[embed_dim, embed_dim]` に対して:
    /// `a = sqrt(6 / (embed_dim + embed_dim))`、`w ~ U(-a, +a)`。
    pub fn init_xavier<R: rand::Rng>(&mut self, rng: &mut R) {
        let e = self.config.embed_dim;
        let a = (6.0_f32 / (2 * e) as f32).sqrt();
        for w in self
            .w_q
            .iter_mut()
            .chain(self.w_k.iter_mut())
            .chain(self.w_v.iter_mut())
            .chain(self.w_o.iter_mut())
        {
            *w = rng.gen_range(-a..=a);
        }
        for b in self
            .b_q
            .iter_mut()
            .chain(self.b_k.iter_mut())
            .chain(self.b_v.iter_mut())
            .chain(self.b_o.iter_mut())
        {
            *b = 0.0;
        }
    }

    /// Q projection weight への可変参照 (optimizer 更新用)。
    pub fn w_q_mut(&mut self) -> &mut [f32] {
        &mut self.w_q
    }

    /// K projection weight への可変参照。
    pub fn w_k_mut(&mut self) -> &mut [f32] {
        &mut self.w_k
    }

    /// V projection weight への可変参照。
    pub fn w_v_mut(&mut self) -> &mut [f32] {
        &mut self.w_v
    }

    /// O projection weight への可変参照。
    pub fn w_o_mut(&mut self) -> &mut [f32] {
        &mut self.w_o
    }

    /// Q projection bias への可変参照。
    pub fn b_q_mut(&mut self) -> &mut [f32] {
        &mut self.b_q
    }

    /// K projection bias への可変参照。
    pub fn b_k_mut(&mut self) -> &mut [f32] {
        &mut self.b_k
    }

    /// V projection bias への可変参照。
    pub fn b_v_mut(&mut self) -> &mut [f32] {
        &mut self.b_v
    }

    /// O projection bias への可変参照。
    pub fn b_o_mut(&mut self) -> &mut [f32] {
        &mut self.b_o
    }

    /// SGD update: `w -= lr * grad` を全 8 tensor に適用する (Q/K/V/O weight + bias)。
    pub fn apply_sgd(&mut self, grads: &MhaGrads, lr: f32) {
        for (w, g) in self.w_q.iter_mut().zip(&grads.w_q) {
            *w -= lr * g;
        }
        for (w, g) in self.w_k.iter_mut().zip(&grads.w_k) {
            *w -= lr * g;
        }
        for (w, g) in self.w_v.iter_mut().zip(&grads.w_v) {
            *w -= lr * g;
        }
        for (w, g) in self.w_o.iter_mut().zip(&grads.w_o) {
            *w -= lr * g;
        }
        if self.config.bias {
            for (b, g) in self.b_q.iter_mut().zip(&grads.b_q) {
                *b -= lr * g;
            }
            for (b, g) in self.b_k.iter_mut().zip(&grads.b_k) {
                *b -= lr * g;
            }
            for (b, g) in self.b_v.iter_mut().zip(&grads.b_v) {
                *b -= lr * g;
            }
            for (b, g) in self.b_o.iter_mut().zip(&grads.b_o) {
                *b -= lr * g;
            }
        }
    }

    /// self-attention forward (Q=K=V=input)。
    ///
    /// # Errors
    ///
    /// input shape mismatch。
    pub fn forward_self_attention(
        &self,
        input: &[f32],
        batch: usize,
        seq_len: usize,
        causal: bool,
    ) -> Result<Vec<f32>, MhaError> {
        self.forward(input, input, input, batch, seq_len, seq_len, causal)
    }

    /// cross-attention forward。
    ///
    /// # 引数
    ///
    /// - `q`: `[batch, seq_q, embed_dim]` flatten
    /// - `k`, `v`: `[batch, seq_k, embed_dim]` flatten
    /// - `causal`: causal mask 有効化 (self-attention 用途、cross-attention では通常 false)
    ///
    /// # Errors
    ///
    /// shape 不整合。
    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        causal: bool,
    ) -> Result<Vec<f32>, MhaError> {
        let cfg = self.config;
        let e = cfg.embed_dim;
        check_input_shape("q", q.len(), batch * seq_q * e)?;
        check_input_shape("k", k.len(), batch * seq_k * e)?;
        check_input_shape("v", v.len(), batch * seq_k * e)?;

        // 1. Project Q/K/V
        let q_proj = linear_forward(q, &self.w_q, &self.b_q, batch * seq_q, e, e, cfg.bias);
        let k_proj = linear_forward(k, &self.w_k, &self.b_k, batch * seq_k, e, e, cfg.bias);
        let v_proj = linear_forward(v, &self.w_v, &self.b_v, batch * seq_k, e, e, cfg.bias);

        // 2. Split heads: [B, S, E] -> [B, H, S, D]  (index-only reshape, no data move)
        // 3-4. scaled dot-product + optional causal mask + softmax
        // 5. attn @ v_h -> out_h: [B, H, Sq, D]
        // 6. concat heads: [B, H, Sq, D] -> [B, Sq, E]
        let attn_out = self
            .scaled_dot_product_attention(&q_proj, &k_proj, &v_proj, batch, seq_q, seq_k, causal);

        // 7. Output projection
        let output = linear_forward(
            &attn_out,
            &self.w_o,
            &self.b_o,
            batch * seq_q,
            e,
            e,
            cfg.bias,
        );
        Ok(output)
    }

    /// backward pass (self-attention Q=K=V=input)。
    ///
    /// # 戻り値
    ///
    /// `(grad_input, grads)` where `grads` は Q/K/V/O の weight + bias 勾配を保持。
    ///
    /// # Errors
    ///
    /// shape 不整合。
    pub fn backward_self_attention(
        &self,
        input: &[f32],
        grad_output: &[f32],
        batch: usize,
        seq_len: usize,
        causal: bool,
    ) -> Result<(Vec<f32>, MhaGrads), MhaError> {
        let (grad_q, grad_k, grad_v, grads) = self.backward(
            input,
            input,
            input,
            grad_output,
            batch,
            seq_len,
            seq_len,
            causal,
        )?;
        let mut grad_input = grad_q;
        for i in 0..grad_input.len() {
            grad_input[i] += grad_k[i] + grad_v[i];
        }
        Ok((grad_input, grads))
    }

    /// cross-attention backward。grad_q / grad_k / grad_v をそれぞれ返す。
    ///
    /// # Errors
    ///
    /// shape 不整合。
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        grad_output: &[f32],
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        causal: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, MhaGrads), MhaError> {
        let cfg = self.config;
        let e = cfg.embed_dim;
        check_input_shape("q", q.len(), batch * seq_q * e)?;
        check_input_shape("k", k.len(), batch * seq_k * e)?;
        check_input_shape("v", v.len(), batch * seq_k * e)?;
        check_input_shape("grad_output", grad_output.len(), batch * seq_q * e)?;

        // === forward recompute (need intermediate activations) ===
        let q_proj = linear_forward(q, &self.w_q, &self.b_q, batch * seq_q, e, e, cfg.bias);
        let k_proj = linear_forward(k, &self.w_k, &self.b_k, batch * seq_k, e, e, cfg.bias);
        let v_proj = linear_forward(v, &self.w_v, &self.b_v, batch * seq_k, e, e, cfg.bias);

        // scaled dot-product returns (attn_out, attn_weights) for backward reuse
        let (attn_out, attn_weights) = self.scaled_dot_product_attention_with_weights(
            &q_proj, &k_proj, &v_proj, batch, seq_q, seq_k, causal, None,
        );

        // === backward ===
        // 7. Output projection backward
        let (grad_attn_out, grad_w_o, grad_b_o) = linear_backward(
            &attn_out,
            grad_output,
            &self.w_o,
            batch * seq_q,
            e,
            e,
            cfg.bias,
        );

        // 6. concat heads → split heads (grad_attn_out shape: [B, Sq, E] flatten)
        // 5. attn @ v_h -> out_h backward
        // Layout: attn_out[b, sq, e] = Σ_h Σ_sk attn_weights[b, h, sq, sk] * v_h[b, h, sk, d]  (d = e - h*D within head)
        let (grad_attn_weights, grad_v_proj) = attn_matmul_v_backward(
            &attn_weights,
            &v_proj,
            &grad_attn_out,
            batch,
            seq_q,
            seq_k,
            cfg.num_heads,
            cfg.head_dim(),
        );

        // 4. softmax backward
        let grad_scores = softmax_backward_4d(
            &attn_weights,
            &grad_attn_weights,
            batch,
            cfg.num_heads,
            seq_q,
            seq_k,
        );

        // 3. scaled dot-product Q_h @ K_h^T / sqrt(D) backward
        let (grad_q_proj, grad_k_proj) = score_backward(
            &q_proj,
            &k_proj,
            &grad_scores,
            batch,
            seq_q,
            seq_k,
            cfg.num_heads,
            cfg.head_dim(),
        );

        // 1. Q/K/V projection backward
        let (grad_q, grad_w_q, grad_b_q) =
            linear_backward(q, &grad_q_proj, &self.w_q, batch * seq_q, e, e, cfg.bias);
        let (grad_k, grad_w_k, grad_b_k) =
            linear_backward(k, &grad_k_proj, &self.w_k, batch * seq_k, e, e, cfg.bias);
        let (grad_v, grad_w_v, grad_b_v) =
            linear_backward(v, &grad_v_proj, &self.w_v, batch * seq_k, e, e, cfg.bias);

        Ok((
            grad_q,
            grad_k,
            grad_v,
            MhaGrads {
                w_q: grad_w_q,
                w_k: grad_w_k,
                w_v: grad_w_v,
                w_o: grad_w_o,
                b_q: grad_b_q,
                b_k: grad_b_k,
                b_v: grad_b_v,
                b_o: grad_b_o,
            },
        ))
    }

    /// self-attention forward with key_mask (Phase T.4a Phase D)。
    ///
    /// `key_mask`: `[batch, seq_len]` bool、false = padded position (attention から除外)。
    ///
    /// # Errors
    ///
    /// shape mismatch。
    pub fn forward_self_attention_masked(
        &self,
        input: &[f32],
        batch: usize,
        seq_len: usize,
        causal: bool,
        key_mask: &[bool],
    ) -> Result<Vec<f32>, MhaError> {
        self.forward_masked(
            input,
            input,
            input,
            batch,
            seq_len,
            seq_len,
            causal,
            Some(key_mask),
        )
    }

    /// cross-attention forward with optional key_mask。
    ///
    /// # Errors
    ///
    /// - shape 不整合 (q/k/v/`key_mask`)
    /// - forward path のエラー
    #[allow(clippy::too_many_arguments)]
    pub fn forward_masked(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        causal: bool,
        key_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>, MhaError> {
        let cfg = self.config;
        let e = cfg.embed_dim;
        check_input_shape("q", q.len(), batch * seq_q * e)?;
        check_input_shape("k", k.len(), batch * seq_k * e)?;
        check_input_shape("v", v.len(), batch * seq_k * e)?;
        if let Some(m) = key_mask {
            check_input_shape("key_mask", m.len(), batch * seq_k)?;
        }
        let q_proj = linear_forward(q, &self.w_q, &self.b_q, batch * seq_q, e, e, cfg.bias);
        let k_proj = linear_forward(k, &self.w_k, &self.b_k, batch * seq_k, e, e, cfg.bias);
        let v_proj = linear_forward(v, &self.w_v, &self.b_v, batch * seq_k, e, e, cfg.bias);
        let (attn_out, _) = self.scaled_dot_product_attention_with_weights(
            &q_proj, &k_proj, &v_proj, batch, seq_q, seq_k, causal, key_mask,
        );
        let output = linear_forward(
            &attn_out,
            &self.w_o,
            &self.b_o,
            batch * seq_q,
            e,
            e,
            cfg.bias,
        );
        Ok(output)
    }

    /// self-attention backward with key_mask (Phase T.4a Phase D)。
    ///
    /// # Errors
    ///
    /// shape mismatch。
    pub fn backward_self_attention_masked(
        &self,
        input: &[f32],
        grad_output: &[f32],
        batch: usize,
        seq_len: usize,
        causal: bool,
        key_mask: &[bool],
    ) -> Result<(Vec<f32>, MhaGrads), MhaError> {
        let (grad_q, grad_k, grad_v, grads) = self.backward_masked(
            input,
            input,
            input,
            grad_output,
            batch,
            seq_len,
            seq_len,
            causal,
            Some(key_mask),
        )?;
        let mut grad_input = grad_q;
        for i in 0..grad_input.len() {
            grad_input[i] += grad_k[i] + grad_v[i];
        }
        Ok((grad_input, grads))
    }

    /// cross-attention backward with optional key_mask。
    ///
    /// # Errors
    ///
    /// - shape 不整合 (q/k/v/`grad_output`/`key_mask`)
    /// - backward path のエラー
    #[allow(clippy::too_many_arguments)]
    pub fn backward_masked(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        grad_output: &[f32],
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        causal: bool,
        key_mask: Option<&[bool]>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, MhaGrads), MhaError> {
        let cfg = self.config;
        let e = cfg.embed_dim;
        check_input_shape("q", q.len(), batch * seq_q * e)?;
        check_input_shape("k", k.len(), batch * seq_k * e)?;
        check_input_shape("v", v.len(), batch * seq_k * e)?;
        check_input_shape("grad_output", grad_output.len(), batch * seq_q * e)?;
        if let Some(m) = key_mask {
            check_input_shape("key_mask", m.len(), batch * seq_k)?;
        }
        let q_proj = linear_forward(q, &self.w_q, &self.b_q, batch * seq_q, e, e, cfg.bias);
        let k_proj = linear_forward(k, &self.w_k, &self.b_k, batch * seq_k, e, e, cfg.bias);
        let v_proj = linear_forward(v, &self.w_v, &self.b_v, batch * seq_k, e, e, cfg.bias);
        let (attn_out, attn_weights) = self.scaled_dot_product_attention_with_weights(
            &q_proj, &k_proj, &v_proj, batch, seq_q, seq_k, causal, key_mask,
        );
        let (grad_attn_out, grad_w_o, grad_b_o) = linear_backward(
            &attn_out,
            grad_output,
            &self.w_o,
            batch * seq_q,
            e,
            e,
            cfg.bias,
        );
        let (grad_attn_weights, grad_v_proj) = attn_matmul_v_backward(
            &attn_weights,
            &v_proj,
            &grad_attn_out,
            batch,
            seq_q,
            seq_k,
            cfg.num_heads,
            cfg.head_dim(),
        );
        let grad_scores = softmax_backward_4d(
            &attn_weights,
            &grad_attn_weights,
            batch,
            cfg.num_heads,
            seq_q,
            seq_k,
        );
        let (grad_q_proj, grad_k_proj) = score_backward(
            &q_proj,
            &k_proj,
            &grad_scores,
            batch,
            seq_q,
            seq_k,
            cfg.num_heads,
            cfg.head_dim(),
        );
        let (grad_q, grad_w_q, grad_b_q) =
            linear_backward(q, &grad_q_proj, &self.w_q, batch * seq_q, e, e, cfg.bias);
        let (grad_k, grad_w_k, grad_b_k) =
            linear_backward(k, &grad_k_proj, &self.w_k, batch * seq_k, e, e, cfg.bias);
        let (grad_v, grad_w_v, grad_b_v) =
            linear_backward(v, &grad_v_proj, &self.w_v, batch * seq_k, e, e, cfg.bias);
        Ok((
            grad_q,
            grad_k,
            grad_v,
            MhaGrads {
                w_q: grad_w_q,
                w_k: grad_w_k,
                w_v: grad_w_v,
                w_o: grad_w_o,
                b_q: grad_b_q,
                b_k: grad_b_k,
                b_v: grad_b_v,
                b_o: grad_b_o,
            },
        ))
    }

    /// scaled dot-product attention forward (attn weights は捨てる版、forward 用)。
    fn scaled_dot_product_attention(
        &self,
        q_proj: &[f32],
        k_proj: &[f32],
        v_proj: &[f32],
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        causal: bool,
    ) -> Vec<f32> {
        let (attn_out, _) = self.scaled_dot_product_attention_with_weights(
            q_proj, k_proj, v_proj, batch, seq_q, seq_k, causal, None,
        );
        attn_out
    }

    /// scaled dot-product attention forward (attn weights も保存、backward 用)。
    ///
    /// `key_mask` (optional): `[batch, seq_k]` bool、false = padded key を attention から除外。
    fn scaled_dot_product_attention_with_weights(
        &self,
        q_proj: &[f32],
        k_proj: &[f32],
        v_proj: &[f32],
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        causal: bool,
        key_mask: Option<&[bool]>,
    ) -> (Vec<f32>, Vec<f32>) {
        let cfg = self.config;
        let h = cfg.num_heads;
        let d = cfg.head_dim();
        let e = cfg.embed_dim;
        let scale = 1.0 / (d as f32).sqrt();

        // attn_weights shape: [batch, h, seq_q, seq_k]
        let mut attn_weights = vec![0.0_f32; batch * h * seq_q * seq_k];
        // attn_out shape: [batch, seq_q, e] (concat heads 後)
        let mut attn_out = vec![0.0_f32; batch * seq_q * e];

        // Phase T.4b-A: Per-head 抽出 buffer (Q_h, K_h, V_h, attn_h) を再利用
        // per-head 抽出は O(seq × d) の memory copy、その後 matmul O(seq² × d) が BLAS 化
        let mut q_h = vec![0.0_f32; seq_q * d];
        let mut k_h = vec![0.0_f32; seq_k * d];
        let mut v_h = vec![0.0_f32; seq_k * d];
        let mut attn_h = vec![0.0_f32; seq_q * d];
        let mut scores = vec![0.0_f32; seq_q * seq_k];

        for b in 0..batch {
            for head in 0..h {
                // Q_h / K_h / V_h を contiguous 抽出
                for sq in 0..seq_q {
                    let src = b * seq_q * e + sq * e + head * d;
                    q_h[sq * d..sq * d + d].copy_from_slice(&q_proj[src..src + d]);
                }
                for sk in 0..seq_k {
                    let src_k = b * seq_k * e + sk * e + head * d;
                    k_h[sk * d..sk * d + d].copy_from_slice(&k_proj[src_k..src_k + d]);
                    v_h[sk * d..sk * d + d].copy_from_slice(&v_proj[src_k..src_k + d]);
                }

                // 1. scores = Q_h @ K_h^T (BLAS), shape [seq_q, seq_k]
                crate::blas::blas_matmul_bt(&q_h, &k_h, &mut scores, seq_q, seq_k, d);
                // scale by 1/sqrt(d)
                for v in scores.iter_mut() {
                    *v *= scale;
                }

                // 2. causal mask
                if causal {
                    for sq in 0..seq_q {
                        for sk in (sq + 1)..seq_k {
                            scores[sq * seq_k + sk] = f32::NEG_INFINITY;
                        }
                    }
                }

                // 2b. key_mask: false 位置の key を attention から除外
                if let Some(mask) = key_mask {
                    for sq in 0..seq_q {
                        for sk in 0..seq_k {
                            if !mask[b * seq_k + sk] {
                                scores[sq * seq_k + sk] = f32::NEG_INFINITY;
                            }
                        }
                    }
                }

                // 3. softmax (per row)
                for sq in 0..seq_q {
                    let row_start = sq * seq_k;
                    let row = &mut scores[row_start..row_start + seq_k];
                    softmax_row(row);
                }

                // save attn weights for backward
                for sq in 0..seq_q {
                    let src = sq * seq_k;
                    let dst = b * h * seq_q * seq_k + head * seq_q * seq_k + sq * seq_k;
                    attn_weights[dst..dst + seq_k].copy_from_slice(&scores[src..src + seq_k]);
                }

                // 4. attn @ V_h (BLAS), shape [seq_q, d]
                crate::blas::blas_matmul_nn(&scores, &v_h, &mut attn_h, seq_q, d, seq_k);
                // Write attn_h back to attn_out[b, sq, head*d..head*d+d]
                for sq in 0..seq_q {
                    let dst = b * seq_q * e + sq * e + head * d;
                    attn_out[dst..dst + d].copy_from_slice(&attn_h[sq * d..sq * d + d]);
                }
            }
        }

        (attn_out, attn_weights)
    }
}

/// MHA backward で返される weight/bias 勾配 bundle。
#[derive(Clone, Debug)]
pub struct MhaGrads {
    /// Q projection weight 勾配 `[embed_dim, embed_dim]` flatten。
    pub w_q: Vec<f32>,
    /// K projection weight 勾配。
    pub w_k: Vec<f32>,
    /// V projection weight 勾配。
    pub w_v: Vec<f32>,
    /// O projection weight 勾配。
    pub w_o: Vec<f32>,
    /// Q projection bias 勾配 `[embed_dim]`。
    pub b_q: Vec<f32>,
    /// K projection bias 勾配。
    pub b_k: Vec<f32>,
    /// V projection bias 勾配。
    pub b_v: Vec<f32>,
    /// O projection bias 勾配。
    pub b_o: Vec<f32>,
}

/// 線形層 forward: `y[m, out] = Σ_in x[m, in] * w[out, in] + b[out]` (Phase T.4b: BLAS 化)。
fn linear_forward(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    m: usize,
    in_dim: usize,
    out_dim: usize,
    has_bias: bool,
) -> Vec<f32> {
    let mut y = vec![0.0_f32; m * out_dim];
    // y[m, out] = x[m, in] × w[out, in]^T
    crate::blas::blas_matmul_bt(x, w, &mut y, m, out_dim, in_dim);
    if has_bias {
        for row in 0..m {
            for o in 0..out_dim {
                y[row * out_dim + o] += b[o];
            }
        }
    }
    y
}

/// 線形層 backward: return (grad_x, grad_w, grad_b) (Phase T.4b: BLAS 化)。
fn linear_backward(
    x: &[f32],
    grad_y: &[f32],
    w: &[f32],
    m: usize,
    in_dim: usize,
    out_dim: usize,
    has_bias: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut grad_x = vec![0.0_f32; m * in_dim];
    let mut grad_w = vec![0.0_f32; out_dim * in_dim];
    let mut grad_b = vec![0.0_f32; out_dim];

    // grad_x[m, in] = grad_y[m, out] × w[out, in]
    crate::blas::blas_matmul_nn(grad_y, w, &mut grad_x, m, in_dim, out_dim);
    // grad_w[out, in] = grad_y[m, out]^T × x[m, in]
    crate::blas::blas_matmul_tn(grad_y, x, &mut grad_w, out_dim, in_dim, m);
    if has_bias {
        for row in 0..m {
            for o in 0..out_dim {
                grad_b[o] += grad_y[row * out_dim + o];
            }
        }
    }

    (grad_x, grad_w, grad_b)
}

/// softmax row in-place (numerical stability: subtract max)。
fn softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max.is_infinite() {
        // 全 -inf の場合、uniform 0 に (mask で全 masked 状態、実運用ではまず起きない)
        for v in row.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// `attn @ V` backward (Phase T.4b-A: per-head BLAS 化):
/// `attn_out[b, h, sq, d] = Σ_sk attn_w[b, h, sq, sk] * v_h[b, h, sk, d]`
///
/// Returns (grad_attn_weights, grad_v_proj)。
fn attn_matmul_v_backward(
    attn_weights: &[f32],
    v_proj: &[f32],
    grad_attn_out: &[f32],
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    h: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>) {
    let e = h * d;
    let mut grad_attn_weights = vec![0.0_f32; batch * h * seq_q * seq_k];
    let mut grad_v_proj = vec![0.0_f32; batch * seq_k * e];

    // per-head buffer 再利用
    let mut v_h = vec![0.0_f32; seq_k * d];
    let mut grad_out_h = vec![0.0_f32; seq_q * d];
    let mut grad_v_h = vec![0.0_f32; seq_k * d];

    for b in 0..batch {
        for head in 0..h {
            // V_h と grad_attn_out_h を contiguous 抽出
            for sk in 0..seq_k {
                let src = b * seq_k * e + sk * e + head * d;
                v_h[sk * d..sk * d + d].copy_from_slice(&v_proj[src..src + d]);
            }
            for sq in 0..seq_q {
                let src = b * seq_q * e + sq * e + head * d;
                grad_out_h[sq * d..sq * d + d].copy_from_slice(&grad_attn_out[src..src + d]);
            }

            // grad_attn_h[seq_q, seq_k] = grad_out_h[seq_q, d] × V_h[seq_k, d]^T
            let g_attn_base = b * h * seq_q * seq_k + head * seq_q * seq_k;
            crate::blas::blas_matmul_bt(
                &grad_out_h,
                &v_h,
                &mut grad_attn_weights[g_attn_base..g_attn_base + seq_q * seq_k],
                seq_q,
                seq_k,
                d,
            );

            // grad_V_h[seq_k, d] = attn_h[seq_q, seq_k]^T × grad_out_h[seq_q, d]
            let attn_h = &attn_weights[g_attn_base..g_attn_base + seq_q * seq_k];
            crate::blas::blas_matmul_tn(attn_h, &grad_out_h, &mut grad_v_h, seq_k, d, seq_q);
            // grad_V_h → grad_v_proj (per-head 位置に加算 = copy)
            for sk in 0..seq_k {
                let dst = b * seq_k * e + sk * e + head * d;
                for dd in 0..d {
                    grad_v_proj[dst + dd] += grad_v_h[sk * d + dd];
                }
            }
        }
    }

    (grad_attn_weights, grad_v_proj)
}

/// Softmax backward (row-wise、per [b, h, sq]):
/// `grad_scores[sk] = attn[sk] * (grad_attn[sk] - Σ_j attn[j] * grad_attn[j])`
fn softmax_backward_4d(
    attn_weights: &[f32],
    grad_attn_weights: &[f32],
    batch: usize,
    h: usize,
    seq_q: usize,
    seq_k: usize,
) -> Vec<f32> {
    let mut grad_scores = vec![0.0_f32; batch * h * seq_q * seq_k];
    for b in 0..batch {
        for head in 0..h {
            for sq in 0..seq_q {
                let base = b * h * seq_q * seq_k + head * seq_q * seq_k + sq * seq_k;
                let attn = &attn_weights[base..base + seq_k];
                let g_attn = &grad_attn_weights[base..base + seq_k];
                let mut sum = 0.0_f32;
                for sk in 0..seq_k {
                    sum += attn[sk] * g_attn[sk];
                }
                for sk in 0..seq_k {
                    grad_scores[base + sk] = attn[sk] * (g_attn[sk] - sum);
                }
            }
        }
    }
    grad_scores
}

/// Scaled dot-product `scores = Q_h @ K_h^T / sqrt(D)` backward (Phase T.4b-A: per-head BLAS 化):
/// `grad_Q_h[sq, d] = (1/sqrt(D)) * Σ_sk grad_scores[sq, sk] * K_h[sk, d]`
/// `grad_K_h[sk, d] = (1/sqrt(D)) * Σ_sq grad_scores[sq, sk] * Q_h[sq, d]`
fn score_backward(
    q_proj: &[f32],
    k_proj: &[f32],
    grad_scores: &[f32],
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    h: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>) {
    let e = h * d;
    let scale = 1.0 / (d as f32).sqrt();
    let mut grad_q_proj = vec![0.0_f32; batch * seq_q * e];
    let mut grad_k_proj = vec![0.0_f32; batch * seq_k * e];

    // per-head buffer
    let mut q_h = vec![0.0_f32; seq_q * d];
    let mut k_h = vec![0.0_f32; seq_k * d];
    let mut grad_scores_scaled = vec![0.0_f32; seq_q * seq_k];
    let mut grad_q_h = vec![0.0_f32; seq_q * d];
    let mut grad_k_h = vec![0.0_f32; seq_k * d];

    for b in 0..batch {
        for head in 0..h {
            // Q_h / K_h contiguous 抽出
            for sq in 0..seq_q {
                let src = b * seq_q * e + sq * e + head * d;
                q_h[sq * d..sq * d + d].copy_from_slice(&q_proj[src..src + d]);
            }
            for sk in 0..seq_k {
                let src = b * seq_k * e + sk * e + head * d;
                k_h[sk * d..sk * d + d].copy_from_slice(&k_proj[src..src + d]);
            }
            // scaled grad_scores を per-head 抽出 (scale をここで適用)
            let gs_base = b * h * seq_q * seq_k + head * seq_q * seq_k;
            for i in 0..(seq_q * seq_k) {
                grad_scores_scaled[i] = grad_scores[gs_base + i] * scale;
            }

            // grad_Q_h[seq_q, d] = grad_scores_scaled[seq_q, seq_k] @ K_h[seq_k, d]
            crate::blas::blas_matmul_nn(&grad_scores_scaled, &k_h, &mut grad_q_h, seq_q, d, seq_k);
            // grad_K_h[seq_k, d] = grad_scores_scaled[seq_q, seq_k]^T @ Q_h[seq_q, d]
            crate::blas::blas_matmul_tn(&grad_scores_scaled, &q_h, &mut grad_k_h, seq_k, d, seq_q);

            // 位置に加算 (accumulate は copy_from_slice ではなく += が必要)
            for sq in 0..seq_q {
                let dst = b * seq_q * e + sq * e + head * d;
                for dd in 0..d {
                    grad_q_proj[dst + dd] += grad_q_h[sq * d + dd];
                }
            }
            for sk in 0..seq_k {
                let dst = b * seq_k * e + sk * e + head * d;
                for dd in 0..d {
                    grad_k_proj[dst + dd] += grad_k_h[sk * d + dd];
                }
            }
        }
    }

    (grad_q_proj, grad_k_proj)
}

fn check_input_shape(field: &'static str, actual: usize, expected: usize) -> Result<(), MhaError> {
    if actual == expected {
        Ok(())
    } else {
        Err(MhaError::ShapeMismatch {
            field,
            expected,
            actual,
        })
    }
}

/// Multi-Head Attention 操作で発生し得るエラー。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MhaError {
    /// config が不正 (embed_dim=0、num_heads=0、embed_dim % num_heads != 0)。
    InvalidConfig {
        /// 具体的な理由。
        reason: String,
    },
    /// weight/bias/input/grad_output の shape 不整合。
    ShapeMismatch {
        /// 対象 field 名。
        field: &'static str,
        /// 期待 len。
        expected: usize,
        /// 実際 len。
        actual: usize,
    },
}

impl std::fmt::Display for MhaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig { reason } => {
                write!(f, "invalid MultiHeadAttention config: {reason}")
            }
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch on '{field}': expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for MhaError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mha_with_random(
        config: MultiHeadAttentionConfig,
        seed_offset: f32,
    ) -> MultiHeadAttention {
        let e = config.embed_dim;
        let n_w = e * e;
        let n_b = if config.bias { e } else { 0 };
        let mk_w = |off: f32| {
            (0..n_w)
                .map(|i| ((i as f32 + off) * 0.13).sin() * 0.3)
                .collect::<Vec<f32>>()
        };
        let mk_b = |off: f32| {
            (0..n_b)
                .map(|i| ((i as f32 + off) * 0.07).cos() * 0.1)
                .collect::<Vec<f32>>()
        };
        MultiHeadAttention::new(
            config,
            mk_w(1.0 + seed_offset),
            mk_w(2.0 + seed_offset),
            mk_w(3.0 + seed_offset),
            mk_w(4.0 + seed_offset),
            mk_b(5.0 + seed_offset),
            mk_b(6.0 + seed_offset),
            mk_b(7.0 + seed_offset),
            mk_b(8.0 + seed_offset),
        )
        .unwrap()
    }

    #[test]
    fn config_head_dim_and_validation() {
        let cfg = MultiHeadAttentionConfig::new(64, 4);
        assert_eq!(cfg.head_dim(), 16);
        cfg.validate().expect("valid");

        let bad = MultiHeadAttentionConfig::new(64, 5);
        assert!(bad.validate().is_err());

        let bad = MultiHeadAttentionConfig::new(0, 4);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn zero_weights_produce_bias_only_output() {
        // 全 weight = 0 → attn = softmax(0) = uniform 1/seq_k, attn_out = 平均、しかし O_proj weight = 0
        // → output = O_bias 全て (b_o) を各 [b, sq] に broadcast
        let cfg = MultiHeadAttentionConfig::new(4, 2);
        let mut mha = MultiHeadAttention::zeros(cfg).unwrap();
        mha.b_o[0] = 1.0;
        mha.b_o[1] = 2.0;
        mha.b_o[2] = 3.0;
        mha.b_o[3] = 4.0;
        let input = vec![0.1_f32; 2 * 3 * 4]; // batch=2, seq=3, embed=4
        let out = mha.forward_self_attention(&input, 2, 3, false).unwrap();
        for b in 0..2 {
            for sq in 0..3 {
                for e in 0..4 {
                    let idx = b * 3 * 4 + sq * 4 + e;
                    assert!(
                        (out[idx] - mha.b_o[e]).abs() < 1e-4,
                        "out[{b}, {sq}, {e}]={} expected b_o[{e}]={}",
                        out[idx],
                        mha.b_o[e]
                    );
                }
            }
        }
    }

    #[test]
    fn output_shape_matches_input_shape() {
        let cfg = MultiHeadAttentionConfig::new(8, 2);
        let mha = make_mha_with_random(cfg, 0.0);
        let input: Vec<f32> = (0..2 * 5 * 8).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = mha.forward_self_attention(&input, 2, 5, false).unwrap();
        assert_eq!(out.len(), 2 * 5 * 8);
    }

    #[test]
    fn causal_mask_prevents_future_attention() {
        // causal で attention weight upper triangular が 0
        // これは attn_weights 内部変数の検証、外部から見えないので indirect test:
        // 同じ input で causal on/off して結果が違うこと
        let cfg = MultiHeadAttentionConfig::new(4, 2);
        let mha = make_mha_with_random(cfg, 0.0);
        let input: Vec<f32> = (0..4 * 4).map(|i| (i as f32 * 0.3).cos()).collect();
        let out_causal = mha.forward_self_attention(&input, 1, 4, true).unwrap();
        let out_bi = mha.forward_self_attention(&input, 1, 4, false).unwrap();
        // causal と bidirectional で少なくとも 1 element 違う (position 0 は同じ、後方は違う)
        let mut any_diff = false;
        for (a, b) in out_causal.iter().zip(&out_bi) {
            if (a - b).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "causal and bidirectional outputs should differ");
    }

    #[test]
    fn backward_gradient_matches_finite_difference() {
        let cfg = MultiHeadAttentionConfig::new(4, 2);
        let mha = make_mha_with_random(cfg, 0.0);
        let batch = 2;
        let seq_len = 3;
        let e = 4;
        let input: Vec<f32> = (0..batch * seq_len * e)
            .map(|i| (i as f32 * 0.2).sin() * 0.5)
            .collect();
        let grad_output: Vec<f32> = (0..batch * seq_len * e)
            .map(|i| (i as f32 * 0.15 + 0.3).cos() * 0.3)
            .collect();

        let (analytical, _) = mha
            .backward_self_attention(&input, &grad_output, batch, seq_len, false)
            .unwrap();

        let h = 5e-3_f32;
        for i in 0..input.len() {
            let mut ip = input.clone();
            ip[i] += h;
            let out_p = mha
                .forward_self_attention(&ip, batch, seq_len, false)
                .unwrap();
            let mut im = input.clone();
            im[i] -= h;
            let out_m = mha
                .forward_self_attention(&im, batch, seq_len, false)
                .unwrap();
            let loss_p: f32 = out_p.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let loss_m: f32 = out_m.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let ana = analytical[i];
            let diff = (ana - num).abs();
            let scale = ana.abs().max(num.abs()).max(1e-3);
            assert!(
                diff / scale < 5e-2,
                "input[{i}] analytical={ana}, numerical={num}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn backward_causal_gradient_matches_finite_difference() {
        let cfg = MultiHeadAttentionConfig::new(4, 2);
        let mha = make_mha_with_random(cfg, 0.5);
        let batch = 1;
        let seq_len = 3;
        let e = 4;
        let input: Vec<f32> = (0..batch * seq_len * e)
            .map(|i| (i as f32 * 0.2).cos() * 0.5)
            .collect();
        let grad_output: Vec<f32> = (0..batch * seq_len * e)
            .map(|i| (i as f32 * 0.11).sin() * 0.4)
            .collect();

        let (analytical, _) = mha
            .backward_self_attention(&input, &grad_output, batch, seq_len, true)
            .unwrap();

        let h = 5e-3_f32;
        for i in 0..input.len() {
            let mut ip = input.clone();
            ip[i] += h;
            let out_p = mha
                .forward_self_attention(&ip, batch, seq_len, true)
                .unwrap();
            let mut im = input.clone();
            im[i] -= h;
            let out_m = mha
                .forward_self_attention(&im, batch, seq_len, true)
                .unwrap();
            let loss_p: f32 = out_p.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let loss_m: f32 = out_m.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let ana = analytical[i];
            let diff = (ana - num).abs();
            let scale = ana.abs().max(num.abs()).max(1e-3);
            assert!(
                diff / scale < 5e-2,
                "causal input[{i}] analytical={ana}, numerical={num}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn backward_weight_gradient_matches_finite_difference() {
        // grad_w_o の finite-diff 検証 (全部やると重いので w_o のみ)
        let cfg = MultiHeadAttentionConfig::new(4, 2);
        let mha = make_mha_with_random(cfg, 0.0);
        let batch = 1;
        let seq_len = 3;
        let e = 4;
        let input: Vec<f32> = (0..batch * seq_len * e)
            .map(|i| (i as f32 * 0.25).sin() * 0.5)
            .collect();
        let grad_output: Vec<f32> = (0..batch * seq_len * e)
            .map(|i| (i as f32 * 0.17).cos() * 0.3)
            .collect();

        let (_, grads) = mha
            .backward_self_attention(&input, &grad_output, batch, seq_len, false)
            .unwrap();

        let h = 1e-3_f32;
        for i in 0..mha.w_o.len() {
            let mut mha_p = mha.clone();
            mha_p.w_o[i] += h;
            let out_p = mha_p
                .forward_self_attention(&input, batch, seq_len, false)
                .unwrap();
            let mut mha_m = mha.clone();
            mha_m.w_o[i] -= h;
            let out_m = mha_m
                .forward_self_attention(&input, batch, seq_len, false)
                .unwrap();
            let loss_p: f32 = out_p.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let loss_m: f32 = out_m.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let num = (loss_p - loss_m) / (2.0 * h);
            let ana = grads.w_o[i];
            let diff = (ana - num).abs();
            let scale = ana.abs().max(num.abs()).max(1e-3);
            assert!(
                diff / scale < 5e-2,
                "w_o[{i}] analytical={ana}, numerical={num}, rel_err={}",
                diff / scale
            );
        }
    }

    #[test]
    fn cross_attention_forward_returns_seq_q_shape() {
        // Q: seq_q=2, K/V: seq_k=5, output shape [batch, seq_q, embed]
        let cfg = MultiHeadAttentionConfig::new(4, 2);
        let mha = make_mha_with_random(cfg, 0.0);
        let q: Vec<f32> = (0..2 * 4).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..5 * 4).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..5 * 4).map(|i| (i as f32 * 0.3).sin()).collect();
        let out = mha.forward(&q, &k, &v, 1, 2, 5, false).unwrap();
        assert_eq!(out.len(), 2 * 4);
    }

    #[test]
    fn softmax_row_sums_to_one() {
        let mut row = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_all_neg_inf_returns_zero() {
        let mut row = vec![f32::NEG_INFINITY; 3];
        softmax_row(&mut row);
        for &v in &row {
            assert!((v).abs() < 1e-6);
        }
    }

    #[test]
    fn invalid_config_and_shape_errors() {
        let bad_cfg = MultiHeadAttentionConfig::new(0, 4);
        assert!(MultiHeadAttention::zeros(bad_cfg).is_err());

        let cfg = MultiHeadAttentionConfig::new(4, 2);
        let mha = MultiHeadAttention::zeros(cfg).unwrap();
        let err = mha
            .forward_self_attention(&[0.0; 10], 1, 3, false)
            .expect_err("shape mismatch");
        assert!(matches!(err, MhaError::ShapeMismatch { .. }));
    }

    #[test]
    fn error_display_and_error_trait() {
        let e = MhaError::InvalidConfig {
            reason: "test".to_string(),
        };
        let s = format!("{e}");
        assert!(s.contains("invalid MultiHeadAttention"));
        let boxed: Box<dyn std::error::Error> = Box::new(e);
        assert!(boxed.to_string().contains("test"));
    }

    #[test]
    fn without_bias_returns_zero_bias_grad() {
        let cfg = MultiHeadAttentionConfig::new(4, 2).with_bias(false);
        let mha = MultiHeadAttention::new(
            cfg,
            vec![0.1_f32; 16],
            vec![0.2_f32; 16],
            vec![0.3_f32; 16],
            vec![0.4_f32; 16],
            vec![],
            vec![],
            vec![],
            vec![],
        )
        .unwrap();
        let input: Vec<f32> = (0..3 * 4).map(|i| i as f32 * 0.1).collect();
        let grad_out: Vec<f32> = (0..3 * 4).map(|i| i as f32 * 0.05).collect();
        let (_, grads) = mha
            .backward_self_attention(&input, &grad_out, 1, 3, false)
            .unwrap();
        // bias 勾配は全て 0
        for &v in &grads.b_q {
            assert!(v.abs() < 1e-9);
        }
        for &v in &grads.b_o {
            assert!(v.abs() < 1e-9);
        }
    }
}
