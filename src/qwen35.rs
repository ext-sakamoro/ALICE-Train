//! Qwen3.5 アーキテクチャ定義 — Gated DeltaNet ハイブリッド QAT 学習用。
//!
//! Qwen3.5 は 3:1 ハイブリッド構造:
//! - 3/4 の層: Gated DeltaNet（線形再帰 + Delta Rule）
//! - 1/4 の層: Full Attention（QK-norm 付き GQA + partial RoPE）
//!
//! 9B モデルで 120B 超えのベンチマークを達成。

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// 層の種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Gated DeltaNet 線形再帰層。
    LinearAttention,
    /// Softmax GQA + QK-norm + partial RoPE。
    FullAttention,
}

/// Qwen3.5 モデル構成。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen35Config {
    /// 語彙サイズ。
    pub vocab_size: usize,
    /// 隠れ層次元。
    pub hidden_size: usize,
    /// FFN 中間次元 (SwiGLU)。
    pub intermediate_size: usize,
    /// Transformer 総レイヤー数。
    pub num_hidden_layers: usize,
    /// RMSNorm ε。
    pub rms_norm_eps: f32,

    // ── Full Attention 設定 ──
    /// Full Attention ヘッド数。
    pub num_attention_heads: usize,
    /// Full Attention KV ヘッド数 (GQA)。
    pub num_key_value_heads: usize,
    /// Full Attention ヘッド次元。
    pub head_dim: usize,
    /// RoPE θ。
    pub rope_theta: f32,
    /// RoPE を適用するヘッド次元の割合 (0.25 = 25%)。
    pub partial_rotary_factor: f32,

    // ── Linear Attention (DeltaNet) 設定 ──
    /// DeltaNet Key ヘッド次元。
    pub linear_key_head_dim: usize,
    /// DeltaNet Key ヘッド数。
    pub linear_num_key_heads: usize,
    /// DeltaNet Value ヘッド次元。
    pub linear_value_head_dim: usize,
    /// DeltaNet Value ヘッド数。
    pub linear_num_value_heads: usize,
    /// Causal Conv1d カーネルサイズ。
    pub linear_conv_kernel_dim: usize,

    /// Full Attention を挿入する間隔 (4 = 毎4層目)。
    pub full_attention_interval: usize,

    /// 各層の種別リスト (省略時は `full_attention_interval` から自動生成)。
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
}

impl Qwen35Config {
    /// Qwen3.5-9B のデフォルト設定。
    #[must_use]
    pub fn qwen35_9b() -> Self {
        let num_layers = 32;
        let interval = 4;
        let layer_types = (0..num_layers)
            .map(|i| {
                if (i + 1) % interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();

        Self {
            vocab_size: 248_320,
            hidden_size: 4096,
            intermediate_size: 12_288,
            num_hidden_layers: num_layers,
            rms_norm_eps: 1e-6,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            linear_key_head_dim: 128,
            linear_num_key_heads: 16,
            linear_value_head_dim: 128,
            linear_num_value_heads: 32,
            linear_conv_kernel_dim: 4,
            full_attention_interval: interval,
            layer_types,
        }
    }

    /// 層 i の種別を返す。
    #[must_use]
    pub fn layer_type(&self, i: usize) -> LayerType {
        if i < self.layer_types.len() {
            self.layer_types[i]
        } else if (i + 1).is_multiple_of(self.full_attention_interval) {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        }
    }

    /// DeltaNet の key 次元合計 (key_head_dim × num_key_heads)。
    #[must_use]
    pub fn linear_key_dim(&self) -> usize {
        self.linear_key_head_dim * self.linear_num_key_heads
    }

    /// DeltaNet の value 次元合計 (value_head_dim × num_value_heads)。
    #[must_use]
    pub fn linear_value_dim(&self) -> usize {
        self.linear_value_head_dim * self.linear_num_value_heads
    }

    /// Conv1d の入力チャネル数 (key_dim × 2 + value_dim)。
    #[must_use]
    pub fn conv_dim(&self) -> usize {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }

    /// RoPE を適用する次元数 (Full Attention)。
    #[must_use]
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    /// Full Attention の KV 次元。
    #[must_use]
    pub fn full_attn_kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// 1 DeltaNet レイヤーのパラメータ数。
    #[must_use]
    pub fn deltanet_params_per_layer(&self) -> usize {
        let key_dim = self.linear_key_dim();
        let val_dim = self.linear_value_dim();
        let conv_dim = self.conv_dim();
        let n_v_heads = self.linear_num_value_heads;

        let in_proj_qkv = self.hidden_size * (key_dim * 2 + val_dim);
        let in_proj_z = self.hidden_size * val_dim;
        let in_proj_b = self.hidden_size * n_v_heads;
        let in_proj_a = self.hidden_size * n_v_heads;
        let conv1d = conv_dim * self.linear_conv_kernel_dim;
        let a_log = n_v_heads;
        let dt_bias = n_v_heads;
        let norm = self.linear_value_head_dim;
        let out_proj = val_dim * self.hidden_size;
        // FFN
        let ffn = self.intermediate_size * self.hidden_size * 3;
        let norms = self.hidden_size * 2;

        in_proj_qkv
            + in_proj_z
            + in_proj_b
            + in_proj_a
            + conv1d
            + a_log
            + dt_bias
            + norm
            + out_proj
            + ffn
            + norms
    }

    /// 1 Full Attention レイヤーのパラメータ数。
    #[must_use]
    pub fn full_attn_params_per_layer(&self) -> usize {
        let attn_dim = self.num_attention_heads * self.head_dim;
        let kv_dim = self.full_attn_kv_dim();
        let q_proj = self.hidden_size * attn_dim;
        let k_proj = self.hidden_size * kv_dim;
        let v_proj = self.hidden_size * kv_dim;
        let o_proj = attn_dim * self.hidden_size;
        let q_norm = self.head_dim;
        let k_norm = self.head_dim;
        let ffn = self.intermediate_size * self.hidden_size * 3;
        let norms = self.hidden_size * 2;

        q_proj + k_proj + v_proj + o_proj + q_norm + k_norm + ffn + norms
    }

    /// 全パラメータ数（言語モデル部のみ、Vision エンコーダ除外）。
    #[must_use]
    pub fn total_params(&self) -> usize {
        let embedding = self.vocab_size * self.hidden_size;
        let output_norm = self.hidden_size;
        let lm_head = self.vocab_size * self.hidden_size;

        let mut layers = 0;
        for i in 0..self.num_hidden_layers {
            layers += match self.layer_type(i) {
                LayerType::LinearAttention => self.deltanet_params_per_layer(),
                LayerType::FullAttention => self.full_attn_params_per_layer(),
            };
        }

        embedding + layers + output_norm + lm_head
    }
}

// ── 重み構造体 ──────────────────────────────────────────────────────────────

/// FP32 テンソルを ternary fake quantize する (新規 Vec 確保版、テスト用)。
///
/// γ = mean(|W|), W_fq = round(W/γ).clamp(-1,1) × γ
fn fq_vec(w: &[f32]) -> Vec<f32> {
    if w.is_empty() {
        return Vec::new();
    }
    // Rayon並列: 大規模テンソル (256M+ elements) で 100-200倍高速化
    let sum_abs: f64 = w.par_iter().map(|&v| v.abs() as f64).sum();
    let gamma = (sum_abs / w.len() as f64) as f32;
    let inv_gamma = if gamma > 1e-10 { 1.0 / gamma } else { 0.0 };
    w.par_iter()
        .map(|&v| (v * inv_gamma).round().clamp(-1.0, 1.0) * gamma)
        .collect()
}

/// FP32 テンソルを ternary fake quantize する (in-place 版、ゼロアロケーション)。
///
/// `src` から読み、`dst` に上書き。`dst` は `src` と同じ長さでなければならない。
fn fq_vec_inplace(src: &[f32], dst: &mut [f32]) {
    if src.is_empty() {
        return;
    }
    debug_assert_eq!(src.len(), dst.len());
    // Rayon並列: 学習ループのホットパス、256コアで ~200倍高速化
    let sum_abs: f64 = src.par_iter().map(|&v| v.abs() as f64).sum();
    let gamma = (sum_abs / src.len() as f64) as f32;
    let inv_gamma = if gamma > 1e-10 { 1.0 / gamma } else { 0.0 };
    dst.par_iter_mut()
        .zip(src.par_iter())
        .for_each(|(d, &s)| {
            *d = (s * inv_gamma).round().clamp(-1.0, 1.0) * gamma;
        });
}

/// FP32 テンソルを in-place で fake quantize（非量子化重みはコピー）。
fn copy_slice(src: &[f32], dst: &mut [f32]) {
    dst.copy_from_slice(src);
}

/// Gated DeltaNet レイヤーの FP32 重み。
#[derive(Clone)]
pub struct DeltaNetLayerWeights {
    /// Input layernorm weight。
    pub input_layernorm: Vec<f32>,
    /// Post-attention layernorm weight。
    pub post_attn_layernorm: Vec<f32>,

    // ── DeltaNet projections ──
    /// QKV projection (hidden_size → key_dim*2 + value_dim)。
    pub in_proj_qkv: Vec<f32>,
    /// Output gate projection (hidden_size → value_dim)。
    pub in_proj_z: Vec<f32>,
    /// Write gate (beta) projection (hidden_size → num_v_heads)。
    pub in_proj_b: Vec<f32>,
    /// Decay gate projection (hidden_size → num_v_heads)。
    pub in_proj_a: Vec<f32>,
    /// Log decay base (num_v_heads)。
    pub a_log: Vec<f32>,
    /// Timestep bias (num_v_heads)。
    pub dt_bias: Vec<f32>,
    /// Depthwise causal conv1d weight (conv_dim × kernel_size)。
    pub conv1d_weight: Vec<f32>,
    /// Per-head output norm weight (value_head_dim)。
    pub norm_weight: Vec<f32>,
    /// Output projection (value_dim → hidden_size)。
    pub out_proj: Vec<f32>,

    // ── SwiGLU FFN ──
    /// Gate projection (intermediate_size × hidden_size)。
    pub gate_proj: Vec<f32>,
    /// Up projection (intermediate_size × hidden_size)。
    pub up_proj: Vec<f32>,
    /// Down projection (hidden_size × intermediate_size)。
    pub down_proj: Vec<f32>,
}

/// Full Attention レイヤーの FP32 重み (QK-norm 付き GQA)。
#[derive(Clone)]
pub struct FullAttnLayerWeights {
    /// Input layernorm weight。
    pub input_layernorm: Vec<f32>,
    /// Post-attention layernorm weight。
    pub post_attn_layernorm: Vec<f32>,

    // ── Attention projections ──
    /// Q projection (hidden_size → num_heads * head_dim)。
    pub q_proj: Vec<f32>,
    /// K projection (hidden_size → num_kv_heads * head_dim)。
    pub k_proj: Vec<f32>,
    /// V projection (hidden_size → num_kv_heads * head_dim)。
    pub v_proj: Vec<f32>,
    /// O projection (num_heads * head_dim → hidden_size)。
    pub o_proj: Vec<f32>,
    /// Q RMSNorm weight (head_dim)。
    pub q_norm: Vec<f32>,
    /// K RMSNorm weight (head_dim)。
    pub k_norm: Vec<f32>,

    // ── SwiGLU FFN ──
    /// Gate projection (intermediate_size × hidden_size)。
    pub gate_proj: Vec<f32>,
    /// Up projection (intermediate_size × hidden_size)。
    pub up_proj: Vec<f32>,
    /// Down projection (hidden_size × intermediate_size)。
    pub down_proj: Vec<f32>,
}

/// ハイブリッドレイヤーの重み (DeltaNet or Full Attention)。
#[derive(Clone)]
pub enum Qwen35LayerWeights {
    /// Gated DeltaNet 層。
    DeltaNet(DeltaNetLayerWeights),
    /// Full Attention 層。
    FullAttention(FullAttnLayerWeights),
}

impl Qwen35LayerWeights {
    /// 量子化対象の全 projection 重みへのミュータブル参照リスト。
    pub fn proj_weights_mut(&mut self) -> Vec<(&str, &mut [f32])> {
        match self {
            Self::DeltaNet(w) => vec![
                ("in_proj_qkv", &mut w.in_proj_qkv),
                ("in_proj_z", &mut w.in_proj_z),
                ("in_proj_b", &mut w.in_proj_b),
                ("in_proj_a", &mut w.in_proj_a),
                ("out_proj", &mut w.out_proj),
                ("gate_proj", &mut w.gate_proj),
                ("up_proj", &mut w.up_proj),
                ("down_proj", &mut w.down_proj),
            ],
            Self::FullAttention(w) => vec![
                ("q_proj", &mut w.q_proj),
                ("k_proj", &mut w.k_proj),
                ("v_proj", &mut w.v_proj),
                ("o_proj", &mut w.o_proj),
                ("gate_proj", &mut w.gate_proj),
                ("up_proj", &mut w.up_proj),
                ("down_proj", &mut w.down_proj),
            ],
        }
    }

    /// FakeQuantize — 全 projection 重みを ternary 疑似量子化したクローンを返す。
    ///
    /// γ = mean(|W|) → round(W/γ).clamp(-1,1) × γ。
    /// layernorm, a_log, dt_bias, conv1d, norm 等の非量子化重みはそのままコピー。
    #[must_use]
    pub fn fake_quantize(&self) -> Self {
        match self {
            Self::DeltaNet(w) => Self::DeltaNet(DeltaNetLayerWeights {
                input_layernorm: w.input_layernorm.clone(),
                post_attn_layernorm: w.post_attn_layernorm.clone(),
                in_proj_qkv: fq_vec(&w.in_proj_qkv),
                in_proj_z: fq_vec(&w.in_proj_z),
                in_proj_b: fq_vec(&w.in_proj_b),
                in_proj_a: fq_vec(&w.in_proj_a),
                a_log: w.a_log.clone(),
                dt_bias: w.dt_bias.clone(),
                conv1d_weight: w.conv1d_weight.clone(),
                norm_weight: w.norm_weight.clone(),
                out_proj: fq_vec(&w.out_proj),
                gate_proj: fq_vec(&w.gate_proj),
                up_proj: fq_vec(&w.up_proj),
                down_proj: fq_vec(&w.down_proj),
            }),
            Self::FullAttention(w) => Self::FullAttention(FullAttnLayerWeights {
                input_layernorm: w.input_layernorm.clone(),
                post_attn_layernorm: w.post_attn_layernorm.clone(),
                q_proj: fq_vec(&w.q_proj),
                k_proj: fq_vec(&w.k_proj),
                v_proj: fq_vec(&w.v_proj),
                o_proj: fq_vec(&w.o_proj),
                q_norm: w.q_norm.clone(),
                k_norm: w.k_norm.clone(),
                gate_proj: fq_vec(&w.gate_proj),
                up_proj: fq_vec(&w.up_proj),
                down_proj: fq_vec(&w.down_proj),
            }),
        }
    }

    /// In-place FakeQuantize — 事前確保済みバッファに上書き（ゼロアロケーション）。
    ///
    /// `dst` は `self` と同じ構造・同じサイズで事前確保されていること。
    /// projection は ternary 化、非量子化重みはコピー。
    pub fn fake_quantize_into(&self, dst: &mut Self) {
        match (self, dst) {
            (Self::DeltaNet(src), Self::DeltaNet(d)) => {
                copy_slice(&src.input_layernorm, &mut d.input_layernorm);
                copy_slice(&src.post_attn_layernorm, &mut d.post_attn_layernorm);
                fq_vec_inplace(&src.in_proj_qkv, &mut d.in_proj_qkv);
                fq_vec_inplace(&src.in_proj_z, &mut d.in_proj_z);
                fq_vec_inplace(&src.in_proj_b, &mut d.in_proj_b);
                fq_vec_inplace(&src.in_proj_a, &mut d.in_proj_a);
                copy_slice(&src.a_log, &mut d.a_log);
                copy_slice(&src.dt_bias, &mut d.dt_bias);
                copy_slice(&src.conv1d_weight, &mut d.conv1d_weight);
                copy_slice(&src.norm_weight, &mut d.norm_weight);
                fq_vec_inplace(&src.out_proj, &mut d.out_proj);
                fq_vec_inplace(&src.gate_proj, &mut d.gate_proj);
                fq_vec_inplace(&src.up_proj, &mut d.up_proj);
                fq_vec_inplace(&src.down_proj, &mut d.down_proj);
            }
            (Self::FullAttention(src), Self::FullAttention(d)) => {
                copy_slice(&src.input_layernorm, &mut d.input_layernorm);
                copy_slice(&src.post_attn_layernorm, &mut d.post_attn_layernorm);
                fq_vec_inplace(&src.q_proj, &mut d.q_proj);
                fq_vec_inplace(&src.k_proj, &mut d.k_proj);
                fq_vec_inplace(&src.v_proj, &mut d.v_proj);
                fq_vec_inplace(&src.o_proj, &mut d.o_proj);
                copy_slice(&src.q_norm, &mut d.q_norm);
                copy_slice(&src.k_norm, &mut d.k_norm);
                fq_vec_inplace(&src.gate_proj, &mut d.gate_proj);
                fq_vec_inplace(&src.up_proj, &mut d.up_proj);
                fq_vec_inplace(&src.down_proj, &mut d.down_proj);
            }
            _ => {}
        }
    }

    /// 全 projection 重みのパラメータ数合計。
    #[must_use]
    pub fn proj_param_count(&self) -> usize {
        match self {
            Self::DeltaNet(w) => {
                w.in_proj_qkv.len()
                    + w.in_proj_z.len()
                    + w.in_proj_b.len()
                    + w.in_proj_a.len()
                    + w.out_proj.len()
                    + w.gate_proj.len()
                    + w.up_proj.len()
                    + w.down_proj.len()
            }
            Self::FullAttention(w) => {
                w.q_proj.len()
                    + w.k_proj.len()
                    + w.v_proj.len()
                    + w.o_proj.len()
                    + w.gate_proj.len()
                    + w.up_proj.len()
                    + w.down_proj.len()
            }
        }
    }
}

impl DeltaNetLayerWeights {
    /// safetensors からレイヤー重みを構築。
    ///
    /// `prefix` は `model.language_model.layers.{i}` 形式。
    pub fn from_tensors(
        prefix: &str,
        get_tensor: &dyn Fn(&str) -> Option<Vec<f32>>,
    ) -> Option<Self> {
        Some(Self {
            input_layernorm: get_tensor(&format!("{prefix}.input_layernorm.weight"))?,
            post_attn_layernorm: get_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            in_proj_qkv: get_tensor(&format!("{prefix}.linear_attn.in_proj_qkv.weight"))?,
            in_proj_z: get_tensor(&format!("{prefix}.linear_attn.in_proj_z.weight"))?,
            in_proj_b: get_tensor(&format!("{prefix}.linear_attn.in_proj_b.weight"))?,
            in_proj_a: get_tensor(&format!("{prefix}.linear_attn.in_proj_a.weight"))?,
            a_log: get_tensor(&format!("{prefix}.linear_attn.A_log"))?,
            dt_bias: get_tensor(&format!("{prefix}.linear_attn.dt_bias"))?,
            conv1d_weight: get_tensor(&format!("{prefix}.linear_attn.conv1d.weight"))?,
            norm_weight: get_tensor(&format!("{prefix}.linear_attn.norm.weight"))?,
            out_proj: get_tensor(&format!("{prefix}.linear_attn.out_proj.weight"))?,
            gate_proj: get_tensor(&format!("{prefix}.mlp.gate_proj.weight"))?,
            up_proj: get_tensor(&format!("{prefix}.mlp.up_proj.weight"))?,
            down_proj: get_tensor(&format!("{prefix}.mlp.down_proj.weight"))?,
        })
    }
}

impl FullAttnLayerWeights {
    /// safetensors からレイヤー重みを構築。
    pub fn from_tensors(
        prefix: &str,
        get_tensor: &dyn Fn(&str) -> Option<Vec<f32>>,
    ) -> Option<Self> {
        Some(Self {
            input_layernorm: get_tensor(&format!("{prefix}.input_layernorm.weight"))?,
            post_attn_layernorm: get_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            q_proj: get_tensor(&format!("{prefix}.self_attn.q_proj.weight"))?,
            k_proj: get_tensor(&format!("{prefix}.self_attn.k_proj.weight"))?,
            v_proj: get_tensor(&format!("{prefix}.self_attn.v_proj.weight"))?,
            o_proj: get_tensor(&format!("{prefix}.self_attn.o_proj.weight"))?,
            q_norm: get_tensor(&format!("{prefix}.self_attn.q_norm.weight"))?,
            k_norm: get_tensor(&format!("{prefix}.self_attn.k_norm.weight"))?,
            gate_proj: get_tensor(&format!("{prefix}.mlp.gate_proj.weight"))?,
            up_proj: get_tensor(&format!("{prefix}.mlp.up_proj.weight"))?,
            down_proj: get_tensor(&format!("{prefix}.mlp.down_proj.weight"))?,
        })
    }
}

/// QAT 学習設定 (Qwen3.5 用)。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen35QatConfig {
    /// モデル構成。
    pub model: Qwen35Config,
    /// 学習率。
    pub learning_rate: f32,
    /// 最小学習率 (cosine decay)。
    pub min_lr: f32,
    /// ウォームアップステップ数。
    pub warmup_steps: usize,
    /// 総学習ステップ数。
    pub total_steps: usize,
    /// 勾配累積ステップ数。
    pub gradient_accumulation_steps: usize,
    /// 評価間隔 (ステップ)。
    pub eval_interval: usize,
    /// チェックポイント保存間隔 (ステップ)。
    pub checkpoint_interval: usize,
    /// チェックポイント保存ディレクトリ。
    pub checkpoint_dir: String,
    /// 学習データパス。
    pub train_data_path: String,
    /// 評価データパス。
    pub eval_data_path: Option<String>,
    /// safetensors モデルパス。
    pub model_path: String,
    /// バッチサイズ。
    pub batch_size: usize,
    /// シーケンス長。
    pub seq_len: usize,
    /// AdamW weight decay。
    pub weight_decay: f32,
    /// 勾配クリッピング。
    pub max_grad_norm: f32,
    /// BF16 混合精度。
    pub use_bf16: bool,
    /// レジュームチェックポイントパス。
    pub resume_from: Option<String>,
    /// 全レイヤー RAM プリロード。
    #[serde(default = "default_preload")]
    pub preload_all_layers: bool,
    /// delta を BF16 圧縮。
    #[serde(default)]
    pub bf16_delta: bool,
    /// safetensors 重み名プレフィックス ("model.language_model" or "model")。
    #[serde(default = "default_weight_prefix")]
    pub weight_prefix: String,
}

fn default_preload() -> bool {
    true
}

fn default_weight_prefix() -> String {
    "model.language_model".to_string()
}

impl Default for Qwen35QatConfig {
    fn default() -> Self {
        Self {
            model: Qwen35Config::qwen35_9b(),
            learning_rate: 1e-4,
            min_lr: 1e-6,
            warmup_steps: 200,
            total_steps: 5000,
            gradient_accumulation_steps: 4,
            eval_interval: 100,
            checkpoint_interval: 500,
            checkpoint_dir: "checkpoints/qwen35_9b".to_string(),
            train_data_path: "data/qwen35/train.bin".to_string(),
            eval_data_path: Some("data/qwen35/eval.bin".to_string()),
            model_path: "models/Qwen--Qwen3.5-9B".to_string(),
            batch_size: 1,
            seq_len: 2048,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            use_bf16: true,
            resume_from: None,
            preload_all_layers: true,
            bf16_delta: false,
            weight_prefix: "model.language_model".to_string(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_9b_config() {
        let c = Qwen35Config::qwen35_9b();
        assert_eq!(c.num_hidden_layers, 32);
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.vocab_size, 248_320);
    }

    #[test]
    fn layer_type_pattern() {
        let c = Qwen35Config::qwen35_9b();
        // 3:1 パターン: [lin, lin, lin, full] × 8
        for i in 0..32 {
            let expected = if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            };
            assert_eq!(c.layer_type(i), expected, "layer {i}");
        }
    }

    #[test]
    fn deltanet_layer_count() {
        let c = Qwen35Config::qwen35_9b();
        let dn = (0..32)
            .filter(|&i| c.layer_type(i) == LayerType::LinearAttention)
            .count();
        let fa = (0..32)
            .filter(|&i| c.layer_type(i) == LayerType::FullAttention)
            .count();
        assert_eq!(dn, 24);
        assert_eq!(fa, 8);
    }

    #[test]
    fn linear_dims() {
        let c = Qwen35Config::qwen35_9b();
        assert_eq!(c.linear_key_dim(), 2048); // 128 × 16
        assert_eq!(c.linear_value_dim(), 4096); // 128 × 32
        assert_eq!(c.conv_dim(), 8192); // 2048*2 + 4096
    }

    #[test]
    fn rotary_dim() {
        let c = Qwen35Config::qwen35_9b();
        assert_eq!(c.rotary_dim(), 64); // 256 * 0.25
    }

    #[test]
    fn total_params_9b() {
        let c = Qwen35Config::qwen35_9b();
        let total = c.total_params();
        // 9B 前後であることを検証
        assert!(total > 7_000_000_000, "should have >7B params, got {total}");
        assert!(
            total < 12_000_000_000,
            "should have <12B params, got {total}"
        );
    }

    #[test]
    fn qat_config_serialization() {
        let cfg = Qwen35QatConfig::default();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let parsed: Qwen35QatConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model.hidden_size, 4096);
        assert_eq!(parsed.total_steps, 5000);
    }

    #[test]
    fn proj_weights_deltanet() {
        let c = Qwen35Config::qwen35_9b();
        let w = DeltaNetLayerWeights {
            input_layernorm: vec![0.0; c.hidden_size],
            post_attn_layernorm: vec![0.0; c.hidden_size],
            in_proj_qkv: vec![0.0; c.hidden_size * (c.linear_key_dim() * 2 + c.linear_value_dim())],
            in_proj_z: vec![0.0; c.hidden_size * c.linear_value_dim()],
            in_proj_b: vec![0.0; c.hidden_size * c.linear_num_value_heads],
            in_proj_a: vec![0.0; c.hidden_size * c.linear_num_value_heads],
            a_log: vec![0.0; c.linear_num_value_heads],
            dt_bias: vec![1.0; c.linear_num_value_heads],
            conv1d_weight: vec![0.0; c.conv_dim() * c.linear_conv_kernel_dim],
            norm_weight: vec![1.0; c.linear_value_head_dim],
            out_proj: vec![0.0; c.linear_value_dim() * c.hidden_size],
            gate_proj: vec![0.0; c.intermediate_size * c.hidden_size],
            up_proj: vec![0.0; c.intermediate_size * c.hidden_size],
            down_proj: vec![0.0; c.hidden_size * c.intermediate_size],
        };
        let mut layer = Qwen35LayerWeights::DeltaNet(w);
        assert_eq!(layer.proj_weights_mut().len(), 8);
    }

    // ── fq_vec テスト ──

    #[test]
    fn fq_vec_empty() {
        assert!(fq_vec(&[]).is_empty());
    }

    #[test]
    fn fq_vec_ternary_values_only() {
        let w = vec![0.5, -0.3, 0.01, 0.8, -0.9, 0.0, -0.05, 0.7];
        let fq = fq_vec(&w);
        assert_eq!(fq.len(), w.len());
        let gamma: f64 = w.iter().map(|v| v.abs() as f64).sum::<f64>() / w.len() as f64;
        let g = gamma as f32;
        for &v in &fq {
            let norm = (v / g).round();
            assert!(
                (norm - -1.0).abs() < 0.01
                    || (norm - 0.0).abs() < 0.01
                    || (norm - 1.0).abs() < 0.01,
                "fq value {v} not in {{-γ, 0, +γ}}, γ={g}"
            );
        }
    }

    #[test]
    fn fq_vec_preserves_sign() {
        let w = vec![1.0, -1.0, 0.5, -0.5];
        let fq = fq_vec(&w);
        assert!(fq[0] > 0.0);
        assert!(fq[1] < 0.0);
    }

    #[test]
    fn fq_vec_all_zeros() {
        let w = vec![0.0; 10];
        let fq = fq_vec(&w);
        assert!(fq.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fq_vec_scale_is_mean_abs() {
        let w = vec![1.0, -1.0, 0.5, -0.5];
        let fq = fq_vec(&w);
        // gamma = mean(|W|) = 0.75
        // round(1.0/0.75) = round(1.33) = 1, clamp → 1 → 1*0.75 = 0.75
        assert!((fq[0] - 0.75).abs() < 1e-5);
        assert!((fq[1] - (-0.75)).abs() < 1e-5);
        // round(0.5/0.75) = round(0.67) = 1 → 0.75
        assert!((fq[2] - 0.75).abs() < 1e-5);
    }

    // ── fake_quantize テスト ──

    #[test]
    fn fake_quantize_deltanet_preserves_norms() {
        let c = Qwen35Config::qwen35_9b();
        let w = DeltaNetLayerWeights {
            input_layernorm: vec![1.5; c.hidden_size],
            post_attn_layernorm: vec![2.0; c.hidden_size],
            in_proj_qkv: vec![0.1; c.hidden_size * (c.linear_key_dim() * 2 + c.linear_value_dim())],
            in_proj_z: vec![0.1; c.hidden_size * c.linear_value_dim()],
            in_proj_b: vec![0.1; c.hidden_size * c.linear_num_value_heads],
            in_proj_a: vec![0.1; c.hidden_size * c.linear_num_value_heads],
            a_log: vec![0.5; c.linear_num_value_heads],
            dt_bias: vec![0.3; c.linear_num_value_heads],
            conv1d_weight: vec![0.2; c.conv_dim() * c.linear_conv_kernel_dim],
            norm_weight: vec![1.0; c.linear_value_head_dim],
            out_proj: vec![0.1; c.linear_value_dim() * c.hidden_size],
            gate_proj: vec![0.1; c.intermediate_size * c.hidden_size],
            up_proj: vec![0.1; c.intermediate_size * c.hidden_size],
            down_proj: vec![0.1; c.hidden_size * c.intermediate_size],
        };
        let layer = Qwen35LayerWeights::DeltaNet(w);
        let fq = layer.fake_quantize();
        match &fq {
            Qwen35LayerWeights::DeltaNet(wf) => {
                // layernorm, a_log, dt_bias, conv1d, norm はそのまま
                assert_eq!(wf.input_layernorm[0], 1.5);
                assert_eq!(wf.post_attn_layernorm[0], 2.0);
                assert_eq!(wf.a_log[0], 0.5);
                assert_eq!(wf.dt_bias[0], 0.3);
                assert_eq!(wf.norm_weight[0], 1.0);
                // projection は量子化される（0.1 は gamma=0.1 で round(1.0)=1 → 0.1）
                assert!((wf.in_proj_qkv[0] - 0.1).abs() < 1e-5);
            }
            _ => panic!("expected DeltaNet"),
        }
    }

    #[test]
    fn fake_quantize_fullattn_preserves_norms() {
        let c = Qwen35Config::qwen35_9b();
        let w = FullAttnLayerWeights {
            input_layernorm: vec![1.5; c.hidden_size],
            post_attn_layernorm: vec![2.0; c.hidden_size],
            q_proj: vec![0.1; c.hidden_size * c.num_attention_heads * c.head_dim],
            k_proj: vec![0.1; c.hidden_size * c.num_key_value_heads * c.head_dim],
            v_proj: vec![0.1; c.hidden_size * c.num_key_value_heads * c.head_dim],
            o_proj: vec![0.1; c.num_attention_heads * c.head_dim * c.hidden_size],
            q_norm: vec![1.0; c.head_dim],
            k_norm: vec![1.0; c.head_dim],
            gate_proj: vec![0.1; c.intermediate_size * c.hidden_size],
            up_proj: vec![0.1; c.intermediate_size * c.hidden_size],
            down_proj: vec![0.1; c.hidden_size * c.intermediate_size],
        };
        let layer = Qwen35LayerWeights::FullAttention(w);
        let fq = layer.fake_quantize();
        match &fq {
            Qwen35LayerWeights::FullAttention(wf) => {
                assert_eq!(wf.input_layernorm[0], 1.5);
                assert_eq!(wf.q_norm[0], 1.0);
                assert_eq!(wf.k_norm[0], 1.0);
            }
            _ => panic!("expected FullAttention"),
        }
    }

    #[test]
    fn fake_quantize_mixed_weights() {
        let w = vec![0.5, -0.3, 0.01, 0.8, -0.9];
        let fq = fq_vec(&w);
        // gamma = (0.5+0.3+0.01+0.8+0.9)/5 = 0.502
        let gamma = 0.502;
        // 0.5/0.502 = 0.996 → round=1 → 0.502
        assert!((fq[0] - gamma).abs() < 0.01);
        // -0.3/0.502 = -0.598 → round=-1 → -0.502
        assert!((fq[1] - (-gamma)).abs() < 0.01);
        // 0.01/0.502 = 0.02 → round=0 → 0
        assert!(fq[2].abs() < 0.01);
    }

    // ── fq_vec_inplace テスト ──

    #[test]
    fn fq_vec_inplace_matches_fq_vec() {
        let w = vec![0.5, -0.3, 0.01, 0.8, -0.9, 0.0, -0.05, 0.7];
        let fq_alloc = fq_vec(&w);
        let mut fq_inplace = vec![0.0; w.len()];
        fq_vec_inplace(&w, &mut fq_inplace);
        for (a, b) in fq_alloc.iter().zip(fq_inplace.iter()) {
            assert!((a - b).abs() < 1e-10, "mismatch: alloc={a} inplace={b}");
        }
    }

    #[test]
    fn fq_vec_inplace_empty() {
        let mut dst = vec![];
        fq_vec_inplace(&[], &mut dst);
    }

    // ── fake_quantize_into テスト ──

    #[test]
    fn fake_quantize_into_matches_fake_quantize() {
        let c = Qwen35Config::qwen35_9b();
        let w = DeltaNetLayerWeights {
            input_layernorm: vec![1.5; c.hidden_size],
            post_attn_layernorm: vec![2.0; c.hidden_size],
            in_proj_qkv: vec![0.1; c.hidden_size * (c.linear_key_dim() * 2 + c.linear_value_dim())],
            in_proj_z: vec![0.1; c.hidden_size * c.linear_value_dim()],
            in_proj_b: vec![0.1; c.hidden_size * c.linear_num_value_heads],
            in_proj_a: vec![0.1; c.hidden_size * c.linear_num_value_heads],
            a_log: vec![0.5; c.linear_num_value_heads],
            dt_bias: vec![0.3; c.linear_num_value_heads],
            conv1d_weight: vec![0.2; c.conv_dim() * c.linear_conv_kernel_dim],
            norm_weight: vec![1.0; c.linear_value_head_dim],
            out_proj: vec![0.1; c.linear_value_dim() * c.hidden_size],
            gate_proj: vec![0.1; c.intermediate_size * c.hidden_size],
            up_proj: vec![0.1; c.intermediate_size * c.hidden_size],
            down_proj: vec![0.1; c.hidden_size * c.intermediate_size],
        };
        let layer = Qwen35LayerWeights::DeltaNet(w);
        let fq_alloc = layer.fake_quantize();
        let mut fq_inplace = layer.clone();
        layer.fake_quantize_into(&mut fq_inplace);
        // projection の値が一致
        match (&fq_alloc, &fq_inplace) {
            (Qwen35LayerWeights::DeltaNet(a), Qwen35LayerWeights::DeltaNet(b)) => {
                assert_eq!(a.in_proj_qkv.len(), b.in_proj_qkv.len());
                for (x, y) in a.in_proj_qkv.iter().zip(b.in_proj_qkv.iter()) {
                    assert!((x - y).abs() < 1e-10);
                }
                assert_eq!(a.input_layernorm, b.input_layernorm);
            }
            _ => panic!("variant mismatch"),
        }
    }
}
