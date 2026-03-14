//! Llama-3 アーキテクチャ定義 — QAT 学習用。
//!
//! ALICE-ML の `llama3_ternary` は推論専用。
//! 本モジュールは学習時に必要な「潜在 FP32 重み」と「レイヤー単位 forward/backward」を提供する。
//!
//! # 設計
//!
//! - 重みは safetensors から FP32 で読み込み、CPU RAM に保持
//! - レイヤー単位で forward → backward → update を逐次実行（メモリ節約）
//! - `FakeQuantize` で ternary に量子化した forward を実行
//! - `OffloadOptimizer` で AdamW の m/v を CPU に保持

use serde::{Deserialize, Serialize};

/// Llama-3 モデル構成。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    /// 語彙サイズ
    pub vocab_size: usize,
    /// 隠れ層次元
    pub hidden_dim: usize,
    /// FFN 中間次元 (SwiGLU)
    pub intermediate_dim: usize,
    /// アテンションヘッド数
    pub num_heads: usize,
    /// KV ヘッド数 (GQA)
    pub num_kv_heads: usize,
    /// Transformer レイヤー数
    pub num_layers: usize,
    /// 最大シーケンス長
    pub max_seq_len: usize,
    /// ヘッド次元
    pub head_dim: usize,
    /// RoPE θ
    pub rope_theta: f32,
    /// RMSNorm ε
    pub norm_eps: f32,
    /// Attention projection にバイアスがあるか (Qwen2.5: true, Llama: false)
    #[serde(default)]
    pub attention_bias: bool,
}

impl LlamaConfig {
    /// Llama-3 8B のデフォルト設定。
    #[must_use]
    pub fn llama3_8b() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_dim: 4096,
            intermediate_dim: 14_336,
            num_heads: 32,
            num_kv_heads: 8,
            num_layers: 32,
            max_seq_len: 8192,
            head_dim: 128,
            rope_theta: 500_000.0,
            norm_eps: 1e-5,
            attention_bias: false,
        }
    }

    /// Llama-3.2 1B のデフォルト設定。
    #[must_use]
    pub fn llama3_1b() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_dim: 2048,
            intermediate_dim: 8192,
            num_heads: 32,
            num_kv_heads: 8,
            num_layers: 16,
            max_seq_len: 131_072,
            head_dim: 64,
            rope_theta: 500_000.0,
            norm_eps: 1e-5,
            attention_bias: false,
        }
    }

    /// Llama-3.2 3B のデフォルト設定。
    #[must_use]
    pub fn llama3_3b() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_dim: 3072,
            intermediate_dim: 8192,
            num_heads: 24,
            num_kv_heads: 8,
            num_layers: 28,
            max_seq_len: 131_072,
            head_dim: 128,
            rope_theta: 500_000.0,
            norm_eps: 1e-5,
            attention_bias: false,
        }
    }

    /// Llama-3 70B のデフォルト設定。
    #[must_use]
    pub fn llama3_70b() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_dim: 8192,
            intermediate_dim: 28_672,
            num_heads: 64,
            num_kv_heads: 8,
            num_layers: 80,
            max_seq_len: 8192,
            head_dim: 128,
            rope_theta: 500_000.0,
            norm_eps: 1e-5,
            attention_bias: false,
        }
    }

    /// Qwen2.5-7B のデフォルト設定。
    #[must_use]
    pub fn qwen25_7b() -> Self {
        Self {
            vocab_size: 152_064,
            hidden_dim: 3584,
            intermediate_dim: 18_944,
            num_heads: 28,
            num_kv_heads: 4,
            num_layers: 28,
            max_seq_len: 131_072,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-6,
            attention_bias: true,
        }
    }

    /// 1レイヤーのパラメータ数。
    #[must_use]
    pub fn params_per_layer(&self) -> usize {
        let kv_dim = self.num_kv_heads * self.head_dim;
        // q_proj + o_proj
        let attn = self.hidden_dim * self.hidden_dim * 2;
        // k_proj + v_proj
        let kv = kv_dim * self.hidden_dim * 2;
        // gate + up + down (SwiGLU)
        let ffn = self.intermediate_dim * self.hidden_dim * 3;
        // norm weights (FP32, not quantized)
        let norms = self.hidden_dim * 2;
        attn + kv + ffn + norms
    }

    /// 全パラメータ数（embedding + layers + output）。
    #[must_use]
    pub fn total_params(&self) -> usize {
        let embedding = self.vocab_size * self.hidden_dim;
        let layers = self.params_per_layer() * self.num_layers;
        let output_norm = self.hidden_dim;
        let output_proj = self.vocab_size * self.hidden_dim;
        embedding + layers + output_norm + output_proj
    }

    /// Ternary 化後の推定メモリ (bytes)。
    #[must_use]
    pub fn ternary_memory_bytes(&self) -> usize {
        // 2 bits per weight → 4 weights per byte
        // embedding と norm は FP32 のまま
        let emb_bytes = self.vocab_size * self.hidden_dim * 4;
        let norm_bytes = self.hidden_dim * (self.num_layers * 2 + 1) * 4;
        let ternary_params = self.total_params()
            - self.vocab_size * self.hidden_dim
            - self.hidden_dim * (self.num_layers * 2 + 1);
        let ternary_bytes = ternary_params / 4;
        emb_bytes + norm_bytes + ternary_bytes
    }
}

/// Llama レイヤーの潜在 FP32 重み（QAT 学習用）。
///
/// 学習中はこの FP32 重みを保持し、forward 時に `FakeQuantize` で ternary 化する。
/// backward 後は STE で FP32 重みに勾配を反映。
pub struct LlamaLayerWeights {
    /// 入力 RMSNorm weight (FP32, 量子化しない)
    pub attn_norm: Vec<f32>,
    /// Q projection (hidden_dim × hidden_dim)
    pub q_proj: Vec<f32>,
    /// K projection (kv_dim × hidden_dim)
    pub k_proj: Vec<f32>,
    /// V projection (kv_dim × hidden_dim)
    pub v_proj: Vec<f32>,
    /// O projection (hidden_dim × hidden_dim)
    pub o_proj: Vec<f32>,
    /// Q bias (Qwen2.5: Some, Llama: None) — FP32, 量子化しない
    pub q_bias: Option<Vec<f32>>,
    /// K bias (Qwen2.5: Some, Llama: None) — FP32, 量子化しない
    pub k_bias: Option<Vec<f32>>,
    /// V bias (Qwen2.5: Some, Llama: None) — FP32, 量子化しない
    pub v_bias: Option<Vec<f32>>,
    /// FFN RMSNorm weight (FP32, 量子化しない)
    pub ffn_norm: Vec<f32>,
    /// Gate projection (intermediate_dim × hidden_dim)
    pub gate_proj: Vec<f32>,
    /// Up projection (intermediate_dim × hidden_dim)
    pub up_proj: Vec<f32>,
    /// Down projection (hidden_dim × intermediate_dim)
    pub down_proj: Vec<f32>,
}

impl LlamaLayerWeights {
    /// safetensors テンソルデータからレイヤー重みを構築。
    ///
    /// `tensors` は `(name, f32_data)` のイテレータ。
    /// 重み名は HuggingFace 形式: `model.layers.{i}.self_attn.q_proj.weight` 等。
    pub fn from_tensors(
        layer_idx: usize,
        get_tensor: &dyn Fn(&str) -> Option<Vec<f32>>,
        config: &LlamaConfig,
    ) -> Option<Self> {
        let prefix = format!("model.layers.{layer_idx}");
        // kv_dim は将来の重みバリデーションで使用予定
        let _ = config.num_kv_heads * config.head_dim;

        let q_bias = if config.attention_bias {
            Some(get_tensor(&format!("{prefix}.self_attn.q_proj.bias"))?)
        } else {
            None
        };
        let k_bias = if config.attention_bias {
            Some(get_tensor(&format!("{prefix}.self_attn.k_proj.bias"))?)
        } else {
            None
        };
        let v_bias = if config.attention_bias {
            Some(get_tensor(&format!("{prefix}.self_attn.v_proj.bias"))?)
        } else {
            None
        };

        Some(Self {
            attn_norm: get_tensor(&format!("{prefix}.input_layernorm.weight"))?,
            q_proj: get_tensor(&format!("{prefix}.self_attn.q_proj.weight"))?,
            k_proj: get_tensor(&format!("{prefix}.self_attn.k_proj.weight"))?,
            v_proj: get_tensor(&format!("{prefix}.self_attn.v_proj.weight"))?,
            o_proj: get_tensor(&format!("{prefix}.self_attn.o_proj.weight"))?,
            q_bias,
            k_bias,
            v_bias,
            ffn_norm: get_tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            gate_proj: get_tensor(&format!("{prefix}.mlp.gate_proj.weight"))?,
            up_proj: get_tensor(&format!("{prefix}.mlp.up_proj.weight"))?,
            down_proj: get_tensor(&format!("{prefix}.mlp.down_proj.weight"))?,
        })
    }

    /// 量子化対象の全 projection 重みへのミュータブル参照リスト。
    /// (名前, &mut [f32]) のペア。norm weight は含まない。
    pub fn proj_weights_mut(&mut self) -> Vec<(&str, &mut [f32])> {
        vec![
            ("q_proj", &mut self.q_proj),
            ("k_proj", &mut self.k_proj),
            ("v_proj", &mut self.v_proj),
            ("o_proj", &mut self.o_proj),
            ("gate_proj", &mut self.gate_proj),
            ("up_proj", &mut self.up_proj),
            ("down_proj", &mut self.down_proj),
        ]
    }

    /// 全 projection 重みのパラメータ数合計。
    #[must_use]
    pub fn proj_param_count(&self) -> usize {
        self.q_proj.len()
            + self.k_proj.len()
            + self.v_proj.len()
            + self.o_proj.len()
            + self.gate_proj.len()
            + self.up_proj.len()
            + self.down_proj.len()
    }

    /// Attention bias への参照（Q, K, V 順）。bias がない場合は空。
    pub fn attention_biases(&self) -> Vec<(&str, &[f32])> {
        let mut out = Vec::new();
        if let Some(ref b) = self.q_bias {
            out.push(("q_bias", b.as_slice()));
        }
        if let Some(ref b) = self.k_bias {
            out.push(("k_bias", b.as_slice()));
        }
        if let Some(ref b) = self.v_bias {
            out.push(("v_bias", b.as_slice()));
        }
        out
    }

    /// Attention bias へのミュータブル参照（Q, K, V 順）。
    pub fn attention_biases_mut(&mut self) -> Vec<(&str, &mut [f32])> {
        let mut out = Vec::new();
        if let Some(ref mut b) = self.q_bias {
            out.push(("q_bias", b.as_mut_slice()));
        }
        if let Some(ref mut b) = self.k_bias {
            out.push(("k_bias", b.as_mut_slice()));
        }
        if let Some(ref mut b) = self.v_bias {
            out.push(("v_bias", b.as_mut_slice()));
        }
        out
    }

    /// Attention bias が存在するか。
    #[must_use]
    pub fn has_attention_bias(&self) -> bool {
        self.q_bias.is_some()
    }
}

/// QAT 学習設定。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QatTrainConfig {
    /// モデル構成
    pub model: LlamaConfig,
    /// 学習率
    pub learning_rate: f32,
    /// 最小学習率 (cosine decay)
    pub min_lr: f32,
    /// ウォームアップステップ数
    pub warmup_steps: usize,
    /// 総学習ステップ数
    pub total_steps: usize,
    /// 勾配累積ステップ数
    pub gradient_accumulation_steps: usize,
    /// 評価間隔 (ステップ)
    pub eval_interval: usize,
    /// チェックポイント保存間隔 (ステップ)
    pub checkpoint_interval: usize,
    /// チェックポイント保存ディレクトリ
    pub checkpoint_dir: String,
    /// 学習データパス (トークンファイル)
    pub train_data_path: String,
    /// 評価データパス (トークンファイル)
    pub eval_data_path: Option<String>,
    /// safetensors モデルパス (ディレクトリ)
    pub model_path: String,
    /// バッチサイズ (シーケンス数)
    pub batch_size: usize,
    /// シーケンス長
    pub seq_len: usize,
    /// AdamW weight decay
    pub weight_decay: f32,
    /// 勾配クリッピング
    pub max_grad_norm: f32,
    /// BF16 混合精度を使用するか
    pub use_bf16: bool,
    /// レジュームするチェックポイントパス (None = 最初から)
    pub resume_from: Option<String>,
}

impl Default for QatTrainConfig {
    fn default() -> Self {
        Self {
            model: LlamaConfig::llama3_70b(),
            learning_rate: 1e-4,
            min_lr: 1e-6,
            warmup_steps: 200,
            total_steps: 10_000,
            gradient_accumulation_steps: 8,
            eval_interval: 100,
            checkpoint_interval: 500,
            checkpoint_dir: "checkpoints".to_string(),
            train_data_path: "data/train.bin".to_string(),
            eval_data_path: Some("data/eval.bin".to_string()),
            model_path: "models/llama-3-70b".to_string(),
            batch_size: 1,
            seq_len: 2048,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            use_bf16: true,
            resume_from: None,
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
    fn llama3_8b_config() {
        let c = LlamaConfig::llama3_8b();
        assert_eq!(c.hidden_dim, 4096);
        assert_eq!(c.num_layers, 32);
        assert_eq!(c.head_dim, 128);
    }

    #[test]
    fn llama3_70b_config() {
        let c = LlamaConfig::llama3_70b();
        assert_eq!(c.hidden_dim, 8192);
        assert_eq!(c.num_layers, 80);
        assert_eq!(c.num_heads, 64);
        assert_eq!(c.num_kv_heads, 8);
    }

    #[test]
    fn llama3_70b_params() {
        let c = LlamaConfig::llama3_70b();
        let total = c.total_params();
        // 70B モデル: 約 70B パラメータ
        assert!(
            total > 60_000_000_000,
            "70B should have >60B params, got {total}"
        );
        assert!(
            total < 80_000_000_000,
            "70B should have <80B params, got {total}"
        );
    }

    #[test]
    fn llama3_8b_params() {
        let c = LlamaConfig::llama3_8b();
        let total = c.total_params();
        // 8B モデル: 約 8B パラメータ
        assert!(
            total > 7_000_000_000,
            "8B should have >7B params, got {total}"
        );
        assert!(
            total < 9_000_000_000,
            "8B should have <9B params, got {total}"
        );
    }

    #[test]
    fn llama3_70b_ternary_memory() {
        let c = LlamaConfig::llama3_70b();
        let bytes = c.ternary_memory_bytes();
        let gb = bytes as f64 / 1024.0 / 1024.0 / 1024.0;
        // ternary 70B ≈ 10-15 GB
        assert!(gb > 5.0, "ternary memory should be >5 GB, got {gb:.1} GB");
        assert!(gb < 25.0, "ternary memory should be <25 GB, got {gb:.1} GB");
    }

    #[test]
    fn params_per_layer_70b() {
        let c = LlamaConfig::llama3_70b();
        let per_layer = c.params_per_layer();
        // 70B / 80 layers ≈ 800M per layer (+ embedding/output)
        assert!(
            per_layer > 500_000_000,
            "per layer should be >500M, got {per_layer}"
        );
    }

    #[test]
    fn qat_train_config_default() {
        let cfg = QatTrainConfig::default();
        assert_eq!(cfg.model.num_layers, 80);
        assert!((cfg.learning_rate - 1e-4).abs() < 1e-8);
        assert_eq!(cfg.gradient_accumulation_steps, 8);
    }

    #[test]
    fn qat_train_config_serialization() {
        let cfg = QatTrainConfig::default();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let parsed: QatTrainConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model.hidden_dim, cfg.model.hidden_dim);
        assert_eq!(parsed.total_steps, cfg.total_steps);
    }

    #[test]
    fn layer_weights_proj_count() {
        // 8B のレイヤー重みサイズを検証
        let c = LlamaConfig::llama3_8b();
        let kv_dim = c.num_kv_heads * c.head_dim;
        let expected = c.hidden_dim * c.hidden_dim * 2  // q + o
            + kv_dim * c.hidden_dim * 2                  // k + v
            + c.intermediate_dim * c.hidden_dim * 3; // gate + up + down
                                                     // params_per_layer は norm weight も含むので、proj のみと比較
        assert!(c.params_per_layer() > expected);
    }
}
