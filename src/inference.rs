//! .alice モデル推論エンジン — ternary QAT モデルのテキスト生成。
//!
//! `.alice` ファイルからモデルをロードし、autoregressive にテキストを生成する。

use crate::export::{
    read_alice_meta, read_deltanet_layer, read_embedding, read_fullattn_layer, read_lm_head,
    read_output_norm, AliceModelMeta,
};
use crate::qwen35::{LayerType, Qwen35Config, Qwen35LayerWeights};
use crate::tokenizer::BpeTokenizer;
use std::io::{self, BufReader, Read};
use std::path::Path;

/// ロード済みモデル。
pub struct AliceModel {
    /// メタデータ。
    pub meta: AliceModelMeta,
    /// Embedding テーブル (FP32)。
    pub embedding: Vec<f32>,
    /// Output layernorm (FP32)。
    pub output_norm: Vec<f32>,
    /// lm_head (FP32)。embedding と tied の場合は embedding のクローン。
    pub lm_head: Vec<f32>,
    /// 全レイヤー重み (dequantized FP32)。
    pub layers: Vec<Qwen35LayerWeights>,
}

/// 生成パラメータ。
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    /// 最大生成トークン数。
    pub max_tokens: usize,
    /// Temperature (0.0 = greedy)。
    pub temperature: f32,
    /// Top-k サンプリング (0 = 無効)。
    pub top_k: usize,
    /// 繰り返しペナルティ。
    pub repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_k: 50,
            repetition_penalty: 1.1,
        }
    }
}

impl GenerationConfig {
    /// Greedy decoding (temperature=0)。
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.0,
            top_k: 0,
            repetition_penalty: 1.0,
        }
    }
}

impl AliceModel {
    /// `.alice` ファイルからモデルをロード。
    ///
    /// 全レイヤーを RAM にプリロードする (dequantized FP32)。
    /// Qwen3.5-9B で約 35GB RAM 必要。
    ///
    /// # Errors
    ///
    /// ファイル読み込みまたはフォーマットエラー時。
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);
        Self::from_reader(&mut reader)
    }

    /// Reader からモデルをロード。
    ///
    /// # Errors
    ///
    /// I/O またはフォーマットエラー時。
    pub fn from_reader<R: Read>(reader: &mut R) -> io::Result<Self> {
        let meta = read_alice_meta(reader)?;
        let config = &meta.config;

        eprintln!("  モデル読み込み中...");
        eprintln!(
            "    config: {}層, hidden={}, vocab={}",
            config.num_hidden_layers, config.hidden_size, config.vocab_size
        );

        // Embedding
        eprintln!("    embedding...");
        let embedding = read_embedding(reader, config.vocab_size, config.hidden_size)?;

        // Output norm
        let output_norm = read_output_norm(reader, config.hidden_size)?;

        // lm_head
        let lm_head_opt = read_lm_head(
            reader,
            meta.tied_embeddings,
            config.vocab_size,
            config.hidden_size,
        )?;
        let lm_head = lm_head_opt.unwrap_or_else(|| embedding.clone());

        // Layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = match config.layer_type(i) {
                LayerType::LinearAttention => {
                    Qwen35LayerWeights::DeltaNet(read_deltanet_layer(reader, config)?)
                }
                LayerType::FullAttention => {
                    Qwen35LayerWeights::FullAttention(read_fullattn_layer(reader, config)?)
                }
            };
            layers.push(layer);

            if (i + 1) % 8 == 0 || i == config.num_hidden_layers - 1 {
                eprintln!("    layers: {}/{}", i + 1, config.num_hidden_layers);
            }
        }

        eprintln!("  モデル読み込み完了");

        Ok(Self {
            meta,
            embedding,
            output_norm,
            lm_head,
            layers,
        })
    }

    /// Config への参照。
    #[must_use]
    pub fn config(&self) -> &Qwen35Config {
        &self.meta.config
    }

    /// 入力トークン列から logits を計算。
    ///
    /// 全シーケンスの logits を返す `[seq_len × vocab_size]`。
    #[must_use]
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        crate::qwen35_forward::qwen35_model_forward_eval(
            token_ids,
            &self.embedding,
            &self.layers,
            &self.output_norm,
            &self.lm_head,
            self.config(),
        )
    }

    /// テキストを生成 (autoregressive)。
    ///
    /// `prompt_ids`: 入力トークン ID 列。
    /// 生成されたトークン ID 列（プロンプト部分は含まない）を返す。
    #[must_use]
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        config: &GenerationConfig,
        eos_token_id: u32,
    ) -> Vec<u32> {
        let vocab_size = self.config().vocab_size;
        let mut generated = Vec::new();
        let mut all_ids: Vec<u32> = prompt_ids.to_vec();

        for _ in 0..config.max_tokens {
            // Forward: 全シーケンスを再計算 (KV-cache なし)
            let logits = self.forward(&all_ids);

            // 最後のトークンの logits を取得
            let last_logits_start = (all_ids.len() - 1) * vocab_size;
            let last_logits = &logits[last_logits_start..last_logits_start + vocab_size];

            // Repetition penalty
            let mut logits_vec: Vec<f32> = last_logits.to_vec();
            if (config.repetition_penalty - 1.0).abs() > f32::EPSILON {
                apply_repetition_penalty(&mut logits_vec, &all_ids, config.repetition_penalty);
            }

            // サンプリング
            let next_token = if config.temperature <= 0.0 {
                argmax(&logits_vec)
            } else {
                sample_top_k(&mut logits_vec, config.temperature, config.top_k)
            };

            // EOS チェック
            if next_token == eos_token_id {
                break;
            }

            generated.push(next_token);
            all_ids.push(next_token);
        }

        generated
    }

    /// 1トークンずつ生成し、コールバックで出力。
    ///
    /// ストリーミング生成用。コールバックが `false` を返すと生成停止。
    pub fn generate_streaming<F>(
        &self,
        prompt_ids: &[u32],
        config: &GenerationConfig,
        eos_token_id: u32,
        tokenizer: &BpeTokenizer,
        mut callback: F,
    ) where
        F: FnMut(&str) -> bool,
    {
        let vocab_size = self.config().vocab_size;
        let mut all_ids: Vec<u32> = prompt_ids.to_vec();

        for _ in 0..config.max_tokens {
            let logits = self.forward(&all_ids);

            let last_logits_start = (all_ids.len() - 1) * vocab_size;
            let last_logits = &logits[last_logits_start..last_logits_start + vocab_size];

            let mut logits_vec: Vec<f32> = last_logits.to_vec();
            if (config.repetition_penalty - 1.0).abs() > f32::EPSILON {
                apply_repetition_penalty(&mut logits_vec, &all_ids, config.repetition_penalty);
            }

            let next_token = if config.temperature <= 0.0 {
                argmax(&logits_vec)
            } else {
                sample_top_k(&mut logits_vec, config.temperature, config.top_k)
            };

            if next_token == eos_token_id {
                break;
            }

            let text = tokenizer.decode(&[next_token]);
            if !callback(&text) {
                break;
            }

            all_ids.push(next_token);
        }
    }
}

// ── Sampling ─────────────────────────────────────────────────────────────

/// Argmax: 最大値のインデックスを返す。
fn argmax(logits: &[f32]) -> u32 {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx as u32
}

/// Top-k サンプリング with temperature。
fn sample_top_k(logits: &mut [f32], temperature: f32, top_k: usize) -> u32 {
    // Temperature scaling
    let inv_temp = 1.0 / temperature.max(1e-10);
    for v in logits.iter_mut() {
        *v *= inv_temp;
    }

    // Top-k フィルタリング
    let k = if top_k > 0 && top_k < logits.len() {
        top_k
    } else {
        logits.len()
    };

    // k 番目に大きい値を見つける
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = logits[indices[k - 1]];

    // threshold 未満を -inf にマスク
    for (i, v) in logits.iter_mut().enumerate() {
        if *v < threshold && !indices[..k].contains(&i) {
            *v = f32::NEG_INFINITY;
        }
    }

    // Softmax
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    let mut probs: Vec<f64> = logits
        .iter()
        .map(|&v| {
            let p = ((v - max_val) as f64).exp();
            sum += p;
            p
        })
        .collect();
    for p in &mut probs {
        *p /= sum;
    }

    // 累積分布からサンプリング (簡易 PRNG)
    let rand_val = simple_random() as f64;
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= rand_val {
            return i as u32;
        }
    }

    // フォールバック: 最後のトークン
    (probs.len() - 1) as u32
}

/// Repetition penalty 適用。
fn apply_repetition_penalty(logits: &mut [f32], past_ids: &[u32], penalty: f32) {
    for &id in past_ids {
        let idx = id as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// 簡易乱数生成 (xorshift64)。
///
/// スレッドローカルな状態を使用。暗号用途には不適。
fn simple_random() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(12345, |d| d.as_nanos() as u64)
        );
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        // [0, 1) の範囲
        (x >> 11) as f32 / (1u64 << 53) as f32
    })
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(argmax(&logits), 3);
    }

    #[test]
    fn test_argmax_negative() {
        let logits = vec![-1.0, -0.5, -2.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        apply_repetition_penalty(&mut logits, &[1, 3], 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6); // 未出現: そのまま
        assert!((logits[1] - 1.0).abs() < 1e-6); // 2.0 / 2.0 = 1.0
        assert!((logits[2] - 3.0).abs() < 1e-6); // 未出現: そのまま
        assert!((logits[3] - 2.0).abs() < 1e-6); // 4.0 / 2.0 = 2.0
    }

    #[test]
    fn test_repetition_penalty_negative() {
        let mut logits = vec![-1.0, -2.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        assert!((logits[0] - (-2.0)).abs() < 1e-6); // -1.0 * 2.0 = -2.0
        assert!((logits[1] - (-4.0)).abs() < 1e-6); // -2.0 * 2.0 = -4.0
    }

    #[test]
    fn test_simple_random_range() {
        for _ in 0..100 {
            let r = simple_random();
            assert!((0.0..1.0).contains(&r), "random {r} out of range");
        }
    }

    #[test]
    fn test_sample_top_k_deterministic() {
        // temperature=0 相当の高い差があるとき、top-1 はほぼ argmax
        let mut logits = vec![-100.0; 10];
        logits[7] = 100.0;
        let result = sample_top_k(&mut logits, 0.01, 1);
        assert_eq!(result, 7);
    }

    #[test]
    fn test_generation_config_default() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.max_tokens, 256);
        assert!((cfg.temperature - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_generation_config_greedy() {
        let cfg = GenerationConfig::greedy();
        assert!((cfg.temperature - 0.0).abs() < 1e-6);
        assert_eq!(cfg.top_k, 0);
    }
}
