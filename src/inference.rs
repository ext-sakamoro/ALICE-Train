//! .alice モデル推論エンジン — ternary QAT モデルのテキスト生成。
//!
//! `.alice` ファイルからモデルをロードし、autoregressive にテキストを生成する。

use crate::export::{
    read_alice_meta, read_deltanet_layer, read_embedding, read_fullattn_layer, read_lm_head,
    read_output_norm, AliceModelMeta,
};
use crate::qwen35::{LayerType, Qwen35Config, Qwen35LayerWeights};
use crate::tokenizer::BpeTokenizer;
use std::io::{self, BufReader, Read, Seek};
use std::path::{Path, PathBuf};

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

    /// ストリーミング生成 + ストップシーケンス対応。
    ///
    /// `stop_sequences` に含まれる文字列が生成テキストに出現したら生成を停止する。
    /// ツール呼び出しフォーマットの `</tool_use>` 検出に使用。
    ///
    /// 返り値: 生成されたテキスト全体。
    pub fn generate_streaming_with_stop<F>(
        &self,
        prompt_ids: &[u32],
        config: &GenerationConfig,
        eos_token_id: u32,
        stop_sequences: &[&str],
        tokenizer: &BpeTokenizer,
        mut callback: F,
    ) -> String
    where
        F: FnMut(&str) -> bool,
    {
        let vocab_size = self.config().vocab_size;
        let mut all_ids: Vec<u32> = prompt_ids.to_vec();
        let mut generated_text = String::new();

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
            generated_text.push_str(&text);

            if !callback(&text) {
                break;
            }

            // ストップシーケンスチェック
            let should_stop = stop_sequences
                .iter()
                .any(|seq| generated_text.ends_with(seq));
            if should_stop {
                break;
            }

            all_ids.push(next_token);
        }

        generated_text
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

// ── Incremental Inference (KV-cache + DeltaNet state cache) ─────────────────

/// DeltaNet 再帰状態 + Full Attention KV キャッシュ。
///
/// `create_cache` で初期化し、`forward_incremental` でトークンごとに更新する。
pub struct InferenceCache {
    /// DeltaNet 層の再帰状態: 層ごとに `[num_v_heads][dk × dv]`。
    /// レイヤー順に格納し、Full Attention 層は空 Vec。
    pub deltanet_states: Vec<Vec<Vec<f32>>>,
    /// Full Attention 層の K キャッシュ: 層ごとに `[seq_so_far × num_kv_heads × head_dim]`。
    pub full_attn_k_cache: Vec<Vec<f32>>,
    /// Full Attention 層の V キャッシュ: 層ごとに `[seq_so_far × num_kv_heads × head_dim]`。
    pub full_attn_v_cache: Vec<Vec<f32>>,
    /// キャッシュに積まれたトークン数 (プロンプト含む)。
    pub seq_len: usize,
}

impl AliceModel {
    /// `InferenceCache` を初期化 (すべてゼロ状態)。
    #[must_use]
    pub fn create_cache(&self) -> InferenceCache {
        let config = self.config();
        let num_layers = config.num_hidden_layers;
        let dk = config.linear_key_head_dim;
        let dv = config.linear_value_head_dim;
        let n_v_heads = config.linear_num_value_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let mut deltanet_states = Vec::with_capacity(num_layers);
        let mut full_attn_k_cache = Vec::with_capacity(num_layers);
        let mut full_attn_v_cache = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            match config.layer_type(i) {
                crate::qwen35::LayerType::LinearAttention => {
                    let states: Vec<Vec<f32>> =
                        (0..n_v_heads).map(|_| vec![0.0f32; dk * dv]).collect();
                    deltanet_states.push(states);
                    full_attn_k_cache.push(Vec::new());
                    full_attn_v_cache.push(Vec::new());
                }
                crate::qwen35::LayerType::FullAttention => {
                    deltanet_states.push(Vec::new());
                    // 空で開始 — append していく
                    full_attn_k_cache.push(Vec::with_capacity(4096 * n_kv_heads * head_dim));
                    full_attn_v_cache.push(Vec::with_capacity(4096 * n_kv_heads * head_dim));
                }
            }
        }

        InferenceCache {
            deltanet_states,
            full_attn_k_cache,
            full_attn_v_cache,
            seq_len: 0,
        }
    }

    /// 1トークンの incremental forward。
    ///
    /// `token_id`: 入力トークン ID (単一)。
    /// `cache`: 更新対象のキャッシュ。
    /// 返り値: vocab_size の logits Vec。
    #[must_use]
    pub fn forward_incremental(&self, token_id: u32, cache: &mut InferenceCache) -> Vec<f32> {
        use crate::blas::{blas_matmul_bt, blas_rmsnorm};

        let config = self.config();
        let hidden = config.hidden_size;
        let vocab_size = config.vocab_size;
        let pos = cache.seq_len; // 現在位置 (0-indexed)

        // 1. Embedding lookup
        let tok = (token_id as usize) % vocab_size;
        let mut hidden_states = self.embedding[tok * hidden..(tok + 1) * hidden].to_vec();

        // 2. 各レイヤーを順番に処理
        for (layer_idx, layer_weights) in self.layers.iter().enumerate() {
            match layer_weights {
                crate::qwen35::Qwen35LayerWeights::DeltaNet(w) => {
                    hidden_states = self.incremental_deltanet_layer(
                        hidden_states,
                        w,
                        config,
                        &mut cache.deltanet_states[layer_idx],
                    );
                }
                crate::qwen35::Qwen35LayerWeights::FullAttention(w) => {
                    hidden_states = self.incremental_full_attn_layer(
                        hidden_states,
                        w,
                        config,
                        &mut cache.full_attn_k_cache[layer_idx],
                        &mut cache.full_attn_v_cache[layer_idx],
                        pos,
                    );
                }
            }
        }

        // 3. Output norm
        blas_rmsnorm(&mut hidden_states, &self.output_norm, hidden, config.rms_norm_eps);

        // 4. lm_head
        let mut logits = vec![0.0f32; vocab_size];
        blas_matmul_bt(&hidden_states, &self.lm_head, &mut logits, 1, vocab_size, hidden);

        cache.seq_len += 1;
        logits
    }

    /// DeltaNet 層の incremental forward (1トークン)。
    fn incremental_deltanet_layer(
        &self,
        mut input: Vec<f32>,
        weights: &crate::qwen35::DeltaNetLayerWeights,
        config: &crate::qwen35::Qwen35Config,
        states: &mut Vec<Vec<f32>>,
    ) -> Vec<f32> {
        use crate::blas::{blas_matmul_bt, blas_rmsnorm, blas_swiglu_ffn};
        use crate::deltanet::{
            causal_conv1d_silu_row_major, compute_gates_fused, gated_rmsnorm,
            head_recurrence_forward_eval, l2norm_and_gqa_expand,
        };

        let hidden = config.hidden_size;
        let key_dim = config.linear_key_dim();
        let val_dim = config.linear_value_dim();
        let n_k_heads = config.linear_num_key_heads;
        let n_v_heads = config.linear_num_value_heads;
        let dk = config.linear_key_head_dim;
        let dv = config.linear_value_head_dim;
        let kernel_size = config.linear_conv_kernel_dim;
        let qkv_dim = key_dim * 2 + val_dim;
        let seq_len = 1;

        let residual = input.clone();
        blas_rmsnorm(&mut input, &weights.input_layernorm, hidden, config.rms_norm_eps);
        let normed = input.clone();

        let mut qkv = vec![0.0f32; qkv_dim];
        blas_matmul_bt(&normed, &weights.in_proj_qkv, &mut qkv, 1, qkv_dim, hidden);

        let mut z = vec![0.0f32; val_dim];
        blas_matmul_bt(&normed, &weights.in_proj_z, &mut z, 1, val_dim, hidden);

        let mut b_raw = vec![0.0f32; n_v_heads];
        blas_matmul_bt(&normed, &weights.in_proj_b, &mut b_raw, 1, n_v_heads, hidden);

        let mut a_raw = vec![0.0f32; n_v_heads];
        blas_matmul_bt(&normed, &weights.in_proj_a, &mut a_raw, 1, n_v_heads, hidden);

        // causal conv1d (seq_len=1 では past_context なしの単純適用)
        let mut qkv_conv = vec![0.0f32; qkv_dim];
        causal_conv1d_silu_row_major(
            &qkv,
            &weights.conv1d_weight,
            &mut qkv_conv,
            qkv_dim,
            seq_len,
            kernel_size,
        );

        // Q/K/V 分割
        let q_raw = qkv_conv[..key_dim].to_vec();
        let k_raw = qkv_conv[key_dim..key_dim * 2].to_vec();
        let v_all = qkv_conv[key_dim * 2..qkv_dim].to_vec();

        // L2 norm + GQA expand
        let mut q_expanded = vec![0.0f32; n_v_heads * dk];
        let mut k_expanded = vec![0.0f32; n_v_heads * dk];
        l2norm_and_gqa_expand(
            &q_raw,
            &k_raw,
            &mut q_expanded,
            &mut k_expanded,
            seq_len,
            n_k_heads,
            n_v_heads,
            dk,
            1e-6,
        );

        // Gates
        let mut beta = vec![0.0f32; n_v_heads];
        let mut g = vec![0.0f32; n_v_heads];
        compute_gates_fused(
            &b_raw,
            &a_raw,
            &weights.a_log,
            &weights.dt_bias,
            &mut beta,
            &mut g,
            seq_len,
            n_v_heads,
        );

        // DeltaNet 再帰: 各ヘッドの state を in-place 更新
        let mut attn_out_raw = vec![0.0f32; n_v_heads * dv];
        for h in 0..n_v_heads {
            let q_h = &q_expanded[h * dk..(h + 1) * dk];
            let k_h = &k_expanded[h * dk..(h + 1) * dk];
            let v_h = &v_all[h * dv..(h + 1) * dv];
            let out_h = &mut attn_out_raw[h * dv..(h + 1) * dv];
            head_recurrence_forward_eval(
                q_h,
                k_h,
                v_h,
                &[beta[h]],
                &[g[h]],
                out_h,
                &mut states[h],
                dk,
                dv,
                1,
            );
        }

        // Gated RMSNorm
        let mut attn_normed = vec![0.0f32; val_dim];
        gated_rmsnorm(
            &attn_out_raw,
            &z,
            &weights.norm_weight,
            &mut attn_normed,
            dv,
            config.rms_norm_eps,
        );

        // Output projection
        let mut attn_out = vec![0.0f32; hidden];
        blas_matmul_bt(&attn_normed, &weights.out_proj, &mut attn_out, 1, hidden, val_dim);

        // Residual add
        let mut output: Vec<f32> = residual.iter().zip(attn_out.iter()).map(|(r, a)| r + a).collect();

        // FFN
        let residual_ffn = output.clone();
        let mut normed_ffn = output.clone();
        blas_rmsnorm(&mut normed_ffn, &weights.post_attn_layernorm, hidden, config.rms_norm_eps);

        let inter = config.intermediate_size;
        let mut ffn_out = vec![0.0f32; hidden];
        let mut gate_buf = vec![0.0f32; inter];
        let mut up_buf = vec![0.0f32; inter];
        let mut gate_silu_buf = vec![0.0f32; inter];
        blas_swiglu_ffn(
            &normed_ffn,
            &weights.gate_proj,
            &weights.up_proj,
            &weights.down_proj,
            &mut ffn_out,
            &mut gate_buf,
            &mut up_buf,
            &mut gate_silu_buf,
            1,
            hidden,
            inter,
        );

        for i in 0..output.len() {
            output[i] = residual_ffn[i] + ffn_out[i];
        }

        output
    }

    /// Full Attention 層の incremental forward (1トークン)。
    ///
    /// K/V を cache に append し、新 Q × 全 K^T で attention を計算する。
    fn incremental_full_attn_layer(
        &self,
        mut input: Vec<f32>,
        weights: &crate::qwen35::FullAttnLayerWeights,
        config: &crate::qwen35::Qwen35Config,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
        position: usize,
    ) -> Vec<f32> {
        use crate::blas::{blas_matmul_bt, blas_rmsnorm, blas_swiglu_ffn};
        use crate::deltanet::qk_norm;

        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let rotary_dim = config.rotary_dim();
        let inter = config.intermediate_size;

        let residual = input.clone();
        blas_rmsnorm(&mut input, &weights.input_layernorm, hidden, config.rms_norm_eps);
        let normed = input.clone();

        // Q/K/V projection (1トークン)
        let mut q = vec![0.0f32; num_heads * head_dim];
        let mut k_new = vec![0.0f32; num_kv_heads * head_dim];
        let mut v_new = vec![0.0f32; num_kv_heads * head_dim];

        blas_matmul_bt(&normed, &weights.q_proj, &mut q, 1, num_heads * head_dim, hidden);
        blas_matmul_bt(&normed, &weights.k_proj, &mut k_new, 1, num_kv_heads * head_dim, hidden);
        blas_matmul_bt(&normed, &weights.v_proj, &mut v_new, 1, num_kv_heads * head_dim, hidden);

        // QK-norm
        qk_norm(&mut q, &weights.q_norm, num_heads, head_dim, config.rms_norm_eps);
        qk_norm(&mut k_new, &weights.k_norm, num_kv_heads, head_dim, config.rms_norm_eps);

        // RoPE (position offset 付き)
        apply_rope_with_offset(&mut q, num_heads, head_dim, rotary_dim, position, config.rope_theta);
        apply_rope_with_offset(&mut k_new, num_kv_heads, head_dim, rotary_dim, position, config.rope_theta);

        // K/V cache に append
        k_cache.extend_from_slice(&k_new);
        v_cache.extend_from_slice(&v_new);

        let cache_len = k_cache.len() / (num_kv_heads * head_dim);

        // Attention: Q (1×heads×head_dim) × K^T (cache_len×kv_heads×head_dim)
        let kv_groups = num_heads / num_kv_heads;
        let scale = (head_dim as f32).sqrt().recip();
        let mut attn_out_raw = vec![0.0f32; num_heads * head_dim];

        for h in 0..num_heads {
            let kv_h = h / kv_groups;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let out_h = &mut attn_out_raw[h * head_dim..(h + 1) * head_dim];

            // score[t] = q · k[t] * scale
            let mut scores = vec![0.0f32; cache_len];
            for t in 0..cache_len {
                let k_t = &k_cache[(t * num_kv_heads + kv_h) * head_dim..
                                   (t * num_kv_heads + kv_h + 1) * head_dim];
                let dot: f32 = q_h.iter().zip(k_t.iter()).map(|(a, b)| a * b).sum();
                scores[t] = dot * scale;
            }

            // softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            let inv_sum = sum.recip();
            for s in &mut scores {
                *s *= inv_sum;
            }

            // weighted sum of V
            for t in 0..cache_len {
                let v_t = &v_cache[(t * num_kv_heads + kv_h) * head_dim..
                                   (t * num_kv_heads + kv_h + 1) * head_dim];
                let w = scores[t];
                for d in 0..head_dim {
                    out_h[d] += w * v_t[d];
                }
            }
        }

        // O projection
        let mut attn_out = vec![0.0f32; hidden];
        blas_matmul_bt(&attn_out_raw, &weights.o_proj, &mut attn_out, 1, hidden, num_heads * head_dim);

        // Residual
        let mut output: Vec<f32> = residual.iter().zip(attn_out.iter()).map(|(r, a)| r + a).collect();

        // FFN
        let residual_ffn = output.clone();
        let mut normed_ffn = output.clone();
        blas_rmsnorm(&mut normed_ffn, &weights.post_attn_layernorm, hidden, config.rms_norm_eps);

        let mut ffn_out = vec![0.0f32; hidden];
        let mut gate_buf = vec![0.0f32; inter];
        let mut up_buf = vec![0.0f32; inter];
        let mut gate_silu_buf = vec![0.0f32; inter];
        blas_swiglu_ffn(
            &normed_ffn,
            &weights.gate_proj,
            &weights.up_proj,
            &weights.down_proj,
            &mut ffn_out,
            &mut gate_buf,
            &mut up_buf,
            &mut gate_silu_buf,
            1,
            hidden,
            inter,
        );

        for i in 0..output.len() {
            output[i] = residual_ffn[i] + ffn_out[i];
        }

        output
    }

    /// キャッシュ付きストリーミング生成。
    ///
    /// プロンプトをキャッシュにプリフィルしてから、
    /// 1トークンずつ incremental forward で生成する。
    /// プロンプト部分は含まず、生成されたトークン ID 列を返す。
    #[must_use]
    pub fn generate_cached(
        &self,
        prompt_ids: &[u32],
        gen_config: &GenerationConfig,
        eos_token_id: u32,
    ) -> Vec<u32> {
        let mut cache = self.create_cache();

        // プロンプトをキャッシュにプリフィル
        let mut all_ids: Vec<u32> = prompt_ids.to_vec();
        for &tok in prompt_ids {
            let _ = self.forward_incremental(tok, &mut cache);
        }

        // 生成ループ
        let mut generated = Vec::new();
        let mut last_token = *prompt_ids.last().unwrap_or(&0);

        for _ in 0..gen_config.max_tokens {
            let mut logits = self.forward_incremental(last_token, &mut cache);

            if (gen_config.repetition_penalty - 1.0).abs() > f32::EPSILON {
                apply_repetition_penalty(&mut logits, &all_ids, gen_config.repetition_penalty);
            }

            let next_token = if gen_config.temperature <= 0.0 {
                argmax(&logits)
            } else {
                sample_top_k(&mut logits, gen_config.temperature, gen_config.top_k)
            };

            if next_token == eos_token_id {
                break;
            }

            generated.push(next_token);
            all_ids.push(next_token);
            last_token = next_token;
        }

        generated
    }
}

/// RoPE を position offset 付きで適用 (incremental inference 用)。
///
/// `x`: `[n_heads × head_dim]` (1トークン分)。
/// `position`: 現在のトークン位置 (0-indexed)。
fn apply_rope_with_offset(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    theta: f32,
) {
    let half_rot = rotary_dim / 2;
    for h in 0..n_heads {
        let row = &mut x[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_rot {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / rotary_dim as f32);
            let angle = position as f32 * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = row[i];
            let x1 = row[i + half_rot];
            row[i] = x0 * cos_a - x1 * sin_a;
            row[i + half_rot] = x0 * sin_a + x1 * cos_a;
        }
    }
}

// ── mmap + JIT Ternary 推論 ──────────────────────────────────────────────

/// mmap 上の packed ternary projection の位置情報。
#[derive(Clone, Debug)]
struct PackedRef {
    /// mmap 内の scale (f32 LE) の開始バイト。
    scale_off: usize,
    /// mmap 内の packed ternary data の開始バイト。
    packed_off: usize,
    /// packed bytes 数。
    packed_len: usize,
    /// 要素数。
    count: usize,
}

/// mmap 上の FP32 スライスの位置情報。
#[derive(Clone, Debug)]
struct F32Ref {
    off: usize,
    count: usize,
}

/// DeltaNet 層の mmap オフセット。
#[derive(Clone, Debug)]
struct MmapDeltaNet {
    in_proj_qkv: PackedRef,
    in_proj_z: PackedRef,
    in_proj_b: PackedRef,
    in_proj_a: PackedRef,
    out_proj: PackedRef,
    gate_proj: PackedRef,
    up_proj: PackedRef,
    down_proj: PackedRef,
    input_layernorm: F32Ref,
    post_attn_layernorm: F32Ref,
    a_log: F32Ref,
    dt_bias: F32Ref,
    conv1d_weight: F32Ref,
    norm_weight: F32Ref,
}

/// FullAttn 層の mmap オフセット。
#[derive(Clone, Debug)]
struct MmapFullAttn {
    q_proj: PackedRef,
    k_proj: PackedRef,
    v_proj: PackedRef,
    o_proj: PackedRef,
    gate_proj: PackedRef,
    up_proj: PackedRef,
    down_proj: PackedRef,
    input_layernorm: F32Ref,
    post_attn_layernorm: F32Ref,
    q_norm: F32Ref,
    k_norm: F32Ref,
}

/// レイヤーの mmap インデックス。
#[derive(Clone, Debug)]
enum MmapLayer {
    DeltaNet(MmapDeltaNet),
    FullAttn(MmapFullAttn),
}

/// ストリーミング推論モデル — mmap + JIT ternary unpacking。
///
/// ternary packed データを FP32 に展開**せず** mmap 上に保持。
/// matmul の瞬間にのみ 2-bit→{-1,0,+1}×γ をレジスタ内で実行。
///
/// RAM: embedding (~4GB) + lm\_head (~4GB or tied) + mmap (OS管理) + buffers (~100MB)。
/// FP32 展開の ~35GB は一切発生しない。
pub struct StreamingAliceModel {
    /// メタデータ。
    pub meta: AliceModelMeta,
    /// Embedding テーブル (FP32)。
    pub embedding: Vec<f32>,
    /// Output layernorm (FP32)。
    pub output_norm: Vec<f32>,
    /// lm\_head (FP32)。None = tied (embedding と共有)。
    lm_head_storage: Option<Vec<f32>>,
    /// mmap されたファイル全体。
    mmap: memmap2::Mmap,
    /// 各レイヤーのオフセットインデックス。
    layer_index: Vec<MmapLayer>,
}

impl std::fmt::Debug for StreamingAliceModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingAliceModel")
            .field("layers", &self.meta.config.num_hidden_layers)
            .field("hidden", &self.meta.config.hidden_size)
            .field("vocab", &self.meta.config.vocab_size)
            .field("tied", &self.lm_head_storage.is_none())
            .field("mmap_size", &self.mmap.len())
            .finish_non_exhaustive()
    }
}

/// 現在のオフセットから packed projection のインデックスを構築し、オフセットを進める。
fn build_packed_ref(cursor: &mut usize, count: usize) -> PackedRef {
    let scale_off = *cursor;
    *cursor += 4; // f32 scale
    let packed_len = count.div_ceil(4);
    let packed_off = *cursor;
    *cursor += packed_len;
    PackedRef { scale_off, packed_off, packed_len, count }
}

/// 現在のオフセットから FP32 スライスのインデックスを構築し、オフセットを進める。
fn build_f32_ref(cursor: &mut usize, count: usize) -> F32Ref {
    let off = *cursor;
    *cursor += count * 4;
    F32Ref { off, count }
}

impl StreamingAliceModel {
    /// `.alice` ファイルを mmap でオープンし、JIT 推論モデルを構築。
    ///
    /// embedding + output\_norm + lm\_head のみ FP32 でメモリに配置。
    /// レイヤー重みは mmap 上に packed ternary のまま保持。
    ///
    /// # Errors
    ///
    /// ファイル読み込みまたはフォーマットエラー時。
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref();

        // 1. mmap
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

        // 2. ヘッダー読み込み (mmap からカーソルで読む)
        let mut reader = io::Cursor::new(&mmap[..]);
        let meta = read_alice_meta(&mut reader)?;
        let config = &meta.config;

        eprintln!("  [mmap] ヘッダー読み込み完了");
        eprintln!(
            "    config: {}層, hidden={}, vocab={}",
            config.num_hidden_layers, config.hidden_size, config.vocab_size
        );

        // 3. embedding / output_norm / lm_head — FP32 展開 (計算で毎回必要)
        eprintln!("    embedding...");
        let embedding = read_embedding(&mut reader, config.vocab_size, config.hidden_size)?;
        let output_norm = read_output_norm(&mut reader, config.hidden_size)?;
        let lm_head_storage = read_lm_head(
            &mut reader,
            meta.tied_embeddings,
            config.vocab_size,
            config.hidden_size,
        )?;

        // 4. レイヤーオフセットインデックス構築 (mmap バイト走査、データコピーなし)
        let mut cursor = reader.position() as usize;
        let mut layer_index = Vec::with_capacity(config.num_hidden_layers);

        for i in 0..config.num_hidden_layers {
            let h = config.hidden_size;
            let inter = config.intermediate_size;

            match config.layer_type(i) {
                LayerType::LinearAttention => {
                    let kd = config.linear_key_dim();
                    let vd = config.linear_value_dim();
                    let nv = config.linear_num_value_heads;
                    let conv_dim = config.conv_dim();
                    let ks = config.linear_conv_kernel_dim;
                    let vhd = config.linear_value_head_dim;

                    let idx = MmapDeltaNet {
                        in_proj_qkv: build_packed_ref(&mut cursor, h * (kd * 2 + vd)),
                        in_proj_z: build_packed_ref(&mut cursor, h * vd),
                        in_proj_b: build_packed_ref(&mut cursor, h * nv),
                        in_proj_a: build_packed_ref(&mut cursor, h * nv),
                        out_proj: build_packed_ref(&mut cursor, vd * h),
                        gate_proj: build_packed_ref(&mut cursor, inter * h),
                        up_proj: build_packed_ref(&mut cursor, inter * h),
                        down_proj: build_packed_ref(&mut cursor, h * inter),
                        input_layernorm: build_f32_ref(&mut cursor, h),
                        post_attn_layernorm: build_f32_ref(&mut cursor, h),
                        a_log: build_f32_ref(&mut cursor, nv),
                        dt_bias: build_f32_ref(&mut cursor, nv),
                        conv1d_weight: build_f32_ref(&mut cursor, conv_dim * ks),
                        norm_weight: build_f32_ref(&mut cursor, vhd),
                    };
                    layer_index.push(MmapLayer::DeltaNet(idx));
                }
                LayerType::FullAttention => {
                    let nh = config.num_attention_heads;
                    let nkv = config.num_key_value_heads;
                    let hd = config.head_dim;

                    let idx = MmapFullAttn {
                        q_proj: build_packed_ref(&mut cursor, h * nh * hd),
                        k_proj: build_packed_ref(&mut cursor, h * nkv * hd),
                        v_proj: build_packed_ref(&mut cursor, h * nkv * hd),
                        o_proj: build_packed_ref(&mut cursor, nh * hd * h),
                        gate_proj: build_packed_ref(&mut cursor, inter * h),
                        up_proj: build_packed_ref(&mut cursor, inter * h),
                        down_proj: build_packed_ref(&mut cursor, h * inter),
                        input_layernorm: build_f32_ref(&mut cursor, h),
                        post_attn_layernorm: build_f32_ref(&mut cursor, h),
                        q_norm: build_f32_ref(&mut cursor, hd),
                        k_norm: build_f32_ref(&mut cursor, hd),
                    };
                    layer_index.push(MmapLayer::FullAttn(idx));
                }
            }
        }

        let emb_mb = (embedding.len() * 4) as f64 / 1e6;
        let lm_mb = lm_head_storage.as_ref().map_or(0.0, |v| (v.len() * 4) as f64 / 1e6);
        let layer_data_bytes = mmap.len() - reader.position() as usize;
        eprintln!(
            "  [mmap] RAM: embedding={:.0}MB, lm_head={:.0}MB, mmap={}MB (packed ternary, OS管理)",
            emb_mb, lm_mb, mmap.len() / 1_000_000
        );
        eprintln!(
            "  [mmap] layer data: {:.0}MB packed (FP32展開なし), index: {} layers",
            layer_data_bytes as f64 / 1e6,
            layer_index.len()
        );

        Ok(Self {
            meta,
            embedding,
            output_norm,
            lm_head_storage,
            mmap,
            layer_index,
        })
    }

    /// Config への参照。
    #[must_use]
    pub fn config(&self) -> &Qwen35Config {
        &self.meta.config
    }

    /// lm\_head 重みへの参照 (tied なら embedding を返す)。
    #[must_use]
    pub fn lm_head(&self) -> &[f32] {
        self.lm_head_storage.as_deref().unwrap_or(&self.embedding)
    }

    /// mmap から packed ternary data のスライスを取得。
    fn packed(&self, p: &PackedRef) -> (&[u8], f32) {
        let scale_bytes = &self.mmap[p.scale_off..p.scale_off + 4];
        let scale = f32::from_le_bytes([scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]]);
        let packed = &self.mmap[p.packed_off..p.packed_off + p.packed_len];
        (packed, scale)
    }

    /// mmap から FP32 スライスを取得 (ゼロコピー reinterpret)。
    fn f32_slice(&self, r: &F32Ref) -> Vec<f32> {
        let bytes = &self.mmap[r.off..r.off + r.count * 4];
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// `InferenceCache` を初期化。KV cache は遅延確保 (capacity=0)。
    #[must_use]
    pub fn create_cache(&self) -> InferenceCache {
        let config = self.config();
        let num_layers = config.num_hidden_layers;
        let dk = config.linear_key_head_dim;
        let dv = config.linear_value_head_dim;
        let n_v_heads = config.linear_num_value_heads;

        let mut deltanet_states = Vec::with_capacity(num_layers);
        let mut full_attn_k_cache = Vec::with_capacity(num_layers);
        let mut full_attn_v_cache = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            match config.layer_type(i) {
                LayerType::LinearAttention => {
                    let states: Vec<Vec<f32>> =
                        (0..n_v_heads).map(|_| vec![0.0f32; dk * dv]).collect();
                    deltanet_states.push(states);
                    full_attn_k_cache.push(Vec::new());
                    full_attn_v_cache.push(Vec::new());
                }
                LayerType::FullAttention => {
                    deltanet_states.push(Vec::new());
                    // 遅延確保: capacity=0、extend_from_slice で自動 grow
                    full_attn_k_cache.push(Vec::new());
                    full_attn_v_cache.push(Vec::new());
                }
            }
        }

        InferenceCache {
            deltanet_states,
            full_attn_k_cache,
            full_attn_v_cache,
            seq_len: 0,
        }
    }

    /// 1トークンの incremental forward (mmap + JIT ternary)。
    ///
    /// Ping-Pong バッファで hidden\_states を使い回し。
    /// ternary projection は mmap 上の packed data を直接参照、FP32 展開なし。
    pub fn forward_incremental_streaming(
        &self,
        token_id: u32,
        cache: &mut InferenceCache,
    ) -> io::Result<Vec<f32>> {
        use crate::blas::{blas_matmul_bt, blas_rmsnorm, ternary_matmul_bt, ternary_swiglu_ffn};

        let config = self.config();
        let hidden = config.hidden_size;
        let vocab_size = config.vocab_size;
        let inter = config.intermediate_size;
        let pos = cache.seq_len;

        // Embedding lookup → buf_a
        let tok = (token_id as usize) % vocab_size;
        let mut buf_a = self.embedding[tok * hidden..(tok + 1) * hidden].to_vec();
        let mut buf_b = vec![0.0f32; hidden];

        // Reusable FFN buffers
        let mut gate_buf = vec![0.0f32; inter];
        let mut up_buf = vec![0.0f32; inter];

        for (layer_idx, layer_ref) in self.layer_index.iter().enumerate() {
            match layer_ref {
                MmapLayer::DeltaNet(idx) => {
                    self.jit_deltanet_incremental(
                        &mut buf_a, &mut buf_b, idx, config,
                        &mut cache.deltanet_states[layer_idx],
                        &mut gate_buf, &mut up_buf,
                    );
                    // buf_a に結果が入っている
                }
                MmapLayer::FullAttn(idx) => {
                    self.jit_fullattn_incremental(
                        &mut buf_a, &mut buf_b, idx, config,
                        &mut cache.full_attn_k_cache[layer_idx],
                        &mut cache.full_attn_v_cache[layer_idx],
                        pos,
                        &mut gate_buf, &mut up_buf,
                    );
                }
            }
        }

        // Output norm
        blas_rmsnorm(&mut buf_a, &self.output_norm, hidden, config.rms_norm_eps);

        // lm_head (FP32、BLAS)
        let mut logits = vec![0.0f32; vocab_size];
        blas_matmul_bt(&buf_a, self.lm_head(), &mut logits, 1, vocab_size, hidden);

        cache.seq_len += 1;
        Ok(logits)
    }

    /// DeltaNet incremental (JIT ternary)。buf_a: 入力兼出力。
    #[allow(clippy::too_many_arguments)]
    fn jit_deltanet_incremental(
        &self,
        buf_a: &mut Vec<f32>,
        buf_b: &mut Vec<f32>,
        idx: &MmapDeltaNet,
        config: &Qwen35Config,
        states: &mut Vec<Vec<f32>>,
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
    ) {
        use crate::blas::{blas_rmsnorm, ternary_matmul_bt, ternary_swiglu_ffn};
        use crate::deltanet::{
            causal_conv1d_silu_row_major, compute_gates_fused, gated_rmsnorm,
            head_recurrence_forward_eval, l2norm_and_gqa_expand,
        };

        let hidden = config.hidden_size;
        let key_dim = config.linear_key_dim();
        let val_dim = config.linear_value_dim();
        let n_k_heads = config.linear_num_key_heads;
        let n_v_heads = config.linear_num_value_heads;
        let dk = config.linear_key_head_dim;
        let dv = config.linear_value_head_dim;
        let kernel_size = config.linear_conv_kernel_dim;
        let qkv_dim = key_dim * 2 + val_dim;
        let inter = config.intermediate_size;

        // residual = buf_a のコピー
        let residual: Vec<f32> = buf_a.clone();

        // LayerNorm (FP32、mmap から小さいスライスをコピー)
        let ln = self.f32_slice(&idx.input_layernorm);
        blas_rmsnorm(buf_a, &ln, hidden, config.rms_norm_eps);

        // Ternary projections — JIT matmul
        let mut qkv = vec![0.0f32; qkv_dim];
        let (p, s) = self.packed(&idx.in_proj_qkv);
        ternary_matmul_bt(buf_a, p, s, &mut qkv, 1, qkv_dim, hidden);

        let mut z = vec![0.0f32; val_dim];
        let (p, s) = self.packed(&idx.in_proj_z);
        ternary_matmul_bt(buf_a, p, s, &mut z, 1, val_dim, hidden);

        let mut b_raw = vec![0.0f32; n_v_heads];
        let (p, s) = self.packed(&idx.in_proj_b);
        ternary_matmul_bt(buf_a, p, s, &mut b_raw, 1, n_v_heads, hidden);

        let mut a_raw = vec![0.0f32; n_v_heads];
        let (p, s) = self.packed(&idx.in_proj_a);
        ternary_matmul_bt(buf_a, p, s, &mut a_raw, 1, n_v_heads, hidden);

        // conv1d
        let conv_w = self.f32_slice(&idx.conv1d_weight);
        let mut qkv_conv = vec![0.0f32; qkv_dim];
        causal_conv1d_silu_row_major(&qkv, &conv_w, &mut qkv_conv, qkv_dim, 1, kernel_size);

        let q_raw = &qkv_conv[..key_dim];
        let k_raw = &qkv_conv[key_dim..key_dim * 2];
        let v_all = &qkv_conv[key_dim * 2..qkv_dim];

        let mut q_expanded = vec![0.0f32; n_v_heads * dk];
        let mut k_expanded = vec![0.0f32; n_v_heads * dk];
        l2norm_and_gqa_expand(q_raw, k_raw, &mut q_expanded, &mut k_expanded, 1, n_k_heads, n_v_heads, dk, 1e-6);

        let a_log = self.f32_slice(&idx.a_log);
        let dt_bias = self.f32_slice(&idx.dt_bias);
        let mut beta = vec![0.0f32; n_v_heads];
        let mut g = vec![0.0f32; n_v_heads];
        compute_gates_fused(&b_raw, &a_raw, &a_log, &dt_bias, &mut beta, &mut g, 1, n_v_heads);

        let mut attn_out_raw = vec![0.0f32; n_v_heads * dv];
        for h in 0..n_v_heads {
            head_recurrence_forward_eval(
                &q_expanded[h * dk..(h + 1) * dk],
                &k_expanded[h * dk..(h + 1) * dk],
                &v_all[h * dv..(h + 1) * dv],
                &[beta[h]], &[g[h]],
                &mut attn_out_raw[h * dv..(h + 1) * dv],
                &mut states[h], dk, dv, 1,
            );
        }

        let norm_w = self.f32_slice(&idx.norm_weight);
        let mut attn_normed = vec![0.0f32; val_dim];
        gated_rmsnorm(&attn_out_raw, &z, &norm_w, &mut attn_normed, dv, config.rms_norm_eps);

        // out_proj — JIT ternary
        buf_b.fill(0.0);
        buf_b.resize(hidden, 0.0);
        let (p, s) = self.packed(&idx.out_proj);
        ternary_matmul_bt(&attn_normed, p, s, buf_b, 1, hidden, val_dim);

        // residual add → buf_a
        for i in 0..hidden {
            buf_a[i] = residual[i] + buf_b[i];
        }

        // FFN — JIT ternary SwiGLU
        let residual_ffn: Vec<f32> = buf_a.clone();
        let post_ln = self.f32_slice(&idx.post_attn_layernorm);
        blas_rmsnorm(buf_a, &post_ln, hidden, config.rms_norm_eps);

        buf_b.fill(0.0);
        buf_b.resize(hidden, 0.0);
        let (gp, gs) = self.packed(&idx.gate_proj);
        let (up, us) = self.packed(&idx.up_proj);
        let (dp, ds) = self.packed(&idx.down_proj);
        ternary_swiglu_ffn(buf_a, gp, gs, up, us, dp, ds, buf_b, gate_buf, up_buf, 1, hidden, inter);

        // residual add → buf_a
        for i in 0..hidden {
            buf_a[i] = residual_ffn[i] + buf_b[i];
        }
    }

    /// FullAttn incremental (JIT ternary)。buf_a: 入力兼出力。
    #[allow(clippy::too_many_arguments)]
    fn jit_fullattn_incremental(
        &self,
        buf_a: &mut Vec<f32>,
        buf_b: &mut Vec<f32>,
        idx: &MmapFullAttn,
        config: &Qwen35Config,
        k_cache: &mut Vec<f32>,
        v_cache: &mut Vec<f32>,
        position: usize,
        gate_buf: &mut [f32],
        up_buf: &mut [f32],
    ) {
        use crate::blas::{blas_rmsnorm, ternary_matmul_bt, ternary_swiglu_ffn};
        use crate::deltanet::qk_norm;

        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let rotary_dim = config.rotary_dim();
        let inter = config.intermediate_size;

        let residual: Vec<f32> = buf_a.clone();

        let ln = self.f32_slice(&idx.input_layernorm);
        blas_rmsnorm(buf_a, &ln, hidden, config.rms_norm_eps);

        // QKV projections — JIT ternary
        let mut q = vec![0.0f32; num_heads * head_dim];
        let mut k_new = vec![0.0f32; num_kv_heads * head_dim];
        let mut v_new = vec![0.0f32; num_kv_heads * head_dim];

        let (p, s) = self.packed(&idx.q_proj);
        ternary_matmul_bt(buf_a, p, s, &mut q, 1, num_heads * head_dim, hidden);
        let (p, s) = self.packed(&idx.k_proj);
        ternary_matmul_bt(buf_a, p, s, &mut k_new, 1, num_kv_heads * head_dim, hidden);
        let (p, s) = self.packed(&idx.v_proj);
        ternary_matmul_bt(buf_a, p, s, &mut v_new, 1, num_kv_heads * head_dim, hidden);

        let q_norm_w = self.f32_slice(&idx.q_norm);
        let k_norm_w = self.f32_slice(&idx.k_norm);
        qk_norm(&mut q, &q_norm_w, num_heads, head_dim, config.rms_norm_eps);
        qk_norm(&mut k_new, &k_norm_w, num_kv_heads, head_dim, config.rms_norm_eps);

        apply_rope_with_offset(&mut q, num_heads, head_dim, rotary_dim, position, config.rope_theta);
        apply_rope_with_offset(&mut k_new, num_kv_heads, head_dim, rotary_dim, position, config.rope_theta);

        // KV cache grow (遅延確保)
        k_cache.extend_from_slice(&k_new);
        v_cache.extend_from_slice(&v_new);

        let cache_len = k_cache.len() / (num_kv_heads * head_dim);
        let kv_groups = num_heads / num_kv_heads;
        let scale = (head_dim as f32).sqrt().recip();
        let mut attn_out_raw = vec![0.0f32; num_heads * head_dim];

        for h in 0..num_heads {
            let kv_h = h / kv_groups;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let out_h = &mut attn_out_raw[h * head_dim..(h + 1) * head_dim];

            let mut scores = vec![0.0f32; cache_len];
            for t in 0..cache_len {
                let k_t = &k_cache[(t * num_kv_heads + kv_h) * head_dim..
                                   (t * num_kv_heads + kv_h + 1) * head_dim];
                let dot: f32 = q_h.iter().zip(k_t.iter()).map(|(a, b)| a * b).sum();
                scores[t] = dot * scale;
            }

            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores { *s = (*s - max_s).exp(); sum += *s; }
            let inv_sum = sum.recip();
            for s in &mut scores { *s *= inv_sum; }

            for t in 0..cache_len {
                let v_t = &v_cache[(t * num_kv_heads + kv_h) * head_dim..
                                   (t * num_kv_heads + kv_h + 1) * head_dim];
                let w = scores[t];
                for d in 0..head_dim { out_h[d] += w * v_t[d]; }
            }
        }

        // O projection — JIT ternary
        buf_b.fill(0.0);
        buf_b.resize(hidden, 0.0);
        let (p, s) = self.packed(&idx.o_proj);
        ternary_matmul_bt(&attn_out_raw, p, s, buf_b, 1, hidden, num_heads * head_dim);

        for i in 0..hidden { buf_a[i] = residual[i] + buf_b[i]; }

        // FFN — JIT ternary SwiGLU
        let residual_ffn: Vec<f32> = buf_a.clone();
        let post_ln = self.f32_slice(&idx.post_attn_layernorm);
        blas_rmsnorm(buf_a, &post_ln, hidden, config.rms_norm_eps);

        buf_b.fill(0.0);
        buf_b.resize(hidden, 0.0);
        let (gp, gs) = self.packed(&idx.gate_proj);
        let (up, us) = self.packed(&idx.up_proj);
        let (dp, ds) = self.packed(&idx.down_proj);
        ternary_swiglu_ffn(buf_a, gp, gs, up, us, dp, ds, buf_b, gate_buf, up_buf, 1, hidden, inter);

        for i in 0..hidden { buf_a[i] = residual_ffn[i] + buf_b[i]; }
    }

    /// キャッシュ付きストリーミング生成。
    ///
    /// # Errors
    ///
    /// ファイル I/O エラー時。
    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        gen_config: &GenerationConfig,
        eos_token_id: u32,
    ) -> io::Result<Vec<u32>> {
        let mut cache = self.create_cache();

        eprintln!("  [mmap] prefill {} tokens...", prompt_ids.len());
        for (i, &tok) in prompt_ids.iter().enumerate() {
            let _ = self.forward_incremental_streaming(tok, &mut cache)?;
            if (i + 1) % 10 == 0 || i == prompt_ids.len() - 1 {
                eprintln!("    prefill: {}/{}", i + 1, prompt_ids.len());
            }
        }

        eprintln!("  [mmap] generating...");
        let mut generated = Vec::new();
        let mut all_ids: Vec<u32> = prompt_ids.to_vec();
        let mut last_token = *prompt_ids.last().unwrap_or(&0);

        for step in 0..gen_config.max_tokens {
            let mut logits = self.forward_incremental_streaming(last_token, &mut cache)?;

            if (gen_config.repetition_penalty - 1.0).abs() > f32::EPSILON {
                apply_repetition_penalty(&mut logits, &all_ids, gen_config.repetition_penalty);
            }

            let next_token = if gen_config.temperature <= 0.0 {
                argmax(&logits)
            } else {
                sample_top_k(&mut logits, gen_config.temperature, gen_config.top_k)
            };

            if next_token == eos_token_id {
                eprintln!("    EOS at step {step}");
                break;
            }

            generated.push(next_token);
            all_ids.push(next_token);
            last_token = next_token;
        }

        Ok(generated)
    }

    /// 1トークンずつストリーミング生成し、コールバックで出力。
    ///
    /// # Errors
    ///
    /// ファイル I/O エラー時。
    pub fn generate_streaming_callback<F>(
        &self,
        prompt_ids: &[u32],
        gen_config: &GenerationConfig,
        eos_token_id: u32,
        tokenizer: &BpeTokenizer,
        mut callback: F,
    ) -> io::Result<()>
    where
        F: FnMut(&str) -> bool,
    {
        let mut cache = self.create_cache();

        for &tok in prompt_ids {
            let _ = self.forward_incremental_streaming(tok, &mut cache)?;
        }

        let mut all_ids: Vec<u32> = prompt_ids.to_vec();
        let mut last_token = *prompt_ids.last().unwrap_or(&0);

        for _ in 0..gen_config.max_tokens {
            let mut logits = self.forward_incremental_streaming(last_token, &mut cache)?;

            if (gen_config.repetition_penalty - 1.0).abs() > f32::EPSILON {
                apply_repetition_penalty(&mut logits, &all_ids, gen_config.repetition_penalty);
            }

            let next_token = if gen_config.temperature <= 0.0 {
                argmax(&logits)
            } else {
                sample_top_k(&mut logits, gen_config.temperature, gen_config.top_k)
            };

            if next_token == eos_token_id { break; }

            let text = tokenizer.decode(&[next_token]);
            if !callback(&text) { break; }

            all_ids.push(next_token);
            last_token = next_token;
        }

        Ok(())
    }
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

    // ── InferenceCache / incremental 関連テスト ──────────────────────────────

    /// apply_rope_with_offset: position=0 では角度0なので入力そのまま。
    #[test]
    fn test_rope_offset_position_zero() {
        let mut x = vec![1.0f32, 0.0, 0.0, 1.0]; // 1ヘッド, head_dim=4, rotary_dim=4
        apply_rope_with_offset(&mut x, 1, 4, 4, 0, 10000.0);
        // position=0 → cos(0)=1, sin(0)=0 → 変化なし
        assert!((x[0] - 1.0).abs() < 1e-5, "x[0]={}", x[0]);
        assert!((x[1] - 0.0).abs() < 1e-5, "x[1]={}", x[1]);
        assert!((x[2] - 0.0).abs() < 1e-5, "x[2]={}", x[2]);
        assert!((x[3] - 1.0).abs() < 1e-5, "x[3]={}", x[3]);
    }

    /// apply_rope_with_offset: 回転は逆方向に適用すると元に戻る。
    #[test]
    fn test_rope_offset_invertible() {
        let original = vec![1.0f32, 0.5, -0.5, 0.3];
        let mut x = original.clone();
        let theta = 10_000.0f32;
        let pos = 7usize;
        // 順方向 (pos)
        apply_rope_with_offset(&mut x, 1, 4, 4, pos, theta);
        // 逆方向 (negating angle = applying at -pos angle via negation of sin)
        // 逆回転: angle -> -angle => pos=0 で sin=0/cos=1, or apply pos forward then
        // verify norm is preserved (rotation preserves L2 norm)
        let norm_before: f32 = original.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-4,
            "norm変化: before={norm_before} after={norm_after}");
    }

    /// apply_rope_with_offset: 複数ヘッドでも各ヘッドが独立に回転される。
    #[test]
    fn test_rope_offset_multi_head() {
        // 2ヘッド, head_dim=4, rotary_dim=4
        let mut x = vec![1.0f32, 0.0, 0.0, 1.0, // head 0
                         1.0, 0.0, 0.0, 1.0]; // head 1
        apply_rope_with_offset(&mut x, 2, 4, 4, 0, 10000.0);
        // position=0 → 変化なし (両ヘッド)
        for i in 0..8 {
            let expected = if i % 4 == 0 || i % 4 == 3 { 1.0f32 } else { 0.0f32 };
            assert!((x[i] - expected).abs() < 1e-5, "x[{i}]={}", x[i]);
        }
    }

    /// apply_rope_with_offset: position が増えると角度が変化する。
    #[test]
    fn test_rope_offset_position_changes() {
        let x_pos0 = {
            let mut x = vec![1.0f32, 0.0, 0.0, 1.0];
            apply_rope_with_offset(&mut x, 1, 4, 4, 0, 10000.0);
            x
        };
        let x_pos5 = {
            let mut x = vec![1.0f32, 0.0, 0.0, 1.0];
            apply_rope_with_offset(&mut x, 1, 4, 4, 5, 10000.0);
            x
        };
        // position 0 と 5 では結果が異なる
        let diff: f32 = x_pos0.iter().zip(x_pos5.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "position 0 と 5 の出力が同じ: diff={diff}");
    }
}
