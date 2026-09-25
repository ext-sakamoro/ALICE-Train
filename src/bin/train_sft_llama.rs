//! Llama 系アーキテクチャの LoRA SFT 学習 CLI。
//!
//! base 重みを凍結し、[`alice_train::lora`] の低ランクアダプタだけを AdamW で学習する。
//! 対象は `LlamaForCausalLM` 互換 (MiniCPM5-2B / Llama-3 系)。
//!
//! # なぜ `model_forward` を使わないか
//!
//! [`alice_train::llama_forward::model_forward`] は出力 RMSNorm を in-place で掛けた後の
//! hidden しか返さないが、norm の backward には **norm 前の hidden** が要る。
//! そのため forward をここで展開し、pre-norm hidden を保持する。
//!
//! # メモリ
//!
//! merged 重み (`W + s·B·A`) を全層分持つと base と同量 (2.5B で約 10 GB) 増えるため、
//! **1 層分の scratch を使い回す**。forward と backward で各 1 回ずつ merge する。
//!
//! # 使い方
//!
//! ```bash
//! cargo run --release --features qat-cuda --bin train-sft-llama -- \
//!     --model-dir models/MiniCPM5-2B \
//!     --data data/lol_pairs/train.jsonl \
//!     --out checkpoints/lol_sft
//! ```

use alice_train::llama::{LlamaConfig, LlamaLayerWeights};
use alice_train::llama_backward::{layer_backward, rmsnorm_backward};
use alice_train::llama_forward::{layer_forward, matmul, matmul_bt, rmsnorm};
use alice_train::lora::{LoraConfig, LoraLayer, LoraLayerGrads};
use alice_train::safetensors_loader::ShardedModel;
use alice_train::tokenizer::BpeTokenizer;
use clap::Parser;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// LoRA SFT CLI 引数。
#[derive(Parser, Debug)]
#[command(about = "LoRA SFT for Llama-compatible models (base weights frozen)")]
struct Args {
    /// HF 形式のモデルディレクトリ (config.json / tokenizer.json / *.safetensors)
    #[arg(long)]
    model_dir: PathBuf,
    /// 学習データ JSONL (`caption_en` と `lol` を持つ行)
    #[arg(long)]
    data: PathBuf,
    /// チェックポイント出力ディレクトリ
    #[arg(long)]
    out: PathBuf,
    /// LoRA rank
    #[arg(long, default_value_t = 16)]
    rank: usize,
    /// LoRA alpha
    #[arg(long, default_value_t = 32.0)]
    alpha: f32,
    /// 学習率
    #[arg(long, default_value_t = 1e-4)]
    lr: f32,
    /// weight decay (AdamW)
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f32,
    /// epoch 数
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    /// 勾配累積ステップ数 (実効 batch size)
    #[arg(long, default_value_t = 8)]
    grad_accum: usize,
    /// 最大系列長 (超過サンプルは捨てる)
    #[arg(long, default_value_t = 512)]
    max_seq: usize,
    /// 先頭 N 件だけ使う (smoke 用、0 で全件)
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// 何ステップごとにログを出すか
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    /// 何ステップごとに checkpoint を書くか
    #[arg(long, default_value_t = 200)]
    save_every: usize,
    /// 既存 checkpoint から再開する
    #[arg(long, default_value_t = false)]
    resume: bool,
}

/// `llm_bench` (`alice-lol/examples/llm_bench.rs`) と逐語一致させる system prompt。
///
/// baseline (grammar 7/20 / think 9/20) はこの prompt で測られているため、
/// SFT 側で書式を変えると測定差分が SFT の効果に帰属できなくなる。
const SYSTEM_PROMPT: &str = "You write ALICE-LOL, a tiny SDF DSL. Output exactly one expression and nothing else: no comments, no indentation, single spaces only, no trailing text.";

/// 学習サンプル 1 件。
struct Sample {
    /// prompt + completion の token 列。
    tokens: Vec<u32>,
    /// completion (= LOL 式) の開始位置。ここより前の target には loss を乗せない。
    completion_start: usize,
}

/// 数値安定な cross-entropy loss と logits 勾配。
///
/// `src/bin/train_qat_qwen35.rs` の同名関数と同じ定式 (softmax - onehot)。
fn cross_entropy_loss(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();
    let loss = -(probs[target].max(1e-10)).ln();
    let mut grad = probs;
    grad[target] -= 1.0;
    (loss, grad)
}

/// HF `config.json` を [`LlamaConfig`] に写す。
fn load_config(model_dir: &Path) -> std::io::Result<LlamaConfig> {
    let text = fs::read_to_string(model_dir.join("config.json"))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let get_usize = |k: &str| -> usize {
        v.get(k)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_else(|| panic!("config.json に {k} がない")) as usize
    };
    let hidden_dim = get_usize("hidden_size");
    let num_heads = get_usize("num_attention_heads");
    // head_dim は明示されていれば優先 (MiniCPM5 は hidden/heads と一致しないモデルもある)
    let head_dim = v
        .get("head_dim")
        .and_then(serde_json::Value::as_u64)
        .map_or(hidden_dim / num_heads, |x| x as usize);
    let attention_bias = v
        .get("attention_bias")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    Ok(LlamaConfig {
        vocab_size: get_usize("vocab_size"),
        hidden_dim,
        intermediate_dim: get_usize("intermediate_size"),
        num_heads,
        num_kv_heads: get_usize("num_key_value_heads"),
        num_layers: get_usize("num_hidden_layers"),
        max_seq_len: get_usize("max_position_embeddings"),
        head_dim,
        #[allow(clippy::cast_possible_truncation)]
        rope_theta: v
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .expect("config.json に rope_theta がない") as f32,
        #[allow(clippy::cast_possible_truncation)]
        norm_eps: v
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .expect("config.json に rms_norm_eps がない") as f32,
        attention_bias,
    })
}

/// ChatML 組み立て。`llm_bench.rs:405` の `chatml()` と同じ文字列になるよう token 単位で構成する。
///
/// `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{lol}<|im_end|>`
///
/// BOS は付けない (`llm_bench` が付けていないため。ここを変えると baseline と比較できない)。
fn build_sample(
    tok: &BpeTokenizer,
    im_start: u32,
    im_end: u32,
    caption: &str,
    lol: &str,
) -> Sample {
    let mut t = Vec::with_capacity(256);
    t.push(im_start);
    t.extend(tok.encode(&format!("system\n{SYSTEM_PROMPT}")));
    t.push(im_end);
    t.extend(tok.encode("\n"));
    t.push(im_start);
    t.extend(tok.encode(&format!("user\n{caption}")));
    t.push(im_end);
    t.extend(tok.encode("\n"));
    t.push(im_start);
    t.extend(tok.encode("assistant\n"));
    let completion_start = t.len();
    t.extend(tok.encode(lol));
    t.push(im_end);
    Sample {
        tokens: t,
        completion_start,
    }
}

/// datagen の JSONL を読み、ChatML 化した学習サンプルにする。
fn load_samples(
    path: &Path,
    tok: &BpeTokenizer,
    im_start: u32,
    im_end: u32,
    max_seq: usize,
    limit: usize,
) -> std::io::Result<(Vec<Sample>, usize)> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    let mut skipped_too_long = 0usize;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  JSONL 行を skip (parse 失敗): {e}");
                continue;
            }
        };
        let (Some(caption), Some(lol)) = (
            v.get("caption_en").and_then(serde_json::Value::as_str),
            v.get("lol").and_then(serde_json::Value::as_str),
        ) else {
            eprintln!("  JSONL 行を skip (caption_en / lol が無い)");
            continue;
        };
        let s = build_sample(tok, im_start, im_end, caption, lol);
        if s.tokens.len() > max_seq {
            skipped_too_long += 1;
            continue;
        }
        out.push(s);
        if limit > 0 && out.len() >= limit {
            break;
        }
    }
    Ok((out, skipped_too_long))
}

/// LoRA パラメータ 1 本分の AdamW 状態。
struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamState {
    fn zeros(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    /// AdamW 更新 (bias correction 込み)。
    fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32, wd: f32, t: i32, scale: f32) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        let bc1 = 1.0 - B1.powi(t);
        let bc2 = 1.0 - B2.powi(t);
        for (((p, g), m), v) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(self.m.iter_mut())
            .zip(self.v.iter_mut())
        {
            let g = g * scale;
            *m = B1 * *m + (1.0 - B1) * g;
            *v = B2 * *v + (1.0 - B2) * g * g;
            let m_hat = *m / bc1;
            let v_hat = *v / bc2;
            *p -= lr * (m_hat / (v_hat.sqrt() + EPS) + wd * *p);
        }
    }
}

/// 1 レイヤー分の AdamW 状態 (7 projection × A/B)。
struct LayerAdam {
    states: Vec<AdamState>,
}

impl LayerAdam {
    fn zeros(layer: &LoraLayer) -> Self {
        let mut states = Vec::with_capacity(14);
        for ad in [
            &layer.q,
            &layer.k,
            &layer.v,
            &layer.o,
            &layer.gate,
            &layer.up,
            &layer.down,
        ] {
            states.push(AdamState::zeros(ad.a.len()));
            states.push(AdamState::zeros(ad.b.len()));
        }
        Self { states }
    }
}

/// 7 projection を固定順で列挙する (AdamW 状態との対応を 1 箇所に閉じる)。
macro_rules! for_each_adapter {
    ($layer:expr, $grads:expr, $adam:expr, |$ad:ident, $ga:ident, $gb:ident, $sa:ident, $sb:ident| $body:block) => {{
        let l = $layer;
        let g = $grads;
        let a = $adam;
        let mut idx = 0usize;
        let pairs: [(&mut _, &_); 7] = [
            (&mut l.q, &g.q),
            (&mut l.k, &g.k),
            (&mut l.v, &g.v),
            (&mut l.o, &g.o),
            (&mut l.gate, &g.gate),
            (&mut l.up, &g.up),
            (&mut l.down, &g.down),
        ];
        for (ad_ref, gr) in pairs {
            let (left, right) = a.states.split_at_mut(idx + 1);
            let $sa = &mut left[idx];
            let $sb = &mut right[0];
            let $ad = ad_ref;
            let $ga = &gr.d_a;
            let $gb = &gr.d_b;
            $body
            idx += 2;
        }
    }};
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let t_start = Instant::now();

    println!("[sft] model_dir = {}", args.model_dir.display());
    let config = load_config(&args.model_dir)?;
    println!(
        "[sft] config: layers {} hidden {} inter {} heads {}/{} head_dim {} vocab {} rope_theta {} eps {}",
        config.num_layers,
        config.hidden_dim,
        config.intermediate_dim,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        config.vocab_size,
        config.rope_theta,
        config.norm_eps
    );

    // ── tokenizer + 特殊 token ──
    let tok = BpeTokenizer::from_file(args.model_dir.join("tokenizer.json"))?;
    let im_start = tok
        .token_id("<|im_start|>")
        .expect("tokenizer に <|im_start|> がない (ChatML 非対応の vocab)");
    let im_end = tok
        .token_id("<|im_end|>")
        .expect("tokenizer に <|im_end|> がない (ChatML 非対応の vocab)");
    println!(
        "[sft] im_start = {im_start}, im_end = {im_end}, vocab = {}",
        tok.vocab_size()
    );

    // ── データ ──
    let (samples, skipped) =
        load_samples(&args.data, &tok, im_start, im_end, args.max_seq, args.limit)?;
    assert!(!samples.is_empty(), "学習サンプルが 0 件");
    let total_completion_tokens: usize = samples
        .iter()
        .map(|s| s.tokens.len().saturating_sub(s.completion_start))
        .sum();
    println!(
        "[sft] samples {} (max_seq 超過で skip {}), completion token 合計 {}",
        samples.len(),
        skipped,
        total_completion_tokens
    );
    // 書式を目視できるよう先頭 1 件を decode して出す (log-first)
    let head = &samples[0];
    println!(
        "[sft] sample[0] prompt = {:?}",
        tok.decode(&head.tokens[..head.completion_start])
    );
    println!(
        "[sft] sample[0] completion = {:?}",
        tok.decode(&head.tokens[head.completion_start..])
    );

    // ── base 重み (凍結) ──
    println!("[sft] loading safetensors ...");
    let model = ShardedModel::open(&args.model_dir)?;
    let get = |name: &str| model.get_tensor_f32(name);
    let embedding = get("model.embed_tokens.weight").expect("embed_tokens が無い");
    let output_norm = get("model.norm.weight").expect("model.norm.weight が無い");
    // tie_word_embeddings=false のモデルは lm_head を別に持つ
    let output_proj = get("lm_head.weight").unwrap_or_else(|| {
        println!("[sft] lm_head.weight が無いので embed_tokens を tied として使う");
        embedding.clone()
    });
    let mut base_layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let w = LlamaLayerWeights::from_tensors(i, &get, &config)
            .unwrap_or_else(|| panic!("layer {i} の重みが揃っていない"));
        base_layers.push(w);
    }
    println!(
        "[sft] base 重み load 完了 ({:.1}s), 凍結 param {:.2} B",
        t_start.elapsed().as_secs_f32(),
        config.total_params() as f64 / 1e9
    );

    // ── LoRA ──
    let lora_cfg = LoraConfig::try_new(args.rank, args.alpha)
        .unwrap_or_else(|e| panic!("LoRA config が不正: {e}"));
    let mut lora: Vec<LoraLayer> = (0..config.num_layers)
        .map(|i| LoraLayer::new(&config, lora_cfg, i))
        .collect();
    let mut lora_grads: Vec<LoraLayerGrads> = lora.iter().map(LoraLayerGrads::zeros).collect();
    let mut adam: Vec<LayerAdam> = lora.iter().map(LayerAdam::zeros).collect();
    let trainable: usize = lora.iter().map(LoraLayer::num_params).sum();
    println!(
        "[sft] LoRA rank {} alpha {} scale {:.3} → 学習対象 {:.2} M param ({:.2} % of base)",
        lora_cfg.rank(),
        lora_cfg.alpha(),
        lora_cfg.scale(),
        trainable as f64 / 1e6,
        trainable as f64 / config.total_params() as f64 * 100.0
    );

    fs::create_dir_all(&args.out)?;
    let ckpt_path = args.out.join("lora_adapter.bin");
    let mut global_step: u64 = 0;
    if args.resume && ckpt_path.exists() {
        global_step = load_adapter(&ckpt_path, &mut lora)?;
        println!("[sft] resume: step {global_step} から再開");
    }

    // merge 用 scratch (1 層分だけ確保して使い回す = 全層 merge より約 10 GB 節約)
    let mut merged = base_layers[0].clone();

    let mut log_file = fs::File::create(args.out.join("train.log"))?;
    let mut accum_in_batch = 0usize;
    let mut running_loss = 0.0f32;
    let mut running_tokens = 0usize;
    let mut adam_t: i32 = 0;
    let step_timer = Instant::now();

    for epoch in 0..args.epochs {
        for (si, sample) in samples.iter().enumerate() {
            let seq_len = sample.tokens.len();
            if seq_len < 2 {
                continue;
            }

            // ── forward (pre-norm hidden を保持するため展開) ──
            let mut hidden = vec![0.0f32; seq_len * config.hidden_dim];
            for (t, &tid) in sample.tokens.iter().enumerate() {
                let tid = tid as usize;
                assert!(tid < config.vocab_size, "token id {tid} が vocab 外");
                hidden[t * config.hidden_dim..(t + 1) * config.hidden_dim].copy_from_slice(
                    &embedding[tid * config.hidden_dim..(tid + 1) * config.hidden_dim],
                );
            }
            let mut caches = Vec::with_capacity(config.num_layers);
            for l in 0..config.num_layers {
                lora[l].merge_into(&base_layers[l], &mut merged);
                caches.push(layer_forward(&mut hidden, &merged, &config, seq_len));
            }
            let pre_norm = hidden.clone();
            rmsnorm(
                &mut hidden,
                &output_norm,
                config.hidden_dim,
                config.norm_eps,
            );

            // ── completion 位置だけ loss / 勾配 ──
            //
            // logits と d_normed は行をまとめて 2 回の matmul にする。
            // token ごとに回すと vocab × hidden の演算が自前 scalar ループになり
            // BLAS / CUDA dispatch (`blas.rs`) に乗らない。
            let rows: Vec<usize> = (0..seq_len - 1)
                .filter(|t| t + 1 >= sample.completion_start)
                .collect();
            if rows.is_empty() {
                continue;
            }
            let hd = config.hidden_dim;
            let vocab = config.vocab_size;
            let mut d_normed = vec![0.0f32; seq_len * hd];
            let mut sample_loss = 0.0f32;
            let mut sample_tokens = 0usize;
            // 1 チャンクあたり最大 64 行 (vocab 13 万 × f32 なので行数に比例して重い)
            for chunk in rows.chunks(64) {
                let n = chunk.len();
                let mut h_sel = vec![0.0f32; n * hd];
                for (i, &t) in chunk.iter().enumerate() {
                    h_sel[i * hd..(i + 1) * hd].copy_from_slice(&hidden[t * hd..(t + 1) * hd]);
                }
                let mut logits = vec![0.0f32; n * vocab];
                matmul_bt(&h_sel, &output_proj, &mut logits, n, vocab, hd);

                let mut d_logits = vec![0.0f32; n * vocab];
                for (i, &t) in chunk.iter().enumerate() {
                    let target = sample.tokens[t + 1] as usize;
                    let (loss, g) = cross_entropy_loss(&logits[i * vocab..(i + 1) * vocab], target);
                    sample_loss += loss;
                    sample_tokens += 1;
                    d_logits[i * vocab..(i + 1) * vocab].copy_from_slice(&g);
                }

                // d_sel = d_logits × output_proj   (n × vocab)·(vocab × hidden)
                let mut d_sel = vec![0.0f32; n * hd];
                matmul(&d_logits, &output_proj, &mut d_sel, n, hd, vocab);
                for (i, &t) in chunk.iter().enumerate() {
                    d_normed[t * hd..(t + 1) * hd].copy_from_slice(&d_sel[i * hd..(i + 1) * hd]);
                }
            }
            if sample_tokens == 0 {
                continue;
            }

            // ── backward ──
            let mut d_hidden = vec![0.0f32; seq_len * config.hidden_dim];
            let mut d_out_norm = vec![0.0f32; config.hidden_dim]; // 凍結なので捨てる
            rmsnorm_backward(
                &d_normed,
                &pre_norm,
                &output_norm,
                &mut d_hidden,
                &mut d_out_norm,
                config.hidden_dim,
                config.norm_eps,
            );
            for l in (0..config.num_layers).rev() {
                lora[l].merge_into(&base_layers[l], &mut merged);
                let (d_in, grads) =
                    layer_backward(&d_hidden, &caches[l], &merged, &config, seq_len);
                lora[l].project_grads(&grads, &mut lora_grads[l]);
                d_hidden = d_in;
            }

            running_loss += sample_loss;
            running_tokens += sample_tokens;
            accum_in_batch += 1;

            // ── optimizer ──
            if accum_in_batch >= args.grad_accum {
                adam_t += 1;
                // token 数で正規化 (loss の平均と勾配のスケールを一致させる)
                let scale = 1.0 / running_tokens.max(1) as f32;
                for l in 0..config.num_layers {
                    let (layer, grads, st) = (&mut lora[l], &lora_grads[l], &mut adam[l]);
                    for_each_adapter!(layer, grads, st, |ad, ga, gb, sa, sb| {
                        sa.step(&mut ad.a, ga, args.lr, args.weight_decay, adam_t, scale);
                        sb.step(&mut ad.b, gb, args.lr, args.weight_decay, adam_t, scale);
                    });
                    lora_grads[l].zero_out();
                }
                global_step += 1;

                if global_step.is_multiple_of(args.log_every as u64) {
                    let avg = running_loss / running_tokens.max(1) as f32;
                    let line = format!(
                        "[sft] epoch {epoch} step {global_step} sample {}/{} loss {avg:.4} tokens {running_tokens} elapsed {:.1}s",
                        si + 1,
                        samples.len(),
                        step_timer.elapsed().as_secs_f32()
                    );
                    println!("{line}");
                    writeln!(log_file, "{line}")?;
                    log_file.flush()?;
                }
                running_loss = 0.0;
                running_tokens = 0;
                accum_in_batch = 0;

                if global_step.is_multiple_of(args.save_every as u64) {
                    save_adapter(&ckpt_path, &lora, global_step, lora_cfg)?;
                    println!(
                        "[sft] checkpoint 保存: {} (step {global_step})",
                        ckpt_path.display()
                    );
                }
            }
        }
    }

    save_adapter(&ckpt_path, &lora, global_step, lora_cfg)?;
    println!(
        "[sft] 完了: step {global_step}, 所要 {:.1}s, checkpoint {}",
        t_start.elapsed().as_secs_f32(),
        ckpt_path.display()
    );
    Ok(())
}

const ADAPTER_MAGIC: &[u8; 8] = b"ALORA001";

/// LoRA アダプタを保存する (base 重みは含まない)。
fn save_adapter(
    path: &Path,
    lora: &[LoraLayer],
    step: u64,
    cfg: LoraConfig,
) -> std::io::Result<()> {
    let tmp = path.with_extension("bin.tmp");
    let mut f = std::io::BufWriter::new(fs::File::create(&tmp)?);
    f.write_all(ADAPTER_MAGIC)?;
    f.write_all(&(cfg.rank() as u32).to_le_bytes())?;
    f.write_all(&cfg.alpha().to_le_bytes())?;
    f.write_all(&(lora.len() as u32).to_le_bytes())?;
    f.write_all(&step.to_le_bytes())?;
    for layer in lora {
        for ad in [
            &layer.q,
            &layer.k,
            &layer.v,
            &layer.o,
            &layer.gate,
            &layer.up,
            &layer.down,
        ] {
            for buf in [&ad.a, &ad.b] {
                f.write_all(&(buf.len() as u64).to_le_bytes())?;
                for x in buf {
                    f.write_all(&x.to_le_bytes())?;
                }
            }
        }
    }
    f.flush()?;
    drop(f);
    fs::rename(&tmp, path)?; // 書き込み途中の checkpoint を残さない
    Ok(())
}

/// 保存済みアダプタを読み戻し、`step` を返す。
fn load_adapter(path: &Path, lora: &mut [LoraLayer]) -> std::io::Result<u64> {
    let bytes = fs::read(path)?;
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    if bytes.len() < 24 || &bytes[..8] != ADAPTER_MAGIC {
        return Err(bad("adapter の magic が合わない"));
    }
    let num_layers = u32::from_le_bytes(bytes[16..20].try_into().expect("4 bytes")) as usize;
    if num_layers != lora.len() {
        return Err(bad("adapter の層数がモデルと合わない"));
    }
    let step = u64::from_le_bytes(bytes[20..28].try_into().expect("8 bytes"));
    let mut off = 28usize;
    for layer in lora.iter_mut() {
        for ad in [
            &mut layer.q,
            &mut layer.k,
            &mut layer.v,
            &mut layer.o,
            &mut layer.gate,
            &mut layer.up,
            &mut layer.down,
        ] {
            for buf in [&mut ad.a, &mut ad.b] {
                let n = u64::from_le_bytes(
                    bytes[off..off + 8]
                        .try_into()
                        .map_err(|_| bad("長さ欠損"))?,
                ) as usize;
                off += 8;
                if n != buf.len() || off + n * 4 > bytes.len() {
                    return Err(bad("adapter の形がモデルと合わない"));
                }
                for v in buf.iter_mut() {
                    *v = f32::from_le_bytes(
                        bytes[off..off + 4]
                            .try_into()
                            .map_err(|_| bad("f32 欠損"))?,
                    );
                    off += 4;
                }
            }
        }
    }
    Ok(step)
}
