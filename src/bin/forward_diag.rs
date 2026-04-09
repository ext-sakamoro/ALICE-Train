//! Forward Pass 診断 — .alice モデルの logits 統計 + loss 確認。
//!
//! 目的: forward が正しく動作しているか (loss < random=12.42 になるべき) を確認。
//!
//! ```bash
//! cargo run --release --features qat-cli --bin forward-diag -- \
//!     --model models/ALICE-Cognitive-9B-Ternary.alice \
//!     --tokenizer models/Qwen--Qwen3.5-9B/tokenizer.json
//! ```

use clap::Parser;
use std::time::Instant;

use alice_train::inference::StreamingAliceModel;
use alice_train::qwen35::LayerType;
use alice_train::tokenizer::BpeTokenizer;

#[derive(Parser)]
#[command(about = "Forward pass 診断 — .alice モデル logits 統計")]
struct Cli {
    /// .alice モデルファイル。
    #[arg(long)]
    model: String,

    /// tokenizer.json パス。
    #[arg(long)]
    tokenizer: String,

    /// テストプロンプト。
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,

    /// 生成トークン数。
    #[arg(long, default_value = "8")]
    max_tokens: usize,
}

/// テンソルの統計。
fn tensor_stats(data: &[f32]) -> (f32, f32, f32, f32, usize, usize) {
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    for &v in data {
        if v.is_nan() { nan_count += 1; continue; }
        if v.is_infinite() { inf_count += 1; continue; }
        sum += v as f64;
        sum_sq += (v as f64) * (v as f64);
        if v < min { min = v; }
        if v > max { max = v; }
    }
    let n = (data.len() - nan_count - inf_count) as f64;
    let mean = if n > 0.0 { (sum / n) as f32 } else { 0.0 };
    let std = if n > 1.0 { ((sum_sq / n - (sum / n).powi(2)).max(0.0).sqrt()) as f32 } else { 0.0 };
    (mean, std, min, max, nan_count, inf_count)
}

/// Cross-entropy loss。
fn cross_entropy_loss(logits: &[f32], target: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&x| ((x - max) as f64).exp()).sum();
    let log_prob = (logits[target] - max) as f64 - sum_exp.ln();
    -log_prob as f32
}

/// Top-k tokens from logits。
fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

fn main() {
    let cli = Cli::parse();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Forward Pass 診断 — .alice streaming mode             ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // トークナイザー読み込み
    let tokenizer = BpeTokenizer::from_file(&cli.tokenizer).unwrap_or_else(|e| {
        eprintln!("[ERROR] トークナイザー読み込み失敗: {e}");
        std::process::exit(1);
    });
    println!("  tokenizer vocab: {}", tokenizer.vocab_size());

    // モデル読み込み (mmap + JIT)
    let t0 = Instant::now();
    println!("  モデル: {}", cli.model);
    let model = StreamingAliceModel::from_file(&cli.model).unwrap_or_else(|e| {
        eprintln!("[ERROR] モデル読み込み失敗: {e}");
        std::process::exit(1);
    });
    println!("  読み込み: {:.1}s", t0.elapsed().as_secs_f64());

    let config = model.config();
    let vocab_size = config.vocab_size;
    let random_loss = (vocab_size as f64).ln();

    println!("  vocab_size: {vocab_size}");
    println!("  hidden_size: {}", config.hidden_size);
    println!("  layers: {} ({} DeltaNet + {} FullAttn)",
        config.num_hidden_layers,
        config.layer_types.iter().filter(|t| **t == LayerType::LinearAttention).count(),
        config.layer_types.iter().filter(|t| **t == LayerType::FullAttention).count(),
    );
    println!("  quantization: {}", model.meta.quantization);
    println!("  source: step {}, loss {:.4}", model.meta.source_step, model.meta.source_loss);
    println!("  random baseline loss: {random_loss:.4}");
    println!();

    // プロンプトトークン化
    let prompt_ids = tokenizer.encode(&cli.prompt);
    println!("━━━ プロンプト ━━━");
    println!("  text: \"{}\"", cli.prompt);
    println!("  tokens ({}):", prompt_ids.len());
    for (i, &tok) in prompt_ids.iter().enumerate() {
        let decoded = tokenizer.decode(&[tok]);
        println!("    [{i}] {tok} → \"{decoded}\"");
    }
    println!();

    // Incremental forward — 1トークンずつ処理
    println!("━━━ Incremental Forward (1 token at a time) ━━━");
    let mut cache = model.create_cache();

    // プロンプト処理 (prefill)
    println!("  Prefill ({} tokens):", prompt_ids.len());
    for (i, &tok) in prompt_ids.iter().enumerate() {
        let t_tok = Instant::now();
        let logits = model.forward_incremental_streaming(tok, &mut cache)
            .unwrap_or_else(|e| {
                eprintln!("  [ERROR] token {i} forward失敗: {e}");
                std::process::exit(1);
            });
        let elapsed_ms = t_tok.elapsed().as_millis();

        let (mean, std, min, max, nans, infs) = tensor_stats(&logits);
        let next_target = prompt_ids.get(i + 1).copied().unwrap_or(0) as usize;
        let loss = cross_entropy_loss(&logits, next_target % vocab_size);
        let top5 = top_k(&logits, 5);

        println!("    tok[{i}]={tok:>6} | logits: mean={mean:.4} std={std:.4} min={min:.2} max={max:.2} NaN={nans} Inf={infs} | loss={loss:.4} | {elapsed_ms}ms");
        if i == prompt_ids.len() - 1 || nans > 0 || infs > 0 {
            println!("      top5: {:?}", top5.iter().map(|(id, s)| {
                let t = tokenizer.decode(&[*id as u32]);
                format!("{id}(\"{t}\")={s:.2}")
            }).collect::<Vec<_>>());
        }

        if nans > 0 || infs > 0 {
            eprintln!("  *** 異常: NaN={nans} Inf={infs} — forward pass 破損 ***");
        }
    }

    // 生成 (autoregressive)
    println!();
    println!("  Generation ({} tokens):", cli.max_tokens);
    let mut last_token = *prompt_ids.last().unwrap_or(&1);
    let mut generated_tokens = Vec::new();
    let mut total_gen_loss = 0.0f32;

    for i in 0..cli.max_tokens {
        let t_tok = Instant::now();
        let logits = model.forward_incremental_streaming(last_token, &mut cache)
            .unwrap_or_else(|e| {
                eprintln!("  [ERROR] gen token {i} forward失敗: {e}");
                std::process::exit(1);
            });
        let elapsed_ms = t_tok.elapsed().as_millis();

        // Greedy
        let next_id = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id as u32)
            .unwrap_or(0);

        let (mean, std, _min, _max, nans, infs) = tensor_stats(&logits);
        let loss = cross_entropy_loss(&logits, next_id as usize);
        total_gen_loss += loss;

        let decoded = tokenizer.decode(&[next_id]);
        println!("    gen[{i}] → {next_id:>6} \"{decoded}\" | logits: mean={mean:.4} std={std:.4} NaN={nans} Inf={infs} | loss={loss:.4} | {elapsed_ms}ms");

        generated_tokens.push(next_id);
        last_token = next_id;

        if next_id == tokenizer.eos_token_id {
            println!("    [EOS]");
            break;
        }
    }

    // サマリー
    let gen_count = generated_tokens.len().max(1);
    let avg_gen_loss = total_gen_loss / gen_count as f32;
    let full_text = tokenizer.decode(&generated_tokens);

    println!();
    println!("━━━ 結果 ━━━");
    println!("  生成テキスト: \"{full_text}\"");
    println!("  生成時 平均loss: {avg_gen_loss:.4}");
    println!("  ランダム baseline: {random_loss:.4}");
    println!();

    if avg_gen_loss < random_loss as f32 {
        println!("  ✓ ランダムより良い — モデルは何かを学習している");
    } else {
        println!("  ✗ ランダムより悪い — forward pass or 量子化にバグ");
        println!();
        println!("  考えられる原因:");
        println!("    1. ternary quantization で重みが壊れている (export時の scale/pack バグ)");
        println!("    2. forward pass 実装のバグ (DeltaNet 再帰, RoPE, RMSNorm 等)");
        println!("    3. safetensors → FP32 → ternary の変換でテンソル名マッピング不一致");
        println!("    4. embedding/lm_head は量子化対象外のはずが、量子化されている");
    }
}
