//! ALICE-Cognitive Ternary モデルの Perplexity (PPL) 評価。
//!
//! テキスト生成は一切行わず、Forward パスのみで cross entropy loss を計測し、
//! PPL = exp(avg_loss) を算出する。
//!
//! ```bash
//! cargo run --release --features qat-cli --bin eval-ppl -- \
//!     --model models/ALICE-Cognitive-9B-Ternary.alice \
//!     --tokenizer models/Qwen--Qwen3.5-9B/tokenizer.json \
//!     --data data/qwen35/eval.bin \
//!     --seq-len 512 \
//!     --max-chunks 10
//! ```

use clap::Parser;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "eval-ppl", about = "ALICE Ternary モデルの Perplexity 計測")]
struct Args {
    /// .alice モデルファイル
    #[arg(long)]
    model: String,

    /// tokenizer.json パス
    #[arg(long)]
    tokenizer: String,

    /// 評価用トークンバイナリ (u32 LE)
    #[arg(long)]
    data: String,

    /// チャンクサイズ (トークン数)
    #[arg(long, default_value = "512")]
    seq_len: usize,

    /// 最大チャンク数 (0=全部)
    #[arg(long, default_value = "0")]
    max_chunks: usize,
}

fn main() {
    let args = Args::parse();

    println!("━━━ ALICE Perplexity 評価 ━━━");
    println!("  モデル: {}", args.model);
    println!("  データ: {}", args.data);
    println!("  seq_len: {}", args.seq_len);

    // トークンデータ読み込み
    let token_data = std::fs::read(&args.data).unwrap_or_else(|e| {
        eprintln!("データ読み込み失敗: {e}");
        std::process::exit(1);
    });
    let tokens: Vec<u32> = token_data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    println!("  トークン数: {}", tokens.len());

    let num_chunks = if args.seq_len + 1 > tokens.len() {
        0
    } else {
        (tokens.len() - 1) / args.seq_len
    };
    let num_chunks = if args.max_chunks > 0 {
        num_chunks.min(args.max_chunks)
    } else {
        num_chunks
    };
    println!("  チャンク数: {num_chunks}");
    if num_chunks == 0 {
        eprintln!("チャンク数が0。データが短すぎます。");
        std::process::exit(1);
    }

    // モデル読み込み (streaming: mmap + JIT ternary)
    println!("\n  モデル読み込み中 (streaming mmap)...");
    let load_start = Instant::now();
    let model = alice_train::inference::StreamingAliceModel::from_file(&args.model)
        .unwrap_or_else(|e| {
            eprintln!("モデル読み込み失敗: {e}");
            std::process::exit(1);
        });
    println!("  モデル読み込み完了 ({:.1}s)", load_start.elapsed().as_secs_f64());

    let vocab_size = model.config().vocab_size;

    // チャンクごとに forward → loss 計算
    println!("\n━━━ PPL 計測開始 ━━━");
    let eval_start = Instant::now();
    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * args.seq_len;
        let end = (start + args.seq_len + 1).min(tokens.len());
        let chunk = &tokens[start..end];

        if chunk.len() < 2 {
            continue;
        }

        // チャンクごとにキャッシュ再作成 (状態リセット)
        let mut cache = model.create_cache();
        let mut chunk_loss = 0.0f64;
        let mut chunk_count = 0usize;

        for t in 0..chunk.len() - 1 {
            let input_token = chunk[t];
            let target_token = chunk[t + 1] as usize;

            let logits = model
                .forward_incremental_streaming(input_token, &mut cache)
                .unwrap_or_else(|e| {
                    eprintln!("Forward 失敗 (chunk {chunk_idx}, pos {t}): {e}");
                    std::process::exit(1);
                });

            // Cross entropy: -log(softmax(logits)[target])
            // = -logits[target] + log(sum(exp(logits)))
            // 数値安定性: log-sum-exp trick
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp = max_logit
                + logits
                    .iter()
                    .map(|&x| (x - max_logit).exp())
                    .sum::<f32>()
                    .ln();

            let target_logit = if target_token < logits.len() {
                logits[target_token]
            } else {
                logits[0] // fallback
            };
            let ce = -target_logit + log_sum_exp;
            chunk_loss += ce as f64;
            chunk_count += 1;
        }

        let avg_chunk_loss = chunk_loss / chunk_count as f64;
        let chunk_ppl = avg_chunk_loss.exp();
        total_loss += chunk_loss;
        total_tokens += chunk_count;

        let elapsed = eval_start.elapsed().as_secs_f64();
        let tok_per_sec = total_tokens as f64 / elapsed;
        let eta = if tok_per_sec > 0.0 {
            ((num_chunks * args.seq_len) as f64 - total_tokens as f64) / tok_per_sec
        } else {
            0.0
        };

        println!(
            "  chunk {}/{} | loss: {:.4} | ppl: {:.1} | {:.1} tok/s | ETA: {:.0}s",
            chunk_idx + 1,
            num_chunks,
            avg_chunk_loss,
            chunk_ppl,
            tok_per_sec,
            eta,
        );
    }

    let avg_loss = total_loss / total_tokens as f64;
    let ppl = avg_loss.exp();
    let elapsed = eval_start.elapsed().as_secs_f64();

    println!("\n━━━ 結果 ━━━");
    println!("  評価トークン数: {total_tokens}");
    println!("  平均 Loss: {avg_loss:.4}");
    println!("  Perplexity (PPL): {ppl:.2}");
    println!("  計測時間: {elapsed:.1}s ({:.2} tok/s)", total_tokens as f64 / elapsed);
    println!("  学習時最低 Loss: 5.59 → 期待 PPL: {:.2}", (5.59f64).exp());
}
