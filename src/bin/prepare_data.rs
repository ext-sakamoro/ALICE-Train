//! ALICE-Train データ前処理 — JSONL → raw u32 トークンファイル変換。
//!
//! alice-data-prep が生成した JSONL を、DataLoader が読み込める
//! raw u32 (LE) バイナリに変換する。
//!
//! Phase 1: バイトレベルトークナイズ（各 UTF-8 バイト → token ID）。
//! Phase 2+: HuggingFace tokenizer 統合予定。
//!
//! # 使用法
//!
//! ```bash
//! cargo run --release --features qat-cli --bin prepare-data -- \
//!     --input data/general/train.jsonl \
//!     --output data/general/train.bin
//!
//! # 全ドメイン一括変換
//! cargo run --release --features qat-cli --bin prepare-data -- --all
//! ```

use clap::Parser;
use serde::Deserialize;
use std::fs;
use std::io::{BufRead, Write};
use std::path::Path;

/// ALICE-Train: JSONL → raw u32 トークンファイル変換
#[derive(Parser, Debug)]
#[command(author = "Moroya Sakamoto")]
#[command(about = "Convert JSONL training data to raw u32 token binary")]
struct Cli {
    /// 入力 JSONL ファイルパス
    #[arg(short, long, required_unless_present = "all")]
    input: Option<String>,

    /// 出力 bin ファイルパス
    #[arg(short, long, required_unless_present = "all")]
    output: Option<String>,

    /// 全ドメイン一括変換（data/*/train.jsonl → data/*/train.bin）
    #[arg(long)]
    all: bool,

    /// データディレクトリ（--all 時に使用）
    #[arg(long, default_value = "data")]
    data_dir: String,
}

/// JSONL のメッセージ構造。
#[derive(Deserialize)]
struct Message {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
}

/// JSONL のサンプル構造。
#[derive(Deserialize)]
struct Sample {
    messages: Vec<Message>,
    #[allow(dead_code)]
    domain: Option<String>,
}

/// バイトレベルトークナイズ。
///
/// トークン割当:
/// - 0: PAD
/// - 1: BOS (Begin of Sequence)
/// - 2: EOS (End of Sequence)
/// - 3..258: UTF-8 バイト値 0..255
fn tokenize_byte_level(text: &str) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(text.len() + 2);
    tokens.push(1); // BOS
    for byte in text.bytes() {
        tokens.push(u32::from(byte) + 3);
    }
    tokens.push(2); // EOS
    tokens
}

/// 1 ファイル分の変換。
fn convert_file(input: &Path, output: &Path) -> std::io::Result<(usize, usize)> {
    let file = fs::File::open(input)?;
    let reader = std::io::BufReader::new(file);

    let mut all_tokens: Vec<u32> = Vec::new();
    let mut sample_count = 0usize;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let sample: Sample = serde_json::from_str(&line).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("JSON parse error at sample {sample_count}: {e}"),
            )
        })?;

        // メッセージを連結（Llama-3 形式の簡易版）
        let mut text = String::new();
        for msg in &sample.messages {
            if let Some(content) = &msg.content {
                text.push_str(&msg.role);
                text.push_str(": ");
                text.push_str(content);
                text.push('\n');
            }
        }

        let tokens = tokenize_byte_level(&text);
        all_tokens.extend_from_slice(&tokens);
        sample_count += 1;
    }

    // 親ディレクトリ作成
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    // raw u32 LE で書き出し
    let mut out = std::io::BufWriter::new(fs::File::create(output)?);
    for &t in &all_tokens {
        out.write_all(&t.to_le_bytes())?;
    }
    out.flush()?;

    Ok((sample_count, all_tokens.len()))
}

/// 全ドメイン一括変換。
fn convert_all(data_dir: &str) -> std::io::Result<()> {
    let domains = [
        "general", "code", "japanese", "math", "finance", "medical", "legal", "security",
        "robotics", "creative", "spatial", "infra",
    ];

    let mut total_samples = 0usize;
    let mut total_tokens = 0usize;

    for domain in &domains {
        let input = Path::new(data_dir).join(domain).join("train.jsonl");
        let output = Path::new(data_dir).join(domain).join("train.bin");

        if !input.exists() {
            eprintln!("  {domain}: train.jsonl が見つかりません。スキップ。");
            continue;
        }

        match convert_file(&input, &output) {
            Ok((samples, tokens)) => {
                eprintln!(
                    "  {domain}: {samples} サンプル → {tokens} トークン → {}",
                    output.display()
                );
                total_samples += samples;
                total_tokens += tokens;
            }
            Err(e) => {
                eprintln!("  {domain}: エラー: {e}");
            }
        }
    }

    eprintln!();
    eprintln!(
        "合計: {total_samples} サンプル, {total_tokens} トークン ({:.1} MB)",
        total_tokens as f64 * 4.0 / 1024.0 / 1024.0
    );

    Ok(())
}

fn main() {
    let cli = Cli::parse();

    if cli.all {
        eprintln!("╔══════════════════════════════════════════════════════════╗");
        eprintln!("║  ALICE-Train: 全ドメイン JSONL → bin 変換               ║");
        eprintln!("╚══════════════════════════════════════════════════════════╝");
        eprintln!();

        if let Err(e) = convert_all(&cli.data_dir) {
            eprintln!("エラー: {e}");
            std::process::exit(1);
        }
    } else {
        let input = cli.input.as_deref().unwrap();
        let output = cli.output.as_deref().unwrap();

        match convert_file(Path::new(input), Path::new(output)) {
            Ok((samples, tokens)) => {
                eprintln!(
                    "{samples} サンプル → {tokens} トークン ({:.1} MB) → {output}",
                    tokens as f64 * 4.0 / 1024.0 / 1024.0
                );
            }
            Err(e) => {
                eprintln!("エラー: {e}");
                std::process::exit(1);
            }
        }
    }
}
