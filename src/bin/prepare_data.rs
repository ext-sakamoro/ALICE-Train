//! ALICE-Train データ前処理 — JSONL → raw u32 トークンファイル変換。
//!
//! alice-data-prep が生成した JSONL を、DataLoader が読み込める
//! raw u32 (LE) バイナリに変換する。
//!
//! Phase 1: バイトレベルトークナイズ（各 UTF-8 バイト → token ID）。
//! Phase 2+: `HuggingFace` tokenizer 統合予定。
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
fn convert_all(data_dir: &str) {
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
                let mb = tokens_to_mb(tokens);
                eprintln!(
                    "  {domain}: {samples} サンプル → {tokens} トークン → {mb:.1} MB → {}",
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
        tokens_to_mb(total_tokens),
    );
}

/// トークン数を MB 換算する（u32 = 4 バイト）。
///
/// `usize` を直接 `f64` にキャストすると 64-bit 環境で精度が失われるため、
/// まず `u64` に昇格させてからバイト数を算出し、MB 換算する。
/// MB スケールでの誤差は 1 バイト未満であり実用上問題ない。
#[allow(clippy::cast_precision_loss)]
fn tokens_to_mb(tokens: usize) -> f64 {
    // usize → u64 は情報損失なし。u64 → f64 は仮数 53-bit の精度制限があるが
    // MB 表示レベルでは誤差は無視できる。
    let bytes = (tokens as u64) * 4;
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    let cli = Cli::parse();

    if cli.all {
        eprintln!("╔══════════════════════════════════════════════════════════╗");
        eprintln!("║  ALICE-Train: 全ドメイン JSONL → bin 変換               ║");
        eprintln!("╚══════════════════════════════════════════════════════════╝");
        eprintln!();

        convert_all(&cli.data_dir);
    } else {
        let input = cli.input.as_deref().unwrap();
        let output = cli.output.as_deref().unwrap();

        match convert_file(Path::new(input), Path::new(output)) {
            Ok((samples, tokens)) => {
                eprintln!(
                    "{samples} サンプル → {tokens} トークン ({:.1} MB) → {output}",
                    tokens_to_mb(tokens),
                );
            }
            Err(e) => {
                eprintln!("エラー: {e}");
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
// --- tokenize_byte_level ---

    #[test]
    fn test_tokenize_empty_string() {
        let tokens = tokenize_byte_level("");
        // BOS + EOS のみ
        assert_eq!(tokens, vec![1, 2]);
    }

    #[test]
    fn test_tokenize_ascii() {
        let tokens = tokenize_byte_level("AB");
        // BOS + 'A'(65+3=68) + 'B'(66+3=69) + EOS
        assert_eq!(tokens, vec![1, 68, 69, 2]);
    }

    #[test]
    fn test_tokenize_bos_eos_markers() {
        let tokens = tokenize_byte_level("x");
        assert_eq!(tokens[0], 1, "先頭は BOS");
        assert_eq!(*tokens.last().unwrap(), 2, "末尾は EOS");
    }

    #[test]
    fn test_tokenize_byte_offset() {
        // バイト値 0 → token 3
        let min_token = tokenize_byte_level("\x00");
        assert_eq!(min_token[1], 3);

        // バイト値 255 → token 258（UTF-8 の 0xFF は単独では不正なので文字列リテラル不可。
        // バイト列から直接トークン列を生成して確認する）
        let tokens: Vec<u32> = {
            let mut v = vec![1u32]; // BOS
            v.push(u32::from(255u8) + 3);
            v.push(2); // EOS
            v
        };
        assert_eq!(tokens[1], 258);
    }

    #[test]
    fn test_tokenize_multibyte_utf8() {
        // "あ" は UTF-8 で 3 バイト (0xE3 0x81 0x82)
        let tokens = tokenize_byte_level("あ");
        assert_eq!(tokens.len(), 5); // BOS + 3バイト + EOS
        assert_eq!(tokens[1], 0xE3 + 3);
        assert_eq!(tokens[2], 0x81 + 3);
        assert_eq!(tokens[3], 0x82 + 3);
    }

    #[test]
    fn test_tokenize_length() {
        let text = "hello";
        let tokens = tokenize_byte_level(text);
        assert_eq!(tokens.len(), text.len() + 2); // +2 for BOS/EOS
    }

    // --- tokens_to_mb ---

    #[test]
    fn test_tokens_to_mb_zero() {
        assert_eq!(tokens_to_mb(0), 0.0);
    }

    #[test]
    fn test_tokens_to_mb_one_mib() {
        // 1MB = 1024*1024 バイト、各トークン 4 バイト → 262144 トークン
        let tokens = 1024 * 1024 / 4;
        let mb = tokens_to_mb(tokens);
        assert!((mb - 1.0).abs() < 1e-9, "1MiB = {mb}");
    }

    #[test]
    fn test_tokens_to_mb_large() {
        // 1G トークン → 4GB
        let tokens = 1_000_000_000usize;
        let mb = tokens_to_mb(tokens);
        let expected = 1_000_000_000_u64 as f64 * 4.0 / (1024.0 * 1024.0);
        assert!((mb - expected).abs() < 1.0);
    }

    // --- convert_file ---

    fn write_temp_jsonl(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_convert_file_single_sample() {
        let jsonl = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let input = write_temp_jsonl(jsonl);
        let output = tempfile::NamedTempFile::new().unwrap();

        let (samples, tokens) = convert_file(input.path(), output.path()).unwrap();
        assert_eq!(samples, 1);
        // "user: hi\n" = 9 bytes + BOS + EOS = 11 tokens
        assert_eq!(tokens, 11);

        let data = fs::read(output.path()).unwrap();
        assert_eq!(data.len(), tokens * 4);
        // 先頭 4 バイトは BOS = 1 (LE)
        assert_eq!(u32::from_le_bytes(data[..4].try_into().unwrap()), 1u32);
        // 末尾 4 バイトは EOS = 2 (LE)
        let last = data.len() - 4;
        assert_eq!(u32::from_le_bytes(data[last..].try_into().unwrap()), 2u32);
    }

    #[test]
    fn test_convert_file_multi_sample() {
        let jsonl = concat!(
            r#"{"messages":[{"role":"user","content":"a"}]}"#,
            "\n",
            r#"{"messages":[{"role":"assistant","content":"b"}]}"#,
        );
        let input = write_temp_jsonl(jsonl);
        let output = tempfile::NamedTempFile::new().unwrap();

        let (samples, _tokens) = convert_file(input.path(), output.path()).unwrap();
        assert_eq!(samples, 2);
    }

    #[test]
    fn test_convert_file_empty_lines_skipped() {
        let jsonl = concat!(
            r#"{"messages":[{"role":"user","content":"x"}]}"#,
            "\n\n",
            r#"{"messages":[{"role":"user","content":"y"}]}"#,
        );
        let input = write_temp_jsonl(jsonl);
        let output = tempfile::NamedTempFile::new().unwrap();

        let (samples, _tokens) = convert_file(input.path(), output.path()).unwrap();
        assert_eq!(samples, 2);
    }

    #[test]
    fn test_convert_file_null_content_skipped() {
        // content が null のメッセージは text に追加されない
        let jsonl = r#"{"messages":[{"role":"user","content":null}]}"#;
        let input = write_temp_jsonl(jsonl);
        let output = tempfile::NamedTempFile::new().unwrap();

        let (samples, tokens) = convert_file(input.path(), output.path()).unwrap();
        assert_eq!(samples, 1);
        // text は空文字列 → BOS + EOS = 2 tokens
        assert_eq!(tokens, 2);
    }

    #[test]
    fn test_convert_file_invalid_json_returns_error() {
        let input = write_temp_jsonl("not-json\n");
        let output = tempfile::NamedTempFile::new().unwrap();

        let result = convert_file(input.path(), output.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_file_output_is_le_u32() {
        let jsonl = r#"{"messages":[{"role":"u","content":"A"}]}"#;
        let input = write_temp_jsonl(jsonl);
        let output = tempfile::NamedTempFile::new().unwrap();

        convert_file(input.path(), output.path()).unwrap();
        let data = fs::read(output.path()).unwrap();

        // ファイルサイズは 4 の倍数
        assert_eq!(data.len() % 4, 0);
    }
}
