//! ALICE 推論 CLI — .alice モデルでテキスト生成。
//!
//! # 使い方
//!
//! ```bash
//! cargo run --release --features qat-cli --bin alice-inference -- \
//!     --model qwen35_9b_ternary.alice \
//!     --tokenizer models/Qwen--Qwen3.5-9B/tokenizer.json \
//!     --prompt "What is 2+2?"
//! ```

use alice_train::inference::{AliceModel, GenerationConfig, StreamingAliceModel};
use alice_train::qwen35::LayerType;
use alice_train::tokenizer::BpeTokenizer;
use clap::Parser;
use std::io::Write;
use std::time::Instant;

/// ALICE ternary モデル推論。
#[derive(Parser)]
#[command(name = "alice-inference")]
#[command(about = ".alice ternary モデルでテキスト生成")]
struct Args {
    /// .alice モデルファイルパス。
    #[arg(long, short)]
    model: String,

    /// tokenizer.json パス。
    #[arg(long, short)]
    tokenizer: String,

    /// プロンプト (指定しない場合は対話モード)。
    #[arg(long, short)]
    prompt: Option<String>,

    /// システムプロンプト。
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,

    /// 最大生成トークン数。
    #[arg(long, default_value = "256")]
    max_tokens: usize,

    /// Temperature (0.0 = greedy)。
    #[arg(long, default_value = "0.7")]
    temperature: f32,

    /// Top-k サンプリング (0 = 無効)。
    #[arg(long, default_value = "50")]
    top_k: usize,

    /// Repetition penalty。
    #[arg(long, default_value = "1.1")]
    repetition_penalty: f32,

    /// モデル情報のみ表示。
    #[arg(long)]
    info: bool,

    /// ストリーミングモード (1層ずつ読み込み、低RAM)。
    #[arg(long)]
    streaming: bool,
}

fn main() {
    let args = Args::parse();

    // トークナイザー読み込み
    println!("[ALICE] トークナイザー読み込み: {}", args.tokenizer);
    let tokenizer = BpeTokenizer::from_file(&args.tokenizer).unwrap_or_else(|e| {
        eprintln!("[ERROR] トークナイザー読み込み失敗: {e}");
        std::process::exit(1);
    });
    println!("  vocab_size: {}", tokenizer.vocab_size());

    let gen_config = GenerationConfig {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
    };

    if args.streaming {
        // ストリーミングモード (低RAM)
        println!("[ALICE] ストリーミングモード: {}", args.model);
        let t0 = Instant::now();
        let model = StreamingAliceModel::from_file(&args.model).unwrap_or_else(|e| {
            eprintln!("[ERROR] モデル読み込み失敗: {e}");
            std::process::exit(1);
        });
        println!(
            "[ALICE] ヘッダー読み込み完了 ({:.1}s)",
            t0.elapsed().as_secs_f64()
        );

        if args.info {
            print_streaming_model_info(&model);
            return;
        }

        if let Some(ref prompt) = args.prompt {
            run_single_streaming(&model, &tokenizer, &args.system, prompt, &gen_config);
        } else {
            run_interactive_streaming(&model, &tokenizer, &args.system, &gen_config);
        }
    } else {
        // 通常モード (全プリロード)
        println!("[ALICE] モデル読み込み: {}", args.model);
        let t0 = Instant::now();
        let model = AliceModel::from_file(&args.model).unwrap_or_else(|e| {
            eprintln!("[ERROR] モデル読み込み失敗: {e}");
            std::process::exit(1);
        });
        println!(
            "[ALICE] モデル読み込み完了 ({:.1}s)",
            t0.elapsed().as_secs_f64()
        );

        if args.info {
            print_model_info(&model);
            return;
        }

        if let Some(ref prompt) = args.prompt {
            run_single(&model, &tokenizer, &args.system, prompt, &gen_config);
        } else {
            run_interactive(&model, &tokenizer, &args.system, &gen_config);
        }
    }
}

/// 単発テキスト生成。
fn run_single(
    model: &AliceModel,
    tokenizer: &BpeTokenizer,
    system: &str,
    prompt: &str,
    config: &GenerationConfig,
) {
    let prompt_ids = tokenizer.format_chat(system, prompt);
    println!("[ALICE] プロンプト: {} tokens", prompt_ids.len());

    let t0 = Instant::now();
    print!("[ALICE] ");
    std::io::stdout().flush().unwrap_or(());

    let mut token_count = 0usize;
    model.generate_streaming(
        &prompt_ids,
        config,
        tokenizer.eos_token_id,
        tokenizer,
        |text| {
            print!("{text}");
            std::io::stdout().flush().unwrap_or(());
            token_count += 1;
            true
        },
    );

    let elapsed = t0.elapsed();
    println!();
    println!(
        "[ALICE] {} tokens in {:.1}s ({:.1} tokens/s)",
        token_count,
        elapsed.as_secs_f64(),
        token_count as f64 / elapsed.as_secs_f64().max(0.001)
    );
}

/// 対話モード。
fn run_interactive(
    model: &AliceModel,
    tokenizer: &BpeTokenizer,
    system: &str,
    config: &GenerationConfig,
) {
    println!();
    println!("━━━ ALICE Interactive Mode ━━━");
    println!("  Type your message. Enter empty line to quit.");
    println!();

    loop {
        print!("> ");
        std::io::stdout().flush().unwrap_or(());

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            println!("[ALICE] Goodbye.");
            break;
        }

        let prompt_ids = tokenizer.format_chat(system, input);

        let t0 = Instant::now();
        let mut token_count = 0usize;

        model.generate_streaming(
            &prompt_ids,
            config,
            tokenizer.eos_token_id,
            tokenizer,
            |text| {
                print!("{text}");
                std::io::stdout().flush().unwrap_or(());
                token_count += 1;
                true
            },
        );

        let elapsed = t0.elapsed();
        println!();
        println!(
            "  [{} tokens, {:.1}s, {:.1} tok/s]",
            token_count,
            elapsed.as_secs_f64(),
            token_count as f64 / elapsed.as_secs_f64().max(0.001)
        );
        println!();
    }
}

// ── Streaming mode ─────────────────────────────────────────────────────

/// 単発テキスト生成 (ストリーミング)。
fn run_single_streaming(
    model: &StreamingAliceModel,
    tokenizer: &BpeTokenizer,
    system: &str,
    prompt: &str,
    config: &GenerationConfig,
) {
    let prompt_ids = tokenizer.format_chat(system, prompt);
    println!("[ALICE] プロンプト: {} tokens", prompt_ids.len());

    let t0 = Instant::now();
    print!("[ALICE] ");
    std::io::stdout().flush().unwrap_or(());

    let mut token_count = 0usize;
    model
        .generate_streaming_callback(
            &prompt_ids,
            config,
            tokenizer.eos_token_id,
            tokenizer,
            |text| {
                print!("{text}");
                std::io::stdout().flush().unwrap_or(());
                token_count += 1;
                true
            },
        )
        .unwrap_or_else(|e| {
            eprintln!("\n[ERROR] 生成中にエラー: {e}");
        });

    let elapsed = t0.elapsed();
    println!();
    println!(
        "[ALICE] {} tokens in {:.1}s ({:.1} tokens/s)",
        token_count,
        elapsed.as_secs_f64(),
        token_count as f64 / elapsed.as_secs_f64().max(0.001)
    );
}

/// 対話モード (ストリーミング)。
fn run_interactive_streaming(
    model: &StreamingAliceModel,
    tokenizer: &BpeTokenizer,
    system: &str,
    config: &GenerationConfig,
) {
    println!();
    println!("━━━ ALICE Interactive Mode (streaming) ━━━");
    println!("  Type your message. Enter empty line to quit.");
    println!();

    loop {
        print!("> ");
        std::io::stdout().flush().unwrap_or(());

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            println!("[ALICE] Goodbye.");
            break;
        }

        let prompt_ids = tokenizer.format_chat(system, input);

        let t0 = Instant::now();
        let mut token_count = 0usize;

        model
            .generate_streaming_callback(
                &prompt_ids,
                config,
                tokenizer.eos_token_id,
                tokenizer,
                |text| {
                    print!("{text}");
                    std::io::stdout().flush().unwrap_or(());
                    token_count += 1;
                    true
                },
            )
            .unwrap_or_else(|e| {
                eprintln!("\n[ERROR] 生成中にエラー: {e}");
            });

        let elapsed = t0.elapsed();
        println!();
        println!(
            "  [{} tokens, {:.1}s, {:.1} tok/s]",
            token_count,
            elapsed.as_secs_f64(),
            token_count as f64 / elapsed.as_secs_f64().max(0.001)
        );
        println!();
    }
}

/// ストリーミングモデル情報表示。
fn print_streaming_model_info(model: &StreamingAliceModel) {
    let meta = &model.meta;
    println!();
    println!("=== ALICE Model (streaming) ===");
    println!("Quantization: {}", meta.quantization);
    println!("Tied embeddings: {}", meta.tied_embeddings);
    println!(
        "Source: step {}, loss {:.4}",
        meta.source_step, meta.source_loss
    );
    println!("Parameters:");
    println!("  Quantized:     {}", meta.quantized_params);
    println!("  Non-quantized: {}", meta.non_quantized_params);
    println!("  Total:         {}", meta.total_params);
    println!("Config:");
    println!("  vocab_size: {}", meta.config.vocab_size);
    println!("  hidden_size: {}", meta.config.hidden_size);
    println!("  num_layers: {}", meta.config.num_hidden_layers);
    println!(
        "  layer pattern: {} DeltaNet + {} FullAttn",
        meta.config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count(),
        meta.config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count(),
    );
    println!(
        "RAM (streaming): embedding {:.1} GB + 1 layer ~{:.1} GB",
        model.embedding.len() as f64 * 4.0 / 1e9,
        meta.config.deltanet_params_per_layer() as f64 * 4.0 / 1e9,
    );

    if !meta.layer_scales.is_empty() {
        println!("Layer 0 scales:");
        for (name, scale) in &meta.layer_scales[0].scales {
            println!("  {name}: {scale:.6}");
        }
    }
}

/// モデル情報表示。
fn print_model_info(model: &AliceModel) {
    let meta = &model.meta;
    println!();
    println!("=== ALICE Model ===");
    println!("Quantization: {}", meta.quantization);
    println!("Tied embeddings: {}", meta.tied_embeddings);
    println!(
        "Source: step {}, loss {:.4}",
        meta.source_step, meta.source_loss
    );
    println!("Parameters:");
    println!("  Quantized:     {}", meta.quantized_params);
    println!("  Non-quantized: {}", meta.non_quantized_params);
    println!("  Total:         {}", meta.total_params);
    println!("Config:");
    println!("  vocab_size: {}", meta.config.vocab_size);
    println!("  hidden_size: {}", meta.config.hidden_size);
    println!("  num_layers: {}", meta.config.num_hidden_layers);
    println!(
        "  layer pattern: {} DeltaNet + {} FullAttn",
        meta.config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count(),
        meta.config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count(),
    );
    println!(
        "Memory: embedding {:.1} GB, layers ~{:.1} GB",
        model.embedding.len() as f64 * 4.0 / 1e9,
        model.layers.len() as f64 * meta.config.deltanet_params_per_layer() as f64 * 4.0 / 1e9,
    );

    if !meta.layer_scales.is_empty() {
        println!("Layer 0 scales:");
        for (name, scale) in &meta.layer_scales[0].scales {
            println!("  {name}: {scale:.6}");
        }
    }
}
