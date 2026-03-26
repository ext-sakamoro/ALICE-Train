//! Ternary エクスポート CLI — FP32 チェックポイント → .alice ファイル変換。
//!
//! # 使い方
//!
//! ```bash
//! cargo run --release --features qat-cli --bin export-ternary -- \
//!     --checkpoint-dir checkpoints/qwen35_9b \
//!     --model-path models/Qwen--Qwen3.5-9B \
//!     --output qwen35_9b_ternary.alice \
//!     --step 5000 \
//!     --loss 3.5
//! ```

use alice_train::export::{export_alice_model, read_alice_meta};
use alice_train::qwen35::Qwen35Config;
use alice_train::safetensors_loader::ShardedModel;
use clap::Parser;
use std::fs;
use std::io::{self, BufWriter};
use std::time::Instant;

/// FP32 チェックポイントを .alice ternary 形式にエクスポート。
#[derive(Parser)]
#[command(name = "export-ternary")]
#[command(about = "QAT学習済みFP32重みを.alice ternary形式にエクスポート")]
struct Args {
    /// チェックポイントディレクトリ (fp32_cache を含む)。
    #[arg(long, default_value = "checkpoints/qwen35_9b")]
    checkpoint_dir: String,

    /// safetensors モデルパス (embedding/lm_head 用)。
    #[arg(long, default_value = "models/Qwen--Qwen3.5-9B")]
    model_path: String,

    /// 出力 .alice ファイルパス。
    #[arg(long, short, default_value = "qwen35_9b_ternary.alice")]
    output: String,

    /// 元チェックポイントのステップ番号。
    #[arg(long, default_value = "0")]
    step: usize,

    /// 元チェックポイントの loss 値。
    #[arg(long, default_value = "0.0")]
    loss: f32,

    /// safetensors 重み名プレフィックス。
    #[arg(long, default_value = "model.language_model")]
    weight_prefix: String,

    /// メタデータのみ表示 (既存 .alice ファイルの情報確認)。
    #[arg(long)]
    info: Option<String>,
}

fn main() {
    let args = Args::parse();

    // --info モード
    if let Some(ref path) = args.info {
        print_info(path);
        return;
    }

    println!("[ALICE-Train] Ternary エクスポート開始");
    println!("  checkpoint_dir: {}", args.checkpoint_dir);
    println!("  model_path: {}", args.model_path);
    println!("  output: {}", args.output);
    println!("  step: {}, loss: {:.4}", args.step, args.loss);

    let t0 = Instant::now();

    let config = Qwen35Config::qwen35_9b();
    let prefix = &args.weight_prefix;

    // FP32 キャッシュ存在確認
    if !alice_train::fp32_cache::cache_exists(&args.checkpoint_dir, &config) {
        eprintln!(
            "[ERROR] FP32 キャッシュが見つかりません: {}/fp32_cache/",
            args.checkpoint_dir
        );
        eprintln!("  学習完了後のチェックポイントディレクトリを指定してください");
        std::process::exit(1);
    }

    // safetensors 読み込み (embedding, output_norm, lm_head)
    println!("  safetensors 読み込み: {}", args.model_path);
    let model = ShardedModel::open(&args.model_path).unwrap_or_else(|e| {
        eprintln!("[ERROR] safetensors 読み込み失敗: {e}");
        std::process::exit(1);
    });

    let embedding = model
        .get_tensor_f32(&format!("{prefix}.embed_tokens.weight"))
        .unwrap_or_else(|| {
            eprintln!("[ERROR] {prefix}.embed_tokens.weight が見つかりません");
            std::process::exit(1);
        });
    println!(
        "    embed_tokens: {} params ({:.1} MB)",
        embedding.len(),
        embedding.len() as f64 * 4.0 / 1e6
    );

    let output_norm = model
        .get_tensor_f32(&format!("{prefix}.norm.weight"))
        .unwrap_or_else(|| {
            eprintln!("[ERROR] {prefix}.norm.weight が見つかりません");
            std::process::exit(1);
        });

    let lm_head_opt = model.get_tensor_f32("lm_head.weight");
    let tied = lm_head_opt.is_none();
    if tied {
        println!("    lm_head: tied (embed_tokens と共有)");
    } else {
        println!(
            "    lm_head: {} params ({:.1} MB)",
            lm_head_opt.as_ref().unwrap().len(),
            lm_head_opt.as_ref().unwrap().len() as f64 * 4.0 / 1e6
        );
    }

    // safetensors を閉じて RAM 解放
    drop(model);
    println!("  safetensors 解放完了");

    // エクスポート
    println!("  .alice ファイル書き込み: {}", args.output);
    let file = fs::File::create(&args.output).unwrap_or_else(|e| {
        eprintln!("[ERROR] 出力ファイル作成失敗: {e}");
        std::process::exit(1);
    });
    let mut writer = BufWriter::new(file);

    let stats = export_alice_model(
        &mut writer,
        &config,
        &args.checkpoint_dir,
        &embedding,
        &output_norm,
        lm_head_opt.as_deref(),
        args.step,
        args.loss,
    )
    .unwrap_or_else(|e| {
        eprintln!("[ERROR] エクスポート失敗: {e}");
        std::process::exit(1);
    });

    let elapsed = t0.elapsed();

    println!();
    println!(
        "[ALICE-Train] エクスポート完了 ({:.1}s)",
        elapsed.as_secs_f64()
    );
    println!("  出力: {}", args.output);
    println!(
        "  ファイルサイズ: {:.2} GB ({} bytes)",
        stats.total_bytes as f64 / 1e9,
        stats.total_bytes
    );
    println!("  内訳:");
    println!(
        "    Embedding (BF16): {:.2} GB",
        stats.embed_bytes as f64 / 1e9
    );
    println!(
        "    lm_head (BF16):   {:.2} GB",
        stats.lm_head_bytes as f64 / 1e9
    );
    println!(
        "    Ternary layers:   {:.2} GB",
        stats.ternary_bytes as f64 / 1e9
    );
    println!(
        "    FP32 (norms等):   {:.2} MB",
        stats.layer_fp32_bytes as f64 / 1e6
    );
    println!("  量子化パラメータ: {}", stats.quantized_params);
    println!(
        "  圧縮率: {:.1}x (vs FP32 {:.2} GB)",
        (stats.meta.total_params as f64 * 4.0) / stats.total_bytes as f64,
        stats.meta.total_params as f64 * 4.0 / 1e9
    );
}

/// .alice ファイルのメタデータを表示。
fn print_info(path: &str) {
    let file = fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("[ERROR] ファイルを開けません: {path}: {e}");
        std::process::exit(1);
    });
    let mut reader = io::BufReader::new(file);
    let meta = read_alice_meta(&mut reader).unwrap_or_else(|e| {
        eprintln!("[ERROR] メタデータ読み込み失敗: {e}");
        std::process::exit(1);
    });

    println!("=== ALICE Model Info ===");
    println!("Version: {}", meta.version);
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
    println!("Model config:");
    println!("  vocab_size: {}", meta.config.vocab_size);
    println!("  hidden_size: {}", meta.config.hidden_size);
    println!("  num_layers: {}", meta.config.num_hidden_layers);
    println!("  num_attention_heads: {}", meta.config.num_attention_heads);

    if !meta.layer_scales.is_empty() {
        println!("Layer scales (first layer):");
        for (name, scale) in &meta.layer_scales[0].scales {
            println!("  {name}: {scale:.6}");
        }
    }
}
