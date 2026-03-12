//! ALICE-Train QAT 実行バイナリ — Llama-3 70B → 1.1-bit Ternary。
//!
//! # 使用法
//!
//! ```bash
//! # 新規学習
//! cargo run --release --features qat-cli --bin train-qat-70b -- \
//!     --config configs/qat_70b.json
//!
//! # チェックポイントからレジューム
//! cargo run --release --features qat-cli --bin train-qat-70b -- \
//!     --config configs/qat_70b.json \
//!     --resume checkpoints/step_5000.bin
//! ```
//!
//! # Spot インスタンス自動レジューム
//!
//! ```bash
//! scripts/auto_resume.sh configs/qat_70b.json
//! ```

use clap::Parser;
use std::fs;
use std::path::Path;
use std::time::Instant;

use alice_train::llama::QatTrainConfig;
use alice_train::{
    CheckpointData, FakeQuantize, LogEntry, LossScaler, LrScheduler, MixedPrecisionConfig,
    OffloadConfig, QatConfig, TrainLog, WarmupCosineScheduler,
};

/// ALICE-Train QAT: Llama-3 → 1.1-bit Sparse Ternary
#[derive(Parser, Debug)]
#[command(author = "Moroya Sakamoto")]
#[command(about = "Quantize Llama-3 70B to 1.1-bit ternary via QAT")]
struct Cli {
    /// 設定ファイルパス (JSON)
    #[arg(short, long)]
    config: String,

    /// チェックポイントからレジューム
    #[arg(short, long)]
    resume: Option<String>,

    /// ドライラン (設定の表示のみ)
    #[arg(long)]
    dry_run: bool,
}

fn main() {
    let cli = Cli::parse();

    // --- 設定読み込み ---
    let config_str = fs::read_to_string(&cli.config).unwrap_or_else(|e| {
        eprintln!(
            "[ALICE-Train] 設定ファイル読み込み失敗: {}: {e}",
            cli.config
        );
        std::process::exit(1);
    });
    let mut config: QatTrainConfig = serde_json::from_str(&config_str).unwrap_or_else(|e| {
        eprintln!("[ALICE-Train] 設定パースエラー: {e}");
        std::process::exit(1);
    });

    // CLI のレジュームオプションで設定を上書き
    if let Some(resume_path) = &cli.resume {
        config.resume_from = Some(resume_path.clone());
    }

    // --- メモリ見積もり表示 ---
    let model_config = &config.model;
    let total_params = model_config.total_params();
    let ternary_bytes = model_config.ternary_memory_bytes();
    let params_per_layer = model_config.params_per_layer();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  ALICE-Train QAT — 1.1-bit Sparse Ternary 量子化学習   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("モデル: Llama-3 {}B", total_params / 1_000_000_000);
    println!("  隠れ層: {}", model_config.hidden_dim);
    println!("  レイヤー: {}", model_config.num_layers);
    println!(
        "  ヘッド: {} (KV: {})",
        model_config.num_heads, model_config.num_kv_heads
    );
    println!("  総パラメータ: {:.2}B", total_params as f64 / 1e9);
    println!(
        "  パラメータ/レイヤー: {:.2}M",
        params_per_layer as f64 / 1e6
    );
    println!();
    println!("メモリ見積もり:");
    println!(
        "  FP32 全体: {:.1} GB",
        total_params as f64 * 4.0 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "  Ternary 出力: {:.1} GB",
        ternary_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "  圧縮率: {:.1}x",
        (total_params * 4) as f64 / ternary_bytes.max(1) as f64
    );
    println!();

    let offload_budget = alice_train::MemoryBudget::estimate(params_per_layer);
    println!("ZeRO-Offload (レイヤー単位):");
    println!(
        "  GPU VRAM: {:.1} MB (重み + 勾配)",
        offload_budget.vram_bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "  CPU RAM: {:.1} MB (AdamW m/v)",
        offload_budget.ram_bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "  VRAM 節約: {:.0}%",
        offload_budget.vram_savings_ratio() * 100.0
    );
    println!();

    println!("学習設定:");
    println!("  学習率: {} → {}", config.learning_rate, config.min_lr);
    println!("  ウォームアップ: {} steps", config.warmup_steps);
    println!("  総ステップ: {}", config.total_steps);
    println!("  勾配累積: {}", config.gradient_accumulation_steps);
    println!(
        "  バッチ: {} × seq_len={}",
        config.batch_size, config.seq_len
    );
    println!(
        "  実効バッチサイズ: {}",
        config.batch_size * config.gradient_accumulation_steps
    );
    println!("  BF16: {}", if config.use_bf16 { "有効" } else { "無効" });
    println!(
        "  チェックポイント: {} steps ごと",
        config.checkpoint_interval
    );
    println!("  評価: {} steps ごと", config.eval_interval);
    println!();

    if let Some(ref resume) = config.resume_from {
        println!("レジューム: {resume}");
    }

    if cli.dry_run {
        println!("[ドライラン完了 — 実行は行いません]");
        return;
    }

    // --- チェックポイントディレクトリ作成 ---
    fs::create_dir_all(&config.checkpoint_dir).unwrap_or_else(|e| {
        eprintln!(
            "[ALICE-Train] チェックポイントディレクトリ作成失敗: {}: {e}",
            config.checkpoint_dir
        );
        std::process::exit(1);
    });

    // --- モデル読み込み ---
    println!("━━━ モデル読み込み開始 ━━━");
    let model_path = Path::new(&config.model_path);
    if !model_path.exists() {
        eprintln!(
            "[ALICE-Train] モデルパスが存在しません: {}",
            config.model_path
        );
        eprintln!();
        eprintln!("モデルをダウンロードしてください:");
        eprintln!(
            "  huggingface-cli download meta-llama/Llama-3-70B --local-dir {}",
            config.model_path
        );
        std::process::exit(1);
    }

    // safetensors ファイル一覧
    let st_files: Vec<_> = fs::read_dir(model_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect();

    if st_files.is_empty() {
        eprintln!(
            "[ALICE-Train] safetensors ファイルが見つかりません: {}",
            config.model_path
        );
        std::process::exit(1);
    }

    println!("  safetensors ファイル: {} 個", st_files.len());

    // --- QAT コンポーネント初期化 ---
    println!("━━━ QAT コンポーネント初期化 ━━━");

    let _fq = FakeQuantize::new(QatConfig::ternary());
    let scheduler = WarmupCosineScheduler::new(
        config.learning_rate,
        config.min_lr,
        config.warmup_steps,
        config.total_steps,
    );

    let scaler = if config.use_bf16 {
        Some(LossScaler::new(MixedPrecisionConfig::default()))
    } else {
        None
    };

    let _offload_config = OffloadConfig {
        weight_decay: config.weight_decay,
        max_grad_norm: Some(config.max_grad_norm),
        ..OffloadConfig::default()
    };

    let mut log = TrainLog::new();
    let mut global_step: usize = 0;

    // レジューム
    if let Some(ref resume_path) = config.resume_from {
        if Path::new(resume_path).exists() {
            println!("  チェックポイントからレジューム: {resume_path}");
            match std::fs::File::open(resume_path).and_then(|mut f| CheckpointData::load(&mut f)) {
                Ok(ckpt) => {
                    global_step = ckpt.meta.epoch; // epoch フィールドを step として使用
                    println!("  レジューム成功: step {global_step}");
                }
                Err(e) => {
                    eprintln!("  チェックポイント読み込み失敗: {e}");
                    eprintln!("  最初から学習を開始します");
                }
            }
        } else {
            println!("  チェックポイントが見つかりません: {resume_path}");
            println!("  最初から学習を開始します");
        }
    }

    println!("  FakeQuantize: 初期化完了");
    println!(
        "  WarmupCosineScheduler: {}/{} steps",
        config.warmup_steps, config.total_steps
    );
    println!(
        "  LossScaler: {}",
        if scaler.is_some() {
            "BF16有効"
        } else {
            "無効"
        }
    );
    println!("  OffloadOptimizer: weight_decay={}", config.weight_decay);

    // --- 学習ループ ---
    println!();
    println!(
        "━━━ QAT 学習開始 (step {global_step}/{}) ━━━",
        config.total_steps
    );
    let train_start = Instant::now();

    // TODO: safetensors からレイヤー重みを mmap で読み込み、
    // レイヤーごとに forward → backward → OffloadOptimizer.step() を実行
    //
    // 現時点では以下のスケルトンを提供:
    // 1. 各 safetensors ファイルを mmap で開く
    // 2. レイヤー i の重みを FP32 で取得
    // 3. FakeQuantize で ternary forward
    // 4. Cross-entropy loss 計算
    // 5. STE backward
    // 6. OffloadOptimizer.step() (m/v は CPU RAM)
    // 7. checkpoint_interval ごとに保存

    while global_step < config.total_steps {
        let lr = scheduler.get_lr(global_step);
        let step_start = Instant::now();

        // --- ここにレイヤー単位の forward/backward ループが入る ---
        // 各レイヤーについて:
        //   let mut layer_weights = LlamaLayerWeights::from_tensors(i, &get_tensor, &config.model);
        //   let mut optimizer = OffloadOptimizer::new(layer_weights.proj_param_count(), offload_config.clone());
        //   for micro_batch in 0..config.gradient_accumulation_steps {
        //       // forward: FakeQuantize → ternary matvec
        //       // backward: STE → grad accumulation
        //   }
        //   optimizer.step(&mut weights, &mut grads, lr);

        let step_loss = 0.0_f32; // placeholder
        let _step_duration = step_start.elapsed();

        // ログ記録
        log.append(LogEntry::new(0, global_step, step_loss, lr, 0.0));

        // 進捗表示
        if global_step.is_multiple_of(10) {
            let elapsed = train_start.elapsed();
            let steps_per_sec = if elapsed.as_secs() > 0 {
                global_step as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            let eta_secs = if steps_per_sec > 0.0 {
                (config.total_steps - global_step) as f64 / steps_per_sec
            } else {
                0.0
            };
            println!(
                "  step {global_step}/{} | loss: {step_loss:.4} | lr: {lr:.2e} | {:.2} steps/s | ETA: {:.0}s",
                config.total_steps, steps_per_sec, eta_secs,
            );
        }

        // チェックポイント保存
        if global_step > 0 && global_step.is_multiple_of(config.checkpoint_interval) {
            let ckpt_path = format!("{}/step_{global_step}.bin", config.checkpoint_dir);
            println!("  チェックポイント保存: {ckpt_path}");
            // TODO: CheckpointData::save() で重み + optimizer state を保存
        }

        // 評価
        if global_step > 0 && global_step.is_multiple_of(config.eval_interval) {
            println!("  評価実行中...");
            // TODO: eval_data_path のトークンで perplexity 計算
        }

        global_step += 1;
    }

    // --- 学習完了 ---
    let total_duration = train_start.elapsed();
    println!();
    println!("━━━ QAT 学習完了 ━━━");
    println!("  総ステップ: {global_step}");
    println!(
        "  学習時間: {:.1} 時間",
        total_duration.as_secs_f64() / 3600.0
    );

    // ログ保存
    let log_path = format!("{}/train_log.csv", config.checkpoint_dir);
    if let Err(e) = log.save_csv_to_file(&log_path) {
        eprintln!("  ログ保存失敗: {e}");
    } else {
        println!("  ログ保存: {log_path}");
    }

    // 最終重み出力
    println!("  最終 ternary 重みの出力...");
    // TODO: finalize_weights → safetensors 出力
    println!("  完了。");
}
