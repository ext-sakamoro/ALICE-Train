//! ALICE-Train QAT 実行バイナリ — Llama-3 → 1.1-bit Ternary。
//!
//! # Phase 1 (パイプライン検証)
//!
//! モデルパスが存在しない場合、ランダム初期化の小型モデル（embedding → output）で
//! パイプライン全体の動作を検証する。loss が減少することを確認。
//!
//! ```bash
//! cargo run --release --features qat-cli --bin prepare-data -- --all
//! cargo run --release --features qat-cli --bin train-qat-70b -- \
//!     --config configs/qat_phase1_test.json
//! ```
//!
//! # Phase 2+ (実モデル学習)
//!
//! safetensors モデルが配置されている場合、レイヤー単位で
//! forward → backward → OffloadOptimizer.step() を実行。
//!
//! ```bash
//! cargo run --release --features qat-cli --bin train-qat-70b -- \
//!     --config configs/qat_8b_general.json
//! ```

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use rand::Rng;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use alice_train::llama::{LlamaConfig, QatTrainConfig};
use alice_train::{
    CheckpointData, CudaMatmul, DataLoader, DataLoaderConfig, FakeQuantize, LayerWeightGrads,
    LogEntry, LossScaler, LrScheduler, MixedPrecisionConfig, MmapDataset, OffloadConfig,
    QatConfig, TrainLog, WarmupCosineScheduler,
};

/// プロセスの RSS (Resident Set Size) を MB 単位で表示する。
fn print_rss(label: &str) {
    #[cfg(unix)]
    {
        // /proc/self/statm (Linux) から RSS を取得
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            let fields: Vec<&str> = statm.split_whitespace().collect();
            if fields.len() >= 2 {
                if let Ok(rss_pages) = fields[1].parse::<u64>() {
                    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
                    let rss_mb = rss_pages * page_size / 1024 / 1024;
                    println!("  [RSS] {label}: {rss_mb} MB");
                    return;
                }
            }
        }
        // macOS fallback: rusage
        unsafe {
            let mut usage: libc::rusage = std::mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
                // macOS: ru_maxrss は bytes、Linux: KB
                #[cfg(target_os = "macos")]
                let rss_mb = usage.ru_maxrss as u64 / 1024 / 1024;
                #[cfg(not(target_os = "macos"))]
                let rss_mb = usage.ru_maxrss as u64 / 1024;
                println!("  [RSS] {label}: {rss_mb} MB (peak)");
                return;
            }
        }
    }
    let _ = label;
}

/// ALICE-Train QAT: Llama-3 → 1.1-bit Sparse Ternary
#[derive(Parser, Debug)]
#[command(author = "Moroya Sakamoto")]
#[command(about = "Quantize Llama-3 to 1.1-bit ternary via QAT")]
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

// ── Cross-Entropy Loss ──────────────────────────────────────────────────

/// 数値安定な cross-entropy loss と勾配。
///
/// 戻り値: (loss, d_logits) where d_logits = softmax(logits) - one_hot(target)
fn cross_entropy_loss(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

    // loss = -log(probs[target])
    let loss = -(probs[target].max(1e-10)).ln();

    // gradient: softmax - one_hot
    let mut grad = probs;
    grad[target] -= 1.0;

    (loss, grad)
}

// ── メイン ────────────────────────────────────────────────────────────────

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

    let model_path_exists = Path::new(&config.model_path).exists();
    let phase1_mode = !model_path_exists;

    println!("╔══════════════════════════════════════════════════════════╗");
    if phase1_mode {
        println!("║  ALICE-Train QAT — Phase 1 パイプライン検証             ║");
    } else {
        println!("║  ALICE-Train QAT — 1.1-bit Sparse Ternary 量子化学習   ║");
    }
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    if phase1_mode {
        println!("モード: Phase 1 (ランダム初期化 — パイプライン検証)");
        println!("  モデルパス '{}' が見つかりません", config.model_path);
        println!("  → ランダム重みで embedding → output モデルを初期化");
        println!();
    }

    println!(
        "モデル: vocab={}, hidden={}, layers={}",
        model_config.vocab_size, model_config.hidden_dim, model_config.num_layers
    );
    println!("  総パラメータ: {:.2}M", total_params as f64 / 1e6);
    println!(
        "  Ternary 推定: {:.1} MB",
        ternary_bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "  パラメータ/レイヤー: {:.2}M",
        params_per_layer as f64 / 1e6
    );
    println!();

    println!("学習設定:");
    println!("  学習率: {} → {}", config.learning_rate, config.min_lr);
    println!("  ウォームアップ: {} steps", config.warmup_steps);
    println!("  総ステップ: {}", config.total_steps);
    println!(
        "  バッチ: {} × seq_len={}",
        config.batch_size, config.seq_len
    );
    println!(
        "  勾配累積: {} (実効バッチ={})",
        config.gradient_accumulation_steps,
        config.batch_size * config.gradient_accumulation_steps
    );
    println!("  BF16: {}", if config.use_bf16 { "有効" } else { "無効" });
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

    // --- データ読み込み ---
    println!("━━━ データ読み込み ━━━");
    let train_data_path = &config.train_data_path;
    let dataset = MmapDataset::open(train_data_path).unwrap_or_else(|e| {
        eprintln!(
            "[ALICE-Train] データ読み込み失敗: {train_data_path}: {e}"
        );
        eprintln!();
        eprintln!("データを準備してください:");
        eprintln!(
            "  cargo run --release --features qat-cli --bin prepare-data -- --all"
        );
        std::process::exit(1);
    });

    println!("  データ: {} ({} トークン)", train_data_path, dataset.len());

    let dl_config = DataLoaderConfig::new()
        .with_seq_len(config.seq_len)
        .with_batch_size(config.batch_size)
        .with_shuffle(true)
        .with_seed(42);

    let mut loader = DataLoader::new(&dataset, dl_config);
    let num_batches = loader.num_batches();
    println!(
        "  DataLoader: {} サンプル, {} バッチ/epoch",
        loader.num_samples(),
        num_batches
    );

    // Eval データ読み込み（オプション）
    let eval_dataset: Option<MmapDataset> = config
        .eval_data_path
        .as_ref()
        .and_then(|p| {
            if Path::new(p).exists() {
                match MmapDataset::open(p) {
                    Ok(ds) => {
                        println!("  Eval: {} ({} トークン)", p, ds.len());
                        Some(ds)
                    }
                    Err(e) => {
                        eprintln!("  Eval データ読み込み失敗: {e}");
                        None
                    }
                }
            } else {
                println!("  Eval: なし (eval_data_path 未設定 or 存在しない)");
                None
            }
        });
    println!();

    // --- コンポーネント初期化 ---
    println!("━━━ コンポーネント初期化 ━━━");

    let mut fq = FakeQuantize::new(QatConfig::ternary());
    let scheduler = WarmupCosineScheduler::new(
        config.learning_rate,
        config.min_lr,
        config.warmup_steps,
        config.total_steps,
    );

    let _scaler = if config.use_bf16 {
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

    // レジューム: resume_state.json から step を復元（delta + AdamW は load_from_disk で自動復元）
    let resume_state_path = format!("{}/resume_state.json", config.checkpoint_dir);
    if Path::new(&resume_state_path).exists() {
        if let Ok(json_str) = std::fs::read_to_string(&resume_state_path) {
            // 簡易JSONパース: {"step":123,"loss":2.5,"lr":1e-5}
            if let Some(step_str) = json_str.split("\"step\":").nth(1) {
                if let Some(step_val) = step_str.split([',', '}']).next() {
                    if let Ok(step) = step_val.trim().parse::<usize>() {
                        global_step = step;
                        println!("  resume_state.json からレジューム: step {global_step}");
                    }
                }
            }
        }
    } else if let Some(ref resume_path) = config.resume_from {
        if Path::new(resume_path).exists() {
            println!("  チェックポイントからレジューム: {resume_path}");
            match std::fs::File::open(resume_path).and_then(|mut f| CheckpointData::load(&mut f)) {
                Ok(ckpt) => {
                    global_step = ckpt.meta.step;
                    println!("  レジューム成功: step {global_step}");
                }
                Err(e) => {
                    eprintln!("  チェックポイント読み込み失敗: {e}");
                    eprintln!("  最初から学習を開始します");
                }
            }
        }
    }

    println!("  FakeQuantize: 初期化完了 (Ternary 1.58-bit, STE backward)");
    println!(
        "  Scheduler: warmup {}/{} steps",
        config.warmup_steps, config.total_steps
    );
    println!();

    // --- モデル初期化 ---
    let vocab_size = config.model.vocab_size;
    let hidden_dim = config.model.hidden_dim;

    if phase1_mode {
        // Phase 1: ランダム初期化 (embedding → output projection)
        println!("━━━ Phase 1: ランダムモデル初期化 ━━━");
        println!(
            "  Embedding: [{} × {}] ({:.1} MB)",
            vocab_size,
            hidden_dim,
            (vocab_size * hidden_dim * 4) as f64 / 1024.0 / 1024.0
        );
        println!(
            "  Output Proj: [{} × {}] ({:.1} MB)",
            hidden_dim,
            vocab_size,
            (hidden_dim * vocab_size * 4) as f64 / 1024.0 / 1024.0
        );

        let mut rng = rand::thread_rng();

        // Xavier 初期化
        let scale = (2.0 / (vocab_size + hidden_dim) as f32).sqrt();
        let mut embedding: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let mut output_proj: Vec<f32> = (0..hidden_dim * vocab_size)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();

        // 勾配バッファ
        let mut grad_embedding = vec![0.0f32; vocab_size * hidden_dim];
        let mut grad_output_proj = vec![0.0f32; hidden_dim * vocab_size];

        println!("  初期化完了");
        println!();

        // --- Phase 1 学習ループ ---
        println!(
            "━━━ Phase 1 学習開始 (step {global_step}/{}) ━━━",
            config.total_steps
        );
        let train_start = Instant::now();
        let mut batch_idx = 0usize;

        while global_step < config.total_steps {
            let lr = scheduler.get_lr(global_step);
            let step_start = Instant::now();

            // エポック境界
            if batch_idx >= num_batches {
                loader.shuffle_epoch();
                batch_idx = 0;
            }

            let batch = loader.get_batch(batch_idx, &dataset);
            batch_idx += 1;

            // --- Forward + Loss ---
            let mut total_loss = 0.0f32;
            let mut token_count = 0usize;
            let seq_len = config.seq_len;

            for b in 0..batch.actual_batch_size {
                for t in 0..seq_len {
                    let input_token = batch.input_ids[b * seq_len + t] as usize;
                    let target_token = batch.target_ids[b * seq_len + t] as usize;

                    // 範囲外トークンはスキップ
                    if input_token >= vocab_size || target_token >= vocab_size {
                        continue;
                    }

                    // Embedding lookup
                    let emb_offset = input_token * hidden_dim;

                    // Output projection: logits[v] = sum_h embedding[input][h] * output_proj[v * hidden + h]
                    let mut logits = vec![0.0f32; vocab_size];
                    for v in 0..vocab_size {
                        let proj_offset = v * hidden_dim;
                        let mut sum = 0.0f32;
                        for h in 0..hidden_dim {
                            sum = embedding[emb_offset + h]
                                .mul_add(output_proj[proj_offset + h], sum);
                        }
                        logits[v] = sum;
                    }

                    // Cross-entropy loss
                    let (loss, d_logits) = cross_entropy_loss(&logits, target_token);
                    total_loss += loss;
                    token_count += 1;

                    // --- Backward ---
                    // Output projection 勾配: d_proj[v][h] += d_logits[v] * embedding[input][h]
                    for v in 0..vocab_size {
                        let proj_offset = v * hidden_dim;
                        let dl = d_logits[v];
                        if dl.abs() < 1e-10 {
                            continue;
                        }
                        for h in 0..hidden_dim {
                            grad_output_proj[proj_offset + h] += dl * embedding[emb_offset + h];
                        }
                    }

                    // Embedding 勾配: d_emb[input][h] += sum_v d_logits[v] * output_proj[v][h]
                    for h in 0..hidden_dim {
                        let mut grad_h = 0.0f32;
                        for v in 0..vocab_size {
                            grad_h += d_logits[v] * output_proj[v * hidden_dim + h];
                        }
                        grad_embedding[emb_offset + h] += grad_h;
                    }
                }
            }

            let avg_loss = if token_count > 0 {
                total_loss / token_count as f32
            } else {
                0.0
            };
            let step_duration = step_start.elapsed();

            // --- Weight Update (SGD + weight decay) ---
            let inv_tokens = 1.0 / token_count.max(1) as f32;
            for (w, g) in embedding.iter_mut().zip(grad_embedding.iter_mut()) {
                *w -= lr * (*g * inv_tokens + config.weight_decay * *w);
                *g = 0.0;
            }
            for (w, g) in output_proj.iter_mut().zip(grad_output_proj.iter_mut()) {
                *w -= lr * (*g * inv_tokens + config.weight_decay * *w);
                *g = 0.0;
            }

            // --- ログ記録 ---
            log.append(LogEntry::new(0, global_step, avg_loss, lr, 0.0));

            // --- 進捗表示 ---
            if global_step % 10 == 0 || global_step == config.total_steps - 1 {
                let elapsed = train_start.elapsed();
                let steps_per_sec = if elapsed.as_secs() > 0 {
                    (global_step + 1) as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                let eta_secs = if steps_per_sec > 0.0 {
                    (config.total_steps - global_step) as f64 / steps_per_sec
                } else {
                    0.0
                };
                println!(
                    "  step {global_step:>5}/{} | loss: {avg_loss:.4} | lr: {lr:.2e} | \
                     {:.1}ms/step | {steps_per_sec:.1} steps/s | ETA: {eta_secs:.0}s",
                    config.total_steps,
                    step_duration.as_secs_f64() * 1000.0,
                );
            }

            // --- チェックポイント保存 ---
            if global_step > 0 && global_step % config.checkpoint_interval == 0 {
                let ckpt_path = format!("{}/step_{global_step}.bin", config.checkpoint_dir);
                println!("  チェックポイント保存: {ckpt_path}");

                let meta = alice_train::CheckpointMeta {
                    version: 1,
                    epoch: 0,
                    step: global_step,
                    loss: avg_loss,
                    learning_rate: lr,
                    weight_count: embedding.len(),
                    optimizer_state_count: 0,
                };
                let ckpt = CheckpointData {
                    meta,
                    weights: embedding.clone(),
                    optimizer_state: vec![],
                };
                if let Err(e) =
                    std::fs::File::create(&ckpt_path).and_then(|mut f| ckpt.save(&mut f))
                {
                    eprintln!("  チェックポイント保存失敗: {e}");
                }
            }

            global_step += 1;
        }

        // --- 学習完了 ---
        let total_duration = train_start.elapsed();
        println!();
        println!("━━━ Phase 1 学習完了 ━━━");
        println!("  総ステップ: {global_step}");
        println!(
            "  学習時間: {:.1}s ({:.1} steps/s)",
            total_duration.as_secs_f64(),
            global_step as f64 / total_duration.as_secs_f64().max(0.001)
        );

        // ログ保存
        let log_path = format!("{}/train_log.csv", config.checkpoint_dir);
        if let Err(e) = log.save_csv_to_file(&log_path) {
            eprintln!("  ログ保存失敗: {e}");
        } else {
            println!("  ログ: {log_path}");
        }

        // 最終 loss 表示
        if let Some(last) = log.entries().last() {
            if let Some(first) = log.entries().first() {
                println!(
                    "  Loss: {:.4} → {:.4} (Δ {:.4})",
                    first.loss,
                    last.loss,
                    first.loss - last.loss
                );
                if last.loss < first.loss {
                    println!("  ✓ Loss が減少 — パイプライン正常動作を確認");
                } else {
                    println!("  ✗ Loss が減少していません — 設定を確認してください");
                }
            }
        }
    } else {
        // Phase 2+: ストリーミング方式 — 1レイヤーずつ load/forward/backward/update
        println!("━━━ Phase 2: ストリーミング学習 ━━━");
        println!("  ※ メモリ節約: レイヤーを1つずつ読み込み・破棄");
        println!();

        use alice_train::llama::LlamaLayerWeights;
        use alice_train::llama_backward::rmsnorm_backward_output;
        use alice_train::llama_forward::rmsnorm;
        use alice_train::cuda_matmul::{cuda_layer_forward_eval_ws, cuda_layer_forward_eval_ws_vram, cuda_layer_forward_ws_vram, cuda_layer_backward_ws_vram, CudaLayerWorkspace, VramLayerWeights};
        use alice_train::safetensors_loader::ShardedModel;

        let model = ShardedModel::open(&config.model_path).unwrap_or_else(|e| {
            eprintln!("[ALICE-Train] モデル読み込み失敗: {e}");
            std::process::exit(1);
        });

        let get_tensor = |name: &str| -> Option<Vec<f32>> { model.get_tensor_f32(name) };

        // Embedding table (常駐 ~2GB)
        println!("  embedding 読み込み...");
        let embedding_table = get_tensor("model.embed_tokens.weight").unwrap_or_else(|| {
            eprintln!("[ALICE-Train] embed_tokens.weight が見つかりません");
            std::process::exit(1);
        });
        println!(
            "    embed_tokens: {:.1} MB",
            embedding_table.len() as f64 * 4.0 / 1024.0 / 1024.0
        );

        // Output norm + lm_head (常駐 ~2GB)
        let mut output_norm = get_tensor("model.norm.weight").unwrap_or_else(|| {
            eprintln!("[ALICE-Train] model.norm.weight が見つかりません");
            std::process::exit(1);
        });
        let output_proj = get_tensor("lm_head.weight").unwrap_or_else(|| {
            println!("    lm_head.weight なし — embed_tokens と共有");
            embedding_table.clone()
        });
        println!(
            "    output_norm: {} 要素, output_proj: {:.1} MB",
            output_norm.len(),
            output_proj.len() as f64 * 4.0 / 1024.0 / 1024.0
        );

        let num_layers = config.model.num_layers;

        // delta 適用: output_norm（小さいので delta ファイル方式）
        apply_global_delta(&mut output_norm, "output_norm", &config.checkpoint_dir);

        let layer_bytes = config.model.params_per_layer() * 4;

        // base_weights: preload_all_layers=true なら全 RAM 保持、false なら mmap 都度ロード
        let base_weights: Vec<LlamaLayerWeights> = if config.preload_all_layers {
            println!("  全レイヤー重みをRAMにプリロード中...");
            let weights: Vec<LlamaLayerWeights> = (0..num_layers)
                .map(|i| {
                    LlamaLayerWeights::from_tensors(i, &get_tensor, &config.model)
                        .unwrap_or_else(|| {
                            eprintln!("[ALICE-Train] レイヤー {i} の重み読み込み失敗");
                            std::process::exit(1);
                        })
                })
                .collect();
            println!(
                "    {} レイヤー読み込み完了 ({:.1} GB RAM)",
                num_layers,
                (num_layers * layer_bytes) as f64 / 1024.0 / 1024.0 / 1024.0
            );
            weights
        } else {
            // mmap モード: base_weights は空 — 都度 from_tensors で読む
            println!("  mmap モード: base weights はオンデマンドロード（RAM 節約）");
            println!(
                "    節約: {:.1} GB (全 {} レイヤー分)",
                (num_layers * layer_bytes) as f64 / 1024.0 / 1024.0 / 1024.0,
                num_layers
            );
            Vec::new()
        };

        /// base_weights がプリロード済みならインデックス参照、未ロードなら mmap から読む。
        fn get_base_layer(
            base_weights: &[LlamaLayerWeights],
            layer_idx: usize,
            get_tensor: &dyn Fn(&str) -> Option<Vec<f32>>,
            config: &LlamaConfig,
        ) -> LlamaLayerWeights {
            if !base_weights.is_empty() {
                base_weights[layer_idx].clone()
            } else {
                LlamaLayerWeights::from_tensors(layer_idx, get_tensor, config)
                    .unwrap_or_else(|| {
                        eprintln!("[ALICE-Train] レイヤー {layer_idx} の mmap 読み込み失敗");
                        std::process::exit(1);
                    })
            }
        }

        // Delta をRAMにキャッシュ（ファイルI/O排除）
        // bf16_delta=true の場合は BF16 圧縮して RAM 半減
        // ★ 1層ずつ変換してピークメモリを抑制（旧: 全層FP32→全層BF16 で2倍ピーク）
        let mut delta_cache: Vec<LayerDeltaCache>;
        let mut bf16_delta_store: Vec<Bf16DeltaStore>;
        if config.bf16_delta {
            bf16_delta_store = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                let f32_cache = LayerDeltaCache::load_from_disk(i, &config.model, &config.checkpoint_dir);
                bf16_delta_store.push(Bf16DeltaStore::from_f32(&f32_cache));
                // f32_cache は即 drop — 1層分のFP32しかピークに載らない
            }
            delta_cache = Vec::new();
            println!("    delta キャッシュ初期化完了 (BF16 圧縮: {:.1} GB)",
                (num_layers * layer_bytes / 2) as f64 / 1024.0 / 1024.0 / 1024.0);
        } else {
            delta_cache = (0..num_layers)
                .map(|i| LayerDeltaCache::load_from_disk(i, &config.model, &config.checkpoint_dir))
                .collect();
            bf16_delta_store = Vec::new();
            println!("    delta キャッシュ初期化完了 (FP32: {:.1} GB)",
                (num_layers * layer_bytes) as f64 / 1024.0 / 1024.0 / 1024.0);
        }

        // CUDA 初期化
        println!("  CUDA 初期化...");
        let cuda = CudaMatmul::new();
        println!("    CUDA ready: cuBLAS sgemm 初期化完了");

        // CudaLayerWorkspace: 全レイヤー共通バッファ（起動時1回だけ確保）
        let mut cuda_ws = CudaLayerWorkspace::new(&config.model, config.seq_len);
        let ws_mb = cuda_ws.size_bytes() as f64 / 1024.0 / 1024.0;
        println!("    CudaLayerWorkspace: {:.1} MB (forward+backward共通バッファ)", ws_mb);

        // メモリ見積もり
        let embed_mb = (embedding_table.len() + output_proj.len() + output_norm.len()) as f64 * 4.0 / 1024.0 / 1024.0;
        let layers_mb = (num_layers * layer_bytes) as f64 / 1024.0 / 1024.0;
        let delta_mb = if config.bf16_delta { layers_mb / 2.0 } else { layers_mb };
        println!();
        if config.preload_all_layers {
            println!("  メモリ見積もり (プリロード):");
            println!("    embedding + output: {:.0} MB", embed_mb);
            println!("    全レイヤー重み: {:.0} MB (RAM常駐)", layers_mb);
            println!("    delta キャッシュ: {:.0} MB (RAM常駐)", delta_mb);
            println!("    ピーク: {:.1} GB + activations",
                (embed_mb + layers_mb + delta_mb) / 1024.0);
        } else {
            println!("  メモリ見積もり (mmap モード):");
            println!("    embedding + output: {:.0} MB", embed_mb);
            println!("    base weights: mmap (OS ページキャッシュ管理)");
            println!("    delta キャッシュ: {:.0} MB (RAM常駐)", delta_mb);
            println!("    ピーク: {:.1} GB + activations",
                (embed_mb + delta_mb) / 1024.0);
        }
        // RSS 実測値表示
        print_rss("初期化完了");

        // mmap モードでは初期化後にページキャッシュを解放
        #[cfg(unix)]
        if !config.preload_all_layers {
            model.advise_dontneed_all();
            println!("  mmap ページキャッシュ解放ヒント送信済み");
        }
        println!();

        // --- シグナルハンドラ (SIGTERM/SIGINT → graceful shutdown) ---
        #[cfg(unix)]
        {
            unsafe {
                // SIGTERM (Paperspace 停止時)
                libc::signal(libc::SIGTERM, signal_handler as *const () as libc::sighandler_t);
                // SIGINT (Ctrl+C)
                libc::signal(libc::SIGINT, signal_handler as *const () as libc::sighandler_t);
            }
            println!("  SIGTERM/SIGINT ハンドラ登録済み（graceful shutdown 有効）");
        }

        // --- Phase 2 学習ループ ---
        println!(
            "━━━ Phase 2 学習開始 (step {global_step}/{}) ━━━",
            config.total_steps
        );
        let _ = std::io::stdout().flush();
        let train_start = Instant::now();
        let mut batch_idx = 0usize;

        let accum_steps = config.gradient_accumulation_steps.max(1);

        // ゼロアロケーション用ワークスペース（ループ外で1回だけ確保）
        let mut workspace_lw = get_base_layer(&base_weights, 0, &get_tensor, &config.model);
        let mut quantize_buf: Vec<f32> = Vec::new();
        let mut logits_workspace = vec![0.0f32; config.seq_len * vocab_size];
        let mut d_logits_flat = vec![0.0f32; config.seq_len * vocab_size];
        let mut d_hidden_normed_buf = vec![0.0f32; config.seq_len * hidden_dim];

        // BF16 delta モード: 1レイヤー分の FP32 作業用 delta
        let mut working_delta = if config.bf16_delta {
            Some(LayerDeltaCache::load_from_disk(0, &config.model, &config.checkpoint_dir))
        } else {
            None
        };
        let use_bf16_delta = config.bf16_delta;

        // 究極チューン2: 真の勾配累積 — micro-batch間で勾配を加算し、最後に1回だけ apply_adam
        let mut accumulated_grads: Vec<LayerWeightGrads> = (0..num_layers)
            .map(|_| LayerWeightGrads::zeros(&config.model))
            .collect();

        // AdamW オプティマイザ状態 (m, v) — 全レイヤー分を RAM に保持
        // resume時はディスクから復元（存在しなければゼロ初期化）
        let mut adam_states: Vec<LayerAdamState> = (0..num_layers)
            .map(|i| {
                if global_step > 0 {
                    LayerAdamState::load_from_disk(i, &config.model, &config.checkpoint_dir)
                } else {
                    LayerAdamState::zeros(&config.model)
                }
            })
            .collect();
        // output_norm 用 AdamW 状態
        let mut output_norm_adam_m = vec![0.0f32; hidden_dim];
        let mut output_norm_adam_v = vec![0.0f32; hidden_dim];

        let adam_mem_per_layer = adam_states[0].memory_bytes();
        let adam_mem_total = adam_mem_per_layer * num_layers + hidden_dim * 4 * 2;
        println!("  AdamW 状態: {:.1} MB/layer × {} layers = {:.1} MB",
            adam_mem_per_layer as f64 / 1024.0 / 1024.0,
            num_layers,
            adam_mem_total as f64 / 1024.0 / 1024.0,
        );
        print_rss("AdamW 初期化後");
        println!();

        while global_step < config.total_steps {

            let lr = scheduler.get_lr(global_step);
            // d_logits で既に accum_steps で割っているので、lr はそのまま使用
            let effective_lr = lr;
            let step_start = Instant::now();

            let seq_len = config.seq_len;
            let mut total_loss = 0.0f32;
            let mut token_count = 0usize;
            let mut step_mae = 0.0f64;
            let mut step_cos = 0.0f64;

            // Soft-to-Hard Quantization: alpha = 0→1 (sigmoid schedule)
            // total_steps の 75% 地点で α≈1.0 に到達し、残り 25% を α=1.0 で安定化
            let quant_warmup_steps = (config.total_steps as f32 * 0.75) as usize;
            let alpha = if global_step >= quant_warmup_steps {
                1.0f32
            } else {
                // Sigmoid schedule: 中盤で急上昇、初期と末期はなだらか
                let t = global_step as f32 / quant_warmup_steps as f32;
                let sigmoid_input = 12.0 * (t - 0.5); // -6..+6 の範囲
                1.0 / (1.0 + (-sigmoid_input).exp())
            };

            // ===== 究極チューン8: 全 micro-batch 一括 Layer-first =====
            // 全 accum_steps 分のデータを一括ロードし、expand+quantize を
            // Forward/Backward 各28回 = 計56回のみ実行（従来: 56×accum_steps 回）。

            // Phase A: 全 micro-batch × 全バッチの Embedding を一括ロード
            let mut all_hiddens: Vec<Vec<f32>> = Vec::new();
            let mut all_token_ids: Vec<Vec<u32>> = Vec::new();
            let mut all_targets: Vec<Vec<u32>> = Vec::new();
            let mut total_samples = 0usize;

            for _mb in 0..accum_steps {
                if batch_idx >= num_batches {
                    loader.shuffle_epoch();
                    batch_idx = 0;
                    fq.step_temperature();
                    println!("    [epoch boundary] temperature: {:.4}", fq.temperature());
                }

                let batch = loader.get_batch(batch_idx, &dataset);
                batch_idx += 1;

                for b in 0..batch.actual_batch_size {
                    let token_ids: Vec<u32> = batch.input_ids[b * seq_len..(b + 1) * seq_len].to_vec();
                    let targets: Vec<u32> = batch.target_ids[b * seq_len..(b + 1) * seq_len].to_vec();
                    let mut hidden = vec![0.0f32; seq_len * hidden_dim];
                    for (t, &tok) in token_ids.iter().enumerate() {
                        let tok = tok as usize;
                        if tok < vocab_size {
                            hidden[t * hidden_dim..(t + 1) * hidden_dim]
                                .copy_from_slice(&embedding_table[tok * hidden_dim..(tok + 1) * hidden_dim]);
                        }
                    }
                    all_hiddens.push(hidden);
                    all_token_ids.push(token_ids);
                    all_targets.push(targets);
                    total_samples += 1;
                }
            }

            // Phase B: Forward (Layer-first) — expand+quantize は全 micro-batch で共有
            let phase_b_start = Instant::now();
            let mut fwd_expand_ms = 0u128;
            let mut fwd_quantize_ms = 0u128;
            let mut fwd_forward_ms = 0u128;
            let mut fwd_mmap_ms = 0u128;
            let mut all_layer_inputs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                // 全サンプルの現在の hidden を保存（backward の forward 再計算用）
                let inputs: Vec<Vec<f32>> = all_hiddens.iter().map(|h| h.clone()).collect();
                all_layer_inputs.push(inputs);

                // ★ expand + quantize は全 micro-batch で1回だけ
                let t0 = Instant::now();
                let base_lw = get_base_layer(&base_weights, i, &get_tensor, &config.model);
                if use_bf16_delta {
                    let wd = working_delta.as_mut().unwrap();
                    bf16_delta_store[i].expand_into(wd);
                    fwd_expand_ms += t0.elapsed().as_millis();
                    let t1 = Instant::now();
                    wd.fused_merge_and_quantize_alpha(&base_lw, &mut workspace_lw, &mut fq, &mut quantize_buf, alpha);
                    fwd_quantize_ms += t1.elapsed().as_millis();
                } else {
                    fwd_expand_ms += t0.elapsed().as_millis();
                    let t1 = Instant::now();
                    delta_cache[i].fused_merge_and_quantize_alpha(&base_lw, &mut workspace_lw, &mut fq, &mut quantize_buf, alpha);
                    fwd_quantize_ms += t1.elapsed().as_millis();
                }

                // [DEBUG] 全レイヤーの重み診断（step 0のみ）
                if global_step == 0 {
                    let mut any_nan = false;
                    let mut any_inf = false;
                    for (name, w) in [
                        ("q_proj", &workspace_lw.q_proj),
                        ("k_proj", &workspace_lw.k_proj),
                        ("v_proj", &workspace_lw.v_proj),
                        ("o_proj", &workspace_lw.o_proj),
                        ("gate", &workspace_lw.gate_proj),
                        ("up", &workspace_lw.up_proj),
                        ("down", &workspace_lw.down_proj),
                        ("attn_n", &workspace_lw.attn_norm),
                        ("ffn_n", &workspace_lw.ffn_norm),
                    ] {
                        let has_nan = w.iter().any(|x| x.is_nan());
                        let has_inf = w.iter().any(|x| x.is_infinite());
                        if has_nan || has_inf {
                            let w_min = w.iter().cloned().fold(f32::INFINITY, f32::min);
                            let w_max = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            println!("    [DEBUG] L{i} {name}: min={w_min:.4} max={w_max:.4} NaN={has_nan} Inf={has_inf} len={}", w.len());
                            any_nan |= has_nan;
                            any_inf |= has_inf;
                        }
                    }
                    if any_nan || any_inf {
                        println!("    [DEBUG] *** L{i} に NaN/Inf 重みあり! ***");
                    } else if i == 0 || i == 25 || i == 26 || i == 27 {
                        // 注目レイヤーは正常でも出力
                        let w_min = workspace_lw.q_proj.iter().cloned().fold(f32::INFINITY, f32::min);
                        let w_max = workspace_lw.q_proj.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        println!("    [DEBUG] L{i} weights OK (q_proj: {w_min:.4}..{w_max:.4})");
                    }
                    if i == 0 {
                        let h_min = all_hiddens[0].iter().cloned().fold(f32::INFINITY, f32::min);
                        let h_max = all_hiddens[0].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let h_nan = all_hiddens[0].iter().any(|x| x.is_nan());
                        println!("    [DEBUG] hidden pre-L0: min={h_min:.6} max={h_max:.6} nan={h_nan}");
                    }
                }

                // 全サンプルを同じ量子化済み重みで forward
                let t2 = Instant::now();
                for b in 0..total_samples {
                    cuda_layer_forward_eval_ws(&cuda, &mut all_hiddens[b], &workspace_lw, &config.model, seq_len, &mut cuda_ws);
                }
                fwd_forward_ms += t2.elapsed().as_millis();

                // [DEBUG] forward後のhidden診断（各レイヤー後、最初のステップのみ）
                if global_step == 0 {
                    let h_min = all_hiddens[0].iter().cloned().fold(f32::INFINITY, f32::min);
                    let h_max = all_hiddens[0].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let h_nan = all_hiddens[0].iter().any(|x| x.is_nan());
                    let h_inf = all_hiddens[0].iter().any(|x| x.is_infinite());
                    println!("    [DEBUG] hidden post-L{i}: min={h_min:.6} max={h_max:.6} nan={h_nan} inf={h_inf}");
                }

                // mmap page cache 解放
                let t3 = Instant::now();
                #[cfg(unix)]
                if !config.preload_all_layers {
                    model.advise_dontneed_all();
                }
                fwd_mmap_ms += t3.elapsed().as_millis();
            }
            println!("    [TIMING] Phase B Forward: expand={fwd_expand_ms}ms quantize={fwd_quantize_ms}ms forward={fwd_forward_ms}ms mmap={fwd_mmap_ms}ms total={}ms (samples={total_samples})", phase_b_start.elapsed().as_millis());

            // Phase C: Loss + Output backward (全サンプル一括)
            let phase_c_start = Instant::now();
            let mut all_d_hiddens: Vec<Vec<f32>> = Vec::with_capacity(total_samples);
            let mut d_output_norm_w_accum = vec![0.0f32; hidden_dim];
            for b in 0..total_samples {
                // Output RMSNorm + projection
                let hidden_pre_norm = all_hiddens[b].clone();
                rmsnorm(&mut all_hiddens[b], &output_norm, hidden_dim, config.model.norm_eps);

                // logits — CUDA cuBLAS
                cuda.matmul_bt_inplace(&all_hiddens[b], &output_proj, &mut logits_workspace, seq_len, vocab_size, hidden_dim);

                // Loss + 勾配計算
                d_logits_flat.iter_mut().for_each(|x| *x = 0.0);

                // [DEBUG] logits 診断（最初のサンプル・最初のステップのみ）
                if b == 0 && global_step == 0 {
                    let logits_t0 = &logits_workspace[0..vocab_size];
                    let l_min = logits_t0.iter().cloned().fold(f32::INFINITY, f32::min);
                    let l_max = logits_t0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let l_nan = logits_t0.iter().any(|x| x.is_nan());
                    let l_inf = logits_t0.iter().any(|x| x.is_infinite());
                    println!("    [DEBUG] logits[0]: min={l_min:.4} max={l_max:.4} nan={l_nan} inf={l_inf}");

                    // softmax して正解トークンの確率を確認
                    let target0 = all_targets[b][0] as usize;
                    if target0 < vocab_size {
                        let max_l = l_max;
                        let sum_exp: f64 = logits_t0.iter().map(|&x| ((x - max_l) as f64).exp()).sum();
                        let p_target = ((logits_t0[target0] - max_l) as f64).exp() / sum_exp;
                        println!("    [DEBUG] target[0]={target0} p(target)={p_target:.10e} -ln(p)={:.4}", -(p_target.max(1e-30)).ln());
                    }
                }

                let mut tokens_this_batch = 0usize;
                for t in 0..seq_len {
                    let target = all_targets[b][t] as usize;
                    if target >= vocab_size {
                        continue;
                    }
                    let logits_t = &logits_workspace[t * vocab_size..(t + 1) * vocab_size];
                    let (loss, grad) = cross_entropy_loss(logits_t, target);
                    total_loss += loss;
                    token_count += 1;
                    tokens_this_batch += 1;
                    d_logits_flat[t * vocab_size..(t + 1) * vocab_size].copy_from_slice(&grad);
                }

                // Mean化: d_logits を有効トークン数 × 勾配累積ステップ数で正規化
                // これにより backward 全体のスケールが適正化され、grad_norm が数十〜数百に収まる
                let d_scale = 1.0 / (tokens_this_batch.max(1) * accum_steps) as f32;
                for v in d_logits_flat.iter_mut() {
                    *v *= d_scale;
                }

                // [DEBUG] d_logits ノルム（最初のサンプル・最初のステップのみ）
                if b == 0 && global_step == 0 {
                    let d_logits_norm: f64 = d_logits_flat.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                    let d_logits_max = d_logits_flat.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
                    println!("    [DEBUG] d_logits: L2={d_logits_norm:.6e} max_abs={d_logits_max:.6e}");
                }

                // Output projection backward
                cuda.matmul_nn_inplace(
                    &d_logits_flat, &output_proj, &mut d_hidden_normed_buf,
                    seq_len, hidden_dim, vocab_size,
                );

                // Output RMSNorm backward
                let mut d_hidden = vec![0.0f32; seq_len * hidden_dim];
                let mut d_output_norm_w = vec![0.0f32; hidden_dim];
                rmsnorm_backward_output(
                    &d_hidden_normed_buf,
                    &hidden_pre_norm,
                    &output_norm,
                    &mut d_hidden,
                    &mut d_output_norm_w,
                    hidden_dim,
                    config.model.norm_eps,
                );

                // output_norm 勾配を累積
                for h in 0..hidden_dim {
                    d_output_norm_w_accum[h] += d_output_norm_w[h];
                }

                all_d_hiddens.push(d_hidden);
            }
            // output_norm: 全 micro-batch 完了後に AdamW 更新
            // d_logits で既に Mean 化済みなので inv_tokens 不要
            {
                let beta1: f32 = 0.9;
                let beta2: f32 = 0.999;
                let eps: f32 = 1e-8;
                let adam_t = (global_step + 1) as f32;
                let bc1 = 1.0 - beta1.powf(adam_t);
                let bc2 = 1.0 - beta2.powf(adam_t);
                for h in 0..hidden_dim {
                    let g = d_output_norm_w_accum[h];
                    output_norm_adam_m[h] = beta1 * output_norm_adam_m[h] + (1.0 - beta1) * g;
                    output_norm_adam_v[h] = beta2 * output_norm_adam_v[h] + (1.0 - beta2) * g * g;
                    let m_hat = output_norm_adam_m[h] / bc1;
                    let v_hat = output_norm_adam_v[h] / bc2;
                    output_norm[h] -= effective_lr * (m_hat / (v_hat.sqrt() + eps) + config.weight_decay * output_norm[h]);
                }
            }
            drop(all_hiddens);
            drop(all_token_ids);
            drop(all_targets);
            println!("    [TIMING] Phase C Loss+OutBwd: {}ms", phase_c_start.elapsed().as_millis());

            // Phase D: Backward (Layer-first 逆順) — expand+quantize は全 micro-batch で共有
            let phase_d_start = Instant::now();
            let mut bwd_expand_ms = 0u128;
            let mut bwd_quantize_ms = 0u128;
            let mut bwd_vram_ms = 0u128;
            let mut bwd_backward_ms = 0u128;
            let mut bwd_mmap_ms = 0u128;
            for i in (0..num_layers).rev() {
                let base_lw = get_base_layer(&base_weights, i, &get_tensor, &config.model);
                let t0 = Instant::now();
                if use_bf16_delta {
                    let wd = working_delta.as_mut().unwrap();
                    bf16_delta_store[i].expand_into(wd);
                    bwd_expand_ms += t0.elapsed().as_millis();
                    let t1 = Instant::now();
                    let (mae, cos, cnt) = wd.fused_merge_and_quantize_alpha(&base_lw, &mut workspace_lw, &mut fq, &mut quantize_buf, alpha);
                    bwd_quantize_ms += t1.elapsed().as_millis();
                    if i == 0 {
                        step_mae = mae / cnt.max(1) as f64;
                        step_cos = cos / cnt.max(1) as f64;
                    }
                } else {
                    bwd_expand_ms += t0.elapsed().as_millis();
                    let t1 = Instant::now();
                    let (mae, cos, cnt) = delta_cache[i].fused_merge_and_quantize_alpha(&base_lw, &mut workspace_lw, &mut fq, &mut quantize_buf, alpha);
                    bwd_quantize_ms += t1.elapsed().as_millis();
                    if i == 0 {
                        step_mae = mae / cnt.max(1) as f64;
                        step_cos = cos / cnt.max(1) as f64;
                    }
                }

                // VRAM常駐 — 全サンプルで使い回し
                let t2 = Instant::now();
                let vram_w = VramLayerWeights::upload(&cuda, &workspace_lw);
                bwd_vram_ms += t2.elapsed().as_millis();

                // 全サンプルの backward を同じ重みで計算
                let t3 = Instant::now();
                let mut _fwd_recomp_us = 0u128;
                let mut _bwd_core_us = 0u128;
                let mut _clip_us = 0u128;
                let mut _accum_us = 0u128;
                for b in 0..total_samples {
                    let _tb0 = Instant::now();
                    let mut recompute_hidden = std::mem::take(&mut all_layer_inputs[i][b]);
                    let cache = cuda_layer_forward_ws_vram(&cuda, &mut recompute_hidden, &workspace_lw, &vram_w, &config.model, seq_len, &mut cuda_ws);
                    _fwd_recomp_us += _tb0.elapsed().as_micros();

                    let _tb1 = Instant::now();
                    let (d_input, grads) = cuda_layer_backward_ws_vram(
                        &cuda,
                        &all_d_hiddens[b],
                        &cache,
                        &workspace_lw,
                        &vram_w,
                        &config.model,
                        seq_len,
                        &mut cuda_ws,
                    );
                    all_d_hiddens[b] = d_input;
                    _bwd_core_us += _tb1.elapsed().as_micros();

                    // d_hidden clipping: 各レイヤー後にL2ノルムを制限
                    let _tb2 = Instant::now();
                    let dh = &mut all_d_hiddens[b];
                    let dh_norm: f64 = dh.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                    let max_dh_norm = 100.0f64;
                    if dh_norm > max_dh_norm {
                        let scale = (max_dh_norm / dh_norm) as f32;
                        for v in dh.iter_mut() { *v *= scale; }
                    }
                    // NaN/Inf ガード
                    for v in dh.iter_mut() {
                        if v.is_nan() || v.is_infinite() { *v = 0.0; }
                    }
                    _clip_us += _tb2.elapsed().as_micros();

                    // 勾配累積
                    let _tb3 = Instant::now();
                    accumulated_grads[i].accumulate(&grads);
                    _accum_us += _tb3.elapsed().as_micros();
                }
                bwd_backward_ms += t3.elapsed().as_millis();
                // 最初のステップの最初の3レイヤーだけ詳細出力
                if global_step == 0 && i >= num_layers - 3 {
                    eprintln!("    [BWD-LOOP L{i}] fwd_recomp={:.1}ms bwd_core={:.1}ms clip={:.1}ms accum={:.1}ms total={:.1}ms",
                        _fwd_recomp_us as f64 / 1000.0, _bwd_core_us as f64 / 1000.0,
                        _clip_us as f64 / 1000.0, _accum_us as f64 / 1000.0,
                        t3.elapsed().as_secs_f64() * 1000.0);
                }

                // mmap page cache 解放（OOM 防止）
                let t4 = Instant::now();
                #[cfg(unix)]
                if !config.preload_all_layers {
                    model.advise_dontneed_all();
                }
                bwd_mmap_ms += t4.elapsed().as_millis();
            }
            println!("    [TIMING] Phase D Backward: expand={bwd_expand_ms}ms quantize={bwd_quantize_ms}ms vram={bwd_vram_ms}ms backward={bwd_backward_ms}ms mmap={bwd_mmap_ms}ms total={}ms", phase_d_start.elapsed().as_millis());

            // 勾配NaN検出 + ガード（全ステップ）
            let mut grad_has_nan = false;
            let mut grad_nan_layers = Vec::new();
            for i in 0..num_layers {
                let g = &mut accumulated_grads[i];
                let sanitize = |s: &mut [f32]| -> usize {
                    let mut cnt = 0usize;
                    for v in s.iter_mut() {
                        if v.is_nan() || v.is_infinite() {
                            *v = 0.0;
                            cnt += 1;
                        }
                    }
                    cnt
                };
                let mut layer_nan_count = 0usize;
                layer_nan_count += sanitize(&mut g.attn_norm);
                layer_nan_count += sanitize(&mut g.q_proj);
                layer_nan_count += sanitize(&mut g.k_proj);
                layer_nan_count += sanitize(&mut g.v_proj);
                layer_nan_count += sanitize(&mut g.o_proj);
                layer_nan_count += sanitize(&mut g.ffn_norm);
                layer_nan_count += sanitize(&mut g.gate_proj);
                layer_nan_count += sanitize(&mut g.up_proj);
                layer_nan_count += sanitize(&mut g.down_proj);
                if let Some(ref mut b) = g.q_bias { layer_nan_count += sanitize(b); }
                if let Some(ref mut b) = g.k_bias { layer_nan_count += sanitize(b); }
                if let Some(ref mut b) = g.v_bias { layer_nan_count += sanitize(b); }
                if layer_nan_count > 0 {
                    grad_has_nan = true;
                    grad_nan_layers.push((i, layer_nan_count));
                }
            }
            if grad_has_nan {
                println!("    [WARN] NaN/Inf勾配検出! layers: {grad_nan_layers:?} → ゼロ置換済み");
            }

            // [DEBUG] 勾配診断（最初の5ステップ + 100ステップごと）
            if global_step < 5 || global_step % 100 == 0 {
                let g = &accumulated_grads[0];
                let q_norm: f64 = g.q_proj.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                let gate_norm: f64 = g.gate_proj.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                let g27 = &accumulated_grads[num_layers - 1];
                let q27_norm: f64 = g27.q_proj.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                println!("    [DEBUG] grads: L0 q={q_norm:.4e} gate={gate_norm:.4e} | L27 q={q27_norm:.4e}");
            }

            // メモリ解放
            drop(all_layer_inputs);
            drop(all_d_hiddens);

            // 全 micro-batch 完了後に一括で重み更新
            // Gradient Clipping: 全レイヤーの勾配を結合してグローバル L2 Norm を計算し、
            // max_grad_norm (1.0) を超えていたら全体をスケーリング
            let max_grad_norm = config.max_grad_norm;
            let mut global_grad_sq_sum: f64 = 0.0;
            for i in 0..num_layers {
                let g = &accumulated_grads[i];
                for field in [&g.attn_norm, &g.q_proj, &g.k_proj, &g.v_proj, &g.o_proj,
                              &g.ffn_norm, &g.gate_proj, &g.up_proj, &g.down_proj] {
                    global_grad_sq_sum += field.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
                }
            }
            let global_grad_norm = global_grad_sq_sum.sqrt();
            let clip_scale = if global_grad_norm > max_grad_norm as f64 && global_grad_norm > 1e-10 {
                (max_grad_norm as f64) / global_grad_norm
            } else {
                1.0
            };
            if global_step < 5 || global_step % 100 == 0 {
                println!("    [DEBUG] global_grad_norm={global_grad_norm:.4e} clip_scale={clip_scale:.6e}");
            }
            if clip_scale < 1.0 {
                let s = clip_scale as f32;
                for i in 0..num_layers {
                    let g = &mut accumulated_grads[i];
                    for v in g.attn_norm.iter_mut() { *v *= s; }
                    for v in g.q_proj.iter_mut() { *v *= s; }
                    for v in g.k_proj.iter_mut() { *v *= s; }
                    for v in g.v_proj.iter_mut() { *v *= s; }
                    for v in g.o_proj.iter_mut() { *v *= s; }
                    for v in g.ffn_norm.iter_mut() { *v *= s; }
                    for v in g.gate_proj.iter_mut() { *v *= s; }
                    for v in g.up_proj.iter_mut() { *v *= s; }
                    for v in g.down_proj.iter_mut() { *v *= s; }
                }
            }

            let apply_start = Instant::now();
            let adam_step = global_step + 1; // bias correction は 1-indexed
            let mut _adam_expand_us = 0u128;
            let mut _adam_step_us = 0u128;
            let mut _adam_compress_us = 0u128;
            let mut _adam_zero_us = 0u128;
            for i in 0..num_layers {
                if use_bf16_delta {
                    let wd = working_delta.as_mut().unwrap();
                    let _ta0 = Instant::now();
                    bf16_delta_store[i].expand_into(wd);
                    _adam_expand_us += _ta0.elapsed().as_micros();
                    let _ta1 = Instant::now();
                    wd.apply_adam(&accumulated_grads[i], &mut adam_states[i], effective_lr, config.weight_decay, adam_step);
                    _adam_step_us += _ta1.elapsed().as_micros();
                    // apply_adam 後の delta NaN チェック
                    let delta_nan = wd.q_proj.iter().any(|x| x.is_nan());
                    if delta_nan && (global_step < 5 || i == 0 || i == 26) {
                        println!("    [WARN] delta L{i} にNaN検出 (apply_adam後)");
                    }
                    let _ta2 = Instant::now();
                    bf16_delta_store[i].update_from_f32(wd);
                    _adam_compress_us += _ta2.elapsed().as_micros();
                } else {
                    let _ta1 = Instant::now();
                    delta_cache[i].apply_adam(&accumulated_grads[i], &mut adam_states[i], effective_lr, config.weight_decay, adam_step);
                    _adam_step_us += _ta1.elapsed().as_micros();
                }
                let _ta3 = Instant::now();
                accumulated_grads[i].zero_out();
                _adam_zero_us += _ta3.elapsed().as_micros();
            }
            println!("    [TIMING] Apply AdamW: {}ms (expand={:.1}ms adam={:.1}ms compress={:.1}ms zero={:.1}ms)",
                apply_start.elapsed().as_millis(),
                _adam_expand_us as f64 / 1000.0, _adam_step_us as f64 / 1000.0,
                _adam_compress_us as f64 / 1000.0, _adam_zero_us as f64 / 1000.0);

            // [DEBUG] delta 変化量（最初のステップのみ）
            if global_step == 0 {
                if use_bf16_delta {
                    let wd = working_delta.as_mut().unwrap();
                    bf16_delta_store[0].expand_into(wd);
                    let delta_norm: f64 = wd.q_proj.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                    let delta_max = wd.q_proj.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
                    println!("    [DEBUG] delta L0 q_proj after update: L2={delta_norm:.4e} max={delta_max:.4e}");
                }
            }

            let accumulated_loss = if token_count > 0 {
                total_loss / token_count as f32
            } else {
                0.0
            };
            let step_duration = step_start.elapsed();

            log.append(LogEntry::new(0, global_step, accumulated_loss, lr, 0.0));

            if global_step % 10 == 0 || global_step == config.total_steps - 1 {
                let elapsed = train_start.elapsed();
                let steps_per_sec = if elapsed.as_secs() > 0 {
                    (global_step + 1) as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                let eta_secs = if steps_per_sec > 0.0 {
                    (config.total_steps - global_step) as f64 / steps_per_sec
                } else {
                    0.0
                };
                println!(
                    "  step {global_step:>5}/{} | loss: {accumulated_loss:.4} | lr: {lr:.2e} | \
                     alpha: {alpha:.3} | qat_mae: {step_mae:.4} | cos: {step_cos:.4} | temp: {:.3} | \
                     {:.1}ms/step | {steps_per_sec:.1} steps/s | ETA: {eta_secs:.0}s",
                    config.total_steps,
                    fq.temperature(),
                    step_duration.as_secs_f64() * 1000.0,
                );
                // 最初の5ステップとその後100ステップごとにRSSを表示
                if global_step < 5 || global_step % 100 == 0 {
                    print_rss(&format!("step {global_step}"));
                }
                let _ = std::io::stdout().flush();
            }

            // Eval (eval_data_path が設定されている場合)
            // ★ Layer-first Streaming: 量子化を28回に削減（従来: バッチ数×28回）
            if config.eval_interval > 0
                && global_step % config.eval_interval == 0
                && eval_dataset.is_some()
            {
                println!("    [eval] 評価開始 (Layer-first Streaming)...");
                let _ = std::io::stdout().flush();
                let eval_start = Instant::now();

                let eval_ds = eval_dataset.as_ref().unwrap();
                let eval_dl_config = DataLoaderConfig::new()
                    .with_seq_len(config.seq_len)
                    .with_batch_size(config.batch_size)
                    .with_shuffle(false);
                let eval_loader = DataLoader::new(eval_ds, eval_dl_config);
                let eval_batches = eval_loader.num_batches().min(50);

                // Phase 1: 全バッチの embedding → hidden 状態とターゲットをメモリに保持
                let mut all_hiddens: Vec<Vec<f32>> = Vec::with_capacity(eval_batches);
                let mut all_targets: Vec<Vec<u32>> = Vec::with_capacity(eval_batches);

                for eb in 0..eval_batches {
                    let eval_batch = eval_loader.get_batch(eb, eval_ds);
                    let token_ids = &eval_batch.input_ids[0..seq_len];
                    let targets: Vec<u32> = eval_batch.target_ids[0..seq_len].to_vec();

                    let mut hidden_b = vec![0.0f32; seq_len * hidden_dim];
                    for (t, &tok) in token_ids.iter().enumerate() {
                        let tok = tok as usize;
                        if tok < vocab_size {
                            hidden_b[t * hidden_dim..(t + 1) * hidden_dim]
                                .copy_from_slice(&embedding_table[tok * hidden_dim..(tok + 1) * hidden_dim]);
                        }
                    }
                    all_hiddens.push(hidden_b);
                    all_targets.push(targets);
                }

                // Phase 2: レイヤー・ファースト + VRAM常駐（量子化1回 + 重みH2D転送1回/レイヤー）
                for i in 0..num_layers {
                    let base_lw = get_base_layer(&base_weights, i, &get_tensor, &config.model);
                    if use_bf16_delta {
                        let wd = working_delta.as_mut().unwrap();
                        bf16_delta_store[i].expand_into(wd);
                        wd.fused_merge_and_quantize(&base_lw, &mut workspace_lw, &mut fq, &mut quantize_buf);
                    } else {
                        delta_cache[i].fused_merge_and_quantize(&base_lw, &mut workspace_lw, &mut fq, &mut quantize_buf);
                    }

                    // 量子化済み重みを1回だけ VRAM に転送
                    let vram_w = VramLayerWeights::upload(&cuda, &workspace_lw);

                    // 全バッチで VRAM 上の重みを直接参照（重みの H2D 転送ゼロ）
                    for eb in 0..eval_batches {
                        cuda_layer_forward_eval_ws_vram(&cuda, &mut all_hiddens[eb], &workspace_lw, &vram_w, &config.model, seq_len, &mut cuda_ws);
                    }
                    // vram_w はスコープ抜けで自動解放 → 次レイヤーの VRAM に再利用
                }

                // Phase 3: Output Norm & Logits → Loss 計算
                let mut eval_loss_sum = 0.0f32;
                let mut eval_token_count = 0usize;

                for eb in 0..eval_batches {
                    let hidden = &mut all_hiddens[eb];
                    rmsnorm(hidden, &output_norm, hidden_dim, config.model.norm_eps);
                    cuda.matmul_bt_inplace(hidden, &output_proj, &mut logits_workspace, seq_len, vocab_size, hidden_dim);
                    for t in 0..seq_len {
                        let target = all_targets[eb][t] as usize;
                        if target < vocab_size {
                            let logits_t = &logits_workspace[t * vocab_size..(t + 1) * vocab_size];
                            let (loss, _) = cross_entropy_loss(logits_t, target);
                            eval_loss_sum += loss;
                            eval_token_count += 1;
                        }
                    }
                }

                let eval_loss = if eval_token_count > 0 {
                    eval_loss_sum / eval_token_count as f32
                } else {
                    0.0
                };
                let eval_ppl = eval_loss.exp();
                let eval_elapsed = eval_start.elapsed().as_secs_f32();
                println!(
                    "    [eval] step {global_step} | eval_loss: {eval_loss:.4} | ppl: {eval_ppl:.1} | {eval_token_count} tokens | {eval_elapsed:.1}s"
                );
                let _ = std::io::stdout().flush();
            }

            // チェックポイント: step 番号 + delta + AdamW state を保存
            if global_step > 0 && global_step % config.checkpoint_interval == 0 {
                let ckpt_start = Instant::now();
                // Delta キャッシュをディスクにフラッシュ
                if use_bf16_delta {
                    for (i, store) in bf16_delta_store.iter().enumerate() {
                        store.save_to_disk(i, &config.checkpoint_dir);
                    }
                } else {
                    for (i, dc) in delta_cache.iter().enumerate() {
                        dc.save_to_disk(i, &config.checkpoint_dir);
                    }
                }
                // AdamW 状態をディスクにフラッシュ
                for (i, adam) in adam_states.iter().enumerate() {
                    adam.save_to_disk(i, &config.checkpoint_dir);
                }
                let ckpt_path = format!("{}/step_{global_step}.bin", config.checkpoint_dir);
                let meta = alice_train::CheckpointMeta {
                    version: 1,
                    epoch: 0,
                    step: global_step,
                    loss: accumulated_loss,
                    learning_rate: lr,
                    weight_count: embedding_table.len(),
                    optimizer_state_count: 0,
                };
                let ckpt = CheckpointData {
                    meta,
                    weights: embedding_table.clone(),
                    optimizer_state: vec![],
                };
                if let Err(e) =
                    std::fs::File::create(&ckpt_path).and_then(|mut f| ckpt.save(&mut f))
                {
                    eprintln!("  チェックポイント保存失敗: {e}");
                }
                // resume_state.json: 再起動時に step/loss を自動復元
                let resume_json = format!(
                    "{{\"step\":{global_step},\"loss\":{accumulated_loss:.6},\"lr\":{lr:.2e}}}",
                );
                let resume_path = format!("{}/resume_state.json", config.checkpoint_dir);
                let _ = std::fs::write(&resume_path, &resume_json);
                println!("  チェックポイント保存: {ckpt_path} (delta+adam+resume {:.1}s)", ckpt_start.elapsed().as_secs_f32());
            }

            global_step += 1;

            // Graceful shutdown: SIGTERM/SIGINT を受信したらチェックポイント保存して終了
            #[cfg(unix)]
            if SHUTDOWN_FLAG.load(Ordering::SeqCst) {
                println!("\n━━━ Graceful Shutdown (step {global_step}) ━━━");
                println!("  Delta + AdamW をディスクにフラッシュ中...");
                if use_bf16_delta {
                    for (i, store) in bf16_delta_store.iter().enumerate() {
                        store.save_to_disk(i, &config.checkpoint_dir);
                    }
                } else {
                    for (i, dc) in delta_cache.iter().enumerate() {
                        dc.save_to_disk(i, &config.checkpoint_dir);
                    }
                }
                for (i, adam) in adam_states.iter().enumerate() {
                    adam.save_to_disk(i, &config.checkpoint_dir);
                }
                let ckpt_path = format!("{}/step_{global_step}.bin", config.checkpoint_dir);
                let meta = alice_train::CheckpointMeta {
                    version: 1,
                    epoch: 0,
                    step: global_step,
                    loss: accumulated_loss,
                    learning_rate: lr,
                    weight_count: embedding_table.len(),
                    optimizer_state_count: 0,
                };
                let ckpt = CheckpointData {
                    meta,
                    weights: embedding_table.clone(),
                    optimizer_state: vec![],
                };
                if let Err(e) =
                    std::fs::File::create(&ckpt_path).and_then(|mut f| ckpt.save(&mut f))
                {
                    eprintln!("  チェックポイント保存失敗: {e}");
                } else {
                    println!("  チェックポイント保存完了: {ckpt_path}");
                }
                let resume_json = format!(
                    "{{\"step\":{global_step},\"loss\":{accumulated_loss:.6},\"lr\":{lr:.2e}}}",
                );
                let resume_path = format!("{}/resume_state.json", config.checkpoint_dir);
                let _ = std::fs::write(&resume_path, &resume_json);
                println!("━━━ 学習中断（再開可能） ━━━");
                let _ = std::io::stdout().flush();
                break;
            }
        }

        // 学習終了時に delta + AdamW 状態をディスクにフラッシュ
        if use_bf16_delta {
            for (i, store) in bf16_delta_store.iter().enumerate() {
                store.save_to_disk(i, &config.checkpoint_dir);
            }
        } else {
            for (i, dc) in delta_cache.iter().enumerate() {
                dc.save_to_disk(i, &config.checkpoint_dir);
            }
        }
        for (i, adam) in adam_states.iter().enumerate() {
            adam.save_to_disk(i, &config.checkpoint_dir);
        }
        // resume_state.json: 再起動時に step/loss を自動復元
        let final_lr = scheduler.get_lr(global_step);
        let final_loss = log.entries().last().map_or(0.0, |e| e.loss);
        let resume_json = format!(
            "{{\"step\":{global_step},\"loss\":{final_loss:.6},\"lr\":{final_lr:.2e}}}",
        );
        let resume_path = format!("{}/resume_state.json", config.checkpoint_dir);
        let _ = std::fs::write(&resume_path, &resume_json);

        let total_duration = train_start.elapsed();
        println!();
        println!("━━━ Phase 2 学習完了 ━━━");
        println!("  総ステップ: {global_step}");
        println!(
            "  学習時間: {:.1}s ({:.1} steps/s)",
            total_duration.as_secs_f64(),
            global_step as f64 / total_duration.as_secs_f64().max(0.001)
        );

        let log_path = format!("{}/train_log.csv", config.checkpoint_dir);
        if let Err(e) = log.save_csv_to_file(&log_path) {
            eprintln!("  ログ保存失敗: {e}");
        } else {
            println!("  ログ: {log_path}");
        }

        if let Some(last) = log.entries().last() {
            if let Some(first) = log.entries().first() {
                println!(
                    "  Loss: {:.4} → {:.4} (Δ {:.4})",
                    first.loss,
                    last.loss,
                    first.loss - last.loss
                );
            }
        }
    }
}

// ── AdamW オプティマイザ状態 ──────────────────────────────────────────────

/// レイヤーごとの AdamW m (1次モーメント) / v (2次モーメント) 状態。
/// LayerDeltaCache と同じフィールド構成。
struct LayerAdamState {
    attn_norm_m: Vec<f32>,
    attn_norm_v: Vec<f32>,
    q_proj_m: Vec<f32>,
    q_proj_v: Vec<f32>,
    k_proj_m: Vec<f32>,
    k_proj_v: Vec<f32>,
    v_proj_m: Vec<f32>,
    v_proj_v: Vec<f32>,
    o_proj_m: Vec<f32>,
    o_proj_v: Vec<f32>,
    ffn_norm_m: Vec<f32>,
    ffn_norm_v: Vec<f32>,
    gate_proj_m: Vec<f32>,
    gate_proj_v: Vec<f32>,
    up_proj_m: Vec<f32>,
    up_proj_v: Vec<f32>,
    down_proj_m: Vec<f32>,
    down_proj_v: Vec<f32>,
    q_bias_m: Option<Vec<f32>>,
    q_bias_v: Option<Vec<f32>>,
    k_bias_m: Option<Vec<f32>>,
    k_bias_v: Option<Vec<f32>>,
    v_bias_m: Option<Vec<f32>>,
    v_bias_v: Option<Vec<f32>>,
}

impl LayerAdamState {
    /// ゼロ初期化。LayerDeltaCache と同じサイズ。
    fn zeros(config: &LlamaConfig) -> Self {
        let hidden = config.hidden_dim;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let inter = config.intermediate_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let (q_bias_m, q_bias_v, k_bias_m, k_bias_v, v_bias_m, v_bias_v) = if config.attention_bias {
            (
                Some(vec![0.0f32; q_dim]), Some(vec![0.0f32; q_dim]),
                Some(vec![0.0f32; kv_dim]), Some(vec![0.0f32; kv_dim]),
                Some(vec![0.0f32; kv_dim]), Some(vec![0.0f32; kv_dim]),
            )
        } else {
            (None, None, None, None, None, None)
        };

        Self {
            attn_norm_m: vec![0.0; hidden], attn_norm_v: vec![0.0; hidden],
            q_proj_m: vec![0.0; q_dim * hidden], q_proj_v: vec![0.0; q_dim * hidden],
            k_proj_m: vec![0.0; kv_dim * hidden], k_proj_v: vec![0.0; kv_dim * hidden],
            v_proj_m: vec![0.0; kv_dim * hidden], v_proj_v: vec![0.0; kv_dim * hidden],
            o_proj_m: vec![0.0; hidden * q_dim], o_proj_v: vec![0.0; hidden * q_dim],
            ffn_norm_m: vec![0.0; hidden], ffn_norm_v: vec![0.0; hidden],
            gate_proj_m: vec![0.0; inter * hidden], gate_proj_v: vec![0.0; inter * hidden],
            up_proj_m: vec![0.0; inter * hidden], up_proj_v: vec![0.0; inter * hidden],
            down_proj_m: vec![0.0; hidden * inter], down_proj_v: vec![0.0; hidden * inter],
            q_bias_m, q_bias_v, k_bias_m, k_bias_v, v_bias_m, v_bias_v,
        }
    }

    /// AdamW 状態をディスクに保存。
    fn save_to_disk(&self, layer_idx: usize, checkpoint_dir: &str) {
        let sv = |name: &str, data: &[f32]| save_layer_delta(layer_idx, name, data, checkpoint_dir);
        sv("adam_attn_norm_m", &self.attn_norm_m);
        sv("adam_attn_norm_v", &self.attn_norm_v);
        sv("adam_q_proj_m", &self.q_proj_m);
        sv("adam_q_proj_v", &self.q_proj_v);
        sv("adam_k_proj_m", &self.k_proj_m);
        sv("adam_k_proj_v", &self.k_proj_v);
        sv("adam_v_proj_m", &self.v_proj_m);
        sv("adam_v_proj_v", &self.v_proj_v);
        sv("adam_o_proj_m", &self.o_proj_m);
        sv("adam_o_proj_v", &self.o_proj_v);
        sv("adam_ffn_norm_m", &self.ffn_norm_m);
        sv("adam_ffn_norm_v", &self.ffn_norm_v);
        sv("adam_gate_proj_m", &self.gate_proj_m);
        sv("adam_gate_proj_v", &self.gate_proj_v);
        sv("adam_up_proj_m", &self.up_proj_m);
        sv("adam_up_proj_v", &self.up_proj_v);
        sv("adam_down_proj_m", &self.down_proj_m);
        sv("adam_down_proj_v", &self.down_proj_v);
        if let Some(ref d) = self.q_bias_m { sv("adam_q_bias_m", d); }
        if let Some(ref d) = self.q_bias_v { sv("adam_q_bias_v", d); }
        if let Some(ref d) = self.k_bias_m { sv("adam_k_bias_m", d); }
        if let Some(ref d) = self.k_bias_v { sv("adam_k_bias_v", d); }
        if let Some(ref d) = self.v_bias_m { sv("adam_v_bias_m", d); }
        if let Some(ref d) = self.v_bias_v { sv("adam_v_bias_v", d); }
    }

    /// ディスクから AdamW 状態を読み込む。存在しなければゼロ初期化。
    fn load_from_disk(layer_idx: usize, config: &LlamaConfig, checkpoint_dir: &str) -> Self {
        let hidden = config.hidden_dim;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let inter = config.intermediate_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let ld = |name: &str, size: usize| load_layer_delta(layer_idx, name, size, checkpoint_dir);

        let (q_bias_m, q_bias_v, k_bias_m, k_bias_v, v_bias_m, v_bias_v) = if config.attention_bias {
            (
                Some(ld("adam_q_bias_m", q_dim)), Some(ld("adam_q_bias_v", q_dim)),
                Some(ld("adam_k_bias_m", kv_dim)), Some(ld("adam_k_bias_v", kv_dim)),
                Some(ld("adam_v_bias_m", kv_dim)), Some(ld("adam_v_bias_v", kv_dim)),
            )
        } else {
            (None, None, None, None, None, None)
        };

        Self {
            attn_norm_m: ld("adam_attn_norm_m", hidden),
            attn_norm_v: ld("adam_attn_norm_v", hidden),
            q_proj_m: ld("adam_q_proj_m", q_dim * hidden),
            q_proj_v: ld("adam_q_proj_v", q_dim * hidden),
            k_proj_m: ld("adam_k_proj_m", kv_dim * hidden),
            k_proj_v: ld("adam_k_proj_v", kv_dim * hidden),
            v_proj_m: ld("adam_v_proj_m", kv_dim * hidden),
            v_proj_v: ld("adam_v_proj_v", kv_dim * hidden),
            o_proj_m: ld("adam_o_proj_m", q_dim * hidden),
            o_proj_v: ld("adam_o_proj_v", q_dim * hidden),
            ffn_norm_m: ld("adam_ffn_norm_m", hidden),
            ffn_norm_v: ld("adam_ffn_norm_v", hidden),
            gate_proj_m: ld("adam_gate_proj_m", inter * hidden),
            gate_proj_v: ld("adam_gate_proj_v", inter * hidden),
            up_proj_m: ld("adam_up_proj_m", inter * hidden),
            up_proj_v: ld("adam_up_proj_v", inter * hidden),
            down_proj_m: ld("adam_down_proj_m", hidden * inter),
            down_proj_v: ld("adam_down_proj_v", hidden * inter),
            q_bias_m, q_bias_v,
            k_bias_m, k_bias_v,
            v_bias_m, v_bias_v,
        }
    }

    /// RAM 使用量 (bytes) — m + v 合計。
    fn memory_bytes(&self) -> usize {
        let count = |v: &[f32]| v.len() * 4;
        let mut total = 0;
        total += count(&self.attn_norm_m) + count(&self.attn_norm_v);
        total += count(&self.q_proj_m) + count(&self.q_proj_v);
        total += count(&self.k_proj_m) + count(&self.k_proj_v);
        total += count(&self.v_proj_m) + count(&self.v_proj_v);
        total += count(&self.o_proj_m) + count(&self.o_proj_v);
        total += count(&self.ffn_norm_m) + count(&self.ffn_norm_v);
        total += count(&self.gate_proj_m) + count(&self.gate_proj_v);
        total += count(&self.up_proj_m) + count(&self.up_proj_v);
        total += count(&self.down_proj_m) + count(&self.down_proj_v);
        if let Some(ref v) = self.q_bias_m { total += count(v) * 2; }
        if let Some(ref v) = self.k_bias_m { total += count(v) * 2; }
        if let Some(ref v) = self.v_bias_m { total += count(v) * 2; }
        total
    }
}

// ── Delta Weight RAMキャッシュ ────────────────────────────────────────────

/// レイヤー delta のRAMキャッシュ。ファイルI/Oを排除して高速化。
struct LayerDeltaCache {
    attn_norm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    q_bias: Option<Vec<f32>>,
    k_bias: Option<Vec<f32>>,
    v_bias: Option<Vec<f32>>,
}

impl LayerDeltaCache {
    /// ディスクから既存 delta を読み込む。存在しなければゼロ初期化。
    fn load_from_disk(layer_idx: usize, config: &LlamaConfig, checkpoint_dir: &str) -> Self {
        let hidden = config.hidden_dim;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let inter = config.intermediate_dim;

        let ld = |name: &str, size: usize| load_layer_delta(layer_idx, name, size, checkpoint_dir);

        let q_bias = if config.attention_bias {
            Some(ld("q_bias", num_heads * head_dim))
        } else {
            None
        };
        let k_bias = if config.attention_bias {
            Some(ld("k_bias", num_kv_heads * head_dim))
        } else {
            None
        };
        let v_bias = if config.attention_bias {
            Some(ld("v_bias", num_kv_heads * head_dim))
        } else {
            None
        };

        Self {
            attn_norm: ld("attn_norm", hidden),
            q_proj: ld("q_proj", num_heads * head_dim * hidden),
            k_proj: ld("k_proj", num_kv_heads * head_dim * hidden),
            v_proj: ld("v_proj", num_kv_heads * head_dim * hidden),
            o_proj: ld("o_proj", hidden * num_heads * head_dim),
            ffn_norm: ld("ffn_norm", hidden),
            gate_proj: ld("gate_proj", inter * hidden),
            up_proj: ld("up_proj", inter * hidden),
            down_proj: ld("down_proj", hidden * inter),
            q_bias,
            k_bias,
            v_bias,
        }
    }

    /// 事前確保済みの workspace に base + delta を書き込み、同時に FakeQuantize を適用。
    /// アロケーションゼロで merge + quantize を 1 パスで実行。
    fn fused_merge_and_quantize(
        &self,
        base: &LlamaLayerWeights,
        workspace: &mut LlamaLayerWeights,
        fq: &mut FakeQuantize,
        _quantize_buf: &mut Vec<f32>,
    ) -> (f64, f64, usize) {
        self.fused_merge_and_quantize_alpha(base, workspace, fq, _quantize_buf, 1.0)
    }

    /// Alpha Blending 対応版: alpha=0 → FP32、alpha=1 → Ternary
    fn fused_merge_and_quantize_alpha(
        &self,
        base: &LlamaLayerWeights,
        workspace: &mut LlamaLayerWeights,
        fq: &mut FakeQuantize,
        _quantize_buf: &mut Vec<f32>,
        alpha: f32,
    ) -> (f64, f64, usize) {
        use rayon::prelude::*;

        let mut total_mae = 0.0f64;
        let mut total_cos = 0.0f64;
        let mut count = 0usize;

        // 量子化対象の projection 重み — workspace に in-place 書き込み + quantize
        let pairs: Vec<(&[f32], &[f32], &mut [f32])> = vec![
            (&base.q_proj, &self.q_proj, workspace.q_proj.as_mut_slice()),
            (&base.k_proj, &self.k_proj, workspace.k_proj.as_mut_slice()),
            (&base.v_proj, &self.v_proj, workspace.v_proj.as_mut_slice()),
            (&base.o_proj, &self.o_proj, workspace.o_proj.as_mut_slice()),
            (&base.gate_proj, &self.gate_proj, workspace.gate_proj.as_mut_slice()),
            (&base.up_proj, &self.up_proj, workspace.up_proj.as_mut_slice()),
            (&base.down_proj, &self.down_proj, workspace.down_proj.as_mut_slice()),
        ];

        let temp = fq.temperature().max(1e-10);
        let inv_temp = 1.0 / temp;

        for (b, d, w) in pairs {
            if b.is_empty() { continue; }

            // パス1 (Rayon並列): base+delta → workspace + sum(|w|) を同時計算
            let sum_abs: f64 = w.par_iter_mut()
                .zip(b.par_iter().zip(d.par_iter()))
                .map(|(out, (&bv, &dv))| {
                    let val = bv + dv;
                    *out = val;
                    val.abs() as f64
                })
                .sum();

            // calibrate_scale: γ = mean(|w|)
            let scale = (sum_abs / w.len() as f64).max(1e-10) as f32;
            fq.set_scale(scale);
            let inv_scale = 1.0 / scale;

            // パス2 (Rayon並列): 量子化 + Alpha Blending + 統計 + 上書き を完全融合
            let blend_alpha = alpha;
            let one_minus_alpha = 1.0 - blend_alpha;
            let (sum_err, dot_wq, dot_ww, dot_qq) = w.par_iter_mut()
                .map(|wv_mut| {
                    let wv = *wv_mut; // FP32 (base + delta)
                    // Ternary fake quantize: round(w/γ/temp) → clamp(-1,1) → ×γ
                    let scaled = wv * inv_scale * inv_temp;
                    let qv_hard = scaled.round().clamp(-1.0, 1.0) * scale;

                    // Alpha Blending: w_effective = w_fp32 * (1-α) + w_quant * α
                    let qv = wv * one_minus_alpha + qv_hard * blend_alpha;

                    // 統計計算（FP32 vs blended）
                    let wv_f64 = wv as f64;
                    let qv_f64 = qv as f64;

                    // blended 値で上書き
                    *wv_mut = qv;

                    ((wv_f64 - qv_f64).abs(), wv_f64 * qv_f64, wv_f64 * wv_f64, qv_f64 * qv_f64)
                })
                .reduce(
                    || (0.0f64, 0.0f64, 0.0f64, 0.0f64),
                    |(a, b, c, d), (e, f, g, h)| (a + e, b + f, c + g, d + h),
                );

            let n = w.len() as f64;
            total_mae += sum_err / n;
            let denom = (dot_ww * dot_qq).sqrt();
            if denom > 1e-10 {
                total_cos += dot_wq / denom;
            }
            count += 1;
        }

        // Norm / Bias は量子化しない — 単なる加算（rayon 並列）
        workspace.attn_norm.par_iter_mut().zip(base.attn_norm.par_iter().zip(self.attn_norm.par_iter()))
            .for_each(|(out, (&b, &d))| *out = b + d);
        workspace.ffn_norm.par_iter_mut().zip(base.ffn_norm.par_iter().zip(self.ffn_norm.par_iter()))
            .for_each(|(out, (&b, &d))| *out = b + d);

        // Bias
        if let (Some(ref mut wb), Some(ref bb), Some(ref db)) = (&mut workspace.q_bias, &base.q_bias, &self.q_bias) {
            wb.iter_mut().zip(bb.iter().zip(db.iter())).for_each(|(out, (&b, &d))| *out = b + d);
        }
        if let (Some(ref mut wb), Some(ref bb), Some(ref db)) = (&mut workspace.k_bias, &base.k_bias, &self.k_bias) {
            wb.iter_mut().zip(bb.iter().zip(db.iter())).for_each(|(out, (&b, &d))| *out = b + d);
        }
        if let (Some(ref mut wb), Some(ref bb), Some(ref db)) = (&mut workspace.v_bias, &base.v_bias, &self.v_bias) {
            wb.iter_mut().zip(bb.iter().zip(db.iter())).for_each(|(out, (&b, &d))| *out = b + d);
        }

        (total_mae, total_cos, count)
    }

    /// AdamW 更新を delta キャッシュに適用（rayon 並列）。
    /// d_logits で Mean 化済み + グローバル gradient clipping 適用済み。
    fn apply_adam(&mut self, grads: &LayerWeightGrads, adam: &mut LayerAdamState, lr: f32, weight_decay: f32, step: usize) {
        use rayon::prelude::*;

        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let eps: f32 = 1e-8;
        let t = step as f32;
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);

        let adamw_update = |delta: &mut [f32], grad: &[f32], m: &mut [f32], v: &mut [f32], wd: f32| {
            for i in 0..delta.len() {
                let g = grad[i];
                m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
                let m_hat = m[i] / bc1;
                let v_hat = v[i] / bc2;
                delta[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * delta[i]);
            }
        };

        // 大きい projection テンソルを並列更新
        let wd = weight_decay;
        // projection 7本を並列に処理
        rayon::scope(|s| {
            s.spawn(|_| adamw_update(&mut self.q_proj, &grads.q_proj, &mut adam.q_proj_m, &mut adam.q_proj_v, wd));
            s.spawn(|_| adamw_update(&mut self.k_proj, &grads.k_proj, &mut adam.k_proj_m, &mut adam.k_proj_v, wd));
            s.spawn(|_| adamw_update(&mut self.v_proj, &grads.v_proj, &mut adam.v_proj_m, &mut adam.v_proj_v, wd));
            s.spawn(|_| adamw_update(&mut self.o_proj, &grads.o_proj, &mut adam.o_proj_m, &mut adam.o_proj_v, wd));
            s.spawn(|_| adamw_update(&mut self.gate_proj, &grads.gate_proj, &mut adam.gate_proj_m, &mut adam.gate_proj_v, wd));
            s.spawn(|_| adamw_update(&mut self.up_proj, &grads.up_proj, &mut adam.up_proj_m, &mut adam.up_proj_v, wd));
            s.spawn(|_| adamw_update(&mut self.down_proj, &grads.down_proj, &mut adam.down_proj_m, &mut adam.down_proj_v, wd));
        });

        // norm は小さいのでシーケンシャル
        adamw_update(&mut self.attn_norm, &grads.attn_norm, &mut adam.attn_norm_m, &mut adam.attn_norm_v, weight_decay);
        adamw_update(&mut self.ffn_norm, &grads.ffn_norm, &mut adam.ffn_norm_m, &mut adam.ffn_norm_v, weight_decay);

        // Bias: weight_decay なし
        if let (Some(ref mut d), Some(ref g), Some(ref mut m), Some(ref mut v)) =
            (&mut self.q_bias, &grads.q_bias, &mut adam.q_bias_m, &mut adam.q_bias_v) {
            adamw_update(d, g, m, v, 0.0);
        }
        if let (Some(ref mut d), Some(ref g), Some(ref mut m), Some(ref mut v)) =
            (&mut self.k_bias, &grads.k_bias, &mut adam.k_bias_m, &mut adam.k_bias_v) {
            adamw_update(d, g, m, v, 0.0);
        }
        if let (Some(ref mut d), Some(ref g), Some(ref mut m), Some(ref mut v)) =
            (&mut self.v_bias, &grads.v_bias, &mut adam.v_bias_m, &mut adam.v_bias_v) {
            adamw_update(d, g, m, v, 0.0);
        }
    }

    /// GPU Fused AdamW 更新。projection 7本を順次 GPU カーネルで更新。
    #[cfg(feature = "cuda")]
    fn apply_adam_gpu(&mut self, grads: &LayerWeightGrads, adam: &mut LayerAdamState,
                      cuda: &alice_train::cuda_matmul::CudaMatmul, lr: f32, weight_decay: f32, step: usize) {
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let eps: f32 = 1e-8;
        let t = step as f32;
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);
        let wd = weight_decay;

        // 大きい projection テンソルを GPU で更新
        cuda.gpu_adamw_update(&mut self.q_proj, &grads.q_proj, &mut adam.q_proj_m, &mut adam.q_proj_v, lr, beta1, beta2, eps, bc1, bc2, wd);
        cuda.gpu_adamw_update(&mut self.k_proj, &grads.k_proj, &mut adam.k_proj_m, &mut adam.k_proj_v, lr, beta1, beta2, eps, bc1, bc2, wd);
        cuda.gpu_adamw_update(&mut self.v_proj, &grads.v_proj, &mut adam.v_proj_m, &mut adam.v_proj_v, lr, beta1, beta2, eps, bc1, bc2, wd);
        cuda.gpu_adamw_update(&mut self.o_proj, &grads.o_proj, &mut adam.o_proj_m, &mut adam.o_proj_v, lr, beta1, beta2, eps, bc1, bc2, wd);
        cuda.gpu_adamw_update(&mut self.gate_proj, &grads.gate_proj, &mut adam.gate_proj_m, &mut adam.gate_proj_v, lr, beta1, beta2, eps, bc1, bc2, wd);
        cuda.gpu_adamw_update(&mut self.up_proj, &grads.up_proj, &mut adam.up_proj_m, &mut adam.up_proj_v, lr, beta1, beta2, eps, bc1, bc2, wd);
        cuda.gpu_adamw_update(&mut self.down_proj, &grads.down_proj, &mut adam.down_proj_m, &mut adam.down_proj_v, lr, beta1, beta2, eps, bc1, bc2, wd);

        // norm は小さいのでCPU（GPU転送オーバーヘッドの方が大きい）
        let adamw_cpu = |delta: &mut [f32], grad: &[f32], m: &mut [f32], v: &mut [f32], wd: f32| {
            for i in 0..delta.len() {
                let g = grad[i];
                m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
                let m_hat = m[i] / bc1;
                let v_hat = v[i] / bc2;
                delta[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * delta[i]);
            }
        };
        adamw_cpu(&mut self.attn_norm, &grads.attn_norm, &mut adam.attn_norm_m, &mut adam.attn_norm_v, weight_decay);
        adamw_cpu(&mut self.ffn_norm, &grads.ffn_norm, &mut adam.ffn_norm_m, &mut adam.ffn_norm_v, weight_decay);
        if let (Some(ref mut d), Some(ref g), Some(ref mut m), Some(ref mut v)) =
            (&mut self.q_bias, &grads.q_bias, &mut adam.q_bias_m, &mut adam.q_bias_v) {
            adamw_cpu(d, g, m, v, 0.0);
        }
        if let (Some(ref mut d), Some(ref g), Some(ref mut m), Some(ref mut v)) =
            (&mut self.k_bias, &grads.k_bias, &mut adam.k_bias_m, &mut adam.k_bias_v) {
            adamw_cpu(d, g, m, v, 0.0);
        }
        if let (Some(ref mut d), Some(ref g), Some(ref mut m), Some(ref mut v)) =
            (&mut self.v_bias, &grads.v_bias, &mut adam.v_bias_m, &mut adam.v_bias_v) {
            adamw_cpu(d, g, m, v, 0.0);
        }
    }

    /// delta をディスクに保存。
    fn save_to_disk(&self, layer_idx: usize, checkpoint_dir: &str) {
        let sv = |name: &str, data: &[f32]| save_layer_delta(layer_idx, name, data, checkpoint_dir);
        sv("attn_norm", &self.attn_norm);
        sv("q_proj", &self.q_proj);
        sv("k_proj", &self.k_proj);
        sv("v_proj", &self.v_proj);
        sv("o_proj", &self.o_proj);
        sv("ffn_norm", &self.ffn_norm);
        sv("gate_proj", &self.gate_proj);
        sv("up_proj", &self.up_proj);
        sv("down_proj", &self.down_proj);
        if let Some(ref d) = self.q_bias { sv("q_bias", d); }
        if let Some(ref d) = self.k_bias { sv("k_bias", d); }
        if let Some(ref d) = self.v_bias { sv("v_bias", d); }
    }
}

// ── BF16 Delta 圧縮ストレージ ──────────────────────────────────────────
//
// 70B で delta を FP32 保持すると ~272 GB 必要。
// BF16 (16-bit) で保持すれば ~136 GB に半減。
// 学習ループでは 1 レイヤーずつ FP32 に展開して使い、更新後に BF16 に戻す。

#[inline]
fn f32_to_bf16(v: f32) -> u16 {
    // BF16: 上位16ビット（符号1 + 指数8 + 仮数7）
    // round-to-nearest-even: bit 16 が 1 かつ bit 15 以下が非ゼロならインクリメント
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding = 0x7FFF + lsb;
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

#[inline]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

fn f32_slice_to_bf16(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&v| f32_to_bf16(v)).collect()
}

fn bf16_slice_to_f32(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&v| bf16_to_f32(v)).collect()
}

/// BF16 圧縮 delta ストレージ。RAM 使用量を半減。
struct Bf16DeltaStore {
    attn_norm: Vec<u16>,
    q_proj: Vec<u16>,
    k_proj: Vec<u16>,
    v_proj: Vec<u16>,
    o_proj: Vec<u16>,
    ffn_norm: Vec<u16>,
    gate_proj: Vec<u16>,
    up_proj: Vec<u16>,
    down_proj: Vec<u16>,
    q_bias: Option<Vec<u16>>,
    k_bias: Option<Vec<u16>>,
    v_bias: Option<Vec<u16>>,
}

impl Bf16DeltaStore {
    /// FP32 delta キャッシュから BF16 に圧縮。
    fn from_f32(cache: &LayerDeltaCache) -> Self {
        Self {
            attn_norm: f32_slice_to_bf16(&cache.attn_norm),
            q_proj: f32_slice_to_bf16(&cache.q_proj),
            k_proj: f32_slice_to_bf16(&cache.k_proj),
            v_proj: f32_slice_to_bf16(&cache.v_proj),
            o_proj: f32_slice_to_bf16(&cache.o_proj),
            ffn_norm: f32_slice_to_bf16(&cache.ffn_norm),
            gate_proj: f32_slice_to_bf16(&cache.gate_proj),
            up_proj: f32_slice_to_bf16(&cache.up_proj),
            down_proj: f32_slice_to_bf16(&cache.down_proj),
            q_bias: cache.q_bias.as_ref().map(|b| f32_slice_to_bf16(b)),
            k_bias: cache.k_bias.as_ref().map(|b| f32_slice_to_bf16(b)),
            v_bias: cache.v_bias.as_ref().map(|b| f32_slice_to_bf16(b)),
        }
    }

    /// BF16 から FP32 delta キャッシュに展開。
    fn to_f32(&self) -> LayerDeltaCache {
        LayerDeltaCache {
            attn_norm: bf16_slice_to_f32(&self.attn_norm),
            q_proj: bf16_slice_to_f32(&self.q_proj),
            k_proj: bf16_slice_to_f32(&self.k_proj),
            v_proj: bf16_slice_to_f32(&self.v_proj),
            o_proj: bf16_slice_to_f32(&self.o_proj),
            ffn_norm: bf16_slice_to_f32(&self.ffn_norm),
            gate_proj: bf16_slice_to_f32(&self.gate_proj),
            up_proj: bf16_slice_to_f32(&self.up_proj),
            down_proj: bf16_slice_to_f32(&self.down_proj),
            q_bias: self.q_bias.as_ref().map(|b| bf16_slice_to_f32(b)),
            k_bias: self.k_bias.as_ref().map(|b| bf16_slice_to_f32(b)),
            v_bias: self.v_bias.as_ref().map(|b| bf16_slice_to_f32(b)),
        }
    }

    /// FP32 delta キャッシュの内容で BF16 ストアを更新（再アロケーションなし）。Rayon並列。
    fn update_from_f32(&mut self, cache: &LayerDeltaCache) {
        use rayon::prelude::*;
        let compress = |dst: &mut [u16], src: &[f32]| {
            dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, s)| *d = f32_to_bf16(*s));
        };
        compress(&mut self.attn_norm, &cache.attn_norm);
        compress(&mut self.q_proj, &cache.q_proj);
        compress(&mut self.k_proj, &cache.k_proj);
        compress(&mut self.v_proj, &cache.v_proj);
        compress(&mut self.o_proj, &cache.o_proj);
        compress(&mut self.ffn_norm, &cache.ffn_norm);
        compress(&mut self.gate_proj, &cache.gate_proj);
        compress(&mut self.up_proj, &cache.up_proj);
        compress(&mut self.down_proj, &cache.down_proj);
        if let (Some(dst), Some(src)) = (&mut self.q_bias, &cache.q_bias) {
            compress(dst, src);
        }
        if let (Some(dst), Some(src)) = (&mut self.k_bias, &cache.k_bias) {
            compress(dst, src);
        }
        if let (Some(dst), Some(src)) = (&mut self.v_bias, &cache.v_bias) {
            compress(dst, src);
        }
    }

    /// delta をディスクに保存（FP32 で書き出し — チェックポイント互換）。
    fn save_to_disk(&self, layer_idx: usize, checkpoint_dir: &str) {
        self.to_f32().save_to_disk(layer_idx, checkpoint_dir);
    }

    /// FP32 delta キャッシュの内容で BF16 ストアをリロード。
    /// SIMD-friendly 単一スレッドループ（Rayon par_iter より高速 — メモリバウンド）。
    fn expand_into(&self, dst: &mut LayerDeltaCache) {
        use alice_train::cuda_matmul::CudaMatmul;
        let expand = CudaMatmul::cpu_bf16_expand_fast;
        expand(&self.attn_norm, &mut dst.attn_norm);
        expand(&self.q_proj, &mut dst.q_proj);
        expand(&self.k_proj, &mut dst.k_proj);
        expand(&self.v_proj, &mut dst.v_proj);
        expand(&self.o_proj, &mut dst.o_proj);
        expand(&self.ffn_norm, &mut dst.ffn_norm);
        expand(&self.gate_proj, &mut dst.gate_proj);
        expand(&self.up_proj, &mut dst.up_proj);
        expand(&self.down_proj, &mut dst.down_proj);
        if let (Some(ref src), Some(ref mut d)) = (&self.q_bias, &mut dst.q_bias) {
            expand(src, d);
        }
        if let (Some(ref src), Some(ref mut d)) = (&self.k_bias, &mut dst.k_bias) {
            expand(src, d);
        }
        if let (Some(ref src), Some(ref mut d)) = (&self.v_bias, &mut dst.v_bias) {
            expand(src, d);
        }
    }
}

// ── Delta Weight ファイルI/O ────────────────────────────────────────────
use alice_train::llama::LlamaLayerWeights;

/// レイヤーiの重み delta をファイルから読み込む。存在しなければゼロ初期化。
fn load_layer_delta(layer_idx: usize, weight_name: &str, size: usize, checkpoint_dir: &str) -> Vec<f32> {
    let path = format!("{}/delta_layer{layer_idx}_{weight_name}.bin", checkpoint_dir);
    if let Ok(data) = std::fs::read(&path) {
        if data.len() == size * 4 {
            let mut out = vec![0.0f32; size];
            for i in 0..size {
                out[i] = f32::from_le_bytes([data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]]);
            }
            return out;
        }
    }
    vec![0.0f32; size]
}

/// レイヤーiの重み delta をファイルに保存。
fn save_layer_delta(layer_idx: usize, weight_name: &str, delta: &[f32], checkpoint_dir: &str) {
    let path = format!("{}/delta_layer{layer_idx}_{weight_name}.bin", checkpoint_dir);
    let mut data = Vec::with_capacity(delta.len() * 4);
    for &v in delta {
        data.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(e) = std::fs::write(&path, &data) {
        eprintln!("  delta保存失敗 layer{layer_idx}/{weight_name}: {e}");
    }
}

/// 勾配ベクトルの L2 ノルムを計算。
fn grad_l2_norm(grad: &[f32]) -> f32 {
    use rayon::prelude::*;
    let sum: f64 = grad.par_iter().map(|&g| (g as f64) * (g as f64)).sum();
    sum.sqrt() as f32
}

/// 勾配をクリップ（max_norm 以下にスケーリング）。
fn clip_grad(grad: &mut [f32], max_norm: f32) -> f32 {
    use rayon::prelude::*;
    let norm = grad_l2_norm(grad);
    if norm > max_norm && norm > 1e-10 {
        let scale = max_norm / norm;
        grad.par_iter_mut().for_each(|g| *g *= scale);
    }
    norm
}

/// SIGTERM/SIGINT ハンドラ — AtomicBool を立てるだけ（async-signal-safe）。
#[cfg(unix)]
extern "C" fn signal_handler(sig: libc::c_int) {
    use std::sync::atomic::Ordering;
    // signal_handler 内では安全な操作のみ（atomic store + write syscall）
    SHUTDOWN_FLAG.store(true, Ordering::SeqCst);
    // stderr に直接書き出し（async-signal-safe）
    let msg = match sig {
        libc::SIGTERM => b"\n[ALICE-Train] SIGTERM received, saving checkpoint...\n" as &[u8],
        libc::SIGINT => b"\n[ALICE-Train] SIGINT received, saving checkpoint...\n" as &[u8],
        _ => b"\n[ALICE-Train] Signal received, saving checkpoint...\n" as &[u8],
    };
    unsafe { libc::write(2, msg.as_ptr().cast(), msg.len()); }
}

#[cfg(unix)]
static SHUTDOWN_FLAG: AtomicBool = AtomicBool::new(false);

/// グローバル delta をモデル重みに適用（起動時に呼ぶ）。
fn apply_global_delta(weights: &mut [f32], name: &str, checkpoint_dir: &str) {
    let delta_path = format!("{}/delta_layer0_global_{name}.bin", checkpoint_dir);
    if let Ok(data) = std::fs::read(&delta_path) {
        if data.len() == weights.len() * 4 {
            for i in 0..weights.len() {
                weights[i] += f32::from_le_bytes([data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]]);
            }
            println!("    {name} delta 適用済み ({:.1} MB)", data.len() as f64 / 1024.0 / 1024.0);
        }
    }
}
