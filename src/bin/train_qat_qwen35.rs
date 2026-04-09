//! ALICE-Train QAT — Qwen3.5 Gated DeltaNet → 1.1-bit Ternary。
//!
//! # Phase 1 (パイプライン検証)
//!
//! ```bash
//! cargo run --release --features qat-cli --bin train-qat-qwen35 -- \
//!     --config configs/qat_qwen35_9b.json
//! ```
//!
//! # Phase 2 (実モデル学習)
//!
//! ```bash
//! cargo run --release --features qat-cli --bin train-qat-qwen35 -- \
//!     --config configs/qat_qwen35_9b.json
//! ```

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use rand::Rng;
use std::fs;
use std::path::Path;
use std::time::Instant;

use alice_train::qwen35::{LayerType, Qwen35LayerWeights, Qwen35QatConfig};
use alice_train::qwen35_backward::{qwen35_layer_backward, Qwen35WeightGrads};
use alice_train::qwen35_forward::qwen35_model_forward_eval;
use alice_train::{
    CheckpointData, DataLoader, DataLoaderConfig, FakeQuantize, LogEntry, LrScheduler, MmapDataset,
    QatConfig, TrainLog, WarmupCosineScheduler,
};

/// RSS 表示。
fn print_rss(label: &str) {
    #[cfg(unix)]
    {
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
        unsafe {
            let mut usage: libc::rusage = std::mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
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

/// ホスト名取得。
fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// GPU名取得 (nvidia-smi)。
fn gpu_name() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// 現在時刻 (UTC ISO 8601)。
fn chrono_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let days = secs / 86400;
    let rem = secs % 86400;
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let s = rem % 60;
    // Simple epoch-days to Y-M-D (good enough for logging)
    let mut y = 1970u64;
    let mut remaining_days = days;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            366
        } else {
            365
        };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut mo = 0;
    for (i, &md) in month_days.iter().enumerate() {
        if remaining_days < md {
            mo = i + 1;
            break;
        }
        remaining_days -= md;
    }
    let day = remaining_days + 1;
    format!("{y}-{mo:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

/// ALICE-Train QAT: Qwen3.5 → 1.1-bit Sparse Ternary
#[derive(Parser, Debug)]
#[command(author = "Moroya Sakamoto")]
#[command(about = "Quantize Qwen3.5 to 1.1-bit ternary via QAT")]
struct Cli {
    /// 設定ファイルパス (JSON)
    #[arg(short, long)]
    config: String,

    /// チェックポイントからレジューム
    #[arg(short, long)]
    resume: Option<String>,

    /// ドライラン
    #[arg(long)]
    dry_run: bool,
}

/// 数値安定な cross-entropy loss と勾配。
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
    let mut config: Qwen35QatConfig = serde_json::from_str(&config_str).unwrap_or_else(|e| {
        eprintln!("[ALICE-Train] 設定パースエラー: {e}");
        std::process::exit(1);
    });

    if let Some(resume_path) = &cli.resume {
        config.resume_from = Some(resume_path.clone());
    }

    // layer_types が空なら自動生成
    if config.model.layer_types.is_empty() {
        config.model.layer_types = (0..config.model.num_hidden_layers)
            .map(|i| {
                if (i + 1) % config.model.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
    }

    let model_config = &config.model;
    let total_params = model_config.total_params();
    let model_path_exists = Path::new(&config.model_path).exists();
    let phase1_mode = !model_path_exists;

    let dn_count = (0..model_config.num_hidden_layers)
        .filter(|&i| model_config.layer_type(i) == LayerType::LinearAttention)
        .count();
    let fa_count = model_config.num_hidden_layers - dn_count;

    println!("╔══════════════════════════════════════════════════════════╗");
    if phase1_mode {
        println!("║  ALICE-Train QAT — Qwen3.5 Phase 1 パイプライン検証     ║");
    } else {
        println!("║  ALICE-Train QAT — Qwen3.5 Gated DeltaNet → Ternary    ║");
    }
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    if phase1_mode {
        println!("モード: Phase 1 (ランダム初期化 — パイプライン検証)");
        println!("  モデルパス '{}' が見つかりません", config.model_path);
        println!();
    }

    println!(
        "モデル: Qwen3.5 vocab={}, hidden={}, layers={} ({dn_count} DeltaNet + {fa_count} FullAttn)",
        model_config.vocab_size, model_config.hidden_size, model_config.num_hidden_layers
    );
    println!("  総パラメータ: {:.2}B", total_params as f64 / 1e9);
    println!(
        "  DeltaNet: key_dim={}, value_dim={}, conv_kernel={}",
        model_config.linear_key_dim(),
        model_config.linear_value_dim(),
        model_config.linear_conv_kernel_dim
    );
    println!(
        "  FullAttn: heads={}, kv_heads={}, head_dim={}, rotary_dim={}",
        model_config.num_attention_heads,
        model_config.num_key_value_heads,
        model_config.head_dim,
        model_config.rotary_dim()
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
    println!("  重みプレフィックス: {}", config.weight_prefix);
    println!();

    if cli.dry_run {
        println!("[ドライラン完了]");
        return;
    }

    // --- チェックポイントディレクトリ ---
    fs::create_dir_all(&config.checkpoint_dir).unwrap_or_else(|e| {
        eprintln!("[ALICE-Train] チェックポイントDir作成失敗: {e}");
        std::process::exit(1);
    });

    // --- データ読み込み ---
    println!("━━━ データ読み込み ━━━");
    let dataset = MmapDataset::open(&config.train_data_path).unwrap_or_else(|e| {
        eprintln!(
            "[ALICE-Train] データ読み込み失敗: {}: {e}",
            config.train_data_path
        );
        std::process::exit(1);
    });
    println!(
        "  データ: {} ({} トークン)",
        config.train_data_path,
        dataset.len()
    );

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

    // Eval データ
    let eval_dataset: Option<MmapDataset> = config.eval_data_path.as_ref().and_then(|p| {
        if Path::new(p).exists() {
            MmapDataset::open(p).ok().map(|ds| {
                println!("  Eval: {} ({} トークン)", p, ds.len());
                ds
            })
        } else {
            None
        }
    });
    println!();

    // --- コンポーネント初期化 ---
    println!("━━━ コンポーネント初期化 ━━━");

    // THP無効化 — チェックポイント保存時のcompactionストールを防止
    #[cfg(target_os = "linux")]
    unsafe {
        libc::prctl(libc::PR_SET_THP_DISABLE, 1, 0, 0, 0);
        println!("  THP無効化: compactionストール防止");
    }

    // CUDA cuBLAS 初期化 — 以降の全 blas_matmul_bt/nn が自動的に GPU を使用
    #[cfg(feature = "cuda")]
    {
        println!("  CUDA cuBLAS 初期化中...");
        alice_train::blas::init_cuda_blas();
        println!("  CUDA cuBLAS: 有効 (TF32 Tensor Cores)");
    }

    let _fq = FakeQuantize::new(QatConfig::ternary());
    let scheduler = WarmupCosineScheduler::new(
        config.learning_rate,
        config.min_lr,
        config.warmup_steps,
        config.total_steps,
    );
    let mut log = TrainLog::new();
    let mut global_step: usize = 0;
    let mut resume_start_step: usize = 0;

    // resume_state.json から step 復元
    let resume_path = format!("{}/resume_state.json", config.checkpoint_dir);
    if let Ok(json_str) = fs::read_to_string(&resume_path) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json_str) {
            if let Some(step) = v.get("step").and_then(|s| s.as_u64()) {
                global_step = step as usize;
                resume_start_step = global_step;
                let prev_loss = v.get("loss").and_then(|l| l.as_f64()).unwrap_or(0.0);
                println!("  Resume: step {global_step}, loss {prev_loss:.4}");
            }
        }
    }

    println!("  FakeQuantize: Ternary 1.58-bit, STE backward");
    println!();

    let vocab_size = config.model.vocab_size;
    let hidden = config.model.hidden_size;

    if phase1_mode {
        // ================================================================
        // Phase 1: ランダム初期化 (embedding → output)
        // ================================================================
        println!("━━━ Phase 1: ランダムモデル初期化 ━━━");
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (vocab_size + hidden) as f32).sqrt();
        let mut embedding: Vec<f32> = (0..vocab_size * hidden)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let mut output_proj: Vec<f32> = (0..hidden * vocab_size)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let mut grad_embedding = vec![0.0f32; vocab_size * hidden];
        let mut grad_output_proj = vec![0.0f32; hidden * vocab_size];

        println!(
            "  Embedding: [{vocab_size} × {hidden}] ({:.1} MB)",
            (vocab_size * hidden * 4) as f64 / 1e6
        );
        println!("  初期化完了");
        println!();

        println!(
            "━━━ Phase 1 学習開始 (step {global_step}/{}) ━━━",
            config.total_steps
        );
        let train_start = Instant::now();
        let mut batch_idx = 0usize;

        while global_step < config.total_steps {
            let lr = scheduler.get_lr(global_step);
            let step_start = Instant::now();

            if batch_idx >= num_batches {
                loader.shuffle_epoch();
                batch_idx = 0;
            }
            let batch = loader.get_batch(batch_idx, &dataset);
            batch_idx += 1;

            let mut total_loss = 0.0f32;
            let mut token_count = 0usize;
            let seq_len = config.seq_len;

            for b in 0..batch.actual_batch_size {
                for t in 0..seq_len {
                    let input_token = batch.input_ids[b * seq_len + t] as usize % vocab_size;
                    let target_token = batch.target_ids[b * seq_len + t] as usize % vocab_size;

                    let emb_offset = input_token * hidden;
                    let mut logits = vec![0.0f32; vocab_size];
                    for v in 0..vocab_size {
                        let proj_offset = v * hidden;
                        let mut sum = 0.0f32;
                        for h in 0..hidden {
                            sum = embedding[emb_offset + h]
                                .mul_add(output_proj[proj_offset + h], sum);
                        }
                        logits[v] = sum;
                    }

                    let (loss, d_logits) = cross_entropy_loss(&logits, target_token);
                    total_loss += loss;
                    token_count += 1;

                    for v in 0..vocab_size {
                        let proj_offset = v * hidden;
                        let dl = d_logits[v];
                        if dl.abs() < 1e-10 {
                            continue;
                        }
                        for h in 0..hidden {
                            grad_output_proj[proj_offset + h] += dl * embedding[emb_offset + h];
                        }
                    }
                    for h in 0..hidden {
                        let mut grad_h = 0.0f32;
                        for v in 0..vocab_size {
                            grad_h += d_logits[v] * output_proj[v * hidden + h];
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

            // Phase 1: 勾配を token_count で正規化（loss の平均と一致させる）
            let grad_scale = 1.0 / token_count.max(1) as f32;

            for (w, g) in embedding.iter_mut().zip(grad_embedding.iter_mut()) {
                *w -= lr * (*g * grad_scale + config.weight_decay * *w);
                *g = 0.0;
            }
            for (w, g) in output_proj.iter_mut().zip(grad_output_proj.iter_mut()) {
                *w -= lr * (*g * grad_scale + config.weight_decay * *w);
                *g = 0.0;
            }

            log.append(LogEntry::new(0, global_step, avg_loss, lr, 0.0));

            {
                let elapsed = train_start.elapsed();
                let steps_done = (global_step - resume_start_step).max(1) as f64;
                let steps_per_sec = steps_done / elapsed.as_secs_f64().max(0.001);
                let eta_secs = (config.total_steps - global_step) as f64 / steps_per_sec.max(0.001);
                println!(
                    "  step {global_step:>5}/{} | loss: {avg_loss:.4} | lr: {lr:.2e} | {:.1}ms/step | ETA: {eta_secs:.0}s",
                    config.total_steps,
                    step_duration.as_secs_f64() * 1000.0,
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }

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

        let total_duration = train_start.elapsed();
        println!();
        println!("━━━ Phase 1 学習完了 ━━━");
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
                if last.loss < first.loss {
                    println!("  ✓ Loss が減少 — パイプライン正常動作を確認");
                } else {
                    println!("  ✗ Loss が減少していません — 設定を確認してください");
                }
            }
        }
    } else {
        // ================================================================
        // Phase 2: ストリーミング学習（実モデル）
        // ================================================================
        println!("━━━ Phase 2: Qwen3.5 ストリーミング学習 ━━━");

        use alice_train::safetensors_loader::ShardedModel;

        let mut model_opt: Option<ShardedModel> =
            Some(ShardedModel::open(&config.model_path).unwrap_or_else(|e| {
                eprintln!("[ALICE-Train] モデル読み込み失敗: {e}");
                std::process::exit(1);
            }));

        let prefix = &config.weight_prefix;

        // Embedding (常駐)
        println!("  embedding 読み込み...");
        let embedding_table = {
            let model = model_opt.as_ref().unwrap();
            model
                .get_tensor_f32(&format!("{prefix}.embed_tokens.weight"))
                .unwrap_or_else(|| {
                    eprintln!("[ALICE-Train] {prefix}.embed_tokens.weight が見つかりません");
                    std::process::exit(1);
                })
        };
        println!(
            "    embed_tokens: {:.1} MB",
            embedding_table.len() as f64 * 4.0 / 1e6
        );

        // Output norm + lm_head
        let output_norm = {
            let model = model_opt.as_ref().unwrap();
            model
                .get_tensor_f32(&format!("{prefix}.norm.weight"))
                .unwrap_or_else(|| {
                    eprintln!("[ALICE-Train] {prefix}.norm.weight が見つかりません");
                    std::process::exit(1);
                })
        };
        let lm_head = {
            let model = model_opt.as_ref().unwrap();
            model.get_tensor_f32("lm_head.weight").unwrap_or_else(|| {
                println!("    lm_head.weight なし — embed_tokens と共有");
                embedding_table.clone()
            })
        };

        // レイヤー重み読み込み (L10: preload/streaming 分岐)
        let num_layers = config.model.num_hidden_layers;
        let mut layers: Vec<Qwen35LayerWeights> = if config.preload_all_layers {
            // L14: FP32キャッシュ構築 → SafetensorsModel解放 → 全層プリロード (A100 83GB RAM用)
            println!("  L14: FP32 キャッシュ → SafetensorsModel解放 → 全層プリロード");
            let cache_base = &config.checkpoint_dir;

            // FP32キャッシュ構築（未構築の場合のみ）
            if !alice_train::fp32_cache::cache_exists(cache_base, &config.model) {
                println!("    FP32 キャッシュ構築中 (初回のみ, 1層ずつ)...");
                let t0 = Instant::now();
                {
                    let model = model_opt.as_ref().unwrap();
                    let get_tensor =
                        |name: &str| -> Option<Vec<f32>> { model.get_tensor_f32(name) };
                    alice_train::fp32_cache::build_cache(
                        &get_tensor,
                        &config.weight_prefix,
                        cache_base,
                        &config.model,
                    )
                    .unwrap_or_else(|e| {
                        eprintln!("[ALICE-Train] FP32 キャッシュ構築失敗: {e}");
                        std::process::exit(1);
                    });
                }
                let cache_mb =
                    alice_train::fp32_cache::cache_size_bytes(&config.model) as f64 / 1e6;
                println!(
                    "    FP32 キャッシュ完了: {cache_mb:.0} MB ({:.1}s)",
                    t0.elapsed().as_secs_f64()
                );
            } else {
                println!("    FP32 キャッシュ検出済み");
            }

            // SafetensorsModel 解放 → mmap RAM 回収
            let _ = model_opt.take();
            print_rss("SafetensorsModel 解放後");

            // FP32キャッシュから全層プリロード
            println!("    FP32 キャッシュから全層プリロード中...");
            let t0 = Instant::now();
            let mut layers = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                let layer =
                    alice_train::fp32_cache::load_layer_from_cache(cache_base, i, &config.model)
                        .unwrap_or_else(|e| {
                            eprintln!("[ALICE-Train] FP32 キャッシュ layer {i} 読み込み失敗: {e}");
                            std::process::exit(1);
                        });
                layers.push(layer);
                if (i + 1) % 8 == 0 || i == num_layers - 1 {
                    println!("    {}/{num_layers} レイヤー読み込み完了", i + 1);
                }
            }
            println!(
                "    全層プリロード完了 ({:.1}s)",
                t0.elapsed().as_secs_f64()
            );
            print_rss("全層プリロード後");
            layers
        } else {
            // FP32キャッシュ + ストリーミングモード (CUDA matmul + DeltaNet)
            println!("  L10+L12+L13: ストリーミング + FP32 キャッシュ + CUDA DeltaNet");
            let cache_base = &config.checkpoint_dir;
            if !alice_train::fp32_cache::cache_exists(cache_base, &config.model) {
                println!("    FP32 キャッシュ構築中 (初回のみ, 1層ずつ)...");
                let t0 = Instant::now();
                let model = model_opt.as_ref().unwrap();
                let get_tensor = |name: &str| -> Option<Vec<f32>> { model.get_tensor_f32(name) };
                alice_train::fp32_cache::build_cache(
                    &get_tensor,
                    &config.weight_prefix,
                    cache_base,
                    &config.model,
                )
                .unwrap_or_else(|e| {
                    eprintln!("[ALICE-Train] FP32 キャッシュ構築失敗: {e}");
                    std::process::exit(1);
                });
                let cache_mb =
                    alice_train::fp32_cache::cache_size_bytes(&config.model) as f64 / 1e6;
                println!(
                    "    FP32 キャッシュ完了: {:.0} MB ({:.1}s)",
                    cache_mb,
                    t0.elapsed().as_secs_f64()
                );
            } else {
                println!("    FP32 キャッシュ検出済み");
            }
            println!("    RAM 節約: 全層プリロード不要 (~35GB → 1層分 ~1GB)");
            Vec::new()
        };
        println!();

        // --- 実行環境記録 ---
        {
            let run_record = format!(
                concat!(
                    "{{",
                    "\"timestamp\":\"{}\",",
                    "\"config_file\":\"{}\",",
                    "\"seq_len\":{},",
                    "\"batch_size\":{},",
                    "\"gradient_accumulation_steps\":{},",
                    "\"learning_rate\":{},",
                    "\"min_lr\":{},",
                    "\"max_grad_norm\":{},",
                    "\"total_steps\":{},",
                    "\"warmup_steps\":{},",
                    "\"weight_decay\":{},",
                    "\"preload_all_layers\":{},",
                    "\"bf16_delta\":{},",
                    "\"hidden_size\":{},",
                    "\"num_layers\":{},",
                    "\"vocab_size\":{},",
                    "\"hostname\":\"{}\",",
                    "\"gpu\":\"{}\",",
                    "\"cuda_features\":{},",
                    "\"rust_version\":\"{}\",",
                    "\"resume_step\":{}",
                    "}}"
                ),
                chrono_now(),
                cli.config,
                config.seq_len,
                config.batch_size,
                config.gradient_accumulation_steps,
                config.learning_rate,
                config.min_lr,
                config.max_grad_norm,
                config.total_steps,
                config.warmup_steps,
                config.weight_decay,
                config.preload_all_layers,
                config.bf16_delta,
                config.model.hidden_size,
                config.model.num_hidden_layers,
                config.model.vocab_size,
                hostname(),
                gpu_name(),
                cfg!(feature = "cuda"),
                env!("CARGO_PKG_VERSION"),
                global_step,
            );
            let record_path = format!("{}/run_record.json", config.checkpoint_dir);
            match fs::write(&record_path, &run_record) {
                Ok(()) => println!("  実行環境記録: {record_path}"),
                Err(e) => eprintln!("  実行環境記録失敗: {e}"),
            }
            // ログにも出力
            println!("  config: {}", cli.config);
            println!(
                "  seq_len={} batch={} grad_accum={} lr={:.2e} max_grad_norm={} steps={}",
                config.seq_len,
                config.batch_size,
                config.gradient_accumulation_steps,
                config.learning_rate,
                config.max_grad_norm,
                config.total_steps,
            );
            println!(
                "  gpu={} cuda={} preload={} hostname={}",
                gpu_name(),
                cfg!(feature = "cuda"),
                config.preload_all_layers,
                hostname(),
            );
        }

        // --- 学習ループ ---
        println!(
            "━━━ 学習開始 (step {global_step}/{}) ━━━",
            config.total_steps
        );
        let train_start = Instant::now();
        let mut batch_idx = 0usize;
        let seq_len = config.seq_len;
        let mut collapse_countdown: i32 = -1; // -1 = 正常, >0 = 崩壊猶予カウント
        let mut last_ckpt_time = Instant::now();
        const CKPT_KEEP_COUNT: usize = 10; // チェックポイント保持世代数
        let mut ckpt_history: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let mut last_loss = 0.0f32;
        let mut stuck_loss_count: usize = 0; // 同一loss連続カウント
        let mut stuck_loss_value: f32 = 0.0;

        // Gradient accumulation 用アキュムレータ
        let mut accumulated_grads: Vec<Option<Qwen35WeightGrads>> =
            (0..config.model.num_hidden_layers).map(|_| None).collect();
        let mut accum_count: usize = 0;

        // FakeQuantize バッファ — 事前確保してゼロアロケーション (preload mode 用)
        let mut fq_buffers: Vec<Qwen35LayerWeights> = if config.preload_all_layers {
            layers.iter().map(|l| l.clone()).collect()
        } else {
            Vec::new()
        };

        while global_step < config.total_steps {
            let lr = scheduler.get_lr(global_step);
            let step_start = Instant::now();

            if batch_idx >= num_batches {
                loader.shuffle_epoch();
                batch_idx = 0;
            }
            let batch = loader.get_batch(batch_idx, &dataset);
            batch_idx += 1;

            let mut total_loss = 0.0f32;
            let mut token_count = 0usize;

            for b in 0..batch.actual_batch_size {
                let token_ids: Vec<u32> = batch.input_ids[b * seq_len..(b + 1) * seq_len].to_vec();

                if config.preload_all_layers {
                    // ── Preloaded Gradient Checkpointing ──
                    use alice_train::qwen35_forward::qwen35_layer_forward;

                    let num_l = config.model.num_hidden_layers;
                    let profile = global_step < 3 && b == 0; // 最初の3 stepでプロファイル

                    // 1. Embedding
                    let t_embed = Instant::now();
                    let mut hidden_states = vec![0.0f32; seq_len * hidden];
                    for (t, &tok) in token_ids.iter().enumerate() {
                        let tok = (tok as usize) % vocab_size;
                        hidden_states[t * hidden..(t + 1) * hidden]
                            .copy_from_slice(&embedding_table[tok * hidden..(tok + 1) * hidden]);
                    }
                    let embed_ms = t_embed.elapsed().as_millis();

                    // 2. Eval forward: 各層入力を保存
                    let t_fwd = Instant::now();
                    let mut saved_inputs: Vec<Vec<f32>> = Vec::with_capacity(num_l);
                    let mut clone_ms_total = 0u128;
                    let mut layer_ms_total = 0u128;
                    for i in 0..num_l {
                        let tc = Instant::now();
                        saved_inputs.push(hidden_states.clone());
                        clone_ms_total += tc.elapsed().as_millis();
                        let tl = Instant::now();
                        // FakeQuantize: in-place (ゼロアロケーション)
                        layers[i].fake_quantize_into(&mut fq_buffers[i]);
                        alice_train::qwen35_forward::qwen35_layer_forward_eval_inplace(
                            &mut hidden_states,
                            &fq_buffers[i],
                            &config.model,
                            seq_len,
                        );
                        layer_ms_total += tl.elapsed().as_millis();
                    }
                    let fwd_ms = t_fwd.elapsed().as_millis();
                    if profile {
                        eprintln!(
                            "  [FWD_DETAIL] clone_inputs={}ms layers={}ms total={}ms",
                            clone_ms_total, layer_ms_total, fwd_ms,
                        );
                    }

                    // 3. Output norm + lm_head → logits
                    let t_head = Instant::now();
                    alice_train::blas::blas_rmsnorm(
                        &mut hidden_states,
                        &output_norm,
                        hidden,
                        config.model.rms_norm_eps,
                    );
                    let mut logits = vec![0.0f32; seq_len * vocab_size];
                    alice_train::blas::blas_matmul_bt(
                        &hidden_states,
                        &lm_head,
                        &mut logits,
                        seq_len,
                        vocab_size,
                        hidden,
                    );
                    drop(hidden_states);
                    let head_ms = t_head.elapsed().as_millis();

                    // 4. Loss + d_logits (seq_len で正規化 — loss 平均と一致)
                    let t_loss = Instant::now();
                    let mut d_logits_all = vec![0.0f32; seq_len * vocab_size];
                    let grad_scale = 1.0 / seq_len as f32;
                    for t in 0..seq_len {
                        let target = batch.target_ids[b * seq_len + t] as usize % vocab_size;
                        let tl = &logits[t * vocab_size..(t + 1) * vocab_size];
                        let (loss, dl) = cross_entropy_loss(tl, target);
                        total_loss += loss;
                        token_count += 1;
                        for (dst, &src) in d_logits_all[t * vocab_size..(t + 1) * vocab_size]
                            .iter_mut().zip(dl.iter())
                        {
                            *dst = src * grad_scale;
                        }
                    }
                    drop(logits);
                    let loss_ms = t_loss.elapsed().as_millis();

                    // 5. Backward through lm_head
                    let t_bwd_head = Instant::now();
                    let mut d_hidden = vec![0.0f32; seq_len * hidden];
                    alice_train::blas::blas_matmul_nn(
                        &d_logits_all,
                        &lm_head,
                        &mut d_hidden,
                        seq_len,
                        hidden,
                        vocab_size,
                    );
                    drop(d_logits_all);
                    let bwd_head_ms = t_bwd_head.elapsed().as_millis();

                    if profile {
                        eprintln!(
                            "  [PROFILE step={global_step} b={b}] embed={}ms fwd_32layers={}ms lm_head={}ms loss={}ms bwd_lm_head={}ms",
                            embed_ms, fwd_ms, head_ms, loss_ms, bwd_head_ms,
                        );
                    }

                    // 6. Gradient checkpointing backward with FakeQuantize:
                    let t_bwd_layers = Instant::now();
                    for i in (0..num_l).rev() {
                        let saved_input = saved_inputs.pop().unwrap();

                        // FakeQuantize — backward の forward 再計算も fq 重みで (in-place)
                        layers[i].fake_quantize_into(&mut fq_buffers[i]);

                        #[cfg(feature = "cuda")]
                        let use_fused = alice_train::blas::cuda_blas_available()
                            && matches!(layers[i], Qwen35LayerWeights::DeltaNet(_));

                        #[cfg(not(feature = "cuda"))]
                        let use_fused = false;

                        let (d_input, grads) = if use_fused {
                            #[cfg(feature = "cuda")]
                            {
                                match &mut fq_buffers[i] {
                                    Qwen35LayerWeights::DeltaNet(w_fq) => {
                                        let mut wg = alice_train::qwen35_backward::DeltaNetWeightGrads::zeros(&config.model);
                                        let d_in =
                                            alice_train::qwen35_backward::deltanet_layer_gc_fused(
                                                &saved_input, &d_hidden, w_fq,
                                                &config.model, seq_len, &mut wg,
                                            );
                                        (d_in, Some(Qwen35WeightGrads::DeltaNet(wg)))
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            #[cfg(not(feature = "cuda"))]
                            {
                                unreachable!()
                            }
                        } else {
                            let mut recompute_input = saved_input;
                            let cache = qwen35_layer_forward(
                                &mut recompute_input, &fq_buffers[i], &config.model, seq_len,
                            );
                            let (d_in, grads) = qwen35_layer_backward(
                                &d_hidden, &cache, &fq_buffers[i], &config.model, seq_len, lr, config.weight_decay,
                            );
                            drop(cache);
                            (d_in, Some(grads))
                        };

                        d_hidden = d_input;

                        // 勾配蓄積
                        if let Some(grads) = grads {
                            if let Some(acc) = &mut accumulated_grads[i] {
                                acc.add_assign(&grads);
                            } else {
                                accumulated_grads[i] = Some(grads);
                            }
                        }
                    }

                    // gradient accumulation: 蓄積完了時に SGD を FP32 shadow weights に適用 (STE)
                    accum_count += 1;
                    if accum_count >= config.gradient_accumulation_steps {
                        for i in 0..num_l {
                            if let Some(ref mut g) = accumulated_grads[i] {
                                let accum_scale = 1.0 / accum_count as f32;
                                g.scale(accum_scale);
                                g.clip_grad_norm(config.max_grad_norm);

                                // Layer 0: フリップ率計測 (STE の効果を可視化)
                                if i == 0 {
                                    // 更新前の Ternary 重み
                                    let old_fq = layers[0].fake_quantize();
                                    g.apply_sgd(&mut layers[0], lr, config.weight_decay);
                                    // 更新後の Ternary 重み
                                    let new_fq = layers[0].fake_quantize();
                                    // in_proj_qkv でフリップ率を計算
                                    if let (Qwen35LayerWeights::DeltaNet(ref o), Qwen35LayerWeights::DeltaNet(ref n)) = (&old_fq, &new_fq) {
                                        let flipped = o.in_proj_qkv.iter().zip(n.in_proj_qkv.iter())
                                            .filter(|(&a, &b)| (a - b).abs() > 1e-5).count();
                                        let total = o.in_proj_qkv.len();
                                        let ratio = flipped as f64 / total as f64 * 100.0;
                                        eprintln!("    [Layer 0 qkv] Ternary Flip: {:.4}% ({}/{})", ratio, flipped, total);
                                    }
                                } else {
                                    g.apply_sgd(&mut layers[i], lr, config.weight_decay);
                                }
                            }
                        }
                        for g in accumulated_grads.iter_mut() { *g = None; }
                        accum_count = 0;
                    }
                    if profile {
                        let bwd_layers_ms = t_bwd_layers.elapsed().as_millis();
                        eprintln!(
                            "  [PROFILE step={global_step} b={b}] bwd_32layers={}ms | total_micro_batch={}ms",
                            bwd_layers_ms,
                            embed_ms + fwd_ms + head_ms + loss_ms + bwd_head_ms + bwd_layers_ms,
                        );
                    }
                } else {
                    // ── Streaming QAT (FakeQuantize + gradient accumulation) ──
                    // Forward: FP32重みを fake_quantize → eval forward
                    // Backward: fake_quantize された重みで forward 再計算 → backward
                    // SGD: 元の FP32 重み (shadow weights) に適用 (STE)
                    use alice_train::qwen35_forward::qwen35_layer_forward;

                    let cache_base = &config.checkpoint_dir;
                    let num_l = config.model.num_hidden_layers;

                    // 1. Embedding
                    let mut hidden_states = vec![0.0f32; seq_len * hidden];
                    for (t, &tok) in token_ids.iter().enumerate() {
                        let tok = (tok as usize) % vocab_size;
                        hidden_states[t * hidden..(t + 1) * hidden]
                            .copy_from_slice(&embedding_table[tok * hidden..(tok + 1) * hidden]);
                    }

                    // 2. Streaming eval forward with FakeQuantize (in-place)
                    let mut saved_inputs: Vec<Vec<f32>> = Vec::with_capacity(num_l);
                    // FQ バッファ: DeltaNet/FullAttn 各1つずつ事前確保
                    let mut fq_dn: Option<Qwen35LayerWeights> = None;
                    let mut fq_fa: Option<Qwen35LayerWeights> = None;
                    for i in 0..num_l {
                        saved_inputs.push(hidden_states.clone());
                        let layer_w = alice_train::fp32_cache::load_layer_from_cache(
                            cache_base, i, &config.model,
                        ).unwrap_or_else(|e| {
                            eprintln!("[ALICE-Train] layer {i} 読み込み失敗: {e}");
                            std::process::exit(1);
                        });
                        // FakeQuantize: in-place (variant別バッファ)
                        let fq_buf = match &layer_w {
                            Qwen35LayerWeights::DeltaNet(_) => &mut fq_dn,
                            Qwen35LayerWeights::FullAttention(_) => &mut fq_fa,
                        };
                        if let Some(ref mut buf) = fq_buf {
                            layer_w.fake_quantize_into(buf);
                        } else {
                            *fq_buf = Some(layer_w.fake_quantize());
                        }
                        alice_train::qwen35_forward::qwen35_layer_forward_eval_inplace(
                            &mut hidden_states, fq_buf.as_ref().unwrap(), &config.model, seq_len,
                        );
                    }

                    // 3. Output norm + lm_head
                    alice_train::blas::blas_rmsnorm(
                        &mut hidden_states, &output_norm, hidden, config.model.rms_norm_eps,
                    );
                    let mut logits = vec![0.0f32; seq_len * vocab_size];
                    alice_train::blas::blas_matmul_bt(
                        &hidden_states, &lm_head, &mut logits, seq_len, vocab_size, hidden,
                    );

                    // 4. Loss + d_logits (seq_len で正規化)
                    let mut d_logits_all = vec![0.0f32; seq_len * vocab_size];
                    let grad_scale = 1.0 / seq_len as f32;
                    for t in 0..seq_len {
                        let target = batch.target_ids[b * seq_len + t] as usize % vocab_size;
                        let tl = &logits[t * vocab_size..(t + 1) * vocab_size];
                        let (loss, dl) = cross_entropy_loss(tl, target);
                        total_loss += loss;
                        token_count += 1;
                        for (dst, &src) in d_logits_all[t * vocab_size..(t + 1) * vocab_size]
                            .iter_mut().zip(dl.iter())
                        {
                            *dst = src * grad_scale;
                        }
                    }
                    drop(logits);

                    // 5. Backward through lm_head
                    let mut d_hidden = vec![0.0f32; seq_len * hidden];
                    alice_train::blas::blas_matmul_nn(
                        &d_logits_all, &lm_head, &mut d_hidden, seq_len, hidden, vocab_size,
                    );
                    drop(d_logits_all);

                    // 6. Streaming backward (FakeQuantize + gradient accumulation)
                    // backward は fake_quantize 済み重みで forward 再計算 → backward
                    // 勾配は accumulated_grads に蓄積 (STE: 勾配を素通り)
                    for i in (0..num_l).rev() {
                        let layer_w_orig = alice_train::fp32_cache::load_layer_from_cache(
                            cache_base, i, &config.model,
                        ).unwrap_or_else(|e| {
                            eprintln!("[ALICE-Train] backward layer {i} 読み込み失敗: {e}");
                            std::process::exit(1);
                        });
                        // FakeQuantize — backward の forward 再計算も fq 重みで (in-place)
                        let fq_buf = match &layer_w_orig {
                            Qwen35LayerWeights::DeltaNet(_) => &mut fq_dn,
                            Qwen35LayerWeights::FullAttention(_) => &mut fq_fa,
                        };
                        if let Some(ref mut buf) = fq_buf {
                            layer_w_orig.fake_quantize_into(buf);
                        } else {
                            *fq_buf = Some(layer_w_orig.fake_quantize());
                        }
                        let fq_ref = fq_buf.as_ref().unwrap();

                        let mut recompute_input = saved_inputs.pop().unwrap();
                        let cache = qwen35_layer_forward(
                            &mut recompute_input, fq_ref, &config.model, seq_len,
                        );

                        // backward は fq 重みに対して計算 → STE で勾配素通り
                        let (d_input, grads) = qwen35_layer_backward(
                            &d_hidden, &cache, fq_ref, &config.model, seq_len, lr, config.weight_decay,
                        );
                        d_hidden = d_input;
                        drop(cache);

                        // 勾配蓄積 (gradient accumulation)
                        if let Some(acc) = &mut accumulated_grads[i] {
                            acc.add_assign(&grads);
                        } else {
                            accumulated_grads[i] = Some(grads);
                        }
                    }

                    // gradient accumulation: 蓄積完了時に SGD 適用
                    accum_count += 1;
                    if accum_count >= config.gradient_accumulation_steps {
                        for i in 0..num_l {
                            if let Some(ref mut g) = accumulated_grads[i] {
                                let accum_scale = 1.0 / accum_count as f32;
                                g.scale(accum_scale);
                                g.clip_grad_norm(config.max_grad_norm);

                                // FP32 shadow weights を読み込み → SGD → 書き戻し (STE)
                                let mut layer_w = alice_train::fp32_cache::load_layer_from_cache(
                                    cache_base, i, &config.model,
                                ).unwrap_or_else(|e| {
                                    eprintln!("[ALICE-Train] SGD layer {i} 読み込み失敗: {e}");
                                    std::process::exit(1);
                                });
                                g.apply_sgd(&mut layer_w, lr, config.weight_decay);
                                alice_train::fp32_cache::save_layer_to_cache(
                                    cache_base, i, &layer_w, &config.model,
                                ).unwrap_or_else(|e| {
                                    eprintln!("[ALICE-Train] layer {i} 書き戻し失敗: {e}");
                                });
                            }
                        }
                        // リセット
                        for g in accumulated_grads.iter_mut() { *g = None; }
                        accum_count = 0;
                    }
                }
            }

            let avg_loss = if token_count > 0 {
                total_loss / token_count as f32
            } else {
                0.0
            };
            last_loss = avg_loss;
            let step_duration = step_start.elapsed();

            log.append(LogEntry::new(0, global_step, avg_loss, lr, 0.0));

            // ── 崩壊検知 (NaN/Inf/異常Loss/固定Loss) → 自動停止 ──
            let is_collapsed = avg_loss.is_nan()
                || avg_loss.is_infinite()
                || (global_step > 200 && avg_loss > 15.0);

            // 固定Loss検知: 同じ値が5step以上続く = モデル崩壊
            if (avg_loss - stuck_loss_value).abs() < 1e-4 {
                stuck_loss_count += 1;
            } else {
                stuck_loss_count = 0;
                stuck_loss_value = avg_loss;
            }
            let is_stuck = stuck_loss_count >= 5 && global_step > 200;

            if is_collapsed || is_stuck {
                let reason = if avg_loss.is_nan() || avg_loss.is_infinite() {
                    "NaN/Inf"
                } else if is_stuck {
                    "固定Loss (モデル崩壊)"
                } else {
                    "異常Loss上昇"
                };
                eprintln!("╔══════════════════════════════════════════════════════════╗");
                eprintln!("║  *** 崩壊検知: {reason} — 学習を緊急停止 ***");
                eprintln!("╚══════════════════════════════════════════════════════════╝");
                eprintln!("  step: {global_step}, loss: {avg_loss:.4}, lr: {lr:.2e}");
                eprintln!("  fp32_cacheは書き戻しません（崩壊前の重みを保全）");
                let log_path = format!("{}/train_log.csv", config.checkpoint_dir);
                let _ = log.save_csv_to_file(&log_path);
                let crash_json = format!(
                    "{{\"step\":{},\"loss\":{:.6},\"lr\":{:.2e},\"status\":\"COLLAPSED\",\"reason\":\"{}\"}}",
                    global_step, avg_loss, lr, reason
                );
                let _ = fs::write(
                    format!("{}/resume_state.json", config.checkpoint_dir),
                    &crash_json,
                );
                // 崩壊時: 全チェックポイント+FP32キャッシュをBoxに退避
                let _ = std::process::Command::new("bash")
                    .arg("scripts/upload_box.sh")
                    .arg(&config.checkpoint_dir)
                    .arg("--all-checkpoints")
                    .status();
                std::process::exit(1);
            }
            // Loss 緩やかな上昇検知 — 5ステップ猶予後に自動停止
            if global_step > 100 {
                if let Some(prev) = log.entries().iter().rev().nth(10) {
                    if prev.loss > 0.0 && avg_loss > prev.loss * 2.0 && avg_loss > 12.0 {
                        if collapse_countdown < 0 {
                            collapse_countdown = 5;
                            eprintln!("  *** 崩壊警告: Loss {:.2} → {:.2} (2倍超) — 残り{collapse_countdown}ステップで自動停止 ***", prev.loss, avg_loss);
                        }
                    } else if collapse_countdown > 0 {
                        eprintln!("  崩壊警告解除: Loss回復 ({avg_loss:.4})");
                        collapse_countdown = -1;
                    }
                }
            }
            if collapse_countdown > 0 {
                collapse_countdown -= 1;
                if collapse_countdown == 0 {
                    eprintln!("╔══════════════════════════════════════════════════════════╗");
                    eprintln!("║  *** 崩壊確定: Loss回復せず — 学習を自動停止 ***        ║");
                    eprintln!("╚══════════════════════════════════════════════════════════╝");
                    eprintln!("  step: {global_step}, loss: {avg_loss:.4}");
                    eprintln!("  fp32_cacheは書き戻しません（崩壊前の重みを保全）");
                    let log_path = format!("{}/train_log.csv", config.checkpoint_dir);
                    let _ = log.save_csv_to_file(&log_path);
                    let crash_json = format!(
                        "{{\"step\":{},\"loss\":{:.6},\"lr\":{:.2e},\"status\":\"COLLAPSED_DIVERGED\"}}",
                        global_step, avg_loss, lr
                    );
                    let _ = fs::write(
                        format!("{}/resume_state.json", config.checkpoint_dir),
                        &crash_json,
                    );
                    // 崩壊時: 全チェックポイント+FP32キャッシュをBoxに退避
                    let _ = std::process::Command::new("bash")
                        .arg("scripts/upload_box.sh")
                        .arg(&config.checkpoint_dir)
                        .arg("--all-checkpoints")
                        .status();
                    std::process::exit(1);
                }
            }

            // resume_state.json 保存 (毎ステップ — 再起動時に復元)
            let resume_json = format!(
                "{{\"step\":{},\"loss\":{:.6},\"lr\":{:.2e}}}",
                global_step, avg_loss, lr
            );
            let resume_path = format!("{}/resume_state.json", config.checkpoint_dir);
            let _ = fs::write(&resume_path, &resume_json);

            {
                let elapsed = train_start.elapsed();
                let steps_done = (global_step - resume_start_step).max(1) as f64;
                let steps_per_sec = steps_done / elapsed.as_secs_f64().max(0.001);
                let eta_secs = (config.total_steps - global_step) as f64 / steps_per_sec.max(0.001);
                println!(
                    "  step {global_step:>5}/{} | loss: {avg_loss:.4} | lr: {lr:.2e} | {:.1}ms/step | ETA: {eta_secs:.0}s",
                    config.total_steps,
                    step_duration.as_secs_f64() * 1000.0,
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }

            // Eval (オプション)
            if global_step > 0 && global_step % config.eval_interval == 0 && eval_dataset.is_some()
            {
                let eval_ds = eval_dataset.as_ref().unwrap();
                let eval_dl_config = DataLoaderConfig::new()
                    .with_seq_len(seq_len)
                    .with_batch_size(1)
                    .with_shuffle(false)
                    .with_seed(0);
                let eval_loader = DataLoader::new(eval_ds, eval_dl_config);
                let eval_batches = eval_loader.num_batches().min(10);

                let mut eval_loss = 0.0f32;
                let mut eval_tokens = 0usize;
                for eb in 0..eval_batches {
                    let eval_batch = eval_loader.get_batch(eb, eval_ds);
                    let eval_ids: Vec<u32> = eval_batch.input_ids[..seq_len].to_vec();
                    let eval_logits = qwen35_model_forward_eval(
                        &eval_ids,
                        &embedding_table,
                        &layers,
                        &output_norm,
                        &lm_head,
                        &config.model,
                    );
                    for t in 0..seq_len {
                        let target = eval_batch.target_ids[t] as usize % vocab_size;
                        let tl = &eval_logits[t * vocab_size..(t + 1) * vocab_size];
                        let (l, _) = cross_entropy_loss(tl, target);
                        eval_loss += l;
                        eval_tokens += 1;
                    }
                }
                if eval_tokens > 0 {
                    let avg_eval = eval_loss / eval_tokens as f32;
                    println!("    eval_loss: {avg_eval:.4} ({eval_tokens} tokens)");
                }
            }

            // チェックポイント (15分間隔、2世代保持)
            const CKPT_INTERVAL_SECS: u64 = 15 * 60;
            if global_step > 0 && last_ckpt_time.elapsed().as_secs() >= CKPT_INTERVAL_SECS {
                // preloadモード: 更新済み重みをFP32キャッシュに書き戻し (resume用)
                if config.preload_all_layers {
                    let cache_base = &config.checkpoint_dir;
                    for li in 0..config.model.num_hidden_layers {
                        if let Err(e) = alice_train::fp32_cache::save_layer_to_cache(
                            cache_base,
                            li,
                            &layers[li],
                            &config.model,
                        ) {
                            eprintln!("  レイヤー{li} キャッシュ書き戻し失敗: {e}");
                        }
                    }
                    println!(
                        "    FP32キャッシュ書き戻し完了 ({} layers)",
                        config.model.num_hidden_layers
                    );
                    // ページキャッシュ解放 — 26GB書き込み後のメモリ効率低下を防止
                    #[cfg(target_os = "linux")]
                    alice_train::fp32_cache::drop_page_cache(cache_base, &config.model);
                    // ヒープ圧縮 — checkpointファイル書き込み後のフラグメンテーション防止
                    #[cfg(target_os = "linux")]
                    unsafe { libc::malloc_trim(0); }
                }

                let ckpt_path = format!("{}/step_{global_step}.bin", config.checkpoint_dir);
                println!("  チェックポイント保存: {ckpt_path}");
                let meta = alice_train::CheckpointMeta {
                    version: 1,
                    epoch: 0,
                    step: global_step,
                    loss: avg_loss,
                    learning_rate: lr,
                    weight_count: 0,
                    optimizer_state_count: 0,
                };
                let ckpt = CheckpointData {
                    meta,
                    weights: vec![],
                    optimizer_state: vec![],
                };
                if let Err(e) =
                    std::fs::File::create(&ckpt_path).and_then(|mut f| ckpt.save(&mut f))
                {
                    eprintln!("  チェックポイント保存失敗: {e}");
                }

                // 10世代保持 — 崩壊時にロールバック可能
                ckpt_history.push_back(global_step);
                if ckpt_history.len() > CKPT_KEEP_COUNT {
                    if let Some(oldest) = ckpt_history.pop_front() {
                        let old_path = format!("{}/step_{oldest}.bin", config.checkpoint_dir);
                        if std::path::Path::new(&old_path).exists() {
                            let _ = std::fs::remove_file(&old_path);
                            println!("    古いチェックポイント削除: {old_path}");
                        }
                    }
                }
                last_ckpt_time = Instant::now();
            }

            global_step += 1;
        }

        let total_duration = train_start.elapsed();
        println!();
        println!("━━━ 学習完了 ━━━");
        println!("  総ステップ: {global_step}");
        println!(
            "  学習時間: {:.1}s ({:.1} steps/s)",
            total_duration.as_secs_f64(),
            global_step as f64 / total_duration.as_secs_f64().max(0.001)
        );
        print_rss("学習完了");

        let log_path = format!("{}/train_log.csv", config.checkpoint_dir);
        if let Err(e) = log.save_csv_to_file(&log_path) {
            eprintln!("  ログ保存失敗: {e}");
        } else {
            println!("  ログ: {log_path}");
        }

        // ── 自動 ternary エクスポート ──
        // 学習完了後、FP32キャッシュを書き戻してから .alice ファイルを生成
        if config.preload_all_layers {
            println!();
            println!("━━━ Ternary エクスポート ━━━");
            let export_start = Instant::now();

            // FP32キャッシュ最終書き戻し
            let cache_base = &config.checkpoint_dir;
            for li in 0..config.model.num_hidden_layers {
                if let Err(e) = alice_train::fp32_cache::save_layer_to_cache(
                    cache_base,
                    li,
                    &layers[li],
                    &config.model,
                ) {
                    eprintln!("  レイヤー{li} キャッシュ書き戻し失敗: {e}");
                }
            }

            let lm_head_ref = if embedding_table == lm_head {
                None // tied
            } else {
                Some(lm_head.as_slice())
            };

            let output_path = format!("{}/ALICE-Cognitive-9B-Ternary.alice", config.checkpoint_dir);
            match std::fs::File::create(&output_path) {
                Ok(file) => {
                    let mut writer = std::io::BufWriter::new(file);
                    match alice_train::export::export_alice_model(
                        &mut writer,
                        &config.model,
                        cache_base,
                        &embedding_table,
                        &output_norm,
                        lm_head_ref,
                        global_step,
                        last_loss,
                    ) {
                        Ok(stats) => {
                            println!(
                                "  .alice エクスポート完了: {} ({:.2} GB, {:.1}s)",
                                output_path,
                                stats.total_bytes as f64 / 1e9,
                                export_start.elapsed().as_secs_f64()
                            );
                        }
                        Err(e) => {
                            eprintln!("  .alice エクスポート失敗: {e}");
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  出力ファイル作成失敗: {output_path}: {e}");
                }
            }
        }

        // ── Box SFTP アップロード ──
        println!();
        println!("━━━ Box SFTP アップロード ━━━");
        let upload_status = std::process::Command::new("bash")
            .arg("scripts/upload_box.sh")
            .arg(&config.checkpoint_dir)
            .status();
        match upload_status {
            Ok(s) if s.success() => println!("  Box アップロード完了"),
            Ok(s) => eprintln!("  Box アップロード失敗 (exit={})", s.code().unwrap_or(-1)),
            Err(e) => eprintln!("  Box アップロード実行失敗: {e}"),
        }
    }
}
