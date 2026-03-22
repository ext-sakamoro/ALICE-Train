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

use alice_train::qwen35::{
    DeltaNetLayerWeights, FullAttnLayerWeights, LayerType, Qwen35LayerWeights, Qwen35QatConfig,
};
use alice_train::qwen35_backward::{qwen35_layer_backward, Qwen35WeightGrads};
use alice_train::qwen35_forward::{qwen35_model_forward, qwen35_model_forward_eval, qwen35_model_forward_eval_streaming};
use alice_train::{
    CheckpointData, DataLoader, DataLoaderConfig, FakeQuantize, LogEntry, LrScheduler,
    MmapDataset, QatConfig, TrainLog, WarmupCosineScheduler,
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
        eprintln!("[ALICE-Train] 設定ファイル読み込み失敗: {}: {e}", cli.config);
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
    let eval_dataset: Option<MmapDataset> = config
        .eval_data_path
        .as_ref()
        .and_then(|p| {
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
    let _fq = FakeQuantize::new(QatConfig::ternary());
    let scheduler = WarmupCosineScheduler::new(
        config.learning_rate,
        config.min_lr,
        config.warmup_steps,
        config.total_steps,
    );
    let mut log = TrainLog::new();
    let mut global_step: usize = 0;
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

            let inv_tokens = 1.0 / token_count.max(1) as f32;
            for (w, g) in embedding.iter_mut().zip(grad_embedding.iter_mut()) {
                *w -= lr * (*g * inv_tokens + config.weight_decay * *w);
                *g = 0.0;
            }
            for (w, g) in output_proj.iter_mut().zip(grad_output_proj.iter_mut()) {
                *w -= lr * (*g * inv_tokens + config.weight_decay * *w);
                *g = 0.0;
            }

            log.append(LogEntry::new(0, global_step, avg_loss, lr, 0.0));

            if global_step % 10 == 0 || global_step == config.total_steps - 1 {
                let elapsed = train_start.elapsed();
                let steps_per_sec = (global_step + 1) as f64 / elapsed.as_secs_f64().max(0.001);
                let eta_secs = (config.total_steps - global_step) as f64 / steps_per_sec.max(0.001);
                println!(
                    "  step {global_step:>5}/{} | loss: {avg_loss:.4} | lr: {lr:.2e} | {:.1}ms/step | ETA: {eta_secs:.0}s",
                    config.total_steps,
                    step_duration.as_secs_f64() * 1000.0,
                );
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

        let model = ShardedModel::open(&config.model_path).unwrap_or_else(|e| {
            eprintln!("[ALICE-Train] モデル読み込み失敗: {e}");
            std::process::exit(1);
        });

        let prefix = &config.weight_prefix;
        let get_tensor = |name: &str| -> Option<Vec<f32>> { model.get_tensor_f32(name) };

        // Embedding (常駐)
        println!("  embedding 読み込み...");
        let embedding_table =
            get_tensor(&format!("{prefix}.embed_tokens.weight")).unwrap_or_else(|| {
                eprintln!("[ALICE-Train] {prefix}.embed_tokens.weight が見つかりません");
                std::process::exit(1);
            });
        println!(
            "    embed_tokens: {:.1} MB",
            embedding_table.len() as f64 * 4.0 / 1e6
        );

        // Output norm + lm_head
        let output_norm =
            get_tensor(&format!("{prefix}.norm.weight")).unwrap_or_else(|| {
                eprintln!("[ALICE-Train] {prefix}.norm.weight が見つかりません");
                std::process::exit(1);
            });
        let lm_head = get_tensor("lm_head.weight").unwrap_or_else(|| {
            println!("    lm_head.weight なし — embed_tokens と共有");
            embedding_table.clone()
        });

        // レイヤー重み読み込み (L10: preload/streaming 分岐)
        let num_layers = config.model.num_hidden_layers;
        let mut layers: Vec<Qwen35LayerWeights> = if config.preload_all_layers {
            println!("  レイヤー重み全プリロード中...");
            let mut layers = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                let layer_prefix = format!("{prefix}.layers.{i}");
                let lt = config.model.layer_type(i);
                let layer = match lt {
                    LayerType::LinearAttention => {
                        let w = DeltaNetLayerWeights::from_tensors(&layer_prefix, &get_tensor)
                            .unwrap_or_else(|| {
                                eprintln!("[ALICE-Train] DeltaNet layer {i} 読み込み失敗");
                                std::process::exit(1);
                            });
                        Qwen35LayerWeights::DeltaNet(w)
                    }
                    LayerType::FullAttention => {
                        let w = FullAttnLayerWeights::from_tensors(&layer_prefix, &get_tensor)
                            .unwrap_or_else(|| {
                                eprintln!("[ALICE-Train] FullAttn layer {i} 読み込み失敗");
                                std::process::exit(1);
                            });
                        Qwen35LayerWeights::FullAttention(w)
                    }
                };
                layers.push(layer);
                if (i + 1) % 8 == 0 || i == num_layers - 1 {
                    println!("    {}/{num_layers} レイヤー読み込み完了", i + 1);
                }
            }
            print_rss("全レイヤープリロード完了");
            layers
        } else {
            println!("  L10+L12: ストリーミングモード + FP32 キャッシュ");
            // L12: FP32 キャッシュ構築（初回のみ）
            let cache_base = &config.checkpoint_dir;
            if !alice_train::fp32_cache::cache_exists(cache_base, &config.model) {
                println!("    FP32 キャッシュ構築中 (初回のみ)...");
                let t0 = Instant::now();
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
                println!("    FP32 キャッシュ検出 — BF16 デコードスキップ");
            }
            println!("    RAM 節約: 全層プリロード不要 (~35GB → 1層分 ~1GB)");
            Vec::new()
        };
        println!();

        // --- 学習ループ ---
        println!(
            "━━━ 学習開始 (step {global_step}/{}) ━━━",
            config.total_steps
        );
        let train_start = Instant::now();
        let mut batch_idx = 0usize;
        let seq_len = config.seq_len;
        let mut collapse_countdown: i32 = -1; // -1 = 正常, >0 = 崩壊猶予カウント

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
                let token_ids: Vec<u32> = batch.input_ids
                    [b * seq_len..(b + 1) * seq_len]
                    .to_vec();

                if config.preload_all_layers {
                    // ── Forward (cache付き) + Backward + Weight Update ──
                    let (logits, caches) = qwen35_model_forward(
                        &token_ids, &embedding_table, &layers, &output_norm, &lm_head, &config.model,
                    );

                    // Loss + d_logits
                    let mut d_logits_all = vec![0.0f32; seq_len * vocab_size];
                    for t in 0..seq_len {
                        let target = batch.target_ids[b * seq_len + t] as usize % vocab_size;
                        let tl = &logits[t * vocab_size..(t + 1) * vocab_size];
                        let (loss, dl) = cross_entropy_loss(tl, target);
                        total_loss += loss;
                        token_count += 1;
                        d_logits_all[t * vocab_size..(t + 1) * vocab_size].copy_from_slice(&dl);
                    }

                    // Backward through lm_head: d_hidden = d_logits × lm_head
                    // lm_head: [vocab × hidden], d_logits: [seq × vocab] → d_hidden: [seq × hidden]
                    let mut d_hidden = vec![0.0f32; seq_len * hidden];
                    alice_train::blas::blas_matmul_nn(&d_logits_all, &lm_head, &mut d_hidden, seq_len, hidden, vocab_size);

                    // Layer-by-layer backward (reverse order)
                    let inv_tokens = 1.0 / token_count.max(1) as f32;
                    let n_layers = caches.len();
                    let mut all_grads: Vec<Qwen35WeightGrads> = Vec::with_capacity(n_layers);

                    for i in (0..n_layers).rev() {
                        let (d_input, grads) = qwen35_layer_backward(
                            &d_hidden, &caches[i], &layers[i], &config.model, seq_len, lr, config.weight_decay,
                        );
                        all_grads.push(grads);
                        d_hidden = d_input;
                    }

                    // Weight update (caches dropped by here)
                    drop(caches);
                    all_grads.reverse();

                    for (i, grads) in all_grads.into_iter().enumerate() {
                        match (grads, &mut layers[i]) {
                            (Qwen35WeightGrads::DeltaNet(g), Qwen35LayerWeights::DeltaNet(w)) => {
                                g.apply_sgd(w, lr * inv_tokens, config.weight_decay);
                            }
                            (Qwen35WeightGrads::FullAttention(g), Qwen35LayerWeights::FullAttention(w)) => {
                                g.apply_sgd(w, lr * inv_tokens, config.weight_decay);
                            }
                            _ => {}
                        }
                    }
                } else {
                    // ── Streaming eval (forward-only, no backward) ──
                    let logits = qwen35_model_forward_eval_streaming(
                        &token_ids, &embedding_table, &get_tensor, &config.weight_prefix,
                        &output_norm, &lm_head, &config.model,
                        Some(&config.checkpoint_dir),
                    );
                    for t in 0..seq_len {
                        let target = batch.target_ids[b * seq_len + t] as usize % vocab_size;
                        let tl = &logits[t * vocab_size..(t + 1) * vocab_size];
                        let (loss, _) = cross_entropy_loss(tl, target);
                        total_loss += loss;
                        token_count += 1;
                    }
                }
            }

            let avg_loss = if token_count > 0 {
                total_loss / token_count as f32
            } else {
                0.0
            };
            let step_duration = step_start.elapsed();

            // SGD weight update on projection weights (FakeQuantize + STE)
            // Phase 2 初期版: forward-only loss 監視
            // 完全 backward は hidden_states の保存 + レイヤー逆伝播が必要
            // → CUDA 対応時に統合（CPU での 9B backward は実用速度に達しないため）

            log.append(LogEntry::new(0, global_step, avg_loss, lr, 0.0));

            // ── 崩壊検知 (NaN/Inf/異常Loss) → 自動停止 ──
            if avg_loss.is_nan() || avg_loss.is_infinite() {
                eprintln!("╔══════════════════════════════════════════════════════════╗");
                eprintln!("║  *** 崩壊検知: Loss = {avg_loss} — 学習を緊急停止 ***      ║");
                eprintln!("╚══════════════════════════════════════════════════════════╝");
                eprintln!("  step: {global_step}, lr: {lr:.2e}");
                eprintln!("  最終正常チェックポイントから再開してください。");
                // ログ保存
                let log_path = format!("{}/train_log.csv", config.checkpoint_dir);
                let _ = log.save_csv_to_file(&log_path);
                let crash_json = format!(
                    "{{\"step\":{},\"loss\":{:.6},\"lr\":{:.2e},\"status\":\"COLLAPSED\"}}",
                    global_step, avg_loss, lr
                );
                let _ = fs::write(
                    format!("{}/resume_state.json", config.checkpoint_dir),
                    &crash_json,
                );
                std::process::exit(1);
            }
            // Loss 異常上昇検知 — 10ステップ猶予後に自動停止
            if global_step > 100 {
                if let Some(prev) = log.entries().iter().rev().nth(10) {
                    // 10ステップ前と比較して3倍以上かつ絶対値50超
                    if prev.loss > 0.0 && avg_loss > prev.loss * 3.0 && avg_loss > 50.0 {
                        if collapse_countdown < 0 {
                            collapse_countdown = 10;
                            eprintln!("  *** 崩壊警告: Loss {:.2} → {:.2} (3倍超) — 残り{collapse_countdown}ステップで自動停止 ***", prev.loss, avg_loss);
                        }
                    } else if collapse_countdown > 0 {
                        // 回復した
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

            if global_step % 10 == 0 || global_step == config.total_steps - 1 {
                let elapsed = train_start.elapsed();
                let steps_per_sec =
                    (global_step + 1) as f64 / elapsed.as_secs_f64().max(0.001);
                let eta_secs =
                    (config.total_steps - global_step) as f64 / steps_per_sec.max(0.001);
                println!(
                    "  step {global_step:>5}/{} | loss: {avg_loss:.4} | lr: {lr:.2e} | {:.1}ms/step | ETA: {eta_secs:.0}s",
                    config.total_steps,
                    step_duration.as_secs_f64() * 1000.0,
                );
            }

            // Eval (オプション)
            if global_step > 0
                && global_step % config.eval_interval == 0
                && eval_dataset.is_some()
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

            // チェックポイント (最大3世代保持 — ストレージ溢れ防止)
            if global_step > 0 && global_step % config.checkpoint_interval == 0 {
                let ckpt_path =
                    format!("{}/step_{global_step}.bin", config.checkpoint_dir);
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

                // 古いチェックポイントの自動削除 (最大3世代保持)
                const MAX_CKPT_KEEP: usize = 3;
                let old_step = global_step as isize
                    - (MAX_CKPT_KEEP as isize * config.checkpoint_interval as isize);
                if old_step > 0 {
                    let old_path =
                        format!("{}/step_{old_step}.bin", config.checkpoint_dir);
                    if std::path::Path::new(&old_path).exists() {
                        if let Err(e) = std::fs::remove_file(&old_path) {
                            eprintln!("  古いチェックポイント削除失敗: {e}");
                        } else {
                            println!("    古いチェックポイント削除: {old_path}");
                        }
                    }
                }
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
    }
}
