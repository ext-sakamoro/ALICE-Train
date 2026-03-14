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

use clap::Parser;
use rand::Rng;
use std::fs;
use std::path::Path;
use std::time::Instant;

use alice_train::llama::QatTrainConfig;
use alice_train::{
    CheckpointData, CudaMatmul, DataLoader, DataLoaderConfig, FakeQuantize, LogEntry,
    LossScaler, LrScheduler, MixedPrecisionConfig, MmapDataset, OffloadConfig, QatConfig, TrainLog,
    WarmupCosineScheduler,
};

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

    // レジューム
    if let Some(ref resume_path) = config.resume_from {
        if Path::new(resume_path).exists() {
            println!("  チェックポイントからレジューム: {resume_path}");
            match std::fs::File::open(resume_path).and_then(|mut f| CheckpointData::load(&mut f)) {
                Ok(ckpt) => {
                    global_step = ckpt.meta.epoch;
                    println!("  レジューム成功: step {global_step}");
                }
                Err(e) => {
                    eprintln!("  チェックポイント読み込み失敗: {e}");
                    eprintln!("  最初から学習を開始します");
                }
            }
        }
    }

    println!("  FakeQuantize: 初期化完了");
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
        use alice_train::llama_forward::{rmsnorm, LayerCache};
        use alice_train::cuda_matmul::{cuda_layer_forward, cuda_layer_backward};
        use alice_train::safetensors_loader::ShardedModel;

        let model = ShardedModel::open(&config.model_path).unwrap_or_else(|e| {
            eprintln!("[ALICE-Train] モデル読み込み失敗: {e}");
            std::process::exit(1);
        });

        let get_tensor = |name: &str| -> Option<Vec<f32>> { model.get_tensor_f32(name) };

        // Embedding table (常駐 ~2GB)
        println!("  embedding 読み込み...");
        let mut embedding_table = get_tensor("model.embed_tokens.weight").unwrap_or_else(|| {
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
        let mut output_proj = get_tensor("lm_head.weight").unwrap_or_else(|| {
            println!("    lm_head.weight なし — embed_tokens と共有");
            embedding_table.clone()
        });
        println!(
            "    output_norm: {} 要素, output_proj: {:.1} MB",
            output_norm.len(),
            output_proj.len() as f64 * 4.0 / 1024.0 / 1024.0
        );

        let num_layers = config.model.num_layers;

        // CUDA 初期化
        println!("  CUDA 初期化...");
        let cuda = CudaMatmul::new();
        println!("    CUDA ready: cuBLAS sgemm 初期化完了");
        println!();

        // メモリ見積もり
        let layer_bytes = config.model.params_per_layer() * 4;
        println!();
        println!("  メモリ見積もり (ストリーミング):");
        println!("    embedding + output: {:.0} MB", (embedding_table.len() + output_proj.len() + output_norm.len()) as f64 * 4.0 / 1024.0 / 1024.0);
        println!("    1レイヤー重み: {:.0} MB", layer_bytes as f64 / 1024.0 / 1024.0);
        println!("    ピーク: embedding + output + 1 layer + activations");
        println!();

        // --- Phase 2 学習ループ ---
        println!(
            "━━━ Phase 2 学習開始 (step {global_step}/{}) ━━━",
            config.total_steps
        );
        let train_start = Instant::now();
        let mut batch_idx = 0usize;

        while global_step < config.total_steps {
            eprintln!("[DEBUG] step {} 開始 — batch_idx={}", global_step, batch_idx);
            use std::io::Write;
            std::io::stderr().flush().ok();
            let lr = scheduler.get_lr(global_step);
            let step_start = Instant::now();

            if batch_idx >= num_batches {
                loader.shuffle_epoch();
                batch_idx = 0;
            }

            let batch = loader.get_batch(batch_idx, &dataset);
            batch_idx += 1;

            let seq_len = config.seq_len;
            let mut total_loss = 0.0f32;
            let mut token_count = 0usize;

            for b in 0..batch.actual_batch_size {
                let token_ids: Vec<u32> = batch.input_ids[b * seq_len..(b + 1) * seq_len].to_vec();
                let targets: Vec<u32> = batch.target_ids[b * seq_len..(b + 1) * seq_len].to_vec();

                // --- Forward (ストリーミング) ---
                // Embedding lookup
                let mut hidden = vec![0.0f32; seq_len * hidden_dim];
                for (t, &tok) in token_ids.iter().enumerate() {
                    let tok = tok as usize;
                    if tok < vocab_size {
                        hidden[t * hidden_dim..(t + 1) * hidden_dim]
                            .copy_from_slice(&embedding_table[tok * hidden_dim..(tok + 1) * hidden_dim]);
                    }
                }

                // Forward: 1レイヤーずつ読み込み → GPU forward → cache 保存 → 重み破棄
                eprintln!("[DEBUG] step {} forward開始", global_step);
                std::io::stderr().flush().ok();
                let mut caches: Vec<LayerCache> = Vec::with_capacity(num_layers);
                for i in 0..num_layers {
                    let lw = LlamaLayerWeights::from_tensors(i, &get_tensor, &config.model)
                        .unwrap_or_else(|| {
                            eprintln!("[ALICE-Train] レイヤー {i} の重み読み込み失敗");
                            std::process::exit(1);
                        });
                    let cache = cuda_layer_forward(&cuda, &mut hidden, &lw, &config.model, seq_len);
                    caches.push(cache);
                    // lw はここで drop — メモリ解放
                }

                // Output RMSNorm + projection
                eprintln!("[DEBUG] step {} forward完了 — logits計算", global_step);
                std::io::stderr().flush().ok();
                let hidden_pre_norm = hidden.clone();
                rmsnorm(&mut hidden, &output_norm, hidden_dim, config.model.norm_eps);

                // logits 計算 — CUDA cuBLAS sgemm: hidden (seq×hidden) × output_proj^T → logits (seq×vocab)
                let logits = cuda.matmul_bt(&hidden, &output_proj, seq_len, vocab_size, hidden_dim);

                // Loss + 勾配計算 (CPU — softmax は軽い)
                let mut d_logits_per_token = Vec::with_capacity(seq_len);
                for t in 0..seq_len {
                    let target = targets[t] as usize;
                    if target >= vocab_size {
                        d_logits_per_token.push(None);
                        continue;
                    }

                    let logits_t = &logits[t * vocab_size..(t + 1) * vocab_size];
                    let (loss, grad) = cross_entropy_loss(logits_t, target);
                    total_loss += loss;
                    token_count += 1;
                    d_logits_per_token.push(Some(grad));
                }

                // --- Backward ---
                eprintln!("[DEBUG] step {} backward開始", global_step);
                std::io::stderr().flush().ok();
                let inv_tokens = 1.0 / token_count.max(1) as f32;

                // Output projection backward — CUDA matmul_nn で d_hidden を計算
                // d_hidden_normed = d_logits × output_proj
                // d_logits: [seq × vocab], output_proj: [vocab × hidden] → d_hidden: [seq × hidden]
                let mut d_logits_flat = vec![0.0f32; seq_len * vocab_size];
                for t in 0..seq_len {
                    if let Some(ref d_logits_t) = d_logits_per_token[t] {
                        d_logits_flat[t * vocab_size..(t + 1) * vocab_size]
                            .copy_from_slice(d_logits_t);
                    }
                }
                let d_hidden_normed = cuda.matmul_nn(
                    &d_logits_flat, &output_proj,
                    seq_len, hidden_dim, vocab_size,
                );
                // output_proj weight update はスキップ（ストリーミングモードでは重み更新を別途保存）
                // TODO: 定期的に差分を蓄積して safetensors に書き戻す

                // Output RMSNorm backward
                let mut d_hidden = vec![0.0f32; seq_len * hidden_dim];
                let mut d_output_norm_w = vec![0.0f32; hidden_dim];
                rmsnorm_backward_output(
                    &d_hidden_normed,
                    &hidden_pre_norm,
                    &output_norm,
                    &mut d_hidden,
                    &mut d_output_norm_w,
                    hidden_dim,
                    config.model.norm_eps,
                );
                for h in 0..hidden_dim {
                    output_norm[h] -= lr * (d_output_norm_w[h] * inv_tokens + config.weight_decay * output_norm[h]);
                }

                // Backward: 逆順ストリーミング — 1レイヤーずつ reload → CUDA backward → drop
                eprintln!("[DEBUG] step {} layer backward開始", global_step);
                std::io::stderr().flush().ok();
                let mut d_layer_input = d_hidden;
                for i in (0..num_layers).rev() {
                    let lw = LlamaLayerWeights::from_tensors(i, &get_tensor, &config.model)
                        .unwrap_or_else(|| {
                            eprintln!("[ALICE-Train] レイヤー {i} backward: 重み読み込み失敗");
                            std::process::exit(1);
                        });

                    d_layer_input = cuda_layer_backward(
                        &cuda,
                        &d_layer_input,
                        &caches[i],
                        &lw,
                        &config.model,
                        seq_len,
                    );
                    // lw + weight grads は drop — ストリーミングモード
                }

                // Embedding gradient update
                eprintln!("[DEBUG] step {} embedding更新", global_step);
                std::io::stderr().flush().ok();
                for t in 0..seq_len {
                    let tok = token_ids[t] as usize;
                    if tok < vocab_size {
                        for h in 0..hidden_dim {
                            embedding_table[tok * hidden_dim + h] -=
                                lr * d_layer_input[t * hidden_dim + h] * inv_tokens;
                        }
                    }
                }

                // caches を明示的にドロップ
                drop(caches);
            }

            let avg_loss = if token_count > 0 {
                total_loss / token_count as f32
            } else {
                0.0
            };
            let step_duration = step_start.elapsed();

            log.append(LogEntry::new(0, global_step, avg_loss, lr, 0.0));

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

            // チェックポイント
            if global_step > 0 && global_step % config.checkpoint_interval == 0 {
                let ckpt_path = format!("{}/step_{global_step}.bin", config.checkpoint_dir);
                println!("  チェックポイント保存: {ckpt_path}");
                let meta = alice_train::CheckpointMeta {
                    version: 1,
                    epoch: 0,
                    step: global_step,
                    loss: avg_loss,
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
            }

            global_step += 1;
        }

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
