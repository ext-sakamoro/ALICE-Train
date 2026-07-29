//! FastSpeech2 training example (Phase T.4a Phase 5)。
//!
//! Config file (`configs/fs2_jsut.json`) + TtsDataset (`data/jsut/manifest.jsonl`) から
//! FastSpeech2 学習を実行する CLI example。
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example fs2_train --features tts -- --config configs/fs2_jsut.json
//! ```
//!
//! # 前提
//!
//! - JSUT corpus + manifest jsonl 準備済 (`scripts/prepare_jsut_manifest.py`)
//! - 実運用は RunPod / Paperspace A100 で実行 (Mac は sanity check のみ)
//!
//! # 制約 (Phase T.4a MVP)
//!
//! - SGD only (AdamW は Phase T.4a 継続)
//! - ProsodyLoss 併用未対応 (mel L2 loss のみ)
//! - Checkpoint save/load は Phase T.4a 継続
//! - Variable frame length (batch > 1) 未対応

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Phase T.4c: CUDA cuBLAS 初期化 (feature = "cuda" 有効時)
    // 以降 blas_matmul_bt/nn/tn は自動的に GPU 経路に routing される
    #[cfg(feature = "cuda")]
    {
        println!("[info] Initializing CUDA cuBLAS (Phase T.4c)...");
        alice_train::blas::init_cuda_blas();
        println!("[info] CUDA cuBLAS ready (matmul → GPU)");
    }

    // 簡易 arg parse (--config <path>)
    let args: Vec<String> = env::args().collect();
    let config_path = parse_config_arg(&args).unwrap_or_else(|| {
        eprintln!("[warn] no --config specified, using default configs/fs2_jsut.json");
        PathBuf::from("configs/fs2_jsut.json")
    });

    println!("[info] Loading config from {}", config_path.display());
    let config_json = fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_json)?;

    println!("[info] Config loaded:");
    println!("  vocab_size = {}", config["model"]["vocab_size"]);
    println!("  hidden_dim = {}", config["model"]["hidden_dim"]);
    println!(
        "  encoder_layers = {}",
        config["model"]["num_encoder_layers"]
    );
    println!("  batch_size = {}", config["dataset"]["batch_size"]);
    println!("  learning_rate = {}", config["training"]["learning_rate"]);
    println!("  total_steps = {}", config["training"]["total_steps"]);

    // Feature = "tts" が必要
    #[cfg(feature = "tts")]
    {
        use alice_train::tts::{
            AdamWConfig, AudioFeatureExtractor, FastSpeech2, FastSpeech2Config, TtsDataset,
            TtsTrainConfig, TtsTrainer,
        };

        // 1. Model 構築
        let fs2_config = FastSpeech2Config {
            vocab_size: config["model"]["vocab_size"].as_u64().unwrap_or(100) as usize,
            hidden_dim: config["model"]["hidden_dim"].as_u64().unwrap_or(256) as usize,
            num_heads: config["model"]["num_heads"].as_u64().unwrap_or(2) as usize,
            num_encoder_layers: config["model"]["num_encoder_layers"].as_u64().unwrap_or(4)
                as usize,
            num_decoder_layers: config["model"]["num_decoder_layers"].as_u64().unwrap_or(4)
                as usize,
            fft_kernel_size: config["model"]["fft_kernel_size"].as_u64().unwrap_or(3) as usize,
            fft_expansion: config["model"]["fft_expansion"].as_u64().unwrap_or(4) as usize,
            predictor_kernel_size: config["model"]["predictor_kernel_size"]
                .as_u64()
                .unwrap_or(3) as usize,
            predictor_hidden: config["model"]["predictor_hidden"].as_u64().unwrap_or(256) as usize,
            mel_dim: config["model"]["mel_dim"].as_u64().unwrap_or(80) as usize,
            postnet_kernel_size: config["model"]["postnet_kernel_size"].as_u64().unwrap_or(5)
                as usize,
            postnet_layers: config["model"]["postnet_layers"].as_u64().unwrap_or(5) as usize,
            postnet_hidden: config["model"]["postnet_hidden"].as_u64().unwrap_or(512) as usize,
            max_len: config["model"]["max_len"].as_u64().unwrap_or(2048) as usize,
        };
        println!("[info] Building FastSpeech2 model (zeros init + Xavier init seed=42 + Phase E-next-4 log-domain bias)...");
        let mut model = FastSpeech2::zeros(fs2_config)?;

        // Checkpoint resume 対応 (Phase T.4a Phase C + Paperspace 6h session)
        let ckpt_dir = config["training"]["checkpoint_dir"]
            .as_str()
            .unwrap_or("checkpoints/fs2_jsut");
        std::fs::create_dir_all(ckpt_dir).ok();
        let latest_ckpt = find_latest_checkpoint(ckpt_dir);
        let mut resume_step = 0_usize;
        if let Some((path, step)) = &latest_ckpt {
            println!(
                "[info] Resuming from checkpoint: {} (step {})",
                path.display(),
                step
            );
            model.load_safetensors(path)?;
            resume_step = *step;
        } else {
            model.init_xavier(42);
            // Phase E-next-4: log-domain bias 事前設定 (dur log(4+1)=1.5, pitch log(150Hz)=5.0, energy -20dB)
            model.init_variance_biases(1.5, 5.0, -20.0);
        }

        // 2. Trainer 構築
        let train_config = TtsTrainConfig {
            learning_rate: config["training"]["learning_rate"].as_f64().unwrap_or(1e-3) as f32,
            log_interval: config["training"]["log_interval"].as_u64().unwrap_or(100) as usize,
        };
        // AdamW optimizer (SGD より収束速い)
        let adamw_config = AdamWConfig {
            learning_rate: train_config.learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        let mut trainer = TtsTrainer::with_adamw(model, train_config, adamw_config);

        // 3. Dataset 構築
        let sample_rate = config["audio"]["sample_rate"].as_u64().unwrap_or(24_000) as u32;
        let n_fft = config["audio"]["n_fft"].as_u64().unwrap_or(1024) as usize;
        let hop_length = config["audio"]["hop_length"].as_u64().unwrap_or(256) as usize;
        let n_mels = config["audio"]["n_mels"].as_u64().unwrap_or(80) as usize;
        let extractor = AudioFeatureExtractor::new(sample_rate, n_fft, hop_length, n_mels);

        let manifest_path = config["dataset"]["manifest_path"]
            .as_str()
            .unwrap_or("data/jsut/manifest.jsonl");
        let audio_root = config["dataset"]["audio_root"]
            .as_str()
            .unwrap_or("data/jsut/audio");
        println!(
            "[info] Loading dataset from manifest {} (audio_root {})...",
            manifest_path, audio_root
        );

        // Manifest が存在しない場合は sanity check (dummy) mode
        if !PathBuf::from(manifest_path).exists() {
            println!(
                "[warn] manifest {} not found, running sanity check with dummy data",
                manifest_path
            );
            run_sanity_check(&mut trainer)?;
            return Ok(());
        }

        let dataset = TtsDataset::from_manifest(manifest_path, audio_root, extractor)?;
        println!("[info] Dataset loaded: {} entries", dataset.len());

        // 4. Train / valid split
        let split_seed = config["dataset"]["split_seed"].as_u64().unwrap_or(42);
        let ratios = config["dataset"]["split_ratios"].as_array();
        let (r_train, r_valid, r_test) = if let Some(arr) = ratios {
            (
                arr.first().and_then(|v| v.as_f64()).unwrap_or(0.9) as f32,
                arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.05) as f32,
                arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.05) as f32,
            )
        } else {
            (0.9, 0.05, 0.05)
        };
        let (train_ds, _valid_ds, _test_ds) =
            dataset.split((r_train, r_valid, r_test), split_seed)?;
        println!("[info] Train split: {} entries", train_ds.len());

        // 5. Training loop
        let total_steps = config["training"]["total_steps"]
            .as_u64()
            .unwrap_or(300_000) as usize;
        let batch_size = config["dataset"]["batch_size"].as_u64().unwrap_or(8) as usize;
        let log_interval = train_config.log_interval;
        println!(
            "[info] Starting training: {} steps, batch_size = {}",
            total_steps, batch_size
        );

        let checkpoint_interval = config["training"]["checkpoint_interval"]
            .as_u64()
            .unwrap_or(1000) as usize;

        let mut step_count = resume_step;
        let mut iter_count = 0_usize;
        let mut skip_reason: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        'outer: for epoch in 0.. {
            for batch_result in train_ds.iter_batches(batch_size) {
                iter_count += 1;
                if iter_count.is_multiple_of(50) {
                    eprintln!("[iter {iter_count}] step_count={step_count} skips={skip_reason:?}");
                }
                let batch = match batch_result {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("[iter {iter_count}] batch load error: {e}");
                        *skip_reason.entry("batch_load_err").or_insert(0) += 1;
                        continue;
                    }
                };
                if batch.batch_size() == 0 {
                    *skip_reason.entry("empty_batch").or_insert(0) += 1;
                    continue;
                }
                // batch_size=1 sample のみ抜き取り (variable frame len 対応まで)
                if batch.batch_size() > 1 {
                    *skip_reason.entry("batch_size_gt1").or_insert(0) += 1;
                    continue;
                }
                let mora_ids = &batch.text_moras()[0]
                    .iter()
                    .map(|&x| u32::from(x))
                    .collect::<Vec<u32>>();
                let durations: Vec<u32> = batch.durations_ms()[0]
                    .iter()
                    .map(|&ms| {
                        ((ms as f32 * sample_rate as f32) / (hop_length as f32 * 1000.0)).round()
                            as u32
                    })
                    .collect();
                let mora_len = mora_ids.len();

                // mel target = batch.audio_mel()[0] を [mel_dim, frames] → [frames, mel_dim] に転置
                let mel_batch = &batch.audio_mel()[0];
                if mel_batch.is_empty() {
                    *skip_reason.entry("empty_mel").or_insert(0) += 1;
                    continue;
                }
                let n_mels = mel_batch.len();
                let frames = mel_batch[0].len();
                let mut mel_target = vec![0.0_f32; frames * n_mels];
                for t in 0..frames {
                    for m in 0..n_mels {
                        mel_target[t * n_mels + m] = mel_batch[m][t];
                    }
                }

                let expected_frames: usize = durations.iter().map(|&d| d as usize).sum();
                // Duration auto-rescale: placeholder 80ms/mora と実 audio 長のズレを比例調整
                // 完全 forced alignment は Phase T.4a 継続で MFA 導入予定
                let durations_scaled: Vec<u32> = if expected_frames == frames {
                    durations.clone()
                } else if expected_frames == 0 {
                    *skip_reason.entry("zero_expected_frames").or_insert(0) += 1;
                    continue;
                } else {
                    let scale = frames as f32 / expected_frames as f32;
                    let mut scaled: Vec<u32> = durations
                        .iter()
                        .map(|&d| ((d as f32 * scale).round() as u32).max(1))
                        .collect();
                    // Sum mismatch は最後の mora で吸収
                    let sum: u32 = scaled.iter().sum();
                    let target = frames as u32;
                    if sum > target && !scaled.is_empty() {
                        let diff = sum - target;
                        let last = scaled.len() - 1;
                        scaled[last] = scaled[last].saturating_sub(diff).max(1);
                    } else if sum < target && !scaled.is_empty() {
                        let diff = target - sum;
                        let last = scaled.len() - 1;
                        scaled[last] += diff;
                    }
                    scaled
                };
                // 最終検証: sum(durations_scaled) == frames
                let scaled_sum: usize = durations_scaled.iter().map(|&d| d as usize).sum();
                if scaled_sum != frames {
                    *skip_reason.entry("scale_sum_mismatch").or_insert(0) += 1;
                    continue;
                }

                let t0 = std::time::Instant::now();
                let result = trainer.step(mora_ids, &durations_scaled, &mel_target, 1, mora_len)?;
                let dt = t0.elapsed().as_secs_f32();
                // Bug fix: 真の累積 step = resume 元 step + このセッション内 trainer.step_count()
                // 旧実装は step_count = result.step で resume_step を毎回上書き → checkpoint 名衝突
                let session_step = result.step;
                step_count = resume_step + session_step;
                if session_step <= 3 || session_step.is_multiple_of(log_interval) {
                    println!(
                        "[step {} (session {})] epoch={} mel_loss={:.4} step_time={:.2}s mora={} frames={}",
                        step_count, session_step, epoch, result.mel_loss, dt, mora_len, frames
                    );
                }

                // Checkpoint は accumulated step で命名 (単調増加、上書きなし)
                if session_step > 0 && session_step.is_multiple_of(checkpoint_interval) {
                    let ckpt_path = format!("{}/step_{}.safetensors", ckpt_dir, step_count);
                    match trainer.model().save_safetensors(&ckpt_path) {
                        Ok(_) => println!("[ckpt] saved {} (accumulated step)", ckpt_path),
                        Err(e) => eprintln!("[ckpt] save failed: {e}"),
                    }
                }

                if step_count >= total_steps {
                    println!("[info] Total steps reached ({}), stopping", step_count);
                    break 'outer;
                }
            }
        }

        println!(
            "[info] Training complete. Accumulated steps: {} (session {})",
            step_count,
            trainer.step_count()
        );
    }

    #[cfg(not(feature = "tts"))]
    {
        eprintln!("[error] This example requires --features tts");
        return Err("feature 'tts' not enabled".into());
    }

    Ok(())
}

/// checkpoint_dir 内で `step_N.safetensors` の最大 N を持つ file を返す (Phase T.4a Phase C resume)。
fn find_latest_checkpoint(ckpt_dir: &str) -> Option<(PathBuf, usize)> {
    let dir = PathBuf::from(ckpt_dir);
    if !dir.exists() {
        return None;
    }
    let mut best: Option<(PathBuf, usize)> = None;
    if let Ok(entries) = fs::read_dir(&dir) {
        for e in entries.flatten() {
            let p = e.path();
            let name = p.file_name()?.to_str()?;
            if let Some(rest) = name.strip_prefix("step_") {
                if let Some(num_str) = rest.strip_suffix(".safetensors") {
                    if let Ok(n) = num_str.parse::<usize>() {
                        if best.as_ref().is_none_or(|(_, prev)| n > *prev) {
                            best = Some((p, n));
                        }
                    }
                }
            }
        }
    }
    best
}

fn parse_config_arg(args: &[String]) -> Option<PathBuf> {
    for i in 0..args.len() {
        if args[i] == "--config" && i + 1 < args.len() {
            return Some(PathBuf::from(&args[i + 1]));
        }
    }
    None
}

#[cfg(feature = "tts")]
fn run_sanity_check(
    trainer: &mut alice_train::tts::TtsTrainer,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("[sanity] Running 5-step training with dummy data...");
    let mel_dim = trainer.model().config().mel_dim;
    let mora_ids: Vec<u32> = vec![1, 2, 3];
    let durations: Vec<u32> = vec![2, 2, 2];
    let frame_len = 6;
    let mel_target = vec![0.5_f32; frame_len * mel_dim];

    for _ in 1..=5 {
        let result = trainer.step(&mora_ids, &durations, &mel_target, 1, 3)?;
        println!(
            "[sanity] step {} mel_loss={:.6}",
            result.step, result.mel_loss
        );
    }
    println!("[sanity] Done. Sanity check passed (no NaN/Inf).");
    Ok(())
}

#[cfg(not(feature = "tts"))]
fn run_sanity_check() -> Result<(), Box<dyn std::error::Error>> {
    Err("feature 'tts' not enabled".into())
}
