//! CPU-only デバッグ版 — 各ステップにタイミングログを出力。
//! メモリ制約: 16GB RAM で 8B モデルを扱うため、各フェーズ後に不要データを解放。

use clap::Parser;
use std::fs;
use std::io::Write as IoWrite;
use std::path::Path;
use std::time::Instant;

use alice_train::llama::QatTrainConfig;
use alice_train::{
    DataLoader, DataLoaderConfig, MmapDataset, WarmupCosineScheduler, LrScheduler,
};

#[derive(Parser, Debug)]
#[command(about = "CPU-only debug: レイヤーごとのタイミングを計測")]
struct Cli {
    #[arg(short, long)]
    config: String,
}

fn log(msg: &str) {
    print!("{msg}");
    std::io::stdout().flush().ok();
}

fn logln(msg: &str) {
    println!("{msg}");
    std::io::stdout().flush().ok();
}

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

    let config_str = fs::read_to_string(&cli.config).unwrap_or_else(|e| {
        eprintln!("[ALICE-Train] 設定ファイル読み込み失敗: {e}");
        std::process::exit(1);
    });
    let config: QatTrainConfig = serde_json::from_str(&config_str).unwrap_or_else(|e| {
        eprintln!("[ALICE-Train] 設定パースエラー: {e}");
        std::process::exit(1);
    });

    if !Path::new(&config.model_path).exists() {
        eprintln!("[ALICE-Train] モデルパスが存在しません: {}", config.model_path);
        std::process::exit(1);
    }

    logln("╔══════════════════════════════════════════════════════════╗");
    logln("║  ALICE-Train DEBUG — CPU-only タイミング計測            ║");
    logln("╚══════════════════════════════════════════════════════════╝");
    logln(&format!("seq_len={}, total_steps={}, layers={}", config.seq_len, config.total_steps, config.model.num_layers));

    let dataset = MmapDataset::open(&config.train_data_path).unwrap_or_else(|e| {
        eprintln!("[ALICE-Train] データ読み込み失敗: {e}");
        std::process::exit(1);
    });
    let dl_config = DataLoaderConfig::new()
        .with_seq_len(config.seq_len)
        .with_batch_size(config.batch_size)
        .with_shuffle(true)
        .with_seed(42);
    let loader = DataLoader::new(&dataset, dl_config);
    logln(&format!("DataLoader: {} トークン", dataset.len()));

    use alice_train::llama::LlamaLayerWeights;
    use alice_train::llama_backward::{layer_backward, rmsnorm_backward_output};
    use alice_train::llama_forward::{layer_forward, rmsnorm, matmul_bt, LayerCache};
    use alice_train::safetensors_loader::ShardedModel;

    let t0 = Instant::now();
    let model = ShardedModel::open(&config.model_path).unwrap();
    logln(&format!("ShardedModel::open: {:.1}s", t0.elapsed().as_secs_f64()));

    let get_tensor = |name: &str| -> Option<Vec<f32>> { model.get_tensor_f32(name) };

    let hidden_dim = config.model.hidden_dim;
    let vocab_size = config.model.vocab_size;
    let num_layers = config.model.num_layers;
    let seq_len = config.seq_len;

    let scheduler = WarmupCosineScheduler::new(config.learning_rate, config.min_lr, config.warmup_steps, config.total_steps);
    let mut output_norm = get_tensor("model.norm.weight").unwrap();

    logln("━━━ 学習ループ開始 ━━━");

    for step in 0..config.total_steps {
        let lr = scheduler.get_lr(step);
        let step_start = Instant::now();

        let batch = loader.get_batch(step % loader.num_batches(), &dataset);
        let token_ids: Vec<u32> = batch.input_ids[..seq_len].to_vec();
        let targets: Vec<u32> = batch.target_ids[..seq_len].to_vec();

        // ── Phase 1: Embedding ──
        log("  [embed] ");
        let t0 = Instant::now();
        let mut hidden = vec![0.0f32; seq_len * hidden_dim];
        {
            let embedding_table = get_tensor("model.embed_tokens.weight").unwrap();
            for (t, &tok) in token_ids.iter().enumerate() {
                let tok = tok as usize;
                if tok < vocab_size {
                    hidden[t * hidden_dim..(t + 1) * hidden_dim]
                        .copy_from_slice(&embedding_table[tok * hidden_dim..(tok + 1) * hidden_dim]);
                }
            }
        }
        logln(&format!("{:.1}ms", t0.elapsed().as_secs_f64() * 1000.0));

        // ── Phase 2: Forward layers ──
        let t0 = Instant::now();
        let mut caches: Vec<LayerCache> = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            log(&format!("  [fwd L{i:>2}] "));
            let lt = Instant::now();
            let lw = LlamaLayerWeights::from_tensors(i, &get_tensor, &config.model).unwrap();
            let load_ms = lt.elapsed().as_millis();

            let ft = Instant::now();
            let cache = layer_forward(&mut hidden, &lw, &config.model, seq_len);
            let fwd_ms = ft.elapsed().as_millis();
            caches.push(cache);
            logln(&format!("load={load_ms}ms fwd={fwd_ms}ms"));
        }
        let fwd_total = t0.elapsed();

        // ── Phase 3: RMSNorm ──
        log("  [rmsnorm] ");
        let t0 = Instant::now();
        let hidden_pre_norm = hidden.clone();
        rmsnorm(&mut hidden, &output_norm, hidden_dim, config.model.norm_eps);
        logln(&format!("{:.1}ms", t0.elapsed().as_secs_f64() * 1000.0));

        // ── Phase 4: Logits (per-token) ──
        log("  [proj load] ");
        let t0 = Instant::now();
        let output_proj = get_tensor("lm_head.weight").unwrap_or_else(|| {
            get_tensor("model.embed_tokens.weight").unwrap()
        });
        logln(&format!("{:.1}ms ({:.0} MB)", t0.elapsed().as_secs_f64() * 1000.0, output_proj.len() as f64 * 4.0 / 1e6));

        log("  [logits] ");
        let t0 = Instant::now();
        let mut total_loss = 0.0f32;
        let mut token_count = 0usize;
        let mut d_logits_sparse: Vec<(usize, Vec<f32>)> = Vec::new();

        for t in 0..seq_len {
            let target = targets[t] as usize;
            if target >= vocab_size { continue; }
            let h_off = t * hidden_dim;
            let mut logits_t = vec![0.0f32; vocab_size];
            matmul_bt(&hidden[h_off..h_off + hidden_dim], &output_proj, &mut logits_t, 1, vocab_size, hidden_dim);
            let (loss, grad) = cross_entropy_loss(&logits_t, target);
            total_loss += loss;
            token_count += 1;
            d_logits_sparse.push((t, grad));
        }
        let logits_time = t0.elapsed();
        let avg_loss = if token_count > 0 { total_loss / token_count as f32 } else { 0.0 };
        logln(&format!("{:.1}ms ({} tokens)", logits_time.as_secs_f64() * 1000.0, token_count));

        // ── Phase 5: Backward d_hidden (cache-friendly loop order) ──
        log("  [bwd d_hid] ");
        let t0 = Instant::now();
        let inv_tokens = 1.0 / token_count.max(1) as f32;
        let mut d_hidden_normed = vec![0.0f32; seq_len * hidden_dim];
        for &(t, ref d_logits_t) in &d_logits_sparse {
            let h_off = t * hidden_dim;
            // Cache-friendly: v 外ループ → output_proj 行を連続読み
            for v in 0..vocab_size {
                let dl = d_logits_t[v];
                if dl.abs() < 1e-10 { continue; }
                let p_off = v * hidden_dim;
                for h in 0..hidden_dim {
                    d_hidden_normed[h_off + h] += dl * output_proj[p_off + h];
                }
            }
        }
        logln(&format!("{:.1}ms", t0.elapsed().as_secs_f64() * 1000.0));

        drop(output_proj);
        drop(d_logits_sparse);

        // ── Phase 6: Backward RMSNorm ──
        log("  [bwd norm] ");
        let t0 = Instant::now();
        let mut d_hidden = vec![0.0f32; seq_len * hidden_dim];
        let mut d_output_norm_w = vec![0.0f32; hidden_dim];
        rmsnorm_backward_output(&d_hidden_normed, &hidden_pre_norm, &output_norm, &mut d_hidden, &mut d_output_norm_w, hidden_dim, config.model.norm_eps);
        for h in 0..hidden_dim {
            output_norm[h] -= lr * (d_output_norm_w[h] * inv_tokens + config.weight_decay * output_norm[h]);
        }
        logln(&format!("{:.1}ms", t0.elapsed().as_secs_f64() * 1000.0));
        drop(d_hidden_normed);

        // ── Phase 7: Backward layers ──
        let t0 = Instant::now();
        let mut d_layer_input = d_hidden;
        for i in (0..num_layers).rev() {
            log(&format!("  [bwd L{i:>2}] "));
            let lt = Instant::now();
            let lw = LlamaLayerWeights::from_tensors(i, &get_tensor, &config.model).unwrap();
            let load_ms = lt.elapsed().as_millis();
            let bt = Instant::now();
            let (d_in, _weight_grads) = layer_backward(&d_layer_input, &caches[i], &lw, &config.model, seq_len);
            let bwd_ms = bt.elapsed().as_millis();
            d_layer_input = d_in;
            logln(&format!("load={load_ms}ms bwd={bwd_ms}ms"));
        }
        let bwd_layers_time = t0.elapsed();
        drop(caches);

        let step_total = step_start.elapsed();
        logln(&format!("\n  ═══ step {step} | loss={avg_loss:.4} | lr={lr:.2e} | total={:.1}s ═══", step_total.as_secs_f64()));
        logln(&format!("    fwd layers: {:.1}s | bwd layers: {:.1}s", fwd_total.as_secs_f64(), bwd_layers_time.as_secs_f64()));
        logln("");
    }

    logln("━━━ デバッグ完了 ━━━");
}
