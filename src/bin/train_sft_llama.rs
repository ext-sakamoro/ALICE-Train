//! Llama 系アーキテクチャの LoRA SFT 学習 CLI。
//!
//! base 重みを凍結し、[`alice_train::lora`] の低ランクアダプタだけを AdamW で学習する。
//! 対象は `LlamaForCausalLM` 互換 (MiniCPM5-2B / Llama-3 系)。
//!
//! # なぜ `model_forward` を使わないか
//!
//! [`alice_train::llama_forward::model_forward`] は出力 RMSNorm を in-place で掛けた後の
//! hidden しか返さないが、norm の backward には **norm 前の hidden** が要る。
//! そのため forward をここで展開し、pre-norm hidden を保持する。
//!
//! # メモリ
//!
//! merged 重み (`W + s·B·A`) を全層分持つと base と同量 (2.5B で約 10 GB) 増えるため、
//! **1 層分の scratch を使い回す**。forward と backward で各 1 回ずつ merge する。
//!
//! # 使い方
//!
//! ```bash
//! cargo run --release --features qat-cuda --bin train-sft-llama -- \
//!     --model-dir models/MiniCPM5-2B \
//!     --data data/lol_pairs/train.jsonl \
//!     --out checkpoints/lol_sft
//! ```

#[cfg(feature = "cuda")]
use alice_train::cuda_matmul::{CudaLayerWorkspace, VramLayerWeights};
use alice_train::llama::{LlamaConfig, LlamaLayerWeights};
use alice_train::llama_backward::{layer_backward, rmsnorm_backward};
use alice_train::llama_forward::{layer_forward, matmul, matmul_bt, rmsnorm};
use alice_train::lora::{LoraConfig, LoraLayer, LoraLayerGrads};
use alice_train::safetensors_loader::ShardedModel;
use alice_train::tokenizer::BpeTokenizer;
use clap::Parser;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// LoRA SFT CLI 引数。
#[derive(Parser, Debug)]
#[command(about = "LoRA SFT for Llama-compatible models (base weights frozen)")]
struct Args {
    /// HF 形式のモデルディレクトリ (config.json / tokenizer.json / *.safetensors)
    #[arg(long)]
    model_dir: PathBuf,
    /// 学習データ JSONL (`caption_en` と `lol` を持つ行)
    #[arg(long)]
    data: PathBuf,
    /// チェックポイント出力ディレクトリ
    #[arg(long)]
    out: PathBuf,
    /// system prompt の text file (llm_bench.rs から機械抽出したもの、逐語一致が必須)
    #[arg(long)]
    system_prompt_file: PathBuf,
    /// LoRA rank
    #[arg(long, default_value_t = 16)]
    rank: usize,
    /// LoRA alpha
    #[arg(long, default_value_t = 32.0)]
    alpha: f32,
    /// 学習率
    #[arg(long, default_value_t = 1e-4)]
    lr: f32,
    /// weight decay (AdamW)
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f32,
    /// epoch 数
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    /// 勾配累積ステップ数 (実効 batch size)
    #[arg(long, default_value_t = 8)]
    grad_accum: usize,
    /// 最大系列長 (超過サンプルは捨てる)
    #[arg(long, default_value_t = 512)]
    max_seq: usize,
    /// 先頭 N 件だけ使う (smoke 用、0 で全件)
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// 何ステップごとにログを出すか
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    /// 何ステップごとに checkpoint を書くか
    #[arg(long, default_value_t = 200)]
    save_every: usize,
    /// 既存 checkpoint から再開する
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// CPU path と CUDA path を層ごとに突合して終了する (GPU 必須、学習しない)
    #[arg(long, default_value_t = false)]
    verify_cuda_parity: bool,
    /// CUDA が使える環境でも layer は CPU path で回す
    ///
    /// CUDA layer path は 2026-09-25 時点で backward の勾配が CPU path と符号レベルで
    /// 食い違っており (`--verify-cuda-parity` で再現)、CPU path の方だけが
    /// 数値微分 oracle (`tests/layer_backward_oracle.rs`) を通っている。
    #[arg(long, default_value_t = false)]
    force_cpu_layers: bool,
}

// system prompt は **埋め込まない**。
//
// baseline (grammar 7/20 / think 9/20) は `alice-lol/examples/llm_bench.rs` の
// `SYSTEM_PROMPT` (8 行 1,306 文字) で測られているため、SFT 側が 1 文字でも違うと
// 測定差分を SFT の効果に帰属できない。ここに手でコピーすると law の二重管理になる
// (canonical source rule) ので、`--system-prompt-file` で外から与える。
//
// file は llm_bench.rs から機械抽出して作る (手写し禁止):
//   python3 scripts/extract_system_prompt.py \
//       ../ALICE-LOL/alice-lol/examples/llm_bench.rs data/lol_system_prompt.txt

/// 2 実装の tensor を突合し、最大絶対誤差と最大相対誤差を返す。
///
/// 同一概念を 2 経路で実装したら突合する (port parity oracle)。
fn tensor_diff(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    assert_eq!(a.len(), b.len(), "長さが違う: {} vs {}", a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let abs = (x - y).abs();
        if abs > max_abs {
            max_abs = abs;
            at = i;
        }
        let denom = x.abs().max(y.abs()).max(1e-6);
        max_rel = max_rel.max(abs / denom);
    }
    (max_abs, max_rel, at)
}

/// allclose 判定 (`atol + rtol`)。相対だけで見るとゼロ近傍要素で誤検知する
/// (最初この設計で forward が「全層乖離」に見え、修正後も TF32 の雑音で
/// `d_input rel 5.06e-3 / max_abs 1e-7` を「乖離」と報告していた)。
fn is_diverged(a: &[f32], b: &[f32], abs: f32, rel: f32, tol: f32) -> bool {
    let scale = a.iter().chain(b.iter()).fold(0.0f32, |m, v| m.max(v.abs()));
    let atol = 1e-4 * scale.max(1e-3);
    abs > atol && rel > tol
}

/// 突合結果を 1 行で出す。閾値超えなら `!` を付ける。
fn report_diff(label: &str, a: &[f32], b: &[f32], tol: f32) -> bool {
    let (abs, rel, at) = tensor_diff(a, b);
    let bad = is_diverged(a, b, abs, rel, tol);
    println!(
        "    {mark} {label:<14} max_abs {abs:>12.3e}  max_rel {rel:>10.3e}  at {at}  (n={n})",
        mark = if bad { "!!" } else { "ok" },
        n = a.len()
    );
    bad
}

/// 学習サンプル 1 件。
struct Sample {
    /// prompt + completion の token 列。
    tokens: Vec<u32>,
    /// completion (= LOL 式) の開始位置。ここより前の target には loss を乗せない。
    completion_start: usize,
}

/// 数値安定な cross-entropy loss と logits 勾配。
///
/// `src/bin/train_qat_qwen35.rs` の同名関数と同じ定式 (softmax - onehot)。
fn cross_entropy_loss(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    // NaN / Inf を **絶対に黙って通さない**。
    //
    // 以前は `probs[target].max(1e-10)` で下限を切っていたが、Rust の `f32::max` は
    // NaN を受けると他方を返すため、logits が NaN でも loss が -ln(1e-10) = 23.0259 と
    // いう「もっともらしい数字」に化けて、CUDA path の NaN を丸ごと隠していた。
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "logits に NaN / Inf がある (target={target}, len={}) — forward が壊れている",
        logits.len()
    );
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    assert!(sum.is_finite() && sum > 0.0, "softmax の分母が不正: {sum}");
    let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();
    // ここまで finite が保証されているので、下限は f32 の真の underflow 用。
    // (logit 差が -104 を下回ると exp が subnormal → 0 になるのは正当な underflow)
    const P_FLOOR: f32 = 1e-30;
    let p = probs[target];
    debug_assert!(p.is_finite(), "probs に NaN が残っている");
    let loss = -p.max(P_FLOOR).ln();
    let mut grad = probs;
    grad[target] -= 1.0;
    (loss, grad)
}

/// HF `config.json` を [`LlamaConfig`] に写す。
fn load_config(model_dir: &Path) -> std::io::Result<LlamaConfig> {
    let text = fs::read_to_string(model_dir.join("config.json"))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let get_usize = |k: &str| -> usize {
        v.get(k)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_else(|| panic!("config.json に {k} がない")) as usize
    };
    let hidden_dim = get_usize("hidden_size");
    let num_heads = get_usize("num_attention_heads");
    // head_dim は明示されていれば優先 (MiniCPM5 は hidden/heads と一致しないモデルもある)
    let head_dim = v
        .get("head_dim")
        .and_then(serde_json::Value::as_u64)
        .map_or(hidden_dim / num_heads, |x| x as usize);
    let attention_bias = v
        .get("attention_bias")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    Ok(LlamaConfig {
        vocab_size: get_usize("vocab_size"),
        hidden_dim,
        intermediate_dim: get_usize("intermediate_size"),
        num_heads,
        num_kv_heads: get_usize("num_key_value_heads"),
        num_layers: get_usize("num_hidden_layers"),
        max_seq_len: get_usize("max_position_embeddings"),
        head_dim,
        #[allow(clippy::cast_possible_truncation)]
        rope_theta: v
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .expect("config.json に rope_theta がない") as f32,
        #[allow(clippy::cast_possible_truncation)]
        norm_eps: v
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .expect("config.json に rms_norm_eps がない") as f32,
        attention_bias,
    })
}

/// ChatML 組み立て。`llm_bench.rs:405` の `chatml()` と同じ文字列になるよう token 単位で構成する。
///
/// `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{lol}<|im_end|>`
///
/// BOS は付けない (`llm_bench` が付けていないため。ここを変えると baseline と比較できない)。
fn build_sample(
    tok: &BpeTokenizer,
    im_start: u32,
    im_end: u32,
    system: &str,
    caption: &str,
    lol: &str,
) -> Sample {
    let mut t = Vec::with_capacity(256);
    t.push(im_start);
    t.extend(tok.encode(&format!("system\n{system}")));
    t.push(im_end);
    t.extend(tok.encode("\n"));
    t.push(im_start);
    t.extend(tok.encode(&format!("user\n{caption}")));
    t.push(im_end);
    t.extend(tok.encode("\n"));
    t.push(im_start);
    t.extend(tok.encode("assistant\n"));
    let completion_start = t.len();
    t.extend(tok.encode(lol));
    t.push(im_end);
    Sample {
        tokens: t,
        completion_start,
    }
}

/// datagen の JSONL を読み、ChatML 化した学習サンプルにする。
fn load_samples(
    path: &Path,
    tok: &BpeTokenizer,
    im_start: u32,
    im_end: u32,
    system: &str,
    max_seq: usize,
    limit: usize,
) -> std::io::Result<(Vec<Sample>, usize)> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    let mut skipped_too_long = 0usize;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  JSONL 行を skip (parse 失敗): {e}");
                continue;
            }
        };
        let (Some(caption), Some(lol)) = (
            v.get("caption_en").and_then(serde_json::Value::as_str),
            v.get("lol").and_then(serde_json::Value::as_str),
        ) else {
            eprintln!("  JSONL 行を skip (caption_en / lol が無い)");
            continue;
        };
        let s = build_sample(tok, im_start, im_end, system, caption, lol);
        if s.tokens.len() > max_seq {
            skipped_too_long += 1;
            continue;
        }
        out.push(s);
        if limit > 0 && out.len() >= limit {
            break;
        }
    }
    Ok((out, skipped_too_long))
}

/// LoRA パラメータ 1 本分の AdamW 状態。
struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamState {
    fn zeros(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    /// AdamW 更新 (bias correction 込み)。
    fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32, wd: f32, t: i32, scale: f32) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        let bc1 = 1.0 - B1.powi(t);
        let bc2 = 1.0 - B2.powi(t);
        for (((p, g), m), v) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(self.m.iter_mut())
            .zip(self.v.iter_mut())
        {
            let g = g * scale;
            *m = B1 * *m + (1.0 - B1) * g;
            *v = B2 * *v + (1.0 - B2) * g * g;
            let m_hat = *m / bc1;
            let v_hat = *v / bc2;
            *p -= lr * (m_hat / (v_hat.sqrt() + EPS) + wd * *p);
        }
    }
}

/// 1 レイヤー分の AdamW 状態 (7 projection × A/B)。
struct LayerAdam {
    states: Vec<AdamState>,
}

impl LayerAdam {
    fn zeros(layer: &LoraLayer) -> Self {
        let mut states = Vec::with_capacity(14);
        for ad in [
            &layer.q,
            &layer.k,
            &layer.v,
            &layer.o,
            &layer.gate,
            &layer.up,
            &layer.down,
        ] {
            states.push(AdamState::zeros(ad.a.len()));
            states.push(AdamState::zeros(ad.b.len()));
        }
        Self { states }
    }
}

/// 7 projection を固定順で列挙する (AdamW 状態との対応を 1 箇所に閉じる)。
macro_rules! for_each_adapter {
    ($layer:expr, $grads:expr, $adam:expr, |$ad:ident, $ga:ident, $gb:ident, $sa:ident, $sb:ident| $body:block) => {{
        let l = $layer;
        let g = $grads;
        let a = $adam;
        let mut idx = 0usize;
        let pairs: [(&mut _, &_); 7] = [
            (&mut l.q, &g.q),
            (&mut l.k, &g.k),
            (&mut l.v, &g.v),
            (&mut l.o, &g.o),
            (&mut l.gate, &g.gate),
            (&mut l.up, &g.up),
            (&mut l.down, &g.down),
        ];
        for (ad_ref, gr) in pairs {
            let (left, right) = a.states.split_at_mut(idx + 1);
            let $sa = &mut left[idx];
            let $sb = &mut right[0];
            let $ad = ad_ref;
            let $ga = &gr.d_a;
            let $gb = &gr.d_b;
            $body
            idx += 2;
        }
    }};
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let t_start = Instant::now();

    println!("[sft] model_dir = {}", args.model_dir.display());

    // BLAS dispatch を初期化する。これを呼ばないと `blas.rs` の CUDA 経路が
    // 使われず、行列積が tiled CPU に落ちる (llama path は `blas.rs` 経由なので
    // 効果は forward / backward の両方に効く)。
    #[cfg(feature = "cuda")]
    {
        alice_train::blas::init_cuda_blas();
        println!(
            "[sft] CUDA BLAS: {}",
            if alice_train::blas::cuda_blas_available() {
                "有効 (cuBLAS TF32)"
            } else {
                "初期化失敗 → CPU fallback"
            }
        );
    }
    #[cfg(not(feature = "cuda"))]
    println!("[sft] CUDA BLAS: 無効 (feature off) → CPU 経路");
    let config = load_config(&args.model_dir)?;
    println!(
        "[sft] config: layers {} hidden {} inter {} heads {}/{} head_dim {} vocab {} rope_theta {} eps {}",
        config.num_layers,
        config.hidden_dim,
        config.intermediate_dim,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        config.vocab_size,
        config.rope_theta,
        config.norm_eps
    );

    // ── tokenizer + 特殊 token ──
    let tok = BpeTokenizer::from_file(args.model_dir.join("tokenizer.json"))?;
    let im_start = tok
        .token_id("<|im_start|>")
        .expect("tokenizer に <|im_start|> がない (ChatML 非対応の vocab)");
    let im_end = tok
        .token_id("<|im_end|>")
        .expect("tokenizer に <|im_end|> がない (ChatML 非対応の vocab)");
    println!(
        "[sft] im_start = {im_start}, im_end = {im_end}, vocab = {}",
        tok.vocab_size()
    );

    // ── system prompt (逐語一致が要るので埋め込まず file から) ──
    let raw_system = fs::read_to_string(&args.system_prompt_file)?;
    let system_prompt = raw_system.strip_suffix('\n').unwrap_or(&raw_system);
    println!(
        "[sft] system prompt: {} chars, {} lines ({})",
        system_prompt.chars().count(),
        system_prompt.lines().count(),
        args.system_prompt_file.display()
    );

    // ── データ ──
    let (samples, skipped) = load_samples(
        &args.data,
        &tok,
        im_start,
        im_end,
        system_prompt,
        args.max_seq,
        args.limit,
    )?;
    assert!(!samples.is_empty(), "学習サンプルが 0 件");
    let total_completion_tokens: usize = samples
        .iter()
        .map(|s| s.tokens.len().saturating_sub(s.completion_start))
        .sum();
    println!(
        "[sft] samples {} (max_seq 超過で skip {}), completion token 合計 {}",
        samples.len(),
        skipped,
        total_completion_tokens
    );
    // 書式を目視できるよう先頭 1 件を decode して出す (log-first)
    let head = &samples[0];
    println!(
        "[sft] sample[0] prompt = {:?}",
        tok.decode(&head.tokens[..head.completion_start])
    );
    println!(
        "[sft] sample[0] completion = {:?}",
        tok.decode(&head.tokens[head.completion_start..])
    );

    // ── base 重み (凍結) ──
    println!("[sft] loading safetensors ...");
    let model = ShardedModel::open(&args.model_dir)?;
    let get = |name: &str| model.get_tensor_f32(name);
    let embedding = get("model.embed_tokens.weight").expect("embed_tokens が無い");
    let output_norm = get("model.norm.weight").expect("model.norm.weight が無い");
    // tie_word_embeddings=false のモデルは lm_head を別に持つ
    let output_proj = get("lm_head.weight").unwrap_or_else(|| {
        println!("[sft] lm_head.weight が無いので embed_tokens を tied として使う");
        embedding.clone()
    });
    let mut base_layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let w = LlamaLayerWeights::from_tensors(i, &get, &config)
            .unwrap_or_else(|| panic!("layer {i} の重みが揃っていない"));
        base_layers.push(w);
    }
    println!(
        "[sft] base 重み load 完了 ({:.1}s), 凍結 param {:.2} B",
        t_start.elapsed().as_secs_f32(),
        config.total_params() as f64 / 1e9
    );

    // ── LoRA ──
    let lora_cfg = LoraConfig::try_new(args.rank, args.alpha)
        .unwrap_or_else(|e| panic!("LoRA config が不正: {e}"));
    let mut lora: Vec<LoraLayer> = (0..config.num_layers)
        .map(|i| LoraLayer::new(&config, lora_cfg, i))
        .collect();
    let mut lora_grads: Vec<LoraLayerGrads> = lora.iter().map(LoraLayerGrads::zeros).collect();
    let mut adam: Vec<LayerAdam> = lora.iter().map(LayerAdam::zeros).collect();
    let trainable: usize = lora.iter().map(LoraLayer::num_params).sum();
    println!(
        "[sft] LoRA rank {} alpha {} scale {:.3} → 学習対象 {:.2} M param ({:.2} % of base)",
        lora_cfg.rank(),
        lora_cfg.alpha(),
        lora_cfg.scale(),
        trainable as f64 / 1e6,
        trainable as f64 / config.total_params() as f64 * 100.0
    );

    fs::create_dir_all(&args.out)?;
    let ckpt_path = args.out.join("lora_adapter.bin");
    let mut global_step: u64 = 0;
    if args.resume && ckpt_path.exists() {
        global_step = load_adapter(&ckpt_path, &mut lora)?;
        println!("[sft] resume: step {global_step} から再開");
    }

    // merge 用 scratch (1 層分だけ確保して使い回す = 全層 merge より約 10 GB 節約)
    let mut merged = base_layers[0].clone();

    // ── CUDA 常駐状態 ──
    //
    // `cuda_layer_forward_ws_vram` / `cuda_layer_backward_ws_vram` は llama layer
    // 全体 (attention + FFN) を GPU で回し、projection 重みを VRAM 常駐にして
    // H2D を 1/14 に削る (qwen35 path 由来、L5/L6/L15/L16)。
    //
    // LoRA では merged 重みが **optimizer step ごと** にしか変わらないので、
    // upload も step ごとで済む (grad_accum 分だけ H2D を共有できる)。
    // 42 層 × 47.2M float = 約 7.9 GB VRAM (A6000 48 GB に収まる)。
    #[cfg(feature = "cuda")]
    let mut cuda_ws = CudaLayerWorkspace::new(&config, args.max_seq);
    #[cfg(feature = "cuda")]
    let mut vram_layers: Vec<VramLayerWeights> = Vec::new();
    #[cfg(feature = "cuda")]
    let use_cuda_layers = alice_train::blas::cuda_blas_available() && !args.force_cpu_layers;
    #[cfg(not(feature = "cuda"))]
    let use_cuda_layers = false;
    #[cfg(not(feature = "cuda"))]
    let _ = args.force_cpu_layers;

    // merged 重みを全層 VRAM に載せ直す (起動時 + optimizer step ごと)
    #[cfg(feature = "cuda")]
    macro_rules! refresh_vram {
        () => {{
            if use_cuda_layers {
                let cuda_mtx = alice_train::blas::CUDA_MATMUL
                    .get()
                    .expect("cuda_blas_available() が true なのに未初期化");
                let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
                vram_layers.clear();
                for l in 0..config.num_layers {
                    lora[l].merge_into(&base_layers[l], &mut merged);
                    vram_layers.push(VramLayerWeights::upload(&cuda, &merged));
                }
            }
        }};
    }
    #[cfg(feature = "cuda")]
    {
        let t = Instant::now();
        refresh_vram!();
        if use_cuda_layers {
            println!(
                "[sft] VRAM 常駐: {} layer, {:.2} GB, upload {:.1}s",
                vram_layers.len(),
                vram_layers
                    .iter()
                    .map(VramLayerWeights::vram_bytes)
                    .sum::<usize>() as f64
                    / 1e9,
                t.elapsed().as_secs_f32()
            );
        }
    }
    println!(
        "[sft] layer 実行経路: {}",
        if use_cuda_layers {
            "CUDA (attention + FFN を GPU、重みは VRAM 常駐)"
        } else {
            "CPU (blas 経由)"
        }
    );

    // ── CPU ↔ CUDA parity 突合 (--verify-cuda-parity) ──
    //
    // 同一入力から両 path を走らせ、最初に乖離する層 / tensor を特定する。
    // CPU 版 layer_forward は blas 経由で CUDA_MATMUL の mutex を取るので、
    // CPU 呼び出し中は lock を持たない (持つと deadlock)。
    #[cfg(feature = "cuda")]
    if args.verify_cuda_parity {
        assert!(use_cuda_layers, "CUDA が初期化されていない (GPU 必須)");
        let sample = &samples[0];
        let seq_len = sample.tokens.len();
        let hd = config.hidden_dim;
        println!("\n[parity] sample[0] seq_len {seq_len}, 層ごとに同一入力で突合する");
        // tol: TF32 の cuBLAS は f32 比で 1e-3 級の相対誤差が出うる
        let tol = 5e-3f32;

        let mut hidden = vec![0.0f32; seq_len * hd];
        for (t, &tid) in sample.tokens.iter().enumerate() {
            let tid = tid as usize;
            hidden[t * hd..(t + 1) * hd].copy_from_slice(&embedding[tid * hd..(tid + 1) * hd]);
        }

        let mut caches_cpu = Vec::with_capacity(config.num_layers);
        let mut first_bad_fwd: Option<usize> = None;
        for l in 0..config.num_layers {
            lora[l].merge_into(&base_layers[l], &mut merged);
            let mut h_cpu = hidden.clone();
            let c_cpu = layer_forward(&mut h_cpu, &merged, &config, seq_len);
            let mut h_gpu = hidden.clone();
            let c_gpu = {
                let cuda_mtx = alice_train::blas::CUDA_MATMUL.get().expect("CUDA 未初期化");
                let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
                alice_train::cuda_matmul::cuda_layer_forward_ws_vram(
                    &cuda,
                    &mut h_gpu,
                    &base_layers[l],
                    &vram_layers[l],
                    &config,
                    seq_len,
                    &mut cuda_ws,
                )
            };
            let mut bad = false;
            if l < 3 || first_bad_fwd.is_some() || l + 1 == config.num_layers {
                println!("  [fwd] layer {l}");
                bad |= report_diff("hidden", &h_cpu, &h_gpu, tol);
                bad |= report_diff("normed_attn", &c_cpu.normed_attn, &c_gpu.normed_attn, tol);
                bad |= report_diff("q", &c_cpu.q, &c_gpu.q, tol);
                bad |= report_diff("k", &c_cpu.k, &c_gpu.k, tol);
                bad |= report_diff("v", &c_cpu.v, &c_gpu.v, tol);
                bad |= report_diff(
                    "attn_weights",
                    &c_cpu.attn_weights,
                    &c_gpu.attn_weights,
                    tol,
                );
                bad |= report_diff("attn_out", &c_cpu.attn_out, &c_gpu.attn_out, tol);
                bad |= report_diff("gate", &c_cpu.gate, &c_gpu.gate, tol);
                bad |= report_diff("up", &c_cpu.up, &c_gpu.up, tol);
                bad |= report_diff("gate_silu", &c_cpu.gate_silu, &c_gpu.gate_silu, tol);
            } else {
                let (abs, rel, _) = tensor_diff(&h_cpu, &h_gpu);
                bad = is_diverged(&h_cpu, &h_gpu, abs, rel, tol);
                if bad {
                    println!(
                        "  [fwd] layer {l}: hidden max_abs {abs:.3e} max_rel {rel:.3e} ← 乖離"
                    );
                }
            }
            if bad && first_bad_fwd.is_none() {
                first_bad_fwd = Some(l);
                println!("  ==> forward の最初の乖離は layer {l}");
            }
            hidden = h_cpu; // canonical (CPU) で先へ進める
            caches_cpu.push(c_cpu);
        }
        println!(
            "[parity] forward: {}",
            first_bad_fwd.map_or_else(|| "全層一致".to_string(), |l| format!("layer {l} から乖離"))
        );

        // backward は CPU の cache を両方に食わせて backward 計算だけを比べる
        let mut d_hidden = vec![0.0f32; seq_len * hd];
        for (i, v) in d_hidden.iter_mut().enumerate() {
            *v = ((i % 17) as f32 - 8.0) * 1e-3; // 決定論的な疑似勾配
        }
        let mut first_bad_bwd: Option<usize> = None;
        for l in (0..config.num_layers).rev() {
            lora[l].merge_into(&base_layers[l], &mut merged);
            let (d_cpu, g_cpu) =
                layer_backward(&d_hidden, &caches_cpu[l], &merged, &config, seq_len);
            let (d_gpu, g_gpu_raw) = {
                let cuda_mtx = alice_train::blas::CUDA_MATMUL.get().expect("CUDA 未初期化");
                let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
                // CUDA backward は同じ layer の forward が書いた ws を前提にするので、
                // 直前に recompute する (これを省くと stale ws で全層が乖離して見える)
                let mut recompute_hidden = caches_cpu[l].residual_attn.clone();
                let fresh = alice_train::cuda_matmul::cuda_layer_forward_ws_vram(
                    &cuda,
                    &mut recompute_hidden,
                    &base_layers[l],
                    &vram_layers[l],
                    &config,
                    seq_len,
                    &mut cuda_ws,
                );
                alice_train::cuda_matmul::cuda_layer_backward_ws_vram(
                    &cuda,
                    &d_hidden,
                    &fresh,
                    &base_layers[l],
                    &vram_layers[l],
                    &config,
                    seq_len,
                    &mut cuda_ws,
                )
            };
            let g_gpu: alice_train::llama_backward::LayerWeightGrads = g_gpu_raw.into();
            let mut bad = false;
            let verbose = l + 1 == config.num_layers || l < 2 || first_bad_bwd.is_some();
            if verbose {
                println!("  [bwd] layer {l}");
                bad |= report_diff("d_input", &d_cpu, &d_gpu, tol);
                bad |= report_diff("d_q_proj", &g_cpu.d_q_proj, &g_gpu.d_q_proj, tol);
                bad |= report_diff("d_k_proj", &g_cpu.d_k_proj, &g_gpu.d_k_proj, tol);
                bad |= report_diff("d_v_proj", &g_cpu.d_v_proj, &g_gpu.d_v_proj, tol);
                bad |= report_diff("d_o_proj", &g_cpu.d_o_proj, &g_gpu.d_o_proj, tol);
                bad |= report_diff("d_gate_proj", &g_cpu.d_gate_proj, &g_gpu.d_gate_proj, tol);
                bad |= report_diff("d_up_proj", &g_cpu.d_up_proj, &g_gpu.d_up_proj, tol);
                bad |= report_diff("d_down_proj", &g_cpu.d_down_proj, &g_gpu.d_down_proj, tol);
                bad |= report_diff("d_attn_norm", &g_cpu.d_attn_norm, &g_gpu.d_attn_norm, tol);
                bad |= report_diff("d_ffn_norm", &g_cpu.d_ffn_norm, &g_gpu.d_ffn_norm, tol);
            } else {
                let (abs, rel, _) = tensor_diff(&d_cpu, &d_gpu);
                let (abs_q, rel_q, _) = tensor_diff(&g_cpu.d_q_proj, &g_gpu.d_q_proj);
                bad = is_diverged(&d_cpu, &d_gpu, abs, rel, tol)
                    || is_diverged(&g_cpu.d_q_proj, &g_gpu.d_q_proj, abs_q, rel_q, tol);
                if bad {
                    println!("  [bwd] layer {l}: d_input rel {rel:.3e} (abs {abs:.3e}) / d_q_proj rel {rel_q:.3e} (abs {abs_q:.3e}) ← 乖離");
                }
            }
            if bad && first_bad_bwd.is_none() {
                first_bad_bwd = Some(l);
                println!("  ==> backward の最初の乖離は layer {l}");
            }
            d_hidden = d_cpu;
        }
        println!(
            "[parity] backward: {}",
            first_bad_bwd.map_or_else(|| "全層一致".to_string(), |l| format!("layer {l} から乖離"))
        );
        println!("[parity] tol = {tol:e} (相対) で判定 完了");
        return Ok(());
    }

    let mut log_file = fs::File::create(args.out.join("train.log"))?;
    let mut accum_in_batch = 0usize;
    let mut running_loss = 0.0f32;
    let mut running_tokens = 0usize;
    let mut adam_t: i32 = 0;
    let step_timer = Instant::now();

    for epoch in 0..args.epochs {
        for (si, sample) in samples.iter().enumerate() {
            let seq_len = sample.tokens.len();
            if seq_len < 2 {
                continue;
            }

            // ── forward (pre-norm hidden を保持するため展開) ──
            let mut hidden = vec![0.0f32; seq_len * config.hidden_dim];
            for (t, &tid) in sample.tokens.iter().enumerate() {
                let tid = tid as usize;
                assert!(tid < config.vocab_size, "token id {tid} が vocab 外");
                hidden[t * config.hidden_dim..(t + 1) * config.hidden_dim].copy_from_slice(
                    &embedding[tid * config.hidden_dim..(tid + 1) * config.hidden_dim],
                );
            }
            let mut caches = Vec::with_capacity(config.num_layers);
            if use_cuda_layers {
                // Mutex は layer loop の間だけ保持する。logits 側の matmul は
                // blas 経由で同じ mutex を取るので、scope を分けないと deadlock する。
                #[cfg(feature = "cuda")]
                {
                    let cuda_mtx = alice_train::blas::CUDA_MATMUL.get().expect("CUDA 未初期化");
                    let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
                    for l in 0..config.num_layers {
                        caches.push(alice_train::cuda_matmul::cuda_layer_forward_ws_vram(
                            &cuda,
                            &mut hidden,
                            &base_layers[l], // norm / bias のみ参照される (LoRA は触らない)
                            &vram_layers[l],
                            &config,
                            seq_len,
                            &mut cuda_ws,
                        ));
                    }
                }
            } else {
                for l in 0..config.num_layers {
                    lora[l].merge_into(&base_layers[l], &mut merged);
                    caches.push(layer_forward(&mut hidden, &merged, &config, seq_len));
                }
            }
            let pre_norm = hidden.clone();
            rmsnorm(
                &mut hidden,
                &output_norm,
                config.hidden_dim,
                config.norm_eps,
            );

            // ── completion 位置だけ loss / 勾配 ──
            //
            // logits と d_normed は行をまとめて 2 回の matmul にする。
            // token ごとに回すと vocab × hidden の演算が自前 scalar ループになり
            // BLAS / CUDA dispatch (`blas.rs`) に乗らない。
            let rows: Vec<usize> = (0..seq_len - 1)
                .filter(|t| t + 1 >= sample.completion_start)
                .collect();
            if rows.is_empty() {
                continue;
            }
            let hd = config.hidden_dim;
            let vocab = config.vocab_size;
            let mut d_normed = vec![0.0f32; seq_len * hd];
            let mut sample_loss = 0.0f32;
            let mut sample_tokens = 0usize;
            // 1 チャンクあたり最大 64 行 (vocab 13 万 × f32 なので行数に比例して重い)
            for chunk in rows.chunks(64) {
                let n = chunk.len();
                let mut h_sel = vec![0.0f32; n * hd];
                for (i, &t) in chunk.iter().enumerate() {
                    h_sel[i * hd..(i + 1) * hd].copy_from_slice(&hidden[t * hd..(t + 1) * hd]);
                }
                let mut logits = vec![0.0f32; n * vocab];
                matmul_bt(&h_sel, &output_proj, &mut logits, n, vocab, hd);

                let mut d_logits = vec![0.0f32; n * vocab];
                for (i, &t) in chunk.iter().enumerate() {
                    let target = sample.tokens[t + 1] as usize;
                    let (loss, g) = cross_entropy_loss(&logits[i * vocab..(i + 1) * vocab], target);
                    sample_loss += loss;
                    sample_tokens += 1;
                    d_logits[i * vocab..(i + 1) * vocab].copy_from_slice(&g);
                }

                // d_sel = d_logits × output_proj   (n × vocab)·(vocab × hidden)
                let mut d_sel = vec![0.0f32; n * hd];
                matmul(&d_logits, &output_proj, &mut d_sel, n, hd, vocab);
                for (i, &t) in chunk.iter().enumerate() {
                    d_normed[t * hd..(t + 1) * hd].copy_from_slice(&d_sel[i * hd..(i + 1) * hd]);
                }
            }
            if sample_tokens == 0 {
                continue;
            }

            // ── backward ──
            let mut d_hidden = vec![0.0f32; seq_len * config.hidden_dim];
            let mut d_out_norm = vec![0.0f32; config.hidden_dim]; // 凍結なので捨てる
            rmsnorm_backward(
                &d_normed,
                &pre_norm,
                &output_norm,
                &mut d_hidden,
                &mut d_out_norm,
                config.hidden_dim,
                config.norm_eps,
            );
            if use_cuda_layers {
                #[cfg(feature = "cuda")]
                {
                    let cuda_mtx = alice_train::blas::CUDA_MATMUL.get().expect("CUDA 未初期化");
                    let cuda = cuda_mtx.lock().expect("CUDA mutex poisoned");
                    for l in (0..config.num_layers).rev() {
                        // `cuda_layer_backward_ws_vram` は **同じ layer の forward が
                        // 書いた workspace** を前提にしている (内部で ws の中間値を読む)。
                        // 全層 forward → 逆順 backward と並べると最終層以外 ws が stale に
                        // なり、勾配が壊れて NaN に至る。既存の train-qat-70b と同じく
                        // backward の直前にその layer の forward を recompute する
                        // (activation recomputation)。layer の入力は cache.residual_attn。
                        let mut recompute_hidden = caches[l].residual_attn.clone();
                        let fresh = alice_train::cuda_matmul::cuda_layer_forward_ws_vram(
                            &cuda,
                            &mut recompute_hidden,
                            &base_layers[l],
                            &vram_layers[l],
                            &config,
                            seq_len,
                            &mut cuda_ws,
                        );
                        let (d_in, grads) = alice_train::cuda_matmul::cuda_layer_backward_ws_vram(
                            &cuda,
                            &d_hidden,
                            &fresh,
                            &base_layers[l],
                            &vram_layers[l],
                            &config,
                            seq_len,
                            &mut cuda_ws,
                        );
                        // CUDA path の勾配型は重複定義なので canonical 側へ変換する
                        let grads: alice_train::llama_backward::LayerWeightGrads = grads.into();
                        lora[l].project_grads(&grads, &mut lora_grads[l]);
                        d_hidden = d_in;
                    }
                }
            } else {
                for l in (0..config.num_layers).rev() {
                    lora[l].merge_into(&base_layers[l], &mut merged);
                    let (d_in, grads) =
                        layer_backward(&d_hidden, &caches[l], &merged, &config, seq_len);
                    lora[l].project_grads(&grads, &mut lora_grads[l]);
                    d_hidden = d_in;
                }
            }

            running_loss += sample_loss;
            running_tokens += sample_tokens;
            accum_in_batch += 1;

            // ── optimizer ──
            if accum_in_batch >= args.grad_accum {
                adam_t += 1;
                // token 数で正規化 (loss の平均と勾配のスケールを一致させる)
                let scale = 1.0 / running_tokens.max(1) as f32;
                for l in 0..config.num_layers {
                    let (layer, grads, st) = (&mut lora[l], &lora_grads[l], &mut adam[l]);
                    for_each_adapter!(layer, grads, st, |ad, ga, gb, sa, sb| {
                        sa.step(&mut ad.a, ga, args.lr, args.weight_decay, adam_t, scale);
                        sb.step(&mut ad.b, gb, args.lr, args.weight_decay, adam_t, scale);
                    });
                    lora_grads[l].zero_out();
                }
                global_step += 1;
                // 重みが動いたので VRAM 常駐分を作り直す (次の grad_accum 分で共有)
                #[cfg(feature = "cuda")]
                refresh_vram!();

                if global_step.is_multiple_of(args.log_every as u64) {
                    let avg = running_loss / running_tokens.max(1) as f32;
                    let line = format!(
                        "[sft] epoch {epoch} step {global_step} sample {}/{} loss {avg:.4} tokens {running_tokens} elapsed {:.1}s",
                        si + 1,
                        samples.len(),
                        step_timer.elapsed().as_secs_f32()
                    );
                    println!("{line}");
                    writeln!(log_file, "{line}")?;
                    log_file.flush()?;
                }
                running_loss = 0.0;
                running_tokens = 0;
                accum_in_batch = 0;

                if global_step.is_multiple_of(args.save_every as u64) {
                    save_adapter(&ckpt_path, &lora, global_step, lora_cfg)?;
                    println!(
                        "[sft] checkpoint 保存: {} (step {global_step})",
                        ckpt_path.display()
                    );
                }
            }
        }
    }

    save_adapter(&ckpt_path, &lora, global_step, lora_cfg)?;
    println!(
        "[sft] 完了: step {global_step}, 所要 {:.1}s, checkpoint {}",
        t_start.elapsed().as_secs_f32(),
        ckpt_path.display()
    );
    Ok(())
}

const ADAPTER_MAGIC: &[u8; 8] = b"ALORA001";

/// LoRA アダプタを保存する (base 重みは含まない)。
fn save_adapter(
    path: &Path,
    lora: &[LoraLayer],
    step: u64,
    cfg: LoraConfig,
) -> std::io::Result<()> {
    let tmp = path.with_extension("bin.tmp");
    let mut f = std::io::BufWriter::new(fs::File::create(&tmp)?);
    f.write_all(ADAPTER_MAGIC)?;
    f.write_all(&(cfg.rank() as u32).to_le_bytes())?;
    f.write_all(&cfg.alpha().to_le_bytes())?;
    f.write_all(&(lora.len() as u32).to_le_bytes())?;
    f.write_all(&step.to_le_bytes())?;
    for layer in lora {
        for ad in [
            &layer.q,
            &layer.k,
            &layer.v,
            &layer.o,
            &layer.gate,
            &layer.up,
            &layer.down,
        ] {
            for buf in [&ad.a, &ad.b] {
                f.write_all(&(buf.len() as u64).to_le_bytes())?;
                for x in buf {
                    f.write_all(&x.to_le_bytes())?;
                }
            }
        }
    }
    f.flush()?;
    drop(f);
    fs::rename(&tmp, path)?; // 書き込み途中の checkpoint を残さない
    Ok(())
}

/// 保存済みアダプタを読み戻し、`step` を返す。
fn load_adapter(path: &Path, lora: &mut [LoraLayer]) -> std::io::Result<u64> {
    let bytes = fs::read(path)?;
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    if bytes.len() < 24 || &bytes[..8] != ADAPTER_MAGIC {
        return Err(bad("adapter の magic が合わない"));
    }
    let num_layers = u32::from_le_bytes(bytes[16..20].try_into().expect("4 bytes")) as usize;
    if num_layers != lora.len() {
        return Err(bad("adapter の層数がモデルと合わない"));
    }
    let step = u64::from_le_bytes(bytes[20..28].try_into().expect("8 bytes"));
    let mut off = 28usize;
    for layer in lora.iter_mut() {
        for ad in [
            &mut layer.q,
            &mut layer.k,
            &mut layer.v,
            &mut layer.o,
            &mut layer.gate,
            &mut layer.up,
            &mut layer.down,
        ] {
            for buf in [&mut ad.a, &mut ad.b] {
                let n = u64::from_le_bytes(
                    bytes[off..off + 8]
                        .try_into()
                        .map_err(|_| bad("長さ欠損"))?,
                ) as usize;
                off += 8;
                if n != buf.len() || off + n * 4 > bytes.len() {
                    return Err(bad("adapter の形がモデルと合わない"));
                }
                for v in buf.iter_mut() {
                    *v = f32::from_le_bytes(
                        bytes[off..off + 4]
                            .try_into()
                            .map_err(|_| bad("f32 欠損"))?,
                    );
                    off += 4;
                }
            }
        }
    }
    Ok(step)
}
