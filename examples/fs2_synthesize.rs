//! FastSpeech2 音声合成 example (Phase T.4 test)。
//!
//! checkpoint から model を load し、mora 列 → mel → Griffin-Lim → WAV を出力する。
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --features tts --example fs2_synthesize -- \
//!     --checkpoint checkpoints/fs2_jsut/step_8000.safetensors \
//!     --config configs/fs2_jsut.json \
//!     --output test_synth.wav
//! ```
//!
//! 出力: 24kHz mono WAV (Griffin-Lim vocoder、30 iter)

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let ckpt_path = parse_arg(&args, "--checkpoint")
        .unwrap_or_else(|| PathBuf::from("checkpoints/fs2_jsut/step_8000.safetensors"));
    let config_path =
        parse_arg(&args, "--config").unwrap_or_else(|| PathBuf::from("configs/fs2_jsut.json"));
    let output_path =
        parse_arg(&args, "--output").unwrap_or_else(|| PathBuf::from("test_synth.wav"));
    let gl_iter: usize = parse_arg(&args, "--gl-iter")
        .and_then(|p| p.to_str().and_then(|s| s.parse().ok()))
        .unwrap_or(30);

    println!("[info] checkpoint: {}", ckpt_path.display());
    println!("[info] config: {}", config_path.display());
    println!("[info] output: {}", output_path.display());
    println!("[info] Griffin-Lim iter: {}", gl_iter);

    let config_json = fs::read_to_string(&config_path)?;
    let _config: serde_json::Value = serde_json::from_str(&config_json)?;

    #[cfg(feature = "cuda")]
    {
        println!("[info] Initializing CUDA cuBLAS...");
        alice_train::blas::init_cuda_blas();
    }

    #[cfg(feature = "tts")]
    {
        use alice_train::tts::{FastSpeech2, FastSpeech2Config};

        let fs2_config = FastSpeech2Config {
            vocab_size: config["model"]["vocab_size"].as_u64().unwrap_or(100) as usize,
            hidden_dim: config["model"]["hidden_dim"].as_u64().unwrap_or(128) as usize,
            num_heads: config["model"]["num_heads"].as_u64().unwrap_or(2) as usize,
            num_encoder_layers: config["model"]["num_encoder_layers"].as_u64().unwrap_or(2)
                as usize,
            num_decoder_layers: config["model"]["num_decoder_layers"].as_u64().unwrap_or(2)
                as usize,
            fft_kernel_size: config["model"]["fft_kernel_size"].as_u64().unwrap_or(3) as usize,
            fft_expansion: config["model"]["fft_expansion"].as_u64().unwrap_or(4) as usize,
            predictor_kernel_size: config["model"]["predictor_kernel_size"]
                .as_u64()
                .unwrap_or(3) as usize,
            predictor_hidden: config["model"]["predictor_hidden"].as_u64().unwrap_or(128) as usize,
            mel_dim: config["model"]["mel_dim"].as_u64().unwrap_or(80) as usize,
            postnet_kernel_size: config["model"]["postnet_kernel_size"].as_u64().unwrap_or(5)
                as usize,
            postnet_layers: config["model"]["postnet_layers"].as_u64().unwrap_or(3) as usize,
            postnet_hidden: config["model"]["postnet_hidden"].as_u64().unwrap_or(128) as usize,
            max_len: config["model"]["max_len"].as_u64().unwrap_or(2048) as usize,
        };
        println!("[info] Building FastSpeech2 model + loading checkpoint...");
        let mut model = FastSpeech2::zeros(fs2_config)?;
        model.load_safetensors(&ckpt_path)?;
        println!(
            "[info] Model loaded (hidden_dim={}, layers={}/{})",
            fs2_config.hidden_dim, fs2_config.num_encoder_layers, fs2_config.num_decoder_layers
        );

        // 適当な mora sequence: "aiueo aiueo" 相当 (mora vocab は JSUT で 42、a=5, i=18, u=4, e=12, o=32)
        // 実 vocab file: data/jsut/mora_vocab.txt (a=5, i=18, u=4, e=12, o=32 が今の設定)
        let mora_ids: Vec<u32> = vec![5, 18, 4, 12, 32, 5, 18, 4, 12, 32]; // aiueo × 2
        let durations: Vec<u32> = vec![10, 10, 10, 10, 10, 10, 10, 10, 10, 10]; // 10 frame each
        let mora_len = mora_ids.len();
        println!(
            "[info] Synthesizing: {} moras, total {} frames",
            mora_len,
            durations.iter().sum::<u32>()
        );

        let mel = model.forward(&mora_ids, 1, mora_len, &durations)?;
        let frames: usize = durations.iter().map(|&d| d as usize).sum();
        let mel_dim = fs2_config.mel_dim;
        println!(
            "[info] Mel shape: [{}, {}] (frames × mel_dim)",
            frames, mel_dim
        );

        // mel は [batch=1, frames, mel_dim] flat、log-scale
        // Statistics
        let mel_min = mel.iter().cloned().fold(f32::INFINITY, f32::min);
        let mel_max = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mel_mean: f32 = mel.iter().sum::<f32>() / mel.len() as f32;
        println!(
            "[info] Mel stats: min={:.3}, max={:.3}, mean={:.3}",
            mel_min, mel_max, mel_mean
        );

        // 転置 [frames, mel_dim] → [mel_dim, frames] は不要 (すでに time-major flatten)
        // exp() で linear mel に戻す
        let linear_mel: Vec<f32> = mel.iter().map(|&x| x.exp() - 1e-5).collect();

        // Griffin-Lim
        let sample_rate = config["audio"]["sample_rate"].as_u64().unwrap_or(48_000) as u32;
        let n_fft = config["audio"]["n_fft"].as_u64().unwrap_or(2048) as usize;
        let hop_length = config["audio"]["hop_length"].as_u64().unwrap_or(512) as usize;
        println!(
            "[info] Audio params: sr={}, n_fft={}, hop={}",
            sample_rate, n_fft, hop_length
        );

        let wav = griffin_lim_from_mel(
            &linear_mel,
            frames,
            mel_dim,
            n_fft,
            hop_length,
            sample_rate,
            gl_iter,
        );
        println!(
            "[info] Waveform generated: {} samples ({:.2}s)",
            wav.len(),
            wav.len() as f32 / sample_rate as f32
        );

        // WAV 書き込み
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&output_path, spec)?;
        let peak = wav.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let scale = if peak > 0.0 { 0.9 / peak } else { 1.0 };
        for &v in &wav {
            let clipped = (v * scale).clamp(-1.0, 1.0);
            writer.write_sample((clipped * 32767.0) as i16)?;
        }
        writer.finalize()?;
        println!("[info] WAV written: {}", output_path.display());
    }

    #[cfg(not(feature = "tts"))]
    eprintln!("[error] This example requires --features tts");

    Ok(())
}

fn parse_arg(args: &[String], key: &str) -> Option<PathBuf> {
    for i in 0..args.len() {
        if args[i] == key && i + 1 < args.len() {
            return Some(PathBuf::from(&args[i + 1]));
        }
    }
    None
}

/// Griffin-Lim vocoder: mel → linear magnitude → iterative phase estimation → waveform
#[cfg(feature = "tts")]
fn griffin_lim_from_mel(
    linear_mel: &[f32],
    frames: usize,
    mel_dim: usize,
    n_fft: usize,
    hop_length: usize,
    sample_rate: u32,
    iter: usize,
) -> Vec<f32> {
    use alice_train::tts::audio::mel::mel_filterbank;

    let n_bins = n_fft / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;

    // mel filterbank: [n_mels, n_bins]
    let mel_filters = mel_filterbank(mel_dim, n_fft, sample_rate, 0.0, nyquist);

    // Pseudo-inverse via transpose (簡易): linear = mel × filterbank^T
    // 精度低いが Griffin-Lim iter で改善
    let mut magnitude = vec![0.0_f32; frames * n_bins];
    for t in 0..frames {
        for b in 0..n_bins {
            let mut acc = 0.0_f32;
            for m in 0..mel_dim {
                acc += linear_mel[t * mel_dim + m] * mel_filters[m][b];
            }
            magnitude[t * n_bins + b] = acc.max(0.0);
        }
    }

    // Griffin-Lim: initialize with random phase (or zero), iterate STFT/iSTFT
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut phase: Vec<f32> = (0..frames * n_bins)
        .map(|_| rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI))
        .collect();

    let mut wav = vec![0.0_f32; frames * hop_length + n_fft];

    for it in 0..iter {
        // 1. Build complex STFT from magnitude + phase
        let stft_complex: Vec<(f32, f32)> = magnitude
            .iter()
            .zip(&phase)
            .map(|(&mag, &ph)| (mag * ph.cos(), mag * ph.sin()))
            .collect();

        // 2. iSTFT (overlap-add)
        wav = istft(&stft_complex, frames, n_fft, hop_length);

        // 3. STFT again to update phase
        let recomputed = stft_simple(&wav, n_fft, hop_length, frames);

        // 4. Extract new phase (magnitude constraint kept)
        for (i, &(re, im)) in recomputed.iter().enumerate() {
            let new_mag = (re * re + im * im).sqrt().max(1e-12);
            phase[i] = im.atan2(re);
            // consistency check
            let _ = new_mag;
        }

        if it % 5 == 0 {
            println!("[gl] iter {}/{}", it, iter);
        }
    }

    // Final iSTFT
    let stft_complex: Vec<(f32, f32)> = magnitude
        .iter()
        .zip(&phase)
        .map(|(&mag, &ph)| (mag * ph.cos(), mag * ph.sin()))
        .collect();
    istft(&stft_complex, frames, n_fft, hop_length)
}

#[cfg(feature = "tts")]
fn istft(stft_complex: &[(f32, f32)], frames: usize, n_fft: usize, hop_length: usize) -> Vec<f32> {
    use rustfft::{num_complex::Complex, FftPlanner};
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(n_fft);
    let n_bins = n_fft / 2 + 1;

    // Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / (n_fft - 1) as f32;
            x.sin().powi(2)
        })
        .collect();

    let total_len = frames * hop_length + n_fft;
    let mut wav = vec![0.0_f32; total_len];
    let mut norm = vec![0.0_f32; total_len];

    for t in 0..frames {
        // Rebuild full spectrum (symmetric)
        let mut spectrum: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; n_fft];
        for b in 0..n_bins {
            let (re, im) = stft_complex[t * n_bins + b];
            spectrum[b] = Complex { re, im };
            if b > 0 && b < n_fft - b {
                spectrum[n_fft - b] = Complex { re, im: -im };
            }
        }
        ifft.process(&mut spectrum);
        let start = t * hop_length;
        for i in 0..n_fft {
            let v = spectrum[i].re / n_fft as f32;
            wav[start + i] += v * window[i];
            norm[start + i] += window[i] * window[i];
        }
    }
    for i in 0..total_len {
        if norm[i] > 1e-8 {
            wav[i] /= norm[i];
        }
    }
    wav
}

#[cfg(feature = "tts")]
fn stft_simple(wav: &[f32], n_fft: usize, hop_length: usize, frames: usize) -> Vec<(f32, f32)> {
    use rustfft::{num_complex::Complex, FftPlanner};
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let n_bins = n_fft / 2 + 1;

    let window: Vec<f32> = (0..n_fft)
        .map(|i| {
            let x = std::f32::consts::PI * i as f32 / (n_fft - 1) as f32;
            x.sin().powi(2)
        })
        .collect();

    let mut result = vec![(0.0_f32, 0.0_f32); frames * n_bins];
    for t in 0..frames {
        let start = t * hop_length;
        let mut buf: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| {
                let idx = start + i;
                let x = if idx < wav.len() { wav[idx] } else { 0.0 };
                Complex {
                    re: x * window[i],
                    im: 0.0,
                }
            })
            .collect();
        fft.process(&mut buf);
        for b in 0..n_bins {
            result[t * n_bins + b] = (buf[b].re, buf[b].im);
        }
    }
    result
}
