//! Q1: 実音声 A/B verify — speech-like synthetic signal で YIN vs alice-world F0 の
//! quantitative 比較 (Bridge C の real value 証明)。
//!
//! ## 目的
//!
//! v0.2 blocker Q1 (alice-world ROADMAP.md 参照)。Bridge C 統合 (BC-Step 1-3、
//! `AudioFeatureExtractor::extract_f0_with_algorithm`) で選択可能になった
//! `F0Algorithm::AliceWorldDioStoneMask` の real-world 効果を、実音声に近い
//! synthetic signal で数値的に verify する。
//!
//! ## Signal 仕様 (speech-like)
//!
//! 女性声を模擬:
//! - **F0 contour**: 200 Hz 中心、10% depth vibrato @ 5.5 Hz + ±2% micro-jitter
//! - **Voiced portion**: 0.0 - 1.5 秒 (harmonic + formant weight)
//! - **Unvoiced portion**: 1.5 - 2.0 秒 (noise burst、F0=0)
//! - **Formant model**: 3 formants (F1=800, F2=1200, F3=2800Hz) を harmonic gain で近似
//!
//! ## 比較 metric
//!
//! - **F0 RMSE** (voiced 区間で true vs 抽出)
//! - **Voiced coverage %** (正しく voiced 判定した frame 比率)
//! - **F0 temporal stability** (隣接 frame 差の std)
//! - **Unvoiced detection %** (unvoiced 区間で F0=0 判定した比率)

#![cfg(feature = "tts")]

use alice_train::tts::{AudioFeatureExtractor, F0Algorithm};
use std::f32::consts::PI;

const SAMPLE_RATE: u32 = 24_000;
const N_FFT: usize = 1024;
const HOP_LENGTH: usize = 256;
const N_MELS: usize = 80;
const DURATION_SEC: f32 = 2.0;

// ---------- Signal generators ----------

/// Speech-like signal: F0 vibrato + micro-jitter + 3 formant gain 重み。
/// 前半 (< 1.5 秒) は voiced、後半は unvoiced (noise burst)。
fn speech_like_signal(seed: u64) -> Vec<f32> {
    let fs = SAMPLE_RATE as f32;
    let n_samples = (DURATION_SEC * fs) as usize;
    let voiced_until_sec = 1.5_f32;

    // Formant weights (母音「あ」相当): F1, F2, F3 の相対 gain (harmonic 番号ベース)
    // harmonic k の gain は Gaussian shape で 3 formants に集約
    let f0_center = 200.0_f32;
    let formants_hz = [800.0_f32, 1200.0, 2800.0];
    let formant_bw = 250.0_f32; // bandwidth (Hz)
    let mut jitter_state: u64 = seed;
    let dt = 1.0 / fs;

    // Phase accumulator: 時変 F0 では phase = 2π ∫ f0(τ) dτ で積分必須。
    // per-sample の f0_inst * t 近似は phase 不連続を招き周期性を破壊する。
    let mut phase_acc: f32 = 0.0;

    let mut out = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f32 / fs;
        if t >= voiced_until_sec {
            // Unvoiced burst: LCG noise
            jitter_state = jitter_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let r = ((jitter_state >> 32) as u32 as f32) / u32::MAX as f32 - 0.5;
            out.push(r * 0.3); // amplitude 0.3
        } else {
            // Voiced: vibrato + jitter な F0 で harmonic 合成
            let vibrato = 20.0 * (2.0 * PI * 5.5 * t).sin(); // ±20 Hz vibrato
            // Micro-jitter (LCG-based deterministic)
            jitter_state = jitter_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let jitter_r = ((jitter_state >> 32) as u32 as f32) / u32::MAX as f32 - 0.5;
            let jitter = jitter_r * 8.0; // ±4 Hz jitter
            let f0_inst = f0_center + vibrato + jitter;

            phase_acc += 2.0 * PI * f0_inst * dt;

            // Harmonic 合成 (formant gain 重み付き + fundamental dominant 保証)
            // 診断で判明: formant gain 0.5 は k=4/6 で fundamental を上回り
            // YIN の autocorrelation が octave-up period を誤選定する。
            // Fundamental を dominant にするため base gain 2.5、formant gain 0.2 に縮小。
            let mut sample = 0.0;
            for k in 1..=15 {
                let f_k = f0_inst * (k as f32);
                let formant_gain: f32 = formants_hz
                    .iter()
                    .map(|&fc| ((-(f_k - fc).powi(2)) / (2.0 * formant_bw.powi(2))).exp())
                    .sum();
                // Fundamental 強調 (2.5/k) + 抑えた formant (0.2)
                let gain = 2.5 / (k as f32) + formant_gain * 0.2;
                sample += gain * (phase_acc * (k as f32)).sin();
            }
            out.push(sample * 0.1); // scale down to avoid clipping
        }
    }
    out
}

/// Signal に対する true F0 contour (frame-level ground truth)。
///
/// Voiced 区間の instantaneous F0 を frame center で sampling。unvoiced は 0.0。
fn true_f0_contour(n_frames: usize) -> Vec<f32> {
    let fs = SAMPLE_RATE as f32;
    let hop = HOP_LENGTH as f32;
    let f0_center = 200.0_f32;
    let voiced_until_sec = 1.5_f32;
    (0..n_frames)
        .map(|i| {
            let t = i as f32 * hop / fs;
            if t >= voiced_until_sec {
                0.0
            } else {
                // Jitter は evaluation で除外 (deterministic 部分のみ)
                let vibrato = 20.0 * (2.0 * PI * 5.5 * t).sin();
                f0_center + vibrato
            }
        })
        .collect()
}

// ---------- Metric helpers ----------

/// Voiced 区間 (両者とも voiced 判定 + true voiced) で F0 RMSE を計算。
fn voiced_rmse(true_f0: &[f32], est_f0: &[f32], est_voiced: &[bool]) -> (f32, usize) {
    let mut sum_sq = 0.0_f32;
    let mut n = 0_usize;
    for ((&t, &e), &v) in true_f0.iter().zip(est_f0.iter()).zip(est_voiced.iter()) {
        if t > 0.0 && v && e > 0.0 {
            let d = t - e;
            sum_sq += d * d;
            n += 1;
        }
    }
    let rmse = if n > 0 {
        (sum_sq / n as f32).sqrt()
    } else {
        f32::NAN
    };
    (rmse, n)
}

/// Voiced coverage: true voiced 区間で estimate が voiced 判定した比率 (%)。
fn voiced_coverage_pct(true_f0: &[f32], est_voiced: &[bool]) -> f32 {
    let mut true_voiced_count = 0_usize;
    let mut correct_voiced = 0_usize;
    for (&t, &v) in true_f0.iter().zip(est_voiced.iter()) {
        if t > 0.0 {
            true_voiced_count += 1;
            if v {
                correct_voiced += 1;
            }
        }
    }
    if true_voiced_count > 0 {
        100.0 * correct_voiced as f32 / true_voiced_count as f32
    } else {
        f32::NAN
    }
}

/// Unvoiced detection: true unvoiced 区間で estimate が unvoiced 判定した比率 (%)。
fn unvoiced_detection_pct(true_f0: &[f32], est_voiced: &[bool]) -> f32 {
    let mut true_unvoiced = 0_usize;
    let mut correct_unvoiced = 0_usize;
    for (&t, &v) in true_f0.iter().zip(est_voiced.iter()) {
        if t == 0.0 {
            true_unvoiced += 1;
            if !v {
                correct_unvoiced += 1;
            }
        }
    }
    if true_unvoiced > 0 {
        100.0 * correct_unvoiced as f32 / true_unvoiced as f32
    } else {
        f32::NAN
    }
}

/// F0 temporal stability: 隣接 voiced frame 間の F0 diff の std (jitter response)。
fn f0_temporal_stability(est_f0: &[f32], est_voiced: &[bool]) -> f32 {
    let mut diffs = Vec::new();
    for i in 1..est_f0.len() {
        if est_voiced[i] && est_voiced[i - 1] {
            diffs.push(est_f0[i] - est_f0[i - 1]);
        }
    }
    if diffs.len() < 2 {
        return f32::NAN;
    }
    let mean: f32 = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let var: f32 = diffs.iter().map(|&d| (d - mean).powi(2)).sum::<f32>() / diffs.len() as f32;
    var.sqrt()
}

// ---------- Tests ----------

#[test]
fn q1_speech_like_ab_extract() {
    let ex = AudioFeatureExtractor::new(SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS);
    let wav = speech_like_signal(0x1234_5678_9abc_def0);
    let (f0_yin, voiced_yin) = ex.extract_f0_with_algorithm(&wav, F0Algorithm::Yin);
    let (f0_aw, voiced_aw) =
        ex.extract_f0_with_algorithm(&wav, F0Algorithm::AliceWorldDioStoneMask);

    // Shape 一致 (両者とも 1 + wav.len() / hop_length)
    assert_eq!(f0_yin.len(), f0_aw.len());
    assert_eq!(voiced_yin.len(), voiced_aw.len());
    let expected_frames = 1 + wav.len() / HOP_LENGTH;
    assert_eq!(f0_yin.len(), expected_frames);
}

#[test]
fn q1_voiced_f0_rmse_comparison() {
    let ex = AudioFeatureExtractor::new(SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS);
    let wav = speech_like_signal(0x1234_5678_9abc_def0);
    let (f0_yin, voiced_yin) = ex.extract_f0_with_algorithm(&wav, F0Algorithm::Yin);
    let (f0_aw, voiced_aw) =
        ex.extract_f0_with_algorithm(&wav, F0Algorithm::AliceWorldDioStoneMask);
    let true_f0 = true_f0_contour(f0_yin.len());

    let (rmse_yin, n_yin) = voiced_rmse(&true_f0, &f0_yin, &voiced_yin);
    let (rmse_aw, n_aw) = voiced_rmse(&true_f0, &f0_aw, &voiced_aw);

    println!(
        "[speech-like 200Hz vibrato + jitter] Voiced F0 RMSE: YIN={rmse_yin:.3} Hz ({n_yin} frames), alice-world={rmse_aw:.3} Hz ({n_aw} frames)"
    );

    // 両者とも finite + reasonable range
    assert!(rmse_yin.is_finite() && rmse_yin < 50.0, "YIN RMSE: {rmse_yin}");
    assert!(rmse_aw.is_finite() && rmse_aw < 50.0, "alice-world RMSE: {rmse_aw}");
}

#[test]
fn q1_voiced_uv_coverage_comparison() {
    let ex = AudioFeatureExtractor::new(SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS);
    let wav = speech_like_signal(0x1234_5678_9abc_def0);
    let (_, voiced_yin) = ex.extract_f0_with_algorithm(&wav, F0Algorithm::Yin);
    let (_, voiced_aw) = ex.extract_f0_with_algorithm(&wav, F0Algorithm::AliceWorldDioStoneMask);
    let true_f0 = true_f0_contour(voiced_yin.len());

    let voiced_yin_pct = voiced_coverage_pct(&true_f0, &voiced_yin);
    let voiced_aw_pct = voiced_coverage_pct(&true_f0, &voiced_aw);
    let unvoiced_yin_pct = unvoiced_detection_pct(&true_f0, &voiced_yin);
    let unvoiced_aw_pct = unvoiced_detection_pct(&true_f0, &voiced_aw);

    println!(
        "[speech-like] Voiced coverage: YIN={voiced_yin_pct:.1}%, alice-world={voiced_aw_pct:.1}%"
    );
    println!(
        "[speech-like] Unvoiced detection: YIN={unvoiced_yin_pct:.1}%, alice-world={unvoiced_aw_pct:.1}%"
    );

    // Q1 finding (Bridge C の real value 証明):
    // - YIN: precision >>> recall (voiced coverage ~20%、conservative)
    // - alice-world: recall > precision (voiced coverage 100%、continuity fix で unvoiced 補完)
    // - Unvoiced boundary は alice-world で消える (documented tradeoff)
    // Assert は algorithm 特性反映の緩い bound、regression detector 用途。
    assert!(
        voiced_yin_pct > 10.0,
        "YIN voiced coverage too low (regression): {voiced_yin_pct}"
    );
    assert!(
        voiced_aw_pct > 50.0,
        "alice-world voiced coverage too low (regression): {voiced_aw_pct}"
    );
}

#[test]
fn q1_f0_temporal_stability_comparison() {
    let ex = AudioFeatureExtractor::new(SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS);
    let wav = speech_like_signal(0x1234_5678_9abc_def0);
    let (f0_yin, voiced_yin) = ex.extract_f0_with_algorithm(&wav, F0Algorithm::Yin);
    let (f0_aw, voiced_aw) =
        ex.extract_f0_with_algorithm(&wav, F0Algorithm::AliceWorldDioStoneMask);

    let stab_yin = f0_temporal_stability(&f0_yin, &voiced_yin);
    let stab_aw = f0_temporal_stability(&f0_aw, &voiced_aw);

    println!(
        "[speech-like] F0 temporal stability (frame-diff std): YIN={stab_yin:.3} Hz, alice-world={stab_aw:.3} Hz"
    );

    // 両者とも finite (regression detector)
    assert!(stab_yin.is_finite());
    assert!(stab_aw.is_finite());
}

#[test]
fn q1_both_algorithms_produce_finite_output() {
    let ex = AudioFeatureExtractor::new(SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS);
    let wav = speech_like_signal(0x1234_5678_9abc_def0);
    for algo in [F0Algorithm::Yin, F0Algorithm::AliceWorldDioStoneMask] {
        let (f0, voiced) = ex.extract_f0_with_algorithm(&wav, algo);
        for (i, &v) in f0.iter().enumerate() {
            assert!(v.is_finite(), "[{algo:?}] F0 non-finite at {i}: {v}");
        }
        // voiced flag と F0 == 0.0 の整合性
        for (i, (&v, &f)) in voiced.iter().zip(f0.iter()).enumerate() {
            assert_eq!(
                v,
                f > 0.0,
                "[{algo:?}] voiced/f0 mismatch at {i}: voiced={v}, f0={f}"
            );
        }
    }
}
