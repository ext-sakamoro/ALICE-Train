//! Short-Time Fourier Transform (STFT) with Hann window。
//!
//! wav 波形を frame-by-frame の複素スペクトル列に変換する。
//! Vocos vocoder (Phase T.5) と librosa 実装との数値互換性を意識した設計。
//!
//! # 実装仕様
//!
//! - Window: Hann window (`0.5 * (1 - cos(2π n / (N-1)))`)
//! - Center padding: 反射 padding で first/last frame も中心化 (librosa `center=True` 相当)
//! - Frame layout: `frames = 1 + (wav_len / hop_length)` (center=True の場合)
//! - Complex layout: `Vec<Vec<Complex<f32>>>[frames][n_fft/2 + 1]` (real-only 入力の対称性を利用)

use rustfft::{num_complex::Complex, FftPlanner};

/// Hann window を生成する。
///
/// `w[n] = 0.5 * (1 - cos(2π n / (N-1)))` (`n = 0..N`)
///
/// # Panics
///
/// `n_fft < 2` の場合 panic。
#[must_use]
pub fn hann_window(n_fft: usize) -> Vec<f32> {
    assert!(n_fft >= 2, "n_fft must be >= 2");
    let denom = (n_fft - 1) as f32;
    (0..n_fft)
        .map(|n| {
            let phase = 2.0 * std::f32::consts::PI * (n as f32) / denom;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

/// STFT を計算する。
///
/// wav 波形を Hann window + FFT で frame ごとに複素スペクトルに変換する。
/// `center = true` (常に有効) で反射 padding、`return_complex = true` (常に有効) で複素値。
///
/// # 引数
///
/// - `wav`: 入力波形 (mono f32)
/// - `n_fft`: FFT サイズ (通常 1024 or 2048、2 の冪推奨)
/// - `hop_length`: hop サイズ (通常 `n_fft / 4`)
///
/// # 戻り値
///
/// `Vec<Vec<Complex<f32>>>[frames][n_fft/2 + 1]` の複素スペクトログラム。
/// frame 数は `1 + wav.len() / hop_length` (center padding 適用)。
///
/// # Panics
///
/// - `n_fft < 2`
/// - `hop_length == 0`
/// - `wav.len() == 0`
#[must_use]
pub fn stft(wav: &[f32], n_fft: usize, hop_length: usize) -> Vec<Vec<Complex<f32>>> {
    assert!(n_fft >= 2, "n_fft must be >= 2");
    assert!(hop_length > 0, "hop_length must be > 0");
    assert!(!wav.is_empty(), "wav must not be empty");

    let window = hann_window(n_fft);
    let pad = n_fft / 2;
    let padded = reflect_pad(wav, pad);

    let n_freq = n_fft / 2 + 1;
    let n_frames = 1 + wav.len() / hop_length;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut spectrogram = Vec::with_capacity(n_frames);
    let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        if start + n_fft > padded.len() {
            break;
        }

        // window × frame → complex buffer
        for i in 0..n_fft {
            buffer[i] = Complex::new(padded[start + i] * window[i], 0.0);
        }

        fft.process(&mut buffer);

        // real input の対称性を利用して 0..n_freq のみ保持
        let mut frame_spec = Vec::with_capacity(n_freq);
        frame_spec.extend_from_slice(&buffer[..n_freq]);
        spectrogram.push(frame_spec);
    }

    spectrogram
}

/// 反射 padding。`[a, b, c, d]` を左右 `pad` サンプル反射で拡張する (librosa `reflect` mode 準拠)。
fn reflect_pad(wav: &[f32], pad: usize) -> Vec<f32> {
    let n = wav.len();
    if pad == 0 {
        return wav.to_vec();
    }

    let mut out = Vec::with_capacity(n + 2 * pad);

    // 左側反射: wav[pad], wav[pad-1], ..., wav[1] (先頭 wav[0] は含めない = librosa 準拠)
    for i in 0..pad {
        let src_idx = pad - i;
        // Clamp: 波形長より長い pad を要求されても index out of range にならない
        let clamped = src_idx.min(n - 1);
        out.push(wav[clamped]);
    }

    // オリジナル
    out.extend_from_slice(wav);

    // 右側反射: wav[n-2], wav[n-3], ..., wav[n-pad-1]
    for i in 0..pad {
        // n - 2 - i を目指すが、負にならないよう clamp
        let src_idx = if n >= 2 + i { n - 2 - i } else { 0 };
        out.push(wav[src_idx]);
    }

    out
}

/// 複素スペクトログラムの magnitude spectrogram を計算する (`|X|`)。
///
/// # 引数
///
/// - `spec`: STFT 出力 (`Vec<Vec<Complex<f32>>>[frames][n_freq]`)
///
/// # 戻り値
///
/// `Vec<Vec<f32>>[frames][n_freq]` の magnitude spectrogram。
#[must_use]
pub fn magnitude(spec: &[Vec<Complex<f32>>]) -> Vec<Vec<f32>> {
    spec.iter()
        .map(|frame| frame.iter().map(|c| c.re.hypot(c.im)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_endpoints_are_zero() {
        let w = hann_window(1024);
        assert!(w[0].abs() < 1e-6);
        assert!(w[1023].abs() < 1e-6);
        // 中央は最大 (=1.0)
        assert!((w[511] - 1.0).abs() < 1e-2);
    }

    #[test]
    fn hann_window_sum_is_positive() {
        let w = hann_window(512);
        let sum: f32 = w.iter().sum();
        assert!(sum > 0.0);
        // 理論値: Σ hann ≈ (N-1) / 2
        let expected = (512.0 - 1.0) / 2.0;
        assert!(
            (sum - expected).abs() / expected < 0.01,
            "sum={sum}, expected~{expected}"
        );
    }

    #[test]
    fn stft_shape_is_correct() {
        let wav = vec![0.0_f32; 24_000]; // 1 sec silence @ 24kHz
        let n_fft = 1024;
        let hop = 256;
        let spec = stft(&wav, n_fft, hop);

        // center=True: frames = 1 + wav_len / hop
        let expected_frames = 1 + 24_000 / hop;
        assert_eq!(spec.len(), expected_frames);
        assert_eq!(spec[0].len(), n_fft / 2 + 1);
    }

    #[test]
    fn stft_of_silence_is_zero() {
        let wav = vec![0.0_f32; 24_000];
        let spec = stft(&wav, 1024, 256);
        for frame in &spec {
            for c in frame {
                assert!(c.re.abs() < 1e-6);
                assert!(c.im.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn stft_of_sine_has_energy_at_target_bin() {
        // 440 Hz sine @ 24 kHz sample rate
        let sr = 24_000_f32;
        let freq = 440.0_f32;
        let n_samples = 24_000;
        let wav: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32) / sr).sin())
            .collect();

        let n_fft = 2048;
        let spec = stft(&wav, n_fft, 512);
        let mag = magnitude(&spec);

        // 目的 bin index: freq * n_fft / sr = 440 * 2048 / 24000 ≈ 37.5
        let target_bin = (freq * n_fft as f32 / sr).round() as usize;

        // 中央 frame でエネルギーが target_bin 周辺に集中しているか
        let mid_frame = &mag[mag.len() / 2];
        let peak_bin = mid_frame
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or(0, |(i, _)| i);

        // ±2 bin 以内に peak があれば OK
        let diff = peak_bin.abs_diff(target_bin);
        assert!(
            diff <= 2,
            "peak_bin={peak_bin}, target_bin={target_bin}, diff={diff}"
        );
    }

    #[test]
    fn magnitude_is_non_negative() {
        let wav: Vec<f32> = (0..24_000).map(|i| (i as f32 * 0.001).sin()).collect();
        let spec = stft(&wav, 1024, 256);
        let mag = magnitude(&spec);

        for frame in &mag {
            for &m in frame {
                assert!(m >= 0.0);
            }
        }
    }

    #[test]
    #[should_panic(expected = "n_fft must be >= 2")]
    fn hann_window_panics_on_small_n() {
        let _ = hann_window(1);
    }

    #[test]
    #[should_panic(expected = "wav must not be empty")]
    fn stft_panics_on_empty_wav() {
        let _ = stft(&[], 1024, 256);
    }
}
