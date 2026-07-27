//! Mel filterbank (Slaney scale, librosa htk=False 互換)。
//!
//! # 参照
//!
//! - Slaney, M. (1998), "Auditory Toolbox Version 2"
//! - librosa `librosa.filters.mel(htk=False, norm='slaney')` と数値互換
//!
//! # 実装仕様
//!
//! - 周波数変換: 1000 Hz 以下 linear (mel = hz * 3/200)、1000 Hz 超 log
//! - フィルタ形: 三角形 (mel-space で等間隔配置)
//! - 正規化: Slaney (面積正規化、`2 / (mel_freqs[i+2] - mel_freqs[i])`)
//!
//! Phase T.5 Vocos vocoder (`alice-tts-vocoder`) 側の primitive と数値一致を目指す。

/// Hz を mel scale に変換する (Slaney formula)。
///
/// 1000 Hz 以下は linear (`mel = hz * 3 / 200`)、1000 Hz 超は log 変換。
#[must_use]
pub fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_min: f32 = 0.0;
    let f_sp: f32 = 200.0 / 3.0;
    let min_log_hz: f32 = 1000.0;
    let min_log_mel: f32 = (min_log_hz - f_min) / f_sp;
    let logstep: f32 = 6.4_f32.ln() / 27.0;

    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        (hz - f_min) / f_sp
    }
}

/// mel scale を Hz に変換する (Slaney inverse)。
#[must_use]
pub fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_min: f32 = 0.0;
    let f_sp: f32 = 200.0 / 3.0;
    let min_log_hz: f32 = 1000.0;
    let min_log_mel: f32 = (min_log_hz - f_min) / f_sp;
    let logstep: f32 = 6.4_f32.ln() / 27.0;

    if mel >= min_log_mel {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    } else {
        f_min + f_sp * mel
    }
}

/// Mel filterbank 行列を生成する。
///
/// # 引数
///
/// - `n_mels`: mel channel 数 (通常 80)
/// - `n_fft`: FFT サイズ (STFT と一致必須)
/// - `sample_rate`: 入力波形のサンプリング周波数 (Hz)
/// - `fmin`: 最低周波数 (Hz、通常 0.0)
/// - `fmax`: 最高周波数 (Hz、通常 `sample_rate / 2` = ナイキスト周波数)
///
/// # 戻り値
///
/// `Vec<Vec<f32>>[n_mels][n_fft / 2 + 1]` の mel filter 行列。
/// Slaney 正規化済 (面積 = 1 相当)。
///
/// # Panics
///
/// - `n_mels == 0`
/// - `n_fft < 2`
/// - `fmin >= fmax`
/// - `fmax > sample_rate / 2` (ナイキスト超過)
#[must_use]
pub fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    assert!(n_mels > 0, "n_mels must be > 0");
    assert!(n_fft >= 2, "n_fft must be >= 2");
    assert!(fmin < fmax, "fmin must be < fmax");
    let nyquist = sample_rate as f32 / 2.0;
    assert!(fmax <= nyquist, "fmax {fmax} must be <= Nyquist {nyquist}");

    let n_freq = n_fft / 2 + 1;

    // FFT bin 周波数 (librosa の fft_frequencies 相当): linspace(0, sr/2, n_freq)
    let fftfreqs: Vec<f32> = (0..n_freq)
        .map(|i| nyquist * (i as f32) / (n_freq - 1) as f32)
        .collect();

    // mel 空間で n_mels + 2 個の等間隔点 (両端は境界)
    let min_mel = hz_to_mel_slaney(fmin);
    let max_mel = hz_to_mel_slaney(fmax);
    let mel_freqs: Vec<f32> = (0..=n_mels + 1)
        .map(|i| {
            let mel = min_mel + (max_mel - min_mel) * (i as f32) / (n_mels + 1) as f32;
            mel_to_hz_slaney(mel)
        })
        .collect();

    let mut weights = vec![vec![0.0_f32; n_freq]; n_mels];

    for i in 0..n_mels {
        let f_lower = mel_freqs[i];
        let f_center = mel_freqs[i + 1];
        let f_upper = mel_freqs[i + 2];
        let denom_lower = f_center - f_lower;
        let denom_upper = f_upper - f_center;

        for j in 0..n_freq {
            let f = fftfreqs[j];
            // 上り斜面
            let up = if denom_lower > 0.0 {
                (f - f_lower) / denom_lower
            } else {
                0.0
            };
            // 下り斜面
            let down = if denom_upper > 0.0 {
                (f_upper - f) / denom_upper
            } else {
                0.0
            };
            let w = up.min(down).max(0.0);
            weights[i][j] = w;
        }

        // Slaney 正規化: 2 / (mel_freqs[i+2] - mel_freqs[i])
        let enorm = 2.0 / (f_upper - f_lower);
        for w in &mut weights[i] {
            *w *= enorm;
        }
    }

    weights
}

/// magnitude spectrogram (`|X|`) に mel filterbank を適用して mel-spectrogram を得る。
///
/// power=False (magnitude 入力) の librosa `librosa.feature.melspectrogram` と互換。
///
/// # 引数
///
/// - `magnitude`: STFT magnitude (`Vec<Vec<f32>>[frames][n_freq]`)
/// - `mel_filters`: mel filterbank (`Vec<Vec<f32>>[n_mels][n_freq]`)
///
/// # 戻り値
///
/// `Vec<Vec<f32>>[n_mels][frames]` の mel spectrogram (transposed 済、audio_mel 形式に一致)。
///
/// # Panics
///
/// magnitude / mel_filters の n_freq 次元が不一致の場合。
#[must_use]
pub fn apply_mel_filters(magnitude: &[Vec<f32>], mel_filters: &[Vec<f32>]) -> Vec<Vec<f32>> {
    assert!(!magnitude.is_empty(), "magnitude must not be empty");
    assert!(!mel_filters.is_empty(), "mel_filters must not be empty");

    let n_freq = magnitude[0].len();
    let n_mels = mel_filters.len();
    let n_frames = magnitude.len();

    assert_eq!(
        mel_filters[0].len(),
        n_freq,
        "mel_filters n_freq={} does not match magnitude n_freq={n_freq}",
        mel_filters[0].len()
    );

    // [n_mels][frames] レイアウトで返す
    let mut mel_spec = vec![vec![0.0_f32; n_frames]; n_mels];

    for (m, filter) in mel_filters.iter().enumerate() {
        for (t, mag_frame) in magnitude.iter().enumerate() {
            let mut acc = 0.0_f32;
            for k in 0..n_freq {
                acc += filter[k] * mag_frame[k];
            }
            mel_spec[m][t] = acc;
        }
    }

    mel_spec
}

/// mel-spectrogram を log-mel-spectrogram に変換する (`log(max(mel, eps))`)。
///
/// # 引数
///
/// - `mel_spec`: mel spectrogram (`Vec<Vec<f32>>[n_mels][frames]`)
/// - `eps`: log 底値 (通常 1e-5)
///
/// # 戻り値
///
/// log 済 mel spectrogram (同 shape)。
#[must_use]
pub fn log_mel(mel_spec: &[Vec<f32>], eps: f32) -> Vec<Vec<f32>> {
    mel_spec
        .iter()
        .map(|row| row.iter().map(|&x| x.max(eps).ln()).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hz_to_mel_roundtrip() {
        for hz in [10.0_f32, 100.0, 500.0, 1000.0, 2000.0, 8000.0, 12000.0] {
            let mel = hz_to_mel_slaney(hz);
            let back = mel_to_hz_slaney(mel);
            assert!(
                (back - hz).abs() / hz < 1e-4,
                "hz={hz}, back={back}, mel={mel}"
            );
        }
    }

    #[test]
    fn hz_to_mel_monotonic() {
        let mut prev = hz_to_mel_slaney(1.0);
        for hz in [10.0_f32, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0] {
            let mel = hz_to_mel_slaney(hz);
            assert!(mel > prev, "hz={hz} mel={mel} prev={prev}");
            prev = mel;
        }
    }

    #[test]
    fn hz_to_mel_at_1000hz_break_point() {
        // Slaney: 1000 Hz は linear/log の境目、mel = 1000 / (200/3) = 15
        let mel = hz_to_mel_slaney(1000.0);
        assert!((mel - 15.0).abs() < 1e-4);
    }

    #[test]
    fn mel_filterbank_shape() {
        let filters = mel_filterbank(80, 1024, 24_000, 0.0, 12_000.0);
        assert_eq!(filters.len(), 80);
        assert_eq!(filters[0].len(), 1024 / 2 + 1);
    }

    #[test]
    fn mel_filterbank_values_non_negative() {
        let filters = mel_filterbank(80, 1024, 24_000, 0.0, 12_000.0);
        for filter in &filters {
            for &w in filter {
                assert!(w >= 0.0);
            }
        }
    }

    #[test]
    fn mel_filterbank_each_row_has_positive_sum() {
        let filters = mel_filterbank(80, 1024, 24_000, 0.0, 12_000.0);
        for (i, filter) in filters.iter().enumerate() {
            let sum: f32 = filter.iter().sum();
            assert!(sum > 0.0, "filter[{i}] sum={sum}");
        }
    }

    #[test]
    fn apply_mel_filters_shape() {
        let filters = mel_filterbank(80, 1024, 24_000, 0.0, 12_000.0);
        let n_frames = 10;
        let magnitude = vec![vec![1.0_f32; 513]; n_frames];
        let mel_spec = apply_mel_filters(&magnitude, &filters);
        assert_eq!(mel_spec.len(), 80);
        assert_eq!(mel_spec[0].len(), n_frames);
    }

    #[test]
    fn log_mel_never_below_log_eps() {
        let mel_spec = vec![vec![0.0_f32, 1e-10, 1.0, 100.0]];
        let eps = 1e-5;
        let logged = log_mel(&mel_spec, eps);
        assert_eq!(logged.len(), 1);
        assert!((logged[0][0] - eps.ln()).abs() < 1e-6);
        assert!((logged[0][1] - eps.ln()).abs() < 1e-6);
        assert!((logged[0][2] - 1.0_f32.ln()).abs() < 1e-6);
        assert!((logged[0][3] - 100.0_f32.ln()).abs() < 1e-4);
    }

    #[test]
    #[should_panic(expected = "n_mels must be > 0")]
    fn mel_filterbank_panics_on_zero_n_mels() {
        let _ = mel_filterbank(0, 1024, 24_000, 0.0, 12_000.0);
    }

    #[test]
    #[should_panic(expected = "fmax")]
    fn mel_filterbank_panics_on_fmax_over_nyquist() {
        // sr=24000, Nyquist=12000, fmax=20000 → panic
        let _ = mel_filterbank(80, 1024, 24_000, 0.0, 20_000.0);
    }
}
