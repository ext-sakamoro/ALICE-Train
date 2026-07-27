//! `AudioFeatureExtractor` facade — wav 波形から mel/F0/energy を統合抽出。
//!
//! Feature Request R4 に対応する高レベル API。内部で [`super::stft`], [`super::mel`],
//! [`super::f0`] を組み合わせる。同一 `hop_length` に対して 3 種の特徴量が frame-align された
//! `Vec` として返る (frame 数 = `1 + wav.len() / hop_length`)。
//!
//! # 例
//!
//! ```rust
//! # #[cfg(feature = "tts")] {
//! use alice_train::tts::AudioFeatureExtractor;
//!
//! let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
//!
//! // 440 Hz sine 1 秒
//! let sr = 24_000_f32;
//! let wav: Vec<f32> = (0..24_000)
//!     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / sr).sin())
//!     .collect();
//!
//! let mel = extractor.extract_mel(&wav);
//! let (f0, voiced) = extractor.extract_f0(&wav);
//! let energy = extractor.extract_energy(&wav);
//!
//! let expected_frames = 1 + wav.len() / 256;
//! assert_eq!(mel.len(), 80);
//! assert_eq!(mel[0].len(), expected_frames);
//! assert_eq!(f0.len(), expected_frames);
//! assert_eq!(voiced.len(), expected_frames);
//! assert_eq!(energy.len(), expected_frames);
//! # }
//! ```

use super::{f0, mel, stft};

/// Audio feature 抽出 facade。
///
/// 学習前処理で `wav → mel/f0/energy` を統合抽出する高レベル API。
/// 全 method が同一 frame 数の Vec を返す (frame 数 = `1 + wav.len() / hop_length`)。
///
/// # Config
///
/// | Field | 用途 | 推奨値 |
/// |---|---|---|
/// | `sample_rate` | 入力波形の Hz | 24000 (ALICE-TTS v2.0 baseline) |
/// | `n_fft` | STFT FFT サイズ | 1024 |
/// | `hop_length` | STFT hop | 256 |
/// | `n_mels` | mel channel 数 | 80 (Vocos vocoder 準拠) |
///
/// F0 抽出は `f_min = 50 Hz`, `f_max = 550 Hz`, `frame_length = 2048`, `threshold = 0.15` を
/// 固定使用 (speech 用途、女性/男性声共通)。カスタマイズは [`Self::extract_f0_custom`] を使う。
#[derive(Clone, Debug)]
pub struct AudioFeatureExtractor {
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    mel_filters: Vec<Vec<f32>>,
}

impl AudioFeatureExtractor {
    /// 新しい抽出器を構築する。
    ///
    /// 内部で mel filterbank を precompute する (毎回計算しないための最適化)。
    ///
    /// # Panics
    ///
    /// [`mel::mel_filterbank`] と同条件で panic:
    /// - `n_mels == 0`
    /// - `n_fft < 2`
    /// - `sample_rate == 0`
    #[must_use]
    pub fn new(sample_rate: u32, n_fft: usize, hop_length: usize, n_mels: usize) -> Self {
        assert!(sample_rate > 0, "sample_rate must be > 0");
        let nyquist = sample_rate as f32 / 2.0;
        let mel_filters = mel::mel_filterbank(n_mels, n_fft, sample_rate, 0.0, nyquist);
        Self {
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            mel_filters,
        }
    }

    /// サンプリング周波数 (Hz)。
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// FFT サイズ。
    #[must_use]
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// hop サイズ。
    #[must_use]
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// mel channel 数。
    #[must_use]
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// mel-spectrogram (log-scale) を抽出する。
    ///
    /// STFT → magnitude → mel filter → log の順で処理。
    ///
    /// # 戻り値
    ///
    /// `Vec<Vec<f32>>[n_mels][frames]` の log-mel-spectrogram。
    #[must_use]
    pub fn extract_mel(&self, wav: &[f32]) -> Vec<Vec<f32>> {
        let spec = stft::stft(wav, self.n_fft, self.hop_length);
        let magnitude = stft::magnitude(&spec);
        let mel_spec = mel::apply_mel_filters(&magnitude, &self.mel_filters);
        mel::log_mel(&mel_spec, 1e-5)
    }

    /// F0 (fundamental frequency, Hz) と V/UV flag を抽出する。
    ///
    /// YIN algorithm を使用、speech 用 default パラメータ (f_min=50, f_max=550, threshold=0.15,
    /// frame_length=2048)。
    ///
    /// # 戻り値
    ///
    /// `(Vec<f32>, Vec<bool>)`: (F0 Hz, voiced flag)。unvoiced frame は F0=0.0。
    #[must_use]
    pub fn extract_f0(&self, wav: &[f32]) -> (Vec<f32>, Vec<bool>) {
        self.extract_f0_custom(wav, 2048, 50.0, 550.0, 0.15)
    }

    /// F0 抽出のカスタム版 (parameter 明示指定)。
    ///
    /// 男性声のみ (`f_max = 250`) / 女性声のみ (`f_min = 150`) / children (`f_max = 800`) 等で
    /// f_min/f_max を切り替える用途。
    #[must_use]
    pub fn extract_f0_custom(
        &self,
        wav: &[f32],
        frame_length: usize,
        f_min: f32,
        f_max: f32,
        threshold: f32,
    ) -> (Vec<f32>, Vec<bool>) {
        f0::yin_f0(
            wav,
            self.sample_rate,
            frame_length,
            self.hop_length,
            f_min,
            f_max,
            threshold,
        )
    }

    /// RMS energy (dB) を frame ごとに抽出する。
    ///
    /// 各 frame の RMS: `sqrt(mean(x²))`、dB: `20 * log10(max(rms, eps))`。
    /// center padding で STFT / F0 と frame 数を揃える (frame 数 = `1 + wav.len() / hop_length`)。
    ///
    /// # 戻り値
    ///
    /// `Vec<f32>[frames]` の dB energy。silence は `20 * log10(eps)` (通常 -100 dB 相当)。
    #[must_use]
    pub fn extract_energy(&self, wav: &[f32]) -> Vec<f32> {
        // frame_length = n_fft (STFT と同じ window で計算)
        let frame_length = self.n_fft;
        let pad = frame_length / 2;
        let padded = center_pad(wav, pad);
        let n_frames = 1 + wav.len() / self.hop_length;

        let eps: f32 = 1e-5;
        let mut energy = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let end = (start + frame_length).min(padded.len());
            if end <= start {
                energy.push(20.0 * eps.log10());
                continue;
            }
            let frame = &padded[start..end];
            let sum_sq: f32 = frame.iter().map(|&x| x * x).sum();
            let rms = (sum_sq / frame.len() as f32).sqrt();
            let db = 20.0 * rms.max(eps).log10();
            energy.push(db);
        }

        energy
    }
}

fn center_pad(wav: &[f32], pad: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(wav.len() + 2 * pad);
    out.extend(std::iter::repeat_n(0.0_f32, pad));
    out.extend_from_slice(wav);
    out.extend(std::iter::repeat_n(0.0_f32, pad));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extractor_config_getters_work() {
        let ex = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        assert_eq!(ex.sample_rate(), 24_000);
        assert_eq!(ex.n_fft(), 1024);
        assert_eq!(ex.hop_length(), 256);
        assert_eq!(ex.n_mels(), 80);
    }

    #[test]
    fn mel_extraction_shape_matches_batch_expectation() {
        let ex = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let wav = vec![0.0_f32; 24_000];
        let mel = ex.extract_mel(&wav);
        assert_eq!(mel.len(), 80);
        let expected_frames = 1 + 24_000 / 256;
        assert_eq!(mel[0].len(), expected_frames);
    }

    #[test]
    fn mel_of_silence_is_at_floor() {
        let ex = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let wav = vec![0.0_f32; 24_000];
        let mel = ex.extract_mel(&wav);
        // すべての mel が eps=1e-5 相当の log 値 ~= -11.51
        for row in &mel {
            for &v in row {
                assert!(
                    (v - 1e-5_f32.ln()).abs() < 1e-4,
                    "silence mel value {v} not near log(eps)={}",
                    1e-5_f32.ln()
                );
            }
        }
    }

    #[test]
    fn f0_extraction_agrees_with_yin() {
        let ex = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let sr = 24_000_f32;
        let freq = 220.0_f32;
        let wav: Vec<f32> = (0..24_000)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32) / sr).sin())
            .collect();

        let (f0, voiced) = ex.extract_f0(&wav);
        let mid = f0.len() / 2;
        assert!(voiced[mid]);
        let err = (f0[mid] - freq).abs() / freq;
        assert!(err < 0.03, "f0={} err={:.4}", f0[mid], err);
    }

    #[test]
    fn energy_of_silence_is_near_negative_100db() {
        let ex = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let wav = vec![0.0_f32; 24_000];
        let energy = ex.extract_energy(&wav);
        for e in &energy {
            // 20 * log10(1e-5) = -100 dB
            assert!((*e - (-100.0)).abs() < 1e-3, "energy={e}");
        }
    }

    #[test]
    fn energy_of_sine_is_positive_relative_to_silence() {
        let ex = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let sr = 24_000_f32;
        let wav: Vec<f32> = (0..24_000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / sr).sin())
            .collect();

        let energy = ex.extract_energy(&wav);
        let mid = energy.len() / 2;
        // sine 0.5 amplitude → RMS = 0.5/sqrt(2) ≈ 0.354 → 20*log10(0.354) ≈ -9 dB
        assert!(
            energy[mid] > -20.0,
            "energy={} should be well above silence floor",
            energy[mid]
        );
        assert!(energy[mid] < 0.0);
    }

    #[test]
    fn all_features_frame_count_matches() {
        let ex = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let sr = 24_000_f32;
        let wav: Vec<f32> = (0..24_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / sr).sin())
            .collect();

        let mel = ex.extract_mel(&wav);
        let (f0, voiced) = ex.extract_f0(&wav);
        let energy = ex.extract_energy(&wav);

        // 全 feature が同じ frame 数 → TtsBatch の検証条件を満たす
        let n_frames = mel[0].len();
        assert_eq!(f0.len(), n_frames);
        assert_eq!(voiced.len(), n_frames);
        assert_eq!(energy.len(), n_frames);
    }

    #[test]
    fn custom_f0_range_works() {
        let ex = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let sr = 24_000_f32;
        let wav: Vec<f32> = (0..24_000)
            .map(|i| (2.0 * std::f32::consts::PI * 120.0 * (i as f32) / sr).sin())
            .collect();

        // 男性声レンジ (50-250 Hz) で 120 Hz を検出
        let (f0, voiced) = ex.extract_f0_custom(&wav, 2048, 50.0, 250.0, 0.15);
        let mid = f0.len() / 2;
        assert!(voiced[mid]);
        let err = (f0[mid] - 120.0).abs() / 120.0;
        assert!(err < 0.03, "f0={} err={:.4}", f0[mid], err);
    }
}
