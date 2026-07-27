//! TTS 学習用 audio feature 抽出モジュール (Feature Request R4)。
//!
//! wav 波形 (24 kHz mono f32) から以下 3 種類の特徴量を抽出する:
//!
//! - **mel-spectrogram**: STFT + Hann window + Slaney mel filterbank + log
//! - **F0 (fundamental frequency)**: YIN algorithm + V/UV flag
//! - **energy**: RMS energy (dB)
//!
//! # サブモジュール
//!
//! | Module | 内容 |
//! |---|---|
//! | [`stft`] | Short-Time Fourier Transform (Hann window + rustfft) |
//! | [`mel`] | Mel filterbank (Slaney scale、librosa htk=False 互換) |
//! | [`f0`] | YIN algorithm による F0 推定 |
//! | [`feature`] | `AudioFeatureExtractor` facade (上記 3 系統 + energy を統合) |
//!
//! # 例
//!
//! ```rust
//! # #[cfg(feature = "tts")] {
//! use alice_train::tts::AudioFeatureExtractor;
//!
//! let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
//! let wav = vec![0.0_f32; 24_000]; // 1 sec of silence @ 24 kHz
//!
//! let mel = extractor.extract_mel(&wav);
//! let (f0, voiced) = extractor.extract_f0(&wav);
//! let energy = extractor.extract_energy(&wav);
//!
//! assert_eq!(mel.len(), 80); // n_mels channels
//! assert_eq!(f0.len(), voiced.len());
//! assert_eq!(f0.len(), energy.len());
//! # }
//! ```

pub mod f0;
pub mod feature;
pub mod mel;
pub mod stft;

pub use feature::AudioFeatureExtractor;
