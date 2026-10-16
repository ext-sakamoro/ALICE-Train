//! The F0 (fundamental frequency) extractor as a plug-in
//!
//! [`crate::tts::AudioFeatureExtractor::extract_f0_with`] takes any type that
//! implements [`F0Extractor`], so a caller can supply another algorithm without this
//! crate depending on it. The built-in one is `tts::Yin` (feature `tts`), the same
//! YIN path as `AudioFeatureExtractor::extract_f0`.
//!
//! This module has no dependency and is compiled with every feature set, so a crate
//! that implements an extractor needs `alice-train` with `default-features = false`
//! only.

/// An F0 extractor: one F0 value (Hz, `0.0` when unvoiced) and one voiced flag per
/// frame
///
/// An implementation returns `1 + wav.len() / hop_length` frames, the frame count of
/// the rest of `AudioFeatureExtractor`, so the F0 lines up with the mel and energy
/// frames.
pub trait F0Extractor {
    /// F0 and voiced flags of `wav` (mono, `[-1.0, 1.0]`) sampled at `sample_rate` Hz,
    /// one frame every `hop_length` samples
    fn extract_f0(&self, wav: &[f32], sample_rate: u32, hop_length: usize)
        -> (Vec<f32>, Vec<bool>);
}
