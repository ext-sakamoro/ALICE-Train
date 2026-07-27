//! `TtsDataset` — manifest jsonl から `TtsBatch` を streaming で構築する loader。
//!
//! 全 entry を memory 保持しつつ、audio 実データは batch iterate 時に都度 WAV load + feature
//! extract する streaming 設計。50k utterance の JVS 全 audio (24 kHz mono 15 sec × 100 speaker
//! × 500 utt ≈ 72 GB) を memory に載せずに済む。
//!
//! # 例
//!
//! ```rust,no_run
//! # #[cfg(feature = "tts")] {
//! use alice_train::tts::{AudioFeatureExtractor, TtsDataset};
//!
//! let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
//! let dataset = TtsDataset::from_manifest("data/jvs.jsonl", "/data/jvs", extractor)
//!     .expect("load manifest");
//!
//! let (train, valid, test) = dataset.split((0.9, 0.05, 0.05), 42).expect("split");
//!
//! for batch_result in train.iter_batches(4) {
//!     let batch = batch_result.expect("batch");
//!     assert!(batch.batch_size() <= 4);
//! }
//! # }
//! ```

use super::manifest::{TtsDatasetError, TtsManifestEntry};
use crate::tts::audio::AudioFeatureExtractor;
use crate::tts::batch::TtsBatch;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// TTS 学習用 dataset。
///
/// Manifest jsonl から entry 列を保持し、iter_batches で batch 単位に yield する。
/// audio load + feature extract は各 batch 生成時に streaming 実行される。
#[derive(Clone, Debug)]
pub struct TtsDataset {
    entries: Vec<TtsManifestEntry>,
    audio_root: PathBuf,
    extractor: AudioFeatureExtractor,
}

impl TtsDataset {
    /// Manifest jsonl と audio_root を読み込んで dataset を構築する。
    ///
    /// # 引数
    ///
    /// - `manifest_path`: 1 行 1 sample の JSONL ファイル
    /// - `audio_root`: 各 entry の `audio_path` の basedir
    /// - `extractor`: audio feature 抽出器 (sample_rate / n_mels / hop_length は WAV 側と一致必須)
    ///
    /// # Errors
    ///
    /// - manifest ファイルが読めない ([`TtsDatasetError::Io`])
    /// - JSONL の 1 行が parse できない ([`TtsDatasetError::JsonParse`])
    /// - entry 内部が invalid ([`TtsDatasetError::InvalidEntry`])
    /// - 空 manifest ([`TtsDatasetError::EmptyManifest`])
    pub fn from_manifest(
        manifest_path: impl AsRef<Path>,
        audio_root: impl AsRef<Path>,
        extractor: AudioFeatureExtractor,
    ) -> Result<Self, TtsDatasetError> {
        let file = File::open(manifest_path.as_ref())?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for (i, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: TtsManifestEntry =
                serde_json::from_str(trimmed).map_err(|e| TtsDatasetError::JsonParse {
                    line: i + 1,
                    source: e,
                })?;
            entry.validate()?;
            entries.push(entry);
        }

        if entries.is_empty() {
            return Err(TtsDatasetError::EmptyManifest);
        }

        Ok(Self {
            entries,
            audio_root: audio_root.as_ref().to_path_buf(),
            extractor,
        })
    }

    /// Entry 数を返す。
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// dataset が空か。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// audio_root を返す。
    #[must_use]
    pub fn audio_root(&self) -> &Path {
        &self.audio_root
    }

    /// dataset を (train, valid, test) の 3 分割する。
    ///
    /// `seed` で決定論的 shuffle 後、比率に従って分割。合計 1.0 に正規化される。
    ///
    /// # Errors
    ///
    /// 比率のいずれかが負 or 合計が 0 の場合 ([`TtsDatasetError::InvalidSplitRatio`])。
    pub fn split(
        &self,
        ratios: (f32, f32, f32),
        seed: u64,
    ) -> Result<(Self, Self, Self), TtsDatasetError> {
        let (t, v, s) = ratios;
        if t < 0.0 || v < 0.0 || s < 0.0 || (t + v + s) <= 0.0 {
            return Err(TtsDatasetError::InvalidSplitRatio { ratios });
        }

        let total = t + v + s;
        let (n_train, n_valid) = {
            let n = self.entries.len();
            let n_t = ((t / total) * n as f32).round() as usize;
            let n_v = ((v / total) * n as f32).round() as usize;
            let n_t = n_t.min(n);
            let n_v = n_v.min(n - n_t);
            (n_t, n_v)
        };

        let mut shuffled: Vec<TtsManifestEntry> = self.entries.clone();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        shuffled.shuffle(&mut rng);

        let train_entries = shuffled[..n_train].to_vec();
        let valid_entries = shuffled[n_train..n_train + n_valid].to_vec();
        let test_entries = shuffled[n_train + n_valid..].to_vec();

        Ok((
            Self::from_parts(
                train_entries,
                self.audio_root.clone(),
                self.extractor.clone(),
            ),
            Self::from_parts(
                valid_entries,
                self.audio_root.clone(),
                self.extractor.clone(),
            ),
            Self::from_parts(
                test_entries,
                self.audio_root.clone(),
                self.extractor.clone(),
            ),
        ))
    }

    fn from_parts(
        entries: Vec<TtsManifestEntry>,
        audio_root: PathBuf,
        extractor: AudioFeatureExtractor,
    ) -> Self {
        Self {
            entries,
            audio_root,
            extractor,
        }
    }

    /// batch 単位で `TtsBatch` を yield する iterator。
    ///
    /// 各 batch 生成時に entry の audio を WAV load + feature extract し、
    /// max-length に padding して `TtsBatch` を返す。
    ///
    /// # 引数
    ///
    /// - `batch_size`: 1 batch あたりの entry 数 (最後の batch は端数)
    #[must_use]
    pub fn iter_batches(&self, batch_size: usize) -> TtsBatchIterator<'_> {
        TtsBatchIterator {
            dataset: self,
            batch_size: batch_size.max(1),
            cursor: 0,
        }
    }
}

/// [`TtsDataset::iter_batches`] が返す iterator。
pub struct TtsBatchIterator<'a> {
    dataset: &'a TtsDataset,
    batch_size: usize,
    cursor: usize,
}

impl Iterator for TtsBatchIterator<'_> {
    type Item = Result<TtsBatch, TtsDatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.dataset.entries.len() {
            return None;
        }
        let end = (self.cursor + self.batch_size).min(self.dataset.entries.len());
        let batch_entries = &self.dataset.entries[self.cursor..end];
        self.cursor = end;
        Some(build_batch(
            batch_entries,
            &self.dataset.audio_root,
            &self.dataset.extractor,
        ))
    }
}

/// entry 群から TtsBatch を構築する (audio load + feature extract + padding)。
fn build_batch(
    entries: &[TtsManifestEntry],
    audio_root: &Path,
    extractor: &AudioFeatureExtractor,
) -> Result<TtsBatch, TtsDatasetError> {
    if entries.is_empty() {
        return Err(TtsDatasetError::InvalidEntry {
            reason: "empty batch entries".to_string(),
        });
    }

    // Step 1: 各 entry の audio load + feature extract
    let mut per_sample = Vec::with_capacity(entries.len());
    for entry in entries {
        let path = audio_root.join(&entry.audio_path);
        let wav = load_wav_mono_f32(&path, extractor.sample_rate())?;
        let mel = extractor.extract_mel(&wav);
        let (f0, voiced) = extractor.extract_f0(&wav);
        let energy = extractor.extract_energy(&wav);

        // alignment_ms → frame index に変換
        let hop = extractor.hop_length() as f32;
        let sr = extractor.sample_rate() as f32;
        let alignment: Vec<usize> = entry
            .phoneme_alignment_ms
            .iter()
            .map(|&ms| ((ms as f32 * sr) / (hop * 1000.0)).round() as usize)
            .collect();

        per_sample.push(SampleFeatures {
            wav,
            mel,
            f0,
            voiced,
            energy,
            alignment,
            entry: entry.clone(),
        });
    }

    // Step 2: max-length 計算 + padding
    let max_wav = per_sample.iter().map(|s| s.wav.len()).max().unwrap_or(0);
    let max_frames = per_sample
        .iter()
        .map(|s| s.mel.first().map_or(0, Vec::len))
        .max()
        .unwrap_or(0);
    let max_seq = per_sample
        .iter()
        .map(|s| s.entry.text_input_ids.len())
        .max()
        .unwrap_or(0);
    let max_mora = per_sample
        .iter()
        .map(|s| s.entry.text_moras.len())
        .max()
        .unwrap_or(0);
    let max_phrase = per_sample
        .iter()
        .map(|s| s.entry.text_accent_types.len())
        .max()
        .unwrap_or(0);
    let n_mels = extractor.n_mels();

    let log_eps: f32 = 1e-5_f32.ln();

    let mut audio_waveform = Vec::with_capacity(per_sample.len());
    let mut audio_mel = Vec::with_capacity(per_sample.len());
    let mut audio_f0 = Vec::with_capacity(per_sample.len());
    let mut audio_voiced = Vec::with_capacity(per_sample.len());
    let mut audio_energy = Vec::with_capacity(per_sample.len());
    let mut text_input_ids = Vec::with_capacity(per_sample.len());
    let mut text_moras = Vec::with_capacity(per_sample.len());
    let mut text_accent_types = Vec::with_capacity(per_sample.len());
    let mut text_phoneme_alignment = Vec::with_capacity(per_sample.len());
    let mut speaker_id = Vec::with_capacity(per_sample.len());
    let mut durations_ms = Vec::with_capacity(per_sample.len());

    for s in per_sample {
        audio_waveform.push(pad_to(&s.wav, max_wav, 0.0_f32));

        // mel: [n_mels][frames]、frame 方向を padding、n_mels 方向は変わらず
        let mel_padded: Vec<Vec<f32>> = s
            .mel
            .iter()
            .map(|row| pad_to(row, max_frames, log_eps))
            .collect();
        // n_mels が足りない場合の 0 埋め (通常同じ)
        let mel_padded = if mel_padded.len() < n_mels {
            let mut extended = mel_padded;
            while extended.len() < n_mels {
                extended.push(vec![log_eps; max_frames]);
            }
            extended
        } else {
            mel_padded
        };
        audio_mel.push(mel_padded);

        audio_f0.push(pad_to(&s.f0, max_frames, 0.0_f32));
        audio_voiced.push(pad_to(&s.voiced, max_frames, false));
        audio_energy.push(pad_to(&s.energy, max_frames, -100.0_f32));

        text_input_ids.push(pad_to(&s.entry.text_input_ids, max_seq, 0_u32));
        text_moras.push(pad_to(&s.entry.text_moras, max_mora, 0_u8));
        text_accent_types.push(pad_to(&s.entry.text_accent_types, max_phrase, 0_u8));
        text_phoneme_alignment.push(pad_to(&s.alignment, max_mora, 0_usize));
        speaker_id.push(s.entry.speaker_id);
        durations_ms.push(pad_to(&s.entry.durations_ms, max_mora, 0_u32));
    }

    TtsBatch::new(
        audio_waveform,
        audio_mel,
        audio_f0,
        audio_voiced,
        audio_energy,
        text_input_ids,
        text_moras,
        text_accent_types,
        text_phoneme_alignment,
        speaker_id,
        durations_ms,
        n_mels,
    )
    .map_err(|e| TtsDatasetError::BatchConstruction {
        source: format!("{e}"),
    })
}

/// 1 sample 分の抽出済 feature (padding 前)。
struct SampleFeatures {
    wav: Vec<f32>,
    mel: Vec<Vec<f32>>,
    f0: Vec<f32>,
    voiced: Vec<bool>,
    energy: Vec<f32>,
    alignment: Vec<usize>,
    entry: TtsManifestEntry,
}

/// 汎用 padding: `src` を右側に `fill` で pad して `target_len` にする。
fn pad_to<T: Clone>(src: &[T], target_len: usize, fill: T) -> Vec<T> {
    if src.len() >= target_len {
        return src[..target_len].to_vec();
    }
    let mut out = Vec::with_capacity(target_len);
    out.extend_from_slice(src);
    out.extend(std::iter::repeat_n(fill, target_len - src.len()));
    out
}

/// WAV load: mono f32 に変換。stereo は L/R 平均、int は f32 [-1.0, 1.0] に正規化。
fn load_wav_mono_f32(path: &Path, expected_sr: u32) -> Result<Vec<f32>, TtsDatasetError> {
    let reader = hound::WavReader::open(path).map_err(|e| TtsDatasetError::WavFormat {
        path: path.display().to_string(),
        reason: format!("open failed: {e}"),
    })?;
    let spec = reader.spec();

    if spec.sample_rate != expected_sr {
        return Err(TtsDatasetError::WavFormat {
            path: path.display().to_string(),
            reason: format!(
                "sample_rate mismatch: WAV {} vs expected {expected_sr}",
                spec.sample_rate
            ),
        });
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| TtsDatasetError::WavFormat {
                path: path.display().to_string(),
                reason: format!("float sample read failed: {e}"),
            })?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if bits == 0 || bits > 32 {
                return Err(TtsDatasetError::WavFormat {
                    path: path.display().to_string(),
                    reason: format!("unsupported bits_per_sample: {bits}"),
                });
            }
            let max_val = (1_i64 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|r| r.map(|v| (v as f32) / max_val))
                .collect::<Result<_, _>>()
                .map_err(|e| TtsDatasetError::WavFormat {
                    path: path.display().to_string(),
                    reason: format!("int sample read failed: {e}"),
                })?
        }
    };

    // mono 化 (stereo → 平均)
    let mono: Vec<f32> = match spec.channels {
        1 => samples,
        2 => samples
            .chunks_exact(2)
            .map(|pair| 0.5 * (pair[0] + pair[1]))
            .collect(),
        n => {
            return Err(TtsDatasetError::WavFormat {
                path: path.display().to_string(),
                reason: format!("unsupported channel count: {n}"),
            });
        }
    };

    if mono.is_empty() {
        return Err(TtsDatasetError::WavFormat {
            path: path.display().to_string(),
            reason: "wav is empty".to_string(),
        });
    }

    Ok(mono)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// テスト用に短い WAV を生成する (Int16, 24 kHz, mono)。
    fn write_test_wav(path: &Path, samples: &[f32], sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
        for &s in samples {
            let v = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(v).expect("write sample");
        }
        writer.finalize().expect("finalize");
    }

    /// テスト用 manifest + 対応する WAV を tempdir に用意する。
    fn setup_test_dataset(n_entries: usize) -> (TempDir, PathBuf) {
        let tmp = TempDir::new().expect("tmpdir");
        let manifest_path = tmp.path().join("manifest.jsonl");
        let mut manifest_file = File::create(&manifest_path).expect("create manifest");

        for i in 0..n_entries {
            let wav_path = tmp.path().join(format!("sample_{i}.wav"));
            // 0.5 sec of 440 Hz sine @ 24 kHz
            let sr = 24_000_f32;
            let n_samples = 12_000; // 0.5 sec
            let samples: Vec<f32> = (0..n_samples)
                .map(|k| (2.0 * std::f32::consts::PI * 440.0 * (k as f32) / sr).sin() * 0.5)
                .collect();
            write_test_wav(&wav_path, &samples, 24_000);

            let entry = TtsManifestEntry {
                audio_path: format!("sample_{i}.wav"),
                text_input_ids: vec![1, 2, 3],
                text_moras: vec![0, 1],
                text_accent_types: vec![0],
                phoneme_alignment_ms: vec![0, 250],
                speaker_id: i as u32,
                durations_ms: vec![250, 250],
            };
            let json = serde_json::to_string(&entry).expect("serialize");
            writeln!(manifest_file, "{json}").expect("write manifest line");
        }
        manifest_file.sync_all().expect("sync");
        (tmp, manifest_path)
    }

    #[test]
    fn from_manifest_loads_entries() {
        let (tmp, manifest_path) = setup_test_dataset(5);
        let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let ds = TtsDataset::from_manifest(&manifest_path, tmp.path(), extractor).expect("load");
        assert_eq!(ds.len(), 5);
        assert!(!ds.is_empty());
    }

    #[test]
    fn from_manifest_empty_returns_error() {
        let tmp = TempDir::new().expect("tmpdir");
        let manifest_path = tmp.path().join("empty.jsonl");
        File::create(&manifest_path).expect("create");
        let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let err = TtsDataset::from_manifest(&manifest_path, tmp.path(), extractor)
            .expect_err("empty manifest must fail");
        assert!(matches!(err, TtsDatasetError::EmptyManifest));
    }

    #[test]
    fn from_manifest_invalid_json_returns_parse_error() {
        let tmp = TempDir::new().expect("tmpdir");
        let manifest_path = tmp.path().join("invalid.jsonl");
        let mut f = File::create(&manifest_path).expect("create");
        writeln!(f, "not valid json").expect("write");
        f.sync_all().expect("sync");
        let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let err = TtsDataset::from_manifest(&manifest_path, tmp.path(), extractor)
            .expect_err("invalid json must fail");
        assert!(matches!(err, TtsDatasetError::JsonParse { line: 1, .. }));
    }

    #[test]
    fn split_produces_correct_ratios() {
        let (tmp, manifest_path) = setup_test_dataset(100);
        let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let ds = TtsDataset::from_manifest(&manifest_path, tmp.path(), extractor).expect("load");
        let (train, valid, test) = ds.split((0.8, 0.1, 0.1), 42).expect("split");
        assert_eq!(train.len() + valid.len() + test.len(), 100);
        // ±1 の許容 (round() の丸め)
        assert!(train.len().abs_diff(80) <= 1);
        assert!(valid.len().abs_diff(10) <= 1);
        assert!(test.len().abs_diff(10) <= 1);
    }

    #[test]
    fn split_rejects_invalid_ratios() {
        let (tmp, manifest_path) = setup_test_dataset(10);
        let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let ds = TtsDataset::from_manifest(&manifest_path, tmp.path(), extractor).expect("load");
        let err = ds
            .split((-0.1, 0.5, 0.5), 42)
            .expect_err("negative ratio must fail");
        assert!(matches!(err, TtsDatasetError::InvalidSplitRatio { .. }));

        let err = ds
            .split((0.0, 0.0, 0.0), 42)
            .expect_err("zero sum must fail");
        assert!(matches!(err, TtsDatasetError::InvalidSplitRatio { .. }));
    }

    #[test]
    fn split_is_deterministic_with_seed() {
        let (tmp, manifest_path) = setup_test_dataset(30);
        let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let ds = TtsDataset::from_manifest(&manifest_path, tmp.path(), extractor).expect("load");
        let (t1, _, _) = ds.split((0.7, 0.2, 0.1), 42).expect("split1");
        let (t2, _, _) = ds.split((0.7, 0.2, 0.1), 42).expect("split2");
        assert_eq!(t1.len(), t2.len());
        for (a, b) in t1.entries.iter().zip(t2.entries.iter()) {
            assert_eq!(a.audio_path, b.audio_path);
        }
    }

    #[test]
    fn iter_batches_yields_batches_of_correct_size() {
        let (tmp, manifest_path) = setup_test_dataset(10);
        let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let ds = TtsDataset::from_manifest(&manifest_path, tmp.path(), extractor).expect("load");

        let batches: Vec<_> = ds.iter_batches(4).map(|r| r.expect("batch load")).collect();
        // 10 entries / batch_size 4 → 3 batch (4, 4, 2)
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].batch_size(), 4);
        assert_eq!(batches[1].batch_size(), 4);
        assert_eq!(batches[2].batch_size(), 2);
    }

    #[test]
    fn batch_frame_counts_are_consistent() {
        let (tmp, manifest_path) = setup_test_dataset(3);
        let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let ds = TtsDataset::from_manifest(&manifest_path, tmp.path(), extractor).expect("load");
        let batch = ds
            .iter_batches(3)
            .next()
            .expect("has batch")
            .expect("build");
        // 全 sample が同じ audio 長 (0.5 sec) → 同一 frame 数 → padding は元と一致
        assert_eq!(batch.batch_size(), 3);
        assert_eq!(batch.n_mels(), 80);

        let n_frames = batch.audio_mel()[0][0].len();
        for i in 0..batch.batch_size() {
            assert_eq!(batch.audio_mel()[i][0].len(), n_frames);
            assert_eq!(batch.audio_f0()[i].len(), n_frames);
            assert_eq!(batch.audio_voiced()[i].len(), n_frames);
            assert_eq!(batch.audio_energy()[i].len(), n_frames);
        }
    }

    #[test]
    fn wav_sample_rate_mismatch_returns_error() {
        let tmp = TempDir::new().expect("tmpdir");
        let manifest_path = tmp.path().join("manifest.jsonl");
        let wav_path = tmp.path().join("a.wav");

        // WAV は 16 kHz、extractor は 24 kHz → mismatch
        write_test_wav(&wav_path, &vec![0.1_f32; 8_000], 16_000);
        let entry = TtsManifestEntry {
            audio_path: "a.wav".to_string(),
            text_input_ids: vec![1],
            text_moras: vec![0],
            text_accent_types: vec![0],
            phoneme_alignment_ms: vec![0],
            speaker_id: 0,
            durations_ms: vec![100],
        };
        let mut f = File::create(&manifest_path).expect("create");
        writeln!(f, "{}", serde_json::to_string(&entry).unwrap()).expect("write");
        f.sync_all().expect("sync");

        let extractor = AudioFeatureExtractor::new(24_000, 1024, 256, 80);
        let ds = TtsDataset::from_manifest(&manifest_path, tmp.path(), extractor).expect("load");
        let err = ds
            .iter_batches(1)
            .next()
            .expect("has batch")
            .expect_err("sr mismatch must fail");
        assert!(matches!(err, TtsDatasetError::WavFormat { .. }));
    }

    #[test]
    fn pad_to_extends_shorter_slice() {
        let src = vec![1_u32, 2, 3];
        let out = pad_to(&src, 5, 0_u32);
        assert_eq!(out, vec![1, 2, 3, 0, 0]);
    }

    #[test]
    fn pad_to_truncates_longer_slice() {
        let src = vec![1_u32, 2, 3, 4, 5];
        let out = pad_to(&src, 3, 0_u32);
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn pad_to_returns_same_for_equal_len() {
        let src = vec![1_u32, 2, 3];
        let out = pad_to(&src, 3, 0_u32);
        assert_eq!(out, vec![1, 2, 3]);
    }
}
