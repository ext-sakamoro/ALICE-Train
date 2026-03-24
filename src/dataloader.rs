//! データローダー — メモリマップ済みトークンデータの読み込み。
//!
//! 7B クラスの学習では数 GB のトークンデータを扱う。
//! `memmap2` でファイルを mmap し、ゼロコピーでバッチを切り出す。
//!
//! # ファイルフォーマット
//!
//! raw u32 トークン ID の連続配列（リトルエンディアン）。
//! ヘッダなし — tokenizer の出力をそのまま書き込める。

use memmap2::Mmap;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::io;
use std::path::Path;

/// メモリマップ済みトークンデータセット。
///
/// ファイルを mmap で開き、u32 トークン ID として読み取る。
pub struct MmapDataset {
    /// mmap 本体。
    mmap: Mmap,
    /// トークン数。
    token_count: usize,
}

impl MmapDataset {
    /// ファイルからメモリマップ済みデータセットを開く。
    ///
    /// ファイルは raw u32 (LE) のトークン ID 配列であること。
    ///
    /// # Errors
    ///
    /// - ファイルが開けない場合
    /// - ファイルサイズが 4 の倍数でない場合
    ///
    /// # Safety
    ///
    /// `memmap2::Mmap` は外部プロセスによるファイル変更に対して unsafe だが、
    /// 学習データは読み取り専用と仮定する。
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len() as usize;

        if !file_len.is_multiple_of(4) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("file size ({file_len}) is not a multiple of 4 (u32)"),
            ));
        }

        let token_count = file_len / 4;

        if token_count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "empty dataset file",
            ));
        }

        // Safety: ファイルは読み取り専用、学習中に変更されないと仮定。
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self { mmap, token_count })
    }

    /// トークン数を返す。
    #[must_use]
    pub fn len(&self) -> usize {
        self.token_count
    }

    /// データセットが空かどうか（open 時に弾くため常に false）。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_count == 0
    }

    /// 指定範囲のトークンを取得する。
    ///
    /// # Panics
    ///
    /// `start + len > self.token_count` の場合。
    #[must_use]
    pub fn get_tokens(&self, start: usize, len: usize) -> Vec<u32> {
        assert!(
            start + len <= self.token_count,
            "range [{start}..{}] exceeds token_count {}",
            start + len,
            self.token_count
        );
        let mut tokens = vec![0u32; len];
        let bytes = &self.mmap[..];
        for (i, t) in tokens.iter_mut().enumerate() {
            let offset = (start + i) * 4;
            let chunk: [u8; 4] = [
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ];
            *t = u32::from_le_bytes(chunk);
        }
        tokens
    }
}

/// バッチイテレータの設定。
#[derive(Clone, Debug)]
pub struct DataLoaderConfig {
    /// シーケンス長（1サンプルのトークン数）。
    pub seq_len: usize,
    /// バッチサイズ。
    pub batch_size: usize,
    /// エポック毎にシャッフルするか。
    pub shuffle: bool,
    /// 乱数シード（再現性用）。
    pub seed: u64,
}

impl DataLoaderConfig {
    /// デフォルト設定（seq=512, batch=8, shuffle=true, seed=42）。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            seq_len: 512,
            batch_size: 8,
            shuffle: true,
            seed: 42,
        }
    }

    /// シーケンス長を設定。
    #[must_use]
    pub const fn with_seq_len(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// バッチサイズを設定。
    #[must_use]
    pub const fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// シャッフル設定。
    #[must_use]
    pub const fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// シード設定。
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// バッチデータ。
#[derive(Clone, Debug)]
pub struct Batch {
    /// 入力トークン: `[batch_size][seq_len]`（フラットに格納）。
    pub input_ids: Vec<u32>,
    /// ターゲットトークン（1つ右にシフト）: `[batch_size][seq_len]`。
    pub target_ids: Vec<u32>,
    /// バッチ内の実効サンプル数（最終バッチで `batch_size` 未満になる場合）。
    pub actual_batch_size: usize,
}

/// データローダー。
///
/// `MmapDataset` からシーケンスを切り出し、バッチ化する。
/// エポック毎にシャッフル可能。
pub struct DataLoader {
    /// サンプル開始インデックスのリスト。
    sample_indices: Vec<usize>,
    /// 設定。
    config: DataLoaderConfig,
    /// 乱数生成器。
    rng: rand::rngs::StdRng,
}

impl DataLoader {
    /// 新しいデータローダーを構築する。
    ///
    /// # Panics
    ///
    /// - `dataset` のトークン数が `seq_len + 1` 未満の場合
    ///   （ターゲット用に1トークン余分に必要）
    #[must_use]
    pub fn new(dataset: &MmapDataset, config: DataLoaderConfig) -> Self {
        let required = config.seq_len + 1; // input + 1-shifted target
        assert!(
            dataset.len() >= required,
            "dataset has {} tokens but seq_len+1={required} required",
            dataset.len()
        );

        // 非重複サンプルのインデックスを生成
        let n_samples = (dataset.len() - 1) / config.seq_len; // -1 for target shift
        let sample_indices: Vec<usize> = (0..n_samples).map(|i| i * config.seq_len).collect();

        let rng = rand::rngs::StdRng::seed_from_u64(config.seed);

        Self {
            sample_indices,
            config,
            rng,
        }
    }

    /// サンプル数を返す。
    #[must_use]
    pub fn num_samples(&self) -> usize {
        self.sample_indices.len()
    }

    /// バッチ数を返す（切り上げ）。
    #[must_use]
    pub fn num_batches(&self) -> usize {
        self.sample_indices.len().div_ceil(self.config.batch_size)
    }

    /// エポック開始時にシャッフルする。
    pub fn shuffle_epoch(&mut self) {
        if self.config.shuffle {
            self.sample_indices.shuffle(&mut self.rng);
        }
    }

    /// 指定バッチインデックスのバッチを取得する。
    ///
    /// # Panics
    ///
    /// `batch_idx >= num_batches()` の場合。
    #[must_use]
    pub fn get_batch(&self, batch_idx: usize, dataset: &MmapDataset) -> Batch {
        assert!(
            batch_idx < self.num_batches(),
            "batch_idx {batch_idx} >= num_batches {}",
            self.num_batches()
        );

        let start = batch_idx * self.config.batch_size;
        let end = (start + self.config.batch_size).min(self.sample_indices.len());
        let actual_batch_size = end - start;
        let seq_len = self.config.seq_len;

        let mut input_ids = vec![0u32; actual_batch_size * seq_len];
        let mut target_ids = vec![0u32; actual_batch_size * seq_len];

        for (b, &sample_start) in self.sample_indices[start..end].iter().enumerate() {
            // input: tokens[sample_start .. sample_start + seq_len]
            // target: tokens[sample_start + 1 .. sample_start + seq_len + 1]
            let tokens = dataset.get_tokens(sample_start, seq_len + 1);
            let offset = b * seq_len;
            input_ids[offset..offset + seq_len].copy_from_slice(&tokens[..seq_len]);
            target_ids[offset..offset + seq_len].copy_from_slice(&tokens[1..=seq_len]);
        }

        Batch {
            input_ids,
            target_ids,
            actual_batch_size,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as IoWrite;

    /// テスト用にトークンファイルを作成するヘルパー。
    fn create_token_file(dir: &tempfile::TempDir, tokens: &[u32]) -> std::path::PathBuf {
        let path = dir.path().join("tokens.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        for &t in tokens {
            f.write_all(&t.to_le_bytes()).unwrap();
        }
        path
    }

    // --- MmapDataset ---

    #[test]
    fn mmap_dataset_open_and_len() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..100).collect();
        let path = create_token_file(&dir, &tokens);

        let ds = MmapDataset::open(&path).unwrap();
        assert_eq!(ds.len(), 100);
        assert!(!ds.is_empty());
    }

    #[test]
    fn mmap_dataset_get_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..50).collect();
        let path = create_token_file(&dir, &tokens);

        let ds = MmapDataset::open(&path).unwrap();
        let slice = ds.get_tokens(10, 5);
        assert_eq!(slice, vec![10, 11, 12, 13, 14]);
    }

    #[test]
    fn mmap_dataset_get_tokens_start() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (100..110).collect();
        let path = create_token_file(&dir, &tokens);

        let ds = MmapDataset::open(&path).unwrap();
        let slice = ds.get_tokens(0, 3);
        assert_eq!(slice, vec![100, 101, 102]);
    }

    #[test]
    fn mmap_dataset_get_tokens_end() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..10).collect();
        let path = create_token_file(&dir, &tokens);

        let ds = MmapDataset::open(&path).unwrap();
        let slice = ds.get_tokens(7, 3);
        assert_eq!(slice, vec![7, 8, 9]);
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn mmap_dataset_get_tokens_out_of_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..10).collect();
        let path = create_token_file(&dir, &tokens);

        let ds = MmapDataset::open(&path).unwrap();
        let _ = ds.get_tokens(8, 5);
    }

    #[test]
    fn mmap_dataset_invalid_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.bin");
        // 5 bytes — not a multiple of 4
        std::fs::write(&path, &[0, 1, 2, 3, 4]).unwrap();

        let err = MmapDataset::open(&path);
        assert!(err.is_err());
    }

    #[test]
    fn mmap_dataset_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        std::fs::write(&path, &[]).unwrap();

        let err = MmapDataset::open(&path);
        assert!(err.is_err());
    }

    #[test]
    fn mmap_dataset_nonexistent_file() {
        let err = MmapDataset::open("/tmp/nonexistent_alice_tokens_12345.bin");
        assert!(err.is_err());
    }

    // --- DataLoaderConfig ---

    #[test]
    fn dataloader_config_default() {
        let c = DataLoaderConfig::default();
        assert_eq!(c.seq_len, 512);
        assert_eq!(c.batch_size, 8);
        assert!(c.shuffle);
        assert_eq!(c.seed, 42);
    }

    #[test]
    fn dataloader_config_builder() {
        let c = DataLoaderConfig::new()
            .with_seq_len(128)
            .with_batch_size(4)
            .with_shuffle(false)
            .with_seed(123);
        assert_eq!(c.seq_len, 128);
        assert_eq!(c.batch_size, 4);
        assert!(!c.shuffle);
        assert_eq!(c.seed, 123);
    }

    // --- DataLoader ---

    #[test]
    fn dataloader_num_samples_and_batches() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..105).collect(); // 104 usable, seq_len=10 → 10 samples
        let path = create_token_file(&dir, &tokens);
        let ds = MmapDataset::open(&path).unwrap();

        let config = DataLoaderConfig::new()
            .with_seq_len(10)
            .with_batch_size(3)
            .with_shuffle(false);
        let loader = DataLoader::new(&ds, config);
        assert_eq!(loader.num_samples(), 10); // (105-1)/10 = 10
        assert_eq!(loader.num_batches(), 4); // ceil(10/3) = 4
    }

    #[test]
    fn dataloader_get_batch_content() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..25).collect();
        let path = create_token_file(&dir, &tokens);
        let ds = MmapDataset::open(&path).unwrap();

        let config = DataLoaderConfig::new()
            .with_seq_len(4)
            .with_batch_size(2)
            .with_shuffle(false);
        let loader = DataLoader::new(&ds, config);

        let batch = loader.get_batch(0, &ds);
        assert_eq!(batch.actual_batch_size, 2);
        // sample 0: input=[0,1,2,3], target=[1,2,3,4]
        assert_eq!(&batch.input_ids[..4], &[0, 1, 2, 3]);
        assert_eq!(&batch.target_ids[..4], &[1, 2, 3, 4]);
        // sample 1: input=[4,5,6,7], target=[5,6,7,8]
        assert_eq!(&batch.input_ids[4..8], &[4, 5, 6, 7]);
        assert_eq!(&batch.target_ids[4..8], &[5, 6, 7, 8]);
    }

    #[test]
    fn dataloader_last_batch_partial() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..25).collect();
        let path = create_token_file(&dir, &tokens);
        let ds = MmapDataset::open(&path).unwrap();

        let config = DataLoaderConfig::new()
            .with_seq_len(4)
            .with_batch_size(4)
            .with_shuffle(false);
        let loader = DataLoader::new(&ds, config);
        // 6 samples, batch_size=4 → 2 batches, last has 2
        let last_batch = loader.get_batch(1, &ds);
        assert_eq!(last_batch.actual_batch_size, 2);
    }

    #[test]
    fn dataloader_shuffle_changes_order() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..105).collect();
        let path = create_token_file(&dir, &tokens);
        let ds = MmapDataset::open(&path).unwrap();

        let config = DataLoaderConfig::new()
            .with_seq_len(4)
            .with_batch_size(2)
            .with_shuffle(true)
            .with_seed(42);

        let mut loader = DataLoader::new(&ds, config);
        let before = loader.get_batch(0, &ds).input_ids.clone();
        loader.shuffle_epoch();
        let after = loader.get_batch(0, &ds).input_ids.clone();

        // シャッフル後に順序が変わる（確率的だが seed 固定で再現可能）
        // 少なくともサンプル数が多ければほぼ確実に変わる
        assert_ne!(before, after, "shuffle should change batch order");
    }

    #[test]
    fn dataloader_no_shuffle_preserves_order() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..50).collect();
        let path = create_token_file(&dir, &tokens);
        let ds = MmapDataset::open(&path).unwrap();

        let config = DataLoaderConfig::new()
            .with_seq_len(4)
            .with_batch_size(2)
            .with_shuffle(false);

        let mut loader = DataLoader::new(&ds, config);
        let before = loader.get_batch(0, &ds).input_ids.clone();
        loader.shuffle_epoch(); // shuffle=false なので変わらない
        let after = loader.get_batch(0, &ds).input_ids.clone();

        assert_eq!(before, after);
    }

    #[test]
    #[should_panic(expected = "batch_idx")]
    fn dataloader_batch_out_of_range() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..20).collect();
        let path = create_token_file(&dir, &tokens);
        let ds = MmapDataset::open(&path).unwrap();

        let config = DataLoaderConfig::new()
            .with_seq_len(4)
            .with_batch_size(2)
            .with_shuffle(false);
        let loader = DataLoader::new(&ds, config);
        let _ = loader.get_batch(100, &ds);
    }

    #[test]
    fn dataloader_deterministic_shuffle() {
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..200).collect();
        let path = create_token_file(&dir, &tokens);
        let ds = MmapDataset::open(&path).unwrap();

        let config = DataLoaderConfig::new()
            .with_seq_len(4)
            .with_batch_size(2)
            .with_seed(999);

        let mut loader1 = DataLoader::new(&ds, config.clone());
        loader1.shuffle_epoch();
        let batch1 = loader1.get_batch(0, &ds);

        let mut loader2 = DataLoader::new(&ds, config);
        loader2.shuffle_epoch();
        let batch2 = loader2.get_batch(0, &ds);

        assert_eq!(batch1.input_ids, batch2.input_ids, "same seed → same order");
    }

    #[test]
    fn batch_debug() {
        let batch = Batch {
            input_ids: vec![1, 2, 3],
            target_ids: vec![2, 3, 4],
            actual_batch_size: 1,
        };
        let s = format!("{batch:?}");
        assert!(s.contains("Batch"));
    }
}
