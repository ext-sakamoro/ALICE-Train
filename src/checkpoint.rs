//! チェックポイント — 重み保存/復元。
//!
//! safetensors 互換を意識した軽量バイナリフォーマット:
//! `[8 bytes: header_len (LE u64)] [header_len bytes: JSON metadata] [raw f32 weights]`
//!
//! 外部依存なしで safetensors のワークフローに近い保存/復元を実現。

use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use std::path::Path;

/// マジックバイト列 — ALICE-Train チェックポイントであることを識別。
const MAGIC: &[u8; 8] = b"ALICETRN";

/// チェックポイントフォーマットのバージョン。
const FORMAT_VERSION: u32 = 1;

/// チェックポイントのメタデータ（JSON シリアライズ）。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMeta {
    /// フォーマットバージョン。
    pub version: u32,
    /// 保存時のエポック番号。
    pub epoch: usize,
    /// 保存時のステップ番号。
    pub step: usize,
    /// 保存時の損失値。
    pub loss: f32,
    /// 学習率。
    pub learning_rate: f32,
    /// 重みテンソルの要素数。
    pub weight_count: usize,
    /// Optimizer state の要素数（0 = なし）。
    pub optimizer_state_count: usize,
}

/// チェックポイントデータ。
#[derive(Clone, Debug)]
pub struct CheckpointData {
    /// メタデータ。
    pub meta: CheckpointMeta,
    /// 潜在 FP32 重み。
    pub weights: Vec<f32>,
    /// Optimizer state（Adam の m, v など）。空なら optimizer state なし。
    pub optimizer_state: Vec<f32>,
}

impl CheckpointData {
    /// 新しいチェックポイントデータを構築する。
    #[must_use]
    pub fn new(
        epoch: usize,
        step: usize,
        loss: f32,
        learning_rate: f32,
        weights: Vec<f32>,
        optimizer_state: Vec<f32>,
    ) -> Self {
        let meta = CheckpointMeta {
            version: FORMAT_VERSION,
            epoch,
            step,
            loss,
            learning_rate,
            weight_count: weights.len(),
            optimizer_state_count: optimizer_state.len(),
        };
        Self {
            meta,
            weights,
            optimizer_state,
        }
    }

    /// チェックポイントをバイナリにシリアライズする。
    ///
    /// フォーマット:
    /// 1. `ALICETRN` (8 bytes magic)
    /// 2. `header_len` (8 bytes LE u64)
    /// 3. JSON メタデータ (`header_len` bytes)
    /// 4. weights (weight_count * 4 bytes, LE f32)
    /// 5. optimizer_state (optimizer_state_count * 4 bytes, LE f32)
    ///
    /// # Errors
    ///
    /// I/O エラー時。
    pub fn save<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Magic
        writer.write_all(MAGIC)?;

        // Header (JSON metadata)
        let header = serde_json::to_vec(&self.meta)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let header_len = header.len() as u64;
        writer.write_all(&header_len.to_le_bytes())?;
        writer.write_all(&header)?;

        // Weights
        for &w in &self.weights {
            writer.write_all(&w.to_le_bytes())?;
        }

        // Optimizer state
        for &s in &self.optimizer_state {
            writer.write_all(&s.to_le_bytes())?;
        }

        Ok(())
    }

    /// ファイルに保存する。
    ///
    /// # Errors
    ///
    /// I/O エラー時。
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = io::BufWriter::new(file);
        self.save(&mut writer)
    }

    /// バイナリからデシリアライズする。
    ///
    /// # Errors
    ///
    /// - マジックバイト不一致
    /// - JSON パースエラー
    /// - データ不足
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        // Magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid magic bytes: not an ALICE-Train checkpoint",
            ));
        }

        // Header length
        let mut len_buf = [0u8; 8];
        reader.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf) as usize;

        // Header JSON
        let mut header_buf = vec![0u8; header_len];
        reader.read_exact(&mut header_buf)?;
        let meta: CheckpointMeta = serde_json::from_slice(&header_buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Weights
        let mut weights = vec![0.0f32; meta.weight_count];
        let mut f32_buf = [0u8; 4];
        for w in &mut weights {
            reader.read_exact(&mut f32_buf)?;
            *w = f32::from_le_bytes(f32_buf);
        }

        // Optimizer state
        let mut optimizer_state = vec![0.0f32; meta.optimizer_state_count];
        for s in &mut optimizer_state {
            reader.read_exact(&mut f32_buf)?;
            *s = f32::from_le_bytes(f32_buf);
        }

        Ok(Self {
            meta,
            weights,
            optimizer_state,
        })
    }

    /// ファイルから読み込む。
    ///
    /// # Errors
    ///
    /// I/O エラー時、フォーマット不正時。
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        Self::load(&mut io::BufReader::new(&mut file))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_checkpoint(n_weights: usize, n_opt: usize) -> CheckpointData {
        let weights: Vec<f32> = (0..n_weights).map(|i| i as f32 * 0.1).collect();
        let opt: Vec<f32> = (0..n_opt).map(|i| i as f32 * 0.01).collect();
        CheckpointData::new(5, 1000, 0.42, 0.001, weights, opt)
    }

    #[test]
    fn roundtrip_basic() {
        let ckpt = make_checkpoint(10, 0);
        let mut buf = Vec::new();
        ckpt.save(&mut buf).unwrap();

        let loaded = CheckpointData::load(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.meta.epoch, 5);
        assert_eq!(loaded.meta.step, 1000);
        assert!((loaded.meta.loss - 0.42).abs() < 1e-6);
        assert_eq!(loaded.weights.len(), 10);
        for (a, b) in ckpt.weights.iter().zip(loaded.weights.iter()) {
            assert!((a - b).abs() < 1e-8);
        }
    }

    #[test]
    fn roundtrip_with_optimizer_state() {
        let ckpt = make_checkpoint(8, 16);
        let mut buf = Vec::new();
        ckpt.save(&mut buf).unwrap();

        let loaded = CheckpointData::load(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.optimizer_state.len(), 16);
        for (a, b) in ckpt
            .optimizer_state
            .iter()
            .zip(loaded.optimizer_state.iter())
        {
            assert!((a - b).abs() < 1e-8);
        }
    }

    #[test]
    fn roundtrip_empty_weights() {
        let ckpt = make_checkpoint(0, 0);
        let mut buf = Vec::new();
        ckpt.save(&mut buf).unwrap();

        let loaded = CheckpointData::load(&mut Cursor::new(&buf)).unwrap();
        assert!(loaded.weights.is_empty());
        assert!(loaded.optimizer_state.is_empty());
    }

    #[test]
    fn roundtrip_large_weights() {
        let ckpt = make_checkpoint(100_000, 200_000);
        let mut buf = Vec::new();
        ckpt.save(&mut buf).unwrap();

        let loaded = CheckpointData::load(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.weights.len(), 100_000);
        assert_eq!(loaded.optimizer_state.len(), 200_000);
        // スポットチェック
        assert!((loaded.weights[99_999] - 9999.9).abs() < 0.1);
    }

    #[test]
    fn roundtrip_negative_weights() {
        let weights = vec![-1.5, 0.0, 1.5, f32::MIN, f32::MAX];
        let ckpt = CheckpointData::new(0, 0, 0.0, 0.0, weights.clone(), vec![]);
        let mut buf = Vec::new();
        ckpt.save(&mut buf).unwrap();

        let loaded = CheckpointData::load(&mut Cursor::new(&buf)).unwrap();
        for (a, b) in weights.iter().zip(loaded.weights.iter()) {
            assert!((a - b).abs() < 1e-8 || (a.is_infinite() && b.is_infinite()));
        }
    }

    #[test]
    fn invalid_magic_rejected() {
        let mut data = vec![0u8; 100];
        data[..8].copy_from_slice(b"BADMAGIC");
        let err = CheckpointData::load(&mut Cursor::new(&data));
        assert!(err.is_err());
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("magic"), "error should mention magic: {msg}");
    }

    #[test]
    fn truncated_data_errors() {
        let ckpt = make_checkpoint(10, 0);
        let mut buf = Vec::new();
        ckpt.save(&mut buf).unwrap();

        // 末尾を切り詰め
        buf.truncate(buf.len() - 10);
        let err = CheckpointData::load(&mut Cursor::new(&buf));
        assert!(err.is_err());
    }

    #[test]
    fn meta_fields_preserved() {
        let ckpt = CheckpointData::new(42, 99999, 0.123, 0.0042, vec![1.0], vec![2.0, 3.0]);
        let mut buf = Vec::new();
        ckpt.save(&mut buf).unwrap();

        let loaded = CheckpointData::load(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.meta.epoch, 42);
        assert_eq!(loaded.meta.step, 99999);
        assert!((loaded.meta.loss - 0.123).abs() < 1e-6);
        assert!((loaded.meta.learning_rate - 0.0042).abs() < 1e-6);
        assert_eq!(loaded.meta.weight_count, 1);
        assert_eq!(loaded.meta.optimizer_state_count, 2);
    }

    #[test]
    fn version_field_is_one() {
        let ckpt = make_checkpoint(1, 0);
        assert_eq!(ckpt.meta.version, 1);
    }

    #[test]
    fn save_to_file_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.ckpt");

        let ckpt = make_checkpoint(50, 100);
        ckpt.save_to_file(&path).unwrap();

        let loaded = CheckpointData::load_from_file(&path).unwrap();
        assert_eq!(loaded.meta.epoch, 5);
        assert_eq!(loaded.weights.len(), 50);
        assert_eq!(loaded.optimizer_state.len(), 100);
    }

    #[test]
    fn load_nonexistent_file_errors() {
        let err = CheckpointData::load_from_file("/tmp/nonexistent_alice_ckpt_12345.bin");
        assert!(err.is_err());
    }

    #[test]
    fn checkpoint_data_clone() {
        let ckpt = make_checkpoint(3, 2);
        let ckpt2 = ckpt.clone();
        assert_eq!(ckpt2.weights.len(), 3);
        assert_eq!(ckpt2.optimizer_state.len(), 2);
    }

    #[test]
    fn checkpoint_data_debug() {
        let ckpt = make_checkpoint(1, 0);
        let s = format!("{ckpt:?}");
        assert!(s.contains("CheckpointData"));
    }

    #[test]
    fn checkpoint_meta_debug() {
        let ckpt = make_checkpoint(1, 0);
        let s = format!("{:?}", ckpt.meta);
        assert!(s.contains("epoch"));
    }
}
