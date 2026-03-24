//! safetensors シャード読み込み — BF16/F16→FP32 変換。
//!
//! HuggingFace 形式の sharded safetensors モデルを mmap で読み込む。
//! `model.safetensors.index.json` のマッピングに従い、各シャードから
//! テンソルを取得して FP32 に変換する。

use memmap2::Mmap;
use safetensors::tensor::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// シャードインデックスの JSON 構造。
#[derive(Deserialize)]
struct SafetensorsIndex {
    /// テンソル名 → シャードファイル名のマッピング。
    weight_map: HashMap<String, String>,
}

/// sharded safetensors モデルローダー。
pub struct ShardedModel {
    /// モデルディレクトリ。
    model_dir: PathBuf,
    /// テンソル名 → シャードファイル名。
    weight_map: HashMap<String, String>,
    /// シャードファイル名 → mmap データ。
    shards: HashMap<String, Mmap>,
}

impl ShardedModel {
    /// モデルディレクトリからローダーを構築する。
    ///
    /// `model.safetensors.index.json` が存在すればシャードモデル、
    /// `model.safetensors` のみなら単一ファイルモデルとして扱う。
    ///
    /// # Errors
    ///
    /// ファイル読み込みエラー、JSON パースエラー時。
    pub fn open<P: AsRef<Path>>(model_dir: P) -> io::Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let index_path = model_dir.join("model.safetensors.index.json");

        let (weight_map, shard_files) = if index_path.exists() {
            // Sharded model
            let index_str = fs::read_to_string(&index_path)?;
            let index: SafetensorsIndex = serde_json::from_str(&index_str)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let shard_files: Vec<String> = index
                .weight_map
                .values()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            (index.weight_map, shard_files)
        } else {
            // Single file model
            let single = "model.safetensors".to_string();
            let single_path = model_dir.join(&single);
            if !single_path.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "safetensors ファイルが見つかりません: {}",
                        model_dir.display()
                    ),
                ));
            }
            // 単一ファイルの場合、weight_map は後で全テンソル名を列挙して構築
            (HashMap::new(), vec![single])
        };

        // 各シャードを mmap
        let mut shards = HashMap::with_capacity(shard_files.len());
        for shard_name in &shard_files {
            let shard_path = model_dir.join(shard_name);
            // シンボリックリンクを解決
            let resolved = fs::canonicalize(&shard_path).unwrap_or(shard_path.clone());
            let file = fs::File::open(&resolved).map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("シャード読み込み失敗: {}: {e}", resolved.display()),
                )
            })?;
            let mmap = unsafe { Mmap::map(&file)? };
            shards.insert(shard_name.clone(), mmap);
        }

        // 単一ファイルモデルの場合、weight_map を構築
        let weight_map = if weight_map.is_empty() {
            let single_name = "model.safetensors".to_string();
            let mmap = shards.get(&single_name).unwrap();
            let st = SafeTensors::deserialize(mmap).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("safetensors パースエラー: {e}"),
                )
            })?;
            st.names()
                .into_iter()
                .map(|name| (name.to_string(), single_name.clone()))
                .collect()
        } else {
            weight_map
        };

        eprintln!(
            "  ShardedModel: {} テンソル, {} シャード",
            weight_map.len(),
            shards.len()
        );

        Ok(Self {
            model_dir,
            weight_map,
            shards,
        })
    }

    /// 全テンソル名を返す。
    #[must_use]
    pub fn tensor_names(&self) -> Vec<String> {
        self.weight_map.keys().cloned().collect()
    }

    /// モデルディレクトリを返す。
    #[must_use]
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// テンソルを FP32 Vec として取得する。
    ///
    /// BF16/F16 は FP32 に変換する。F32/F64 はそのまま。
    /// テンソルが存在しない場合は `None`。
    #[must_use]
    pub fn get_tensor_f32(&self, name: &str) -> Option<Vec<f32>> {
        let shard_name = self.weight_map.get(name)?;
        let mmap = self.shards.get(shard_name)?;

        let st = SafeTensors::deserialize(mmap).ok()?;
        let tensor = st.tensor(name).ok()?;
        let data = tensor.data();

        use safetensors::Dtype;
        match tensor.dtype() {
            Dtype::BF16 => Some(bf16_bytes_to_f32(data)),
            Dtype::F16 => Some(f16_bytes_to_f32(data)),
            Dtype::F32 => Some(f32_bytes_to_vec(data)),
            Dtype::F64 => Some(f64_bytes_to_f32(data)),
            _ => {
                eprintln!("  警告: 未対応の dtype {:?} for {name}", tensor.dtype());
                None
            }
        }
    }

    /// テンソルの shape を取得する。
    #[must_use]
    pub fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        let shard_name = self.weight_map.get(name)?;
        let mmap = self.shards.get(shard_name)?;
        let st = SafeTensors::deserialize(mmap).ok()?;
        let tensor = st.tensor(name).ok()?;
        Some(tensor.shape().to_vec())
    }

    /// 指定シャードのページキャッシュを解放するようOSにヒントを送る。
    /// mmap データ自体は有効なまま（次回アクセス時にディスクから再読込）。
    #[cfg(unix)]
    pub fn advise_dontneed(&self, shard_name: &str) {
        if let Some(mmap) = self.shards.get(shard_name) {
            unsafe {
                libc::madvise(
                    mmap.as_ptr() as *mut libc::c_void,
                    mmap.len(),
                    libc::MADV_DONTNEED,
                );
            }
        }
    }

    /// 全シャードのページキャッシュを解放するようOSにヒントを送る。
    #[cfg(unix)]
    pub fn advise_dontneed_all(&self) {
        for (_, mmap) in &self.shards {
            unsafe {
                libc::madvise(
                    mmap.as_ptr() as *mut libc::c_void,
                    mmap.len(),
                    libc::MADV_DONTNEED,
                );
            }
        }
    }

    /// テンソル名からシャードファイル名を取得する。
    #[must_use]
    pub fn shard_for_tensor(&self, name: &str) -> Option<&str> {
        self.weight_map.get(name).map(String::as_str)
    }
}

/// BF16 バイト列 → FP32 Vec。
fn bf16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    let n = data.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    out
}

/// F16 バイト列 → FP32 Vec。
fn f16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    let n = data.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        out.push(f16_to_f32(bits));
    }
    out
}

/// IEEE 754 half-precision → f32。
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // ±0
            return f32::from_bits(sign << 31);
        }
        // Subnormal: normalize
        let mut m = mant;
        let mut e = 0i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (m << 13));
    }
    if exp == 31 {
        // Inf / NaN
        let exp32 = 0xFF_u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (mant << 13));
    }

    let exp32 = exp + 127 - 15;
    f32::from_bits((sign << 31) | (exp32 << 23) | (mant << 13))
}

/// F32 バイト列 → Vec<f32>。
fn f32_bytes_to_vec(data: &[u8]) -> Vec<f32> {
    let n = data.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bytes = [
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ];
        out.push(f32::from_le_bytes(bytes));
    }
    out
}

/// F64 バイト列 → FP32 Vec (精度落ち)。
fn f64_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    let n = data.len() / 8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&data[i * 8..(i + 1) * 8]);
        out.push(f64::from_le_bytes(bytes) as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_conversion() {
        // 1.0 in BF16 = 0x3F80
        let data = [0x80, 0x3F];
        let result = bf16_bytes_to_f32(&data);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bf16_zero() {
        let data = [0x00, 0x00];
        let result = bf16_bytes_to_f32(&data);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn bf16_negative() {
        // -1.0 in BF16 = 0xBF80
        let data = [0x80, 0xBF];
        let result = bf16_bytes_to_f32(&data);
        assert!((result[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn f16_conversion_one() {
        // 1.0 in F16 = 0x3C00
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn f16_conversion_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn f32_bytes_roundtrip() {
        let original = [1.5f32, -2.0, 0.0];
        let mut data = Vec::new();
        for &v in &original {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = f32_bytes_to_vec(&data);
        assert_eq!(result, original);
    }
}
