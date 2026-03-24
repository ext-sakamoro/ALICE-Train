//! L12: FP32 重みキャッシュ — BF16→FP32 変換を初回のみ実行。
//!
//! safetensors (BF16) → FP32 変換はレイヤーあたり ~500ms かかる。
//! 32層 × 500ms = 16秒/step が I/O で消費される。
//!
//! 初回起動時に FP32 バイナリをディスクに保存し、2回目以降は
//! FP32 ファイルを直接読み込む (BF16 デコード不要)。
//!
//! # フォーマット
//!
//! レイヤーの全フィールドを固定順序で生 f32 バイト列として連結。
//! サイズは config から確定的に計算できるためヘッダ不要。

use crate::qwen35::{
    DeltaNetLayerWeights, FullAttnLayerWeights, LayerType, Qwen35Config, Qwen35LayerWeights,
};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// FP32 キャッシュディレクトリパスを返す。
fn cache_dir(base_dir: &str) -> PathBuf {
    Path::new(base_dir).join("fp32_cache")
}

/// レイヤーキャッシュファイルパス。
fn layer_path(base_dir: &str, layer_idx: usize, layer_type: LayerType) -> PathBuf {
    let suffix = match layer_type {
        LayerType::LinearAttention => "dn",
        LayerType::FullAttention => "fa",
    };
    cache_dir(base_dir).join(format!("layer_{layer_idx}_{suffix}.fp32"))
}

/// キャッシュが全層存在するか確認。
#[must_use]
pub fn cache_exists(base_dir: &str, config: &Qwen35Config) -> bool {
    for i in 0..config.num_hidden_layers {
        let lt = config.layer_type(i);
        if !layer_path(base_dir, i, lt).exists() {
            return false;
        }
    }
    true
}

/// DeltaNet レイヤーを FP32 バイナリに保存。
fn save_deltanet_layer(w: &DeltaNetLayerWeights, path: &Path) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    let write_slice = |f: &mut fs::File, s: &[f32]| -> io::Result<()> {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len() * 4) };
        f.write_all(bytes)
    };
    write_slice(&mut f, &w.input_layernorm)?;
    write_slice(&mut f, &w.post_attn_layernorm)?;
    write_slice(&mut f, &w.in_proj_qkv)?;
    write_slice(&mut f, &w.in_proj_z)?;
    write_slice(&mut f, &w.in_proj_b)?;
    write_slice(&mut f, &w.in_proj_a)?;
    write_slice(&mut f, &w.a_log)?;
    write_slice(&mut f, &w.dt_bias)?;
    write_slice(&mut f, &w.conv1d_weight)?;
    write_slice(&mut f, &w.norm_weight)?;
    write_slice(&mut f, &w.out_proj)?;
    write_slice(&mut f, &w.gate_proj)?;
    write_slice(&mut f, &w.up_proj)?;
    write_slice(&mut f, &w.down_proj)?;
    Ok(())
}

/// DeltaNet レイヤーを FP32 バイナリから読み込み (mmap)。
fn load_deltanet_layer(path: &Path, config: &Qwen35Config) -> io::Result<DeltaNetLayerWeights> {
    let file = fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    #[allow(clippy::cast_ptr_alignment)]
    let floats: &[f32] =
        unsafe { std::slice::from_raw_parts(mmap.as_ptr().cast::<f32>(), mmap.len() / 4) };

    let h = config.hidden_size;
    let kd = config.linear_key_dim();
    let vd = config.linear_value_dim();
    let nv = config.linear_num_value_heads;
    let conv_dim = config.conv_dim();
    let ks = config.linear_conv_kernel_dim;
    let inter = config.intermediate_size;
    let vhd = config.linear_value_head_dim;

    let mut off = 0;
    let mut take = |n: usize| -> Vec<f32> {
        let s = floats[off..off + n].to_vec();
        off += n;
        s
    };

    Ok(DeltaNetLayerWeights {
        input_layernorm: take(h),
        post_attn_layernorm: take(h),
        in_proj_qkv: take(h * (kd * 2 + vd)),
        in_proj_z: take(h * vd),
        in_proj_b: take(h * nv),
        in_proj_a: take(h * nv),
        a_log: take(nv),
        dt_bias: take(nv),
        conv1d_weight: take(conv_dim * ks),
        norm_weight: take(vhd),
        out_proj: take(vd * h),
        gate_proj: take(inter * h),
        up_proj: take(inter * h),
        down_proj: take(h * inter),
    })
}

/// Full Attention レイヤーを FP32 バイナリに保存。
fn save_fullattn_layer(w: &FullAttnLayerWeights, path: &Path) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    let write_slice = |f: &mut fs::File, s: &[f32]| -> io::Result<()> {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len() * 4) };
        f.write_all(bytes)
    };
    write_slice(&mut f, &w.input_layernorm)?;
    write_slice(&mut f, &w.post_attn_layernorm)?;
    write_slice(&mut f, &w.q_proj)?;
    write_slice(&mut f, &w.k_proj)?;
    write_slice(&mut f, &w.v_proj)?;
    write_slice(&mut f, &w.o_proj)?;
    write_slice(&mut f, &w.q_norm)?;
    write_slice(&mut f, &w.k_norm)?;
    write_slice(&mut f, &w.gate_proj)?;
    write_slice(&mut f, &w.up_proj)?;
    write_slice(&mut f, &w.down_proj)?;
    Ok(())
}

/// Full Attention レイヤーを FP32 バイナリから読み込み (mmap)。
fn load_fullattn_layer(path: &Path, config: &Qwen35Config) -> io::Result<FullAttnLayerWeights> {
    let file = fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    #[allow(clippy::cast_ptr_alignment)]
    let floats: &[f32] =
        unsafe { std::slice::from_raw_parts(mmap.as_ptr().cast::<f32>(), mmap.len() / 4) };

    let h = config.hidden_size;
    let nh = config.num_attention_heads;
    let nkv = config.num_key_value_heads;
    let hd = config.head_dim;
    let inter = config.intermediate_size;

    let mut off = 0;
    let mut take = |n: usize| -> Vec<f32> {
        let s = floats[off..off + n].to_vec();
        off += n;
        s
    };

    Ok(FullAttnLayerWeights {
        input_layernorm: take(h),
        post_attn_layernorm: take(h),
        q_proj: take(h * nh * hd),
        k_proj: take(h * nkv * hd),
        v_proj: take(h * nkv * hd),
        o_proj: take(nh * hd * h),
        q_norm: take(hd),
        k_norm: take(hd),
        gate_proj: take(inter * h),
        up_proj: take(inter * h),
        down_proj: take(h * inter),
    })
}

/// safetensors から全層を FP32 キャッシュに変換・保存。
///
/// 1層ずつ処理するため、全層同時にRAMに載せる必要はない。
///
/// # Errors
///
/// テンソル読み込み失敗またはファイル書き込み失敗時に `io::Error` を返す。
pub fn build_cache(
    get_tensor: &dyn Fn(&str) -> Option<Vec<f32>>,
    weight_prefix: &str,
    base_dir: &str,
    config: &Qwen35Config,
) -> io::Result<()> {
    let dir = cache_dir(base_dir);
    fs::create_dir_all(&dir)?;

    for i in 0..config.num_hidden_layers {
        let lt = config.layer_type(i);
        let path = layer_path(base_dir, i, lt);
        let layer_prefix = format!("{weight_prefix}.layers.{i}");

        match lt {
            LayerType::LinearAttention => {
                let w = DeltaNetLayerWeights::from_tensors(&layer_prefix, get_tensor).ok_or_else(
                    || {
                        io::Error::new(
                            io::ErrorKind::NotFound,
                            format!("DeltaNet layer {i} テンソル読み込み失敗"),
                        )
                    },
                )?;
                save_deltanet_layer(&w, &path)?;
            }
            LayerType::FullAttention => {
                let w = FullAttnLayerWeights::from_tensors(&layer_prefix, get_tensor).ok_or_else(
                    || {
                        io::Error::new(
                            io::ErrorKind::NotFound,
                            format!("FullAttn layer {i} テンソル読み込み失敗"),
                        )
                    },
                )?;
                save_fullattn_layer(&w, &path)?;
            }
        }

        if (i + 1) % 8 == 0 || i == config.num_hidden_layers - 1 {
            eprintln!(
                "    FP32 キャッシュ: {}/{} 層変換完了",
                i + 1,
                config.num_hidden_layers
            );
        }
    }

    Ok(())
}

/// FP32 キャッシュからレイヤー重みを読み込み。
///
/// BF16 デコード不要 — fs::read + ポインタキャストのみ。
///
/// # Errors
///
/// ファイル読み込み失敗時に `io::Error` を返す。
pub fn load_layer_from_cache(
    base_dir: &str,
    layer_idx: usize,
    config: &Qwen35Config,
) -> io::Result<Qwen35LayerWeights> {
    let lt = config.layer_type(layer_idx);
    let path = layer_path(base_dir, layer_idx, lt);

    match lt {
        LayerType::LinearAttention => Ok(Qwen35LayerWeights::DeltaNet(load_deltanet_layer(
            &path, config,
        )?)),
        LayerType::FullAttention => Ok(Qwen35LayerWeights::FullAttention(load_fullattn_layer(
            &path, config,
        )?)),
    }
}

/// 更新済みレイヤー重みを FP32 キャッシュに書き戻す。
///
/// # Errors
///
/// ファイル書き込み失敗時に `io::Error` を返す。
pub fn save_layer_to_cache(
    base_dir: &str,
    layer_idx: usize,
    weights: &Qwen35LayerWeights,
    config: &Qwen35Config,
) -> io::Result<()> {
    let lt = config.layer_type(layer_idx);
    let path = layer_path(base_dir, layer_idx, lt);

    match weights {
        Qwen35LayerWeights::DeltaNet(w) => save_deltanet_layer(w, &path),
        Qwen35LayerWeights::FullAttention(w) => save_fullattn_layer(w, &path),
    }
}

/// FP32 キャッシュのディスク使用量を返す (bytes)。
#[must_use]
pub fn cache_size_bytes(config: &Qwen35Config) -> usize {
    let mut total = 0;
    for i in 0..config.num_hidden_layers {
        total += match config.layer_type(i) {
            LayerType::LinearAttention => config.deltanet_params_per_layer() * 4,
            LayerType::FullAttention => config.full_attn_params_per_layer() * 4,
        };
    }
    total
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_roundtrip_deltanet() {
        let config = Qwen35Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            rms_norm_eps: 1e-6,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            linear_key_head_dim: 4,
            linear_num_key_heads: 2,
            linear_value_head_dim: 4,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 2,
            full_attention_interval: 4,
            layer_types: vec![LayerType::LinearAttention],
        };

        let kd = config.linear_key_dim();
        let vd = config.linear_value_dim();
        let qkv_dim = kd * 2 + vd;
        let h = config.hidden_size;
        let nv = config.linear_num_value_heads;
        let inter = config.intermediate_size;

        let original = DeltaNetLayerWeights {
            input_layernorm: (0..h).map(|i| i as f32 * 0.1).collect(),
            post_attn_layernorm: vec![1.0; h],
            in_proj_qkv: vec![0.01; h * qkv_dim],
            in_proj_z: vec![0.02; h * vd],
            in_proj_b: vec![0.03; h * nv],
            in_proj_a: vec![0.04; h * nv],
            a_log: vec![0.1; nv],
            dt_bias: vec![1.0; nv],
            conv1d_weight: vec![0.5; config.conv_dim() * config.linear_conv_kernel_dim],
            norm_weight: vec![1.0; config.linear_value_head_dim],
            out_proj: vec![0.05; vd * h],
            gate_proj: vec![0.06; inter * h],
            up_proj: vec![0.07; inter * h],
            down_proj: vec![0.08; h * inter],
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_dn.fp32");

        save_deltanet_layer(&original, &path).unwrap();
        let loaded = load_deltanet_layer(&path, &config).unwrap();

        assert_eq!(original.input_layernorm, loaded.input_layernorm);
        assert_eq!(original.in_proj_qkv, loaded.in_proj_qkv);
        assert_eq!(original.a_log, loaded.a_log);
        assert_eq!(original.gate_proj, loaded.gate_proj);
    }

    #[test]
    fn test_cache_roundtrip_fullattn() {
        let config = Qwen35Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            rms_norm_eps: 1e-6,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            linear_key_head_dim: 4,
            linear_num_key_heads: 2,
            linear_value_head_dim: 4,
            linear_num_value_heads: 2,
            linear_conv_kernel_dim: 2,
            full_attention_interval: 4,
            layer_types: vec![LayerType::FullAttention],
        };

        let h = config.hidden_size;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let inter = config.intermediate_size;

        let original = FullAttnLayerWeights {
            input_layernorm: vec![1.0; h],
            post_attn_layernorm: vec![1.0; h],
            q_proj: vec![0.01; h * nh * hd],
            k_proj: vec![0.02; h * nkv * hd],
            v_proj: vec![0.03; h * nkv * hd],
            o_proj: vec![0.04; nh * hd * h],
            q_norm: vec![1.0; hd],
            k_norm: vec![1.0; hd],
            gate_proj: vec![0.05; inter * h],
            up_proj: vec![0.06; inter * h],
            down_proj: vec![0.07; h * inter],
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_fa.fp32");

        save_fullattn_layer(&original, &path).unwrap();
        let loaded = load_fullattn_layer(&path, &config).unwrap();

        assert_eq!(original.q_proj, loaded.q_proj);
        assert_eq!(original.q_norm, loaded.q_norm);
        assert_eq!(original.gate_proj, loaded.gate_proj);
    }
}
