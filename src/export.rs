//! Ternary エクスポート — QAT 学習済み FP32 重みを .alice 形式に変換。
//!
//! # .alice ファイルフォーマット
//!
//! ```text
//! [8 bytes: "ALICEMOD" magic]
//! [4 bytes: format version (LE u32)]
//! [8 bytes: header_len (LE u64)]
//! [header_len bytes: JSON metadata (AliceModelMeta)]
//! [data section: binary tensors in fixed order]
//! ```
//!
//! ## データセクション順序
//!
//! 1. `embed_tokens` — BF16 `[vocab_size × hidden_size]`
//! 2. `output_norm` — FP32 `[hidden_size]`
//! 3. `lm_head` — BF16 `[vocab_size × hidden_size]` (tied=true なら省略)
//! 4. Layer 0..N-1: 量子化 projection (ternary packed) + 非量子化重み (FP32)
//!
//! ## Ternary パッキング
//!
//! 2 bits per value, 4 values per byte, LSB-first:
//! - `0b00` = 0
//! - `0b01` = +1
//! - `0b10` = -1

use crate::fp32_cache;
use crate::mixed_precision::{bf16_to_f32_vec, f32_to_bf16_vec, Bf16};
use crate::qwen35::{
    DeltaNetLayerWeights, FullAttnLayerWeights, LayerType, Qwen35Config, Qwen35LayerWeights,
};
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};

/// マジックバイト列。
const MAGIC: &[u8; 8] = b"ALICEMOD";

/// フォーマットバージョン。
const FORMAT_VERSION: u32 = 1;

/// Ternary エンコード: +1。
const TERNARY_POS: u8 = 0b01;
/// Ternary エンコード: -1。
const TERNARY_NEG: u8 = 0b10;
/// Ternary エンコード: 0。
const TERNARY_ZERO: u8 = 0b00;

// ── Metadata ─────────────────────────────────────────────────────────────

/// .alice ファイルのメタデータ。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AliceModelMeta {
    /// フォーマットバージョン。
    pub version: u32,
    /// モデル構成。
    pub config: Qwen35Config,
    /// 量子化方式。
    pub quantization: String,
    /// embed_tokens と lm_head が共有か。
    pub tied_embeddings: bool,
    /// 量子化対象パラメータ数。
    pub quantized_params: usize,
    /// 非量子化パラメータ数。
    pub non_quantized_params: usize,
    /// 総パラメータ数。
    pub total_params: usize,
    /// 元チェックポイントのステップ数。
    pub source_step: usize,
    /// 元チェックポイントの loss。
    pub source_loss: f32,
    /// 各レイヤーの projection 毎 scale factor。
    pub layer_scales: Vec<LayerScales>,
}

/// 1レイヤーの projection 毎 scale factor (γ = mean(|W|))。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerScales {
    /// レイヤーインデックス。
    pub layer_idx: usize,
    /// レイヤー種別。
    pub layer_type: String,
    /// projection 名 → scale の対応。
    pub scales: Vec<(String, f32)>,
}

// ── Ternary Packing ──────────────────────────────────────────────────────

/// FP32 重みを ternary {-1, 0, +1} に量子化し、パックする。
///
/// # Returns
///
/// `(scale, packed_bytes)` — scale は γ = mean(|W|)。
#[must_use]
pub fn quantize_and_pack(weights: &[f32]) -> (f32, Vec<u8>) {
    if weights.is_empty() {
        return (0.0, Vec::new());
    }

    // γ = mean(|W|)
    let sum_abs: f64 = weights.iter().map(|&w| w.abs() as f64).sum();
    let scale = (sum_abs / weights.len() as f64) as f32;
    let inv_scale = if scale > 1e-10 { 1.0 / scale } else { 0.0 };

    // 量子化: round(w / γ) → clamp(-1, 1)
    let ternary: Vec<i8> = weights
        .iter()
        .map(|&w| (w * inv_scale).round().clamp(-1.0, 1.0) as i8)
        .collect();

    // 2-bit パック (4 values per byte, LSB-first)
    let packed = pack_ternary(&ternary);
    (scale, packed)
}

/// Ternary 値列 ({-1, 0, +1}) を 2-bit パックする。
#[must_use]
pub fn pack_ternary(values: &[i8]) -> Vec<u8> {
    let num_bytes = values.len().div_ceil(4);
    let mut packed = vec![0u8; num_bytes];

    for (i, &v) in values.iter().enumerate() {
        let byte_idx = i / 4;
        let bit_pos = (i % 4) * 2;
        let code = match v {
            1 => TERNARY_POS,
            -1 => TERNARY_NEG,
            _ => TERNARY_ZERO,
        };
        packed[byte_idx] |= code << bit_pos;
    }

    packed
}

/// 2-bit パックから ternary 値列を復元する。
#[must_use]
pub fn unpack_ternary(packed: &[u8], count: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(count);

    for i in 0..count {
        let byte_idx = i / 4;
        let bit_pos = (i % 4) * 2;
        let code = (packed[byte_idx] >> bit_pos) & 0b11;
        let v = match code {
            TERNARY_POS => 1,
            TERNARY_NEG => -1,
            _ => 0,
        };
        values.push(v);
    }

    values
}

/// パック済み ternary + scale から FP32 重みを復元する。
#[must_use]
pub fn dequantize(packed: &[u8], count: usize, scale: f32) -> Vec<f32> {
    let ternary = unpack_ternary(packed, count);
    ternary.iter().map(|&v| v as f32 * scale).collect()
}

// ── Layer Quantization ───────────────────────────────────────────────────

/// DeltaNet レイヤーの量子化結果。
struct QuantizedDeltaNetLayer {
    /// 量子化済み projection: (name, scale, packed_bytes)。
    projections: Vec<(String, f32, Vec<u8>)>,
    /// 非量子化重み (FP32): layernorm, a_log, dt_bias, conv1d, norm。
    fp32_weights: DeltaNetFp32,
}

/// DeltaNet の非量子化 FP32 重み。
struct DeltaNetFp32 {
    input_layernorm: Vec<f32>,
    post_attn_layernorm: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    conv1d_weight: Vec<f32>,
    norm_weight: Vec<f32>,
}

/// FullAttn レイヤーの量子化結果。
struct QuantizedFullAttnLayer {
    /// 量子化済み projection: (name, scale, packed_bytes)。
    projections: Vec<(String, f32, Vec<u8>)>,
    /// 非量子化重み (FP32): layernorm, q_norm, k_norm。
    fp32_weights: FullAttnFp32,
}

/// FullAttn の非量子化 FP32 重み。
struct FullAttnFp32 {
    input_layernorm: Vec<f32>,
    post_attn_layernorm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

/// DeltaNet レイヤーを量子化する。
fn quantize_deltanet_layer(w: &DeltaNetLayerWeights) -> QuantizedDeltaNetLayer {
    let mut projections = Vec::new();

    let proj_list: Vec<(&str, &[f32])> = vec![
        ("in_proj_qkv", &w.in_proj_qkv),
        ("in_proj_z", &w.in_proj_z),
        ("in_proj_b", &w.in_proj_b),
        ("in_proj_a", &w.in_proj_a),
        ("out_proj", &w.out_proj),
        ("gate_proj", &w.gate_proj),
        ("up_proj", &w.up_proj),
        ("down_proj", &w.down_proj),
    ];

    for (name, data) in proj_list {
        let (scale, packed) = quantize_and_pack(data);
        projections.push((name.to_string(), scale, packed));
    }

    QuantizedDeltaNetLayer {
        projections,
        fp32_weights: DeltaNetFp32 {
            input_layernorm: w.input_layernorm.clone(),
            post_attn_layernorm: w.post_attn_layernorm.clone(),
            a_log: w.a_log.clone(),
            dt_bias: w.dt_bias.clone(),
            conv1d_weight: w.conv1d_weight.clone(),
            norm_weight: w.norm_weight.clone(),
        },
    }
}

/// FullAttn レイヤーを量子化する。
fn quantize_fullattn_layer(w: &FullAttnLayerWeights) -> QuantizedFullAttnLayer {
    let mut projections = Vec::new();

    let proj_list: Vec<(&str, &[f32])> = vec![
        ("q_proj", &w.q_proj),
        ("k_proj", &w.k_proj),
        ("v_proj", &w.v_proj),
        ("o_proj", &w.o_proj),
        ("gate_proj", &w.gate_proj),
        ("up_proj", &w.up_proj),
        ("down_proj", &w.down_proj),
    ];

    for (name, data) in proj_list {
        let (scale, packed) = quantize_and_pack(data);
        projections.push((name.to_string(), scale, packed));
    }

    QuantizedFullAttnLayer {
        projections,
        fp32_weights: FullAttnFp32 {
            input_layernorm: w.input_layernorm.clone(),
            post_attn_layernorm: w.post_attn_layernorm.clone(),
            q_norm: w.q_norm.clone(),
            k_norm: w.k_norm.clone(),
        },
    }
}

// ── File Writer ──────────────────────────────────────────────────────────

/// .alice ファイルにエクスポートする。
///
/// # 引数
///
/// - `writer`: 出力先
/// - `config`: モデル構成
/// - `cache_dir`: fp32_cache ディレクトリ (チェックポイントdir)
/// - `embedding`: embed_tokens (FP32)
/// - `output_norm`: output layernorm (FP32)
/// - `lm_head`: lm_head (FP32, None = tied)
/// - `source_step`: 元チェックポイントのステップ
/// - `source_loss`: 元チェックポイントの loss
///
/// # Errors
///
/// I/O エラー時。
#[allow(clippy::too_many_arguments)]
pub fn export_alice_model<W: Write>(
    writer: &mut W,
    config: &Qwen35Config,
    cache_dir: &str,
    embedding: &[f32],
    output_norm: &[f32],
    lm_head: Option<&[f32]>,
    source_step: usize,
    source_loss: f32,
) -> io::Result<ExportStats> {
    let tied = lm_head.is_none();
    let num_layers = config.num_hidden_layers;

    // 全レイヤーを量子化して統計を収集
    let mut layer_scales = Vec::with_capacity(num_layers);
    let mut quantized_layers: Vec<QuantizedLayer> = Vec::with_capacity(num_layers);
    let mut total_quantized = 0usize;
    let mut total_non_quantized = 0usize;

    for i in 0..num_layers {
        let weights = fp32_cache::load_layer_from_cache(cache_dir, i, config)?;

        let (ql, scales) = match weights {
            Qwen35LayerWeights::DeltaNet(ref w) => {
                let q = quantize_deltanet_layer(w);
                let scales: Vec<(String, f32)> = q
                    .projections
                    .iter()
                    .map(|(n, s, _)| (n.clone(), *s))
                    .collect();
                let quantized_count: usize = q
                    .projections
                    .iter()
                    .map(|(_, _, p)| p.len() * 4) // 4 values per byte
                    .sum();
                let fp32_count = q.fp32_weights.input_layernorm.len()
                    + q.fp32_weights.post_attn_layernorm.len()
                    + q.fp32_weights.a_log.len()
                    + q.fp32_weights.dt_bias.len()
                    + q.fp32_weights.conv1d_weight.len()
                    + q.fp32_weights.norm_weight.len();
                total_quantized += quantized_count;
                total_non_quantized += fp32_count;
                (QuantizedLayer::DeltaNet(q), scales)
            }
            Qwen35LayerWeights::FullAttention(ref w) => {
                let q = quantize_fullattn_layer(w);
                let scales: Vec<(String, f32)> = q
                    .projections
                    .iter()
                    .map(|(n, s, _)| (n.clone(), *s))
                    .collect();
                let quantized_count: usize =
                    q.projections.iter().map(|(_, _, p)| p.len() * 4).sum();
                let fp32_count = q.fp32_weights.input_layernorm.len()
                    + q.fp32_weights.post_attn_layernorm.len()
                    + q.fp32_weights.q_norm.len()
                    + q.fp32_weights.k_norm.len();
                total_quantized += quantized_count;
                total_non_quantized += fp32_count;
                (QuantizedLayer::FullAttention(q), scales)
            }
        };

        let lt_str = match config.layer_type(i) {
            LayerType::LinearAttention => "deltanet",
            LayerType::FullAttention => "fullattn",
        };
        layer_scales.push(LayerScales {
            layer_idx: i,
            layer_type: lt_str.to_string(),
            scales,
        });

        quantized_layers.push(ql);

        if (i + 1) % 8 == 0 || i == num_layers - 1 {
            eprintln!("  量子化: {}/{} 層完了", i + 1, num_layers);
        }
    }

    // Embedding + lm_head パラメータ数
    let embed_params = embedding.len();
    let lm_head_params = if tied { 0 } else { lm_head.unwrap().len() };
    let norm_params = output_norm.len();

    total_non_quantized += embed_params + lm_head_params + norm_params;

    // メタデータ
    let meta = AliceModelMeta {
        version: FORMAT_VERSION,
        config: config.clone(),
        quantization: "ternary_1_58bit".to_string(),
        tied_embeddings: tied,
        quantized_params: total_quantized,
        non_quantized_params: total_non_quantized,
        total_params: total_quantized + total_non_quantized,
        source_step,
        source_loss,
        layer_scales,
    };

    // ── Write ──

    // 1. Magic
    writer.write_all(MAGIC)?;

    // 2. Version
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;

    // 3. Header (JSON)
    let header_json =
        serde_json::to_vec(&meta).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let header_len = header_json.len() as u64;
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&header_json)?;

    // 4. Embedding (BF16)
    let embed_bf16 = f32_to_bf16_vec(embedding);
    write_bf16_slice(writer, &embed_bf16)?;

    // 5. Output norm (FP32)
    write_f32_slice(writer, output_norm)?;

    // 6. lm_head (BF16, tied なら省略)
    if let Some(lm) = lm_head {
        let lm_bf16 = f32_to_bf16_vec(lm);
        write_bf16_slice(writer, &lm_bf16)?;
    }

    // 7. Layers
    for ql in &quantized_layers {
        match ql {
            QuantizedLayer::DeltaNet(q) => {
                write_quantized_projections(writer, &q.projections)?;
                write_f32_slice(writer, &q.fp32_weights.input_layernorm)?;
                write_f32_slice(writer, &q.fp32_weights.post_attn_layernorm)?;
                write_f32_slice(writer, &q.fp32_weights.a_log)?;
                write_f32_slice(writer, &q.fp32_weights.dt_bias)?;
                write_f32_slice(writer, &q.fp32_weights.conv1d_weight)?;
                write_f32_slice(writer, &q.fp32_weights.norm_weight)?;
            }
            QuantizedLayer::FullAttention(q) => {
                write_quantized_projections(writer, &q.projections)?;
                write_f32_slice(writer, &q.fp32_weights.input_layernorm)?;
                write_f32_slice(writer, &q.fp32_weights.post_attn_layernorm)?;
                write_f32_slice(writer, &q.fp32_weights.q_norm)?;
                write_f32_slice(writer, &q.fp32_weights.k_norm)?;
            }
        }
    }

    // 統計
    let embed_bytes = embed_bf16.len() * 2;
    let norm_bytes = output_norm.len() * 4;
    let lm_head_bytes = if tied { 0 } else { lm_head.unwrap().len() * 2 };
    let mut ternary_bytes = 0usize;
    let mut layer_fp32_bytes = 0usize;
    for ql in &quantized_layers {
        match ql {
            QuantizedLayer::DeltaNet(q) => {
                for (_, _, p) in &q.projections {
                    ternary_bytes += 4 + p.len(); // scale + packed
                }
                layer_fp32_bytes += (q.fp32_weights.input_layernorm.len()
                    + q.fp32_weights.post_attn_layernorm.len()
                    + q.fp32_weights.a_log.len()
                    + q.fp32_weights.dt_bias.len()
                    + q.fp32_weights.conv1d_weight.len()
                    + q.fp32_weights.norm_weight.len())
                    * 4;
            }
            QuantizedLayer::FullAttention(q) => {
                for (_, _, p) in &q.projections {
                    ternary_bytes += 4 + p.len();
                }
                layer_fp32_bytes += (q.fp32_weights.input_layernorm.len()
                    + q.fp32_weights.post_attn_layernorm.len()
                    + q.fp32_weights.q_norm.len()
                    + q.fp32_weights.k_norm.len())
                    * 4;
            }
        }
    }

    let total_bytes = 8
        + 4
        + 8
        + header_json.len()
        + embed_bytes
        + norm_bytes
        + lm_head_bytes
        + ternary_bytes
        + layer_fp32_bytes;

    Ok(ExportStats {
        total_bytes,
        embed_bytes,
        ternary_bytes,
        layer_fp32_bytes,
        lm_head_bytes,
        quantized_params: total_quantized,
        meta,
    })
}

/// エクスポート統計。
pub struct ExportStats {
    /// 総ファイルサイズ (bytes)。
    pub total_bytes: usize,
    /// Embedding セクションサイズ (bytes)。
    pub embed_bytes: usize,
    /// Ternary パックセクションサイズ (bytes)。
    pub ternary_bytes: usize,
    /// レイヤー FP32 セクションサイズ (bytes)。
    pub layer_fp32_bytes: usize,
    /// lm_head セクションサイズ (bytes)。
    pub lm_head_bytes: usize,
    /// 量子化済みパラメータ数。
    pub quantized_params: usize,
    /// メタデータ。
    pub meta: AliceModelMeta,
}

/// 量子化レイヤーの列挙。
enum QuantizedLayer {
    DeltaNet(QuantizedDeltaNetLayer),
    FullAttention(QuantizedFullAttnLayer),
}

// ── File Reader ──────────────────────────────────────────────────────────

/// .alice ファイルのメタデータのみ読み込む。
///
/// # Errors
///
/// マジック不一致、JSON パースエラー時。
pub fn read_alice_meta<R: Read>(reader: &mut R) -> io::Result<AliceModelMeta> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid magic: expected ALICEMOD, got {:?}", &magic),
        ));
    }

    let mut version_buf = [0u8; 4];
    reader.read_exact(&mut version_buf)?;
    let version = u32::from_le_bytes(version_buf);
    if version != FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported version: {version}"),
        ));
    }

    let mut header_len_buf = [0u8; 8];
    reader.read_exact(&mut header_len_buf)?;
    let header_len = u64::from_le_bytes(header_len_buf) as usize;

    let mut header_buf = vec![0u8; header_len];
    reader.read_exact(&mut header_buf)?;

    serde_json::from_slice(&header_buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// .alice ファイルから embedding を BF16 で読み込み FP32 に変換。
///
/// reader は header 直後の位置にあること。
///
/// # Errors
///
/// I/O エラー時。
pub fn read_embedding<R: Read>(
    reader: &mut R,
    vocab_size: usize,
    hidden_size: usize,
) -> io::Result<Vec<f32>> {
    let count = vocab_size * hidden_size;
    let bf16 = read_bf16_slice(reader, count)?;
    Ok(bf16_to_f32_vec(&bf16))
}

/// .alice ファイルから output_norm を読み込み。
///
/// # Errors
///
/// I/O エラー時。
pub fn read_output_norm<R: Read>(reader: &mut R, hidden_size: usize) -> io::Result<Vec<f32>> {
    read_f32_slice(reader, hidden_size)
}

/// .alice ファイルから lm_head を読み込み (tied なら None)。
///
/// # Errors
///
/// I/O エラー時。
pub fn read_lm_head<R: Read>(
    reader: &mut R,
    tied: bool,
    vocab_size: usize,
    hidden_size: usize,
) -> io::Result<Option<Vec<f32>>> {
    if tied {
        return Ok(None);
    }
    let count = vocab_size * hidden_size;
    let bf16 = read_bf16_slice(reader, count)?;
    Ok(Some(bf16_to_f32_vec(&bf16)))
}

/// .alice ファイルから DeltaNet レイヤーの量子化重みを読み込み、FP32 に復元。
///
/// # Errors
///
/// I/O エラー時。
pub fn read_deltanet_layer<R: Read>(
    reader: &mut R,
    config: &Qwen35Config,
) -> io::Result<DeltaNetLayerWeights> {
    let h = config.hidden_size;
    let kd = config.linear_key_dim();
    let vd = config.linear_value_dim();
    let nv = config.linear_num_value_heads;
    let conv_dim = config.conv_dim();
    let ks = config.linear_conv_kernel_dim;
    let inter = config.intermediate_size;
    let vhd = config.linear_value_head_dim;

    // Quantized projections (固定順序)
    let in_proj_qkv = read_quantized_projection(reader, h * (kd * 2 + vd))?;
    let in_proj_z = read_quantized_projection(reader, h * vd)?;
    let in_proj_b = read_quantized_projection(reader, h * nv)?;
    let in_proj_a = read_quantized_projection(reader, h * nv)?;
    let out_proj = read_quantized_projection(reader, vd * h)?;
    let gate_proj = read_quantized_projection(reader, inter * h)?;
    let up_proj = read_quantized_projection(reader, inter * h)?;
    let down_proj = read_quantized_projection(reader, h * inter)?;

    // FP32 non-quantized
    let input_layernorm = read_f32_slice(reader, h)?;
    let post_attn_layernorm = read_f32_slice(reader, h)?;
    let a_log = read_f32_slice(reader, nv)?;
    let dt_bias = read_f32_slice(reader, nv)?;
    let conv1d_weight = read_f32_slice(reader, conv_dim * ks)?;
    let norm_weight = read_f32_slice(reader, vhd)?;

    Ok(DeltaNetLayerWeights {
        input_layernorm,
        post_attn_layernorm,
        in_proj_qkv,
        in_proj_z,
        in_proj_b,
        in_proj_a,
        a_log,
        dt_bias,
        conv1d_weight,
        norm_weight,
        out_proj,
        gate_proj,
        up_proj,
        down_proj,
    })
}

/// .alice ファイルから FullAttn レイヤーの量子化重みを読み込み、FP32 に復元。
///
/// # Errors
///
/// I/O エラー時。
pub fn read_fullattn_layer<R: Read>(
    reader: &mut R,
    config: &Qwen35Config,
) -> io::Result<FullAttnLayerWeights> {
    let h = config.hidden_size;
    let nh = config.num_attention_heads;
    let nkv = config.num_key_value_heads;
    let hd = config.head_dim;
    let inter = config.intermediate_size;

    // Quantized projections
    let q_proj = read_quantized_projection(reader, h * nh * hd)?;
    let k_proj = read_quantized_projection(reader, h * nkv * hd)?;
    let v_proj = read_quantized_projection(reader, h * nkv * hd)?;
    let o_proj = read_quantized_projection(reader, nh * hd * h)?;
    let gate_proj = read_quantized_projection(reader, inter * h)?;
    let up_proj = read_quantized_projection(reader, inter * h)?;
    let down_proj = read_quantized_projection(reader, h * inter)?;

    // FP32 non-quantized
    let input_layernorm = read_f32_slice(reader, h)?;
    let post_attn_layernorm = read_f32_slice(reader, h)?;
    let q_norm = read_f32_slice(reader, hd)?;
    let k_norm = read_f32_slice(reader, hd)?;

    Ok(FullAttnLayerWeights {
        input_layernorm,
        post_attn_layernorm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm,
        k_norm,
        gate_proj,
        up_proj,
        down_proj,
    })
}

// ── I/O Helpers ──────────────────────────────────────────────────────────

/// FP32 スライスを書き込む。
fn write_f32_slice<W: Write>(writer: &mut W, data: &[f32]) -> io::Result<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    writer.write_all(bytes)
}

/// BF16 スライスを書き込む。
fn write_bf16_slice<W: Write>(writer: &mut W, data: &[Bf16]) -> io::Result<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 2) };
    writer.write_all(bytes)
}

/// 量子化 projection 列を書き込む。各 projection: scale(f32) + packed_bytes。
fn write_quantized_projections<W: Write>(
    writer: &mut W,
    projections: &[(String, f32, Vec<u8>)],
) -> io::Result<()> {
    for (_, scale, packed) in projections {
        writer.write_all(&scale.to_le_bytes())?;
        writer.write_all(packed)?;
    }
    Ok(())
}

/// FP32 スライスを読み込む。
fn read_f32_slice<R: Read>(reader: &mut R, count: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; count * 4];
    reader.read_exact(&mut bytes)?;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(floats)
}

/// BF16 スライスを読み込む。
fn read_bf16_slice<R: Read>(reader: &mut R, count: usize) -> io::Result<Vec<Bf16>> {
    let mut bytes = vec![0u8; count * 2];
    reader.read_exact(&mut bytes)?;
    let bf16: Vec<Bf16> = bytes
        .chunks_exact(2)
        .map(|c| Bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    Ok(bf16)
}

/// 量子化 projection を読み込み FP32 に復元。scale(f32) + packed_ternary。
fn read_quantized_projection<R: Read>(reader: &mut R, count: usize) -> io::Result<Vec<f32>> {
    // scale
    let mut scale_buf = [0u8; 4];
    reader.read_exact(&mut scale_buf)?;
    let scale = f32::from_le_bytes(scale_buf);

    // packed ternary
    let packed_len = count.div_ceil(4);
    let mut packed = vec![0u8; packed_len];
    reader.read_exact(&mut packed)?;

    Ok(dequantize(&packed, count, scale))
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let values: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 0, 1, -1];
        let packed = pack_ternary(&values);
        let unpacked = unpack_ternary(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_unpack_empty() {
        let values: Vec<i8> = vec![];
        let packed = pack_ternary(&values);
        let unpacked = unpack_ternary(&packed, 0);
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_pack_unpack_single() {
        for v in [-1i8, 0, 1] {
            let packed = pack_ternary(&[v]);
            let unpacked = unpack_ternary(&packed, 1);
            assert_eq!(unpacked, vec![v]);
        }
    }

    #[test]
    fn test_pack_unpack_alignment() {
        // 4 の倍数でないケース
        let values: Vec<i8> = vec![1, -1, 0];
        let packed = pack_ternary(&values);
        assert_eq!(packed.len(), 1); // ceil(3/4) = 1 byte
        let unpacked = unpack_ternary(&packed, 3);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_quantize_and_pack_basic() {
        let weights = [0.5, -0.3, 0.01, 0.8, -0.9];
        let (scale, packed) = quantize_and_pack(&weights);

        assert!(scale > 0.0);

        let restored = dequantize(&packed, weights.len(), scale);
        // ternary: 各値は {-scale, 0, +scale}
        for &r in &restored {
            let norm = (r / scale).round();
            assert!(
                (norm - -1.0).abs() < 0.01
                    || (norm - 0.0).abs() < 0.01
                    || (norm - 1.0).abs() < 0.01,
                "restored {r} not ternary with scale {scale}"
            );
        }
    }

    #[test]
    fn test_quantize_empty() {
        let (scale, packed) = quantize_and_pack(&[]);
        assert!((scale - 0.0).abs() < 1e-10);
        assert!(packed.is_empty());
    }

    #[test]
    fn test_quantize_scale_matches_mean_abs() {
        let weights = [1.0f32, -1.0, 0.5, -0.5, 0.0];
        let (scale, _) = quantize_and_pack(&weights);
        // mean(|W|) = (1.0 + 1.0 + 0.5 + 0.5 + 0.0) / 5 = 0.6
        assert!((scale - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_dequantize_values() {
        let scale = 0.5;
        // Pack: [+1, -1, 0, +1]
        let packed = pack_ternary(&[1, -1, 0, 1]);
        let restored = dequantize(&packed, 4, scale);
        assert!((restored[0] - 0.5).abs() < 1e-6);
        assert!((restored[1] - (-0.5)).abs() < 1e-6);
        assert!((restored[2] - 0.0).abs() < 1e-6);
        assert!((restored[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_write_read_f32_roundtrip() {
        let data = vec![1.5f32, -2.3, 0.0, f32::MIN, f32::MAX];
        let mut buf = Vec::new();
        write_f32_slice(&mut buf, &data).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let restored = read_f32_slice(&mut cursor, data.len()).unwrap();
        assert_eq!(data, restored);
    }

    #[test]
    fn test_write_read_quantized_projection_roundtrip() {
        let weights = vec![0.3f32, -0.7, 0.1, 0.9, -0.2, 0.0, 0.6, -0.4];
        let (scale, packed) = quantize_and_pack(&weights);

        let mut buf = Vec::new();
        write_quantized_projections(&mut buf, &[("test".to_string(), scale, packed)]).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let restored = read_quantized_projection(&mut cursor, weights.len()).unwrap();

        // ternary 復元: 各値は {-scale, 0, +scale}
        for &r in &restored {
            let norm = (r / scale).round();
            assert!(
                (norm - -1.0).abs() < 0.01
                    || (norm - 0.0).abs() < 0.01
                    || (norm - 1.0).abs() < 0.01,
            );
        }
    }

    #[test]
    fn test_meta_serialization() {
        let config = Qwen35Config::qwen35_9b();
        let meta = AliceModelMeta {
            version: FORMAT_VERSION,
            config,
            quantization: "ternary_1_58bit".to_string(),
            tied_embeddings: true,
            quantized_params: 1000,
            non_quantized_params: 200,
            total_params: 1200,
            source_step: 5000,
            source_loss: 3.5,
            layer_scales: vec![],
        };

        let json = serde_json::to_vec(&meta).unwrap();
        let restored: AliceModelMeta = serde_json::from_slice(&json).unwrap();
        assert_eq!(restored.version, FORMAT_VERSION);
        assert_eq!(restored.source_step, 5000);
        assert!(restored.tied_embeddings);
    }

    #[test]
    fn test_read_alice_meta() {
        let config = Qwen35Config::qwen35_9b();
        let meta = AliceModelMeta {
            version: FORMAT_VERSION,
            config,
            quantization: "ternary_1_58bit".to_string(),
            tied_embeddings: true,
            quantized_params: 100,
            non_quantized_params: 50,
            total_params: 150,
            source_step: 1000,
            source_loss: 5.0,
            layer_scales: vec![],
        };

        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        let header_json = serde_json::to_vec(&meta).unwrap();
        buf.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        buf.extend_from_slice(&header_json);

        let mut cursor = std::io::Cursor::new(buf);
        let restored = read_alice_meta(&mut cursor).unwrap();
        assert_eq!(restored.source_step, 1000);
        assert!((restored.source_loss - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_read_alice_meta_bad_magic() {
        let buf = b"BADMAGIC\x01\x00\x00\x00";
        let mut cursor = std::io::Cursor::new(buf.as_slice());
        let result = read_alice_meta(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_compression_ratio() {
        // 1000 FP32 values → ternary packed
        let weights: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.001).collect();
        let (_, packed) = quantize_and_pack(&weights);

        let original_bytes = weights.len() * 4; // 4000 bytes
        let packed_bytes = packed.len() + 4; // packed + scale
        let ratio = original_bytes as f64 / packed_bytes as f64;

        // 4000 / (250 + 4) ≈ 15.7x 圧縮
        assert!(ratio > 10.0, "compression ratio {ratio:.1}x too low");
    }
}
