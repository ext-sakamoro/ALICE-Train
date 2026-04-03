//! Vision Encoder スタブ — ViT FP32 後付けインターフェース。
//!
//! Qwen3.5 は全モデルが VLM (Vision-Language Model)。
//! Language Model のみ 1.58-bit QAT し、推論時に FP32 ViT を再結合する。
//!
//! Language Model は事前学習で ViT 出力との接続を学習済みのため、
//! QAT 後も ViT 出力を正しく理解する（追加学習不要）。

/// ViT エンコーダの設定。
#[derive(Clone, Debug)]
pub struct VisionConfig {
    /// 画像サイズ (正方形)。
    pub image_size: usize,
    /// パッチサイズ。
    pub patch_size: usize,
    /// 隠れ次元。
    pub hidden_size: usize,
    /// Transformer レイヤー数。
    pub num_layers: usize,
    /// Attention ヘッド数。
    pub num_heads: usize,
    /// 出力 embedding 次元 (Language Model の hidden_size と一致)。
    pub output_dim: usize,
}

impl VisionConfig {
    /// Qwen3.5 VLM デフォルト設定。
    #[must_use]
    pub fn qwen35_default() -> Self {
        Self {
            image_size: 448,
            patch_size: 14,
            hidden_size: 1280,
            num_layers: 32,
            num_heads: 16,
            output_dim: 4096,
        }
    }

    /// パッチ数。
    #[must_use]
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size) * (self.image_size / self.patch_size)
    }

    /// ViT パラメータ数の概算。
    #[must_use]
    pub fn estimated_params(&self) -> usize {
        let patch_embed = 3 * self.patch_size * self.patch_size * self.hidden_size;
        let transformer = self.num_layers * (4 * self.hidden_size * self.hidden_size + 8 * self.hidden_size);
        let projection = self.hidden_size * self.output_dim;
        patch_embed + transformer + projection
    }

    /// FP32 モデルサイズの概算 (bytes)。
    #[must_use]
    pub fn estimated_size_bytes(&self) -> usize {
        self.estimated_params() * 4
    }
}

/// ViT エンコーダ出力。
#[derive(Clone, Debug)]
pub struct VisionOutput {
    /// 画像 embedding [num_patches × output_dim]。
    pub embeddings: Vec<f32>,
    /// パッチ数。
    pub num_patches: usize,
    /// 出力次元。
    pub output_dim: usize,
}

/// ViT エンコーダ（FP32 重み保持）。
///
/// 将来の実装で safetensors から ViT 重みをロードし、
/// 画像を embedding に変換する。
pub struct VisionEncoder {
    /// 設定。
    pub config: VisionConfig,
    // 将来: 重みフィールドを追加
    // patch_embed: Vec<f32>,
    // transformer_layers: Vec<TransformerLayer>,
    // projection: Vec<f32>,
}

impl VisionEncoder {
    /// 設定から ViT エンコーダを構築（重みなし）。
    #[must_use]
    pub fn new(config: VisionConfig) -> Self {
        Self { config }
    }

    /// 画像ピクセル → embedding 変換（スタブ実装）。
    ///
    /// 将来: ViT forward pass を実装。
    /// 現在はゼロベクトルを返す。
    #[must_use]
    pub fn encode(&self, _image_pixels: &[f32]) -> VisionOutput {
        let num_patches = self.config.num_patches();
        let output_dim = self.config.output_dim;
        VisionOutput {
            embeddings: vec![0.0; num_patches * output_dim],
            num_patches,
            output_dim,
        }
    }

    /// Language Model のプロンプトに画像 embedding を挿入。
    ///
    /// ChatML テンプレートの `<image>` プレースホルダを
    /// ViT embedding で置換する。
    pub fn inject_into_prompt(
        &self,
        text_embeddings: &[f32],
        text_len: usize,
        vision_output: &VisionOutput,
        image_token_pos: usize,
    ) -> Vec<f32> {
        let hidden = self.config.output_dim;
        let total_len = text_len + vision_output.num_patches;
        let mut merged = vec![0.0f32; total_len * hidden];

        // image_token_pos の前のテキスト
        let pre_len = image_token_pos * hidden;
        merged[..pre_len].copy_from_slice(&text_embeddings[..pre_len]);

        // 画像 embedding 挿入
        let img_len = vision_output.num_patches * hidden;
        merged[pre_len..pre_len + img_len].copy_from_slice(&vision_output.embeddings);

        // image_token_pos の後のテキスト
        let post_start = pre_len + img_len;
        let text_post_start = image_token_pos * hidden;
        let text_remaining = text_len * hidden - text_post_start;
        merged[post_start..post_start + text_remaining]
            .copy_from_slice(&text_embeddings[text_post_start..text_post_start + text_remaining]);

        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen35_config() {
        let cfg = VisionConfig::qwen35_default();
        assert_eq!(cfg.image_size, 448);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.num_patches(), 1024); // 32×32
    }

    #[test]
    fn test_estimated_params() {
        let cfg = VisionConfig::qwen35_default();
        let params = cfg.estimated_params();
        // ViT-L/14 は約300M params
        assert!(params > 100_000_000);
        assert!(params < 500_000_000);
    }

    #[test]
    fn test_estimated_size() {
        let cfg = VisionConfig::qwen35_default();
        let size = cfg.estimated_size_bytes();
        // ~1.2GB FP32
        assert!(size > 400_000_000);
        assert!(size < 2_000_000_000);
    }

    #[test]
    fn test_encode_stub() {
        let enc = VisionEncoder::new(VisionConfig::qwen35_default());
        let dummy_image = vec![0.0f32; 3 * 448 * 448];
        let output = enc.encode(&dummy_image);
        assert_eq!(output.num_patches, 1024);
        assert_eq!(output.output_dim, 4096);
        assert_eq!(output.embeddings.len(), 1024 * 4096);
    }

    #[test]
    fn test_inject_into_prompt() {
        let cfg = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            output_dim: 8,
        };
        let enc = VisionEncoder::new(cfg);
        let num_patches = 4; // (28/14)^2
        let text_len = 10;
        let hidden = 8;

        let text_emb = vec![1.0f32; text_len * hidden];
        let vision_out = VisionOutput {
            embeddings: vec![2.0f32; num_patches * hidden],
            num_patches,
            output_dim: hidden,
        };

        let merged = enc.inject_into_prompt(&text_emb, text_len, &vision_out, 3);
        assert_eq!(merged.len(), (text_len + num_patches) * hidden);
        // 最初の3トークンはテキスト (1.0)
        assert!((merged[0] - 1.0).abs() < 1e-6);
        // 位置3からはビジョン (2.0)
        assert!((merged[3 * hidden] - 2.0).abs() < 1e-6);
    }
}
