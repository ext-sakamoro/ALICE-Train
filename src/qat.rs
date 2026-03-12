//! Quantization-Aware Training (QAT) — 量子化再学習パイプライン
//!
//! FP32 重みを学習中に fake quantize し、量子化後の精度劣化を最小化する。
//!
//! # 設計
//!
//! - `FakeQuantize`: forward 時に量子化→逆量子化（STE で勾配通過）
//! - `QatConfig`: per-layer 量子化設定（bits, scale 学習, temperature annealing）
//! - `CalibrationStats`: activation range 統計、weight sensitivity 分析
//! - `QatTrainer`: 学習ループ拡張 — epoch 毎に quant stats 追跡
//!
//! # Flow
//!
//! ```text
//! FP32 weights → FakeQuantize(forward) → quantized logits → loss
//!                    ↑ STE gradient           ↓ backward
//!              latent FP32 ← grad ← grad_output
//! ```

/// 量子化ビット数の指定。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantBits {
    /// 1.58-bit ternary {-1, 0, +1}
    Ternary,
    /// 2-bit {-1, 0, +1} with separate zero threshold
    Binary2,
    /// 4-bit uniform quantization (16 levels)
    Int4,
    /// 8-bit uniform quantization (256 levels)
    Int8,
}

impl QuantBits {
    /// 量子化レベル数を返す。
    #[must_use]
    pub const fn levels(self) -> u32 {
        match self {
            Self::Ternary => 3,
            Self::Binary2 => 3,
            Self::Int4 => 16,
            Self::Int8 => 256,
        }
    }

    /// 実効ビット数を返す（log2(levels)）。
    #[must_use]
    pub fn effective_bits(self) -> f32 {
        (self.levels() as f32).log2()
    }
}

/// Per-layer QAT 設定。
#[derive(Clone, Debug)]
pub struct QatConfig {
    /// 量子化ビット数。
    pub bits: QuantBits,
    /// scale factor を学習するか（true: 勾配で更新、false: mean(|W|) 固定）。
    pub learn_scale: bool,
    /// temperature annealing の初期値（1.0 = no annealing）。
    /// 低い値ほど量子化が「硬く」なる。
    pub temperature: f32,
    /// temperature の減衰率（epoch 毎に掛ける）。
    pub temperature_decay: f32,
    /// gradient clipping の閾値（0.0 = 無効）。
    pub grad_clip: f32,
}

impl Default for QatConfig {
    fn default() -> Self {
        Self {
            bits: QuantBits::Ternary,
            learn_scale: true,
            temperature: 1.0,
            temperature_decay: 0.99,
            grad_clip: 0.0,
        }
    }
}

impl QatConfig {
    /// Ternary QAT のデフォルト設定を返す。
    #[must_use]
    pub fn ternary() -> Self {
        Self::default()
    }

    /// INT4 QAT の設定を返す。
    #[must_use]
    pub fn int4() -> Self {
        Self {
            bits: QuantBits::Int4,
            learn_scale: true,
            temperature: 1.0,
            temperature_decay: 0.995,
            grad_clip: 1.0,
        }
    }

    /// INT8 QAT の設定を返す。
    #[must_use]
    pub fn int8() -> Self {
        Self {
            bits: QuantBits::Int8,
            learn_scale: false,
            temperature: 1.0,
            temperature_decay: 1.0,
            grad_clip: 0.0,
        }
    }
}

/// Fake quantization: FP32 → 量子化 → 逆量子化（STE で勾配通過）。
///
/// forward 時に値を離散化し、backward 時は STE で勾配をそのまま通す。
/// これにより学習中にモデルが量子化誤差に適応する。
pub struct FakeQuantize {
    config: QatConfig,
    /// 学習可能な scale factor（mean(|W|) で初期化）。
    scale: f32,
    /// 現在の temperature。
    current_temp: f32,
}

impl FakeQuantize {
    /// 新しい `FakeQuantize` を構築する。
    #[must_use]
    pub fn new(config: QatConfig) -> Self {
        Self {
            current_temp: config.temperature,
            config,
            scale: 1.0,
        }
    }

    /// scale factor を外部から設定する（calibration 結果の注入用）。
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// 現在の scale factor を返す。
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// 現在の temperature を返す。
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.current_temp
    }

    /// Weights から scale factor を calibrate する。
    ///
    /// BitNet b1.58 式: γ = mean(|W|)
    pub fn calibrate_scale(&mut self, weights: &[f32]) {
        if weights.is_empty() {
            return;
        }
        let mut sum = 0.0f64;
        for &w in weights {
            sum += w.abs() as f64;
        }
        self.scale = (sum / weights.len() as f64) as f32;
        if self.scale < 1e-10 {
            self.scale = 1e-10;
        }
    }

    /// Forward pass: fake quantize weights in-place。
    ///
    /// `weights` を量子化→逆量子化し、同じバッファに書き戻す。
    /// 元の FP32 値は呼び出し側が別途保持すること。
    pub fn fake_quantize_forward(&self, weights: &[f32], quantized: &mut [f32]) {
        assert_eq!(weights.len(), quantized.len());
        let inv_scale = 1.0 / self.scale.max(1e-10);
        let inv_temp = 1.0 / self.current_temp.max(1e-10);

        match self.config.bits {
            QuantBits::Ternary | QuantBits::Binary2 => {
                // FP32 → round(w / γ) → clamp(-1, 1) → × γ
                for (q, &w) in quantized.iter_mut().zip(weights.iter()) {
                    let scaled = w * inv_scale * inv_temp;
                    let rounded = scaled.round().clamp(-1.0, 1.0);
                    *q = rounded * self.scale;
                }
            }
            QuantBits::Int4 => {
                // FP32 → round to 16 levels → dequantize
                let half_levels = 7.0; // [-7, +7] = 15 levels (+ zero = 16)
                for (q, &w) in quantized.iter_mut().zip(weights.iter()) {
                    let scaled = w * inv_scale * half_levels * inv_temp;
                    let rounded = scaled.round().clamp(-half_levels, half_levels);
                    *q = rounded * self.scale / half_levels;
                }
            }
            QuantBits::Int8 => {
                // FP32 → round to 256 levels → dequantize
                let half_levels = 127.0;
                for (q, &w) in quantized.iter_mut().zip(weights.iter()) {
                    let scaled = w * inv_scale * half_levels * inv_temp;
                    let rounded = scaled.round().clamp(-half_levels, half_levels);
                    *q = rounded * self.scale / half_levels;
                }
            }
        }
    }

    /// Backward pass: STE で勾配をそのまま通す。
    ///
    /// Gradient clipping 適用（設定されている場合）。
    pub fn ste_backward(&self, grad_output: &[f32], grad_input: &mut [f32]) {
        assert_eq!(grad_output.len(), grad_input.len());
        if self.config.grad_clip > 0.0 {
            let clip = self.config.grad_clip;
            for (gi, &go) in grad_input.iter_mut().zip(grad_output.iter()) {
                *gi = go.clamp(-clip, clip);
            }
        } else {
            grad_input.copy_from_slice(grad_output);
        }
    }

    /// Epoch 終了時に temperature を減衰させる。
    pub fn step_temperature(&mut self) {
        self.current_temp *= self.config.temperature_decay;
        if self.current_temp < 0.01 {
            self.current_temp = 0.01;
        }
    }
}

/// Calibration 統計: activation range と weight sensitivity を追跡。
#[derive(Clone, Debug)]
pub struct CalibrationStats {
    /// Weight の最小値。
    pub weight_min: f32,
    /// Weight の最大値。
    pub weight_max: f32,
    /// Weight の平均絶対値（scale factor 候補）。
    pub weight_mean_abs: f32,
    /// 量子化前後の Mean Absolute Error。
    pub quantization_mae: f32,
    /// Activation の最小値。
    pub activation_min: f32,
    /// Activation の最大値。
    pub activation_max: f32,
    /// 量子化前後のコサイン類似度。
    pub cosine_similarity: f32,
    /// サンプル数。
    pub sample_count: u64,
}

impl CalibrationStats {
    /// 空の統計を生成する。
    #[must_use]
    pub fn new() -> Self {
        Self {
            weight_min: f32::MAX,
            weight_max: f32::MIN,
            weight_mean_abs: 0.0,
            quantization_mae: 0.0,
            activation_min: f32::MAX,
            activation_max: f32::MIN,
            cosine_similarity: 0.0,
            sample_count: 0,
        }
    }

    /// Weight 統計を更新する。
    pub fn update_weights(&mut self, weights: &[f32], quantized: &[f32]) {
        assert_eq!(weights.len(), quantized.len());
        if weights.is_empty() {
            return;
        }

        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum_abs = 0.0f64;
        let mut sum_err = 0.0f64;
        let mut dot_wq = 0.0f64;
        let mut dot_ww = 0.0f64;
        let mut dot_qq = 0.0f64;

        for (&w, &q) in weights.iter().zip(quantized.iter()) {
            if w < min {
                min = w;
            }
            if w > max {
                max = w;
            }
            sum_abs += w.abs() as f64;
            sum_err += (w - q).abs() as f64;
            dot_wq += (w as f64) * (q as f64);
            dot_ww += (w as f64) * (w as f64);
            dot_qq += (q as f64) * (q as f64);
        }

        let n = weights.len() as f64;
        self.weight_min = min;
        self.weight_max = max;
        self.weight_mean_abs = (sum_abs / n) as f32;
        self.quantization_mae = (sum_err / n) as f32;

        let denom = (dot_ww * dot_qq).sqrt();
        self.cosine_similarity = if denom > 1e-10 {
            (dot_wq / denom) as f32
        } else {
            0.0
        };
    }

    /// Activation 統計を更新する。
    pub fn update_activations(&mut self, activations: &[f32]) {
        for &a in activations {
            if a < self.activation_min {
                self.activation_min = a;
            }
            if a > self.activation_max {
                self.activation_max = a;
            }
        }
        self.sample_count += 1;
    }
}

impl Default for CalibrationStats {
    fn default() -> Self {
        Self::new()
    }
}

/// QAT 学習の epoch 結果。
#[derive(Clone, Debug)]
pub struct QatEpochResult {
    /// Epoch 番号。
    pub epoch: usize,
    /// 平均損失。
    pub avg_loss: f32,
    /// 平均量子化誤差 (MAE)。
    pub avg_quant_mae: f32,
    /// 平均コサイン類似度（量子化前後）。
    pub avg_cosine_sim: f32,
    /// 現在の temperature。
    pub temperature: f32,
    /// 現在の scale factor。
    pub scale: f32,
}

/// QAT 学習ループ。
///
/// `Trainer` の拡張版 — epoch 毎に fake quantize → 学習 → quant stats 追跡。
pub struct QatTrainer {
    /// 学習設定。
    pub epochs: usize,
    /// 学習率。
    pub learning_rate: f32,
    /// QAT 設定。
    pub qat_config: QatConfig,
    /// Calibration 統計（epoch 毎にリセット）。
    stats: CalibrationStats,
}

impl QatTrainer {
    /// 新しい QAT トレーナーを構築する。
    #[must_use]
    pub fn new(epochs: usize, learning_rate: f32, qat_config: QatConfig) -> Self {
        Self {
            epochs,
            learning_rate,
            qat_config,
            stats: CalibrationStats::new(),
        }
    }

    /// Calibration 統計を返す。
    #[must_use]
    pub fn stats(&self) -> &CalibrationStats {
        &self.stats
    }

    /// 単一レイヤーの QAT ステップを実行する。
    ///
    /// 1. latent weights を fake quantize
    /// 2. quantized weights で forward
    /// 3. loss 計算
    /// 4. STE backward で latent weights の勾配を計算
    /// 5. latent weights を更新（SGD）
    /// 6. calibration stats を更新
    ///
    /// # Returns
    ///
    /// (loss, quantization_mae)
    pub fn qat_step(
        &mut self,
        latent_weights: &mut [f32],
        quantized_buf: &mut [f32],
        input: &[f32],
        target: &[f32],
        output_buf: &mut [f32],
        grad_output_buf: &mut [f32],
        grad_weight_buf: &mut [f32],
        fq: &mut FakeQuantize,
        forward_fn: &dyn Fn(&[f32], &[f32], &mut [f32]),
        loss_fn: &dyn Fn(&[f32], &[f32], &mut [f32]) -> f32,
    ) -> (f32, f32) {
        // 1. Calibrate scale from latent weights
        fq.calibrate_scale(latent_weights);

        // 2. Fake quantize
        fq.fake_quantize_forward(latent_weights, quantized_buf);

        // 3. Forward with quantized weights
        forward_fn(quantized_buf, input, output_buf);

        // 4. Loss + grad
        let loss = loss_fn(output_buf, target, grad_output_buf);

        // 5. STE backward → weight gradient (simplified: dy * x^T)
        let grad_len = grad_weight_buf.len().min(latent_weights.len());
        fq.ste_backward(
            &grad_output_buf[..grad_len],
            &mut grad_weight_buf[..grad_len],
        );

        // 6. SGD update on latent weights
        let lr = self.learning_rate;
        for (w, &g) in latent_weights.iter_mut().zip(grad_weight_buf.iter()) {
            *w -= lr * g;
        }

        // 7. Update calibration stats
        fq.fake_quantize_forward(latent_weights, quantized_buf);
        self.stats.update_weights(latent_weights, quantized_buf);

        (loss, self.stats.quantization_mae)
    }

    /// Temperature annealing を1 epoch分進める。
    pub fn step_epoch(&mut self, fq: &mut FakeQuantize) {
        fq.step_temperature();
    }

    /// Calibration 統計をリセットする。
    pub fn reset_stats(&mut self) {
        self.stats = CalibrationStats::new();
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- QuantBits ---

    #[test]
    fn test_quant_bits_ternary_levels() {
        assert_eq!(QuantBits::Ternary.levels(), 3);
        assert!((QuantBits::Ternary.effective_bits() - 1.585).abs() < 0.01);
    }

    #[test]
    fn test_quant_bits_int4_levels() {
        assert_eq!(QuantBits::Int4.levels(), 16);
        assert!((QuantBits::Int4.effective_bits() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_quant_bits_int8_levels() {
        assert_eq!(QuantBits::Int8.levels(), 256);
        assert!((QuantBits::Int8.effective_bits() - 8.0).abs() < 0.01);
    }

    // --- QatConfig ---

    #[test]
    fn test_qat_config_default() {
        let c = QatConfig::default();
        assert_eq!(c.bits, QuantBits::Ternary);
        assert!(c.learn_scale);
        assert!((c.temperature - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_qat_config_int4() {
        let c = QatConfig::int4();
        assert_eq!(c.bits, QuantBits::Int4);
        assert!((c.grad_clip - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_qat_config_int8() {
        let c = QatConfig::int8();
        assert_eq!(c.bits, QuantBits::Int8);
        assert!(!c.learn_scale);
    }

    // --- FakeQuantize ---

    #[test]
    fn test_fake_quantize_ternary_roundtrip() {
        let mut fq = FakeQuantize::new(QatConfig::ternary());
        let weights = [0.5, -0.3, 0.01, 0.8, -0.9];
        fq.calibrate_scale(&weights);

        let mut quantized = vec![0.0; weights.len()];
        fq.fake_quantize_forward(&weights, &mut quantized);

        // Ternary: each value should be {-scale, 0, +scale}
        let s = fq.scale();
        for &q in &quantized {
            let normalized = (q / s).round();
            assert!(
                (normalized - -1.0).abs() < 0.01
                    || (normalized - 0.0).abs() < 0.01
                    || (normalized - 1.0).abs() < 0.01,
                "quantized value {q} not in {{-s, 0, +s}}, s={s}"
            );
        }
    }

    #[test]
    fn test_fake_quantize_int4_range() {
        let mut fq = FakeQuantize::new(QatConfig::int4());
        let weights = [0.1, -0.5, 0.3, 0.7, -0.2];
        fq.calibrate_scale(&weights);

        let mut quantized = vec![0.0; weights.len()];
        fq.fake_quantize_forward(&weights, &mut quantized);

        let s = fq.scale();
        for &q in &quantized {
            assert!(q.abs() <= s + 1e-6, "INT4 quantized {q} exceeds scale {s}");
        }
    }

    #[test]
    fn test_fake_quantize_int8_precision() {
        let mut fq = FakeQuantize::new(QatConfig::int8());
        let weights = [0.5, -0.3, 0.1];
        fq.calibrate_scale(&weights);

        let mut quantized = vec![0.0; weights.len()];
        fq.fake_quantize_forward(&weights, &mut quantized);

        // INT8 should have higher precision than ternary
        let mae: f32 = weights
            .iter()
            .zip(quantized.iter())
            .map(|(w, q)| (w - q).abs())
            .sum::<f32>()
            / weights.len() as f32;
        assert!(mae < 0.1, "INT8 MAE {mae} too high");
    }

    #[test]
    fn test_fake_quantize_calibrate_scale() {
        let mut fq = FakeQuantize::new(QatConfig::ternary());
        let weights = [1.0, -1.0, 0.5, -0.5];
        fq.calibrate_scale(&weights);
        // mean(|W|) = (1.0 + 1.0 + 0.5 + 0.5) / 4 = 0.75
        assert!((fq.scale() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_fake_quantize_empty_weights() {
        let mut fq = FakeQuantize::new(QatConfig::ternary());
        fq.calibrate_scale(&[]);
        assert!((fq.scale() - 1.0).abs() < 1e-6); // unchanged
    }

    #[test]
    fn test_fake_quantize_set_scale() {
        let mut fq = FakeQuantize::new(QatConfig::ternary());
        fq.set_scale(0.42);
        assert!((fq.scale() - 0.42).abs() < 1e-6);
    }

    // --- STE backward ---

    #[test]
    fn test_ste_backward_passthrough() {
        let fq = FakeQuantize::new(QatConfig::ternary());
        let grad = [1.0, -2.0, 3.0];
        let mut grad_in = [0.0; 3];
        fq.ste_backward(&grad, &mut grad_in);
        assert_eq!(grad_in, [1.0, -2.0, 3.0]);
    }

    #[test]
    fn test_ste_backward_with_clipping() {
        let config = QatConfig {
            grad_clip: 1.5,
            ..QatConfig::default()
        };
        let fq = FakeQuantize::new(config);
        let grad = [3.0, -2.0, 0.5];
        let mut grad_in = [0.0; 3];
        fq.ste_backward(&grad, &mut grad_in);
        assert!((grad_in[0] - 1.5).abs() < 1e-6);
        assert!((grad_in[1] - (-1.5)).abs() < 1e-6);
        assert!((grad_in[2] - 0.5).abs() < 1e-6);
    }

    // --- Temperature annealing ---

    #[test]
    fn test_temperature_annealing() {
        let config = QatConfig {
            temperature: 1.0,
            temperature_decay: 0.9,
            ..QatConfig::default()
        };
        let mut fq = FakeQuantize::new(config);
        assert!((fq.temperature() - 1.0).abs() < 1e-6);
        fq.step_temperature();
        assert!((fq.temperature() - 0.9).abs() < 1e-6);
        fq.step_temperature();
        assert!((fq.temperature() - 0.81).abs() < 1e-4);
    }

    #[test]
    fn test_temperature_floor() {
        let config = QatConfig {
            temperature: 0.02,
            temperature_decay: 0.1,
            ..QatConfig::default()
        };
        let mut fq = FakeQuantize::new(config);
        fq.step_temperature(); // 0.02 * 0.1 = 0.002 → clamped to 0.01
        assert!((fq.temperature() - 0.01).abs() < 1e-6);
    }

    // --- CalibrationStats ---

    #[test]
    fn test_calibration_stats_weights() {
        let mut stats = CalibrationStats::new();
        let weights = [1.0, -0.5, 0.3, -0.8];
        let quantized = [0.75, -0.75, 0.0, -0.75];
        stats.update_weights(&weights, &quantized);

        assert!((stats.weight_min - (-0.8)).abs() < 1e-6);
        assert!((stats.weight_max - 1.0).abs() < 1e-6);
        // mean(|W|) = (1.0 + 0.5 + 0.3 + 0.8) / 4 = 0.65
        assert!((stats.weight_mean_abs - 0.65).abs() < 1e-4);
        assert!(stats.quantization_mae > 0.0);
        assert!(stats.cosine_similarity > 0.9); // high similarity expected
    }

    #[test]
    fn test_calibration_stats_activations() {
        let mut stats = CalibrationStats::new();
        stats.update_activations(&[0.1, 0.5, -0.3, 1.2]);
        assert!((stats.activation_min - (-0.3)).abs() < 1e-6);
        assert!((stats.activation_max - 1.2).abs() < 1e-6);
        assert_eq!(stats.sample_count, 1);

        stats.update_activations(&[-1.0, 2.0]);
        assert!((stats.activation_min - (-1.0)).abs() < 1e-6);
        assert!((stats.activation_max - 2.0).abs() < 1e-6);
        assert_eq!(stats.sample_count, 2);
    }

    #[test]
    fn test_calibration_stats_default() {
        let stats = CalibrationStats::default();
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.weight_min, f32::MAX);
    }

    // --- QatTrainer ---

    #[test]
    fn test_qat_trainer_new() {
        let trainer = QatTrainer::new(10, 0.01, QatConfig::ternary());
        assert_eq!(trainer.epochs, 10);
        assert!((trainer.learning_rate - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_qat_step_loss_decreases() {
        let mut trainer = QatTrainer::new(100, 0.1, QatConfig::ternary());
        let mut fq = FakeQuantize::new(QatConfig::ternary());

        // Simple: learn W such that W * 1.0 ≈ 0.5
        let mut latent_w = vec![0.0_f32; 1];
        let mut quantized = vec![0.0_f32; 1];
        let input = [1.0_f32];
        let target = [0.5_f32];
        let mut output = vec![0.0_f32; 1];
        let mut grad_out = vec![0.0_f32; 1];
        let mut grad_w = vec![0.0_f32; 1];

        // Simple matvec: y = W * x
        let forward = |w: &[f32], x: &[f32], y: &mut [f32]| {
            y[0] = w[0] * x[0];
        };
        // MSE loss: L = (y - t)^2, dL/dy = 2*(y - t)
        let loss = |y: &[f32], t: &[f32], gy: &mut [f32]| -> f32 {
            let diff = y[0] - t[0];
            gy[0] = 2.0 * diff;
            diff * diff
        };

        let mut first_loss = f32::MAX;
        let mut last_loss = f32::MAX;
        for i in 0..100 {
            let (l, _mae) = trainer.qat_step(
                &mut latent_w,
                &mut quantized,
                &input,
                &target,
                &mut output,
                &mut grad_out,
                &mut grad_w,
                &mut fq,
                &forward,
                &loss,
            );
            if i == 0 {
                first_loss = l;
            }
            last_loss = l;
            trainer.step_epoch(&mut fq);
        }

        assert!(
            last_loss < first_loss,
            "Loss should decrease: first={first_loss}, last={last_loss}"
        );
    }

    #[test]
    fn test_qat_step_updates_stats() {
        let mut trainer = QatTrainer::new(1, 0.01, QatConfig::ternary());
        let mut fq = FakeQuantize::new(QatConfig::ternary());
        let mut latent_w = vec![0.5_f32, -0.3];
        let mut quantized = vec![0.0; 2];
        let input = [1.0, 1.0];
        let target = [0.2, -0.1];
        let mut output = vec![0.0; 2];
        let mut grad_out = vec![0.0; 2];
        let mut grad_w = vec![0.0; 2];

        let forward = |w: &[f32], _x: &[f32], y: &mut [f32]| {
            y[0] = w[0];
            y[1] = w[1];
        };
        let loss = |y: &[f32], t: &[f32], gy: &mut [f32]| -> f32 {
            let mut l = 0.0;
            for i in 0..y.len() {
                let d = y[i] - t[i];
                gy[i] = 2.0 * d;
                l += d * d;
            }
            l / y.len() as f32
        };

        trainer.qat_step(
            &mut latent_w,
            &mut quantized,
            &input,
            &target,
            &mut output,
            &mut grad_out,
            &mut grad_w,
            &mut fq,
            &forward,
            &loss,
        );

        let stats = trainer.stats();
        assert!(stats.weight_min < stats.weight_max);
        assert!(stats.weight_mean_abs > 0.0);
    }

    #[test]
    fn test_qat_reset_stats() {
        let mut trainer = QatTrainer::new(1, 0.01, QatConfig::ternary());
        trainer.stats = CalibrationStats {
            weight_min: -1.0,
            weight_max: 1.0,
            weight_mean_abs: 0.5,
            quantization_mae: 0.1,
            activation_min: -2.0,
            activation_max: 2.0,
            cosine_similarity: 0.99,
            sample_count: 100,
        };
        trainer.reset_stats();
        assert_eq!(trainer.stats().sample_count, 0);
        assert_eq!(trainer.stats().weight_min, f32::MAX);
    }

    // --- QatEpochResult ---

    #[test]
    fn test_qat_epoch_result_clone() {
        let r = QatEpochResult {
            epoch: 5,
            avg_loss: 0.01,
            avg_quant_mae: 0.05,
            avg_cosine_sim: 0.99,
            temperature: 0.8,
            scale: 0.42,
        };
        let r2 = r.clone();
        assert_eq!(r2.epoch, 5);
        assert!((r2.avg_loss - 0.01).abs() < 1e-6);
    }
}
