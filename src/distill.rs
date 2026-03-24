//! Knowledge Distillation — teacher-student 蒸留パイプライン
//!
//! 大モデル（teacher, FP32）の出力を小モデル（student, 量子化）に蒸留する。
//!
//! # 設計
//!
//! - `DistillConfig`: temperature, alpha（soft/hard label 混合比）
//! - `distill_loss`: KL-divergence (soft) + cross-entropy (hard) の加重混合
//! - `DistillTrainer`: teacher forward → student forward → 蒸留損失 → student backward
//!
//! # Flow
//!
//! ```text
//! input → teacher(FP32) → soft_targets (logits / T)
//!       → student(QAT)  → student_logits
//!       → distill_loss(soft_targets, student_logits, hard_labels) → grad → student update
//! ```

/// 蒸留設定。
#[derive(Clone, Debug)]
pub struct DistillConfig {
    /// 温度パラメータ（soft targets の平滑化度合い）。
    /// 高い値ほど teacher の暗黙知をよりよく伝達する。
    pub temperature: f32,
    /// Soft label 損失の混合比 alpha (0.0–1.0)。
    /// total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    pub alpha: f32,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.7,
        }
    }
}

impl DistillConfig {
    /// デフォルトの蒸留設定を返す（T=4.0, alpha=0.7）。
    #[must_use]
    pub fn new(temperature: f32, alpha: f32) -> Self {
        Self { temperature, alpha }
    }
}

/// Softmax を計算する（数値安定性のため max 減算）。
fn softmax_with_temperature(logits: &[f32], temperature: f32, output: &mut [f32]) {
    assert_eq!(logits.len(), output.len());
    if logits.is_empty() {
        return;
    }

    let inv_t = 1.0 / temperature.max(1e-10);
    let mut max_val = f32::MIN;
    for &l in logits {
        if l > max_val {
            max_val = l;
        }
    }

    let mut sum = 0.0f64;
    for (o, &l) in output.iter_mut().zip(logits.iter()) {
        let e = ((l - max_val) * inv_t).exp();
        *o = e;
        sum += e as f64;
    }

    let inv_sum = 1.0 / sum as f32;
    for o in output.iter_mut() {
        *o *= inv_sum;
    }
}

/// KL-divergence: KL(p || q) = Σ p * log(p / q)
///
/// 勾配: dL/d(student_logit_i) = q_i - p_i (softmax 出力に対して)
fn kl_divergence(teacher_probs: &[f32], student_probs: &[f32]) -> f32 {
    assert_eq!(teacher_probs.len(), student_probs.len());
    let mut kl = 0.0f64;
    for (&p, &q) in teacher_probs.iter().zip(student_probs.iter()) {
        if p > 1e-10 {
            let q_safe = q.max(1e-10);
            kl += (p as f64) * ((p / q_safe) as f64).ln();
        }
    }
    kl as f32
}

/// 蒸留損失を計算する。
///
/// `total_loss = alpha * T^2 * KL(teacher_soft || student_soft) + (1 - alpha) * hard_loss`
///
/// # Arguments
///
/// - `teacher_logits` — teacher モデルの生 logits
/// - `student_logits` — student モデルの生 logits
/// - `hard_labels` — 正解ラベル（one-hot or 確率分布）
/// - `config` — 蒸留設定
/// - `grad_student` — student logits に対する勾配の書き込み先
///
/// # Returns
///
/// (total_loss, soft_loss, hard_loss)
pub fn distill_loss(
    teacher_logits: &[f32],
    student_logits: &[f32],
    hard_labels: &[f32],
    config: &DistillConfig,
    grad_student: &mut [f32],
) -> (f32, f32, f32) {
    let n = teacher_logits.len();
    assert_eq!(student_logits.len(), n);
    assert_eq!(hard_labels.len(), n);
    assert_eq!(grad_student.len(), n);

    // Soft targets (high temperature)
    let mut teacher_soft = vec![0.0f32; n];
    let mut student_soft = vec![0.0f32; n];
    softmax_with_temperature(teacher_logits, config.temperature, &mut teacher_soft);
    softmax_with_temperature(student_logits, config.temperature, &mut student_soft);

    // KL-divergence (soft loss), scaled by T^2
    let t_sq = config.temperature * config.temperature;
    let soft_loss = kl_divergence(&teacher_soft, &student_soft) * t_sq;

    // Hard loss: cross-entropy with temperature=1
    let mut student_hard = vec![0.0f32; n];
    softmax_with_temperature(student_logits, 1.0, &mut student_hard);
    let mut hard_loss = 0.0f64;
    for (&y, &p) in hard_labels.iter().zip(student_hard.iter()) {
        if y > 1e-10 {
            hard_loss -= (y as f64) * (p.max(1e-10) as f64).ln();
        }
    }
    let hard_loss = hard_loss as f32;

    // Total loss
    let total_loss = config.alpha * soft_loss + (1.0 - config.alpha) * hard_loss;

    // Gradient: alpha * T^2 * (student_soft - teacher_soft) / T + (1-alpha) * (student_hard - hard_labels)
    // Simplified: alpha * T * (q_soft - p_soft) + (1-alpha) * (q_hard - y)
    let alpha = config.alpha;
    let t = config.temperature;
    for i in 0..n {
        grad_student[i] = alpha * t * (student_soft[i] - teacher_soft[i])
            + (1.0 - alpha) * (student_hard[i] - hard_labels[i]);
    }

    (total_loss, soft_loss, hard_loss)
}

/// 蒸留トレーナー。
///
/// Teacher (frozen FP32) と Student (QAT) のペアで学習する。
pub struct DistillTrainer {
    /// 蒸留設定。
    pub config: DistillConfig,
    /// 学習率。
    pub learning_rate: f32,
    /// エポック数。
    pub epochs: usize,
}

/// 蒸留エポック結果。
#[derive(Clone, Debug)]
pub struct DistillEpochResult {
    /// エポック番号。
    pub epoch: usize,
    /// 合計損失。
    pub total_loss: f32,
    /// Soft label 損失。
    pub soft_loss: f32,
    /// Hard label 損失。
    pub hard_loss: f32,
}

impl DistillTrainer {
    /// 新しい蒸留トレーナーを構築する。
    #[must_use]
    pub fn new(config: DistillConfig, learning_rate: f32, epochs: usize) -> Self {
        Self {
            config,
            learning_rate,
            epochs,
        }
    }

    /// 単一サンプルの蒸留ステップを実行する。
    ///
    /// teacher_logits は事前計算済み（frozen）。
    /// student_weights を SGD で更新する。
    ///
    /// # Returns
    ///
    /// (total_loss, soft_loss, hard_loss)
    pub fn distill_step(
        &self,
        teacher_logits: &[f32],
        student_logits: &[f32],
        hard_labels: &[f32],
        student_weights: &mut [f32],
        grad_buf: &mut [f32],
    ) -> (f32, f32, f32) {
        let (total, soft, hard) = distill_loss(
            teacher_logits,
            student_logits,
            hard_labels,
            &self.config,
            grad_buf,
        );

        // SGD update (勾配は student logits に対するもの。
        // 実際のアプリでは chain rule で weight 勾配を計算する必要がある。
        // ここでは簡略化: logit 勾配を直接 weight に適用。)
        let lr = self.learning_rate;
        let n = student_weights.len().min(grad_buf.len());
        for i in 0..n {
            student_weights[i] -= lr * grad_buf[i];
        }

        (total, soft, hard)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- DistillConfig ---

    #[test]
    fn test_distill_config_default() {
        let c = DistillConfig::default();
        assert!((c.temperature - 4.0).abs() < 1e-6);
        assert!((c.alpha - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_distill_config_custom() {
        let c = DistillConfig::new(6.0, 0.5);
        assert!((c.temperature - 6.0).abs() < 1e-6);
        assert!((c.alpha - 0.5).abs() < 1e-6);
    }

    // --- softmax ---

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0; 4];
        softmax_with_temperature(&logits, 1.0, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_high_temperature() {
        let logits = [1.0, 10.0];
        let mut out_low = [0.0; 2];
        let mut out_high = [0.0; 2];
        softmax_with_temperature(&logits, 1.0, &mut out_low);
        softmax_with_temperature(&logits, 10.0, &mut out_high);
        // High temp → more uniform distribution
        assert!(
            (out_high[0] - out_high[1]).abs() < (out_low[0] - out_low[1]).abs(),
            "High temperature should flatten distribution"
        );
    }

    #[test]
    fn test_softmax_empty() {
        let logits: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        softmax_with_temperature(&logits, 1.0, &mut out);
    }

    // --- KL divergence ---

    #[test]
    fn test_kl_divergence_identical() {
        let p = [0.3, 0.5, 0.2];
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-5, "KL(p||p) should be ~0, got {kl}");
    }

    #[test]
    fn test_kl_divergence_positive() {
        let p = [0.9, 0.1];
        let q = [0.5, 0.5];
        let kl = kl_divergence(&p, &q);
        assert!(
            kl > 0.0,
            "KL should be positive for different distributions"
        );
    }

    // --- distill_loss ---

    #[test]
    fn test_distill_loss_identical_teacher_student() {
        let logits = [1.0, 2.0, 3.0];
        let labels = [0.0, 0.0, 1.0];
        let config = DistillConfig::default();
        let mut grad = [0.0; 3];

        let (total, soft, _hard) = distill_loss(&logits, &logits, &labels, &config, &mut grad);
        // Same logits → soft loss ≈ 0
        assert!(
            soft < 1e-5,
            "Soft loss should be ~0 for identical logits, got {soft}"
        );
        assert!(total >= 0.0);
    }

    #[test]
    fn test_distill_loss_gradient_direction() {
        let teacher = [5.0, 1.0, 1.0]; // teacher confident in class 0
        let student = [1.0, 1.0, 1.0]; // student uncertain
        let labels = [1.0, 0.0, 0.0]; // hard label: class 0
        let config = DistillConfig::new(4.0, 0.7);
        let mut grad = [0.0; 3];

        distill_loss(&teacher, &student, &labels, &config, &mut grad);
        // Gradient should push student logit[0] up (negative gradient → increase)
        assert!(
            grad[0] < grad[1],
            "Gradient should push class 0 logit up: grad[0]={}, grad[1]={}",
            grad[0],
            grad[1]
        );
    }

    #[test]
    fn test_distill_loss_alpha_zero_hard_only() {
        let teacher = [5.0, 1.0];
        let student = [1.0, 3.0];
        let labels = [1.0, 0.0];
        let config = DistillConfig::new(4.0, 0.0); // alpha=0 → hard loss only
        let mut grad = [0.0; 2];

        let (total, soft, hard) = distill_loss(&teacher, &student, &labels, &config, &mut grad);
        assert!(
            (total - hard).abs() < 1e-5,
            "alpha=0 → total should equal hard loss"
        );
        let _ = soft; // soft loss computed but not weighted
    }

    #[test]
    fn test_distill_loss_alpha_one_soft_only() {
        let teacher = [5.0, 1.0];
        let student = [1.0, 3.0];
        let labels = [1.0, 0.0];
        let config = DistillConfig::new(4.0, 1.0); // alpha=1 → soft loss only
        let mut grad = [0.0; 2];

        let (total, soft, _hard) = distill_loss(&teacher, &student, &labels, &config, &mut grad);
        assert!(
            (total - soft).abs() < 1e-4,
            "alpha=1 → total should equal soft loss"
        );
    }

    // --- DistillTrainer ---

    #[test]
    fn test_distill_trainer_new() {
        let t = DistillTrainer::new(DistillConfig::default(), 0.01, 50);
        assert_eq!(t.epochs, 50);
        assert!((t.learning_rate - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_distill_step_updates_weights() {
        let trainer = DistillTrainer::new(DistillConfig::default(), 0.1, 10);
        let teacher = [3.0, 0.5, 0.5];
        let student = [1.0, 1.0, 1.0];
        let labels = [1.0, 0.0, 0.0];
        let mut weights = [1.0, 1.0, 1.0];
        let mut grad = [0.0; 3];

        let original = weights;
        trainer.distill_step(&teacher, &student, &labels, &mut weights, &mut grad);

        // Weights should have changed
        let changed = weights
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "Weights should be updated after distill step");
    }

    // --- DistillEpochResult ---

    #[test]
    fn test_distill_epoch_result_clone() {
        let r = DistillEpochResult {
            epoch: 3,
            total_loss: 0.5,
            soft_loss: 0.3,
            hard_loss: 0.2,
        };
        let r2 = r.clone();
        assert_eq!(r2.epoch, 3);
        assert!((r2.total_loss - 0.5).abs() < 1e-6);
    }
}
