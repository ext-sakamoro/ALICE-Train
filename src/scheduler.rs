//! 学習率スケジューラ — warmup + cosine decay。
//!
//! 7B クラスの QAT 学習では安定した学習率制御が不可欠。
//! warmup で勾配爆発を防ぎ、cosine decay で終盤の過学習を抑える。

use std::f32::consts::PI;

/// 学習率スケジューラのトレイト。
pub trait LrScheduler {
    /// 現在のステップに対応する学習率を返す。
    fn get_lr(&self, step: usize) -> f32;
}

/// Warmup + Cosine Decay スケジューラ。
///
/// ```text
/// lr
/// ^
/// |    /‾‾‾‾‾‾\
/// |   /         \
/// |  /           \_____ min_lr
/// | /
/// +-------------------------> step
///   warmup  decay
/// ```
#[derive(Clone, Debug)]
pub struct WarmupCosineScheduler {
    /// ピーク学習率。
    pub max_lr: f32,
    /// 最小学習率（cosine decay の下限）。
    pub min_lr: f32,
    /// warmup ステップ数。
    pub warmup_steps: usize,
    /// 全ステップ数（warmup 含む）。
    pub total_steps: usize,
}

impl WarmupCosineScheduler {
    /// 新しいスケジューラを構築する。
    ///
    /// # Panics
    ///
    /// `warmup_steps > total_steps` の場合。
    #[must_use]
    pub fn new(max_lr: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        assert!(
            warmup_steps <= total_steps,
            "warmup_steps ({warmup_steps}) must be <= total_steps ({total_steps})"
        );
        Self {
            max_lr,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }
}

impl LrScheduler for WarmupCosineScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total_steps {
            return self.min_lr;
        }

        if step < self.warmup_steps {
            // 線形 warmup: 0 → max_lr
            if self.warmup_steps == 0 {
                return self.max_lr;
            }
            self.max_lr * (step as f32 / self.warmup_steps as f32)
        } else {
            // Cosine decay: max_lr → min_lr
            let decay_steps = self.total_steps - self.warmup_steps;
            if decay_steps == 0 {
                return self.max_lr;
            }
            let progress = (step - self.warmup_steps) as f32 / decay_steps as f32;
            let cosine = (1.0 + (PI * progress).cos()) * 0.5;
            self.min_lr + (self.max_lr - self.min_lr) * cosine
        }
    }
}

/// 定数学習率スケジューラ（ベースライン用）。
#[derive(Clone, Debug)]
pub struct ConstantScheduler {
    /// 固定学習率。
    pub lr: f32,
}

impl ConstantScheduler {
    /// 新しい定数スケジューラを構築する。
    #[must_use]
    pub const fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LrScheduler for ConstantScheduler {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- WarmupCosineScheduler ---

    #[test]
    fn warmup_cosine_starts_at_zero() {
        let s = WarmupCosineScheduler::new(0.001, 0.0, 100, 1000);
        assert!((s.get_lr(0) - 0.0).abs() < 1e-8);
    }

    #[test]
    fn warmup_cosine_reaches_max_at_warmup_end() {
        let s = WarmupCosineScheduler::new(0.001, 0.0, 100, 1000);
        let lr = s.get_lr(100);
        assert!(
            (lr - 0.001).abs() < 1e-6,
            "lr at warmup end should be max_lr, got {lr}"
        );
    }

    #[test]
    fn warmup_cosine_linear_warmup() {
        let s = WarmupCosineScheduler::new(0.01, 0.0, 10, 100);
        let lr5 = s.get_lr(5);
        assert!(
            (lr5 - 0.005).abs() < 1e-6,
            "half warmup should give half max_lr, got {lr5}"
        );
    }

    #[test]
    fn warmup_cosine_ends_at_min_lr() {
        let s = WarmupCosineScheduler::new(0.001, 1e-5, 100, 1000);
        let lr = s.get_lr(1000);
        assert!(
            (lr - 1e-5).abs() < 1e-8,
            "lr at total_steps should be min_lr, got {lr}"
        );
    }

    #[test]
    fn warmup_cosine_beyond_total_steps() {
        let s = WarmupCosineScheduler::new(0.001, 1e-5, 100, 1000);
        let lr = s.get_lr(2000);
        assert!(
            (lr - 1e-5).abs() < 1e-8,
            "lr beyond total_steps should be min_lr"
        );
    }

    #[test]
    fn warmup_cosine_monotonically_decreasing_after_warmup() {
        let s = WarmupCosineScheduler::new(0.001, 0.0, 100, 1000);
        let mut prev = s.get_lr(100);
        for step in (101..=1000).step_by(10) {
            let lr = s.get_lr(step);
            assert!(
                lr <= prev + 1e-8,
                "lr should decrease after warmup: step={step}, prev={prev}, lr={lr}"
            );
            prev = lr;
        }
    }

    #[test]
    fn warmup_cosine_midpoint_value() {
        // cosine(pi/2) = 0 → midpoint should be (max+min)/2
        let s = WarmupCosineScheduler::new(0.01, 0.0, 0, 100);
        let lr = s.get_lr(50);
        assert!(
            (lr - 0.005).abs() < 1e-4,
            "midpoint lr should be ~0.005, got {lr}"
        );
    }

    #[test]
    fn warmup_cosine_no_warmup() {
        let s = WarmupCosineScheduler::new(0.001, 0.0, 0, 100);
        let lr0 = s.get_lr(0);
        assert!(
            (lr0 - 0.001).abs() < 1e-6,
            "no warmup: step 0 should be max_lr"
        );
    }

    #[test]
    fn warmup_cosine_all_warmup() {
        let s = WarmupCosineScheduler::new(0.001, 0.0, 100, 100);
        let lr50 = s.get_lr(50);
        assert!(
            (lr50 - 0.0005).abs() < 1e-6,
            "all warmup: step 50/100 should be 0.0005"
        );
        let lr100 = s.get_lr(100);
        assert!(
            (lr100 - 0.0).abs() < 1e-6,
            "all warmup: at total_steps should be min_lr"
        );
    }

    #[test]
    fn warmup_cosine_all_values_finite() {
        let s = WarmupCosineScheduler::new(0.001, 1e-6, 50, 500);
        for step in 0..=600 {
            let lr = s.get_lr(step);
            assert!(lr.is_finite(), "lr at step {step} is not finite");
            assert!(lr >= 0.0, "lr at step {step} is negative: {lr}");
        }
    }

    #[test]
    fn warmup_cosine_min_lr_respected() {
        let s = WarmupCosineScheduler::new(0.01, 0.001, 10, 100);
        for step in 0..=200 {
            let lr = s.get_lr(step);
            if step >= 10 {
                assert!(lr >= 0.001 - 1e-8, "lr at step {step} below min_lr: {lr}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "warmup_steps")]
    fn warmup_cosine_panics_if_warmup_exceeds_total() {
        let _ = WarmupCosineScheduler::new(0.001, 0.0, 200, 100);
    }

    // --- ConstantScheduler ---

    #[test]
    fn constant_always_returns_same_lr() {
        let s = ConstantScheduler::new(0.042);
        for step in [0, 1, 100, 10000] {
            assert!((s.get_lr(step) - 0.042).abs() < 1e-8);
        }
    }

    #[test]
    fn constant_clone() {
        let s = ConstantScheduler::new(0.01);
        let s2 = s.clone();
        assert!((s2.lr - 0.01).abs() < 1e-8);
    }

    #[test]
    fn constant_debug() {
        let s = ConstantScheduler::new(0.01);
        let d = format!("{s:?}");
        assert!(d.contains("0.01"));
    }
}
