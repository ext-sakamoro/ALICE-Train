//! ZeRO-Offload — オプティマイザ状態の CPU RAM オフロード。
//!
//! 70B クラスモデルの QAT で GPU VRAM が不足する場合、AdamW の
//! モーメント (m, v) を CPU RAM に配置し、勾配更新を CPU 上で行う。
//!
//! # 設計
//!
//! ```text
//! GPU                              CPU
//! ┌──────────────────┐             ┌──────────────────────┐
//! │ weights (FP32)   │◀── copy ───│ updated weights      │
//! │ gradients        │─── copy ──▶│ gradients            │
//! │                  │             │ m (1st moment)       │
//! │                  │             │ v (2nd moment)       │
//! │                  │             │ AdamW step()         │
//! └──────────────────┘             └──────────────────────┘
//! ```
//!
//! GPU forward/backward → 勾配を CPU にコピー → CPU 上で AdamW 更新 →
//! 更新済み重みを GPU にコピー（または CPU 上の重みを直接使用）。
//!
//! # メモリ節約
//!
//! 通常の AdamW: weights + gradients + m + v = 4N パラメータ分の VRAM
//! ZeRO-Offload: weights + gradients = 2N パラメータ分の VRAM
//!               m + v = 2N パラメータ分の CPU RAM
//!
//! → VRAM 使用量を約 50% 削減。

/// オフロード設定。
#[derive(Clone, Debug)]
pub struct OffloadConfig {
    /// AdamW β1 (1次モーメント減衰率)
    pub beta1: f32,
    /// AdamW β2 (2次モーメント減衰率)
    pub beta2: f32,
    /// AdamW weight decay
    pub weight_decay: f32,
    /// AdamW epsilon (数値安定性)
    pub eps: f32,
    /// 勾配クリッピング閾値 (None = クリッピングなし)
    pub max_grad_norm: Option<f32>,
}

impl Default for OffloadConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: Some(1.0),
        }
    }
}

/// VRAM/RAM メモリ使用量の見積もり。
#[derive(Clone, Debug)]
pub struct MemoryBudget {
    /// パラメータ数
    pub param_count: usize,
    /// GPU VRAM 使用量 (bytes): weights + gradients
    pub vram_bytes: usize,
    /// CPU RAM 使用量 (bytes): m + v
    pub ram_bytes: usize,
    /// オフロードなしの場合の VRAM 使用量 (bytes)
    pub vram_without_offload_bytes: usize,
}

impl MemoryBudget {
    /// パラメータ数からメモリ使用量を計算。
    #[must_use]
    pub fn estimate(param_count: usize) -> Self {
        let f32_size = std::mem::size_of::<f32>();
        // オフロード時: GPU = weights + gradients (2N)
        let vram_bytes = 2 * param_count * f32_size;
        // CPU = m + v (2N)
        let ram_bytes = 2 * param_count * f32_size;
        // オフロードなし: GPU = weights + gradients + m + v (4N)
        let vram_without_offload_bytes = 4 * param_count * f32_size;

        Self {
            param_count,
            vram_bytes,
            ram_bytes,
            vram_without_offload_bytes,
        }
    }

    /// VRAM 節約率 (0.0 - 1.0)。
    #[must_use]
    pub fn vram_savings_ratio(&self) -> f32 {
        if self.vram_without_offload_bytes == 0 {
            return 0.0;
        }
        1.0 - (self.vram_bytes as f32 / self.vram_without_offload_bytes as f32)
    }
}

/// ZeRO-Offload AdamW オプティマイザ。
///
/// モーメント (m, v) を CPU RAM 上に保持し、勾配更新を CPU で実行する。
/// GPU からコピーされた勾配を受け取り、更新済みの重み差分を返す。
pub struct OffloadOptimizer {
    /// 設定
    config: OffloadConfig,
    /// 1次モーメント (CPU RAM)
    m: Vec<f32>,
    /// 2次モーメント (CPU RAM)
    v: Vec<f32>,
    /// ステップカウンタ（bias correction 用）
    step_count: u64,
    /// パラメータ数
    param_count: usize,
}

impl OffloadOptimizer {
    /// 指定パラメータ数で初期化。
    ///
    /// m, v はゼロ初期化される。
    #[must_use]
    pub fn new(param_count: usize, config: OffloadConfig) -> Self {
        Self {
            config,
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
            step_count: 0,
            param_count,
        }
    }

    /// メモリ使用量の見積もりを取得。
    #[must_use]
    pub fn memory_budget(&self) -> MemoryBudget {
        MemoryBudget::estimate(self.param_count)
    }

    /// 現在のステップ数。
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// 勾配ノルムを計算。
    fn grad_norm(gradients: &[f32]) -> f32 {
        let sum_sq: f32 = gradients.iter().map(|&g| g * g).sum();
        sum_sq.sqrt()
    }

    /// 勾配クリッピング（in-place）。
    fn clip_gradients(&self, gradients: &mut [f32]) {
        if let Some(max_norm) = self.config.max_grad_norm {
            let norm = Self::grad_norm(gradients);
            if norm > max_norm {
                let scale = max_norm / (norm + self.config.eps);
                for g in gradients.iter_mut() {
                    *g *= scale;
                }
            }
        }
    }

    /// AdamW 更新ステップ。
    ///
    /// 勾配を受け取り、重みを in-place で更新する。
    /// m, v は CPU RAM 上で更新される。
    ///
    /// # 引数
    ///
    /// - `weights` — 現在の重み (更新される)
    /// - `gradients` — 勾配 (クリッピングが適用される場合がある)
    /// - `lr` — 学習率
    ///
    /// # Panics
    ///
    /// `weights.len()` または `gradients.len()` が `param_count` と異なる場合。
    pub fn step(&mut self, weights: &mut [f32], gradients: &mut [f32], lr: f32) {
        assert_eq!(weights.len(), self.param_count);
        assert_eq!(gradients.len(), self.param_count);

        // 勾配クリッピング
        self.clip_gradients(gradients);

        self.step_count += 1;
        let t = self.step_count as f32;

        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let wd = self.config.weight_decay;

        // Bias correction
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);

        for i in 0..self.param_count {
            let g = gradients[i];

            // モーメント更新
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;

            // Bias-corrected
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            // AdamW: weight decay は勾配に加算するのではなく、重みに直接適用
            weights[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * weights[i]);
        }
    }

    /// モーメントをリセット（学習率リスタート時など）。
    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.step_count = 0;
    }

    /// パラメータ数。
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.param_count
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn offload_config_default() {
        let cfg = OffloadConfig::default();
        assert!((cfg.beta1 - 0.9).abs() < 1e-6);
        assert!((cfg.beta2 - 0.999).abs() < 1e-6);
        assert!((cfg.eps - 1e-8).abs() < 1e-12);
        assert!((cfg.weight_decay - 0.01).abs() < 1e-6);
        assert!(cfg.max_grad_norm.is_some());
    }

    #[test]
    fn memory_budget_estimate() {
        // 7B パラメータの見積もり
        let budget = MemoryBudget::estimate(7_000_000_000);
        // VRAM: 2 * 7B * 4 bytes = 56 GB
        assert_eq!(budget.vram_bytes, 56_000_000_000);
        // RAM: 2 * 7B * 4 bytes = 56 GB
        assert_eq!(budget.ram_bytes, 56_000_000_000);
        // オフロードなし: 4 * 7B * 4 bytes = 112 GB
        assert_eq!(budget.vram_without_offload_bytes, 112_000_000_000);
        // 節約率: 50%
        assert!((budget.vram_savings_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn memory_budget_zero_params() {
        let budget = MemoryBudget::estimate(0);
        assert_eq!(budget.vram_bytes, 0);
        assert!((budget.vram_savings_ratio() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn offload_optimizer_creation() {
        let opt = OffloadOptimizer::new(100, OffloadConfig::default());
        assert_eq!(opt.param_count(), 100);
        assert_eq!(opt.step_count(), 0);
    }

    #[test]
    fn offload_optimizer_step_decreases_loss_proxy() {
        // 簡単な最適化: w を勾配方向に更新して値が変化することを確認
        let mut opt = OffloadOptimizer::new(2, OffloadConfig::default());
        let mut weights = [1.0_f32, -1.0];
        let mut grads = [0.5_f32, -0.3];
        let initial_weights = weights;

        opt.step(&mut weights, &mut grads, 0.01);

        // 重みが更新されていること
        assert_ne!(weights[0], initial_weights[0]);
        assert_ne!(weights[1], initial_weights[1]);
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn offload_optimizer_multi_step() {
        let mut opt = OffloadOptimizer::new(1, OffloadConfig::default());
        let mut w = [5.0_f32];

        // 10ステップ同じ勾配で更新
        for _ in 0..10 {
            let mut g = [1.0_f32];
            opt.step(&mut w, &mut g, 0.01);
        }

        assert_eq!(opt.step_count(), 10);
        // weight decay + 正の勾配 → 重みは減少方向
        assert!(w[0] < 5.0);
    }

    #[test]
    fn offload_optimizer_weight_decay() {
        // weight decay が効いていることを確認
        let config_wd = OffloadConfig {
            weight_decay: 0.1,
            max_grad_norm: None,
            ..OffloadConfig::default()
        };
        let config_no_wd = OffloadConfig {
            weight_decay: 0.0,
            max_grad_norm: None,
            ..OffloadConfig::default()
        };

        let mut opt_wd = OffloadOptimizer::new(1, config_wd);
        let mut opt_no_wd = OffloadOptimizer::new(1, config_no_wd);

        let mut w_wd = [10.0_f32];
        let mut w_no_wd = [10.0_f32];

        for _ in 0..5 {
            let mut g1 = [0.1_f32];
            let mut g2 = [0.1_f32];
            opt_wd.step(&mut w_wd, &mut g1, 0.01);
            opt_no_wd.step(&mut w_no_wd, &mut g2, 0.01);
        }

        // weight decay ありの方が重みが小さい（大きな初期値を縮小する効果）
        assert!(w_wd[0] < w_no_wd[0]);
    }

    #[test]
    fn offload_optimizer_gradient_clipping() {
        let config = OffloadConfig {
            max_grad_norm: Some(1.0),
            ..OffloadConfig::default()
        };
        let mut opt = OffloadOptimizer::new(2, config);
        let mut w = [0.0_f32, 0.0];

        // 大きな勾配
        let mut g = [100.0_f32, 100.0];
        opt.step(&mut w, &mut g, 0.01);

        // クリッピング後の勾配ノルムは max_grad_norm 以下
        // → 更新量が制限されていること
        assert!(w[0].abs() < 0.02); // クリッピングなしなら ~ -1.0
        assert!(w[1].abs() < 0.02);
    }

    #[test]
    fn offload_optimizer_no_clipping() {
        // クリッピングなしでも正常に動作することを検証
        let config = OffloadConfig {
            max_grad_norm: None,
            weight_decay: 0.0,
            ..OffloadConfig::default()
        };
        let mut opt = OffloadOptimizer::new(1, config);
        let mut w = [0.0_f32];
        let mut g = [1000.0_f32];

        opt.step(&mut w, &mut g, 0.01);

        // 更新が発生していること（Adam は勾配正規化するので ~ -0.01）
        assert!(w[0] < 0.0, "weight should decrease with positive gradient");
        assert!(w[0].is_finite());
    }

    #[test]
    fn offload_optimizer_reset() {
        let mut opt = OffloadOptimizer::new(2, OffloadConfig::default());
        let mut w = [1.0_f32, 2.0];
        let mut g = [0.5, 0.5];
        opt.step(&mut w, &mut g, 0.01);
        assert_eq!(opt.step_count(), 1);

        opt.reset();
        assert_eq!(opt.step_count(), 0);
    }

    #[test]
    fn offload_optimizer_zero_lr() {
        let config = OffloadConfig {
            weight_decay: 0.0,
            max_grad_norm: None,
            ..OffloadConfig::default()
        };
        let mut opt = OffloadOptimizer::new(2, config);
        let mut w = [3.0_f32, 7.0];
        let initial = w;
        let mut g = [1.0, 1.0];

        opt.step(&mut w, &mut g, 0.0);

        // lr=0 → 重みは変化しない
        assert!((w[0] - initial[0]).abs() < 1e-6);
        assert!((w[1] - initial[1]).abs() < 1e-6);
    }

    #[test]
    fn offload_optimizer_zero_gradients() {
        let config = OffloadConfig {
            weight_decay: 0.0,
            max_grad_norm: None,
            ..OffloadConfig::default()
        };
        let mut opt = OffloadOptimizer::new(2, config);
        let mut w = [3.0_f32, 7.0];
        let initial = w;
        let mut g = [0.0, 0.0];

        opt.step(&mut w, &mut g, 0.01);

        // 勾配ゼロ → 重みは変化しない
        assert!((w[0] - initial[0]).abs() < 1e-6);
        assert!((w[1] - initial[1]).abs() < 1e-6);
    }

    #[test]
    fn offload_memory_budget_from_optimizer() {
        let opt = OffloadOptimizer::new(1000, OffloadConfig::default());
        let budget = opt.memory_budget();
        assert_eq!(budget.param_count, 1000);
        assert_eq!(budget.vram_bytes, 8000); // 2 * 1000 * 4
        assert_eq!(budget.ram_bytes, 8000); // 2 * 1000 * 4
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn offload_optimizer_panics_weight_size_mismatch() {
        let mut opt = OffloadOptimizer::new(2, OffloadConfig::default());
        let mut w = [1.0_f32; 3]; // サイズ不一致
        let mut g = [0.5; 2];
        opt.step(&mut w, &mut g, 0.01);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn offload_optimizer_panics_grad_size_mismatch() {
        let mut opt = OffloadOptimizer::new(2, OffloadConfig::default());
        let mut w = [1.0_f32; 2];
        let mut g = [0.5; 3]; // サイズ不一致
        opt.step(&mut w, &mut g, 0.01);
    }

    #[test]
    fn offload_optimizer_convergence() {
        // f(x) = x^2 の最小化: 勾配 = 2x
        let config = OffloadConfig {
            weight_decay: 0.0,
            max_grad_norm: None,
            ..OffloadConfig::default()
        };
        let mut opt = OffloadOptimizer::new(1, config);
        let mut w = [5.0_f32];

        for _ in 0..5000 {
            let mut g = [2.0 * w[0]]; // df/dx = 2x
            opt.step(&mut w, &mut g, 0.01);
        }

        // x^2 の最小値は x=0 付近に収束
        assert!(w[0].abs() < 0.5, "should converge near 0, got {}", w[0]);
    }
}
