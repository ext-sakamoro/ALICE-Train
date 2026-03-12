//! GPU コンテキスト — wgpu Device/Queue の初期化と管理。
//!
//! ALICE-Train の GPU アクセラレーション基盤。
//! `GpuContext` はブロッキング初期化を行い、compute pipeline の作成に必要な
//! `Device` / `Queue` を保持する。
//!
//! # Feature Gate
//!
//! `gpu` feature が有効な場合のみコンパイルされる。
//!
//! ```toml
//! [dependencies]
//! alice-train = { path = ".", features = ["gpu"] }
//! ```

use std::sync::Arc;
use wgpu::{Device, Queue};

/// GPU デバイスとキューを保持するコンテキスト。
///
/// `new_blocking()` で同期的に初期化される。
/// 内部で `pollster::block_on` を使用するため、既存の async ランタイム内から
/// 呼ぶ場合は注意が必要（通常は学習スクリプトのメインスレッドから呼ぶ想定）。
pub struct GpuContext {
    /// wgpu Device
    device: Arc<Device>,
    /// wgpu Queue
    queue: Arc<Queue>,
}

impl GpuContext {
    /// ブロッキングで GPU コンテキストを初期化。
    ///
    /// 利用可能な GPU アダプタを自動選択し、Device/Queue を作成する。
    /// GPU が見つからない場合は `None` を返す。
    #[must_use]
    pub fn new_blocking() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    /// 非同期で GPU コンテキストを初期化。
    async fn new_async() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ALICE-Train GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .ok()?;

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// wgpu Device への参照。
    #[inline]
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// wgpu Queue への参照。
    #[inline]
    #[must_use]
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Device の Arc クローン（パイプライン間共有用）。
    #[inline]
    #[must_use]
    pub fn device_arc(&self) -> Arc<Device> {
        Arc::clone(&self.device)
    }

    /// Queue の Arc クローン。
    #[inline]
    #[must_use]
    pub fn queue_arc(&self) -> Arc<Queue> {
        Arc::clone(&self.queue)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_context_creation() {
        // CI 環境では GPU がない場合があるため、None でもパス
        let ctx = GpuContext::new_blocking();
        if let Some(ctx) = &ctx {
            // Device/Queue が有効であることを確認
            let _dev = ctx.device();
            let _q = ctx.queue();
            let _arc_dev = ctx.device_arc();
            let _arc_q = ctx.queue_arc();
        }
        // GPU なし環境でも panic しないことを確認
    }
}
