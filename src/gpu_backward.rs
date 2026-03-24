//! GPU ternary backward — wgpu compute shader による逆伝播。
//!
//! CPU 版 `ternary_matvec_backward` と同一の計算を GPU 上で実行する。
//! 大規模行列（7B パラメータ等）において、CPU 版の O(nnz) ループを
//! GPU の大規模並列処理で高速化する。
//!
//! # アーキテクチャ
//!
//! ```text
//! CPU                              GPU
//! ┌─────────────────┐              ┌──────────────────────┐
//! │ grad_output      │─── upload ──▶│ Storage Buffer (RO)  │
//! │ plus_bits        │─── upload ──▶│ Storage Buffer (RO)  │
//! │ minus_bits       │─── upload ──▶│ Storage Buffer (RO)  │
//! │ params           │─── upload ──▶│ Uniform Buffer       │
//! │                  │              │                      │
//! │ grad_input       │◀── download ─│ Storage Buffer (RW)  │
//! └─────────────────┘              └──────────────────────┘
//!                                   │ ternary_backward.wgsl │
//!                                   │ @compute @wg(256)     │
//!                                   └──────────────────────┘
//! ```

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::{self, util::DeviceExt};

use alice_ml::ops::TernaryWeightKernel;

use crate::gpu::GpuContext;

/// Compute shader の uniform パラメータ。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    out_features: u32,
    in_features: u32,
    words_per_row: u32,
    _pad: u32,
}

/// GPU 上で ternary backward を実行するエンジン。
///
/// `GpuContext` を共有し、compute pipeline をキャッシュする。
/// `ternary_matvec_backward()` が CPU 版と同一の結果を返すことが保証される。
pub struct GpuBackwardEngine {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBackwardEngine {
    /// `GpuContext` から backward エンジンを構築。
    ///
    /// WGSL シェーダーをコンパイルし、compute pipeline を作成する。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device_arc();
        let queue = ctx.queue_arc();

        // WGSL シェーダーを埋め込み
        let shader_source = include_str!("../shaders/ternary_backward.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ternary_backward.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Bind group layout: uniform + 3 storage(RO) + 1 storage(RW)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ternary_backward_layout"),
            entries: &[
                // binding 0: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: grad_output (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: plus_bits (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: minus_bits (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: grad_input (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ternary_backward_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ternary_backward_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        }
    }

    /// GPU で ternary matvec backward を実行: dx = W^T * dy
    ///
    /// CPU 版 `ternary_matvec_backward` と同一の結果を返す。
    /// データ転送 (CPU→GPU→CPU) を含むため、小規模行列では CPU 版の方が速い。
    /// 目安: `in_features * out_features >= 65536` (256x256) 以上で GPU が有利。
    ///
    /// # 引数
    ///
    /// - `grad_output` — 出力勾配 dy (長さ = `out_features`)
    /// - `weights` — `TernaryWeightKernel`
    /// - `grad_input` — 入力勾配 dx の書き込み先 (長さ = `in_features`)
    ///
    /// # Panics
    ///
    /// `grad_output.len() != out_features` または `grad_input.len() != in_features` の場合。
    pub fn ternary_matvec_backward(
        &self,
        grad_output: &[f32],
        weights: &TernaryWeightKernel,
        grad_input: &mut [f32],
    ) {
        let out_features = weights.out_features();
        let in_features = weights.in_features();
        let words_per_row = weights.words_per_row();

        assert_eq!(grad_output.len(), out_features);
        assert_eq!(grad_input.len(), in_features);

        // Uniform パラメータ
        let params = GpuParams {
            out_features: out_features as u32,
            in_features: in_features as u32,
            words_per_row: words_per_row as u32,
            _pad: 0,
        };

        // バッファ作成
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params_buf"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let grad_output_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("grad_output_buf"),
                contents: bytemuck::cast_slice(grad_output),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let plus_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("plus_bits_buf"),
                contents: bytemuck::cast_slice(weights.plus_bits()),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let minus_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("minus_bits_buf"),
                contents: bytemuck::cast_slice(weights.minus_bits()),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (in_features * std::mem::size_of::<f32>()) as u64;
        let grad_input_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_input_buf"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // readback 用バッファ
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buf"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ternary_backward_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: plus_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: minus_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grad_input_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let workgroup_count = in_features.div_ceil(256) as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ternary_backward_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ternary_backward_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // GPU→CPU コピー
        encoder.copy_buffer_to_buffer(&grad_input_buf, 0, &staging_buf, 0, output_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // 同期読み出し
        let buffer_slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("GPU map callback not received")
            .expect("GPU buffer map failed");

        let data = buffer_slice.get_mapped_range();
        let result: &[f32] = bytemuck::cast_slice(&data);
        grad_input.copy_from_slice(result);
        drop(data);
        staging_buf.unmap();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use alice_ml::ops::TernaryWeightKernel;

    /// GPU テスト用ヘルパー: GPU が利用できない場合はスキップ
    fn try_engine() -> Option<GpuBackwardEngine> {
        let ctx = GpuContext::new_blocking()?;
        Some(GpuBackwardEngine::new(&ctx))
    }

    #[test]
    fn gpu_backward_identity() {
        let Some(engine) = try_engine() else {
            return;
        };
        let kernel = TernaryWeightKernel::from_ternary(&[1, 0, 0, 1], 2, 2);
        let grad_out = [3.0_f32, 7.0];
        let mut grad_in = [0.0_f32; 2];
        engine.ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - 3.0).abs() < 1e-5);
        assert!((grad_in[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_backward_mixed() {
        let Some(engine) = try_engine() else {
            return;
        };
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let grad_out = [2.0_f32, 3.0];
        let mut grad_in = [0.0_f32; 2];
        engine.ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - 2.0).abs() < 1e-5);
        assert!((grad_in[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_backward_3x3() {
        let Some(engine) = try_engine() else {
            return;
        };
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1, 1, -1, 0, -1, 1], 3, 3);
        let grad_out = [1.0_f32, 1.0, 1.0];
        let mut grad_in = [0.0_f32; 3];
        engine.ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - 2.0).abs() < 1e-5);
        assert!((grad_in[1] - (-1.0)).abs() < 1e-5);
        assert!((grad_in[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_backward_all_zeros() {
        let Some(engine) = try_engine() else {
            return;
        };
        let kernel = TernaryWeightKernel::from_ternary(&[0, 0, 0, 0], 2, 2);
        let grad_out = [5.0_f32, 3.0];
        let mut grad_in = [0.0_f32; 2];
        engine.ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0]).abs() < 1e-5);
        assert!((grad_in[1]).abs() < 1e-5);
    }

    #[test]
    fn gpu_backward_all_plus() {
        let Some(engine) = try_engine() else {
            return;
        };
        let kernel = TernaryWeightKernel::from_ternary(&[1, 1, 1, 1], 2, 2);
        let grad_out = [2.0_f32, 3.0];
        let mut grad_in = [0.0_f32; 2];
        engine.ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - 5.0).abs() < 1e-5);
        assert!((grad_in[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_backward_all_minus() {
        let Some(engine) = try_engine() else {
            return;
        };
        let kernel = TernaryWeightKernel::from_ternary(&[-1, -1, -1, -1], 2, 2);
        let grad_out = [2.0_f32, 3.0];
        let mut grad_in = [0.0_f32; 2];
        engine.ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - (-5.0)).abs() < 1e-5);
        assert!((grad_in[1] - (-5.0)).abs() < 1e-5);
    }

    #[test]
    fn gpu_backward_matches_cpu() {
        let Some(engine) = try_engine() else {
            return;
        };
        // CPU 版と GPU 版が一致することを検証
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1, -1, 1, 0, -1, 1], 3, 3);
        let grad_out = [1.5_f32, -2.0, 0.7];

        let mut cpu_result = [0.0_f32; 3];
        crate::ternary_matvec_backward(&grad_out, &kernel, &mut cpu_result);

        let mut gpu_result = [0.0_f32; 3];
        engine.ternary_matvec_backward(&grad_out, &kernel, &mut gpu_result);

        for i in 0..3 {
            assert!(
                (cpu_result[i] - gpu_result[i]).abs() < 1e-5,
                "dim {i}: cpu={}, gpu={}",
                cpu_result[i],
                gpu_result[i]
            );
        }
    }

    #[test]
    fn gpu_backward_rectangular_2x4() {
        let Some(engine) = try_engine() else {
            return;
        };
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1, 0, 1, -1, 0], 2, 4);
        let grad_out = [3.0_f32, 5.0];
        let mut grad_in = [0.0_f32; 4];
        engine.ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - 3.0).abs() < 1e-5);
        assert!((grad_in[1] - 2.0).abs() < 1e-5);
        assert!((grad_in[2] - (-5.0)).abs() < 1e-5);
        assert!((grad_in[3] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_backward_4x4_matches_cpu() {
        let Some(engine) = try_engine() else {
            return;
        };
        let vals: Vec<i8> = vec![1, -1, 0, 1, 0, 1, -1, 0, -1, 0, 1, 1, 1, -1, -1, 0];
        let kernel = TernaryWeightKernel::from_ternary(&vals, 4, 4);
        let grad_out = [1.0_f32, 2.0, 3.0, 4.0];

        let mut cpu_result = [0.0_f32; 4];
        crate::ternary_matvec_backward(&grad_out, &kernel, &mut cpu_result);

        let mut gpu_result = [0.0_f32; 4];
        engine.ternary_matvec_backward(&grad_out, &kernel, &mut gpu_result);

        for i in 0..4 {
            assert!(
                (cpu_result[i] - gpu_result[i]).abs() < 1e-5,
                "dim {i}: cpu={}, gpu={}",
                cpu_result[i],
                gpu_result[i]
            );
        }
    }

    #[test]
    fn gpu_backward_single_element() {
        let Some(engine) = try_engine() else {
            return;
        };
        let kernel = TernaryWeightKernel::from_ternary(&[1], 1, 1);
        let grad_out = [42.0_f32];
        let mut grad_in = [0.0_f32; 1];
        engine.ternary_matvec_backward(&grad_out, &kernel, &mut grad_in);
        assert!((grad_in[0] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_backward_negative_grad() {
        let Some(engine) = try_engine() else {
            return;
        };
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let grad_out = [-1.0_f32, -2.0];

        let mut cpu_result = [0.0_f32; 2];
        crate::ternary_matvec_backward(&grad_out, &kernel, &mut cpu_result);

        let mut gpu_result = [0.0_f32; 2];
        engine.ternary_matvec_backward(&grad_out, &kernel, &mut gpu_result);

        for i in 0..2 {
            assert!(
                (cpu_result[i] - gpu_result[i]).abs() < 1e-5,
                "dim {i}: cpu={}, gpu={}",
                cpu_result[i],
                gpu_result[i]
            );
        }
    }
}
