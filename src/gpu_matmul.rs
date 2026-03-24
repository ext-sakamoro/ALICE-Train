//! GPU FP32 行列演算 — wgpu compute shader による高速 matmul。
//!
//! QAT 学習の最大ボトルネックである matmul_bt (C = A × B^T) を
//! GPU で実行する。Vulkan/Metal/DX12 バックエンドで動作。
//!
//! # 設計
//!
//! - ALICE-TRT の Trojan Horse アーキテクチャを FP32 学習に拡張
//! - 1回の dispatch で (M × K) × (N × K)^T → (M × N) を計算
//! - tiled workgroup 版: 各 workgroup が 16×16 タイルを担当

use crate::gpu::GpuContext;

/// FP32 matmul_bt WGSL shader: C[i][j] = sum_k A[i][k] * B[j][k]
///
/// Tiled: 各 workgroup が TILE_SIZE × TILE_SIZE の出力タイルを計算。
/// shared memory で入力タイルをキャッシュし、global memory アクセスを削減。
const MATMUL_BT_SHADER: &str = r"
struct Params {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16u;

var<workgroup> tileA: array<f32, 256>;  // 16 x 16
var<workgroup> tileB: array<f32, 256>;  // 16 x 16

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = gid.x;
    let col = gid.y;
    let lr = lid.x;
    let lc = lid.y;

    var acc: f32 = 0.0;

    let num_tiles = (params.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A: A[row][t*TILE + lc]
        let a_col = t * TILE + lc;
        if (row < params.M && a_col < params.K) {
            tileA[lr * TILE + lc] = A[row * params.K + a_col];
        } else {
            tileA[lr * TILE + lc] = 0.0;
        }

        // Load tile of B: B[col][t*TILE + lr]  (B^T access pattern)
        let b_col = t * TILE + lr;
        if (col < params.N && b_col < params.K) {
            tileB[lc * TILE + lr] = B[col * params.K + b_col];
        } else {
            tileB[lc * TILE + lr] = 0.0;
        }

        workgroupBarrier();

        // Accumulate: dot product of tileA row and tileB row
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            acc = acc + tileA[lr * TILE + k] * tileB[lc * TILE + k];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        C[row * params.N + col] = acc;
    }
}
";

/// Elementwise add shader: A[i] += B[i]
const ADD_SHADER: &str = r"
struct Params {
    len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) { return; }
    A[idx] = A[idx] + B[idx];
}
";

/// SiLU shader: x = x * sigmoid(x)
const SILU_SHADER: &str = r"
struct Params {
    len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) { return; }
    let x = data[idx];
    data[idx] = x / (1.0 + exp(-x));
}
";

/// Elementwise multiply shader: A[i] *= B[i]
const MUL_SHADER: &str = r"
struct Params {
    len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) { return; }
    A[idx] = A[idx] * B[idx];
}
";

/// Matmul パラメータ (uniform buffer)。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

/// Elementwise パラメータ。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ElemParams {
    len: u32,
    _pad: [u32; 3],
}

/// GPU FP32 行列演算エンジン。
pub struct GpuMatmul {
    matmul_pipeline: wgpu::ComputePipeline,
    matmul_layout: wgpu::BindGroupLayout,
    add_pipeline: wgpu::ComputePipeline,
    add_layout: wgpu::BindGroupLayout,
    silu_pipeline: wgpu::ComputePipeline,
    silu_layout: wgpu::BindGroupLayout,
    mul_pipeline: wgpu::ComputePipeline,
    mul_layout: wgpu::BindGroupLayout,
}

impl GpuMatmul {
    /// パイプラインをコンパイル。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();

        let (matmul_pipeline, matmul_layout) = Self::build_pipeline(
            device,
            MATMUL_BT_SHADER,
            "matmul_bt",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let elem_entries = &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        let (add_pipeline, add_layout) =
            Self::build_pipeline(device, ADD_SHADER, "add", elem_entries);
        let (mul_pipeline, mul_layout) =
            Self::build_pipeline(device, MUL_SHADER, "mul", elem_entries);

        let silu_entries = &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        let (silu_pipeline, silu_layout) =
            Self::build_pipeline(device, SILU_SHADER, "silu", silu_entries);

        Self {
            matmul_pipeline,
            matmul_layout,
            add_pipeline,
            add_layout,
            silu_pipeline,
            silu_layout,
            mul_pipeline,
            mul_layout,
        }
    }

    fn build_pipeline(
        device: &wgpu::Device,
        source: &str,
        label: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        (pipeline, layout)
    }

    /// GPU バッファを作成（storage, read-write）。
    fn create_storage_buf(ctx: &GpuContext, data: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        ctx.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// GPU バッファを作成（storage, read-only）。
    fn create_readonly_buf(ctx: &GpuContext, data: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        ctx.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// uniform バッファを作成。
    fn create_uniform_buf<T: bytemuck::Pod>(ctx: &GpuContext, data: &T) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        ctx.device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// GPU から結果をダウンロード。
    fn download(ctx: &GpuContext, buf: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let size = (len * 4) as u64;
        let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        ctx.queue().submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// C = A × B^T を GPU で計算。
    ///
    /// A: (m × k), B: (n × k), C: (m × n)
    ///
    /// B が `max_storage_buffer_binding_size` を超える場合、N方向をチャンク分割して
    /// 複数回 dispatch → 結果を結合する。
    pub fn matmul_bt(
        &self,
        ctx: &GpuContext,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let max_binding = ctx.device().limits().max_storage_buffer_binding_size as usize;
        let b_bytes = n * k * 4;
        let c_bytes = m * n * 4;

        // A, B, C いずれかが制限を超えるならチャンク分割
        if b_bytes > max_binding || c_bytes > max_binding {
            return self.matmul_bt_chunked(ctx, a, b, m, n, k, max_binding);
        }

        self.matmul_bt_single(ctx, a, b, m, n, k)
    }

    /// 単一 dispatch 版 matmul_bt（バッファが制限内の場合）。
    fn matmul_bt_single(
        &self,
        ctx: &GpuContext,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let buf_a = Self::create_readonly_buf(ctx, a);
        let buf_b = Self::create_readonly_buf(ctx, b);

        let output_size = m * n;
        let buf_c = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (output_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = MatmulParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            _pad: 0,
        };
        let buf_params = Self::create_uniform_buf(ctx, &params);

        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.matmul_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_params.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (m as u32 + 15) / 16;
            let wg_y = (n as u32 + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        ctx.queue().submit(Some(encoder.finish()));

        Self::download(ctx, &buf_c, output_size)
    }

    /// チャンク分割版 matmul_bt — B の N 方向を分割して複数回 dispatch。
    ///
    /// C[:,chunk_start..chunk_end] = A × B[chunk_start..chunk_end]^T
    fn matmul_bt_chunked(
        &self,
        ctx: &GpuContext,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        max_binding: usize,
    ) -> Vec<f32> {
        // チャンクサイズ: B の1行 = k*4 bytes, C の1行出力 = m*4 bytes
        // B チャンク: chunk_n 行 × k 要素 ≤ max_binding / 4
        // C チャンク: m × chunk_n 要素 ≤ max_binding / 4
        let max_elems = max_binding / 4;
        let chunk_n_by_b = max_elems / k; // B の行数制限
        let chunk_n_by_c = max_elems / m; // C の列数制限
        let chunk_n = chunk_n_by_b.min(chunk_n_by_c).max(1);

        let mut result = vec![0.0f32; m * n];
        let mut offset = 0;

        while offset < n {
            let end = (offset + chunk_n).min(n);
            let cn = end - offset;

            // B のチャンク: rows [offset..end], 各行 k 要素
            let b_chunk: Vec<f32> = b[offset * k..end * k].to_vec();

            // この chunk の matmul: C_chunk = A × B_chunk^T, shape (m × cn)
            let c_chunk = self.matmul_bt_single(ctx, a, &b_chunk, m, cn, k);

            // 結果を合体: result[i][offset..end] = c_chunk[i][0..cn]
            for i in 0..m {
                result[i * n + offset..i * n + end].copy_from_slice(&c_chunk[i * cn..(i + 1) * cn]);
            }

            offset = end;
        }

        result
    }

    /// SwiGLU FFN を GPU で実行。
    ///
    /// gate = input × gate_proj^T, up = input × up_proj^T
    /// output = (SiLU(gate) ⊙ up) × down_proj^T
    pub fn swiglu_ffn(
        &self,
        ctx: &GpuContext,
        input: &[f32],
        gate_proj: &[f32],
        up_proj: &[f32],
        down_proj: &[f32],
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Vec<f32> {
        // gate = input × gate_proj^T
        let gate = self.matmul_bt(ctx, input, gate_proj, seq_len, intermediate_dim, hidden_dim);
        // up = input × up_proj^T
        let up = self.matmul_bt(ctx, input, up_proj, seq_len, intermediate_dim, hidden_dim);

        // SiLU(gate) ⊙ up on GPU
        let total = seq_len * intermediate_dim;
        let buf_gate = Self::create_storage_buf(ctx, &gate);
        let buf_up = Self::create_readonly_buf(ctx, &up);

        // SiLU in-place on gate
        let silu_params = ElemParams {
            len: total as u32,
            _pad: [0; 3],
        };
        let buf_silu_params = Self::create_uniform_buf(ctx, &silu_params);
        let silu_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.silu_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_gate.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_silu_params.as_entire_binding(),
                },
            ],
        });

        // Mul: gate *= up
        let mul_params = ElemParams {
            len: total as u32,
            _pad: [0; 3],
        };
        let buf_mul_params = Self::create_uniform_buf(ctx, &mul_params);
        let mul_bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.mul_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_gate.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_up.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_mul_params.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            // SiLU
            pass.set_pipeline(&self.silu_pipeline);
            pass.set_bind_group(0, &silu_bg, &[]);
            pass.dispatch_workgroups((total as u32 + 255) / 256, 1, 1);
        }
        ctx.queue().submit(Some(encoder.finish()));

        let mut encoder = ctx.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            // Mul
            pass.set_pipeline(&self.mul_pipeline);
            pass.set_bind_group(0, &mul_bg, &[]);
            pass.dispatch_workgroups((total as u32 + 255) / 256, 1, 1);
        }
        ctx.queue().submit(Some(encoder.finish()));

        // intermediate = SiLU(gate) * up (now in buf_gate)
        let intermediate = Self::download(ctx, &buf_gate, total);

        // output = intermediate × down_proj^T
        self.matmul_bt(
            ctx,
            &intermediate,
            down_proj,
            seq_len,
            hidden_dim,
            intermediate_dim,
        )
    }
}

// ── GPU レイヤー Forward ──────────────────────────────────────────────────

use crate::llama::{LlamaConfig, LlamaLayerWeights};
use crate::llama_forward::{apply_rope, gqa_attention, rmsnorm, LayerCache};

/// GPU 加速版 Transformer レイヤー forward。
///
/// QKV/O projection と SwiGLU FFN を GPU matmul_bt で実行。
/// RMSNorm, RoPE, GQA Attention は CPU（小規模なので十分高速）。
pub fn gpu_layer_forward(
    gpu: &GpuMatmul,
    ctx: &GpuContext,
    input: &mut Vec<f32>,
    weights: &LlamaLayerWeights,
    config: &LlamaConfig,
    seq_len: usize,
) -> LayerCache {
    let hidden_dim = config.hidden_dim;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let intermediate_dim = config.intermediate_dim;

    // 1. 残差保存 + Attention RMSNorm (CPU)
    let residual_attn = input.clone();
    let mut normed = input.clone();
    rmsnorm(&mut normed, &weights.attn_norm, hidden_dim, config.norm_eps);
    let normed_attn = normed.clone();

    // 2. QKV projection (GPU matmul_bt)
    let q = gpu.matmul_bt(
        ctx,
        &normed,
        &weights.q_proj,
        seq_len,
        num_heads * head_dim,
        hidden_dim,
    );
    let k = gpu.matmul_bt(
        ctx,
        &normed,
        &weights.k_proj,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );
    let v = gpu.matmul_bt(
        ctx,
        &normed,
        &weights.v_proj,
        seq_len,
        num_kv_heads * head_dim,
        hidden_dim,
    );

    let mut q = q;
    let mut k = k;

    // 3. RoPE (CPU)
    apply_rope(&mut q, num_heads, head_dim, seq_len, config.rope_theta);
    apply_rope(&mut k, num_kv_heads, head_dim, seq_len, config.rope_theta);

    // 4. GQA Attention (CPU — seq_len^2 × head_dim, 小規模)
    let mut attn_out_raw = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut attn_weights = vec![0.0f32; num_heads * seq_len * seq_len];
    gqa_attention(
        &q,
        &k,
        &v,
        &mut attn_out_raw,
        &mut attn_weights,
        config,
        seq_len,
    );

    // 5. O projection (GPU matmul_bt)
    let attn_out = gpu.matmul_bt(
        ctx,
        &attn_out_raw,
        &weights.o_proj,
        seq_len,
        hidden_dim,
        num_heads * head_dim,
    );

    // 6. Residual add (CPU)
    for i in 0..input.len() {
        input[i] = residual_attn[i] + attn_out[i];
    }

    // 7. FFN: 残差保存 + RMSNorm (CPU)
    let residual_ffn = input.clone();
    let mut normed_ffn_buf = input.clone();
    rmsnorm(
        &mut normed_ffn_buf,
        &weights.ffn_norm,
        hidden_dim,
        config.norm_eps,
    );
    let normed_ffn = normed_ffn_buf.clone();

    // 8. SwiGLU FFN (GPU — 3 matmul_bt + SiLU + mul)
    let ffn_out = gpu.swiglu_ffn(
        ctx,
        &normed_ffn_buf,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
        seq_len,
        hidden_dim,
        intermediate_dim,
    );

    // gate/up/gate_silu は backward 用に再計算が必要（GPU版では CPU で再計算する）
    // ストリーミングモードでは backward 時に重みを再読み込みするので、
    // ここでは空のプレースホルダーでなく、CPU で gate/up を再計算して LayerCache に入れる
    let gate = gpu.matmul_bt(
        ctx,
        &normed_ffn_buf,
        &weights.gate_proj,
        seq_len,
        intermediate_dim,
        hidden_dim,
    );
    let up = gpu.matmul_bt(
        ctx,
        &normed_ffn_buf,
        &weights.up_proj,
        seq_len,
        intermediate_dim,
        hidden_dim,
    );
    let gate_silu: Vec<f32> = gate.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    // 9. Residual add (CPU)
    for i in 0..input.len() {
        input[i] = residual_ffn[i] + ffn_out[i];
    }

    LayerCache {
        residual_attn,
        normed_attn,
        q,
        k,
        v,
        attn_weights,
        attn_out,
        residual_ffn,
        normed_ffn,
        gate,
        up,
        gate_silu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_bt_identity() {
        let ctx = match GpuContext::new_blocking() {
            Some(c) => c,
            None => return,
        };
        let engine = GpuMatmul::new(&ctx);

        // A = [[1, 2], [3, 4]], B = [[1, 0], [0, 1]] (identity)
        // C = A × B^T = A
        let a = vec![1.0, 2.0, 3.0, 4.0f32];
        let b = vec![1.0, 0.0, 0.0, 1.0f32];
        let c = engine.matmul_bt(&ctx, &a, &b, 2, 2, 2);
        assert!((c[0] - 1.0).abs() < 1e-4, "got {}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-4, "got {}", c[1]);
        assert!((c[2] - 3.0).abs() < 1e-4, "got {}", c[2]);
        assert!((c[3] - 4.0).abs() < 1e-4, "got {}", c[3]);
    }

    #[test]
    fn test_matmul_bt_large() {
        let ctx = match GpuContext::new_blocking() {
            Some(c) => c,
            None => return,
        };
        let engine = GpuMatmul::new(&ctx);

        // 64 × 128 × 128 → 64 × 64
        let m = 64;
        let k = 128;
        let n = 64;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.1).collect();

        let gpu_result = engine.matmul_bt(&ctx, &a, &b, m, n, k);

        // CPU reference
        let mut cpu_result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for h in 0..k {
                    sum += a[i * k + h] * b[j * k + h];
                }
                cpu_result[i * n + j] = sum;
            }
        }

        for idx in 0..m * n {
            assert!(
                (gpu_result[idx] - cpu_result[idx]).abs() < 0.1,
                "mismatch at {idx}: gpu={} cpu={}",
                gpu_result[idx],
                cpu_result[idx]
            );
        }
    }
}
