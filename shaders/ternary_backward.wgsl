// ALICE-Train: ternary matvec backward compute shader
//
// dx[i] = sum_j W[j][i] * dy[j]  (W^T * dy)
//
// ternary 重み W ∈ {-1, 0, +1} はビットプレーン形式:
//   plus_bits[row * words_per_row + w]:  bit i が立っていれば W[row][w*32+i] = +1
//   minus_bits[row * words_per_row + w]: bit i が立っていれば W[row][w*32+i] = -1
//
// 各ワークグループは in_features の一部分 (32要素 = 1 word) を担当し、
// 全行の grad_output を走査して grad_input に累積する。

// uniforms
struct Params {
    out_features: u32,
    in_features: u32,
    words_per_row: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read> plus_bits: array<u32>;
@group(0) @binding(3) var<storage, read> minus_bits: array<u32>;
@group(0) @binding(4) var<storage, read_write> grad_input: array<f32>;

// 各スレッドが grad_input の1要素を担当
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= params.in_features {
        return;
    }

    let word_idx = col / 32u;
    let bit_pos = col % 32u;
    let bit_mask = 1u << bit_pos;

    var acc: f32 = 0.0;

    for (var row: u32 = 0u; row < params.out_features; row = row + 1u) {
        let dy = grad_output[row];

        // dy == 0 のスキップ（GPU では分岐コスト vs メモリアクセスのトレードオフ、
        // 大規模行列では dy=0 は稀なのでスキップしない方が効率的な場合もある）

        let idx = row * params.words_per_row + word_idx;
        let p = plus_bits[idx];
        let m = minus_bits[idx];

        // +1 の場合: acc += dy
        if (p & bit_mask) != 0u {
            acc = acc + dy;
        }
        // -1 の場合: acc -= dy
        if (m & bit_mask) != 0u {
            acc = acc - dy;
        }
    }

    grad_input[col] = acc;
}
