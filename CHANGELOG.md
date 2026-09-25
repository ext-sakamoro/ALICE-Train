# Changelog

## [Unreleased]

### Added
- `tests/cuda_layer_oracle.rs` に 2 本追加 (GPU 必須、`--features cuda` 単独で実行): **実 shape** (hidden 2048 / heads 16/2 / head_dim 128 / seq 466) の中心差分 oracle と、同じ `CudaLayerWorkspace` で forward→backward を **2 組連続** 回した時の 2 組目を検査する oracle 小 shape 1 組だけの検査では下の `rmsnorm_backward` の使い回しバグが見えなかった
- `tests/layer_backward_oracle.rs` に CPU 版の実 shape 中心差分 oracle (`#[ignore]`、`-- --ignored` で実行、Mac 実測 86s)
- `tests/analytic_oracle.rs` — 閉形式 / f64 参照との突合 oracle 10 本 (CLAUDE.md § 解析解突合テスト規律、2026-09-17): activation backward の解析微分 / 数値微分、ternary linear の dx = γ·Wᵀg と dW = g·xᵀ (STE)、pre-norm BitLinear の有限差分、BitNet b1.58 ternary / int8 / int4 (tensor-wise + group-wise) の量子化閉形式と Δ/2 再構成 bound、bf16 round-to-nearest-even、dynamic loss scaling の閉形式 schedule、BLAS 3 variant の f64 参照 + rmsnorm、warmup-cosine LR
- CI `test` job (default features、alice-ml sibling checkout) — それまで fmt + actionlint のみで test が CI で走っていなかった

### Fixed (2026-09-25、CUDA layer path の勾配)
- **`rmsnorm_backward` の `d_input` が累積 (`+=`) で、使い回し buffer では前の層の勾配が積み上がっていた**: `cuda_layer_backward_ws[_vram]` は `CudaLayerWorkspace` の `d_input` / `d_pre_ffn_residual` を層をまたいで使い回すが zero 化しておらず (同じ箇所で norm weight 側だけは zero 化していた)、backward を 41 → 0 と回すと **最初の 1 層だけ正しく、2 層目以降が壊れる** 42 層で累積して logits が NaN になり学習が 4 sample 目で停止していた `d_input` を上書き semantics に変更 (全 caller が fresh buffer 渡しで累積に依存していないことを確認)、`d_weight` は従来どおり累積 実測: A6000 / MiniCPM5-2B / seq 466 の 2 組連続 oracle が 許容比 8.36 → 0.00
- `layer_backward` の `d_o_proj` が o_proj 適用**後**の `attn_out` を使っていた (正しくは適用前の `attn_out_raw`): `LayerCache` に `attn_out_raw` を追加 中心差分 oracle (`tests/layer_backward_oracle.rs`) が検出、符号ごと誤っていたが loss は下がっていた
- `cuda_matmul` の SwiGLU / softmax 系 9 箇所を `fast_sigmoid` (相対誤差 < 5%) から `exact_sigmoid` に: `silu'(x) = sig(1 + x(1−sig))` は零点付近で 5% 誤差が符号を反転させる elementwise は backward 175ms 中 5.5ms なので速度影響なし
- cross-entropy が NaN を隠していた: `probs[target].max(1e-10)` は `f32::max` が NaN で他方を返すため、logits が NaN でも loss が `-ln(1e-10) = 23.0259` という妥当な数字に化けていた logits の finite を assert で先に落とす

### Fixed (oracle 先行 red 2 → 修正)
- **`ternary_matvec_backward` が kernel の scale γ を落としていた**: forward は `y = γ·W·x` (alice-ml `ternary_matvec_kernel`) なのに backward は `Wᵀ·dy` → `from_ternary_scaled` の kernel で dx が 1/γ 倍 (γ ≈ 0.01〜0.5 の実 layer で 2〜100 倍過大)、`bitlinear_backward` も経由で同じ → `γ·Wᵀ·dy` に
- **tensor-wise int8 / int4 `calibrate_scale` が mean(|W|) を range にしていた**: `|w| > mean|w|` (layer の約半分) が ±mean に clip、実測 worst 再構成誤差 2.26 (正しい step 0.024 の Δ/2 = 0.012 のはず) → absmax (`max|W|`、group-wise 経路と同じ法則) に ternary は BitNet b1.58 の mean|W| のまま

## [0.2.0] - 2026-06-23

### Added
- Qwen3.5-9B QAT with CUDA DeltaNet (Gated DeltaNet hybrid Config)
- CUDA backward: atomicAdd 排除、warp shuffle reduction 修正
- Fused SwiGLU FFN — GPU 完結 (input H2D 1 回 + gemm×3 + GPU SiLU + D2H 1 回)
- FullAttention GQA CUDA 化 (50s/層 → 数 ms/層、~10000x 高速化)
- FullAttention backward CUDA 化 (GQA attention backward GPU 実行)
- Ternary export (.alice 形式) — マイルストーン 6 完了
- 推論エンジン + トークナイザー + CLI (.alice モデルでテキスト生成)
- Gradient Checkpointing (preload モードの活性化メモリ 27GB → 150MB)
- DeltaNet GPU fused GC — forward (VRAM state 保持) + backward (VRAM 直接参照)
- チェックポイント 15 分間隔の時間ベース保存 + 2-10 世代保持
- 崩壊検知強化
- RunPod QAT ラッパー (学習完了 / 崩壊時に Pod 自動停止)
- ボトルネック計測プロファイリング
- 実行環境自動記録 (run_record.json)

### Changed
- モデル名を `ALICE-Cognitive-9B-Ternary` に統一
- GPU 内完結 gemm + SwiGLU VRAM Zero-Copy パス
- All CPU bottleneck GPU 化 (RMSNorm / conv1d / L2norm / gates / gated-rmsnorm / conv1d-bwd)
- authors メールアドレスを `sakamoro@alicelaw.net` に統一

### Fixed
- resume 修正: preload モードでチェックポイント時に FP32 キャッシュ書き戻し
- runpod_qat.sh コンフリクトマーカー解消
- config 自動選択 + symlink フォールバック
- blockDim 超過修正 (RMSNorm / conv1d は CPU 版に戻し、小次元カーネルのみ GPU)

## [0.1.0] - 2026-03-06

### Added
- `activation` module: `relu_backward`, `silu_backward`, `gelu_backward`
- `backward` module: `ternary_matvec_backward`, `bitlinear_backward`, `ste_weight_grad`
- `trainer` module: `TrainableNetwork` trait, `Trainer`, `TrainConfig`, `EpochResult`
- Full numerical gradient verification for all backward functions
- 100+ tests covering happy path, boundary, error, and convergence
