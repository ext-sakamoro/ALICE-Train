# Changelog

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
