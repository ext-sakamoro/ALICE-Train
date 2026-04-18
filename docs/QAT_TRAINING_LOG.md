# ALICE-Cognitive-9B-Ternary QAT Training Log

Qwen3.5-9B を 1.58-bit Ternary に量子化再学習（QAT）した全記録。

## 概要

| 項目 | 値 |
|------|-----|
| ベースモデル | Qwen3.5-9B (Gated DeltaNet hybrid, 120B超ベンチ性能) |
| 量子化方式 | 1.58-bit Ternary QAT (STE backward) |
| 成果物 | ALICE-Cognitive-9B-Ternary.alice (5.4GB) |
| 環境 | RunPod A100 PCIe 80GB |
| 総ステップ | 5,000 (完走) |
| 学習期間 | 2026-03-31 〜 2026-04-09 (約9日間) |
| 学習時間 | 294,890s (約3.4日 = 81.9時間) |
| 最終Loss | 13.44 (step 4999) |
| 学習率 | cosine decay, peak 3.0e-5 → min 1.0e-6 |
| 実装 | Pure Rust + cudarc (PyTorch不使用) |
| 推定コスト | ~$97 (A100 PCIe $1.19/h × 81.9h) |

## 前回の失敗 (2026-03 旧run)

- step 3213 で Loss 6.6 → 23.0 に崩壊
- チェックポイント消失（保持世代不足）
- grad clipping 未実装が崩壊の主因

## 今回の run (2026-03-31 〜 2026-04-08)

### Phase 1: 再スタート — step 0〜200 (3/31)

- `qat_qwen35_9b_a100_safe.json` で再開（max_grad_norm=1.0 追加）
- warmup 200 steps, lr 0 → 3.0e-5
- 速度: ~135s/step
- Loss: 13.3〜13.5 で安定推移

### Phase 2: 速度低下問題 #1 — step 195 (3/31)

**症状**: チェックポイント保存後に速度が 135s → 186s に恒久低下（+37%）

**原因調査**:
- GPU メモリパターンは安定（per-step alloc/free なし）
- FP32 キャッシュ書き戻しは CPU/ディスク I/O のみ
- CUDA メモリフラグメンテーションが疑われた

**対策**: 再起動で回復。kill 時に FP32 キャッシュ書き込み中断で `layer_29_dn.fp32` 破損 → `build_cache` に既存キャッシュスキップ追加で修復。

**教訓**: `--features qat-cuda` を忘れると CUDA 無しバイナリになる（CUDA cuBLAS メッセージで確認必須）

### Phase 3: ページキャッシュ汚染対策 — step 1371 (4/1〜4/2)

**症状**: チェックポイント保存のたびに速度低下が再発

**原因**: 26GB の FP32 キャッシュ書き込みが OS ページキャッシュを汚染し、学習ループの CPU メモリアクセス効率が低下

**修正**:
```rust
// fp32_cache.rs に追加
#[cfg(target_os = "linux")]
pub fn drop_page_cache(base_dir: &str, config: &Qwen35Config) {
    use std::os::unix::io::AsRawFd;
    for i in 0..config.num_hidden_layers {
        let path = layer_path(base_dir, i, config.layer_type(i));
        if let Ok(f) = std::fs::File::open(&path) {
            unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED); }
        }
    }
}
```

**結果**: 186s → 135s に回復し安定維持。チェックポイント前後で速度差ゼロ。

### Phase 4: ETA 計算修正 (4/3)

**症状**: resume 時の ETA 表示が大幅に楽観的（実際の 1/5 程度の値）

**原因**: `(global_step + 1) / elapsed` で計算していたが、resume 時は `global_step` がゼロからではないため `steps_per_sec` が過大に算出

**修正**: `resume_start_step` 変数を追加し `(global_step - resume_start_step) / elapsed` に変更

### Phase 5: THP compaction ストール — 根本原因特定 (4/5)

**症状**: `drop_page_cache` 適用後も再び速度低下が発生（135s → 179s）。チェックポイント保存の 2 ステップ後に急変。

**調査**:
- `perf stat` はコンテナで権限不足（使用不可）
- `/proc/vmstat` で `compact_stall: 40,068,033` を発見（4000万回のメモリ compaction ストール）
- `/proc/PID/smaps_rollup` で `AnonHugePages: 32GB`（RSS の 76% が 2MB Transparent Huge Pages）
- cgroup メモリ制限: 116GB、使用量: 60GB（余裕あり）
- ページキャッシュ (dirty pages) は正常

**根本原因**: Transparent Huge Pages (THP) のメモリ compaction

1. プロセスの 42GB RSS のうち 32GB が AnonHugePages (2MB pages)
2. チェックポイント保存で 26GB を順次ディスク書き込み → カーネルが大量ページを dirty 化
3. `khugepaged` がバックグラウンドでメモリ compaction 実行（2MB 連続領域の再確保）
4. compaction 完了後も 2MB pages のレイアウトが非最適化状態 → TLB ミス率上昇
5. CPU → GPU 転送と CPU 計算が恒久的に 33% 遅化

**なぜ `drop_page_cache` で直らないか**: ファイルの page cache 解放は anonymous pages（heap 上の Vec<f32>）の THP 再配置とは無関係

**なぜ再起動で直るか**: プロセス再起動でカーネルが全メモリを再割り当て → 新規に連続 2MB ページが確保される

**修正**:
```rust
// 起動時に THP を無効化
#[cfg(target_os = "linux")]
unsafe {
    libc::prctl(libc::PR_SET_THP_DISABLE, 1, 0, 0, 0);
}

// チェックポイント後にヒープ圧縮
#[cfg(target_os = "linux")]
unsafe { libc::malloc_trim(0); }
```

**結果**: 179s → 127s（29% 高速化）。チェックポイント後も速度低下ゼロ。compact_stall 増加なし。

### Phase 6: Box SFTP バックアップ (4/4)

学習完了時・崩壊時に成果物を自動で Box SFTP にアップロードする仕組みを実装。

- 正常完了時: `.alice` モデル + 最新チェックポイント + メタデータ
- 崩壊時: 全チェックポイント (10世代) + FP32 キャッシュ (26GB) + メタデータ
- `scripts/upload_box.sh` — heredoc 方式で SFTP 接続（batch mode は認証失敗するため）
- パスワードは環境変数 `BOX_SFTP_PASS` から取得

### Phase 7: 安定走行 — step 3213 突破 (4/5〜)

- 前回崩壊した step 3213 を無事通過
- grad clipping (max_grad_norm=1.0) が崩壊防止に効果的
- Loss 13.2〜13.5 で安定推移、崩壊の兆候なし

### Phase 8: 完走 — step 4600〜5000 (4/8〜4/9)

- lr = 1.0e-6（最小値）まで cosine decay 完了
- Loss 13.2〜13.5 で最後まで安定、崩壊なし
- 速度 145-155s/step
- **step 5000 で学習完了** (4/9 早朝)
- 学習時間: 294,890s (81.9時間)

### Phase 9: エクスポートと退避 (4/9)

- `.alice` エクスポート初回失敗 — ディスク容量不足 (50GB overlay 100%)
- `/dev/shm` (RAM 58GB) に出力先変更して成功
- ファイルサイズ: **5.4GB** (5,768,802,226 bytes)
  - Embedding (BF16): 2.03 GB
  - lm_head (BF16): 2.03 GB
  - Ternary layers: 1.70 GB
  - FP32 (norms等): 4.23 MB
- 量子化パラメータ: 6,784,286,720 (67.8億)
- 圧縮率: 6.1x (vs FP32 35.28 GB)
- Box SFTP アップロード失敗 — 834MB/file でサイズ上限超過
- scp でローカル Mac に全ファイル転送完了
- RunPod Pod 停止

### ローカル保存先

| ファイル | パス | サイズ |
|---------|------|--------|
| .alice モデル | `~/ALICE-Train/models/ALICE-Cognitive-9B-Ternary.alice` | 5.4GB |
| FP32 キャッシュ | `~/ALICE-Train/checkpoints/qwen35_9b/fp32_cache/` | 25GB (32ファイル) |
| チェックポイント | `~/ALICE-Train/checkpoints/qwen35_9b/step_*.bin` | 10ファイル |
| メタデータ | `~/ALICE-Train/checkpoints/qwen35_9b/{resume_state,run_record,train_log}` | 65KB |
| ログ | `~/ALICE-Train/logs/qwen35_9b_qat.log` | 538KB |

## 速度最適化の全履歴

| # | 手法 | 効果 | step |
|---|------|------|------|
| L14 | Fused SwiGLU FFN (GPU完結) | FFN 409→155ms/層 | 初期 |
| L15 | FullAttention GQA CUDA化 | fwd 417→17s | 初期 |
| L16 | FullAttention backward CUDA化 | bwd 124→115s | 初期 |
| L17 | VRAM常駐 SwiGLU Zero-Copy | H2D/D2H ゼロ | 初期 |
| — | 上記複合 | **919s → 130s/step (7.1x)** | 初期 |
| L18 | `drop_page_cache` (posix_fadvise) | 186s → 135s 安定維持 | ~1371 |
| L19 | `build_cache` 既存スキップ | kill 後の破損からの回復 | ~210 |
| L20 | THP 無効化 (`prctl`) | **179s → 127s (29% 高速化)** | ~2886 |
| L21 | `malloc_trim` ヒープ圧縮 | フラグメンテーション防止 | ~2886 |

**総合**: 919s → 127s/step = **7.2 倍高速化**

## プロファイル (step 3100, 146s/step 時)

| フェーズ | 時間 | 割合 |
|---------|------|------|
| embed | 14ms | 0.01% |
| fwd_32layers | 15,949ms | 10.9% |
| lm_head | 1,471ms | 1.0% |
| loss | 2,306ms | 1.6% |
| bwd_lm_head | 1,125ms | 0.8% |
| **bwd_32layers** | **125,479ms** | **85.7%** |
| **total (1 micro-batch)** | **146,344ms** | 100% |

backward が支配的。grad_accum=16 のため 1 step = 16 micro-batch だが、ストリーミング方式で 1 micro-batch ずつ実行するため total ≈ micro-batch 時間。

## .alice エクスポート仕様

| 項目 | 方式 |
|------|------|
| ternary 重み | 2-bit パック (4 values/byte, LSB-first) |
| エンコード | `00`=0, `01`=+1, `10`=-1 |
| スケール | γ = mean(\|W\|) — projection 毎に FP32 1個 |
| embedding | BF16 (量子化しない) |
| RMSNorm | FP32 (量子化しない) |
| 圧縮率 | FP32 → 2bit = 16x |
| 推定サイズ | ~2.7GB (ternary ~2.1GB + embedding/norm ~0.6GB) |

## 環境詳細

| 項目 | 値 |
|------|-----|
| Pod | `ca6wcohkj5gqv7` RunPod A100 PCIe 80GB |
| OS | Ubuntu 22.04 (kernel 6.5.0-35) |
| CUDA | 12.5 (Driver 555.42.02) |
| RAM | 1TB (cgroup 制限 116GB) |
| CPU | 64 cores, ~3.9GHz boost |
| Rust | 0.2.0 (alice-train) |
| ビルド | `cargo build --release --features qat-cuda` |
| SSH | `ssh -i ~/.runpod/ssh/RunPod-Key-Go root@104.255.9.187 -p 11566` |
