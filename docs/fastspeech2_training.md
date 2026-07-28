# FastSpeech2 学習ガイド (Phase T.4a)

ALICE-TTS v2.0 Phase T.4a の FastSpeech2 モデルを JSUT / JVS で学習するための手順書

## 前提

- Rust 1.87+ (`rust-toolchain.toml` pin)
- Python 3.10+ (`pyopenjtalk` 依存、G2P 用)
- audio 24 kHz mono WAV
- Compute: RunPod A100 or Paperspace A100 (Mac は sanity check のみ)

## 準備手順

### 1. JSUT corpus ダウンロード (~5 GB)

```bash
mkdir -p data/jsut
cd data/jsut
wget https://ss-takashi.sakura.ne.jp/corpus/jsut/jsut_ver1.1.zip
unzip jsut_ver1.1.zip
```

### 2. Manifest jsonl 生成

```bash
pip install pyopenjtalk numpy scipy
python3 scripts/prepare_jsut_manifest.py \
    --jsut_root data/jsut/jsut_ver1.1 \
    --output data/jsut/manifest.jsonl \
    --tokenizer_vocab_file data/jsut/mora_vocab.txt
```

**Note**: `prepare_jsut_manifest.py` の duration は placeholder (80 ms/mora 固定) 精度が必要なら Montreal Forced Aligner (MFA) の出力に置換すること

### 3. Config 確認

`configs/fs2_jsut.json` を必要に応じて編集:
- `vocab_size`: manifest 生成時の mora vocab に合わせる (通常 100-200)
- `hidden_dim`: 256 (JSUT scale)、実運用は 384-512 推奨
- `batch_size`: A100 80GB なら 8-16、Mac なら 1
- `total_steps`: 300k step 目安 (~5-10 日 A100)

## Sanity Check (Mac / Local)

Manifest が無くても dummy 5 step training を実行できる:

```bash
cargo run --release --example fs2_train --features tts -- --config configs/fs2_jsut.json
```

期待出力:
```
[sanity] step 1 mel_loss=0.250000
[sanity] step 2 mel_loss=0.249xxx
...
[sanity] Done. Sanity check passed (no NaN/Inf).
```

## 本番学習 (RunPod / Paperspace)

### RunPod A100 セットアップ

1. RunPod pod 作成 (A100 PCIe 80GB、Community $1.19/h、~250 GB RAM 推奨)
2. Community template `runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04` 選択
3. Web Terminal で:

```bash
# Rust install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH=/root/.cargo/bin:$PATH

# Clone
cd /workspace
git clone git@github.com:ext-sakamoro/ALICE-Train.git
git clone git@github.com:ext-sakamoro/ALICE-ML.git
cd ALICE-Train

# JSUT DL + manifest
mkdir -p data/jsut && cd data/jsut
wget https://ss-takashi.sakura.ne.jp/corpus/jsut/jsut_ver1.1.zip
unzip jsut_ver1.1.zip
cd /workspace/ALICE-Train
pip install pyopenjtalk
python3 scripts/prepare_jsut_manifest.py \
    --jsut_root data/jsut/jsut_ver1.1 \
    --output data/jsut/manifest.jsonl

# Training
nohup ./scripts/run_fastspeech2_train.sh configs/fs2_jsut.json > logs/fs2_train.log 2>&1 &
```

### 進捗監視

```bash
tail -f logs/fs2_train.log

# grep loss trend
grep 'mel_loss' logs/fs2_train.log | tail -20
```

期待する loss trajectory (300k step、SGD lr=1e-3、mel L2 loss):
- Step 0-1000: 0.5 → 0.1 (初期急降下)
- Step 1000-10000: 0.1 → 0.05
- Step 10000-100000: 0.05 → 0.02
- Step 100000-300000: 0.02 → 0.01 (収束)

**Note**: 現 MVP は SGD only、実運用では AdamW 必要 (Phase T.4a 継続で追加)

## Checkpoint (現状未実装、Phase T.4a 継続)

現 MVP は checkpoint save/load 未実装 実運用では以下追加が必要:
- `TtsTrainer::save_checkpoint(path)`: safetensors 形式で FastSpeech2 の全 weight 保存
- `TtsTrainer::load_checkpoint(path)`: 学習再開用
- Interval save: 10k step ごと

## 評価

学習完了後の mel spectrogram 品質評価:

1. Held-out test set (JSUT の 5%) で mel 生成
2. MCD (Mel-Cepstral Distortion) 計測: target vs pred
3. Whisper large-v3 で書き起こし → WER 計測
4. MOS (Mean Opinion Score) 評価: n=20 native speaker (crowdsourcing 想定)

Target values (Phase T.4a DoD):
- MCD ≤ 6.5 dB
- WER ≤ 12%
- MOS-JA ≥ 3.5 (Phase T.6 fine-tune 後 ≥ 4.0 目標)

## トラブルシューティング

### `variable frame length not supported` エラー

現 MVP は batch_size > 1 の variable frame length を未対応
- 解決: batch_size = 1 で学習
- 恒久対応: attention mask + padding awareness を Phase T.4a 継続で追加

### `pyopenjtalk not installed`

```bash
pip install pyopenjtalk
# For Mac (M1/M2/M3):
brew install open-jtalk
```

### mel_loss が NaN になる

- learning_rate を下げる (1e-3 → 1e-4)
- gradient clip 追加 (Phase T.4a 継続で `TtsTrainConfig::gradient_clip_norm` 追加予定)
- 初期 weight の scale を small にする (現 zeros init は LayerNorm eps で保護されるが実際の学習には unsuited)

## 次段階 (Phase T.4a 継続)

- AdamW optimizer 追加 (state file m/v per param)
- Checkpoint save/load (safetensors format)
- Variable frame length + attention mask
- ProsodyLoss 併用 (duration/pitch/energy joint training)
- Gradient clipping
- Weight init (Xavier / He) — 現 zeros init は学習不能
- Weight decay
- Mixed precision (bf16)
- Multi-GPU (Phase T.4a 継続 → Phase T.4b で本格対応)

## 参考

- Paper: Ren et al. "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (ICLR 2021)
- 参照実装: Wataru-Nakata/FastSpeech2-JSUT (日本語 JSUT 特化 fork)
- ALICE-TTS docs: `~/ALICE-TTS/docs/v2/ROADMAP.md` Phase T.4a / `ARCHITECTURE.md` §AD-1
