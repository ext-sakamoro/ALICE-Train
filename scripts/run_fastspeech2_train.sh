#!/bin/bash
# FastSpeech2 学習実行スクリプト (Phase T.4a)
#
# Usage:
#   ./scripts/run_fastspeech2_train.sh [config_file]
#
# 前提条件:
#   - JSUT corpus が data/jsut/ にダウンロード済
#   - Manifest jsonl 生成済: python3 scripts/prepare_jsut_manifest.py --jsut_root ...
#   - cargo が PATH に通っている、または $HOME/.cargo/bin
#
# 実行環境の推奨:
#   - Local Mac (dev): 短時間 sanity check (10-100 step、~5 分)
#   - RunPod A100 (本番): 300k step、~5-10 日、~$100
#
# 参考: docs/fastspeech2_training.md
#

set -euo pipefail
export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"

CONFIG="${1:-configs/fs2_jsut.json}"
LOG_DIR="logs/fs2_$(date +%Y%m%d_%H%M%S)"

echo "[info] Config: ${CONFIG}"
echo "[info] Log dir: ${LOG_DIR}"

if [ ! -f "${CONFIG}" ]; then
    echo "[error] Config file not found: ${CONFIG}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

# Build release binary
echo "[info] Building alice-train example fs2_train..."
cargo build --release --example fs2_train --features tts 2>&1 | tee "${LOG_DIR}/build.log"

# Verify config parses (via a dry-run marker、実 binary は example)
echo "[info] Config summary:"
python3 -c "
import json, sys
with open('${CONFIG}') as f:
    cfg = json.load(f)
print(f'  vocab_size: {cfg[\"model\"][\"vocab_size\"]}')
print(f'  hidden_dim: {cfg[\"model\"][\"hidden_dim\"]}')
print(f'  encoder_layers: {cfg[\"model\"][\"num_encoder_layers\"]}')
print(f'  decoder_layers: {cfg[\"model\"][\"num_decoder_layers\"]}')
print(f'  mel_dim: {cfg[\"model\"][\"mel_dim\"]}')
print(f'  batch_size: {cfg[\"dataset\"][\"batch_size\"]}')
print(f'  learning_rate: {cfg[\"training\"][\"learning_rate\"]}')
print(f'  total_steps: {cfg[\"training\"][\"total_steps\"]}')
print(f'  checkpoint_dir: {cfg[\"training\"][\"checkpoint_dir\"]}')
"

# Run training
echo "[info] Starting training..."
./target/release/examples/fs2_train --config "${CONFIG}" 2>&1 | tee "${LOG_DIR}/train.log"

echo "[info] Training done. Log: ${LOG_DIR}/train.log"
