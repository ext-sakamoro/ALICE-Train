#!/bin/bash
# RunPod QAT ラッパー — 学習完了/崩壊時にPod自動停止
#
# Usage: bash scripts/runpod_qat.sh
#
# 環境変数:
#   RUNPOD_API_KEY  — RunPod API Key (必須)
#   RUNPOD_POD_ID   — 自身のPod ID (必須、$RUNPOD_POD_ID で自動設定される場合あり)

set -u

export PATH=/root/.cargo/bin:${PATH}
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

cd /workspace/alice-train || { echo "alice-train not found"; exit 1; }

RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"

# Pod ID 自動検出
if [ -z "$RUNPOD_POD_ID" ]; then
    RUNPOD_POD_ID=$(hostname 2>/dev/null | grep -oE '[a-z0-9]+' | head -1)
    echo "Pod ID 自動検出: $RUNPOD_POD_ID"
fi

stop_pod() {
    if [ -n "$RUNPOD_POD_ID" ]; then
        echo "$(date '+%H:%M:%S') Pod停止: $RUNPOD_POD_ID"
        curl -s https://api.runpod.io/graphql \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"query\": \"mutation { podStop(input: { podId: \\\"$RUNPOD_POD_ID\\\" }) { id } }\"}"
        echo ""
    else
        echo "RUNPOD_POD_ID 未設定 — Pod停止スキップ"
    fi
}

echo "═══════════════════════════════════════════"
echo "  ALICE-Train QAT on RunPod"
echo "  Pod: $RUNPOD_POD_ID"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Config: configs/qat_qwen35_9b_a100_safe.json"
echo "═══════════════════════════════════════════"

mkdir -p logs

# QAT実行 (フォアグラウンド)
stdbuf -oL ./target/release/train-qat-qwen35 \
    --config configs/qat_qwen35_9b_a100_safe.json \
    2>&1 | tee logs/qwen35_runpod.log

EXIT_CODE=$?

echo ""
echo "═══════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ 学習完了 (exit 0)"
else
    echo "  ✗ 学習停止 (exit $EXIT_CODE) — 崩壊 or エラー"
fi
echo "═══════════════════════════════════════════"

# 最終チェックポイント書き戻し確認
echo "最終 resume_state.json:"
cat checkpoints/qwen35_9b/resume_state.json 2>/dev/null || echo "(なし)"

# Pod停止 (課金停止)
stop_pod
