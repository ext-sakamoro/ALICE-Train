#!/bin/bash
# ALICE-Train: Spot インスタンス自動レジュームスクリプト
#
# Paperspace/AWS Spot がキルされた場合に自動的に最新チェックポイントから再開。
#
# 使用法:
#   ./scripts/auto_resume.sh configs/qat_70b.json
#   ./scripts/auto_resume.sh configs/qat_70b.json --features gpu
#
# 環境変数:
#   RETRY_DELAY   — リトライ間隔 (秒, デフォルト: 30)
#   MAX_RETRIES   — 最大リトライ回数 (デフォルト: 100)
#   CHECKPOINT_DIR — チェックポイントディレクトリ (デフォルト: checkpoints)

set -euo pipefail

CONFIG="${1:?使用法: $0 <config.json> [cargo flags...]}"
shift
EXTRA_FLAGS="${*}"

RETRY_DELAY="${RETRY_DELAY:-30}"
MAX_RETRIES="${MAX_RETRIES:-100}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"

# 最新チェックポイントを検索
find_latest_checkpoint() {
    if [ -d "$CHECKPOINT_DIR" ]; then
        # step_XXXXX.bin の最大番号を検索
        ls -1 "$CHECKPOINT_DIR"/step_*.bin 2>/dev/null \
            | sort -t_ -k2 -n \
            | tail -1
    fi
}

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALICE-Train Auto-Resume — Spot Instance Resilient      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "設定: $CONFIG"
echo "チェックポイント: $CHECKPOINT_DIR"
echo "リトライ間隔: ${RETRY_DELAY}s"
echo "最大リトライ: $MAX_RETRIES"
echo ""

RETRY_COUNT=0

while [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; do
    LATEST=$(find_latest_checkpoint)
    RESUME_FLAG=""

    if [ -n "$LATEST" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] レジューム: $LATEST"
        RESUME_FLAG="--resume $LATEST"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 新規学習開始"
    fi

    # 学習実行
    # shellcheck disable=SC2086
    cargo run --release --features qat-cli --bin train-qat-70b $EXTRA_FLAGS -- \
        --config "$CONFIG" \
        $RESUME_FLAG

    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 学習正常完了"
        exit 0
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] プロセス終了 (exit=$EXIT_CODE)"
    echo "  Spot kill の可能性あり。${RETRY_DELAY}秒後にリトライ ($RETRY_COUNT/$MAX_RETRIES)"
    sleep "$RETRY_DELAY"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 最大リトライ回数に到達。終了。"
exit 1
