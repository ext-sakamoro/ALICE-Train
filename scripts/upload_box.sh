#!/bin/bash
# Box SFTP アップロード — 学習成果物を Project-ALICE/QAT/ に転送
# Usage: upload_box.sh <checkpoint_dir> [--all-checkpoints]
set -euo pipefail

CKPT_DIR="${1:?Usage: upload_box.sh <checkpoint_dir> [--all-checkpoints]}"
ALL_CKPTS="${2:-}"
BOX_USER="sakamoro@extoria.co.jp"
BOX_HOST="sftp.services.box.com"
BOX_DIR="Project-ALICE/QAT"

if [ -z "${BOX_SFTP_PASS:-}" ]; then
    echo "[upload_box] ERROR: BOX_SFTP_PASS not set"
    exit 1
fi

# アップロード対象ファイルを収集
FILES=()

# .alice モデル
for f in "$CKPT_DIR"/*.alice; do [ -f "$f" ] && FILES+=("$f"); done

# チェックポイント
if [ "$ALL_CKPTS" = "--all-checkpoints" ]; then
    # 全チェックポイント（崩壊時：復旧用に全世代保持）
    for f in "$CKPT_DIR"/step_*.bin; do [ -f "$f" ] && FILES+=("$f"); done
else
    # 最新のみ（正常完了時）
    LATEST_CKPT=$(ls -t "$CKPT_DIR"/step_*.bin 2>/dev/null | head -1)
    [ -n "${LATEST_CKPT:-}" ] && FILES+=("$LATEST_CKPT")
fi

# メタデータ
[ -f "$CKPT_DIR/resume_state.json" ] && FILES+=("$CKPT_DIR/resume_state.json")
[ -f "$CKPT_DIR/train_log.csv" ] && FILES+=("$CKPT_DIR/train_log.csv")
[ -f "$CKPT_DIR/run_record.json" ] && FILES+=("$CKPT_DIR/run_record.json")

# FP32キャッシュ（崩壊時は復旧に必要）
if [ "$ALL_CKPTS" = "--all-checkpoints" ] && [ -d "$CKPT_DIR/fp32_cache" ]; then
    for f in "$CKPT_DIR/fp32_cache"/*.fp32; do [ -f "$f" ] && FILES+=("$f"); done
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "[upload_box] No files to upload"
    exit 0
fi

echo "[upload_box] Uploading ${#FILES[@]} files to Box:${BOX_DIR}/"

# fp32_cacheサブディレクトリが必要か確認
HAS_FP32=false
for f in "${FILES[@]}"; do
    case "$f" in */fp32_cache/*) HAS_FP32=true; break;; esac
done

# putコマンドを生成
PUT_CMDS="cd ${BOX_DIR}
"
if [ "$HAS_FP32" = true ]; then
    PUT_CMDS="${PUT_CMDS}-mkdir fp32_cache
"
fi

for f in "${FILES[@]}"; do
    case "$f" in
        */fp32_cache/*)
            PUT_CMDS="${PUT_CMDS}put ${f} fp32_cache/$(basename "$f")
"
            ;;
        *)
            PUT_CMDS="${PUT_CMDS}put ${f}
"
            ;;
    esac
    SIZE=$(du -h "$f" | cut -f1)
    echo "[upload_box]   $(basename "$f") (${SIZE})"
done
PUT_CMDS="${PUT_CMDS}ls -la
"

sshpass -p "$BOX_SFTP_PASS" sftp -o StrictHostKeyChecking=no -o ConnectTimeout=30 "${BOX_USER}@${BOX_HOST}" << EOF
${PUT_CMDS}
EOF

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "[upload_box] Upload complete (${#FILES[@]} files)"
else
    echo "[upload_box] Upload FAILED (exit=$STATUS)"
fi
exit $STATUS
