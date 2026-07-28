#!/bin/bash
# Phase T.4d Box auto-uploader watcher
# 新規 checkpoint (.safetensors) が現れたら Box:Project-ALICE/TTS/fs2_jsut_YYYYMMDD/ に upload
# usage: box_watcher.sh <ckpt_dir> <session_id>

set -uo pipefail

CKPT_DIR="${1:?Usage: box_watcher.sh <ckpt_dir> <session_id>}"
SESSION="${2:?session_id required}"
BOX_USER="sakamoro@extoria.co.jp"
BOX_HOST="sftp.services.box.com"
BOX_DIR="Project-ALICE/TTS/fs2_jsut_${SESSION}"
UPLOADED_LOG="/tmp/box_uploaded_${SESSION}.txt"
CHECK_INTERVAL=60  # 60 秒毎に polling

touch "$UPLOADED_LOG"

echo "[box_watcher] start: ckpt_dir=$CKPT_DIR box_dir=$BOX_DIR"

# 初回に session dir を Box 上に作成
sshpass -f /tmp/.bp sftp -B 32768 -oStrictHostKeyChecking=no -oKexAlgorithms=+diffie-hellman-group14-sha1 "$BOX_USER"@"$BOX_HOST" <<EOF > /tmp/box_mkdir_${SESSION}.log 2>&1
-mkdir Project-ALICE
-mkdir Project-ALICE/TTS
-mkdir $BOX_DIR
bye
EOF
echo "[box_watcher] Box dir prepared: $BOX_DIR"

while true; do
    sleep $CHECK_INTERVAL
    for f in "$CKPT_DIR"/step_*.safetensors; do
        [ -f "$f" ] || continue
        BASENAME=$(basename "$f")
        # 既に upload 済みかチェック
        if grep -qx "$BASENAME" "$UPLOADED_LOG"; then
            continue
        fi
        # file size が安定していることを確認 (書き込み中なら skip)
        SIZE_1=$(stat -c%s "$f" 2>/dev/null || echo 0)
        sleep 3
        SIZE_2=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if [ "$SIZE_1" != "$SIZE_2" ] || [ "$SIZE_1" -eq 0 ]; then
            echo "[box_watcher] $BASENAME still growing (${SIZE_1}→${SIZE_2}), skip this round"
            continue
        fi
        # upload
        echo "[box_watcher] uploading $BASENAME ($SIZE_1 bytes)..."
        UPLOAD_LOG="/tmp/box_upload_${SESSION}_${BASENAME}.log"
        sshpass -f /tmp/.bp sftp -B 32768 -oStrictHostKeyChecking=no -oKexAlgorithms=+diffie-hellman-group14-sha1 "$BOX_USER"@"$BOX_HOST" <<EOF > "$UPLOAD_LOG" 2>&1
cd $BOX_DIR
put "$f"
ls -l $BASENAME
bye
EOF
        # 検証: remote size と local size 一致確認
        REMOTE_LINE=$(grep -E " $BASENAME\$" "$UPLOAD_LOG" | tail -1)
        REMOTE_SIZE=$(echo "$REMOTE_LINE" | awk '{print $5}')
        if [ "$REMOTE_SIZE" = "$SIZE_1" ]; then
            echo "$BASENAME" >> "$UPLOADED_LOG"
            echo "[box_watcher] uploaded OK: $BASENAME (${SIZE_1} bytes)"
        else
            echo "[box_watcher] SIZE MISMATCH $BASENAME local=$SIZE_1 remote=$REMOTE_SIZE — will retry next round"
        fi
    done

    # train.log も上書き upload (毎回)
    LOG_FILE="$CKPT_DIR/../logs/train.log"
    if [ -f "$LOG_FILE" ]; then
        sshpass -f /tmp/.bp sftp -B 32768 -oStrictHostKeyChecking=no -oKexAlgorithms=+diffie-hellman-group14-sha1 "$BOX_USER"@"$BOX_HOST" <<EOF > /dev/null 2>&1
cd $BOX_DIR
put $LOG_FILE
bye
EOF
    fi
done
