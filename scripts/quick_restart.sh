#!/bin/bash
# quick_restart.sh — Paperspace session expire 後の TTS 学習復旧 one-shot
# 新 container 起動時に実行すると学習 + Box auto-upload を再開する

set -uo pipefail

echo "=== ALICE-TTS Paperspace quick restart ==="

# 1. Container reset で消える dep を再 install
apt-get install -y -qq sshpass tmux libopenblas0 libopenblas-dev 2>&1 | tail -1

# 2. Box パスワード file を再作成 (session 毎に消える)
if [ ! -f /tmp/.bp ]; then
    printf '@E99l0103' > /tmp/.bp
    chmod 600 /tmp/.bp
    echo "  [ok] /tmp/.bp created"
fi

# 3. Rust default toolchain
export PATH=/notebooks/.cargo/bin:$PATH
rustup default 1.92.0 > /dev/null 2>&1

# 4. CUDA LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# 5. Session ID for Box upload path
SESSION_ID=$(date +%Y%m%d_%H%M%S)
echo "SESSION_ID=$SESSION_ID" > /tmp/current_session.txt
echo "  [ok] SESSION_ID=$SESSION_ID"

# 6. 既存 tmux 全 kill
tmux kill-server 2>/dev/null || true
pkill -9 -f fs2_train 2>/dev/null || true
pkill -9 -f box_watcher 2>/dev/null || true
sleep 2

# 7. checkpoint 状態表示
CKPT_DIR=/notebooks/ALICE-Train/checkpoints/fs2_jsut
LATEST_CKPT=$(ls -t "$CKPT_DIR"/step_*.safetensors 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "  [ok] Latest checkpoint: $LATEST_CKPT"
else
    echo "  [warn] No checkpoint found, will start from scratch"
fi

# 8. train + box_watcher を tmux で起動
cd /notebooks/ALICE-Train
rm -f logs/train.log
chmod +x scripts/box_watcher.sh

tmux new-session -d -s tts_train "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH; ./target/release/examples/fs2_train --config configs/fs2_jsut.json > logs/train.log 2>&1"
tmux new-session -d -s box_uploader "bash scripts/box_watcher.sh checkpoints/fs2_jsut $SESSION_ID > logs/box_watcher.log 2>&1"

sleep 5
echo ""
echo "=== tmux sessions ==="
tmux list-sessions
echo ""
echo "=== processes ==="
pgrep -a fs2_train
ps -ef | grep box_watcher | grep -v grep | head -2
echo ""
echo "=== first log lines ==="
sleep 5
tail -8 logs/train.log
echo ""
echo "OK. Monitor: tail -f logs/train.log logs/box_watcher.log"
