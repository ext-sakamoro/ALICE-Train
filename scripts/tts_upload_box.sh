#!/bin/bash
# Box SFTP Upload Script for TTS Output
# Usage: bash tts_upload_box.sh /notebooks/tts_output
#
# BOX_SFTP_PASS must be set as environment variable

set -euo pipefail

SOURCE_DIR="${1:-/notebooks/tts_output}"
BOX_HOST="sftp.services.box.com"
BOX_USER="sakamoro@extoria.co.jp"
BOX_DEST="Shizai/DLsite/Voice"

if [ -z "${BOX_SFTP_PASS:-}" ]; then
    echo "ERROR: BOX_SFTP_PASS not set"
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory not found: $SOURCE_DIR"
    exit 1
fi

WAV_COUNT=$(find "$SOURCE_DIR" -name "*.wav" -o -name "*.mp3" | wc -l)
if [ "$WAV_COUNT" -eq 0 ]; then
    echo "No audio files found in $SOURCE_DIR"
    exit 0
fi

echo "Uploading $WAV_COUNT audio files to Box: $BOX_DEST"

# Create remote directory and upload
sshpass -p "$BOX_SFTP_PASS" sftp -oBatchMode=no -oStrictHostKeyChecking=no "$BOX_USER@$BOX_HOST" <<SFTP_EOF
-mkdir $BOX_DEST
cd $BOX_DEST
lcd $SOURCE_DIR
mput *.wav
mput *.mp3
bye
SFTP_EOF

echo "Upload complete: $WAV_COUNT files -> Box:$BOX_DEST"
