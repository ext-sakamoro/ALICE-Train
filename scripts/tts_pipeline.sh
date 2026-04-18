#!/bin/bash
# TTS Pipeline: Generate + Upload to Box
# Usage:
#   Single:  bash tts_pipeline.sh --ref-audio sample.wav --ref-text "transcript" --text "generate this"
#   Batch:   bash tts_pipeline.sh --ref-audio sample.wav --ref-text "transcript" --batch batch.json
#
# Environment:
#   BOX_SFTP_PASS - Box SFTP password (required for upload)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="/notebooks/tts_output"

# Pass all arguments to tts_generate.py
echo "=== Step 1: Generate Audio ==="
python3 "$SCRIPT_DIR/tts_generate.py" --output-dir "$OUTPUT_DIR" "$@"

# Upload to Box
echo ""
echo "=== Step 2: Upload to Box ==="
if [ -n "${BOX_SFTP_PASS:-}" ]; then
    bash "$SCRIPT_DIR/tts_upload_box.sh" "$OUTPUT_DIR"
else
    echo "SKIP: BOX_SFTP_PASS not set. Files are in $OUTPUT_DIR"
fi

echo ""
echo "=== Done ==="
