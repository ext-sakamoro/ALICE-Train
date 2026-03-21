#!/bin/bash
# ============================================================================
# ALICE-Train: RunPod QAT 量子化学習スクリプト
#
# RunPod Pod 上で QAT (Quantization-Aware Training) を実行する。
# CUDA feature 有効でビルドし、cuBLAS + TF32 Tensor Cores を活用。
#
# 使用法:
#   bash scripts/run_qat_runpod.sh 1b        # Llama-3.2 1B
#   bash scripts/run_qat_runpod.sh qwen1.5b  # Qwen2.5 1.5B
#   bash scripts/run_qat_runpod.sh 3b        # Llama-3.2 3B
#   bash scripts/run_qat_runpod.sh qwen7b    # Qwen2.5 7B
#   bash scripts/run_qat_runpod.sh 8b        # Llama-3.1 8B
#   bash scripts/run_qat_runpod.sh 70b       # Llama-3 70B
#   bash scripts/run_qat_runpod.sh qwen35    # Qwen3.5-9B (推奨)
#   bash scripts/run_qat_runpod.sh all       # 1B→3B→Qwen7B→8B 順次
#
# 前提:
#   - bash scripts/setup_runpod.sh 済み
#   - モデルがダウンロード済み（bash scripts/download_models.sh）
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# 環境変数の読み込み
source /workspace/alice-train/.env 2>/dev/null || true
source "$HOME/.cargo/env" 2>/dev/null || true

BINARY="./target/release/train-qat-70b"
BINARY_QWEN35="./target/release/train-qat-qwen35"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALICE-Train — RunPod QAT 量子化学習                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# --------------------------------------------------------------------------
# ビルド確認
# --------------------------------------------------------------------------
if [ ! -f "$BINARY" ]; then
    echo "バイナリが見つかりません。ビルドします (qat-cuda)..."
    cargo build --release --features qat-cuda 2>&1 | tail -3
    echo ""
fi

# --------------------------------------------------------------------------
# GPU 情報表示
# --------------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# --------------------------------------------------------------------------
# データ確認・自動生成
# --------------------------------------------------------------------------
check_data() {
    local model_path=$1
    local data_dir=$2

    local train_bin="$data_dir/train.bin"
    local eval_bin="$data_dir/eval.bin"

    mkdir -p "$data_dir"

    if [ ! -f "$train_bin" ]; then
        echo "  学習データ生成中: $train_bin"
        python3 scripts/prepare_real_data.py \
            --model_path "$model_path" \
            --output "$train_bin" \
            --max_tokens 2000000
    fi

    if [ ! -f "$eval_bin" ]; then
        echo "  評価データ生成中: $eval_bin"
        python3 scripts/prepare_real_data.py \
            --model_path "$model_path" \
            --output "$eval_bin" \
            --max_tokens 100000 \
            --split validation
    fi
}

# --------------------------------------------------------------------------
# 学習関数
# --------------------------------------------------------------------------
run_qat() {
    local name=$1
    local config=$2
    local model_path=$3
    local data_dir=${4:-data/general}

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  QAT 量子化: $name"
    echo "  設定: $config"
    echo "  モデル: $model_path"
    echo "  データ: $data_dir"
    echo "  開始: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if [ ! -d "$model_path" ]; then
        echo "  エラー: モデルが見つかりません: $model_path"
        echo "  bash scripts/download_models.sh でダウンロードしてください。"
        return 1
    fi

    if [ ! -f "$config" ]; then
        echo "  エラー: 設定ファイルが見つかりません: $config"
        return 1
    fi

    check_data "$model_path" "$data_dir"

    mkdir -p logs

    # tmux セッションで実行（SSH切断に耐える）
    local log_file="logs/${name}_$(date +%Y%m%d_%H%M%S).log"

    if command -v tmux &>/dev/null && [ -z "${TMUX:-}" ]; then
        echo "  tmux セッション 'qat' で実行（SSH切断耐性あり）"
        echo "  ログ: $log_file"
        echo "  再接続: tmux attach -t qat"
        echo ""
        tmux new-session -d -s qat "$BINARY $config 2>&1 | tee $log_file"
        echo "  バックグラウンドで起動しました。"
        echo "  監視: tmux attach -t qat"
    else
        # 既にtmux内、または tmux がない場合は直接実行
        $BINARY "$config" 2>&1 | tee "$log_file"
    fi

    echo ""
    echo "  完了: $(date '+%Y-%m-%d %H:%M:%S')"
}

# --------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------
TARGET="${1:-all}"

case "$TARGET" in
    1b)
        run_qat "llama-1b" "configs/qat_1b.json" "models/meta-llama--Llama-3.2-1B-Instruct"
        ;;
    qwen1.5b|1.5b)
        run_qat "qwen-1.5b" "configs/qat_qwen_1.5b.json" "models/Qwen--Qwen2.5-1.5B-Instruct" "data/qwen"
        ;;
    3b)
        run_qat "llama-3b" "configs/qat_3b.json" "models/meta-llama--Llama-3.2-3B-Instruct"
        ;;
    qwen7b|qwen|7b)
        run_qat "qwen-7b" "configs/qat_qwen_7b.json" "models/Qwen--Qwen2.5-7B-Instruct" "data/qwen"
        ;;
    8b)
        run_qat "llama-8b" "configs/qat_8b_full.json" "models/meta-llama--Llama-3.1-8B-Instruct"
        ;;
    qwen35|qwen35_9b|qwen3.5)
        echo "━━━ Qwen3.5-9B QAT (Gated DeltaNet ハイブリッド) ━━━"
        echo "  推奨: A100 SXM 80GB"
        echo ""
        # Qwen3.5 用バイナリのビルド確認
        if [ ! -f "$BINARY_QWEN35" ]; then
            echo "  Qwen3.5 バイナリをビルド中 (qat-cli)..."
            cargo build --release --features qat-cli --bin train-qat-qwen35 2>&1 | tail -3
            echo ""
        fi
        check_data "models/Qwen--Qwen3.5-9B" "data/qwen35"
        mkdir -p logs
        local log_file="logs/qwen35-9b_$(date +%Y%m%d_%H%M%S).log"
        if command -v tmux &>/dev/null && [ -z "${TMUX:-}" ]; then
            echo "  tmux セッション 'qat' で実行"
            echo "  ログ: $log_file"
            tmux new-session -d -s qat "$BINARY_QWEN35 --config configs/qat_qwen35_9b.json 2>&1 | tee $log_file"
        else
            $BINARY_QWEN35 --config configs/qat_qwen35_9b.json 2>&1 | tee "$log_file"
        fi
        ;;
    70b)
        echo "━━━ 70B QAT (mmap + BF16 delta モード) ━━━"
        echo "  必要: GPU ≥ 80GB VRAM (H100 SXM), RAM ≥ 200GB"
        echo ""
        # RAM チェック
        total_ram_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
        if [ -n "$total_ram_kb" ]; then
            total_ram_gb=$((total_ram_kb / 1024 / 1024))
            echo "  検出 RAM: ${total_ram_gb} GB"
            if [ "$total_ram_gb" -lt 180 ]; then
                echo "  警告: RAM が 180GB 未満です。OOM の可能性があります。"
                echo "  5秒後に続行... (Ctrl+C で中止)"
                sleep 5
            fi
        fi
        # VRAM チェック
        if command -v nvidia-smi &>/dev/null; then
            vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
            vram_gb=$((vram_mb / 1024))
            echo "  検出 VRAM: ${vram_gb} GB"
            if [ "$vram_gb" -lt 70 ]; then
                echo "  警告: VRAM が 70GB 未満です。H100 80GB を推奨します。"
            fi
        fi
        echo ""
        run_qat "llama-70b" "configs/qat_70b.json" "models/meta-llama--Llama-3-70B-Instruct"
        ;;
    all)
        echo "全モデル順次実行: 1B → 3B → Qwen 7B → 8B"
        echo ""
        run_qat "llama-1b" "configs/qat_1b.json" "models/meta-llama--Llama-3.2-1B-Instruct"
        run_qat "llama-3b" "configs/qat_3b.json" "models/meta-llama--Llama-3.2-3B-Instruct"
        run_qat "qwen-7b" "configs/qat_qwen_7b.json" "models/Qwen--Qwen2.5-7B-Instruct" "data/qwen"
        run_qat "llama-8b" "configs/qat_8b_full.json" "models/meta-llama--Llama-3.1-8B-Instruct"
        ;;
    *)
        echo "使用法: $0 [1b|qwen1.5b|3b|qwen7b|8b|70b|qwen35|all]"
        echo ""
        echo "  1b        — Llama-3.2 1B"
        echo "  qwen1.5b  — Qwen2.5 1.5B"
        echo "  3b        — Llama-3.2 3B"
        echo "  qwen7b    — Qwen2.5 7B"
        echo "  8b        — Llama-3.1 8B"
        echo "  70b       — Llama-3 70B (要 H100 80GB + 200GB RAM)"
        echo "  qwen35    — Qwen3.5-9B Gated DeltaNet (推奨: A100 SXM)"
        echo "  all       — 全モデル順次実行 (70B 除く)"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  完了                                                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "チェックポイント:"
ls -la checkpoints/ 2>/dev/null || echo "  (なし)"
echo ""
