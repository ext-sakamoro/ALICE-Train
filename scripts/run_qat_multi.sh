#!/bin/bash
# ============================================================================
# ALICE-Train: マルチモデル QAT 学習スクリプト
#
# 1B → 3B → Qwen 7B → 8B の順で QAT 学習を実行する。
# 各モデル完了後に次のモデルへ進む。
#
# 使用法:
#   bash scripts/run_qat_multi.sh              # 全モデル順次実行
#   bash scripts/run_qat_multi.sh 1b           # 1B のみ
#   bash scripts/run_qat_multi.sh 3b           # 3B のみ
#   bash scripts/run_qat_multi.sh qwen         # Qwen 7B のみ
#   bash scripts/run_qat_multi.sh 8b           # 8B のみ
#
# 前提:
#   - モデルがダウンロード済み（scripts/download_models.sh small）
#   - 学習データが data/general/train.bin に存在
#   - cargo build --release 済み
# ============================================================================
set -euo pipefail

# プロジェクトルート検出
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BINARY="./target/release/train_qat_70b"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALICE-Train — マルチモデル QAT 学習                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# --------------------------------------------------------------------------
# ビルド確認
# --------------------------------------------------------------------------
if [ ! -f "$BINARY" ]; then
    echo "バイナリが見つかりません。ビルドします..."
    cargo build --release --bin train_qat_70b 2>&1 | tail -3
    echo ""
fi

# --------------------------------------------------------------------------
# データ確認
# Qwen は vocab_size が異なるため専用データが必要。
# Llama 系は共通データで OK（vocab_size 同一）。
# --------------------------------------------------------------------------
check_data() {
    local model_path=$1
    local data_dir=$2  # "data/general" or "data/qwen"

    local train_bin="$data_dir/train.bin"
    local eval_bin="$data_dir/eval.bin"

    mkdir -p "$data_dir"

    if [ ! -f "$train_bin" ]; then
        echo "  学習データが見つかりません: $train_bin"
        echo "  生成します..."
        pip install -q transformers datasets 2>/dev/null || true
        python3 scripts/prepare_real_data.py \
            --model_path "$model_path" \
            --output "$train_bin" \
            --max_tokens 2000000
    fi

    if [ ! -f "$eval_bin" ]; then
        echo "  評価データが見つかりません: $eval_bin"
        echo "  生成します..."
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
    local data_dir=${4:-data/general}  # Qwen 用は data/qwen

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  QAT 学習: $name"
    echo "  設定: $config"
    echo "  モデル: $model_path"
    echo "  データ: $data_dir"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # モデル存在確認
    if [ ! -d "$model_path" ]; then
        echo "  エラー: モデルが見つかりません: $model_path"
        echo "  scripts/download_models.sh でダウンロードしてください。"
        return 1
    fi

    # 設定ファイル確認
    if [ ! -f "$config" ]; then
        echo "  エラー: 設定ファイルが見つかりません: $config"
        return 1
    fi

    # データ確認（モデル固有のtokenizerでデータ生成）
    check_data "$model_path" "$data_dir"

    # 学習実行
    echo "  学習開始: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    $BINARY "$config" 2>&1 | tee "logs/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "  学習完了: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

# --------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------

mkdir -p logs

TARGET="${1:-all}"

case "$TARGET" in
    1b)
        run_qat "llama-1b" "configs/qat_1b.json" "models/meta-llama--Llama-3.2-1B-Instruct"
        ;;
    3b)
        run_qat "llama-3b" "configs/qat_3b.json" "models/meta-llama--Llama-3.2-3B-Instruct"
        ;;
    qwen|7b|5b)
        run_qat "qwen-7b" "configs/qat_qwen_7b.json" "models/Qwen--Qwen2.5-7B-Instruct" "data/qwen"
        ;;
    8b)
        run_qat "llama-8b" "configs/qat_8b_full.json" "models/meta-llama--Llama-3.1-8B-Instruct"
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
        echo "使用法: $0 [1b|3b|qwen|8b|all]"
        echo ""
        echo "  1b    — Llama-3.2 1B (reCamera 向け)"
        echo "  3b    — Llama-3.2 3B (reCamera 向け)"
        echo "  qwen  — Qwen2.5 7B (5B クラス)"
        echo "  8b    — Llama-3.1 8B full (RPi 5 向け)"
        echo "  all   — 全モデル順次実行"
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
