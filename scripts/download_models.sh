#!/bin/bash
# ============================================================================
# ALICE-Train: ベースモデルダウンロードスクリプト
#
# HuggingFace から QAT 学習に必要な全ベースモデルをダウンロードする。
# Llama-3 70B は約 140GB/モデル。十分なディスク容量を確認すること。
#
# 使用法:
#   ./scripts/download_models.sh              # 全モデル
#   ./scripts/download_models.sh llama        # Llama-3 70B のみ
#   ./scripts/download_models.sh codellama    # CodeLlama 70B のみ
#   ./scripts/download_models.sh elyza        # ELYZA JP 70B のみ
#
# 前提:
#   - huggingface-cli がインストール済み
#   - huggingface-cli login でトークン設定済み（Llama-3 はゲート付き）
# ============================================================================

set -euo pipefail

# モデル保存先（永続ストレージ）
MODEL_BASE="${ALICE_TRAIN_MODELS:-/storage/alice-train/models}"
mkdir -p "$MODEL_BASE"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALICE-Train — ベースモデルダウンロード                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "保存先: $MODEL_BASE"
echo ""

# --------------------------------------------------------------------------
# ディスク容量チェック
# --------------------------------------------------------------------------
check_disk_space() {
    local required_gb=$1
    local available_gb

    # /storage のマウントポイントから空き容量を取得
    available_gb=$(df -BG "$MODEL_BASE" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')

    if [ -z "$available_gb" ]; then
        echo "  警告: ディスク容量の確認ができません。続行します。"
        return 0
    fi

    echo "  必要容量: ${required_gb}GB / 空き容量: ${available_gb}GB"

    if [ "$available_gb" -lt "$required_gb" ]; then
        echo "  エラー: ディスク容量不足。${required_gb}GB 必要ですが ${available_gb}GB しかありません。"
        return 1
    fi
    return 0
}

# --------------------------------------------------------------------------
# モデルダウンロード関数
# --------------------------------------------------------------------------
download_model() {
    local repo_id=$1
    local local_name=$2
    local description=$3
    local local_dir="$MODEL_BASE/$local_name"

    echo "────────────────────────────────────────────────────────"
    echo "モデル: $description"
    echo "リポ:   $repo_id"
    echo "保存先: $local_dir"
    echo ""

    # 既にダウンロード済みか確認
    if [ -f "$local_dir/config.json" ]; then
        echo "  スキップ: 既にダウンロード済み (config.json 検出)"
        echo ""
        return 0
    fi

    mkdir -p "$local_dir"

    echo "  ダウンロード開始..."
    echo "  （中断しても再実行で途中から再開可能）"
    echo ""

    # huggingface-cli で safetensors のみダウンロード（.bin は除外して容量節約）
    huggingface-cli download \
        "$repo_id" \
        --local-dir "$local_dir" \
        --local-dir-use-symlinks False \
        --include "*.safetensors" "*.json" "tokenizer*" "*.model" \
        --exclude "*.bin" "*.ot" "*.msgpack" "consolidated*" \
        --resume-download

    echo ""
    echo "  完了: $local_dir"

    # ダウンロードサイズ表示
    local size
    size=$(du -sh "$local_dir" 2>/dev/null | awk '{print $1}')
    echo "  サイズ: $size"
    echo ""
}

# --------------------------------------------------------------------------
# モデル定義
# --------------------------------------------------------------------------

# Llama-3 70B Instruct — 9 ドメインで使用
#   general, math, finance, medical, legal, security, robotics, creative, spatial, infra
download_llama() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [1/3] Meta Llama-3 70B Instruct"
    echo "  使用ドメイン: general, math, finance, medical, legal,"
    echo "               security, robotics, creative, spatial, infra"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    check_disk_space 150 || return 1

    download_model \
        "meta-llama/Llama-3-70B-Instruct" \
        "meta-llama--Llama-3-70B-Instruct" \
        "Meta Llama-3 70B Instruct (汎用ベースモデル)"
}

# CodeLlama 70B Instruct — コード生成ドメインで使用
download_codellama() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [2/3] CodeLlama 70B Instruct"
    echo "  使用ドメイン: code"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    check_disk_space 150 || return 1

    download_model \
        "codellama/CodeLlama-70b-Instruct-hf" \
        "codellama--CodeLlama-70b-Instruct-hf" \
        "CodeLlama 70B Instruct (コード生成特化)"
}

# ELYZA JP 70B — 日本語ドメインで使用
download_elyza() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [3/3] ELYZA Llama-3 JP 70B"
    echo "  使用ドメイン: japanese"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    check_disk_space 150 || return 1

    download_model \
        "elyza/Llama-3-ELYZA-JP-70B" \
        "elyza--Llama-3-ELYZA-JP-70B" \
        "ELYZA Llama-3 JP 70B (日本語特化)"
}

# Llama-3.2-1B Instruct — reCamera / 小型デバイス向け QAT
download_llama_1b() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Llama-3.2-1B Instruct (reCamera 向け)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    check_disk_space 3 || return 1
    download_model \
        "meta-llama/Llama-3.2-1B-Instruct" \
        "meta-llama--Llama-3.2-1B-Instruct" \
        "Meta Llama-3.2 1B Instruct (~2.5GB)"
}

# Llama-3.2-3B Instruct — reCamera / 中型デバイス向け QAT
download_llama_3b() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Llama-3.2-3B Instruct (reCamera 向け)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    check_disk_space 7 || return 1
    download_model \
        "meta-llama/Llama-3.2-3B-Instruct" \
        "meta-llama--Llama-3.2-3B-Instruct" \
        "Meta Llama-3.2 3B Instruct (~6.4GB)"
}

# Qwen2.5-7B Instruct — reCamera MAX / 中型デバイス向け QAT
download_qwen_7b() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Qwen2.5-7B Instruct (5B クラス, attention_bias=true)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    check_disk_space 16 || return 1
    download_model \
        "Qwen/Qwen2.5-7B-Instruct" \
        "Qwen--Qwen2.5-7B-Instruct" \
        "Qwen2.5 7B Instruct (~15GB)"
}

# --------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------

TARGET="${1:-all}"

case "$TARGET" in
    llama)
        download_llama
        ;;
    codellama)
        download_codellama
        ;;
    elyza)
        download_elyza
        ;;
    1b)
        download_llama_1b
        ;;
    3b)
        download_llama_3b
        ;;
    qwen|7b)
        download_qwen_7b
        ;;
    small)
        echo "小型モデルをダウンロードします（1B + 3B + Qwen 7B, 合計 ~24GB）"
        echo ""
        download_llama_1b
        download_llama_3b
        download_qwen_7b
        ;;
    all)
        echo "全モデルをダウンロードします（合計 ~420GB）"
        echo ""
        check_disk_space 450 || {
            echo "エラー: 全モデルのダウンロードに十分な容量がありません。"
            echo "個別にダウンロードしてください:"
            echo "  ./scripts/download_models.sh llama"
            echo "  ./scripts/download_models.sh codellama"
            echo "  ./scripts/download_models.sh elyza"
            exit 1
        }
        download_llama
        download_codellama
        download_elyza
        ;;
    *)
        echo "使用法: $0 [1b|3b|qwen|small|llama|codellama|elyza|all]"
        echo ""
        echo "  1b        — Llama-3.2 1B Instruct (~2.5GB)"
        echo "  3b        — Llama-3.2 3B Instruct (~6.4GB)"
        echo "  qwen      — Qwen2.5 7B Instruct (~15GB)"
        echo "  small     — 1B + 3B + Qwen 7B (全小型モデル)"
        echo "  llama     — Meta Llama-3 70B Instruct"
        echo "  codellama — CodeLlama 70B Instruct"
        echo "  elyza     — ELYZA Llama-3 JP 70B"
        echo "  all       — 全モデル (デフォルト)"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ダウンロード完了                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "モデル一覧:"
ls -la "$MODEL_BASE/"
echo ""
echo "合計使用量:"
du -sh "$MODEL_BASE"
echo ""
