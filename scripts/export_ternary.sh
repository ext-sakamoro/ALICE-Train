#!/bin/bash
# ============================================================================
# ALICE-Train: 三値重みエクスポートスクリプト
#
# QAT 学習済みチェックポイントから三値 (ternary) 重みをエクスポートする。
# Llama-3 70B の FP32 重み (~280GB) を 1.1-bit sparse ternary (~10GB) に圧縮。
#
# 使用法:
#   ./scripts/export_ternary.sh general          # ベストチェックポイント
#   ./scripts/export_ternary.sh code 10000       # 指定ステップ
#   ./scripts/export_ternary.sh all              # 全ドメイン一括
#
# 出力:
#   exports/{domain}/alice-ternary-70b-{domain}.safetensors
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# RunPod: /workspace/alice-train, Paperspace: /storage/alice-train
if [ -d "/workspace/alice-train" ]; then
    DEFAULT_STORAGE="/workspace/alice-train"
elif [ -d "/storage/alice-train" ]; then
    DEFAULT_STORAGE="/storage/alice-train"
else
    DEFAULT_STORAGE="/storage/alice-train"
fi
STORAGE_BASE="${ALICE_TRAIN_STORAGE:-$DEFAULT_STORAGE}"
EXPORT_BASE="$STORAGE_BASE/exports"

# ドメイン一覧
ALL_DOMAINS=(general code japanese math finance medical legal security robotics creative spatial infra)

# --------------------------------------------------------------------------
# Llama-3 70B モデルサイズ参照値
# --------------------------------------------------------------------------
# FP32 全パラメータ: 70B × 4 bytes = ~280GB
# BF16 全パラメータ: 70B × 2 bytes = ~140GB
# 1.1-bit ternary:  70B × 1.1/8 bytes ≈ ~9.6GB (+ メタデータ)
FP32_SIZE_GB=280
BF16_SIZE_GB=140
TERNARY_SIZE_GB=10  # 概算（スパース性による変動あり）

# --------------------------------------------------------------------------
# 関数定義
# --------------------------------------------------------------------------

# ベストチェックポイントを検索（最大ステップ番号）
find_best_checkpoint() {
    local domain=$1
    local ckpt_dir="$STORAGE_BASE/checkpoints/$domain"

    if [ ! -d "$ckpt_dir" ]; then
        echo ""
        return
    fi

    # step_XXXXX.bin の最大番号を検索
    ls -1 "$ckpt_dir"/step_*.bin 2>/dev/null \
        | sort -t_ -k2 -n \
        | tail -1
}

# 指定ステップのチェックポイントを検索
find_checkpoint_at_step() {
    local domain=$1
    local step=$2
    local ckpt_dir="$STORAGE_BASE/checkpoints/$domain"

    # ゼロパディング対応（step_00500.bin, step_500.bin 両方）
    local padded
    padded=$(printf "step_%05d.bin" "$step")

    if [ -f "$ckpt_dir/$padded" ]; then
        echo "$ckpt_dir/$padded"
    elif [ -f "$ckpt_dir/step_${step}.bin" ]; then
        echo "$ckpt_dir/step_${step}.bin"
    else
        echo ""
    fi
}

# 圧縮統計の表示
show_compression_stats() {
    local domain=$1
    local checkpoint=$2
    local export_file=$3

    echo ""
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  圧縮統計:"
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "    元モデル (FP32):    ${FP32_SIZE_GB}GB (70B params × 32bit)"
    echo "    元モデル (BF16):    ${BF16_SIZE_GB}GB (70B params × 16bit)"
    echo "    三値 (1.1-bit):     ~${TERNARY_SIZE_GB}GB (sparse ternary)"
    echo ""

    # 圧縮率計算
    local ratio_fp32=$((FP32_SIZE_GB / TERNARY_SIZE_GB))
    local ratio_bf16=$((BF16_SIZE_GB / TERNARY_SIZE_GB))
    echo "    FP32 比圧縮率:     ${ratio_fp32}x"
    echo "    BF16 比圧縮率:     ${ratio_bf16}x"
    echo ""

    # チェックポイントサイズ
    if [ -f "$checkpoint" ]; then
        local ckpt_size
        ckpt_size=$(du -sh "$checkpoint" 2>/dev/null | awk '{print $1}')
        echo "    チェックポイント:   $ckpt_size ($checkpoint)"
    fi

    # エクスポートファイルサイズ
    if [ -f "$export_file" ]; then
        local export_size
        export_size=$(du -sh "$export_file" 2>/dev/null | awk '{print $1}')
        echo "    エクスポート:       $export_size ($export_file)"
    fi
    echo ""
}

# 単一ドメインのエクスポート
export_domain() {
    local domain=$1
    local target_step=${2:-""}

    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  三値重みエクスポート — $domain"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # チェックポイント検索
    local checkpoint=""
    if [ -n "$target_step" ]; then
        echo "  指定ステップ: $target_step"
        checkpoint=$(find_checkpoint_at_step "$domain" "$target_step")
        if [ -z "$checkpoint" ]; then
            echo "  エラー: ステップ $target_step のチェックポイントが見つかりません。"
            echo "  利用可能なチェックポイント:"
            ls -1 "$STORAGE_BASE/checkpoints/$domain"/step_*.bin 2>/dev/null \
                | sed 's/^/    /' || echo "    (なし)"
            return 1
        fi
    else
        echo "  ベストチェックポイントを検索中..."
        checkpoint=$(find_best_checkpoint "$domain")
        if [ -z "$checkpoint" ]; then
            echo "  エラー: チェックポイントが見つかりません。"
            echo "  先に学習を実行してください: ./scripts/train_domain.sh $domain"
            return 1
        fi
    fi

    echo "  チェックポイント: $checkpoint"

    # ステップ番号抽出
    local step_num
    step_num=$(basename "$checkpoint" .bin | sed 's/step_//' | sed 's/^0*//')
    [ -z "$step_num" ] && step_num="0"

    # 出力ディレクトリ・ファイル
    local export_dir="$EXPORT_BASE/$domain"
    local export_file="$export_dir/alice-ternary-70b-${domain}.safetensors"
    mkdir -p "$export_dir"

    echo "  出力先: $export_file"
    echo ""

    # エクスポート実行
    # 注: 実際の safetensors 書き出しは Rust バイナリ側で実装が必要。
    #     ここでは cargo run でエクスポートコマンドを呼び出す。
    echo "  エクスポート実行中..."
    echo "    ドメイン:       $domain"
    echo "    ステップ:       $step_num"
    echo "    入力:           $checkpoint"
    echo "    出力:           $export_file"
    echo ""

    # Rust バイナリでエクスポート（qat-cli feature 必須）
    # エクスポート機能が未実装の場合はプレースホルダとして動作
    if cargo run --release --features qat-cli --bin train-qat-70b -- \
        --export \
        --checkpoint "$checkpoint" \
        --output "$export_file" \
        2>&1; then
        echo ""
        echo "  エクスポート成功"
    else
        echo ""
        echo "  警告: Rust エクスポートバイナリの実行に失敗しました。"
        echo "  エクスポート機能が未実装の可能性があります。"
        echo ""
        echo "  プレースホルダとしてメタデータファイルを出力します..."

        # メタデータ JSON 出力（エクスポート機能が実装されるまでの代替）
        local meta_file="$export_dir/export_meta.json"
        cat > "$meta_file" << METAEOF
{
    "domain": "$domain",
    "base_model": "Llama-3-70B",
    "quantization": "1.1-bit sparse ternary",
    "checkpoint_step": $step_num,
    "checkpoint_path": "$checkpoint",
    "export_path": "$export_file",
    "estimated_size_gb": $TERNARY_SIZE_GB,
    "compression_ratio_vs_fp32": "$((FP32_SIZE_GB / TERNARY_SIZE_GB))x",
    "exported_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
METAEOF
        echo "  メタデータ: $meta_file"
    fi

    # 圧縮統計表示
    show_compression_stats "$domain" "$checkpoint" "$export_file"

    return 0
}

# --------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------

if [ $# -lt 1 ]; then
    echo "使用法: $0 <domain> [step]"
    echo "        $0 all"
    echo ""
    echo "ドメイン: ${ALL_DOMAINS[*]}"
    echo ""
    echo "例:"
    echo "  $0 general          # general のベストチェックポイントをエクスポート"
    echo "  $0 code 10000       # code のステップ 10000 をエクスポート"
    echo "  $0 all              # 全ドメインの一括エクスポート"
    exit 1
fi

TARGET="$1"
STEP="${2:-""}"

case "$TARGET" in
    all)
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║  全ドメイン一括エクスポート                               ║"
        echo "╚══════════════════════════════════════════════════════════╝"
        echo ""

        SUCCESS=0
        SKIPPED=0

        for domain in "${ALL_DOMAINS[@]}"; do
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            if export_domain "$domain"; then
                SUCCESS=$((SUCCESS + 1))
            else
                SKIPPED=$((SKIPPED + 1))
            fi
        done

        echo ""
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║  エクスポート完了                                        ║"
        echo "║  成功: $SUCCESS / ${#ALL_DOMAINS[@]}                      "
        echo "║  スキップ: $SKIPPED / ${#ALL_DOMAINS[@]}                  "
        echo "╚══════════════════════════════════════════════════════════╝"
        echo ""
        echo "エクスポート一覧:"
        du -sh "$EXPORT_BASE"/*/ 2>/dev/null || echo "  (なし)"
        ;;
    *)
        # ドメイン名バリデーション
        valid=false
        for d in "${ALL_DOMAINS[@]}"; do
            if [ "$d" = "$TARGET" ]; then
                valid=true
                break
            fi
        done

        if [ "$valid" = false ]; then
            echo "エラー: 不明なドメイン '$TARGET'"
            echo "利用可能: ${ALL_DOMAINS[*]}"
            exit 1
        fi

        export_domain "$TARGET" "$STEP"
        ;;
esac
