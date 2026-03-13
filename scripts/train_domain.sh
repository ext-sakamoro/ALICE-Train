#!/bin/bash
# ============================================================================
# ALICE-Train: ドメイン別学習ランナー
#
# 指定ドメインの QAT 学習を実行する。Spot インスタンスの中断に対応するため
# 内部で auto_resume.sh を利用して自動レジュームを行う。
#
# 使用法:
#   ./scripts/train_domain.sh general        # 汎用ドメイン学習
#   ./scripts/train_domain.sh code           # コード生成ドメイン
#   ./scripts/train_domain.sh all            # 全ドメイン順次実行
#   ./scripts/train_domain.sh all --dry-run  # 設定確認のみ
#
# 環境変数:
#   ALICE_TRAIN_STORAGE — 永続ストレージパス (デフォルト: /storage/alice-train)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STORAGE_BASE="${ALICE_TRAIN_STORAGE:-/storage/alice-train}"

# --------------------------------------------------------------------------
# ドメイン定義
# --------------------------------------------------------------------------
# ドメイン名 → 設定ファイル、説明、推定GPU メモリ、推定学習時間
declare -A DOMAIN_CONFIG
declare -A DOMAIN_DESC
declare -A DOMAIN_GPU_MEM
declare -A DOMAIN_ETA

DOMAIN_CONFIG[general]="configs/qat_70b_general.json"
DOMAIN_CONFIG[code]="configs/qat_70b_code.json"
DOMAIN_CONFIG[japanese]="configs/qat_70b_japanese.json"
DOMAIN_CONFIG[math]="configs/qat_70b_math.json"
DOMAIN_CONFIG[finance]="configs/qat_70b_finance.json"
DOMAIN_CONFIG[medical]="configs/qat_70b_medical.json"
DOMAIN_CONFIG[legal]="configs/qat_70b_legal.json"
DOMAIN_CONFIG[security]="configs/qat_70b_security.json"
DOMAIN_CONFIG[robotics]="configs/qat_70b_robotics.json"
DOMAIN_CONFIG[creative]="configs/qat_70b_creative.json"
DOMAIN_CONFIG[spatial]="configs/qat_70b_spatial.json"
DOMAIN_CONFIG[infra]="configs/qat_70b_infra.json"

DOMAIN_DESC[general]="汎用（一般会話・指示追従）"
DOMAIN_DESC[code]="コード生成"
DOMAIN_DESC[japanese]="日本語特化"
DOMAIN_DESC[math]="数学・科学"
DOMAIN_DESC[finance]="金融・ビジネス"
DOMAIN_DESC[medical]="医療"
DOMAIN_DESC[legal]="法務・コンプライアンス"
DOMAIN_DESC[security]="セキュリティ"
DOMAIN_DESC[robotics]="ロボティクス・組込み"
DOMAIN_DESC[creative]="クリエイティブ（文章生成）"
DOMAIN_DESC[spatial]="空間・3D"
DOMAIN_DESC[infra]="インフラ・DevOps"

# A100 80GB での推定 GPU メモリ使用量
DOMAIN_GPU_MEM[general]="~65GB"
DOMAIN_GPU_MEM[code]="~72GB"       # seq_len=4096
DOMAIN_GPU_MEM[japanese]="~65GB"
DOMAIN_GPU_MEM[math]="~65GB"
DOMAIN_GPU_MEM[finance]="~65GB"
DOMAIN_GPU_MEM[medical]="~65GB"
DOMAIN_GPU_MEM[legal]="~65GB"
DOMAIN_GPU_MEM[security]="~65GB"
DOMAIN_GPU_MEM[robotics]="~55GB"   # seq_len=1024
DOMAIN_GPU_MEM[creative]="~72GB"   # seq_len=4096
DOMAIN_GPU_MEM[spatial]="~65GB"
DOMAIN_GPU_MEM[infra]="~65GB"

# 推定学習時間 (A100 80GB, 単体)
DOMAIN_ETA[general]="~48h"
DOMAIN_ETA[code]="~72h"            # 15000 steps, seq_len=4096
DOMAIN_ETA[japanese]="~56h"        # 12000 steps
DOMAIN_ETA[math]="~38h"
DOMAIN_ETA[finance]="~38h"
DOMAIN_ETA[medical]="~38h"
DOMAIN_ETA[legal]="~38h"
DOMAIN_ETA[security]="~38h"
DOMAIN_ETA[robotics]="~28h"        # seq_len=1024
DOMAIN_ETA[creative]="~48h"        # seq_len=4096
DOMAIN_ETA[spatial]="~38h"
DOMAIN_ETA[infra]="~38h"

# ドメイン順序（all モードで使用）
ALL_DOMAINS=(general code japanese math finance medical legal security robotics creative spatial infra)

# --------------------------------------------------------------------------
# 関数定義
# --------------------------------------------------------------------------

# ドメイン情報バナー表示
show_domain_banner() {
    local domain=$1
    local config="${DOMAIN_CONFIG[$domain]}"
    local desc="${DOMAIN_DESC[$domain]}"
    local gpu_mem="${DOMAIN_GPU_MEM[$domain]}"
    local eta="${DOMAIN_ETA[$domain]}"

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  ALICE-Train QAT — ドメイン学習                         ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  ドメイン:     $domain"
    echo "║  説明:         $desc"
    echo "║  設定:         $config"
    echo "║  推定GPUメモリ: $gpu_mem"
    echo "║  推定学習時間:  $eta"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
}

# ドメイン学習実行
train_domain() {
    local domain=$1
    local dry_run=${2:-false}
    local config="${DOMAIN_CONFIG[$domain]}"
    local checkpoint_dir="$STORAGE_BASE/checkpoints/$domain"
    local log_dir="$STORAGE_BASE/logs/$domain"

    # 設定ファイル存在チェック
    if [ ! -f "$PROJECT_DIR/$config" ]; then
        echo "エラー: 設定ファイルが見つかりません: $PROJECT_DIR/$config"
        return 1
    fi

    show_domain_banner "$domain"

    # ドライラン
    if [ "$dry_run" = true ]; then
        echo "  [DRY-RUN] 設定確認のみ。実行はスキップ。"
        echo "  設定ファイル内容:"
        jq '.' "$PROJECT_DIR/$config" 2>/dev/null || cat "$PROJECT_DIR/$config"
        echo ""
        return 0
    fi

    # ログディレクトリ作成
    mkdir -p "$log_dir"

    # 学習データ存在チェック
    local train_data
    train_data=$(jq -r '.train_data_path' "$PROJECT_DIR/$config" 2>/dev/null || echo "")
    if [ -n "$train_data" ] && [ ! -f "$PROJECT_DIR/$train_data" ] && [ ! -f "$STORAGE_BASE/$train_data" ]; then
        echo "  警告: 学習データが見つかりません: $train_data"
        echo "  $STORAGE_BASE/data/$domain/ にデータを配置してください。"
        echo ""
        return 1
    fi

    # GPU 状態表示
    if command -v nvidia-smi &>/dev/null; then
        echo "GPU 状態:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader
        echo ""
    fi

    # タイムスタンプ付きログファイル
    local log_file="$log_dir/train_$(date '+%Y%m%d_%H%M%S').log"
    echo "ログ: $log_file"
    echo ""

    # auto_resume.sh 経由で学習実行
    # CHECKPOINT_DIR を設定してチェックポイント検索先を指定
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 学習開始: $domain"
    echo ""

    CHECKPOINT_DIR="$checkpoint_dir" \
        "$SCRIPT_DIR/auto_resume.sh" "$PROJECT_DIR/$config" 2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}

    if [ "$exit_code" -eq 0 ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ドメイン '$domain' の学習が正常完了しました。"
    else
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ドメイン '$domain' の学習が異常終了しました (exit=$exit_code)"
        return "$exit_code"
    fi
}

# 全ドメイン一覧表示
show_domains() {
    echo ""
    echo "利用可能なドメイン:"
    echo ""
    printf "  %-12s %-30s %-10s %s\n" "ドメイン" "説明" "GPUメモリ" "推定時間"
    echo "  ────────────────────────────────────────────────────────────────"
    for domain in "${ALL_DOMAINS[@]}"; do
        printf "  %-12s %-30s %-10s %s\n" \
            "$domain" "${DOMAIN_DESC[$domain]}" "${DOMAIN_GPU_MEM[$domain]}" "${DOMAIN_ETA[$domain]}"
    done
    echo ""
}

# --------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------

if [ $# -lt 1 ]; then
    echo "使用法: $0 <domain|all|list> [--dry-run]"
    show_domains
    exit 1
fi

TARGET="$1"
DRY_RUN=false

# オプション解析
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=true
    fi
done

case "$TARGET" in
    list)
        show_domains
        ;;
    all)
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║  全 ${#ALL_DOMAINS[@]} ドメインの順次学習                ║"
        echo "╚══════════════════════════════════════════════════════════╝"

        COMPLETED=0
        FAILED=0
        TOTAL=${#ALL_DOMAINS[@]}

        for domain in "${ALL_DOMAINS[@]}"; do
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  進捗: $COMPLETED/$TOTAL 完了 ($FAILED 失敗)"
            echo "  次のドメイン: $domain"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            if train_domain "$domain" "$DRY_RUN"; then
                COMPLETED=$((COMPLETED + 1))
            else
                FAILED=$((FAILED + 1))
                echo "  ドメイン '$domain' が失敗。次のドメインへ進みます。"
            fi
        done

        echo ""
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║  全ドメイン学習完了                                      ║"
        echo "╠══════════════════════════════════════════════════════════╣"
        echo "║  成功: $COMPLETED / $TOTAL                               "
        echo "║  失敗: $FAILED / $TOTAL                                  "
        echo "╚══════════════════════════════════════════════════════════╝"
        ;;
    *)
        # 個別ドメイン
        if [ -z "${DOMAIN_CONFIG[$TARGET]+x}" ]; then
            echo "エラー: 不明なドメイン '$TARGET'"
            show_domains
            exit 1
        fi

        train_domain "$TARGET" "$DRY_RUN"
        ;;
esac
