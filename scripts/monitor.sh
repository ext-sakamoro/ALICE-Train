#!/bin/bash
# ============================================================================
# ALICE-Train: 学習モニタースクリプト
#
# 指定ドメインの学習進捗をリアルタイム表示する。
# 10秒間隔で自動更新。
#
# 使用法:
#   ./scripts/monitor.sh general       # 汎用ドメインの進捗監視
#   ./scripts/monitor.sh code          # コード生成ドメインの進捗監視
#   ./scripts/monitor.sh all           # 全ドメインのサマリ表示
# ============================================================================

set -euo pipefail

STORAGE_BASE="${ALICE_TRAIN_STORAGE:-/storage/alice-train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 更新間隔（秒）
REFRESH_INTERVAL=10

# ドメイン一覧
ALL_DOMAINS=(general code japanese math finance medical legal security robotics creative spatial infra)

# ドメイン別 total_steps
declare -A DOMAIN_STEPS
DOMAIN_STEPS[general]=10000
DOMAIN_STEPS[code]=15000
DOMAIN_STEPS[japanese]=12000
DOMAIN_STEPS[math]=8000
DOMAIN_STEPS[finance]=8000
DOMAIN_STEPS[medical]=8000
DOMAIN_STEPS[legal]=8000
DOMAIN_STEPS[security]=8000
DOMAIN_STEPS[robotics]=8000
DOMAIN_STEPS[creative]=10000
DOMAIN_STEPS[spatial]=8000
DOMAIN_STEPS[infra]=8000

# --------------------------------------------------------------------------
# 関数定義
# --------------------------------------------------------------------------

# 最新チェックポイントのステップ番号を取得
get_latest_step() {
    local domain=$1
    local ckpt_dir="$STORAGE_BASE/checkpoints/$domain"

    if [ -d "$ckpt_dir" ]; then
        local latest
        latest=$(ls -1 "$ckpt_dir"/step_*.bin 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [ -n "$latest" ]; then
            # step_00500.bin → 500
            basename "$latest" .bin | sed 's/step_//' | sed 's/^0*//' || echo "0"
            return
        fi
    fi
    echo "0"
}

# 最新ログファイルから直近のメトリクスを取得
get_latest_metrics() {
    local domain=$1
    local log_dir="$STORAGE_BASE/logs/$domain"

    # CSV ログ（train_log.csv）を探す
    local csv_file=""
    if [ -f "$log_dir/train_log.csv" ]; then
        csv_file="$log_dir/train_log.csv"
    else
        # 最新の .csv ファイルを探す
        csv_file=$(ls -1t "$log_dir"/*.csv 2>/dev/null | head -1)
    fi

    if [ -n "$csv_file" ] && [ -f "$csv_file" ]; then
        # CSV の最終行からメトリクスを抽出
        # 想定フォーマット: step,loss,lr,grad_norm,timestamp
        local last_line
        last_line=$(tail -1 "$csv_file" 2>/dev/null)

        if [ -n "$last_line" ]; then
            echo "$last_line"
            return
        fi
    fi

    echo ""
}

# プログレスバー生成
progress_bar() {
    local current=$1
    local total=$2
    local width=30

    if [ "$total" -eq 0 ]; then
        printf "[%-${width}s] 0%%" ""
        return
    fi

    local pct=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    printf "[%s%s] %d%%" \
        "$(printf '#%.0s' $(seq 1 "$filled") 2>/dev/null)" \
        "$(printf '.%.0s' $(seq 1 "$empty") 2>/dev/null)" \
        "$pct"
}

# 単一ドメインの詳細表示
show_domain_detail() {
    local domain=$1
    local total_steps="${DOMAIN_STEPS[$domain]}"
    local current_step
    current_step=$(get_latest_step "$domain")
    local metrics
    metrics=$(get_latest_metrics "$domain")

    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  ALICE-Train モニター — $domain"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo ""

    # 進捗
    printf "  進捗: %s/%s  " "$current_step" "$total_steps"
    progress_bar "$current_step" "$total_steps"
    echo ""
    echo ""

    # メトリクス
    if [ -n "$metrics" ]; then
        # CSV パース（step,loss,lr,grad_norm,timestamp）
        local m_step m_loss m_lr m_grad_norm m_time
        m_step=$(echo "$metrics" | cut -d',' -f1 2>/dev/null)
        m_loss=$(echo "$metrics" | cut -d',' -f2 2>/dev/null)
        m_lr=$(echo "$metrics" | cut -d',' -f3 2>/dev/null)
        m_grad_norm=$(echo "$metrics" | cut -d',' -f4 2>/dev/null)
        m_time=$(echo "$metrics" | cut -d',' -f5 2>/dev/null)

        echo "  最新メトリクス:"
        echo "    ステップ:    ${m_step:-N/A}"
        echo "    Loss:        ${m_loss:-N/A}"
        echo "    学習率:      ${m_lr:-N/A}"
        echo "    勾配ノルム:  ${m_grad_norm:-N/A}"
        echo "    タイムスタンプ: ${m_time:-N/A}"
    else
        echo "  メトリクス: ログファイルが見つかりません"
    fi
    echo ""

    # チェックポイント一覧
    local ckpt_dir="$STORAGE_BASE/checkpoints/$domain"
    echo "  チェックポイント ($ckpt_dir):"
    if [ -d "$ckpt_dir" ] && ls "$ckpt_dir"/step_*.bin &>/dev/null; then
        ls -lhS "$ckpt_dir"/step_*.bin 2>/dev/null | awk '{printf "    %s  %s  %s\n", $NF, $5, $6" "$7}'
    else
        echo "    (なし)"
    fi
    echo ""

    # ETA 計算（概算）
    if [ "$current_step" -gt 0 ] && [ -n "$metrics" ]; then
        local remaining=$((total_steps - current_step))
        echo "  残りステップ: $remaining"
    fi
    echo ""

    # GPU 状態
    if command -v nvidia-smi &>/dev/null; then
        echo "  GPU 状態:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
            --format=csv,noheader 2>/dev/null | while IFS=',' read -r name mem_used mem_total util temp; do
            echo "    GPU:     $name"
            echo "    メモリ:  $mem_used /$mem_total"
            echo "    使用率:  $util"
            echo "    温度:    $temp"
        done
    else
        echo "  GPU: nvidia-smi 利用不可"
    fi

    echo ""
    echo "╚══════════════════════════════════════════════════════════╝"
}

# 全ドメインサマリ表示
show_all_summary() {
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  ALICE-Train — 全ドメインサマリ                                     ║"
    echo "╠══════════════════════════════════════════════════════════════════════╣"
    echo ""
    printf "  %-12s  %8s / %-8s  %-35s  %s\n" "ドメイン" "現在" "合計" "進捗" "状態"
    echo "  ──────────────────────────────────────────────────────────────────"

    for domain in "${ALL_DOMAINS[@]}"; do
        local total_steps="${DOMAIN_STEPS[$domain]}"
        local current_step
        current_step=$(get_latest_step "$domain")

        # 状態判定
        local status
        if [ "$current_step" -ge "$total_steps" ]; then
            status="完了"
        elif [ "$current_step" -gt 0 ]; then
            status="進行中"
        else
            status="未開始"
        fi

        # プログレスバー（短縮版）
        local bar
        bar=$(progress_bar "$current_step" "$total_steps")

        printf "  %-12s  %8s / %-8s  %-35s  %s\n" \
            "$domain" "$current_step" "$total_steps" "$bar" "$status"
    done

    echo ""

    # GPU 状態
    if command -v nvidia-smi &>/dev/null; then
        echo "  GPU:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
            | sed 's/^/    /'
    fi

    echo ""
    echo "╚══════════════════════════════════════════════════════════════════════╝"
}

# --------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------

if [ $# -lt 1 ]; then
    echo "使用法: $0 <domain|all>"
    echo ""
    echo "ドメイン: ${ALL_DOMAINS[*]}"
    echo "all: 全ドメインサマリ"
    echo ""
    echo "${REFRESH_INTERVAL}秒間隔で自動更新。Ctrl+C で終了。"
    exit 1
fi

TARGET="$1"

# ドメイン名バリデーション
if [ "$TARGET" != "all" ]; then
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
fi

# 自動更新ループ
echo "モニター開始 (${REFRESH_INTERVAL}秒間隔で更新、Ctrl+C で終了)"

while true; do
    # 画面クリア
    clear

    echo "更新時刻: $(date '+%Y-%m-%d %H:%M:%S')"

    if [ "$TARGET" = "all" ]; then
        show_all_summary
    else
        show_domain_detail "$TARGET"
    fi

    echo ""
    echo "次の更新: ${REFRESH_INTERVAL}秒後 (Ctrl+C で終了)"

    sleep "$REFRESH_INTERVAL"
done
