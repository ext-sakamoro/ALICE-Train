#!/bin/bash
# ============================================================================
# ALICE-Train: Paperspace A100 80GB セットアップスクリプト
#
# Paperspace Gradient インスタンス上で ALICE-Train の学習環境を構築する。
# /storage/ は永続ストレージとしてインスタンス再起動後も保持される。
#
# 使用法:
#   ./scripts/setup_paperspace.sh
# ============================================================================

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALICE-Train — Paperspace A100 80GB セットアップ         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# --------------------------------------------------------------------------
# 1. 永続ストレージのベースディレクトリ
# --------------------------------------------------------------------------
STORAGE_BASE="/storage/alice-train"

echo "[1/7] 永続ストレージディレクトリ作成..."

# ドメイン一覧
DOMAINS=(general code japanese math finance medical legal security robotics creative spatial infra)

# モデル・データ・チェックポイント・ログ・エクスポート用ディレクトリ
mkdir -p "$STORAGE_BASE/models"
mkdir -p "$STORAGE_BASE/exports"

for domain in "${DOMAINS[@]}"; do
    mkdir -p "$STORAGE_BASE/data/$domain"
    mkdir -p "$STORAGE_BASE/checkpoints/$domain"
    mkdir -p "$STORAGE_BASE/logs/$domain"
done

echo "  ディレクトリ構造:"
echo "    $STORAGE_BASE/"
echo "    ├── models/           — ベースモデル (HuggingFace)"
echo "    ├── data/{domain}/    — 学習・評価データ"
echo "    ├── checkpoints/{domain}/ — チェックポイント"
echo "    ├── logs/{domain}/    — 学習ログ"
echo "    └── exports/          — エクスポート済み三値重み"
echo ""

# --------------------------------------------------------------------------
# 2. システム依存パッケージのインストール
# --------------------------------------------------------------------------
echo "[2/7] システムパッケージインストール..."

# Paperspace は Ubuntu ベース
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        build-essential \
        pkg-config \
        libssl-dev \
        cmake \
        git \
        curl \
        wget \
        htop \
        tmux \
        jq
    echo "  apt パッケージインストール完了"
else
    echo "  警告: apt-get が見つかりません。手動でビルド依存をインストールしてください。"
fi

# --------------------------------------------------------------------------
# 3. Rust ツールチェインのインストール
# --------------------------------------------------------------------------
echo "[3/7] Rust ツールチェインインストール..."

if command -v rustc &>/dev/null; then
    RUST_VER=$(rustc --version)
    echo "  既にインストール済み: $RUST_VER"
    # 最新版に更新
    rustup update stable
else
    # 非対話モードでインストール
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
    echo "  Rust インストール完了: $(rustc --version)"
fi

# --------------------------------------------------------------------------
# 4. HuggingFace CLI のインストール
# --------------------------------------------------------------------------
echo "[4/7] HuggingFace CLI インストール..."

if command -v huggingface-cli &>/dev/null; then
    echo "  既にインストール済み"
else
    pip install -q huggingface_hub[cli]
    echo "  HuggingFace CLI インストール完了"
fi

# --------------------------------------------------------------------------
# 5. プロジェクトディレクトリへのシンボリックリンク作成
# --------------------------------------------------------------------------
echo "[5/7] シンボリックリンク作成..."

# プロジェクトルートの検出
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 永続ストレージへのシンボリックリンク
# （既存ディレクトリがある場合はスキップ）
for dir in models data checkpoints; do
    TARGET="$PROJECT_DIR/$dir"
    SOURCE="$STORAGE_BASE/$dir"
    if [ -L "$TARGET" ]; then
        echo "  シンボリックリンク既存: $TARGET -> $(readlink "$TARGET")"
    elif [ -d "$TARGET" ]; then
        echo "  警告: $TARGET は実ディレクトリ。手動で移行してください。"
    else
        ln -s "$SOURCE" "$TARGET"
        echo "  作成: $TARGET -> $SOURCE"
    fi
done

# ログディレクトリ
if [ ! -L "$PROJECT_DIR/logs" ] && [ ! -d "$PROJECT_DIR/logs" ]; then
    ln -s "$STORAGE_BASE/logs" "$PROJECT_DIR/logs"
    echo "  作成: $PROJECT_DIR/logs -> $STORAGE_BASE/logs"
fi

# エクスポートディレクトリ
if [ ! -L "$PROJECT_DIR/exports" ] && [ ! -d "$PROJECT_DIR/exports" ]; then
    ln -s "$STORAGE_BASE/exports" "$PROJECT_DIR/exports"
    echo "  作成: $PROJECT_DIR/exports -> $STORAGE_BASE/exports"
fi

echo ""

# --------------------------------------------------------------------------
# 6. 環境変数の設定
# --------------------------------------------------------------------------
echo "[6/7] 環境変数設定..."

# .env ファイルに書き出し（永続ストレージ）
ENV_FILE="$STORAGE_BASE/.env"
cat > "$ENV_FILE" << 'ENVEOF'
# ALICE-Train 環境変数
# source /storage/alice-train/.env で読み込み

# CUDA 関連
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# Rust 最適化
export RUSTFLAGS="-C target-cpu=native"
export CARGO_INCREMENTAL=0

# ALICE-Train パス
export ALICE_TRAIN_STORAGE="/storage/alice-train"
export ALICE_TRAIN_MODELS="/storage/alice-train/models"
export ALICE_TRAIN_DATA="/storage/alice-train/data"
export ALICE_TRAIN_CHECKPOINTS="/storage/alice-train/checkpoints"
export ALICE_TRAIN_LOGS="/storage/alice-train/logs"

# HuggingFace キャッシュ（永続ストレージに配置）
export HF_HOME="/storage/huggingface"
export HUGGINGFACE_HUB_CACHE="/storage/huggingface/hub"

# Rayon スレッド数（A100 ホストの CPU コア数に合わせる）
export RAYON_NUM_THREADS=$(nproc)
ENVEOF

# 現在のシェルに反映
# shellcheck source=/dev/null
source "$ENV_FILE"

# .bashrc に追記（次回ログイン時に自動読み込み）
BASHRC_LINE="source /storage/alice-train/.env 2>/dev/null || true"
if ! grep -qF "$BASHRC_LINE" "$HOME/.bashrc" 2>/dev/null; then
    echo "$BASHRC_LINE" >> "$HOME/.bashrc"
    echo "  .bashrc に環境変数読み込みを追加"
fi

echo "  環境変数: $ENV_FILE"
echo ""

# --------------------------------------------------------------------------
# 7. ALICE-Train のビルド
# --------------------------------------------------------------------------
echo "[7/7] ALICE-Train ビルド (release + qat-cli)..."

cd "$PROJECT_DIR"

# リリースビルド（qat-cli feature 有効）
cargo build --release --features qat-cli 2>&1 | tail -5

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  セットアップ完了                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "次のステップ:"
echo "  1. モデルダウンロード:  ./scripts/download_models.sh"
echo "  2. 学習データ配置:     $STORAGE_BASE/data/{domain}/"
echo "  3. 学習開始:          ./scripts/train_domain.sh general"
echo ""

# GPU 情報の表示
if command -v nvidia-smi &>/dev/null; then
    echo "GPU 情報:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
fi

# ディスク使用量
echo "ストレージ使用量:"
df -h /storage 2>/dev/null || df -h .
echo ""
