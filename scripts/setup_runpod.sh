#!/bin/bash
# ============================================================================
# ALICE-Train: RunPod GPU Pod セットアップスクリプト
#
# RunPod Pod 上で ALICE-Train の量子化学習環境を構築する。
# /workspace/ は永続ストレージとしてPod停止後も保持される。
#
# 使用法:
#   bash scripts/setup_runpod.sh
#
# 前提:
#   - RunPod Pod (PyTorch template 推奨)
#   - GPU: A100 80GB 以上
# ============================================================================

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALICE-Train — RunPod GPU Pod セットアップ                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# --------------------------------------------------------------------------
# 1. 永続ストレージのベースディレクトリ
# --------------------------------------------------------------------------
STORAGE_BASE="/workspace/alice-train"

echo "[1/7] 永続ストレージディレクトリ作成..."

DOMAINS=(general code japanese math finance medical legal security robotics creative spatial infra)

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

if command -v apt-get &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq \
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
    rustup update stable
else
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
    pip install -q huggingface_hub[cli] transformers datasets
    echo "  HuggingFace CLI + transformers インストール完了"
fi

# --------------------------------------------------------------------------
# 5. プロジェクトディレクトリへのシンボリックリンク作成
# --------------------------------------------------------------------------
echo "[5/7] シンボリックリンク作成..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

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

if [ ! -L "$PROJECT_DIR/logs" ] && [ ! -d "$PROJECT_DIR/logs" ]; then
    ln -s "$STORAGE_BASE/logs" "$PROJECT_DIR/logs"
    echo "  作成: $PROJECT_DIR/logs -> $STORAGE_BASE/logs"
fi

if [ ! -L "$PROJECT_DIR/exports" ] && [ ! -d "$PROJECT_DIR/exports" ]; then
    ln -s "$STORAGE_BASE/exports" "$PROJECT_DIR/exports"
    echo "  作成: $PROJECT_DIR/exports -> $STORAGE_BASE/exports"
fi

echo ""

# --------------------------------------------------------------------------
# 6. 環境変数の設定
# --------------------------------------------------------------------------
echo "[6/7] 環境変数設定..."

ENV_FILE="$STORAGE_BASE/.env"
cat > "$ENV_FILE" << 'ENVEOF'
# ALICE-Train 環境変数 (RunPod)
# source /workspace/alice-train/.env で読み込み

# CUDA 関連
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# Rust 最適化
export RUSTFLAGS="-C target-cpu=native"
export CARGO_INCREMENTAL=0

# ALICE-Train パス
export ALICE_TRAIN_STORAGE="/workspace/alice-train"
export ALICE_TRAIN_MODELS="/workspace/alice-train/models"
export ALICE_TRAIN_DATA="/workspace/alice-train/data"
export ALICE_TRAIN_CHECKPOINTS="/workspace/alice-train/checkpoints"
export ALICE_TRAIN_LOGS="/workspace/alice-train/logs"

# HuggingFace キャッシュ（永続ストレージに配置）
export HF_HOME="/workspace/huggingface"
export HUGGINGFACE_HUB_CACHE="/workspace/huggingface/hub"

# Rayon スレッド数
export RAYON_NUM_THREADS=$(nproc)
ENVEOF

# shellcheck source=/dev/null
source "$ENV_FILE"

# .bashrc に追記
BASHRC_LINE="source /workspace/alice-train/.env 2>/dev/null || true"
if ! grep -qF "$BASHRC_LINE" "$HOME/.bashrc" 2>/dev/null; then
    echo "$BASHRC_LINE" >> "$HOME/.bashrc"
    echo "  .bashrc に環境変数読み込みを追加"
fi

# Rust PATH を .bashrc に追加（RunPod再接続時用）
CARGO_LINE='source "$HOME/.cargo/env" 2>/dev/null || true'
if ! grep -qF '.cargo/env' "$HOME/.bashrc" 2>/dev/null; then
    echo "$CARGO_LINE" >> "$HOME/.bashrc"
    echo "  .bashrc に Cargo PATH を追加"
fi

echo "  環境変数: $ENV_FILE"
echo ""

# --------------------------------------------------------------------------
# 7. ALICE-Train のビルド (CUDA feature 有効)
# --------------------------------------------------------------------------
echo "[7/7] ALICE-Train ビルド (release + qat-cuda)..."

cd "$PROJECT_DIR"

# CUDA ツールキットの確認
if command -v nvcc &>/dev/null; then
    echo "  CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
else
    echo "  警告: nvcc が見つかりません。CUDA_HOME を確認してください。"
    # RunPod PyTorch template では通常 /usr/local/cuda にある
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
        echo "  CUDA_HOME=$CUDA_HOME を設定"
    fi
fi

# qat-cuda feature でビルド（CUDA cuBLAS + safetensors + clap）
cargo build --release --features qat-cuda 2>&1 | tail -5

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  セットアップ完了                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "次のステップ:"
echo "  1. モデルダウンロード:  bash scripts/download_models.sh small"
echo "  2. 量子化実行:         bash scripts/run_qat_runpod.sh 1b"
echo "  3. 全モデル順次:       bash scripts/run_qat_runpod.sh all"
echo ""

# GPU 情報の表示
if command -v nvidia-smi &>/dev/null; then
    echo "GPU 情報:"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
    echo ""
fi

# ディスク使用量
echo "ストレージ使用量:"
df -h /workspace 2>/dev/null || df -h .
echo ""
