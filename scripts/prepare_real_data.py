#!/usr/bin/env python3
"""HuggingFace からテキストデータをダウンロードし、トークナイズして train.bin を生成。

SlimPajama (100M+ tokens) 対応: ストリーミングモードで大規模コーパスから取得。

Usage:
    # wikitext (小規模テスト用)
    python3 scripts/prepare_real_data.py \
        --model_path models/Qwen--Qwen3.5-9B \
        --output data/qwen35/train.bin \
        --max_tokens 2000000

    # SlimPajama (本番用 100M tokens)
    python3 scripts/prepare_real_data.py \
        --model_path models/Qwen--Qwen3.5-9B \
        --output data/qwen35/train.bin \
        --max_tokens 100000000 \
        --dataset cerebras/SlimPajama-627B \
        --streaming

    # eval データ生成 (別 split)
    python3 scripts/prepare_real_data.py \
        --model_path models/Qwen--Qwen3.5-9B \
        --output data/qwen35/eval.bin \
        --max_tokens 10000 \
        --dataset cerebras/SlimPajama-627B \
        --streaming \
        --split validation
"""
import argparse
import struct
import os

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for ALICE-Train QAT")
    parser.add_argument("--model_path", type=str, default="models/Qwen--Qwen3.5-9B",
                        help="Path to model dir (contains tokenizer.json)")
    parser.add_argument("--output", type=str, default="data/qwen35/train.bin",
                        help="Output path for raw u32 LE token binary")
    parser.add_argument("--max_tokens", type=int, default=500_000,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset config name (e.g. wikitext-103-raw-v1)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode (required for large datasets like SlimPajama)")
    parser.add_argument("--text_field", type=str, default=None,
                        help="Name of text field in dataset (auto-detected if not set)")
    args = parser.parse_args()

    # wikitext のデフォルト config
    if args.dataset == "wikitext" and args.dataset_config is None:
        args.dataset_config = "wikitext-103-raw-v1"

    print(f"=== ALICE-Train: Data Preparation ===")
    print(f"  Model: {args.model_path}")
    print(f"  Dataset: {args.dataset}" + (f"/{args.dataset_config}" if args.dataset_config else ""))
    print(f"  Split: {args.split}")
    print(f"  Streaming: {args.streaming}")
    print(f"  Max tokens: {args.max_tokens:,}")
    print()

    # Tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}")
    print()

    # Dataset
    print("Loading dataset...")
    from datasets import load_dataset
    load_kwargs = {
        "split": args.split,
    }
    if args.streaming:
        load_kwargs["streaming"] = True
    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, **load_kwargs)
    else:
        ds = load_dataset(args.dataset, **load_kwargs)

    if not args.streaming:
        print(f"  Samples: {len(ds)}")
    else:
        print(f"  Streaming mode (samples counted during tokenization)")
    print()

    # テキストフィールド自動検出
    text_field = args.text_field
    if text_field is None:
        # 最初のサンプルを見て推定
        if args.streaming:
            first = next(iter(ds))
        else:
            first = ds[0]
        for candidate in ["text", "content", "document", "passage"]:
            if candidate in first:
                text_field = candidate
                break
        if text_field is None:
            # 最初の string フィールド
            for k, v in first.items():
                if isinstance(v, str):
                    text_field = k
                    break
        if text_field is None:
            print("ERROR: テキストフィールドが見つかりません")
            print(f"  Available fields: {list(first.keys())}")
            exit(1)
        print(f"  Text field: '{text_field}'")

    # Tokenize
    print("Tokenizing...")
    all_tokens = []
    sample_count = 0
    for i, sample in enumerate(ds):
        text = sample.get(text_field, "")
        if not text or len(text.strip()) < 10:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        # BOS + tokens + EOS (skip None, e.g. Qwen has no BOS)
        if tokenizer.bos_token_id is not None:
            all_tokens.append(tokenizer.bos_token_id)
        all_tokens.extend(tokens)
        if tokenizer.eos_token_id is not None:
            all_tokens.append(tokenizer.eos_token_id)

        sample_count += 1

        if len(all_tokens) >= args.max_tokens:
            all_tokens = all_tokens[:args.max_tokens]
            break

        if (i + 1) % 10000 == 0:
            print(f"  {i+1} samples processed, {len(all_tokens):,} tokens so far...")

    print(f"  Total: {sample_count:,} samples, {len(all_tokens):,} tokens")
    print()

    # Save as raw u32 LE
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        for tok in all_tokens:
            f.write(struct.pack("<I", tok))

    file_size = os.path.getsize(args.output)
    print(f"Saved: {args.output} ({file_size:,} bytes, {file_size/1024/1024:.1f} MB)")
    print(f"  {len(all_tokens):,} tokens × 4 bytes = {len(all_tokens)*4:,} bytes")

    # Verify: decode first 200 tokens
    first_text = tokenizer.decode(all_tokens[:200])
    print(f"\n  First 200 tokens decoded:\n  {first_text[:300]}...")

if __name__ == "__main__":
    main()
