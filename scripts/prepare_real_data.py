#!/usr/bin/env python3
"""HuggingFace からテキストデータをダウンロードし、Llama-3 tokenizer でトークナイズして train.bin を生成。

Usage:
    python3 scripts/prepare_real_data.py \
        --model_path models/meta-llama--Llama-3.1-8B-Instruct \
        --output data/general/train.bin \
        --max_tokens 1000000
"""
import argparse
import struct
import os

def main():
    parser = argparse.ArgumentParser(description="Prepare real training data for ALICE-Train QAT")
    parser.add_argument("--model_path", type=str, default="models/meta-llama--Llama-3.1-8B-Instruct",
                        help="Path to Llama model dir (contains tokenizer.json)")
    parser.add_argument("--output", type=str, default="data/general/train.bin",
                        help="Output path for raw u32 LE token binary")
    parser.add_argument("--max_tokens", type=int, default=500_000,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1",
                        help="Dataset config name")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split")
    args = parser.parse_args()

    print(f"=== ALICE-Train: Real Data Preparation ===")
    print(f"  Model: {args.model_path}")
    print(f"  Dataset: {args.dataset}/{args.dataset_config} ({args.split})")
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
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    print(f"  Samples: {len(ds)}")
    print()

    # Tokenize
    print("Tokenizing...")
    all_tokens = []
    for i, sample in enumerate(ds):
        text = sample.get("text", "")
        if not text or len(text.strip()) < 10:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        # BOS + tokens + EOS
        all_tokens.append(tokenizer.bos_token_id)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)

        if len(all_tokens) >= args.max_tokens:
            all_tokens = all_tokens[:args.max_tokens]
            break

        if (i + 1) % 10000 == 0:
            print(f"  {i+1} samples processed, {len(all_tokens):,} tokens so far...")

    print(f"  Total: {len(all_tokens):,} tokens")
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
