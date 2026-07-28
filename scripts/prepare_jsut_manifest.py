#!/usr/bin/env python3
"""JSUT audio + transcriptions を TtsDataset manifest jsonl に変換する helper。

Usage:
    python3 prepare_jsut_manifest.py \\
        --jsut_root /path/to/jsut_ver1.1 \\
        --output data/jsut/manifest.jsonl \\
        --tokenizer_vocab_file data/jsut/mora_vocab.txt

Requirements:
    pip install pyopenjtalk numpy scipy
    (pyopenjtalk は日本語 G2P + F0 抽出 + duration alignment に使用)

Notes:
    - JSUT corpus は https://sites.google.com/site/shinnosuketakamichi/publication/jsut
    - CC BY-SA 4.0 license
    - 基本 5000 utt (BASIC5000 subset) を対象
    - 各 utt の transcription (yomikata) から pyopenjtalk で mora 系列を取得
    - Duration は音素 duration を hop_length で frame 化 (簡易 forced alignment、
      精度が必要なら Montreal Forced Aligner (MFA) 併用推奨)
    - 出力は 1 行 1 utt の JSON、TtsManifestEntry と互換
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSUT → TtsDataset manifest jsonl 変換")
    p.add_argument("--jsut_root", required=True, type=Path, help="JSUT corpus root (jsut_ver1.1/)")
    p.add_argument("--output", required=True, type=Path, help="出力 manifest jsonl パス")
    p.add_argument("--subset", default="basic5000", help="対象 subset (basic5000 / travel1000 / etc)")
    p.add_argument("--sample_rate", default=24000, type=int, help="target sample rate")
    p.add_argument("--hop_length", default=256, type=int, help="hop length for frame calculation")
    p.add_argument("--tokenizer_vocab_file", type=Path, help="mora vocab file (省略時は auto build)")
    return p.parse_args()


def load_transcriptions(jsut_root: Path, subset: str) -> dict[str, str]:
    """JSUT の transcript_utf8.txt から {utt_id: text} dict を返す。"""
    trans_file = jsut_root / subset / "transcript_utf8.txt"
    if not trans_file.exists():
        raise FileNotFoundError(f"transcription file not found: {trans_file}")

    trans = {}
    with trans_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: "BASIC5000_0001:水をマレーシアから買わなくてはならないのです。"
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            utt_id, text = parts[0].strip(), parts[1].strip()
            trans[utt_id] = text
    return trans


def text_to_moras_via_pyopenjtalk(text: str) -> tuple[list[str], list[int]]:
    """pyopenjtalk で text → mora list + accent type list を取得する。

    Returns:
        (moras, accent_types) where accent_types is per-phrase list.
    """
    try:
        import pyopenjtalk
    except ImportError:
        raise ImportError(
            "pyopenjtalk not installed. Run: pip install pyopenjtalk"
        )

    # extract_fullcontext returns list of HMM full-context labels
    labels = pyopenjtalk.extract_fullcontext(text)
    moras: list[str] = []
    for label in labels:
        # Full-context label: "phoneme+prev-...+curmora-...+next-..."
        # 抜き出しは簡易: /A:accent_type/E:.../ etc
        # ここでは pyopenjtalk の g2p helper を代わりに使用する簡易版
        m = re.search(r"-([a-zA-Z]+)\+", label)
        if m:
            moras.append(m.group(1))

    # Accent type は簡易 (真の accent は HMM label の /A:X-Y-Z/ から取得)
    accent_types = [0] * max(1, len(moras) // 5)  # 5-mora / phrase 目安
    return moras, accent_types


def moras_to_ids(moras: list[str], vocab: dict[str, int]) -> list[int]:
    """mora string list → id list。vocab に無い mora は 0 (UNK) にする。"""
    return [vocab.get(m, 0) for m in moras]


def compute_durations_ms(text: str, sample_rate: int) -> list[int]:
    """簡易 duration 推定: 各 mora を text 長比例で均等割り (実運用は MFA 推奨)。"""
    try:
        import pyopenjtalk
        # 実 duration は pyopenjtalk.run_frontend で HMM alignment 取得可能
        # ここでは placeholder として 80 ms / mora を返す
    except ImportError:
        pass
    # Placeholder: 80 ms per mora (実運用は forced alignment 必須)
    moras, _ = text_to_moras_via_pyopenjtalk(text)
    return [80] * len(moras)


def build_vocab(all_moras: list[str]) -> dict[str, int]:
    """全 utt の mora 集合から vocab dict {mora: id} を構築。id 0 は UNK 予約。"""
    unique = sorted(set(all_moras))
    vocab = {"<unk>": 0, "<pad>": 1}
    for m in unique:
        if m not in vocab:
            vocab[m] = len(vocab)
    return vocab


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] Loading JSUT transcriptions from {args.jsut_root}/{args.subset}...")
    trans = load_transcriptions(args.jsut_root, args.subset)
    print(f"[info] {len(trans)} utterances loaded")

    # Pass 1: 全 mora 収集 → vocab 構築
    print("[info] Pass 1: G2P (extract moras) ...")
    all_moras: list[str] = []
    utt_moras: dict[str, list[str]] = {}
    for i, (utt_id, text) in enumerate(trans.items()):
        try:
            moras, _ = text_to_moras_via_pyopenjtalk(text)
            utt_moras[utt_id] = moras
            all_moras.extend(moras)
        except Exception as e:
            print(f"[warn] {utt_id}: G2P failed: {e}", file=sys.stderr)
        if (i + 1) % 500 == 0:
            print(f"  [progress] {i + 1}/{len(trans)}")

    vocab = build_vocab(all_moras)
    print(f"[info] Vocabulary size: {len(vocab)}")

    if args.tokenizer_vocab_file:
        args.tokenizer_vocab_file.parent.mkdir(parents=True, exist_ok=True)
        with args.tokenizer_vocab_file.open("w", encoding="utf-8") as f:
            for m, i in vocab.items():
                f.write(f"{m}\t{i}\n")
        print(f"[info] Vocab written to {args.tokenizer_vocab_file}")

    # Pass 2: manifest jsonl 書き出し
    print(f"[info] Pass 2: writing manifest to {args.output}...")
    written = 0
    with args.output.open("w", encoding="utf-8") as f:
        for utt_id, text in trans.items():
            if utt_id not in utt_moras:
                continue
            moras = utt_moras[utt_id]
            durations_ms = compute_durations_ms(text, args.sample_rate)
            if len(durations_ms) != len(moras):
                durations_ms = [80] * len(moras)
            mora_ids = moras_to_ids(moras, vocab)

            # phoneme_alignment_ms: cumulative sum
            alignment_ms = []
            cur = 0
            for d in durations_ms:
                alignment_ms.append(cur)
                cur += d

            entry = {
                "audio_path": f"{args.subset}/wav/{utt_id}.wav",
                "text_input_ids": mora_ids,  # tokenizer 済 id (mora id で兼用)
                "text_moras": mora_ids[: min(255, len(mora_ids))],  # u8 制限
                "text_accent_types": [0],  # 1 phrase 前提 (簡易)
                "phoneme_alignment_ms": alignment_ms,
                "speaker_id": 0,  # JSUT single speaker
                "durations_ms": durations_ms,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"[info] Done. {written} entries written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
