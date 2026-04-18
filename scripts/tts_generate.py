#!/usr/bin/env python3
"""
Qwen3-TTS Voice Clone + Batch Generation Script
場面別リファレンスでテキストから音声を生成し、Boxへ自動転送する。

Usage:
  # Single
  python3 tts_generate.py --ref-type gentle --text "セリフ" --output out.wav

  # Batch (JSON)
  python3 tts_generate.py --batch batch.json

  # Batch JSON format:
  [
    {"text": "セリフ", "filename": "line_01.wav", "ref_type": "gentle", "max_new_tokens": 200},
    {"text": "セリフ", "filename": "line_02.wav", "ref_type": "intense"}
  ]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch
import soundfile as sf


BOX_HOST = "sftp.services.box.com"
BOX_USER = "sakamoro@extoria.co.jp"
BOX_DEST = "Shizai/DLsite/Voice"

REF_DIR = "/notebooks/tts_ref"

REFS = {
    "gentle": {
        "audio": f"{REF_DIR}/ref_gentle.wav",
        "text": "大丈夫、そのままでいいよ。私の膝でくつろいでて。",
    },
    "normal": {
        "audio": f"{REF_DIR}/ref_normal.wav",
        "text": "もう少し休んだ方がいいよ。私も君も。いっぱい逃げて疲れちゃったよね。",
    },
    "intense": {
        "audio": f"{REF_DIR}/ref_intense.wav",
        "text": "あの時の私に説教したい気分だわ",
    },
    "breathy": {
        "audio": f"{REF_DIR}/ref_breathy.wav",
        "text": "鹿の山",
    },
    "playful": {
        "audio": f"{REF_DIR}/ref_playful.wav",
        "text": "君ってば、こういうところは鈍いんだから。",
    },
    "climax": {
        "audio": f"{REF_DIR}/ref_climax.wav",
        "text": "あ、そうだ、頭のマッサージしてあげようか？",
    },
}


def load_model(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda:0"):
    from qwen_tts import Qwen3TTSModel

    print(f"Loading model: {model_name} ...")
    t0 = time.time()

    try:
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        print("flash_attention_2 not available, using sdpa")
        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model


def upload_to_box(file_path):
    box_pass = os.environ.get("BOX_SFTP_PASS", "")
    if not box_pass:
        return False

    fname = os.path.basename(file_path)
    cmds = f"cd {BOX_DEST}\nput {file_path}\nbye\n"

    try:
        result = subprocess.run(
            ["sshpass", "-p", box_pass, "sftp", "-oBatchMode=no", "-oStrictHostKeyChecking=no",
             "-B", "32768", f"{BOX_USER}@{BOX_HOST}"],
            input=cmds, capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            print(f"  [Box] {fname}: OK")
            return True
        else:
            print(f"  [Box ERR] {fname}: {result.stderr[:80]}")
            return False
    except Exception as e:
        print(f"  [Box ERR] {fname}: {e}")
        return False


def generate_one(model, text, ref_type, output_path, max_new_tokens=300, auto_upload=True):
    ref = REFS.get(ref_type)
    if not ref:
        print(f"  [WARN] Unknown ref_type '{ref_type}', falling back to 'normal'")
        ref = REFS["normal"]

    print(f"[{ref_type}] {text[:50]}...")
    t0 = time.time()

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Japanese",
        ref_audio=ref["audio"],
        ref_text=ref["text"],
        max_new_tokens=max_new_tokens,
    )

    sf.write(output_path, wavs[0], sr)
    duration = len(wavs[0]) / sr
    print(f"  -> {os.path.basename(output_path)} ({duration:.1f}s audio, {time.time() - t0:.1f}s gen)")

    if auto_upload:
        upload_to_box(output_path)

    return output_path


def process_batch(model, batch_file, output_dir, default_ref_type="normal",
                  default_max_tokens=300, auto_upload=True):
    with open(batch_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    generated = []

    for i, item in enumerate(items):
        text = item.get("text", "")
        if not text:
            continue
        filename = item.get("filename", f"voice_{i:04d}.wav")
        ref_type = item.get("ref_type", default_ref_type)
        max_tok = item.get("max_new_tokens", default_max_tokens)

        output_path = os.path.join(output_dir, filename)
        generate_one(model, text, ref_type, output_path,
                     max_new_tokens=max_tok, auto_upload=auto_upload)
        generated.append(output_path)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice Clone Generator")
    parser.add_argument("--text", type=str, help="Single text to generate")
    parser.add_argument("--batch", type=str, help="JSON batch file path")
    parser.add_argument("--ref-type", type=str, default="normal",
                        choices=list(REFS.keys()),
                        help="Reference type (default: normal)")
    parser.add_argument("--output", type=str, default="/notebooks/tts_output/output.wav",
                        help="Output file path (single mode)")
    parser.add_argument("--output-dir", type=str, default="/notebooks/tts_output",
                        help="Output directory (batch mode)")
    parser.add_argument("--max-new-tokens", type=int, default=300,
                        help="Max generation tokens (12tok ≈ 1s audio)")
    parser.add_argument("--no-upload", action="store_true", help="Skip Box upload")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--list-refs", action="store_true", help="List available ref types")
    args = parser.parse_args()

    if args.list_refs:
        print("Available reference types:")
        for k, v in REFS.items():
            print(f"  {k:10s} -> {v['text']}")
        return

    if not args.text and not args.batch:
        parser.error("Either --text or --batch is required")

    auto_upload = not args.no_upload
    model = load_model(args.model, args.device)

    if args.batch:
        generated = process_batch(
            model, args.batch, args.output_dir,
            default_ref_type=args.ref_type,
            default_max_tokens=args.max_new_tokens,
            auto_upload=auto_upload,
        )
        print(f"\nDone: {len(generated)} files in {args.output_dir}")
    else:
        generate_one(
            model, args.text, args.ref_type, args.output,
            max_new_tokens=args.max_new_tokens, auto_upload=auto_upload,
        )


if __name__ == "__main__":
    main()
