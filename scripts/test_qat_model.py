#!/usr/bin/env python3
"""ALICE-Train QAT チェックポイントテスト

ベース Qwen2.5-1.5B-Instruct に QAT 学習済み delta を適用し、
三値量子化 (FakeQuantize) を経て推論テストを行う。

delta 適用方式: w_effective = round((base + delta) / γ) * clamp(-1,1) * γ
  γ = mean(|base + delta|)  (per-tensor calibration scale)
"""

import struct
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 設定 ---
CHECKPOINT_DIR = os.path.expanduser("~/ALICE-Train/checkpoints/qat_qwen_1.5b")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_LAYERS = 28

# delta ファイル → HuggingFace パラメータ名のマッピング
# projection 重み（量子化対象）
QUANTIZE_MAPPING = {
    "q_proj":     "self_attn.q_proj.weight",
    "k_proj":     "self_attn.k_proj.weight",
    "v_proj":     "self_attn.v_proj.weight",
    "o_proj":     "self_attn.o_proj.weight",
    "gate_proj":  "mlp.gate_proj.weight",
    "up_proj":    "mlp.up_proj.weight",
    "down_proj":  "mlp.down_proj.weight",
}
# 非量子化パラメータ（norm, bias — delta を直接加算）
DIRECT_MAPPING = {
    "attn_norm":  "input_layernorm.weight",
    "ffn_norm":   "post_attention_layernorm.weight",
    "q_bias":     "self_attn.q_proj.bias",
    "k_bias":     "self_attn.k_proj.bias",
    "v_bias":     "self_attn.v_proj.bias",
}


def load_delta(layer_idx: int, weight_name: str) -> np.ndarray | None:
    path = os.path.join(CHECKPOINT_DIR, f"delta_layer{layer_idx}_{weight_name}.bin")
    if not os.path.exists(path):
        return None
    data = open(path, "rb").read()
    count = len(data) // 4
    return np.array(struct.unpack(f"<{count}f", data), dtype=np.float32)


def ternary_fake_quantize(w: torch.Tensor) -> torch.Tensor:
    """三値 FakeQuantize: round(w/γ) → clamp(-1,1) → ×γ
    γ = mean(|w|) (per-tensor)
    temperature = 0.98 (学習終了時の値)
    """
    temp = 0.98
    gamma = w.abs().mean().clamp(min=1e-10)
    scaled = w / gamma / temp
    q = scaled.round().clamp(-1.0, 1.0) * gamma
    return q


def apply_deltas_with_quantize(model):
    """全レイヤーの delta 適用 + 三値量子化。"""
    params = dict(model.named_parameters())
    applied_q = 0
    applied_d = 0

    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.layers.{layer_idx}."

        # 量子化対象: base + delta → ternary fake quantize
        for delta_name, hf_suffix in QUANTIZE_MAPPING.items():
            delta = load_delta(layer_idx, delta_name)
            if delta is None:
                continue
            param_name = prefix + hf_suffix
            if param_name not in params:
                continue

            param = params[param_name]
            delta_t = torch.from_numpy(delta).reshape(param.shape)
            with torch.no_grad():
                merged = param.data + delta_t.to(param.dtype)
                param.data.copy_(ternary_fake_quantize(merged))
            applied_q += 1

        # 非量子化: delta を直接加算
        for delta_name, hf_suffix in DIRECT_MAPPING.items():
            delta = load_delta(layer_idx, delta_name)
            if delta is None:
                continue
            param_name = prefix + hf_suffix
            if param_name not in params:
                continue

            param = params[param_name]
            delta_t = torch.from_numpy(delta).reshape(param.shape)
            with torch.no_grad():
                param.add_(delta_t.to(param.dtype))
            applied_d += 1

    print(f"  量子化適用: {applied_q}, 直接加算: {applied_d}")
    return applied_q + applied_d


def apply_deltas_no_quantize(model):
    """delta のみ加算（量子化なし）— 比較用。"""
    params = dict(model.named_parameters())
    applied = 0
    all_mapping = {**QUANTIZE_MAPPING, **DIRECT_MAPPING}
    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.layers.{layer_idx}."
        for delta_name, hf_suffix in all_mapping.items():
            delta = load_delta(layer_idx, delta_name)
            if delta is None:
                continue
            param_name = prefix + hf_suffix
            if param_name not in params:
                continue
            param = params[param_name]
            delta_t = torch.from_numpy(delta).reshape(param.shape)
            with torch.no_grad():
                param.add_(delta_t.to(param.dtype))
            applied += 1
    return applied


def generate_response(model, tokenizer, prompt, device, max_tokens=150):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False,  # greedy for reproducibility
        )
    elapsed = time.time() - t0
    n_gen = out.shape[1] - inputs["input_ids"].shape[1]
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    tps = n_gen / elapsed if elapsed > 0 else 0
    return response.strip(), elapsed, tps, n_gen


def run_tests(model, tokenizer, device, label):
    prompts = [
        "What is 2+2? Answer briefly.",
        "日本の首都はどこですか？",
        "Write a Python function to check if a number is prime.",
        "If a train travels 60km in 30 minutes, what is its speed in km/h?",
    ]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for i, p in enumerate(prompts):
        resp, elapsed, tps, n_gen = generate_response(model, tokenizer, p, device)
        print(f"\n[Q{i+1}] {p}")
        print(f"[A]  {resp}")
        print(f"     ({n_gen} tokens, {elapsed:.1f}s, {tps:.1f} tok/s)")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # --- Test 1: ベースモデル（変更なし）---
    print("\n[1/3] ベースモデルロード中...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32, device_map=device)
    model.eval()
    run_tests(model, tokenizer, device, "Base Qwen2.5-1.5B (no delta)")
    del model
    if device == "mps":
        torch.mps.empty_cache()

    # --- Test 2: base + delta（量子化なし）---
    print("\n[2/3] Delta適用モデル（量子化なし）ロード中...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32, device_map=device)
    model.eval()
    n = apply_deltas_no_quantize(model)
    print(f"  {n} パラメータに delta 加算")
    run_tests(model, tokenizer, device, "Base + Delta (NO quantize)")
    del model
    if device == "mps":
        torch.mps.empty_cache()

    # --- Test 3: base + delta + 三値量子化（本来の QAT 推論）---
    print("\n[3/3] QAT モデル（三値量子化）ロード中...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32, device_map=device)
    model.eval()
    apply_deltas_with_quantize(model)
    run_tests(model, tokenizer, device, "Base + Delta + Ternary FakeQuantize (QAT)")

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
