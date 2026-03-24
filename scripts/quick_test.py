#!/usr/bin/env python3
"""ALICE QAT クイックテスト — 3モード比較を1ロードで実行"""

import struct, os, time, torch, numpy as np, copy
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_DIR = os.path.expanduser("~/ALICE-Train/checkpoints/qat_qwen_1.5b")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_LAYERS = 28

QUANTIZE_KEYS = {
    "q_proj": "self_attn.q_proj.weight", "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight", "o_proj": "self_attn.o_proj.weight",
    "gate_proj": "mlp.gate_proj.weight", "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}
DIRECT_KEYS = {
    "attn_norm": "input_layernorm.weight", "ffn_norm": "post_attention_layernorm.weight",
    "q_bias": "self_attn.q_proj.bias", "k_bias": "self_attn.k_proj.bias",
    "v_bias": "self_attn.v_proj.bias",
}

def load_delta(layer_idx, name):
    path = os.path.join(CHECKPOINT_DIR, f"delta_layer{layer_idx}_{name}.bin")
    if not os.path.exists(path): return None
    data = open(path, "rb").read()
    return np.array(struct.unpack(f"<{len(data)//4}f", data), dtype=np.float32)

def ternary_fq(w):
    gamma = w.abs().mean().clamp(min=1e-10)
    return (w / gamma / 0.98).round().clamp(-1, 1) * gamma

def gen(model, tokenizer, prompt):
    msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cpu")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    elapsed = time.time() - t0
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return resp.strip(), elapsed

def main():
    print("Loading tokenizer + base model (CPU, float32)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32, device_map="cpu")
    model.eval()
    print("Loaded.")

    prompts = [
        "What is 2+2?",
        "日本の首都は？",
    ]

    # --- Test 1: Base ---
    print("\n" + "=" * 50)
    print("TEST 1: Base Qwen2.5-1.5B (no delta)")
    print("=" * 50)
    # Save original state_dict for reset
    original_sd = {k: v.clone() for k, v in model.state_dict().items()}

    for p in prompts:
        resp, t = gen(model, tokenizer, p)
        print(f"  Q: {p}")
        print(f"  A: {resp}  ({t:.1f}s)")

    # --- Test 2: Base + Delta (no quantize) ---
    print("\n" + "=" * 50)
    print("TEST 2: Base + Delta (no quantize)")
    print("=" * 50)
    params = dict(model.named_parameters())
    cnt = 0
    for li in range(NUM_LAYERS):
        pfx = f"model.layers.{li}."
        for dn, hs in {**QUANTIZE_KEYS, **DIRECT_KEYS}.items():
            d = load_delta(li, dn)
            if d is None: continue
            pn = pfx + hs
            if pn not in params: continue
            dt = torch.from_numpy(d).reshape(params[pn].shape)
            with torch.no_grad():
                params[pn].add_(dt)
            cnt += 1
    print(f"  Applied {cnt} deltas")

    for p in prompts:
        resp, t = gen(model, tokenizer, p)
        print(f"  Q: {p}")
        print(f"  A: {resp}  ({t:.1f}s)")

    # --- Reset to base ---
    model.load_state_dict(original_sd)

    # --- Test 3: Base + Delta + Ternary FakeQuantize ---
    print("\n" + "=" * 50)
    print("TEST 3: Base + Delta + Ternary FakeQuantize (QAT)")
    print("=" * 50)
    params = dict(model.named_parameters())
    cnt_q = 0
    cnt_d = 0
    for li in range(NUM_LAYERS):
        pfx = f"model.layers.{li}."
        for dn, hs in QUANTIZE_KEYS.items():
            d = load_delta(li, dn)
            if d is None: continue
            pn = pfx + hs
            if pn not in params: continue
            dt = torch.from_numpy(d).reshape(params[pn].shape)
            with torch.no_grad():
                merged = params[pn].data + dt
                params[pn].data.copy_(ternary_fq(merged))
            cnt_q += 1
        for dn, hs in DIRECT_KEYS.items():
            d = load_delta(li, dn)
            if d is None: continue
            pn = pfx + hs
            if pn not in params: continue
            dt = torch.from_numpy(d).reshape(params[pn].shape)
            with torch.no_grad():
                params[pn].add_(dt)
            cnt_d += 1
    print(f"  Quantized: {cnt_q}, Direct: {cnt_d}")

    for p in prompts:
        resp, t = gen(model, tokenizer, p)
        print(f"  Q: {p}")
        print(f"  A: {resp}  ({t:.1f}s)")

    print("\n" + "=" * 50)
    print("Done!")

if __name__ == "__main__":
    main()
