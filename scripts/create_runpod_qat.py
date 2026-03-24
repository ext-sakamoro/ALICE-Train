#!/usr/bin/env python3
"""RunPod A100 Pod作成 — QAT完了/崩壊時にPod自動停止"""
import json
import subprocess
import sys

import os
RUNPOD_KEY = os.environ.get("RUNPOD_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not RUNPOD_KEY or not HF_TOKEN:
    print("環境変数 RUNPOD_API_KEY, HF_TOKEN を設定してください")
    sys.exit(1)

docker_args = (
    "bash -c '"
    # Rust
    "curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y "
    "&& export PATH=/root/.cargo/bin:$PATH "
    # Clone
    "&& cd /workspace "
    f"&& git clone https://sakamoro:{HF_TOKEN}@huggingface.co/sakamoro/alice-ml ALICE-ML "
    f"&& git clone https://sakamoro:{HF_TOKEN}@huggingface.co/sakamoro/alice-train "
    "&& cd alice-train "
    # Build
    "&& cargo build --release --features qat-cuda --bin train-qat-qwen35 "
    # Python deps + model
    "&& pip install -q huggingface_hub transformers datasets fsspec==2024.2.0 "
    f"&& huggingface-cli login --token {HF_TOKEN} "
    "&& huggingface-cli download Qwen/Qwen3.5-9B --local-dir models/Qwen--Qwen3.5-9B "
    # Data
    "&& python3 scripts/prepare_real_data.py --model_path models/Qwen--Qwen3.5-9B --output data/qwen35/train.bin --max_tokens 2000000 "
    "&& python3 scripts/prepare_real_data.py --model_path models/Qwen--Qwen3.5-9B --output data/qwen35/eval.bin --max_tokens 100000 --split validation "
    # QAT (フォアグラウンド — 終了でPod停止)
    "&& bash scripts/runpod_qat.sh "
    # ↑ runpod_qat.sh 内で学習完了/崩壊時にPod停止APIを叩く
    # 万一 runpod_qat.sh が失敗した場合のフォールバック
    "; echo QAT_EXITED; sleep 300'"  # 5分猶予後にPodタイムアウト
)

gpu_types = [
    ("NVIDIA A100 80GB PCIe", "ALICE-QAT-A100-PCIe", 1.19),
    ("NVIDIA A100-SXM4-80GB", "ALICE-QAT-A100-SXM", 1.39),
]

for gpu_id, name, price in gpu_types:
    query = f"""mutation {{ podFindAndDeployOnDemand(input: {{
      name: "{name}",
      gpuTypeId: "{gpu_id}",
      imageName: "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
      gpuCount: 1, volumeInGb: 0, containerDiskInGb: 50,
      minVcpuCount: 4, minMemoryInGb: 64,
      dockerArgs: ARGS_PLACEHOLDER
    }}) {{ id name }} }}"""

    query = query.replace("ARGS_PLACEHOLDER", json.dumps(docker_args))
    payload = json.dumps({"query": query})

    result = subprocess.run(
        ["curl", "-s", "https://api.runpod.io/graphql",
         "-H", f"Authorization: Bearer {RUNPOD_KEY}",
         "-H", "Content-Type: application/json",
         "-d", payload],
        capture_output=True, text=True,
    )
    data = json.loads(result.stdout)

    if data.get("data", {}).get("podFindAndDeployOnDemand"):
        pod = data["data"]["podFindAndDeployOnDemand"]
        print(f"✓ {gpu_id} (${price}/h) → Pod ID: {pod['id']}")
        print(f"  学習完了/崩壊時に自動停止")
        sys.exit(0)
    else:
        err = data.get("errors", [{}])[0].get("message", "unknown")
        print(f"✗ {gpu_id}: {err}")

print("全GPU空きなし")
sys.exit(1)
