#!/usr/bin/env python3
"""RunPod A100 Pod作成 — QAT完了/崩壊時にPod自動停止

必要な環境変数: RUNPOD_API_KEY / HUGGING_FACE_HUB_TOKEN / HF_USER / HF_REPO_ML / HF_REPO_TRAIN
(HF_REPO_* は `<owner>/<name>`)。どれかが空なら何もせず終了する。
token と account は pod の環境変数として渡し、pod の起動 command (dockerArgs) にも URL にも入れない
git は環境変数の credential helper で認証し、huggingface_hub (huggingface-cli download) は
HUGGING_FACE_HUB_TOKEN を自分で読む API key は request の header に入れ、process の引数には出さない
`--dry-run` は送る mutation を秘密の値を伏せて表示し、API は呼ばない
"""
import json
import os
import sys
import urllib.error
import urllib.request

REQUIRED = ("RUNPOD_API_KEY", "HUGGING_FACE_HUB_TOKEN", "HF_USER", "HF_REPO_ML", "HF_REPO_TRAIN")
SECRET = ("HUGGING_FACE_HUB_TOKEN",)
env = {k: os.environ.get(k, "") for k in REQUIRED}
missing = [k for k, v in env.items() if not v]
if missing:
    print(f"環境変数を設定してください: {', '.join(missing)}")
    sys.exit(1)

docker_args = (
    "bash -c '"
    # Rust
    "curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y "
    "&& export PATH=/root/.cargo/bin:$PATH "
    # Clone
    "&& cd /workspace "
    "&& git clone https://huggingface.co/$HF_REPO_ML ALICE-ML "
    "&& git clone https://huggingface.co/$HF_REPO_TRAIN alice-train "
    "&& cd alice-train "
    # Build
    "&& cargo build --release --features qat-cuda --bin train-qat-qwen35 "
    # Python deps + model
    "&& pip install -q huggingface_hub transformers datasets fsspec==2024.2.0 "
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

# pod の環境変数: git の credential helper は実行時に $HF_USER / $HUGGING_FACE_HUB_TOKEN を読む
pod_env = {k: env[k] for k in ("HUGGING_FACE_HUB_TOKEN", "HF_USER", "HF_REPO_ML", "HF_REPO_TRAIN")}
pod_env.update({
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "credential.helper",
    "GIT_CONFIG_VALUE_0": '!f() { echo username="$HF_USER"; echo password="$HUGGING_FACE_HUB_TOKEN"; }; f',
})

gpu_types = [
    ("NVIDIA A100 80GB PCIe", "ALICE-QAT-A100-PCIe", 1.19),
    ("NVIDIA A100-SXM4-80GB", "ALICE-QAT-A100-SXM", 1.39),
]


def mutation(gpu_id, name, values):
    env_list = ", ".join(f"{{ key: {json.dumps(k)}, value: {json.dumps(v)} }}" for k, v in values.items())
    return f"""mutation {{ podFindAndDeployOnDemand(input: {{
      name: "{name}",
      gpuTypeId: "{gpu_id}",
      imageName: "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
      gpuCount: 1, volumeInGb: 0, containerDiskInGb: 50,
      minVcpuCount: 4, minMemoryInGb: 64,
      dockerArgs: {json.dumps(docker_args)},
      env: [{env_list}]
    }}) {{ id name }} }}"""


if "--dry-run" in sys.argv[1:]:
    gpu_id, name, _ = gpu_types[0]
    print(mutation(gpu_id, name, {k: ("***" if k in SECRET else v) for k, v in pod_env.items()}))
    sys.exit(0)

for gpu_id, name, price in gpu_types:
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": mutation(gpu_id, name, pod_env)}).encode(),
        headers={
            "Authorization": f"Bearer {env['RUNPOD_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"✗ {gpu_id}: HTTP {e.code} {e.reason}")
        continue

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
