#!/usr/bin/env python3
"""RunPod A100 Pod 作成 — ALICE-Train Re-QAT 用

必要な環境変数: RUNPOD_API_KEY / HUGGING_FACE_HUB_TOKEN / HF_USER / HF_REPO_ML / HF_REPO_TRAIN
(HF_REPO_* は `<owner>/<name>`)。どれかが空なら何もせず終了する。
token と account は pod の環境変数として渡し、pod の起動 command (dockerArgs) にも URL にも入れない
git は環境変数の credential helper で認証し、huggingface_hub は HUGGING_FACE_HUB_TOKEN を自分で読む
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
    "apt-get update -qq && apt-get install -y -qq git > /dev/null && "
    "curl --proto=https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
    "export PATH=/root/.cargo/bin:$PATH && "
    "cd /workspace && "
    "git clone https://huggingface.co/$HF_REPO_ML ALICE-ML && "
    "git clone https://huggingface.co/$HF_REPO_TRAIN ALICE-Train && "
    "cd ALICE-Train && "
    "export CUDA_PATH=/usr/local/cuda && "
    "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && "
    "cargo build --release --features qat-cuda --bin train-qat-qwen35 2>&1 | tail -5 && "
    "echo SETUP_COMPLETE && "
    "sleep infinity'"
)

# pod の環境変数: git の credential helper は実行時に $HF_USER / $HUGGING_FACE_HUB_TOKEN を読む
pod_env = {k: env[k] for k in ("HUGGING_FACE_HUB_TOKEN", "HF_USER", "HF_REPO_ML", "HF_REPO_TRAIN")}
pod_env.update({
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "credential.helper",
    "GIT_CONFIG_VALUE_0": '!f() { echo username="$HF_USER"; echo password="$HUGGING_FACE_HUB_TOKEN"; }; f',
})


def mutation(values):
    env_list = ", ".join(f"{{ key: {json.dumps(k)}, value: {json.dumps(v)} }}" for k, v in values.items())
    return f"""
mutation {{
  podFindAndDeployOnDemand(input: {{
    name: "alice-qat-rerun"
    gpuTypeId: "NVIDIA A100 80GB PCIe"
    imageName: "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
    containerDiskInGb: 50
    volumeInGb: 0
    minMemoryInGb: 32
    gpuCount: 1
    cloudType: COMMUNITY
    dockerArgs: {json.dumps(docker_args)}
    env: [{env_list}]
  }}) {{
    id
    desiredStatus
    imageName
    gpuCount
    machine {{ gpuDisplayName }}
  }}
}}
"""


if "--dry-run" in sys.argv[1:]:
    print(mutation({k: ("***" if k in SECRET else v) for k, v in pod_env.items()}))
    sys.exit(0)

req = urllib.request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps({"query": mutation(pod_env)}).encode(),
    headers={
        "Authorization": f"Bearer {env['RUNPOD_API_KEY']}",
        "Content-Type": "application/json",
    },
)

try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(json.dumps(result, indent=2))
        if "data" in result and result["data"].get("podFindAndDeployOnDemand"):
            pod = result["data"]["podFindAndDeployOnDemand"]
            print(f"\nPod ID: {pod['id']}")
            print(f"SSH: runpodctl ssh info {pod['id']}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.reason}")
    print(e.read().decode())
