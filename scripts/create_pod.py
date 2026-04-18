#!/usr/bin/env python3
"""RunPod A100 Pod 作成 — ALICE-Train Re-QAT 用"""
import json
import os
import urllib.request

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not API_KEY:
    print("RUNPOD_API_KEY not set")
    exit(1)

docker_args = (
    "bash -c '"
    "apt-get update -qq && apt-get install -y -qq git > /dev/null && "
    "curl --proto=https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && "
    "export PATH=/root/.cargo/bin:$PATH && "
    "cd /workspace && "
    f"git clone https://sakamoro:{HF_TOKEN}@huggingface.co/sakamoro/alice-ml ALICE-ML && "
    f"git clone https://sakamoro:{HF_TOKEN}@huggingface.co/sakamoro/alice-train ALICE-Train && "
    "cd ALICE-Train && "
    "export CUDA_PATH=/usr/local/cuda && "
    "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && "
    "cargo build --release --features qat-cuda --bin train-qat-qwen35 2>&1 | tail -5 && "
    "echo SETUP_COMPLETE && "
    "sleep infinity'"
)

mutation = """
mutation {
  podFindAndDeployOnDemand(input: {
    name: "alice-qat-rerun"
    gpuTypeId: "NVIDIA A100 80GB PCIe"
    imageName: "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
    containerDiskInGb: 50
    volumeInGb: 0
    minMemoryInGb: 32
    gpuCount: 1
    cloudType: COMMUNITY
    dockerArgs: "%s"
  }) {
    id
    desiredStatus
    imageName
    gpuCount
    machine { gpuDisplayName }
  }
}
""" % docker_args.replace('"', '\\"')

req = urllib.request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps({"query": mutation}).encode(),
    headers={
        "Authorization": f"Bearer {API_KEY}",
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
