#!/usr/bin/env bash
set -euo pipefail

trainer_root="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$trainer_root/../.." && pwd)"
gpu_id="${GPU_ID:-0}"
port="${PORT:-9531}"
container_name="${CONTAINER_NAME:-repairdistill-qwen35-9b}"
image="${IMAGE:-vllm/vllm-openai:cu130-nightly}"
model_path="${RCS_OPSD_MODEL_PATH:-$project_root/models/Qwen3.5-9B}"
chat_template="$project_root/templates/chat_template_codex.jinja"

docker run -d \
  --name "$container_name" \
  --runtime=nvidia \
  --gpus "device=$gpu_id" \
  --network=host \
  --ipc=host \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  -v "$model_path:/model:ro" \
  -v "$chat_template:/chat_template.jinja:ro" \
  -v "$trainer_root:$trainer_root:ro" \
  "$image" /model \
  --served-model-name qwen3.5-9b \
  --host 0.0.0.0 --port "$port" \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 8 \
  --language-model-only --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --chat-template /chat_template.jinja --enable-lora --max-lora-rank 64
