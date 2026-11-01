#!/usr/bin/env bash
# Serve GLM-4.6V-Flash with vLLM on a Linux + CUDA machine (a vast.ai instance, or the worker pod), as
# AI-3D-Worker-Example does for its judge. `python vast.py setup` uploads and starts this script.
#   GLM_API_KEY=secret PORT=8002 bash serve_glm.sh
# vLLM listens on 127.0.0.1 by default: reach it through `python vast.py tunnel` (an SSH port forward).
# The first run builds a dedicated venv (GLM-4.6V needs transformers >= 5) and downloads ~20 GB of weights.
set -euo pipefail

VENV="${VENV:-$HOME/vllm-glm-env}"
VLLM_VERSION="${VLLM_VERSION:-0.23.0}"
VLLM_CUDA_TAG="${VLLM_CUDA_TAG:-cu129}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"
MODEL="${MODEL:-zai-org/GLM-4.6V-Flash}"
REVISION="${REVISION:-411bb4d77144a3f03accbf4b780f5acb8b7cde4e}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"  # a judge call is a few thousand tokens
GPU_UTIL="${GPU_UTIL:-0.9}"
GLM_API_KEY="${GLM_API_KEY:-EMPTY}"
DP="${DP:-$(nvidia-smi -L | wc -l)}"  # one replica per GPU
[ -d /workspace ] && export HF_HOME="${HF_HOME:-/workspace/hf}"

if [ ! -x "$VENV/bin/vllm" ]; then
    echo "[glm] building $VENV: vllm ${VLLM_VERSION}+${VLLM_CUDA_TAG}, transformers>=5"
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
    WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${VLLM_CUDA_TAG}-cp38-abi3-manylinux_2_28_$(uname -m).whl"
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" "vllm @ ${WHEEL}" "transformers>=5.0.0rc0" "fastapi<0.137" \
        --torch-backend "$TORCH_BACKEND"
fi

echo "[glm] serving $MODEL@${REVISION:0:8} on $HOST:$PORT, DP=$DP, max_model_len=$MAX_MODEL_LEN"
exec "$VENV/bin/vllm" serve "$MODEL" \
    --revision "$REVISION" --tokenizer-revision "$REVISION" \
    --served-model-name glm-4.6v-flash \
    --host "$HOST" --port "$PORT" --api-key "$GLM_API_KEY" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 64 \
    --data-parallel-size "$DP" --gpu-memory-utilization "$GPU_UTIL" \
    --generation-config vllm --enable-prefix-caching --enable-chunked-prefill \
    --max-num-batched-tokens 16384 --mm-processor-cache-gb 0
