#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/ma-user/work/others/w00911759/models/qwen-3.8-27B}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.11 || command -v python3 || true)}"
VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"
DEVICE_IDS="${NPU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
# 16 卡部署：每个模型副本使用 2 卡张量并行，共启动 8 个数据并行副本。
# 如需临时调整，仍可在启动时覆盖，例如：TP_SIZE=4 DP_SIZE=4 bash start.sh
export TP_SIZE="${TP_SIZE:-2}"
export DP_SIZE="${DP_SIZE:-8}"
unset NPU_VISIBLE_DEVICES
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$DEVICE_IDS}"
export ASCEND_RT_VISIBLE_DEVICES
if [[ -z "${ASCEND_VISIBLE_DEVICES+x}" ]]; then
  export ASCEND_VISIBLE_DEVICES="$DEVICE_IDS"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM="false"
export MODEL_PATH

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "找不到 python3.11/python3；请设置 PYTHON_BIN" >&2
  exit 2
fi
if [[ -z "${VLLM_BIN}" ]]; then
  echo "找不到 vllm；请进入 vLLM-Ascend 环境或设置 VLLM_BIN" >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "模型不存在：${MODEL_PATH}" >&2
  exit 2
fi

# Reuse the bundled vLLM-Ascend custom operators when this project is run from
# the original ModelArts checkout.  A system installation may omit this path.
CUSTOM_OPS_ROOT="${VLLM_CUSTOM_OPS_ROOT:-/home/ma-user/work/others/w00911759/codes/recap_v410_roma_split_b_20260816_run2/runtime/cann_opp_install/vendors/custom_transformer}"
if [[ -f "${CUSTOM_OPS_ROOT}/bin/set_env.bash" ]]; then
  export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPS_ROOT}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
  export LD_LIBRARY_PATH="${CUSTOM_OPS_ROOT}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "[preflight] using custom ops: ${CUSTOM_OPS_ROOT}"
else
  echo "[preflight] custom ops not found; continuing with system vLLM-Ascend"
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import json
import os
import sys
model = os.environ["MODEL_PATH"]
config = json.load(open(os.path.join(model, "config.json"), encoding="utf-8"))
print("[preflight] model_type=", config.get("model_type"), "architecture=", config.get("architectures"))
for name in ("torch", "torch_npu", "transformers", "vllm"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing Python package: {name}; install/use the vLLM-Ascend runtime")
print("[preflight] Python packages: torch/torch_npu/transformers/vllm found")
PY

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --trust-remote-code \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --data-parallel-size "${DP_SIZE}" \
  --distributed-executor-backend "${EXECUTOR_BACKEND:-mp}" \
  --max-model-len "${MAX_MODEL_LEN:-32000}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching
