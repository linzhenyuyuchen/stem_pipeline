#!/usr/bin/env bash
# 后台启动五种推理模式 QA 处理。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="/home/ma-user/work/others/l00788561/data/web_retrival/qa.json"
export QA_API_KEY="sk-tCWORE9AVK9FCmeoni5sDuVeE7uEKDxpAUZkjLhliTxW7Nmi"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "config 不存在: $CONFIG_PATH" >&2
    exit 2
fi

# 多个地址支持逗号、空格或分号分隔；QA_WORKERS 是每个地址的并发数，
# 总并发 = 地址数量 * QA_WORKERS。
export QA_BASE_URL="${QA_BASE_URL:-http://127.0.0.1:8891/v1,http://127.0.0.1:8892/v1,http://127.0.0.1:8893/v1,http://127.0.0.1:8894/v1,http://127.0.0.1:8895/v1}"
export QA_MODEL="${QA_MODEL:-gemini-3.8-flash}"
export QA_WORKERS="${QA_WORKERS:-128}"
export QA_RETRIES="${QA_RETRIES:-1}"
export QA_TIMEOUT="${QA_TIMEOUT:-600}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

CONFIG_PATH="$(realpath "$CONFIG_PATH")"
CONFIG_NAME="$(basename "${CONFIG_PATH%.*}")"
RUN_NAME="qa_5pattern_${CONFIG_NAME}"
LOG_DIR="${QA_LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${QA_LOG_FILE:-${LOG_DIR}/${RUN_NAME}_$(date '+%Y%m%d_%H%M%S').log}"
PID_FILE="${QA_PID_FILE:-${LOG_DIR}/${RUN_NAME}.pid}"
mkdir -p "$LOG_DIR"

if [[ -s "$PID_FILE" ]]; then
    old_pid="$(<"$PID_FILE")"
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "$RUN_NAME 已在运行: pid=$old_pid" >&2
        exit 1
    fi
fi

nohup setsid python3 -u "$SCRIPT_DIR/solve_qa_5pattern.py" \
    --config "$CONFIG_PATH" \
    --resume \
    --base-url "$QA_BASE_URL" \
    --model "$QA_MODEL" \
    --workers "$QA_WORKERS" \
    --retries "$QA_RETRIES" \
    --timeout "$QA_TIMEOUT" \
    "$@" >"$LOG_FILE" 2>&1 </dev/null &

worker_pid=$!
echo "$worker_pid" >"$PID_FILE"
disown "$worker_pid" 2>/dev/null || true

echo "$RUN_NAME 已后台启动: pid=$worker_pid"
echo "日志: $LOG_FILE"
echo "PID 文件: $PID_FILE"
