#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/captioner/label_caption_tar_jsonl.py"

# All paths and tuning knobs may be overridden with environment variables.
RUN_DIR="${RUN_DIR:-/home/ma-user/work/others/w00911759/dataset_0827/purchase_panguml_with_meta_info/purchase_caption/0812_data}"
INPUT_DIR="${INPUT_DIR:-$RUN_DIR/annotation_split_dedup_gemini}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt-mm-pro/s3-g-szc-asset-aifm-mm-exp/j00939224/data/0812_data/images}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/annotation_captioned_gemini}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs_gemini}"
LAUNCH_LOG="${LAUNCH_LOG:-$RUN_DIR/run_0812_data.launcher.log}"
VLM_CONFIG="${VLM_CONFIG:-$SCRIPT_DIR/config/vlm_api.yaml}"
START_ANNOTATION="${START_ANNOTATION:-3800}"
DATA_WORKERS="${DATA_WORKERS:-48}"
API_WORKERS="${API_WORKERS:-200}"
TIMEOUT="${TIMEOUT:-60}"
MAX_RETRIES="${MAX_RETRIES:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-/usr/local/python3.11.15/lib/python3.11/site-packages}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--input-jsonl PATH]

Without arguments, process JSONL files whose numeric suffix is at least
START_ANNOTATION (default: $START_ANNOTATION). With --input-jsonl, process only
that file. A relative PATH is resolved under INPUT_DIR.

Configuration is supplied through environment variables; see README.md.
EOF
}

SINGLE_JSONL=""
while (( $# > 0 )); do
  case "$1" in
    --input-jsonl)
      (( $# >= 2 )) || { echo "error: --input-jsonl requires a path" >&2; exit 2; }
      SINGLE_JSONL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$START_ANNOTATION" =~ ^[0-9]+$ ]] || { echo "error: START_ANNOTATION must be a non-negative integer" >&2; exit 2; }
[[ -f "$PYTHON_SCRIPT" ]] || { echo "error: missing Python entry point: $PYTHON_SCRIPT" >&2; exit 2; }
[[ -d "$INPUT_DIR" ]] || { echo "error: input directory does not exist: $INPUT_DIR" >&2; exit 2; }
[[ -d "$IMAGE_ROOT" ]] || { echo "error: image root does not exist: $IMAGE_ROOT" >&2; exit 2; }
[[ -f "$VLM_CONFIG" ]] || { echo "error: missing VLM config: $VLM_CONFIG (copy config/vlm_api.example.yaml first)" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "error: Python not found: $PYTHON_BIN" >&2; exit 2; }

if [[ -n "$SINGLE_JSONL" && ! -f "$SINGLE_JSONL" ]]; then
  SINGLE_JSONL="$INPUT_DIR/$SINGLE_JSONL"
fi
if [[ -n "$SINGLE_JSONL" ]]; then
  [[ -f "$SINGLE_JSONL" ]] || { echo "error: JSONL file does not exist: $SINGLE_JSONL" >&2; exit 2; }
  [[ "$SINGLE_JSONL" == *.jsonl ]] || { echo "error: --input-jsonl must point to a .jsonl file" >&2; exit 2; }
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$(dirname "$LAUNCH_LOG")"

INPUT_ARGS=(--input-dir "$INPUT_DIR")
if [[ -n "$SINGLE_JSONL" ]]; then
  INPUT_ARGS+=(--input-jsonl "$SINGLE_JSONL")
else
  INPUT_ARGS+=(--start-annotation "$START_ANNOTATION")
fi

nohup setsid bash -c '
  "$@"
  status=$?
  printf "[%s] PROCESS_EXIT status=%s\n" "$(date --iso-8601=seconds)" "$status"
  exit "$status"
' bash env PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -S "$PYTHON_SCRIPT" \
  "${INPUT_ARGS[@]}" \
  --image-root "$IMAGE_ROOT" \
  --tar-name part_000 \
  --output-dir "$OUTPUT_DIR" \
  --log-dir "$LOG_DIR" \
  --config "$VLM_CONFIG" \
  --data-workers "$DATA_WORKERS" \
  --api-workers "$API_WORKERS" \
  --timeout "$TIMEOUT" \
  --max-retries "$MAX_RETRIES" \
  --memory-log-interval 100 \
  --resume \
  > "$LAUNCH_LOG" 2>&1 < /dev/null &

echo "Gemini caption job started in background, pid=$!"
[[ -z "$SINGLE_JSONL" ]] || echo "single JSONL: $SINGLE_JSONL"
echo "launcher log: $LAUNCH_LOG"
echo "per-JSONL logs: $LOG_DIR"
