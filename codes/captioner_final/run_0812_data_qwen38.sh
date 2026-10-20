#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$CODE_DIR/captioner/label_caption_tar_jsonl.py"
RUN_DIR="${RUN_DIR:-/home/ma-user/work/others/w00911759/dataset_0827/purchase_panguml_with_meta_info/purchase_caption/0812_data}"
INPUT_DIR="${INPUT_DIR:-$RUN_DIR/annotation_split_dedup_qwen}"
IMAGE_ROOT="${IMAGE_ROOT:-/mnt-mm-pro/s3-g-szc-asset-aifm-mm-exp/j00939224/data/0812_data/images}"
START_ANNOTATION="${START_ANNOTATION:-0}"
# This job is provisioned with two VJ nodes. The value is used only when a
# rank table is not mounted; an available rank table remains authoritative.
DEFAULT_MACHINE_COUNT="${DEFAULT_MACHINE_COUNT:-2}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-30}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
HEARTBEAT_STALE_SECONDS="${HEARTBEAT_STALE_SECONDS:-120}"
DATA_WORKERS="${DATA_WORKERS:-48}"
API_WORKERS="${API_WORKERS:-128}"
TIMEOUT="${TIMEOUT:-600}"
MAX_RETRIES="${MAX_RETRIES:-0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-/usr/local/python3.11.15/lib/python3.11/site-packages:/usr/local/lib/python3.11/site-packages}"
export PYTHON_BIN PYTHONPATH_VALUE
MACHINE_INDEX="${MACHINE_INDEX:-}"
MACHINE_COUNT="${MACHINE_COUNT:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--input-jsonl PATH] [--machine-index N] [--machine-count N]

Each machine processes only the JSONL shards assigned to its zero-based index.
The index is detected from MA_VJ_NAME (the trailing -vj-N), NODE_RANK/GROUP_RANK,
or a rank table. The machine count is read from the rank table, or defaults to
$DEFAULT_MACHINE_COUNT for this two-node job when no rank table is mounted.
MACHINE_INDEX/MACHINE_COUNT or the matching CLI options may override detection.

Without --input-jsonl, process assigned JSONL files from annotation $START_ANNOTATION onward.
With --input-jsonl, only the machine owning that JSONL processes it.
PATH may be absolute or a filename relative to $INPUT_DIR.
EOF
}

SINGLE_JSONL=""
while (( $# > 0 )); do
  case "$1" in
    --input-jsonl)
      if (( $# < 2 )); then
        echo "error: --input-jsonl requires a path" >&2
        usage >&2
        exit 2
      fi
      SINGLE_JSONL="$2"
      shift 2
      ;;
    --machine-index)
      if (( $# < 2 )); then
        echo "error: --machine-index requires a zero-based integer" >&2
        usage >&2
        exit 2
      fi
      MACHINE_INDEX="$2"
      shift 2
      ;;
    --machine-count)
      if (( $# < 2 )); then
        echo "error: --machine-count requires a positive integer" >&2
        usage >&2
        exit 2
      fi
      MACHINE_COUNT="$2"
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
command -v "$PYTHON_BIN" >/dev/null || { echo "error: Python not found: $PYTHON_BIN" >&2; exit 2; }

if [[ -z "$MACHINE_INDEX" || -z "$MACHINE_COUNT" ]]; then
  if ! read -r DETECTED_MACHINE_INDEX DETECTED_MACHINE_COUNT < <(
    DEFAULT_MACHINE_COUNT="$DEFAULT_MACHINE_COUNT" REQUESTED_MACHINE_COUNT="$MACHINE_COUNT" "$PYTHON_BIN" - <<'PY'
import json
import os
import re
import socket

default_machine_count = int(os.environ["DEFAULT_MACHINE_COUNT"])
requested_machine_count = os.environ.get("REQUESTED_MACHINE_COUNT", "")
ma_vj_name = os.environ.get("MA_VJ_NAME", "")
job_rank_table = "/user/config/jobstart_hccl.json"
configured_rank_table = os.environ.get("RANK_TABLE_FILE", "")
rank_table = configured_rank_table
# MA_VJ_NAME identifies a managed multi-node job. Its jobstart table is the
# authoritative topology even when RANK_TABLE_FILE still points at a notebook
# nbstart table inherited from the surrounding environment.
if configured_rank_table and os.path.basename(configured_rank_table) != "nbstart_hccl.json" and os.path.isfile(configured_rank_table):
    rank_table = configured_rank_table
elif ma_vj_name and os.path.isfile(job_rank_table):
    rank_table = job_rank_table
elif not rank_table or not os.path.isfile(rank_table):
    rank_table = next(
        (path for path in (job_rank_table, "/user/config/nbstart_hccl.json") if os.path.isfile(path)),
        "",
    )

config = {}
servers = []
machine_count = int(requested_machine_count) if requested_machine_count else default_machine_count
if rank_table and os.path.isfile(rank_table):
    with open(rank_table, encoding="utf-8") as handle:
        config = json.load(handle)
    super_pods = config.get("super_pod_list") or []
    super_pod_ids = {str(pod.get("super_pod_id", "")) for pod in super_pods}
    # Ignore stale notebook/job topology belonging to another MA job.
    if ma_vj_name and (not super_pod_ids or ma_vj_name not in super_pod_ids):
        config = {}
    else:
        servers = config.get("server_list") or []
        table_machine_count = int(config.get("server_count") or len(servers))
        if not servers or table_machine_count != len(servers):
            raise SystemExit("rank table has an invalid server_list/server_count")
        if not requested_machine_count:
            machine_count = len(super_pods) or table_machine_count

machine_index = None
match = re.search(r"-vj-(\d+)$", ma_vj_name)
if match:
    vj_index = int(match.group(1))
    if 0 <= vj_index < machine_count:
        machine_index = vj_index
    else:
        raise SystemExit(
            f"MA_VJ_NAME index {vj_index} is outside machine count {machine_count}; "
            "set MACHINE_COUNT explicitly when the rank table is unavailable"
        )

if machine_index is None:
    node_rank_text = os.environ.get("NODE_RANK") or os.environ.get("GROUP_RANK")
    if node_rank_text is not None:
        node_rank = int(node_rank_text)
        if 0 <= node_rank < machine_count:
            machine_index = node_rank

candidates = {
    value
    for value in (
        os.environ.get("MY_HOST_IP"),
        os.environ.get("HOST_IP"),
        os.environ.get("POD_IP"),
    )
    if value
}
try:
    candidates.add(socket.gethostbyname(socket.gethostname()))
except OSError:
    pass

if machine_index is None:
    machine_index = next(
        (index for index, server in enumerate(servers) if str(server.get("server_id")) in candidates),
        None,
    )

rank_text = os.environ.get("RANK_ID")
if machine_index is None and rank_text is not None:
    rank_id = int(rank_text)
    for index, server in enumerate(servers):
        device_ranks = {
            int(device["rank_id"])
            for device in server.get("device", [])
            if "rank_id" in device
        }
        if rank_id in device_ranks:
            machine_index = index
            break

if machine_index is None and machine_count == 1:
    machine_index = 0
if machine_index is None:
    raise SystemExit("cannot match this machine to rank table server_list")

print(machine_index, machine_count)
PY
  ); then
    echo "error: cannot detect machine index/count; set MACHINE_INDEX and MACHINE_COUNT explicitly" >&2
    exit 2
  fi
  MACHINE_INDEX="${MACHINE_INDEX:-$DETECTED_MACHINE_INDEX}"
  MACHINE_COUNT="${MACHINE_COUNT:-$DETECTED_MACHINE_COUNT}"
fi

if ! [[ "$MACHINE_INDEX" =~ ^[0-9]+$ && "$MACHINE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: machine index/count must be integers: index=$MACHINE_INDEX count=$MACHINE_COUNT" >&2
  exit 2
fi
if (( MACHINE_INDEX >= MACHINE_COUNT )); then
  echo "error: machine index must be smaller than machine count: index=$MACHINE_INDEX count=$MACHINE_COUNT" >&2
  exit 2
fi

if [[ -n "$SINGLE_JSONL" && ! -f "$SINGLE_JSONL" ]]; then
  SINGLE_JSONL="$INPUT_DIR/$SINGLE_JSONL"
fi
if [[ -n "$SINGLE_JSONL" ]]; then
  if [[ ! -f "$SINGLE_JSONL" ]]; then
    echo "error: JSONL file does not exist: $SINGLE_JSONL" >&2
    exit 2
  fi
  if [[ "$SINGLE_JSONL" != *.jsonl ]]; then
    echo "error: --input-jsonl must point to a .jsonl file: $SINGLE_JSONL" >&2
    exit 2
  fi
fi

MACHINE_TAG=$(printf "machine_%03d" "$MACHINE_INDEX")
# Output is deliberately independent of MACHINE_INDEX. This makes --resume
# stable when a restarted task uses a different number of nodes.
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/annotation_captioned_qwen38}"
LOG_ROOT="${LOG_ROOT:-$RUN_DIR/logs_qwen38}"
LOG_DIR="$LOG_ROOT/$MACHINE_TAG"
LAUNCH_LOG="$LOG_ROOT/run_0812_data_qwen38.${MACHINE_TAG}.launcher.log"
STATUS_DIR="$LOG_ROOT/node_status"
TOTAL_LOG="$LOG_ROOT/all_nodes_status.log"
MONITOR_SCRIPT="$CODE_DIR/monitor_caption_nodes.sh"
STATUS_FILE="$STATUS_DIR/$MACHINE_TAG.status"

if ! [[ "$START_ANNOTATION" =~ ^[0-9]+$ ]]; then
  echo "error: START_ANNOTATION must be a non-negative integer: $START_ANNOTATION" >&2
  exit 2
fi
[[ -f "$PYTHON_SCRIPT" ]] || { echo "error: missing Python entry point: $PYTHON_SCRIPT" >&2; exit 2; }
[[ -x "$MONITOR_SCRIPT" ]] || { echo "error: missing monitor script: $MONITOR_SCRIPT" >&2; exit 2; }
[[ -d "$INPUT_DIR" ]] || { echo "error: input directory does not exist: $INPUT_DIR" >&2; exit 2; }
[[ -d "$IMAGE_ROOT" ]] || { echo "error: image root does not exist: $IMAGE_ROOT" >&2; exit 2; }

JOB_ID="${MA_VJ_NAME:-manual-job}"
if [[ "$JOB_ID" =~ ^(.+)-vj-[0-9]+$ ]]; then
  JOB_ID="${BASH_REMATCH[1]}"
fi

write_parent_status() {
  local state=$1 exit_code=${2:--} temp_file
  temp_file="$STATUS_FILE.tmp.$BASHPID"
  {
    printf 'job_id=%s\n' "$JOB_ID"
    printf 'machine_index=%s\n' "$MACHINE_INDEX"
    printf 'machine_count=%s\n' "$MACHINE_COUNT"
    printf 'state=%s\n' "$state"
    printf 'heartbeat_epoch=%s\n' "$(date +%s)"
    printf 'pid=%s\n' "$BASHPID"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'current_jsonl=%s\n' '-'
  } > "$temp_file"
  mv -f "$temp_file" "$STATUS_FILE"
}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$STATUS_DIR"
config_temp="$STATUS_DIR/expected_machine_count.tmp.$BASHPID"
printf '%s\n' "$MACHINE_COUNT" > "$config_temp"
mv -f "$config_temp" "$STATUS_DIR/expected_machine_count"
config_temp="$STATUS_DIR/current_job_id.tmp.$BASHPID"
printf '%s\n' "$JOB_ID" > "$config_temp"
mv -f "$config_temp" "$STATUS_DIR/current_job_id"
write_parent_status starting

# All nodes start this command, but flock leaves exactly one shared monitor.
# If its container disappears, another node can acquire the lock on restart.
nohup setsid bash "$MONITOR_SCRIPT" "$STATUS_DIR" "$TOTAL_LOG" \
  "$MONITOR_INTERVAL" "$HEARTBEAT_STALE_SECONDS" > /dev/null 2>&1 < /dev/null &

annotation_number() {
  local filename stem suffix
  filename="${1##*/}"
  stem="${filename%.jsonl}"
  suffix="${stem##*_}"
  [[ "$suffix" =~ ^[0-9]+$ ]] || return 1
  printf '%d\n' "$((10#$suffix))"
}

SHARD_FILES=()
if [[ -n "$SINGLE_JSONL" ]]; then
  if ! number=$(annotation_number "$SINGLE_JSONL"); then
    echo "error: cannot read annotation number from: $SINGLE_JSONL" >&2
    exit 2
  fi
  owner=$(( (number - START_ANNOTATION) % MACHINE_COUNT ))
  (( owner < 0 )) && owner=$((owner + MACHINE_COUNT))
  if (( owner == MACHINE_INDEX )); then
    SHARD_FILES+=("$SINGLE_JSONL")
  fi
else
  shopt -s nullglob
  for input_jsonl in "$INPUT_DIR"/*.jsonl; do
    if ! number=$(annotation_number "$input_jsonl"); then
      echo "warning: skip JSONL without numeric suffix: $input_jsonl" >&2
      continue
    fi
    if (( number >= START_ANNOTATION && (number - START_ANNOTATION) % MACHINE_COUNT == MACHINE_INDEX )); then
      SHARD_FILES+=("$input_jsonl")
    fi
  done
  shopt -u nullglob
fi

if (( ${#SHARD_FILES[@]} == 0 )); then
  write_parent_status completed 0
  echo "machine $MACHINE_INDEX/$MACHINE_COUNT has no assigned JSONL; exiting"
  exit 0
fi

last_index=$((${#SHARD_FILES[@]} - 1))
echo "machine index: $MACHINE_INDEX/$MACHINE_COUNT"
echo "assigned JSONLs: ${#SHARD_FILES[@]} (first=${SHARD_FILES[0]##*/}, last=${SHARD_FILES[$last_index]##*/})"
echo "output dir: $OUTPUT_DIR"

nohup setsid bash -c '
  python_script=$1
  input_dir=$2
  image_root=$3
  output_dir=$4
  log_dir=$5
  status_file=$6
  job_id=$7
  machine_index=$8
  machine_count=$9
  heartbeat_interval=${10}
  shift 10

  runner_pid=$BASHPID
  current_file_path="${status_file}.current"

  write_status() {
    local state=$1 exit_code=$2 current_jsonl="-" temp_file
    if [[ -r "$current_file_path" ]]; then
      IFS= read -r current_jsonl < "$current_file_path" || true
    fi
    temp_file="$status_file.tmp.$BASHPID"
    {
      printf "job_id=%s\n" "$job_id"
      printf "machine_index=%s\n" "$machine_index"
      printf "machine_count=%s\n" "$machine_count"
      printf "state=%s\n" "$state"
      printf "heartbeat_epoch=%s\n" "$(date +%s)"
      printf "pid=%s\n" "$runner_pid"
      printf "exit_code=%s\n" "$exit_code"
      printf "current_jsonl=%s\n" "${current_jsonl##*/}"
    } > "$temp_file"
    mv -f "$temp_file" "$status_file"
  }

  heartbeat_loop() {
    while true; do
      write_status running -
      sleep "$heartbeat_interval"
    done
  }

  cleanup() {
    local exit_code=$? final_state=stopped
    trap - EXIT INT TERM
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    if (( exit_code == 0 )); then
      final_state=completed
    fi
    write_status "$final_state" "$exit_code"
    printf "[%s] PROCESS_EXIT status=%s\n" "$(date --iso-8601=seconds)" "$exit_code"
    exit "$exit_code"
  }

  : > "$current_file_path"
  write_status starting -
  heartbeat_loop &
  heartbeat_pid=$!
  trap cleanup EXIT
  trap "exit 130" INT
  trap "exit 143" TERM

  status=0
  for input_jsonl in "$@"; do
    printf "%s\n" "${input_jsonl##*/}" > "$current_file_path"
    write_status running -
    printf "[%s] JSONL_START file=%s\n" "$(date --iso-8601=seconds)" "$input_jsonl"
    env PYTHONPATH="$PYTHONPATH_VALUE" \
      "$PYTHON_BIN" -S "$python_script" \
      --input-dir "$input_dir" \
      --input-jsonl "$input_jsonl" \
      --image-root "$image_root" \
      --tar-name part_000 \
      --output-dir "$output_dir" \
      --log-dir "$log_dir" \
      --use-qwen-38 \
      --data-workers "$DATA_WORKERS" \
      --api-workers "$API_WORKERS" \
      --timeout "$TIMEOUT" \
      --max-retries "$MAX_RETRIES" \
      --memory-log-interval 100 \
      --resume \
      --max-tokens "$MAX_TOKENS"
    status=$?
    printf "[%s] JSONL_EXIT status=%s file=%s\n" "$(date --iso-8601=seconds)" "$status" "$input_jsonl"
    if (( status != 0 )); then
      break
    fi
  done
  exit "$status"
' bash "$PYTHON_SCRIPT" "$INPUT_DIR" "$IMAGE_ROOT" "$OUTPUT_DIR" "$LOG_DIR" \
  "$STATUS_FILE" "$JOB_ID" "$MACHINE_INDEX" "$MACHINE_COUNT" "$HEARTBEAT_INTERVAL" \
  "${SHARD_FILES[@]}" >> "$LAUNCH_LOG" 2>&1 < /dev/null &

echo "0812_data qwen_38 machine $MACHINE_INDEX/$MACHINE_COUNT started in background, pid=$!"
echo "launcher log: $LAUNCH_LOG"
echo "per-jsonl logs: $LOG_DIR"
echo "all-nodes status log: $TOTAL_LOG"
