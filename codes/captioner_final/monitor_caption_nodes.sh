#!/usr/bin/env bash
set -uo pipefail

if (( $# != 4 )); then
  echo "usage: $0 STATUS_DIR TOTAL_LOG CHECK_INTERVAL_SECONDS STALE_SECONDS" >&2
  exit 2
fi

STATUS_DIR=$1
TOTAL_LOG=$2
CHECK_INTERVAL=$3
STALE_SECONDS=$4
EXPECTED_COUNT_FILE="$STATUS_DIR/expected_machine_count"
JOB_ID_FILE="$STATUS_DIR/current_job_id"
LOCK_FILE="$STATUS_DIR/all_nodes_monitor.lock"

mkdir -p "$STATUS_DIR" "${TOTAL_LOG%/*}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  exit 0
fi

read_field() {
  local field=$1 file=$2
  sed -n "s/^${field}=//p" "$file" | tail -n 1
}

while true; do
  now=$(date +%s)
  timestamp=$(date --iso-8601=seconds)
  machine_count=0
  current_job_id="unknown"
  [[ -r "$EXPECTED_COUNT_FILE" ]] && read -r machine_count < "$EXPECTED_COUNT_FILE"
  [[ -r "$JOB_ID_FILE" ]] && read -r current_job_id < "$JOB_ID_FILE"

  if ! [[ "$machine_count" =~ ^[1-9][0-9]*$ ]]; then
    printf '[%s] MONITOR waiting_for_valid_machine_count value=%s\n' "$timestamp" "$machine_count" >> "$TOTAL_LOG"
    sleep "$CHECK_INTERVAL"
    continue
  fi

  running=0
  completed=0
  stopped=0
  {
    printf '[%s] CLUSTER_STATUS job=%s expected_nodes=%s\n' "$timestamp" "$current_job_id" "$machine_count"
    for ((index = 0; index < machine_count; index++)); do
      tag=$(printf 'machine_%03d' "$index")
      status_file="$STATUS_DIR/$tag.status"
      if [[ ! -r "$status_file" ]]; then
        printf '[%s] NODE node=%s status=STOPPED reason=status_missing\n' "$timestamp" "$index"
        ((stopped += 1))
        continue
      fi

      node_job_id=$(read_field job_id "$status_file")
      state=$(read_field state "$status_file")
      heartbeat=$(read_field heartbeat_epoch "$status_file")
      pid=$(read_field pid "$status_file")
      exit_code=$(read_field exit_code "$status_file")
      current_jsonl=$(read_field current_jsonl "$status_file")

      if [[ "$node_job_id" != "$current_job_id" ]]; then
        printf '[%s] NODE node=%s status=STOPPED reason=old_job heartbeat_job=%s\n' "$timestamp" "$index" "$node_job_id"
        ((stopped += 1))
      elif ! [[ "$heartbeat" =~ ^[0-9]+$ ]]; then
        printf '[%s] NODE node=%s status=STOPPED reason=invalid_heartbeat state=%s\n' "$timestamp" "$index" "$state"
        ((stopped += 1))
      else
        age=$((now - heartbeat))
        if [[ "$state" == "completed" ]]; then
          printf '[%s] NODE node=%s status=COMPLETED heartbeat_age=%ss pid=%s\n' "$timestamp" "$index" "$age" "$pid"
          ((completed += 1))
        elif [[ "$state" == "running" || "$state" == "starting" ]] && (( age <= STALE_SECONDS )); then
          printf '[%s] NODE node=%s status=RUNNING state=%s heartbeat_age=%ss pid=%s current=%s\n' \
            "$timestamp" "$index" "$state" "$age" "$pid" "${current_jsonl:-none}"
          ((running += 1))
        elif (( age > STALE_SECONDS )); then
          printf '[%s] NODE node=%s status=STOPPED reason=heartbeat_stale heartbeat_age=%ss last_state=%s pid=%s current=%s\n' \
            "$timestamp" "$index" "$age" "$state" "$pid" "${current_jsonl:-none}"
          ((stopped += 1))
        else
          printf '[%s] NODE node=%s status=STOPPED reason=process_exit state=%s exit_code=%s pid=%s current=%s\n' \
            "$timestamp" "$index" "$state" "${exit_code:-unknown}" "$pid" "${current_jsonl:-none}"
          ((stopped += 1))
        fi
      fi
    done
    printf '[%s] CLUSTER_SUMMARY running=%s completed=%s stopped=%s expected=%s\n' \
      "$timestamp" "$running" "$completed" "$stopped" "$machine_count"
  } >> "$TOTAL_LOG"

  sleep "$CHECK_INTERVAL"
done
