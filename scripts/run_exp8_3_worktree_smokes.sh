#!/usr/bin/env bash
# 顺序执行 Exp8.3 的 G2 / G2b / G3 训练。
# 支持模式：
#   - smoke
#   - baseline400
# 默认模式：smoke
# 默认顺序：g2 -> g2b -> g3
# 用法：
#   bash scripts/run_exp8_3_worktree_smokes.sh
#   bash scripts/run_exp8_3_worktree_smokes.sh smoke
#   bash scripts/run_exp8_3_worktree_smokes.sh baseline400
#   bash scripts/run_exp8_3_worktree_smokes.sh baseline400 g2 g3

set -euo pipefail

ISAACLAB="${ISAACLAB:-/home/uniubi/projects/forklift_sim/IsaacLab}"
POLL_SECONDS="${POLL_SECONDS:-30}"

default_experiments=(g2 g2b g3)
mode="smoke"
if [[ "${1:-}" == "smoke" || "${1:-}" == "baseline400" ]]; then
  mode="$1"
  shift
fi

if [[ "$#" -gt 0 ]]; then
  experiments=("$@")
else
  experiments=("${default_experiments[@]}")
fi

cleanup_train_processes() {
  echo "[INFO] Cleaning stale train.py processes"
  pkill -f "scripts/reinforcement_learning/rsl_rl/train.py" || true
  sleep 2
  pkill -9 -f "scripts/reinforcement_learning/rsl_rl/train.py" || true
}

print_resources() {
  echo "[INFO] Host memory"
  free -h
  echo "[INFO] GPU state"
  nvidia-smi
}

resolve_experiment() {
  case "$1:$2" in
    smoke:g2)
      echo "/home/uniubi/projects/forklift_sim_wt_g2|scripts/run_exp8_3_g2_smoke.sh|50"
      ;;
    smoke:g2b)
      echo "/home/uniubi/projects/forklift_sim_wt_g2b|scripts/run_exp8_3_g2b_smoke.sh|50"
      ;;
    smoke:g3)
      echo "/home/uniubi/projects/forklift_sim_wt_g3|scripts/run_exp8_3_g3_smoke.sh|50"
      ;;
    baseline400:g2)
      echo "/home/uniubi/projects/forklift_sim_wt_g2|scripts/run_exp8_3_g2_baseline.sh|400"
      ;;
    baseline400:g2b)
      echo "/home/uniubi/projects/forklift_sim_wt_g2b|scripts/run_exp8_3_g2b_baseline.sh|400"
      ;;
    baseline400:g3)
      echo "/home/uniubi/projects/forklift_sim_wt_g3|scripts/run_exp8_3_g3_baseline.sh|400"
      ;;
    *)
      echo "[ERROR] Unsupported mode/experiment: $1 $2" >&2
      return 1
      ;;
  esac
}

latest_iteration() {
  python3 - "$1" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("log-missing")
    raise SystemExit(0)

text = path.read_text(errors="ignore")
matches = re.findall(r"Learning iteration\s+(\d+)/(\d+)", text)
if not matches:
    print("starting")
else:
    cur, total = matches[-1]
    print(f"{cur}/{total}")
PY
}

run_succeeded() {
  python3 - "$1" "$2" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
target_total = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)

text = path.read_text(errors="ignore")
if "Traceback (most recent call last)" in text:
    raise SystemExit(1)

matches = re.findall(r"Learning iteration\s+(\d+)/(\d+)", text)
for cur_str, total_str in matches:
    cur = int(cur_str)
    total = int(total_str)
    if total == target_total and cur >= target_total - 1:
        raise SystemExit(0)

raise SystemExit(1)
PY
}

run_one_experiment() {
  local name="$1"
  local resolved worktree script_rel max_iters script_path branch head launch_output log_path pid

  resolved="$(resolve_experiment "$mode" "$name")"
  IFS="|" read -r worktree script_rel max_iters <<<"$resolved"
  script_path="$worktree/$script_rel"

  if [[ ! -d "$worktree" ]]; then
    echo "[ERROR] Missing worktree: $worktree" >&2
    return 1
  fi
  if [[ ! -f "$script_path" ]]; then
    echo "[ERROR] Missing script: $script_path" >&2
    return 1
  fi

  branch="$(git -C "$worktree" branch --show-current)"
  head="$(git -C "$worktree" rev-parse --short HEAD)"
  echo "[INFO] Running $name from $branch @ $head"

  cleanup_train_processes
  print_resources

  if ! launch_output="$(ISAACLAB="$ISAACLAB" bash "$script_path" 2>&1)"; then
    echo "$launch_output"
    echo "[ERROR] Failed to launch $name" >&2
    return 1
  fi

  echo "$launch_output"

  log_path="$(printf '%s\n' "$launch_output" | awk -F': ' '/^log:/ {print $2}' | tail -n1)"
  pid="$(printf '%s\n' "$launch_output" | awk -F': ' '/^pid:/ {print $2}' | tail -n1)"

  if [[ -z "${log_path}" || -z "${pid}" ]]; then
    echo "[ERROR] Could not parse log path or pid for $name" >&2
    return 1
  fi

  echo "[INFO] Waiting for $name (pid=$pid, log=$log_path)"
  while kill -0 "$pid" 2>/dev/null; do
    echo "[INFO] [$name] latest iteration: $(latest_iteration "$log_path")"
    sleep "$POLL_SECONDS"
  done

  echo "[INFO] $name process exited, validating $mode log"
  if run_succeeded "$log_path" "$max_iters"; then
    echo "[OK] $name $mode passed"
  else
    echo "[ERROR] $name $mode did not reach the expected final iteration" >&2
    return 1
  fi
}

echo "[INFO] Shared IsaacLab: $ISAACLAB"
echo "[INFO] Requested mode: $mode"
for experiment in "${experiments[@]}"; do
  run_one_experiment "$experiment"
done

echo "[DONE] All requested Exp8.3 worktree $mode runs completed."
