#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/uniubi/projects/forklift_sim"
ISAACLAB_DIR="${REPO_ROOT}/IsaacLab"
SCRIPT_PATH="${REPO_ROOT}/scripts/visualize_exp83_runtime_trajectory_topdown.py"

mkdir -p "${REPO_ROOT}/logs"

run_case() {
  local label="$1"
  local x_root="$2"
  local pre_dist="$3"
  local out_dir="$4"
  local ts log
  ts="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
  log="${REPO_ROOT}/logs/${ts}_${label}.log"

  (
    cd "${ISAACLAB_DIR}" || exit 1
    exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PROMPT_MODIFIER -u CONDA_PYTHON_EXE -u _CE_CONDA -u _CE_M \
      TERM=xterm PYTHONUNBUFFERED=1 \
      ./isaaclab.sh -p "${SCRIPT_PATH}" \
      --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
      --headless \
      --label "${label}" \
      --x_root "${x_root}" \
      --y_values=-0.08,0.0,0.08 \
      --yaw_deg_values=-3,0,3 \
      --output_dir "${out_dir}" \
      env.traj_pre_dist_m="${pre_dist}"
  ) >"${log}" 2>&1

  echo "[DONE] ${label} x=${x_root} pre=${pre_dist} log=${log}"
}

OUT_BASE="${REPO_ROOT}/outputs/exp83_stage1_entry_geometry_v3"

# V3-A: mild geometry fix
for X in -3.65 -3.50 -3.35; do
  run_case \
    "exp83_v3A_entry_geom_x${X}" \
    "${X}" \
    "1.10" \
    "${OUT_BASE}/v3A_pre1p10"
done

# V3-B: stronger geometry fix
for X in -3.70 -3.55 -3.40; do
  run_case \
    "exp83_v3B_entry_geom_x${X}" \
    "${X}" \
    "1.05" \
    "${OUT_BASE}/v3B_pre1p05"
done

echo "[ALL DONE] geometry sweep outputs in ${OUT_BASE}"
