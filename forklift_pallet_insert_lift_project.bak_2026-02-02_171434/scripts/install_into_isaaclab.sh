#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_DIR="${1:-}"
if [[ -z "${ISAACLAB_DIR}" ]]; then
  echo "Usage: $0 /path/to/IsaacLab"
  exit 1
fi
if [[ ! -d "${ISAACLAB_DIR}/source/isaaclab_tasks/isaaclab_tasks" ]]; then
  echo "Error: ${ISAACLAB_DIR} does not look like an IsaacLab checkout (missing source/isaaclab_tasks/isaaclab_tasks)"
  exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_SRC="${SRC_DIR}/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift"
DST_DIR="${ISAACLAB_DIR}/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift"

echo "[INFO] Copying task into IsaacLab..."
rm -rf "${DST_DIR}"
mkdir -p "$(dirname "${DST_DIR}")"
cp -R "${PATCH_SRC}" "${DST_DIR}"

INIT_FILE="${ISAACLAB_DIR}/source/isaaclab_tasks/isaaclab_tasks/direct/__init__.py"
IMPORT_LINE="from . import forklift_pallet_insert_lift  # noqa: F401"

echo "[INFO] Patching ${INIT_FILE} to import task registration..."
if ! grep -q "forklift_pallet_insert_lift" "${INIT_FILE}"; then
  echo "" >> "${INIT_FILE}"
  echo "${IMPORT_LINE}" >> "${INIT_FILE}"
  echo "[OK] Added import line."
else
  echo "[OK] Import already present."
fi

echo "[DONE] Patch applied."
echo "Next:"
echo "  cd ${ISAACLAB_DIR}"
echo "  ./isaaclab.sh -i rsl_rl"
echo "  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Forklift-PalletInsertLift-Direct-v0 --headless --num_envs 128"
