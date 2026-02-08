#!/usr/bin/env bash
# 将 forklift_pallet_insert_lift 任务安装到 IsaacLab 目录中。
# 说明：
# - 这是“补丁式”安装：把任务目录复制到 IsaacLab 固定路径
# - 会覆盖 IsaacLab 里已有同名任务目录（全量替换）
# - 并确保 direct/__init__.py 中有导入行以触发 gym.register()
set -euo pipefail  # 遇到错误立即退出；未定义变量报错；管道中任一失败则整体失败

ISAACLAB_DIR="${1:-}"
if [[ -z "${ISAACLAB_DIR}" ]]; then
  echo "Usage: $0 /path/to/IsaacLab"
  exit 1
fi
if [[ ! -d "${ISAACLAB_DIR}/source/isaaclab_tasks/isaaclab_tasks" ]]; then
  echo "Error: ${ISAACLAB_DIR} does not look like an IsaacLab checkout (missing source/isaaclab_tasks/isaaclab_tasks)"
  exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # 当前项目根目录
PATCH_SRC="${SRC_DIR}/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift"
DST_DIR="${ISAACLAB_DIR}/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift"

echo "[INFO] Copying task into IsaacLab..."
rm -rf "${DST_DIR}"  # 全量替换，避免旧文件残留
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
