#!/bin/bash
# S1.0Q 通用实验运行器
# 用法: bash scripts/run_s1.0q_experiment.sh <实验名> <seed> <max_iter> [参数覆盖...]
#
# 示例:
#   bash scripts/run_s1.0q_experiment.sh A1 42 300 k_dead_zone=0.5 milestone_dead_zone_scale=0.0
#   bash scripts/run_s1.0q_experiment.sh B1a 42 300 ins_floor=0.1
#
# 参数覆盖格式: param_name=value（env_cfg.py 中的属性名 = 新值）
# 训练结束后自动恢复 env_cfg.py 到默认值。

set -e

# ---- 参数解析 ----
EXP_NAME=${1:?用法: $0 <实验名> <seed> <max_iter> [参数覆盖...]}
SEED=${2:?缺少 seed}
MAX_ITER=${3:?缺少 max_iter}
shift 3
OVERRIDES=("$@")

# ---- 路径定义 ----
PROJECT=/home/uniubi/projects/forklift_sim
ISAACLAB=$PROJECT/IsaacLab
PATCH_SRC=$PROJECT/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift
DST=$ISAACLAB/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift
CFG_FILE=$PATCH_SRC/env_cfg.py
CFG_BACKUP=$PATCH_SRC/env_cfg.py.bak

# ---- 确保在 feat/s1.0q* 分支 ----
cd "$PROJECT"
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != feat/s1.0q* ]]; then
    echo "[ERROR] 当前分支 $CURRENT_BRANCH，需要在 feat/s1.0q* 上运行"
    exit 1
fi

# ---- 备份 env_cfg.py ----
cp "$CFG_FILE" "$CFG_BACKUP"

# ---- 应用参数覆盖 ----
if [ ${#OVERRIDES[@]} -gt 0 ]; then
    echo "[$(date '+%H:%M:%S')] 应用参数覆盖:"
    for OVERRIDE in "${OVERRIDES[@]}"; do
        PARAM=$(echo "$OVERRIDE" | cut -d= -f1)
        VALUE=$(echo "$OVERRIDE" | cut -d= -f2)
        echo "  $PARAM = $VALUE"

        # 使用 Python 精确替换参数值（处理类型：float/int/bool）
        python3 -c "
import re, sys
param, value = '$PARAM', '$VALUE'
with open('$CFG_FILE', 'r') as f:
    content = f.read()
# 匹配 'param_name: type = old_value' 或 'param_name: type = old_value  # comment'
pattern = r'(\s+' + re.escape(param) + r'\s*:\s*\w+\s*=\s*)([^\s#]+)'
new_content, n = re.subn(pattern, r'\g<1>' + value, content)
if n == 0:
    print(f'[WARNING] 未找到参数 {param}，跳过')
    sys.exit(0)
with open('$CFG_FILE', 'w') as f:
    f.write(new_content)
print(f'  -> 已替换 {param} = {value} ({n} 处)')
"
    done
fi

# ---- 同步文件到 IsaacLab ----
cp "$PATCH_SRC/env.py"      "$DST/env.py"
cp "$PATCH_SRC/env_cfg.py"  "$DST/env_cfg.py"
cp "$PATCH_SRC/clamped_actor_critic.py" "$DST/clamped_actor_critic.py" 2>/dev/null || true
cp "$PATCH_SRC/__init__.py" "$DST/__init__.py"
cp "$PATCH_SRC/agents/rsl_rl_ppo_cfg.py" "$DST/agents/rsl_rl_ppo_cfg.py"
echo "[$(date '+%H:%M:%S')] 同步完成 -> IsaacLab"

# ---- 验证关键参数 ----
echo "[$(date '+%H:%M:%S')] 实验 $EXP_NAME 关键参数:"
grep -E "k_dead_zone|k_retreat|ins_floor|ins_lat_gate_sigma|milestone_dead_zone_scale|k_lat_fine" "$DST/env_cfg.py" | sed 's/^    /  /' || true

# ---- 日志文件 ----
LOG=$PROJECT/logs/$(date +%Y%m%d_%H%M%S)_train_s1.0q_${EXP_NAME}_s${SEED}.log

echo "============================================================"
echo "[$(date '+%H:%M:%S')] Starting s1.0q/$EXP_NAME (seed=$SEED, iter=$MAX_ITER)"
echo "  Log: $LOG"
echo "============================================================"

# ---- 运行训练 ----
cd "$ISAACLAB"
CONDA_PREFIX="" TERM=xterm \
bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
    --headless --num_envs 1024 --max_iterations $MAX_ITER \
    --resume --load_run "2026-02-12_11-14-55" --checkpoint "model_3296.pt" \
    --seed $SEED \
    > "$LOG" 2>&1

echo "[$(date '+%H:%M:%S')] Finished s1.0q/$EXP_NAME -> $LOG"

# ---- 恢复 env_cfg.py ----
cd "$PROJECT"
cp "$CFG_BACKUP" "$CFG_FILE"
rm -f "$CFG_BACKUP"
echo "[$(date '+%H:%M:%S')] 已恢复 env_cfg.py 到默认值"
