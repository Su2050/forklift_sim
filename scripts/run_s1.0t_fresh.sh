#!/bin/bash
# S1.0T 从头训练运行器（不 resume）
# 用法: bash scripts/run_s1.0t_fresh.sh <实验名> <seed> <max_iter> [参数覆盖...]
#
# 示例:
#   bash scripts/run_s1.0t_fresh.sh F1_fresh 42 2000 \
#       sigma_lift=0.15 k_lift_progress=1.2 lift_speed_m_s=0.5
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

# ---- 确保在 feat/s1.0t* 分支 ----
cd "$PROJECT"
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != feat/s1.0t* ]]; then
    echo "[ERROR] 当前分支 $CURRENT_BRANCH，需要在 feat/s1.0t* 上运行"
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

        python3 -c "
import re, sys
param, value = '$PARAM', '$VALUE'
with open('$CFG_FILE', 'r') as f:
    content = f.read()
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
grep -E "lift_delta_m|lift_exit_epsilon|sigma_lift|lift_speed_m_s|lift_pos_scale|k_lift_progress|rew_milestone_lift|global_stall_steps" "$DST/env_cfg.py" | sed 's/^    /  /' || true

# ---- 日志文件 ----
LOG=$PROJECT/logs/$(date +%Y%m%d_%H%M%S)_train_s1.0t_${EXP_NAME}_s${SEED}.log

echo "============================================================"
echo "[$(date '+%H:%M:%S')] Starting s1.0t/$EXP_NAME (seed=$SEED, iter=$MAX_ITER, FRESH)"
echo "  Log: $LOG"
echo "============================================================"

# ---- 运行训练（从头，无 --resume）----
cd "$ISAACLAB"
CONDA_PREFIX="" TERM=xterm \
bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
    --headless --num_envs 1024 --max_iterations $MAX_ITER \
    --seed $SEED \
    > "$LOG" 2>&1

echo "[$(date '+%H:%M:%S')] Finished s1.0t/$EXP_NAME -> $LOG"

# ---- 恢复 env_cfg.py ----
cd "$PROJECT"
cp "$CFG_BACKUP" "$CFG_FILE"
rm -f "$CFG_BACKUP"
echo "[$(date '+%H:%M:%S')] 已恢复 env_cfg.py 到默认值"
