#!/bin/bash
# S1.0S 实验运行器
# 用法: bash scripts/run_s1.0s_experiment.sh <实验名> <seed> <max_iter> <resume_dir> <resume_model> [参数覆盖...]
#
# 示例:
#   # Phase-0.5 P1（方案B单独: y_err_obs_scale=0.8）
#   bash scripts/run_s1.0s_experiment.sh P1_schemeB 42 300 \
#       2026-02-13_18-40-18 model_3595.pt \
#       y_err_obs_scale=0.8
#
#   # Phase-1a L_s15（sigma_lift=0.15, lift_delta_m=0.25）
#   bash scripts/run_s1.0s_experiment.sh L_s15 42 500 \
#       <phase05_best_dir> <phase05_best_model> \
#       lift_delta_m=0.25 lift_exit_epsilon=0.02 sigma_lift=0.15
#
# 参数覆盖格式: param_name=value（env_cfg.py 中的属性名 = 新值）
# 训练结束后自动恢复 env_cfg.py 到默认值。

set -e

# ---- 参数解析 ----
EXP_NAME=${1:?用法: $0 <实验名> <seed> <max_iter> <resume_dir> <resume_model> [参数覆盖...]}
SEED=${2:?缺少 seed}
MAX_ITER=${3:?缺少 max_iter}
RESUME_DIR=${4:?缺少 resume_dir (如 2026-02-13_18-40-18)}
RESUME_MODEL=${5:?缺少 resume_model (如 model_3595.pt)}
shift 5
OVERRIDES=("$@")

# ---- 路径定义 ----
PROJECT=/home/uniubi/projects/forklift_sim
ISAACLAB=$PROJECT/IsaacLab
PATCH_SRC=$PROJECT/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift
DST=$ISAACLAB/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift
CFG_FILE=$PATCH_SRC/env_cfg.py
CFG_BACKUP=$PATCH_SRC/env_cfg.py.bak

# ---- 确保在 feat/s1.0s* 分支 ----
cd "$PROJECT"
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != feat/s1.0s* ]]; then
    echo "[ERROR] 当前分支 $CURRENT_BRANCH，需要在 feat/s1.0s* 上运行"
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
grep -E "lift_delta_m|lift_exit_epsilon|sigma_lift|y_err_obs_scale|y_gate2|k_far_lat|milestone_dead_zone_scale|ins_floor|rew_milestone_lift|global_stall_steps" "$DST/env_cfg.py" | sed 's/^    /  /' || true

# ---- 日志文件 ----
LOG=$PROJECT/logs/$(date +%Y%m%d_%H%M%S)_train_s1.0s_${EXP_NAME}_s${SEED}.log

echo "============================================================"
echo "[$(date '+%H:%M:%S')] Starting s1.0s/$EXP_NAME (seed=$SEED, iter=$MAX_ITER)"
echo "  Resume: $RESUME_DIR / $RESUME_MODEL"
echo "  Log: $LOG"
echo "============================================================"

# ---- 运行训练 ----
cd "$ISAACLAB"
CONDA_PREFIX="" TERM=xterm \
bash isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
    --headless --num_envs 1024 --max_iterations $MAX_ITER \
    --resume --load_run "$RESUME_DIR" --checkpoint "$RESUME_MODEL" \
    --seed $SEED \
    > "$LOG" 2>&1

echo "[$(date '+%H:%M:%S')] Finished s1.0s/$EXP_NAME -> $LOG"

# ---- 恢复 env_cfg.py ----
cd "$PROJECT"
cp "$CFG_BACKUP" "$CFG_FILE"
rm -f "$CFG_BACKUP"
echo "[$(date '+%H:%M:%S')] 已恢复 env_cfg.py 到默认值"
