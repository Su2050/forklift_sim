#!/bin/bash
# S1.0T 离线评估运行器
# 用法: bash scripts/run_s1.0t_eval.sh <实验名> <checkpoint> [评估级别] [参数覆盖...]
#
# 评估级别:
#   quick    - 10 seeds (1-10), 快速迭代
#   standard - 30 seeds (1-30), Phase 决策节点
#   full     - 50 seeds (1-50), 最终验证
#
# 示例:
#   bash scripts/run_s1.0s_eval.sh P0_baseline \
#       logs/rsl_rl/forklift_pallet_insert_lift/2026-02-13_18-40-18/model_3595.pt \
#       quick
#
#   bash scripts/run_s1.0s_eval.sh P1_schemeB \
#       logs/rsl_rl/forklift_pallet_insert_lift/<run_dir>/model_XXX.pt \
#       standard y_err_obs_scale=0.8

set -e

# ---- 参数解析 ----
EXP_NAME=${1:?用法: $0 <实验名> <checkpoint> [评估级别: quick/standard/full] [参数覆盖...]}
CHECKPOINT=${2:?缺少 checkpoint 路径}
EVAL_LEVEL=${3:-quick}
shift 3 2>/dev/null || shift $#
OVERRIDES=("$@")

# ---- 评估级别 -> seed 范围 ----
case "$EVAL_LEVEL" in
    quick)    SEED_END=10 ;;
    standard) SEED_END=30 ;;
    full)     SEED_END=50 ;;
    *)
        echo "[ERROR] 未知评估级别: $EVAL_LEVEL (可选: quick/standard/full)"
        exit 1
        ;;
esac

# ---- 路径定义 ----
PROJECT=/home/uniubi/projects/forklift_sim
ISAACLAB=$PROJECT/IsaacLab
PATCH_SRC=$PROJECT/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift
DST=$ISAACLAB/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift
CFG_FILE=$PATCH_SRC/env_cfg.py
CFG_BACKUP=$PATCH_SRC/env_cfg.py.bak
OUTPUT_DIR=$PROJECT/data/s1.0t_eval/${EXP_NAME}

# ---- 确保在 feat/s1.0s* 分支 ----
cd "$PROJECT"
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != feat/s1.0t* ]]; then
    echo "[ERROR] 当前分支 $CURRENT_BRANCH，需要在 feat/s1.0t* 上运行"
    exit 1
fi

# ---- 备份 + 应用参数覆盖 ----
cp "$CFG_FILE" "$CFG_BACKUP"

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

# ---- 运行评估 ----
echo "============================================================"
echo "[$(date '+%H:%M:%S')] S1.0S Eval: $EXP_NAME ($EVAL_LEVEL, seeds 1-$SEED_END)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"

cd "$ISAACLAB"
CONDA_PREFIX="" TERM=xterm \
bash isaaclab.sh -p ../scripts/eval_s1.0s_diagnostics.py \
    --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
    --headless --num_envs 1024 \
    --checkpoint "$CHECKPOINT" \
    --experiment_name "$EXP_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 2000 \
    --seed_start 1 --seed_end $SEED_END

echo "[$(date '+%H:%M:%S')] Eval 完成: $OUTPUT_DIR"

# ---- 恢复 env_cfg.py ----
cd "$PROJECT"
cp "$CFG_BACKUP" "$CFG_FILE"
rm -f "$CFG_BACKUP"
echo "[$(date '+%H:%M:%S')] 已恢复 env_cfg.py 到默认值"
