# 开启训练（S1.0h）
# 注意：必须先退出 conda（Isaac Sim 使用自带 Python 3.11）
conda deactivate
cd /home/uniubi/projects/forklift_sim/IsaacLab

LOG_FILE="/home/uniubi/projects/forklift_sim/logs/$(date +%Y%m%d_%H%M%S)_train_s1.0h.log"

nohup env TERM=xterm PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 \
  --headless \
  --max_iterations 2000 \
  agent.run_name=exp_s1.0h \
  > "$LOG_FILE" 2>&1 &

echo "训练已在后台启动，PID: $!"
echo "日志文件: $LOG_FILE"

# 实时查看日志
tail -f "$LOG_FILE"

# 查看最新的 mean reward
grep "mean reward" /home/uniubi/projects/forklift_sim/train_reward_v3.log | tail -20


# 在远程机器上启动 TensorBoard
tensorboard --logdir=/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl --port=6006 &

# 在本地机器上建立 SSH 隧道
ssh -L 6006:localhost:6006 用户名@远程主机IP
# 然后浏览器打开 http://localhost:6006




#开启测试
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-04_07-37-47_exp_insert_norm_fix_v2/model_1999.pt" \
  --headless \
  --video --video_length 600
# 注意: video_length 的单位是步数(steps)，不是秒数
# 环境步长约为 0.033秒，所以:
# - 50步 ≈ 1.7秒
# - 100步 ≈ 3.3秒  
# - 300步 ≈ 10秒
# - 600步 ≈ 20秒
# - 1300步 ≈ 43秒

/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-04_19-42-04_exp_gate_optimization_v1/model_1999.pt


cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-04_19-42-04_exp_gate_optimization_v1/model_1999.pt" \
  --headless \
  --video --video_length 600



  /home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-04_22-59-47_exp_gate_optimization_v2_rew_progress_8/model_8599.pt

cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-04_22-59-47_exp_gate_optimization_v2_rew_progress_8/model_8599.pt" \
  --headless \
  --video --video_length 600


# ============================================================
# 验证与诊断
# ============================================================
# 注意：以下脚本必须通过 isaaclab.sh 运行（需要 IsaacLab Python 环境），
#       不能直接 python scripts/xxx.py
# 详细说明：docs/verify_forklift_insert_lift_usage.md

# --- 叉车插入举升功能验证 (verify_forklift_insert_lift.py) ---



# 3) 手动键盘控制 — 自由操控叉车
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --manual
#   键位：W/S 前进后退, A/D 左右转, R/F 货叉升降, G 重置到理想位置, P 打印状态, SPACE 停止

# 4) 手动控制 + 自动对齐 — 先自动将叉车摆到理想插入位置，再交由键盘接管
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --manual --auto-align

# --- 叉车托盘几何兼容性检查 (verify_geometry_compatibility.py) ---
# 分析货叉/托盘插入孔尺寸、碰撞形状、兼容性，生成诊断报告
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_geometry_compatibility.py --headless

# --- Nucleus 托盘资产扫描 (scan_nucleus_pallets.py) ---
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/scan_nucleus_pallets.py --headless