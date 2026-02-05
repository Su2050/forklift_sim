#开启训练
cd /home/uniubi/projects/forklift_sim/IsaacLab

nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1024 \
  --headless \
  --max_iterations 2000 \
  agent.run_name=exp_reward_v3 \
  > ../train_reward_v3.log 2>&1 &

echo "训练已在后台启动，PID: $!"

# 实时查看日志
tail -f /home/uniubi/projects/forklift_sim/train_reward_v3.log

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


#开启验证脚本（手动控制叉车验证插入举升功能）
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --headless

# 注意：必须通过 isaaclab.sh 运行，不能直接运行脚本
# 详细说明请参考：docs/verify_forklift_insert_lift_usage.md


#验证IsaacSim环境
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --headless

#验证叉车托盘几何兼容性（检查货叉与插入孔尺寸是否匹配）
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_geometry_compatibility.py --headless

# 注意：必须通过 isaaclab.sh 运行，不能直接运行脚本
# 该脚本会：
# 1. 分析货叉几何尺寸（宽度、高度、间距、长度）
# 2. 分析托盘插入孔几何尺寸（宽度、高度、间距、深度）
# 3. 检查碰撞形状类型
# 4. 进行几何兼容性分析
# 5. 执行实际碰撞测试
# 6. 生成诊断报告和建议


使用方式：
自动模式（原有逻辑）
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --headless
手动模式（可视化）
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --manual
手动 + 自动对齐
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --manual --auto-align

cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/scan_nucleus_pallets.py --headless

cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --manual --auto-align

cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/verify_forklift_insert_lift.py --manual


cd /home/uniubi/projects/forklift_sim
python scripts/verify_forklift_insert_lift.py