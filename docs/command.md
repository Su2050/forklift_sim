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
  --checkpoint "/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-03_00-07-04_exp_reward_v2/model_1999.pt" \
  --headless \
  --video --video_length 600
# 注意: video_length 的单位是步数(steps)，不是秒数
# 环境步长约为 0.033秒，所以:
# - 50步 ≈ 1.7秒
# - 100步 ≈ 3.3秒  
# - 300步 ≈ 10秒
# - 600步 ≈ 20秒
# - 1300步 ≈ 43秒