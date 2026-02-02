cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 1 \
  --checkpoint "/home/uniubi/projects/forklift_sim/IsaacLab/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-02_20-25-04/model_3998.pt" \
  --headless \
  --video --video_length 1300
# 注意: video_length 的单位是步数(steps)，不是秒数
# 环境步长约为 0.033秒，所以:
# - 50步 ≈ 1.7秒
# - 100步 ≈ 3.3秒  
# - 300步 ≈ 10秒
# - 600步 ≈ 20秒
# - 1300步 ≈ 43秒