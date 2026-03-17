import torch
import math
import matplotlib.pyplot as plt
import numpy as np

def smoothstep(x):
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def generate_bezier_trajectory(p0, h0, pallet_xy, u_in, s_front, d_pre, l0, l1, num_points=32):
    """
    p0: (N, 2) 起点
    h0: (N, 2) 起点切线方向
    pallet_xy: (N, 2) 托盘中心
    u_in: (N, 2) 插入方向
    """
    N = p0.shape[0]
    p_goal = pallet_xy + s_front.unsqueeze(-1) * u_in
    p_pre = pallet_xy + (s_front - d_pre).unsqueeze(-1) * u_in
    
    B0 = p0
    B1 = p0 + l0.unsqueeze(-1) * h0
    B2 = p_pre - l1.unsqueeze(-1) * u_in
    B3 = p_pre
    
    # Bezier curve
    t = torch.linspace(0, 1, num_points).unsqueeze(0).unsqueeze(-1) # (1, P, 1)
    
    traj_bezier = (1-t)**3 * B0.unsqueeze(1) + \
                  3*(1-t)**2*t * B1.unsqueeze(1) + \
                  3*(1-t)*t**2 * B2.unsqueeze(1) + \
                  t**3 * B3.unsqueeze(1)
                  
    # Line segment
    t_line = torch.linspace(0, 1, 10).unsqueeze(0).unsqueeze(-1)
    traj_line = (1-t_line) * p_pre.unsqueeze(1) + t_line * p_goal.unsqueeze(1)
    
    traj = torch.cat([traj_bezier, traj_line[:, 1:, :]], dim=1)
    return traj

def check_fov(p0, yaw, pallet_xy, fov_deg=90.0):
    """检查托盘是否在视野内"""
    rel_pos = pallet_xy - p0
    target_yaw = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
    yaw_diff = torch.abs((target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
    return yaw_diff < math.radians(fov_deg / 2.0)

# 模拟 100 个随机初始位置
N = 100
torch.manual_seed(42)

# 托盘固定在原点，朝向 X 轴负方向
pallet_xy = torch.zeros((N, 2))
pallet_yaw = torch.full((N,), math.pi) # 朝向 -X
u_in = torch.stack([torch.cos(pallet_yaw), torch.sin(pallet_yaw)], dim=-1)
s_front = torch.full((N,), -0.6) # 托盘深度 1.2m 的一半

# 随机化叉车位置
# 保守分布：距离 1.5~2.5m，横向 ±0.5m，偏航 ±15°
dist = torch.empty(N).uniform_(1.5, 2.5)
lat = torch.empty(N).uniform_(-0.5, 0.5)
yaw_deg = torch.empty(N).uniform_(-15.0, 15.0)

# 计算绝对坐标
# 叉车在托盘前方 (X > 0)，朝向 -X 方向
p0_x = dist
p0_y = lat
p0 = torch.stack([p0_x, p0_y], dim=-1)
yaw = pallet_yaw + torch.deg2rad(yaw_deg)
h0 = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

# 生成轨迹
d_pre = torch.full((N,), 1.0)
l0 = torch.full((N,), 0.8)
l1 = torch.full((N,), 1.0)

traj = generate_bezier_trajectory(p0, h0, pallet_xy, u_in, s_front, d_pre, l0, l1)

# 检查 FOV
fov_ok = check_fov(p0, yaw, pallet_xy, fov_deg=90.0)
print(f"FOV 可见率: {fov_ok.float().mean().item() * 100:.1f}%")

# 绘图验证
plt.figure(figsize=(10, 10))
for i in range(min(20, N)):
    pts = traj[i].numpy()
    color = 'green' if fov_ok[i] else 'red'
    plt.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.5)
    
    # 画起点和朝向
    plt.arrow(p0[i, 0].item(), p0[i, 1].item(), 
              h0[i, 0].item()*0.3, h0[i, 1].item()*0.3, 
              head_width=0.05, color=color)

# 画托盘
pallet_rect = plt.Rectangle((-0.6, -0.6), 1.2, 1.2, fill=False, color='blue', linewidth=2)
plt.gca().add_patch(pallet_rect)
plt.plot([-0.6, -0.6], [-0.6, 0.6], color='blue', linewidth=4) # 前沿

plt.xlim(-1, 3)
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.title("Bezier Trajectory & FOV Check (Green: Visible, Red: Blind)")
plt.savefig("/home/uniubi/projects/forklift_sim/docs/0310-0317experiments/images/trajectory_fov_check.png")
print("轨迹图已保存至 docs/0310-0317experiments/images/trajectory_fov_check.png")

# 验证数值范围
# 假设叉车在距离轨迹 0.01m 到 1.0m 之间
r_cd_test = torch.tensor([0.01, 0.05, 0.1, 0.3, 0.5, 1.0])
print("\n数值范围验证:")
print("r_cd (m) | 1/r_cd | exp(-r_cd/0.2)")
print("-" * 40)
for r in r_cd_test:
    inv = 1.0 / r
    exp_val = torch.exp(-r / 0.2)
    print(f"{r.item():.2f}     | {inv.item():.2f}  | {exp_val.item():.4f}")

