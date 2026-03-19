import torch
import numpy as np

# 假设托盘在原点，朝向为 0
p_goal = torch.tensor([[0.0, 0.0]])
u_in = torch.tensor([[1.0, 0.0]])

# p_pre 是预对位点，距离 p_goal 1.2m
p_pre = p_goal - 1.2 * u_in

# 假设初始位置在 -4.0m，偏航角 0
p0 = torch.tensor([[-4.0, 1.0]])
t0 = torch.tensor([[1.0, 0.0]])

dist = torch.norm(p_pre - p0, dim=-1, keepdim=True)
L = dist * 1.5

m0 = t0 * L
m1 = u_in * L

# 离散化
num_curve = 70
t = torch.linspace(0.0, 1.0, num_curve).view(1, -1, 1)
t2 = t ** 2
t3 = t ** 3

h00 = 2*t3 - 3*t2 + 1
h10 = t3 - 2*t2 + t
h01 = -2*t3 + 3*t2
h11 = t3 - t2

p0_exp = p0.unsqueeze(1)
m0_exp = m0.unsqueeze(1)
p1_exp = p_pre.unsqueeze(1)
m1_exp = m1.unsqueeze(1)

pts_curve = h00 * p0_exp + h10 * m0_exp + h01 * p1_exp + h11 * m1_exp

dh00 = 6*t2 - 6*t
dh10 = 3*t2 - 4*t + 1
dh01 = -6*t2 + 6*t
dh11 = 3*t2 - 2*t
vel_curve = dh00 * p0_exp + dh10 * m0_exp + dh01 * p1_exp + dh11 * m1_exp
yaw_curve = torch.atan2(vel_curve[..., 1], vel_curve[..., 0])

# 假设叉车叉尖到达托盘前沿 (p_goal)，此时车体中心在 p_goal - 1.87m
# 也就是 x = -1.87
body_x = -1.87

# 找到曲线上 x 最接近 -1.87 的点
dists = torch.abs(pts_curve[0, :, 0] - body_x)
min_idx = torch.argmin(dists)

print(f"When fork tip is at pallet front (body_x={body_x}):")
print(f"Closest point on curve: x={pts_curve[0, min_idx, 0].item():.2f}, y={pts_curve[0, min_idx, 1].item():.2f}")
print(f"Tangent yaw at this point: {yaw_curve[0, min_idx].item() * 180 / np.pi:.2f} degrees")
