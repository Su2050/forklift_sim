import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_dubins_like_trajectory(start_pos, start_yaw, target_pos, target_yaw, num_points=32, device="cpu"):
    """
    生成一种更符合叉车运动学的平滑轨迹。
    由于真正的 Clothoid 涉及复杂的菲涅尔积分，且在强化学习中难以进行批处理计算，
    我们采用一种基于曲率连续的多项式样条方法，确保轨迹的平滑性。
    """
    N = start_pos.shape[0]
    
    # 将问题转换到以 target 为原点，target_yaw 为 x 轴的局部坐标系
    dx = start_pos[:, 0] - target_pos[:, 0]
    dy = start_pos[:, 1] - target_pos[:, 1]
    
    cos_t = torch.cos(-target_yaw)
    sin_t = torch.sin(-target_yaw)
    
    local_x = dx * cos_t - dy * sin_t
    local_y = dx * sin_t + dy * cos_t
    local_yaw = start_yaw - target_yaw
    local_yaw = torch.atan2(torch.sin(local_yaw), torch.cos(local_yaw))
    
    # 我们使用参数方程 x(t), y(t) 来生成曲线，t 从 0 到 1
    # 边界条件：
    # t=0: x(0)=local_x, y(0)=local_y, dx/dt=L*cos(local_yaw), dy/dt=L*sin(local_yaw)
    # t=1: x(1)=0, y(1)=0, dx/dt=L, dy/dt=0
    # 其中 L 是控制曲线弯曲程度的长度参数，通常取起点到终点直线距离的常数倍
    
    dist = torch.sqrt(local_x**2 + local_y**2)
    L = dist * 1.5  # 经验常数，控制曲线的“肚子”大小
    
    # 构造三次埃尔米特样条 (Cubic Hermite Spline)
    # p(t) = (2t^3 - 3t^2 + 1)p0 + (t^3 - 2t^2 + t)m0 + (-2t^3 + 3t^2)p1 + (t^3 - t^2)m1
    
    t = torch.linspace(0.0, 1.0, num_points, device=device).view(1, -1).expand(N, -1)
    t2 = t ** 2
    t3 = t ** 3
    
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    
    # X 坐标
    p0_x = local_x.unsqueeze(1)
    m0_x = (L * torch.cos(local_yaw)).unsqueeze(1)
    p1_x = torch.zeros_like(p0_x)
    m1_x = L.unsqueeze(1)
    
    traj_x_local = h00 * p0_x + h10 * m0_x + h01 * p1_x + h11 * m1_x
    
    # Y 坐标
    p0_y = local_y.unsqueeze(1)
    m0_y = (L * torch.sin(local_yaw)).unsqueeze(1)
    p1_y = torch.zeros_like(p0_y)
    m1_y = torch.zeros_like(p0_y)
    
    traj_y_local = h00 * p0_y + h10 * m0_y + h01 * p1_y + h11 * m1_y
    
    # 计算切线方向 (dx/dt, dy/dt) 来获取 yaw
    dh00 = 6*t2 - 6*t
    dh10 = 3*t2 - 4*t + 1
    dh01 = -6*t2 + 6*t
    dh11 = 3*t2 - 2*t
    
    dx_dt = dh00 * p0_x + dh10 * m0_x + dh01 * p1_x + dh11 * m1_x
    dy_dt = dh00 * p0_y + dh10 * m0_y + dh01 * p1_y + dh11 * m1_y
    
    traj_yaw_local = torch.atan2(dy_dt, dx_dt)
    
    # 转换回全局坐标系
    cos_inv = torch.cos(target_yaw).unsqueeze(1)
    sin_inv = torch.sin(target_yaw).unsqueeze(1)
    
    traj_x_global = target_pos[:, 0].unsqueeze(1) + traj_x_local * cos_inv - traj_y_local * sin_inv
    traj_y_global = target_pos[:, 1].unsqueeze(1) + traj_x_local * sin_inv + traj_y_local * cos_inv
    traj_yaw_global = traj_yaw_local + target_yaw.unsqueeze(1)
    
    traj_yaw_global = torch.atan2(torch.sin(traj_yaw_global), torch.cos(traj_yaw_global))
    
    trajectory = torch.stack([traj_x_global, traj_y_global, traj_yaw_global], dim=-1)
    return trajectory

if __name__ == "__main__":
    # 测试用例
    start_pos = torch.tensor([[-4.0, 1.0], [-4.0, -1.0], [-3.0, 2.0], [-2.0, -1.5]])
    start_yaw = torch.tensor([0.0, 0.0, -0.5, 0.5])
    target_pos = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    target_yaw = torch.tensor([0.0, 0.0, 0.0, 0.0])
    
    traj = generate_dubins_like_trajectory(start_pos, start_yaw, target_pos, target_yaw, num_points=50)
    
    plt.figure(figsize=(10, 6))
    for i in range(traj.shape[0]):
        x = traj[i, :, 0].numpy()
        y = traj[i, :, 1].numpy()
        plt.plot(x, y, label=f"Traj {i}")
        # 画起点和终点
        plt.arrow(x[0], y[0], np.cos(start_yaw[i].item())*0.5, np.sin(start_yaw[i].item())*0.5, head_width=0.1, color='r')
        plt.arrow(x[-1], y[-1], np.cos(target_yaw[i].item())*0.5, np.sin(target_yaw[i].item())*0.5, head_width=0.1, color='g')
        
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.title("Cubic Hermite Spline (Clothoid Approximation)")
    plt.savefig("clothoid_hermite.png")
    print("Saved to clothoid_hermite.png")
