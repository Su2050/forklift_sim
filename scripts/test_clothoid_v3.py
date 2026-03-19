import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_hermite_trajectory(p0, yaw0, p_goal, yaw_goal, num_points=32, device="cpu"):
    """
    使用三次 Hermite 样条生成平滑轨迹，作为 Clothoid 的近似。
    这种方法能保证起点和终点的位置、切线方向连续，且计算效率高，支持批处理。
    """
    N = p0.shape[0]
    
    # 切线向量
    t0 = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], dim=-1)
    t_goal = torch.stack([torch.cos(yaw_goal), torch.sin(yaw_goal)], dim=-1)
    
    # 计算起点到终点的直线距离
    dist = torch.norm(p_goal - p0, dim=-1, keepdim=True)
    
    # 控制切线长度 (影响曲线的“肚子”大小)
    # 距离越远，切线越长，曲线越平缓
    L = dist * 1.5
    
    m0 = t0 * L
    m1 = t_goal * L
    
    # 构造三次 Hermite 样条
    t = torch.linspace(0.0, 1.0, num_points, device=device).view(1, -1, 1).expand(N, -1, 2)
    t2 = t ** 2
    t3 = t ** 3
    
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    
    p0_exp = p0.unsqueeze(1).expand(-1, num_points, -1)
    m0_exp = m0.unsqueeze(1).expand(-1, num_points, -1)
    p1_exp = p_goal.unsqueeze(1).expand(-1, num_points, -1)
    m1_exp = m1.unsqueeze(1).expand(-1, num_points, -1)
    
    # 轨迹点坐标
    traj_pos = h00 * p0_exp + h10 * m0_exp + h01 * p1_exp + h11 * m1_exp
    
    # 计算导数以获取 yaw
    dh00 = 6*t2 - 6*t
    dh10 = 3*t2 - 4*t + 1
    dh01 = -6*t2 + 6*t
    dh11 = 3*t2 - 2*t
    
    traj_vel = dh00 * p0_exp + dh10 * m0_exp + dh01 * p1_exp + dh11 * m1_exp
    
    traj_yaw = torch.atan2(traj_vel[..., 1], traj_vel[..., 0])
    
    trajectory = torch.cat([traj_pos, traj_yaw.unsqueeze(-1)], dim=-1)
    return trajectory

if __name__ == "__main__":
    # 测试用例
    p0 = torch.tensor([[-4.0, 1.0], [-4.0, -1.0], [-3.0, 2.0], [-2.0, -1.5]])
    yaw0 = torch.tensor([0.0, 0.0, -0.5, 0.5])
    p_goal = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    yaw_goal = torch.tensor([0.0, 0.0, 0.0, 0.0])
    
    traj = generate_hermite_trajectory(p0, yaw0, p_goal, yaw_goal, num_points=50)
    
    plt.figure(figsize=(10, 6))
    for i in range(traj.shape[0]):
        x = traj[i, :, 0].numpy()
        y = traj[i, :, 1].numpy()
        yaw = traj[i, :, 2].numpy()
        plt.plot(x, y, label=f"Traj {i}")
        # 画起点和终点
        plt.arrow(x[0], y[0], np.cos(yaw0[i].item())*0.5, np.sin(yaw0[i].item())*0.5, head_width=0.1, color='r')
        plt.arrow(x[-1], y[-1], np.cos(yaw_goal[i].item())*0.5, np.sin(yaw_goal[i].item())*0.5, head_width=0.1, color='g')
        
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.title("Cubic Hermite Spline (Clothoid Approximation)")
    plt.savefig("clothoid_hermite_v3.png")
    print("Saved to clothoid_hermite_v3.png")
