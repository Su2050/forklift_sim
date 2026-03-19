import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_bezier(p0, yaw0, p_goal, yaw_goal, num_points=50):
    N = p0.shape[0]
    h0 = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], dim=-1)
    u_in = torch.stack([torch.cos(yaw_goal), torch.sin(yaw_goal)], dim=-1)
    
    # 模拟我们代码里的逻辑
    B0 = p0
    B1 = p0 + 1.5 * h0
    B2 = p_goal - 1.5 * u_in
    B3 = p_goal
    
    t = torch.linspace(0.0, 1.0, num_points).view(1, -1, 1).expand(N, -1, 2)
    t_inv = 1.0 - t
    
    B0_exp = B0.unsqueeze(1)
    B1_exp = B1.unsqueeze(1)
    B2_exp = B2.unsqueeze(1)
    B3_exp = B3.unsqueeze(1)
    
    traj = (t_inv**3)*B0_exp + 3*(t_inv**2)*t*B1_exp + 3*t_inv*(t**2)*B2_exp + (t**3)*B3_exp
    
    # 计算导数
    dt = 3*(t_inv**2)*(B1_exp - B0_exp) + 6*t_inv*t*(B2_exp - B1_exp) + 3*(t**2)*(B3_exp - B2_exp)
    yaw = torch.atan2(dt[..., 1], dt[..., 0])
    
    return torch.cat([traj, yaw.unsqueeze(-1)], dim=-1)

def generate_hermite(p0, yaw0, p_goal, yaw_goal, num_points=50):
    N = p0.shape[0]
    t0 = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], dim=-1)
    t_goal = torch.stack([torch.cos(yaw_goal), torch.sin(yaw_goal)], dim=-1)
    
    dist = torch.norm(p_goal - p0, dim=-1, keepdim=True)
    L = dist * 1.5
    
    m0 = t0 * L
    m1 = t_goal * L
    
    t = torch.linspace(0.0, 1.0, num_points).view(1, -1, 1).expand(N, -1, 2)
    t2 = t ** 2
    t3 = t ** 3
    
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    
    p0_exp = p0.unsqueeze(1)
    m0_exp = m0.unsqueeze(1)
    p1_exp = p_goal.unsqueeze(1)
    m1_exp = m1.unsqueeze(1)
    
    traj = h00 * p0_exp + h10 * m0_exp + h01 * p1_exp + h11 * m1_exp
    
    dh00 = 6*t2 - 6*t
    dh10 = 3*t2 - 4*t + 1
    dh01 = -6*t2 + 6*t
    dh11 = 3*t2 - 2*t
    
    dt = dh00 * p0_exp + dh10 * m0_exp + dh01 * p1_exp + dh11 * m1_exp
    yaw = torch.atan2(dt[..., 1], dt[..., 0])
    
    return torch.cat([traj, yaw.unsqueeze(-1)], dim=-1)

if __name__ == "__main__":
    p0 = torch.tensor([[-4.0, 1.5]])
    yaw0 = torch.tensor([0.0])
    p_goal = torch.tensor([[0.0, 0.0]])
    yaw_goal = torch.tensor([0.0])
    
    traj_b = generate_bezier(p0, yaw0, p_goal, yaw_goal)
    traj_h = generate_hermite(p0, yaw0, p_goal, yaw_goal)
    
    plt.figure(figsize=(12, 5))
    
    # 轨迹图
    plt.subplot(1, 2, 1)
    plt.plot(traj_b[0, :, 0].numpy(), traj_b[0, :, 1].numpy(), label="Bezier")
    plt.plot(traj_h[0, :, 0].numpy(), traj_h[0, :, 1].numpy(), label="Hermite (Clothoid Approx)")
    plt.arrow(p0[0,0].item(), p0[0,1].item(), np.cos(yaw0[0].item())*0.5, np.sin(yaw0[0].item())*0.5, head_width=0.1, color='k')
    plt.arrow(p_goal[0,0].item(), p_goal[0,1].item(), np.cos(yaw_goal[0].item())*0.5, np.sin(yaw_goal[0].item())*0.5, head_width=0.1, color='k')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.title("Trajectory Path")
    
    # 曲率变化图 (Yaw 的导数)
    plt.subplot(1, 2, 2)
    yaw_b = traj_b[0, :, 2].numpy()
    yaw_h = traj_h[0, :, 2].numpy()
    
    # 计算曲率 (d_yaw / ds)
    ds_b = np.linalg.norm(np.diff(traj_b[0, :, :2].numpy(), axis=0), axis=1)
    ds_h = np.linalg.norm(np.diff(traj_h[0, :, :2].numpy(), axis=0), axis=1)
    
    curv_b = np.diff(yaw_b) / ds_b
    curv_h = np.diff(yaw_h) / ds_h
    
    plt.plot(curv_b, label="Bezier Curvature")
    plt.plot(curv_h, label="Hermite Curvature")
    plt.grid(True)
    plt.legend()
    plt.title("Curvature Change (Linear is better for Clothoid)")
    
    plt.savefig("compare_curves.png")
    print("Saved to compare_curves.png")
