import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_clothoid_trajectory(start_pos, start_yaw, target_pos, target_yaw, num_points=32, device="cpu"):
    """
    生成从 start_pos 到 target_pos 的 Clothoid (回旋曲线) 近似轨迹。
    
    参数:
        start_pos: [N, 2] 起点坐标 (x, y)
        start_yaw: [N] 起点偏航角 (rad)
        target_pos: [N, 2] 终点坐标 (x, y)
        target_yaw: [N] 终点偏航角 (rad)
    
    返回:
        trajectory: [N, num_points, 3] 轨迹点 (x, y, yaw)
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
    
    # 确保 local_yaw 在 [-pi, pi] 之间
    local_yaw = torch.atan2(torch.sin(local_yaw), torch.cos(local_yaw))
    
    # 我们需要构造一条曲线，从 (local_x, local_y, local_yaw) 到 (0, 0, 0)
    # 论文提到 "approximation of a clothoid curve"。
    # 真正的 Clothoid 涉及 Fresnel 积分，计算代价高。
    # 我们可以使用多项式螺旋线（Polynomial Spiral）或三次样条来近似，
    # 关键是保证曲率（yaw 的导数）是连续且线性变化的。
    
    # 假设 y(x) 是一个多项式。为了满足边界条件：
    # y(0) = 0, y'(0) = 0 (因为终点在原点且朝向 x 轴)
    # y(local_x) = local_y, y'(local_x) = tan(local_yaw)
    
    # 设 y(x) = a*x^3 + b*x^2
    # y'(x) = 3*a*x^2 + 2*b*x
    # 代入起点条件：
    # a*x0^3 + b*x0^2 = y0
    # 3*a*x0^2 + 2*b*x0 = tan(yaw0)
    
    x0 = local_x
    y0 = local_y
    tan_yaw0 = torch.tan(local_yaw)
    
    # 求解 a, b
    # [x0^3, x0^2] [a] = [y0]
    # [3x0^2, 2x0] [b] = [tan_yaw0]
    # 行列式 D = 2x0^4 - 3x0^4 = -x0^4
    
    # 为了避免 x0 接近 0 时除零，加一个极小值
    x0_safe = torch.where(torch.abs(x0) < 1e-3, torch.sign(x0) * 1e-3 + 1e-5, x0)
    
    a = (2 * y0 - x0_safe * tan_yaw0) / (x0_safe ** 3)
    b = (x0_safe * tan_yaw0 - 3 * y0) / (x0_safe ** 2)
    
    # 生成插值点
    t = torch.linspace(1.0, 0.0, num_points, device=device).view(1, -1).expand(N, -1)
    
    # x 坐标线性插值
    traj_x_local = x0.unsqueeze(1) * t
    
    # y 坐标根据多项式计算
    traj_y_local = a.unsqueeze(1) * (traj_x_local ** 3) + b.unsqueeze(1) * (traj_x_local ** 2)
    
    # 计算偏航角 (dy/dx)
    dy_dx = 3 * a.unsqueeze(1) * (traj_x_local ** 2) + 2 * b.unsqueeze(1) * traj_x_local
    traj_yaw_local = torch.atan(dy_dx)
    
    # 转换回全局坐标系
    cos_inv = torch.cos(target_yaw).unsqueeze(1)
    sin_inv = torch.sin(target_yaw).unsqueeze(1)
    
    traj_x_global = target_pos[:, 0].unsqueeze(1) + traj_x_local * cos_inv - traj_y_local * sin_inv
    traj_y_global = target_pos[:, 1].unsqueeze(1) + traj_x_local * sin_inv + traj_y_local * cos_inv
    traj_yaw_global = traj_yaw_local + target_yaw.unsqueeze(1)
    
    # 确保 yaw 在 [-pi, pi]
    traj_yaw_global = torch.atan2(torch.sin(traj_yaw_global), torch.cos(traj_yaw_global))
    
    trajectory = torch.stack([traj_x_global, traj_y_global, traj_yaw_global], dim=-1)
    return trajectory

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 测试用例
    start_pos = torch.tensor([[-4.0, 1.0], [-4.0, -1.0], [-3.0, 0.0]])
    start_yaw = torch.tensor([0.0, 0.0, 0.5])
    target_pos = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    target_yaw = torch.tensor([0.0, 0.0, 0.0])
    
    traj = generate_clothoid_trajectory(start_pos, start_yaw, target_pos, target_yaw, num_points=50)
    
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
    plt.title("Polynomial Spiral (Clothoid Approximation)")
    plt.savefig("clothoid_approx.png")
    print("Saved to clothoid_approx.png")
