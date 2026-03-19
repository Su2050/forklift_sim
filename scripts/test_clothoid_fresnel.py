import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import fresnel

def generate_true_clothoid_scipy(x0, y0, yaw0, x1, y1, yaw1, num_points=50):
    """
    使用 scipy 的 fresnel 积分生成真正的 Clothoid 曲线。
    注意：这只是为了可视化对比，无法在 PyTorch 中直接批处理加速。
    """
    # 这里只是一个简化的演示，真正的两点边界值 Clothoid 求解非常复杂（需要迭代求解非线性方程组）。
    # 我们这里只画一个标准的 Clothoid 螺旋线，看看它的曲率变化特点。
    
    t = np.linspace(-5, 5, num_points)
    y, x = fresnel(t)
    
    plt.figure(figsize=(8, 8))
    plt.plot(x, y)
    plt.title("Standard Clothoid (Euler Spiral)")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("clothoid_true.png")
    print("Saved to clothoid_true.png")

if __name__ == "__main__":
    generate_true_clothoid_scipy(0,0,0,0,0,0)
