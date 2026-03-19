# 重大发现：轨迹生成的几何学 Bug

在分析了 `analyze_resnet34_similarity.py` 的结果后，我们确认了 0.5m 处的视觉特征是足够清晰的。既然视觉没问题，为什么 Agent 在 0.5m 处偏航角卡在 20° 死活插不进去？

我写了一个脚本来检查我们生成的 Clothoid (Hermite) 轨迹的数学属性，发现了一个**极其致命的几何学 Bug**！

## 1. 错误的对象跟踪 (Body Center vs Fork Center)

在 `env.py` 中，我们生成参考轨迹和查询参考轨迹的代码是这样的：
```python
# 生成轨迹的起点
p0 = self.robot.data.root_pos_w[env_ids, :2]  # 这是叉车车体中心！

# 查询轨迹的当前点
root_pos = self.robot.data.root_pos_w[:, :2]  # 这也是叉车车体中心！
```

**问题出在哪里？**
我们的轨迹目标点 `p_goal` 是**托盘前沿中心**。
如果轨迹是为“车体中心”生成的，这意味着轨迹在引导**车体中心**开向**托盘前沿**。

但是，叉尖在车体中心前方 **1.87 米** 处！
当叉尖到达托盘前沿（准备完美插入）时，车体中心实际上还在托盘前沿后方 **1.87 米** 处。

## 2. 致命的切线角度

我们的轨迹设计是：终点前 1.2m (`traj_pre_dist_m`) 是直线，1.2m 以外是曲线。
当叉尖到达托盘前沿时，车体中心在 1.87m 处。因为 $1.87m > 1.2m$，所以**此时车体中心依然处于轨迹的曲线段上**！

我用脚本计算了在 1.87m 处，轨迹的切线角度是多少：
**答案是：-17.35 度！**

## 3. 破案了：Agent 根本没有错，是我们教错了

Agent 在 0.5m 处偏航角保持在 20° 左右，**并不是因为它看不清，也不是因为它不敢插，而是因为它在完美地执行我们给它的错误指令！**

它在 0.5m 处查询轨迹时，轨迹告诉它：“你现在的车体中心在这个位置，切线角度应该是 20 度，请把车头对准 20 度！”
Agent 把车头对准了 20 度，拿到了满分的 $r_{c\psi}$ 奖励，但因为车头是歪的，它永远也插不进托盘。

## 4. 论文是如何避免这个问题的？

重新翻看论文，论文中写得清清楚楚：
> "$r_d$ and $r_{cd}$ are the distances from the **center of the forks** to the pallet and clothoid curve, respectively, $r_{c\psi}$ is the difference between the orientation of the **forks** and the tangent to the clothoid curve."

论文里的轨迹是为**叉臂中心 (center of the forks)** 生成的！

## 5. 修复方案

我们不需要引入相对位姿辅助，也不需要放弃纯视觉路线。我们只需要修复这个几何 Bug：
1. 在 `_build_reference_trajectory` 中，将起点 `p0` 从 `root_pos` 改为 `fork_center`。
2. 在 `_query_reference_trajectory` 中，将查询点从 `root_pos` 改为 `fork_center`。

这样，轨迹就会引导“叉臂中心”平滑地驶入托盘，当叉尖到达托盘前沿时，叉臂中心刚好在直线段上，切线角度完美等于 0 度！
