import time
import rs
from primitives import MIN_TURN_RADIUS, RS_GOAL_X, RS_GOAL_Y, RS_GOAL_TH, DT

# RS 解析终点扩展：离目标的 RS 距离小于此值时，尝试直接用 RS 曲线一杆进洞
RS_EXPANSION_RADIUS = 0.8  # meters

def plan_path_pure_rs(x0, y0, theta0, collision_fn=None, stats=None):
    """
    纯 RS 模式规划（用于调试和三级瀑布的第一级）：
    直接从起点计算一条前往目标的 RS 曲线，并进行碰撞检测。
    如果全段无碰撞则返回成功，否则失败。
    """
    t_start = time.perf_counter()
    
    # 获取纯 RS 路径
    rs_traj = rs.rs_sample_path(
        x0, y0, theta0,
        RS_GOAL_X, RS_GOAL_Y, RS_GOAL_TH,
        MIN_TURN_RADIUS, step=DT * 0.5
    )
    
    if stats is not None:
        stats['use_rs'] = True
        stats['expanded'] = 0
        stats['two_stage'] = False
        stats['pure_rs'] = True

    if not rs_traj:
        if stats is not None:
            stats['elapsed_ms'] = round((time.perf_counter() - t_start) * 1000.0, 1)
        return False, [], None
        
    # 碰撞检测（如果有 collision_fn）
    ok = True
    if collision_fn is not None:
        for pt in rs_traj:
            valid, _ = collision_fn(pt[0], pt[1], pt[2])
            if not valid:
                ok = False
                break
    
    if stats is not None:
        stats['elapsed_ms'] = round((time.perf_counter() - t_start) * 1000.0, 1)
        
    if ok:
        # 为了与统一接口兼容，返回空 acts，以及 rs_traj 即可
        return True, [], rs_traj
    else:
        return False, [], rs_traj
