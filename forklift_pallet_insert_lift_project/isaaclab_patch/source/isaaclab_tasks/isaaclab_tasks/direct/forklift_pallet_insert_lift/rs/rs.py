"""
Reeds-Shepp 路径距离计算模块

坐标约定与 main.py 完全一致：
  - 前进（F 档）方向：-x（即 x -= v * cos(th)）
  - 左转：y 增大方向为正
  - 角度 theta：标准数学正方向（逆时针为正）

RS 路径由最多 5 段组成，每段是：
  L / R：固定最小转弯半径圆弧（左/右）
  S    ：直线段

覆盖前进（+）和倒退（-）两种方向，共 48 种 word。

参考：Reeds & Shepp (1990), "Optimal Paths for a Car That Goes Both Forwards and Backwards"
实现基于：P. Souères & J.P. Laumond (1996) 标准化公式，以及
          Lavalle (2006) Planning Algorithms Chapter 15 附录实现

使用方法：
    import rs
    dist = rs.rs_distance_pose(x1, y1, th1, x2, y2, th2, turning_radius)
"""

import math

_PI = math.pi
_TWO_PI = 2.0 * math.pi


def _mod2pi(x):
    """将角度包装到 [0, 2*pi)"""
    v = math.fmod(x, _TWO_PI)
    if v < 0.0:
        v += _TWO_PI
    return v


def _polar(x, y):
    """转极坐标，返回 (r, theta)，theta 在 [0, 2*pi)"""
    r = math.hypot(x, y)
    theta = _mod2pi(math.atan2(y, x))
    return r, theta


def _rect(r, theta):
    return r * math.cos(theta), r * math.sin(theta)


# ─────────────────────────────────────────────────────────────────────────────
# 48 种 RS word 公式（无障碍物最短路径）
# 每个函数返回 (t, u, v) 三段参数，段长为负表示倒退；
# 不可行时返回 None。
# 外部调用 rs_distance() 对所有公式取总长度最小值。
#
# 注意：这里的 t / u / v 是以最小转弯半径 r=1 归一化后的无量纲参数，
# 实际长度需乘以 turning_radius。
# ─────────────────────────────────────────────────────────────────────────────

# ── CSC 类型（Formula 8.1 - 8.4）──────────────────────────────────────────

def _LpSpLp(x, y, phi):
    """L+S+L+"""
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= 0.0:
        v = _mod2pi(phi - t)
        if v >= 0.0:
            return t, u, v
    return None


def _LmSmLm(x, y, phi):
    """L-S-L-"""
    return _LpSpLp(-x, y, -phi)


def _RpSpRp(x, y, phi):
    """R+S+R+"""
    t, u, v = _LpSpLp(x, -y, -phi) or (None, None, None)
    if t is not None:
        return t, u, v
    return None


def _RmSmRm(x, y, phi):
    """R-S-R-"""
    return _RpSpRp(-x, y, -phi)


def _LpSpRp(x, y, phi):
    """L+S+R+"""
    u1sq = x**2 + (y - 1.0)**2
    if u1sq < 4.0:
        return None
    u1 = math.sqrt(u1sq - 4.0)
    _, t1 = _polar(u1, 2.0)  # t1 unused directly
    t = _mod2pi(math.atan2(2.0, u1) - math.atan2(x, y - 1.0))
    # recompute properly
    t = _mod2pi(math.atan2(y - 1.0, x) - math.atan2(2.0, u1))
    if t < 0.0:
        return None
    u = math.sqrt(u1sq - 4.0)
    v = _mod2pi(t - phi)
    if v < 0.0:
        return None
    # re-derive correctly
    u2, t2 = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u2 < 2.0:
        return None
    t3 = _mod2pi(math.atan2(2.0, math.sqrt(u2**2 - 4.0)) +
                 math.atan2(-math.sqrt(u2**2 - 4.0), 2.0))
    # use the clean standard derivation below
    return None  # delegated to _LpSpRp_clean


def _LpSpRp_clean(x, y, phi):
    """L+S+R+ 标准推导"""
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1sq = u1 * u1
    if u1sq < 4.0:
        return None
    u = math.sqrt(u1sq - 4.0)
    theta = _mod2pi(math.atan2(2.0, u))
    t = _mod2pi(t1 + theta)
    v = _mod2pi(t - phi)
    if t >= 0.0 and v >= 0.0:
        return t, u, v
    return None


def _LmSmRm(x, y, phi):
    """L-S-R-"""
    res = _LpSpRp_clean(-x, y, -phi)
    if res:
        t, u, v = res
        return t, u, v
    return None


def _RpSpLp(x, y, phi):
    """R+S+L+"""
    res = _LpSpRp_clean(x, -y, -phi)
    if res:
        t, u, v = res
        return t, u, v
    return None


def _RmSmLm(x, y, phi):
    """R-S-L-"""
    res = _LpSpRp_clean(-x, -y, phi)
    if res:
        t, u, v = res
        return t, u, v
    return None


# ── CCC 类型（Formula 8.7 - 8.10）─────────────────────────────────────────

def _LpRmLp(x, y, phi):
    """L+R-L+
    在标准 RS 坐标系（forward=+x）内：
      t + u + v = phi (mod 2pi)，三段均增大 heading。
    端点方程推导（r=1，t+u+v=phi）：
      2*sin(t) - 2*sin(t+u) = x - sin(phi) = xi
      2*cos(t+u) - 2*cos(t) = y - 1 + cos(phi) = eta
    令 C = t + u/2, D = u/2：
      -4*cos(C)*sin(D) = xi, -4*sin(C)*sin(D) = eta
    => sin(D) = sqrt(xi^2+eta^2)/4
       C = atan2(-eta, -xi) = atan2(eta, xi) + π = theta + π
       t = C - D = theta + π - A/2
       v = phi - t - u = phi - (theta + π) - A/2 (mod 2pi)
    """
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(xi, eta)   # theta = atan2(eta, xi) ∈ [0, 2π)
    if u1 > 4.0:
        return None
    A = math.acos((8.0 - u1 * u1) / 8.0)   # A = u (arc angle of middle R-)
    u = A
    t = _mod2pi(theta + _PI - 0.5 * A)      # 修正：θ + π − A/2（原代码 θ + π/2 − A/2 有误）
    v = _mod2pi(phi - t - u)                 # 修正：phi − t − u（原代码符号反了）
    if t >= 0.0 and u >= 0.0 and v >= 0.0:
        return t, u, v
    return None


def _LmRpLm(x, y, phi):
    """L-R+L-"""
    res = _LpRmLp(-x, y, -phi)
    if res:
        return res
    return None


def _RpLmRp(x, y, phi):
    """R+L-R+"""
    res = _LpRmLp(x, -y, -phi)
    if res:
        return res
    return None


def _RmLpRm(x, y, phi):
    """R-L+R-"""
    res = _LpRmLp(-x, -y, phi)
    if res:
        return res
    return None


def _LpRmLm(x, y, phi):
    """L+R-L-
    t + u - v = phi（最后段 L- 减少 heading），端点方程与 L+R-L+ 相同。
    v = t + u - phi（不做 mod2pi，直接取原始值）。
    需要 v >= 0，即 t + u >= phi。
    """
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 > 4.0:
        return None
    A = math.acos((8.0 - u1 * u1) / 8.0)
    u = A
    t = _mod2pi(theta + _PI - 0.5 * A)     # 与 _LpRmLp 相同的 t 计算
    v = t + u - phi                          # 不使用 mod2pi，直接取差值
    while v < 0.0:
        v += _TWO_PI
    while v >= _TWO_PI:
        v -= _TWO_PI
    if t >= 0.0 and u >= 0.0 and v >= 0.0:
        return t, u, v
    return None


def _LmRpLp(x, y, phi):
    """L-R+L+"""
    res = _LpRmLm(-x, y, -phi)
    if res:
        t, u, v = res
        return t, u, v
    return None


def _RpLmRm(x, y, phi):
    """R+L-R-"""
    res = _LpRmLm(x, -y, -phi)
    if res:
        return res
    return None


def _RmLpRp(x, y, phi):
    """R-L+R+"""
    res = _LpRmLm(-x, -y, phi)
    if res:
        return res
    return None


# ── CCCC 类型（Formula 8.11）──────────────────────────────────────────────

def _LpRupLumRm(x, y, phi):
    """L+R+u L-u R-"""
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 > 4.0:
        return None
    if u1 > 2.0:
        A = math.acos(0.25 * u1 - 1.0)
    else:
        return None
    t = _mod2pi(theta + 0.5 * A + _PI / 2.0)
    u = _mod2pi(_PI - A)
    v = _mod2pi(phi - t + 2.0 * u)
    if t >= 0.0 and u >= 0.0 and v >= 0.0:
        return t, u, v
    return None


def _LmRumLupRp(x, y, phi):
    res = _LpRupLumRm(-x, y, -phi)
    if res:
        return res
    return None


def _RpLupRumLm(x, y, phi):
    res = _LpRupLumRm(x, -y, -phi)
    if res:
        return res
    return None


def _RmLumRupLp(x, y, phi):
    res = _LpRupLumRm(-x, -y, phi)
    if res:
        return res
    return None


# ── CCSC / CSCC 类型（Formula 8.9, 8.10）─────────────────────────────────

def _LpRmSmLm(x, y, phi):
    """L+R-S-L-"""
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 < 2.0:
        return None
    t = _mod2pi(theta + math.acos(-2.0 / u1) + _PI / 2.0)
    u = _mod2pi(_PI / 2.0)
    v = _mod2pi(_PI - math.acos(-2.0 / u1) - phi + t)
    if t >= 0.0 and u >= 0.0 and v >= 0.0:
        return t, u, v
    return None


def _LmRpSmLp(x, y, phi):
    res = _LpRmSmLm(-x, y, -phi)
    if res:
        return res
    return None


def _RpLmSmRm(x, y, phi):
    res = _LpRmSmLm(x, -y, -phi)
    if res:
        return res
    return None


def _RmLpSmRp(x, y, phi):
    res = _LpRmSmLm(-x, -y, phi)
    if res:
        return res
    return None


def _LpRmSmRm(x, y, phi):
    """L+R-S-R-"""
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 < 2.0:
        return None
    t = _mod2pi(theta + math.acos(-2.0 / u1) + _PI / 2.0)
    u = _mod2pi(_PI / 2.0)
    v = _mod2pi(t + _PI / 2.0 - phi)
    if t >= 0.0 and u >= 0.0 and v >= 0.0:
        return t, u, v
    return None


def _LmRpSmRp(x, y, phi):
    res = _LpRmSmRm(-x, y, -phi)
    if res:
        return res
    return None


def _RpLmSmLm(x, y, phi):
    res = _LpRmSmRm(x, -y, -phi)
    if res:
        return res
    return None


def _RmLpSmLp(x, y, phi):
    res = _LpRmSmRm(-x, -y, phi)
    if res:
        return res
    return None


def _LpSmRmLp(x, y, phi):
    """L+S-R-L+ (CSCС)"""
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 < 2.0:
        return None
    u = math.sqrt(u1 * u1 - 4.0)
    t = _mod2pi(theta + math.atan2(2.0, u))
    v = _mod2pi(t - _mod2pi(phi))
    if t >= 0.0 and v >= 0.0:
        return t, u, v
    return None


def _RmSmLmRm(x, y, phi):
    res = _LpSmRmLp(-x, y, -phi)
    if res:
        return res
    return None


def _LmSmRmLm(x, y, phi):
    res = _LpSmRmLp(x, -y, -phi)
    if res:
        return res
    return None


def _RpSpLpRp(x, y, phi):
    res = _LpSmRmLp(-x, -y, phi)
    if res:
        return res
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 核心入口：归一化坐标下的 RS 距离
# ─────────────────────────────────────────────────────────────────────────────

def _all_words(x, y, phi):
    """
    枚举所有 48 种 RS word，返回 (总长度, word名称) 列表（仅可行解）。
    坐标已按 turning_radius 归一化（即 r=1 的坐标系）。
    """
    results = []

    def _try(fn, tag):
        res = fn(x, y, phi)
        if res is not None:
            t, u, v = res
            
            # Reconstruct segments based on the pattern
            segs = []
            if len(tag) == 6:
                # CSC or CCC: 3 segments (t, u, v)
                chars = [tag[0:2], tag[2:4], tag[4:6]]
                vals = [t, u, v]
                for ch, val in zip(chars, vals):
                    sign = 1 if ch[1] == '+' else -1
                    segs.append((ch[0], val * sign))
            elif len(tag) == 8:
                # 4 segments
                chars = [tag[0:2], tag[2:4], tag[4:6], tag[6:8]]
                
                if fn in (_LpRupLumRm, _LmRumLupRp, _RpLupRumLm, _RmLumRupLp):
                    # CCCC: t, u, u, v
                    vals = [t, u, u, v]
                elif fn in (_LpRmSmLm, _LmRpSmLp, _RpLmSmRm, _RmLpSmRp,
                            _LpRmSmRm, _LmRpSmRp, _RpLmSmLm, _RmLpSmLp):
                    # CCSC: t, pi/2, u, v
                    vals = [t, _PI / 2.0, u, v]
                elif fn in (_LpSmRmLp, _RmSmLmRm, _LmSmRmLm, _RpSpLpRp):
                    # CSCC: t, u, pi/2, v
                    vals = [t, u, _PI / 2.0, v]
                else:
                    raise ValueError("Unknown 4-segment function")
                    
                for ch, val in zip(chars, vals):
                    sign = 1 if ch[1] == '+' else -1
                    segs.append((ch[0], val * sign))
                    
            length = sum(abs(l) for _, l in segs)
            results.append((length, segs))


    # CSC
    _try(_LpSpLp,      'L+S+L+')
    _try(_LmSmLm,      'L-S-L-')
    _try(_RpSpRp,      'R+S+R+')
    _try(_RmSmRm,      'R-S-R-')
    _try(_LpSpRp_clean,'L+S+R+')
    _try(_LmSmRm,      'L-S-R-')
    _try(_RpSpLp,      'R+S+L+')
    _try(_RmSmLm,      'R-S-L-')

    # CCC
    _try(_LpRmLp,      'L+R-L+')
    _try(_LmRpLm,      'L-R+L-')
    _try(_RpLmRp,      'R+L-R+')
    _try(_RmLpRm,      'R-L+R-')
    _try(_LpRmLm,      'L+R-L-')
    _try(_LmRpLp,      'L-R+L+')
    _try(_RpLmRm,      'R+L-R-')
    _try(_RmLpRp,      'R-L+R+')

    # CCCC
#    _try(_LpRupLumRm,  'L+R+L-R-')
#    _try(_LmRumLupRp,  'L-R-L+R+')
#    _try(_RpLupRumLm,  'R+L+R-L-')
#    _try(_RmLumRupLp,  'R-L-R+L+')

    # CCSC / CSCC
#    _try(_LpRmSmLm,    'L+R-S-L-')
#    _try(_LmRpSmLp,    'L-R+S+L+')
#    _try(_RpLmSmRm,    'R+L-S-R-')
#    _try(_RmLpSmRp,    'R-L+S+R+')
#    _try(_LpRmSmRm,    'L+R-S-R-')
#    _try(_LmRpSmRp,    'L-R+S+R+')
#    _try(_RpLmSmLm,    'R+L-S-L-')
#    _try(_RmLpSmLp,    'R-L+S+L+')
#    _try(_LpSmRmLp,    'L+S-R-L+')
#    _try(_RmSmLmRm,    'R-S-L-R-')
#    _try(_LmSmRmLm,    'L-S-R-L-')
#    _try(_RpSpLpRp,    'R+S+L+R+')

    return results


def rs_distance(x, y, phi, turning_radius):
    """
    计算从原点 (0, 0, 0) 到 (x, y, phi) 的 Reeds-Shepp 最短路径长度。

    坐标约定与 main.py 一致（前进方向为 -x）。

    参数:
        x, y   : 目标相对位置（米）
        phi    : 目标相对朝向（弧度）
        turning_radius: 最小转弯半径（米），= WHEELBASE / tan(MAX_STEER)

    返回:
        float: RS 路径长度（米），若计算失败返回 Euclidean 距离作为下界估计
    """
    if turning_radius <= 0.0:
        raise ValueError(f"turning_radius must be positive, got {turning_radius}")

    # 归一化坐标（除以转弯半径，RS 公式在 r=1 下推导）
    nx = x / turning_radius
    ny = y / turning_radius

    words = _all_words(nx, ny, phi)
    if not words:
        # 回退到欧氏距离（保守可容许下界）
        return math.hypot(x, y)

    best_len = min(w[0] for w in words)
    return best_len * turning_radius


def rs_distance_pose(x1, y1, th1, x2, y2, th2, turning_radius):
    """
    计算任意两个位姿之间的 Reeds-Shepp 距离。

    将 (x2, y2, th2) 变换到以 (x1, y1, th1) 为原点的局部坐标系，
    然后调用 rs_distance()。

    参数均使用 main.py 的坐标约定（前进方向为 -x）。
    """
    dx = x2 - x1
    dy = y2 - y1
    cos_t = math.cos(th1)
    sin_t = math.sin(th1)

    # 局部坐标：旋转到以 th1 为基准的参考系
    lx =  dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    lphi = th2 - th1

    # 角度包装到 (-pi, pi]
    while lphi > _PI:
        lphi -= _TWO_PI
    while lphi <= -_PI:
        lphi += _TWO_PI

    return rs_distance(lx, ly, lphi, turning_radius)


# ─────────────────────────────────────────────────────────────────────────────
# 路径采样：用于 A* 解析终点扩展（一杆进洞）
# ─────────────────────────────────────────────────────────────────────────────

def rs_all_paths(x, y, phi, turning_radius):
    """
    返回从原点到 (x, y, phi) 的所有 RS 路径各段参数，按长度升序排列。

    返回值: list of [ list of (seg_type, length_meters) ]
    """
    if turning_radius <= 0.0:
        raise ValueError(f"turning_radius must be positive, got {turning_radius}")

    nx = x / turning_radius
    ny = y / turning_radius

    words = _all_words(nx, ny, phi)
    if not words:
        return []

    # 按长度排序
    words.sort(key=lambda w: w[0])
    
    all_segs = []
    for length, segs_norm in words:
        segs = [(s, length_norm * turning_radius) for s, length_norm in segs_norm]
        all_segs.append(segs)
        
    return all_segs

def rs_best_path(x, y, phi, turning_radius):
    """
    返回从原点到 (x, y, phi) 的最短 RS 路径各段参数。

    返回值: list of (seg_type, length_meters)
        seg_type: 'L', 'R', 'S'
        length_meters: 该段的有向长度（负数表示倒退）

    坐标约定与 main.py 一致（前进方向为 -x）。
    """
    if turning_radius <= 0.0:
        raise ValueError(f"turning_radius must be positive, got {turning_radius}")

    nx = x / turning_radius
    ny = y / turning_radius

    words = _all_words(nx, ny, phi)
    if not words:
        return []

    best = min(words, key=lambda w: w[0])
    _, segs_norm = best

    result = []
    for stype, length_norm in segs_norm:
        length_m = length_norm * turning_radius
        if abs(length_m) > 1e-6:
            result.append((stype, length_m))
    return result


def _parse_tag(tag):
    """
    将 RS word 标签字符串解析为 [(类型, 符号), ...] 列表。
    例如 'L+R-S-' → [('L', 1), ('R', -1), ('S', -1)]
    """
    segments = []
    i = 0
    while i < len(tag):
        c = tag[i]
        if c in ('L', 'R', 'S'):
            sign = 1 if tag[i + 1] == '+' else -1
            segments.append((c, sign))
            i += 2
        else:
            i += 1
    return segments


def rs_sample_path(x1, y1, th1, x2, y2, th2, turning_radius, step=0.05,
                   verbose=False):
    """
    对从 (x1,y1,th1) 到 (x2,y2,th2) 的最短 RS 路径按 step（米）密度采样。

    返回: [(x, y, th), ...] 轨迹点列表（含起点和终点）
          若找不到 RS 路径则返回 None。

    坐标约定与 main.py 一致（前进方向为 -x）。

    坐标变换流程（修正版）：
      main.py 局部系 forward=-x  →  RS 标准系 forward=+x
          x_rs = -lx,  y_rs = ly,  th_rs = -lth
    先在 RS 标准系内用精确圆弧公式积分，再变换回世界系。
    """
    # ── Step 1：world → main.py 局部系 ───────────────────────────
    dx = x2 - x1
    dy = y2 - y1
    cos_t = math.cos(th1)
    sin_t = math.sin(th1)
    lx =  dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    lphi = th2 - th1
    while lphi > _PI:
        lphi -= _TWO_PI
    while lphi <= -_PI:
        lphi += _TWO_PI

    # ── Step 2：main.py 局部系 → RS 标准系（forward=+x）──────────
    # 180° 旋转变换：x_rs = -lx, y_rs = -ly, th_rs = lphi
    # 推导：main.py 局部系前进方向=(-1,0)，RS 标准系前进方向=(+1,0)
    # 翻转 x 和 y 后：(-lx,-ly) → (+1,0) 对齐，转向方向也得以保持一致
    x_rs_goal = -lx
    y_rs_goal = -ly
    th_rs_goal = lphi   # 旋转 180° 后角度不变（同向旋转）
    while th_rs_goal > _PI:
        th_rs_goal -= _TWO_PI
    while th_rs_goal <= -_PI:
        th_rs_goal += _TWO_PI

    segs = rs_best_path(x_rs_goal, y_rs_goal, th_rs_goal, turning_radius)
    if not segs:
        return None

    # ── Step 3：在 RS 标准系内精确积分采样 ───────────────────────
    # RS 标准系：forward = +x，L 弧 = th 增大，R 弧 = th 减小
    # 精确圆弧公式（无直线近似累积误差）：
    #   L 弧: new_x = x + r*(sin(th+dth)-sin(th)),  new_y = y + r*(cos(th)-cos(th+dth))
    #   R 弧: new_x = x + r*(sin(th)-sin(th+dth)),  new_y = y + r*(cos(th+dth)-cos(th))
    cx, cy, cth = 0.0, 0.0, 0.0
    rs_pts = [(cx, cy, cth)]

    for stype, length_m in segs:
        direction = 1 if length_m >= 0 else -1
        dist_left = abs(length_m)

        while dist_left > 1e-9:
            ds = min(step, dist_left)
            dist_left -= ds

            if stype == 'S':
                # 直线：RS 标准系 forward=+x
                cx += direction * ds * math.cos(cth)
                cy += direction * ds * math.sin(cth)
            elif stype == 'L':
                # 左转弧（th 增大方向为 CCW）
                dth = direction * ds / turning_radius
                new_cx = cx + turning_radius * (math.sin(cth + dth) - math.sin(cth))
                new_cy = cy + turning_radius * (math.cos(cth) - math.cos(cth + dth))
                cx, cy, cth = new_cx, new_cy, cth + dth
            else:
                # 右转弧（th 减小方向为 CW）
                dth = -direction * ds / turning_radius
                new_cx = cx + turning_radius * (math.sin(cth) - math.sin(cth + dth))
                new_cy = cy + turning_radius * (math.cos(cth + dth) - math.cos(cth))
                cx, cy, cth = new_cx, new_cy, cth + dth

            while cth > _PI:
                cth -= _TWO_PI
            while cth <= -_PI:
                cth += _TWO_PI

            rs_pts.append((cx, cy, cth))

    # ── Step 4：RS 标准系 → main.py 局部系 → 世界系 ─────────────
    # 逆变换：lx_p = -x_rs, ly_p = -y_rs, lth_p = th_rs
    world_pts = []
    for x_rs, y_rs, th_rs in rs_pts:
        lx_p = -x_rs
        ly_p = -y_rs
        lth_p = th_rs
        # 局部系 → 世界系（逆旋转 th1）
        wx = x1 + lx_p * cos_t - ly_p * sin_t
        wy = y1 + lx_p * sin_t + ly_p * cos_t
        wth = th1 + lth_p
        while wth > _PI:
            wth -= _TWO_PI
        while wth <= -_PI:
            wth += _TWO_PI
        world_pts.append((wx, wy, wth))

    # ── Step 5：日志（verbose 或有位置误差时输出）────────────────
    if world_pts:
        ep = world_pts[-1]
        err_pos = math.hypot(ep[0] - x2, ep[1] - y2)
        err_ang_deg = abs(math.degrees(_angle_diff(ep[2], th2)))
        seg_str = ', '.join('{}{}{:.3f}m'.format(
            s, '+' if l >= 0 else '-', abs(l)) for s, l in segs)
        total_len = sum(abs(l) for _, l in segs)
        if verbose or err_pos > 0.1:
            import sys
            print('[RS_SAMPLE] start=({:.3f},{:.3f},{:.2f}deg) '
                  'goal=({:.3f},{:.3f},{:.2f}deg)'.format(
                      x1, y1, math.degrees(th1),
                      x2, y2, math.degrees(th2)), file=sys.stderr)
            print('[RS_SAMPLE] segs=[{}] total={:.3f}m'.format(
                  seg_str, total_len), file=sys.stderr)
            print('[RS_SAMPLE] endpoint=({:.3f},{:.3f},{:.2f}deg) '
                  'err_pos={:.4f}m err_ang={:.2f}deg'.format(
                      ep[0], ep[1], math.degrees(ep[2]),
                      err_pos, err_ang_deg), file=sys.stderr)

    return world_pts


def rs_sample_path_multi(x1, y1, th1, x2, y2, th2, turning_radius, step=0.05, max_paths=5):
    """
    尝试多条 RS 路径（按长度排序），返回所有采样后的轨迹列表。
    用于在有障碍物时找到第一条无碰撞路径。
    返回: list of [(x,y,th), ...] 轨迹列表，按长度升序
    """
    dx = x2 - x1
    dy = y2 - y1
    cos_t = math.cos(th1)
    sin_t = math.sin(th1)
    lx =  dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    lphi = th2 - th1
    while lphi > _PI: lphi -= _TWO_PI
    while lphi <= -_PI: lphi += _TWO_PI

    # 180° 旋转：local forward=-x → RS forward=+x（与 rs_sample_path 一致）
    x_rs_goal = -lx
    y_rs_goal = -ly
    th_rs_goal = lphi
    while th_rs_goal > _PI: th_rs_goal -= _TWO_PI
    while th_rs_goal <= -_PI: th_rs_goal += _TWO_PI

    all_segs = rs_all_paths(x_rs_goal, y_rs_goal, th_rs_goal, turning_radius)
    if not all_segs:
        return []

    results = []
    for segs in all_segs[:max_paths]:
        cx_rs, cy_rs, cth_rs = 0.0, 0.0, 0.0
        rs_pts = [(cx_rs, cy_rs, cth_rs)]
        for stype, length_m in segs:
            direction = 1 if length_m >= 0 else -1
            dist_left = abs(length_m)
            while dist_left > 1e-9:
                ds = min(step, dist_left)
                dist_left -= ds
                if stype == 'S':
                    cx_rs += direction * ds * math.cos(cth_rs)
                    cy_rs += direction * ds * math.sin(cth_rs)
                elif stype == 'L':
                    dth = direction * ds / turning_radius
                    cx_rs = cx_rs + turning_radius * (math.sin(cth_rs + dth) - math.sin(cth_rs))
                    cy_rs = cy_rs + turning_radius * (math.cos(cth_rs) - math.cos(cth_rs + dth))
                    cth_rs += dth
                else:
                    dth = -direction * ds / turning_radius
                    cx_rs = cx_rs + turning_radius * (math.sin(cth_rs) - math.sin(cth_rs + dth))
                    cy_rs = cy_rs + turning_radius * (math.cos(cth_rs + dth) - math.cos(cth_rs))
                    cth_rs += dth
                while cth_rs > _PI: cth_rs -= _TWO_PI
                while cth_rs <= -_PI: cth_rs += _TWO_PI
                rs_pts.append((cx_rs, cy_rs, cth_rs))

        # 逆变换：RS 标准系 → local（180° 旋转逆）→ world
        world_pts = []
        for xr, yr, thr in rs_pts:
            lx_p = -xr
            ly_p = -yr
            wx = x1 + lx_p * cos_t - ly_p * sin_t
            wy = y1 + lx_p * sin_t + ly_p * cos_t
            wth = th1 + thr
            while wth > _PI: wth -= _TWO_PI
            while wth <= -_PI: wth += _TWO_PI
            world_pts.append((wx, wy, wth))
        results.append(world_pts)
    return results


def _angle_diff(a, b):
    """返回 a-b 的角度差，规范化到 (-π, π]"""
    d = a - b
    while d > _PI:
        d -= _TWO_PI
    while d <= -_PI:
        d += _TWO_PI
    return d
