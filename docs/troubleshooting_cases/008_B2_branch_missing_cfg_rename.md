# B2 分支启动崩溃：env_cfg 参数重命名后 env.py 残留旧引用

**日期**: 2026-02-11
**实验**: DO-O-B2（粗+细双势函数 hold-align shaping）
**严重性**: 启动即崩（`_reset_idx` 阶段 AttributeError）

---

## 症状

B2 训练启动后在第一次 `env.reset()` 即报错退出：

```
File ".../env.py", line 1358, in _reset_idx
    torch.exp(-(y_err_reset / self.cfg.hold_align_sigma_y) ** 2)
AttributeError: 'ForkliftPalletInsertLiftEnvCfg' object has no attribute 'hold_align_sigma_y'
```

日志文件：`20260211_190335_train_s1.0o_B2_s42.log`

---

## 根因

B2 的改动是把 `env_cfg.py` 中的单一 hold-align 参数替换为粗+细双势函数参数：

```python
# 删除的旧参数
k_hold_align: float = 0.1
hold_align_sigma_y: float = 0.15      # <-- 被删除
hold_align_sigma_yaw: float = 8.0     # <-- 被删除

# 新增的参数
k_hold_align_coarse: float = 0.2
k_hold_align_fine: float = 0.15
sigma_y_coarse: float = 0.40
sigma_y_fine: float = 0.12
sigma_yaw_coarse: float = 15.0
sigma_yaw_fine: float = 6.0
```

`env.py` 中的 `_get_rewards()` 部分（主要计算逻辑）已正确替换为新参数名，但**两处隐藏引用被遗漏**：

1. **`_compute_phi_align()` 辅助函数（第 724-727 行）** — 此函数不在 `_get_rewards` 内，是一个独立的辅助方法，用于日志/诊断
2. **`_reset_idx()` 中的 phi_align 初始化（第 1357-1361 行）** — 在 reset 时初始化势函数缓存，防止"开局白嫖"

这两处代码距离主改动位置（`_get_rewards` 第 1093 行附近）相隔 300+ 行和 600+ 行，全文搜索时容易漏掉。

---

## 修复

```python
# _compute_phi_align() — 改用 coarse sigma（此函数仅用于诊断）
phi_align = (
    torch.exp(-(y_err / self.cfg.sigma_y_coarse) ** 2)
    * torch.exp(-(yaw_err_deg / self.cfg.sigma_yaw_coarse) ** 2)
)

# _reset_idx() — 初始化粗+细两个缓存
phi_coarse_init = (
    torch.exp(-(y_err_reset / self.cfg.sigma_y_coarse) ** 2)
    * torch.exp(-(yaw_err_deg_reset / self.cfg.sigma_yaw_coarse) ** 2)
)
phi_fine_init = (
    torch.exp(-(y_err_reset / self.cfg.sigma_y_fine) ** 2)
    * torch.exp(-(yaw_err_deg_reset / self.cfg.sigma_yaw_fine) ** 2)
)
self._prev_phi_coarse[env_ids] = phi_coarse_init.detach()
self._prev_phi_fine[env_ids] = phi_fine_init.detach()
```

修复 commit: `2754afd` (exp/DO-O-B2)

---

## 教训

### 1. 重命名 cfg 参数后，必须全局搜索旧名

在 `env_cfg.py` 中删除/重命名参数后，**必须在 `env.py` 全文搜索旧参数名**，确认零残留：

```bash
grep -n 'hold_align_sigma' env.py
```

仅改动 `_get_rewards()` 是不够的——`env.py` 有 1400 行，同一个 cfg 参数可能在以下位置被引用：
- `__init__`（缓存初始化）
- `_reset_idx`（reset 时缓存重置/初始化）
- `_get_rewards`（主计算逻辑）
- 辅助函数（`_compute_phi_align` 等诊断/日志方法）

### 2. 创建消融分支后应跑冒烟测试

如果每个分支创建后立即跑 2 iter 冒烟测试（`--max_iterations 2 --num_envs 128`），这类启动崩溃可以在 30 秒内发现，而不是等到正式跑实验时才发现浪费时间。

### 3. env.py 中的引用链条

`env.py` 中 cfg 参数的典型引用链条（容易遗漏的用 **粗体** 标记）：

```
env_cfg.py (定义参数)
  └─ env.py
       ├─ __init__()       ← 可能用 cfg 值初始化缓存
       ├─ **_reset_idx()** ← 用 cfg 值重置/初始化缓存（容易漏！）
       ├─ _get_rewards()   ← 主逻辑（通常会改到）
       └─ **辅助函数**      ← _compute_phi_align 等（容易漏！）
```
