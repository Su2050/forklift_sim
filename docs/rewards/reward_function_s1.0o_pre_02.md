你这个“DO 系列做消融”的方向是对的，关键是把它**工程化成一套命名 + 流程 + 判据**，让你一眼就能回答三件事：

1. 这次到底改了啥？
2. 它让哪个指标变好了？
3. 变好是“真因果”还是“随机撞大运”？

下面给你一份**完整可执行**的消融实验方案（含合适命名）。

---

## 一、先定“因素（Factor）”与“版本（Variant）”

把你 S1.0O 的 P0 三刀当作 3 个因素，每个因素下面再列 2~3 个实现版本：

* **A：Lift 触发/梯度（让 frac_lifted 上来）**

  * A1：降低 gate（insert_gate_norm ↓）
  * A2：拉长 ramp（insert_ramp_norm ↑）
  * A3：premature lift 惩罚分段温和化 / 或新增 lift 里程碑 delta 势函数

* **B：Lateral 梯度（让 lateral_mean 真降）**

  * B1：增大 σ_y/σ_yaw（把梯度覆盖到 0.2~0.4m 区间）
  * B2：粗+细双势函数（coarse+fine），都用 delta（只奖进步）

* **C：Hold 去抖/计数稳定（让 success_now→success 转化率上来）**

  * C1：exit debounce（连续越界 N 步才判失败）
  * C2：越界不清零、改衰减（hold_counter decay）

> 这样你面对“改法很多”就不会乱：永远是 **A/B/C 三类**，每类下面 **1/2/3** 个版本。

---

## 二、命名规则：让人一眼看懂（推荐这套）

你提到 DO 系列，那就把 DO 当作“实验活动前缀”（campaign prefix）。核心命名用一个固定格式：

**DO-O-<组合>-s<seed>**

* `DO`：campaign（你这波实验叫 DO）
* `O`：大版本（S1.0O）
* `<组合>`：写清楚用了哪些因素、用了哪个版本
* `s<seed>`：随机种子编号（s0/s1…）

### 例子（你会用得很舒服）

* **DO-O-N-s0**：基线（等价 S1.0N，不改 reward，只换 run_id/记录方式）
* **DO-O-A1-s0**：只改 A1
* **DO-O-B2-s0**：只改 B2
* **DO-O-C1-s0**：只改 C1
* **DO-O-A2B1-s0**：A2 + B1 组合
* **DO-O-A2B1C1-s0**：A2 + B1 + C1 全组合（候选最终版）

> 你原来写错的 OEO 这类就不要了，信息密度太低；这种命名自带“配置说明书”。

---

## 三、实验流程：两轮半就够，不搞组合爆炸

### Round 0：基线对齐（1 个）

**DO-O-N-s0**
目的：确认“你现在的训练环境/吞吐/日志指标”跟 S1.0N 一致，避免后面误把环境变化当 reward 改进。

跑多少：**200~300 iter** 够做对齐。

---

### Round 1：单因素消融筛选（推荐 6 个以内）

只改一个因素，快速判断“这刀到底有没有用”。

建议跑这些（一个 seed 就行）：

* DO-O-A1-s0、DO-O-A2-s0（A 先跑两种，通常收益最大）
* DO-O-B1-s0（或 B2，先选你更信的那种）
* DO-O-C1-s0（debounce 通常比衰减更直观）
* 如果你还有余力，再补：DO-O-A3-s0、DO-O-B2-s0 或 DO-O-C2-s0

跑多少：**400~600 iter**（你这吞吐很高，不用等 2000 才知道死活）

**Round 1 的淘汰规则（硬一点，省时间）：**

* A 类：`phase/frac_lifted` 没有明显抬升（比如从 ~1–2% 拉到 ≥4–5% 的趋势）→ 直接淘汰该 A 版本
* B 类：`err/lateral_mean` 没有持续下降趋势（不是某一刻抖一下）→ 淘汰
* C 类：`frac_success / frac_success_now` 转化率没明显提升 → 淘汰

---

### Round 2：小规模组合验证（3 个）

从 Round 1 里每类挑一个“赢家”：

* DO-O-AB-s0（A* + B*）
* DO-O-AC-s0（A* + C*）
* DO-O-ABC-s0（A* + B* + C*）

跑多少：**800~1200 iter** 看趋势，最强的那个再拉满 2000。

---

### Round 2.5：做“可信度”（2 seeds 复现）

最终候选（通常是 ABC）必须过这一关，否则你会被随机性骗。

* DO-O-A*B*C*-s0
* DO-O-A*B*C*-s1

跑多少：**2000 iter**（或至少 1200+ 看到稳定）

---

## 四、每个实验盯什么 KPI：别被 Mean reward 迷惑

为了“因果归因”，每类因素给一个主 KPI：

* **A（Lift）主 KPI**：`phase/frac_lifted`
  辅助：`frac_success_now`

* **B（Lateral）主 KPI**：`err/lateral_mean`（看趋势）
  辅助：`phase/frac_inserted`

* **C（Hold）主 KPI**：`frac_success / frac_success_now`（转化率）
  辅助：`phase/grace_zone_frac`（别飙太高，飙高说明阈值松到“混过去”）

> 经验法则：
> **A 不起，后面都白聊；B 不降，lift/插入会卡；C 不稳，成功率像彩票。**

---

## 五、强烈建议你同时记录 4 个“实验元数据”（以后救命）

每个 run 除了名字，再写进日志/目录名里：

* `git_commit`（代码版本）
* `cfg_hash`（reward 配置 hash）
* `seed`
* `notes`（一句话：A2B1C1，debounce=3，sigma_y=0.30…）

这样你三天后回看不会“我当时到底改了啥来着”。

---

顺带说一句：你之前上传的训练日志/奖励文档我这边现在已经过期读不到了；不过这条不影响我给你设计“消融实验体系”。等你需要我帮你把 **A1/A2/B1/C1** 具体落到你当前 `env_cfg.py` 的参数表（甚至直接输出一套 Hydra override 清单）时，再把相关文件重新传一次就行。
