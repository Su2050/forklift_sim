# 最佳模型记录 (Best Models)

本文档用于记录训练过程中产生的最佳模型及其表现，方便后续调用、回放和部署。

---

## 1. 叉车插入与举升任务 (Pallet Insert & Lift) - 2026-02-27 (s1.0zB)

* **实验版本**: Exp-A2 (基于 s1.0zB，解决推盘问题)
* **Run ID**: `2026-02-27_17-43-22`
* **最佳 Checkpoint**: `model_1999.pt`
* **模型绝对路径**: `/home/uniubi/projects/forklift_sim/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-27_17-43-22/model_1999.pt`

**为什么它是最好的？**
* **真实 Episode 成功率**: 88% ~ 90%
* **核心突破**: 完美解决了接近阶段“货叉未对准就推着托盘走”的物理交互缺陷。
* **Hold 阶段稳定性**: `fail_ins_frac` 和 `fail_align_frac` 均为 0.0000。
* **视觉验证**: 动作平滑，接近托盘时会主动减速并精准对齐，插入全程托盘几乎无位移。

---

## 2. 叉车插入与举升任务 (Pallet Insert & Lift) - 2026-02-27 (s1.0zA)

* **实验版本**: Exp-A (基于 s1.0zA)
* **Run ID**: `2026-02-27_13-13-32`
* **最佳 Checkpoint**: `model_1200.pt` (推荐范围: 1200 ~ 1269 iter)
* **模型绝对路径**: `/home/uniubi/projects/forklift_sim/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-27_13-13-32/model_1200.pt`

### 为什么它是最佳模型？

1. **真实 Episode 成功率极高**:
   根据修正后的 Episode 级别成功率公式计算，该模型在 1200-1300 iter 期间的真实成功率达到了 **85% ~ 100%**。
2. **Hold 阶段（保持举升）稳如泰山**:
   细粒度诊断日志显示，一旦触发成功条件进入 Hold 阶段，`fail_ins_frac` (插入失效) 和 `fail_align_frac` (对齐失效) 均为 `0.0000`。这意味着叉车插入后非常稳定，完全没有滑脱或偏移。
3. **处于完美的“早停”甜点位**:
   在 1300 iter 之后，模型虽然依然能保持高成功率，但可能会因为过度追求某些次要奖励（如极端的姿态精调）而导致动作不够平滑。因此，1200 iter 左右是一个非常完美的阶段，动作丝滑、高效且未过拟合。

### 如何加载和验收？

可以使用 `play.py` 脚本加载该模型进行视觉验收：

```bash
# 确保退出 conda 环境以防冲突
conda deactivate

cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
 --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
 --num_envs 32 \
 --checkpoint /home/uniubi/projects/forklift_sim/logs/rsl_rl/forklift_pallet_insert_lift/2026-02-27_13-13-32/model_1200.pt
```
