# 文档导航（先读这个）

这套文档面向**不太懂强化学习 / 仿真 / Isaac Sim / Isaac Lab**的读者，目标是让你能：

- **跑起来**：把叉车任务训练跑通并产出 checkpoint
- **看懂**：这项任务在训练什么、奖励如何引导行为
- **会排障**：遇到“不动/不收敛/太慢/倾翻”能定位问题

---

## 阅读路径（按你的情况选）

### A. 我只想尽快跑起来（推荐）

1) [`00_prereqs_and_versions.md`](./00_prereqs_and_versions.md)：前置条件与版本建议（避免卡安装/兼容）
2) [`01_quickstart.md`](./01_quickstart.md)：最短路径：打补丁→开训→回放

### B. 我能跑起来，但想理解“训练目标是什么/为什么这么设定”

1) [`02_task_overview.md`](./02_task_overview.md)：任务边界与场景里有什么物体
2) [`03_task_design_rl.md`](./03_task_design_rl.md)：动作/观测/奖励/成功判定（用大白话解释）

### C. 我想系统掌握训练、日志、checkpoint、导出与评估

1) [`04_training_and_artifacts.md`](./04_training_and_artifacts.md)：训练命令、参数含义、日志与产物结构
2) [`05_evaluation_and_export.md`](./05_evaluation_and_export.md)：回放、录视频、导出 ONNX/JIT

### D. 我想给叉车加相机 / 把输入改成图像

1) [`07_vision_input.md`](./07_vision_input.md)：视觉输入与相机配置、进阶迭代路线图（Phase 0~6）

### E. 我遇到问题（不动/报错/很慢/一直翻车）

1) [`06_troubleshooting.md`](./06_troubleshooting.md)：按现象排障
2) [`troubleshooting_cases/README.md`](./troubleshooting_cases/README.md)：排障案例复盘库（真实问题→定位→修复→验证）

### F. 术语不懂（PPO、episode、headless…）

1) [`99_glossary.md`](./99_glossary.md)：小词典

