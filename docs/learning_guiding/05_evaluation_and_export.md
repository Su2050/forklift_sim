# 评估、回放与导出：怎么录视频、怎么导出 ONNX/JIT

本页适合：已经训练出 checkpoint，想回放效果/录制视频/导出模型的人。

---

## 1. 回放脚本入口在哪里？

- `IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py`

它会做三件事：

- 加载 checkpoint
- 在仿真里用策略控制叉车
- （可选）录视频，并把策略导出到 `exported/`

---

## 2. 最常用：回放 + 录视频

```bash
cd <你的IsaacLab目录>

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Forklift-PalletInsertLift-Direct-v0 \
  --num_envs 16 \
  --headless \
  --video --video_length 300 \
  --load_run ".*" \
  --checkpoint ".*"
```

### 视频会输出到哪里？

一般在（相对 IsaacLab 根目录）：

- `logs/rsl_rl/forklift_pallet_insert_lift/<run>/videos/play/`

`play.py` 里配置的 video_folder 是：

- `<log_dir>/videos/play`

---

## 3. 如何选择某次训练 run 的 checkpoint？

你有三种方式：

- **自动选最新**：`--load_run ".*" --checkpoint ".*"`（推荐）
- **指定目录名**：`--load_run "<时间戳_run_name>"`，`--checkpoint ".*"`
- **指定具体文件**：`--checkpoint "model_1000.pt"`（按你的 rsl-rl 文件名来）

---

## 4. 导出策略（JIT / ONNX）

`play.py` 会在加载 checkpoint 后自动导出：

- JIT：`policy.pt`
- ONNX：`policy.onnx`

导出目录（相对 checkpoint 所在目录）：

- `<checkpoint_dir>/exported/`

也就是说，你会在类似路径看到导出结果：

- `logs/rsl_rl/forklift_pallet_insert_lift/<run>/exported/policy.pt`
- `logs/rsl_rl/forklift_pallet_insert_lift/<run>/exported/policy.onnx`

为什么要在 `play.py` 里导出？

- 因为导出通常需要同时拿到：
  - 策略网络（policy/actor-critic）
  - 观测归一化器（normalizer）
  这些信息在 runner 里最完整。

---

## 5. “导出后怎么用？”

这个属于“部署”主题，取决于你是要：

- 在 Isaac Sim 里直接加载 JIT
- 在别的推理框架里用 ONNX
- 或者做 sim2real

本项目目前提供的是“导出产物”，部署方式建议先按你的目标平台再定。\n如需要，我可以在下一轮补一篇 `07_deployment_notes.md` 专门讲“怎么把 exported 的模型跑在别的程序里”。

