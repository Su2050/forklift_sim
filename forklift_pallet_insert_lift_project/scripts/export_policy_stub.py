"""策略导出脚本占位模板（需要按实际训练结果适配）。

使用背景：
- Isaac Lab 中 RSL-RL 的策略导出通常需要从 runner 中取出 policy 与观测归一化器
- 导出到 ONNX/JIT 时，归一化器与 wrapper 也可能需要一起封装

Isaac Lab 已提供导出辅助函数：
    from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

为什么这是占位脚本：
- 不同 Isaac Lab 版本/训练流程保存的 checkpoint 结构不同
- 需要你自己从 checkpoint 中定位 policy 与 normalizer

Usage（示例）：
    python scripts/export_policy_stub.py --checkpoint /path/to/model_XXXX.pt --out_dir exports/
"""

import argparse
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="exports")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint（只加载，不做结构假设）
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print("[INFO] Loaded checkpoint keys:", list(ckpt.keys())[:20])
    print("[INFO] You should locate the policy / actor-critic module and (optionally) the normalizer.")

    # Example skeleton (you will need to adapt):
    #
    # from isaaclab_rl.rsl_rl import export_policy_as_onnx, export_policy_as_jit
    #
    # policy = <extract policy module>
    # normalizer = <extract normalizer or None>
    #
    # export_policy_as_jit(policy, normalizer, out_dir / "policy.pt")
    # export_policy_as_onnx(policy, out_dir / "policy.onnx")
    #
    # 注意：如果训练时启用了观测归一化，推理时必须使用同一份 normalizer。

    print("[WARN] This is a stub. See comments inside for how to adapt to your run.")

if __name__ == "__main__":
    main()
