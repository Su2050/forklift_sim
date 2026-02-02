"""Starter exporter stub.

In Isaac Lab, exporting a trained RSL-RL policy to ONNX/JIT is typically done from the runner object,
because the observation normalizer (and sometimes policy wrappers) are part of the training stack.

Isaac Lab provides helper functions:
    from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

This script is intentionally conservative: it shows the *shape* of what you need to do, but you will
need to adapt it to the exact checkpoint structure used by your Isaac Lab version/run.

Usage (conceptual):
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

    # Load checkpoint
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

    print("[WARN] This is a stub. See comments inside for how to adapt to your run.")

if __name__ == "__main__":
    main()
