#!/usr/bin/env python3
"""Step-1 camera observation contract checker (no Isaac Lab runtime required)."""

from pathlib import Path

REQUIRED_SNIPPETS = [
    "use_camera",
    "use_asymmetric_critic",
    "camera_width",
    "camera_height",
    "camera_hfov_deg",
    "camera_pos_local",
    "camera_rpy_local_deg",
    "easy8_dim",
    "privileged_dim",
]


def main():
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py"
    env_path = repo_root / "isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env.py"

    cfg_text = cfg_path.read_text(encoding="utf-8")
    env_text = env_path.read_text(encoding="utf-8")

    missing_cfg = [k for k in REQUIRED_SNIPPETS if k not in cfg_text]
    required_env_tokens = ["_get_camera_image", "_get_easy8", "_get_privileged_obs", '"image"', '"proprio"', '"critic"']
    missing_env = [k for k in required_env_tokens if k not in env_text]

    if missing_cfg or missing_env:
        if missing_cfg:
            print(f"❌ Missing cfg tokens: {missing_cfg}")
        if missing_env:
            print(f"❌ Missing env tokens: {missing_env}")
        raise SystemExit(1)

    print("✅ Camera obs scaffold contract check passed")
    print(f"   checked files:\n   - {cfg_path}\n   - {env_path}")


if __name__ == "__main__":
    main()
