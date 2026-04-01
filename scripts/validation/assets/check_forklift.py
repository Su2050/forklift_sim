import os
import numpy as np
from pathlib import Path

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, Gf


REPO_ROOT = Path(__file__).resolve().parents[3]

def main():
    # Check Forklift
    local_override = REPO_ROOT / "assets" / "forklift_c_long.usda"
    
    try:
        from isaacsim.core.utils.nucleus import get_assets_root_path
        assets_root = get_assets_root_path()
        stage = None
        if assets_root:
            forklift_path = assets_root + "/Isaac/Vehicles/Forklift/forklift_c.usd"
            stage = Usd.Stage.Open(forklift_path)
        if stage is None and local_override.exists():
            stage = Usd.Stage.Open(str(local_override))
        if stage is None:
            raise FileNotFoundError(f"Cannot resolve forklift USD from Nucleus or local override: {local_override}")
            
        fork_prim = stage.GetPrimAtPath("/World/forklift_c/SM_Forklift_C01_Fork01_01")
        if fork_prim.IsValid():
            mesh = UsdGeom.Mesh(fork_prim)
            points = np.array(mesh.GetPointsAttr().Get())
            min_b = np.min(points, axis=0)
            max_b = np.max(points, axis=0)
            print(f"Fork 1 length (X): {max_b[0] - min_b[0]}")
            print(f"Fork 1 width (Y): {max_b[1] - min_b[1]}")
            print(f"Fork 1 height (Z): {max_b[2] - min_b[2]}")
        else:
            print("Fork prim not found at expected path")
    except Exception as e:
        print(f"Failed to check forklift: {e}")

    simulation_app.close()

if __name__ == "__main__":
    main()
