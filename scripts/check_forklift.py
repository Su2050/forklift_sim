import os
import numpy as np

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, Gf

def main():
    # Check Forklift
    usd_path = "omniverse://localhost/NVIDIA/Assets/Isaac/5.1/Isaac/Vehicles/Forklift/forklift_c.usd"
    # We can't easily open omniverse:// without Nucleus running, but we can open the local overridden one
    usd_path = "/home/uniubi/projects/forklift_sim/assets/forklift_c_long.usda"
    if not os.path.exists(usd_path):
        usd_path = "/home/uniubi/projects/forklift_sim/forklift_pallet_insert_lift_project/isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/env_cfg.py" # just a fallback, won't work
    
    try:
        from isaacsim.core.utils.nucleus import get_assets_root_path
        assets_root = get_assets_root_path()
        if assets_root:
            forklift_path = assets_root + "/Isaac/Vehicles/Forklift/forklift_c.usd"
            stage = Usd.Stage.Open(forklift_path)
            
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
