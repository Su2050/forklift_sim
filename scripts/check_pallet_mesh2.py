import os
import sys
import numpy as np

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom

def main():
    usd_path = "/home/uniubi/projects/forklift_sim/assets/pallet.usd"
    if not os.path.exists(usd_path):
        return
        
    stage = Usd.Stage.Open(usd_path)
    mesh_prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.array(mesh.GetPointsAttr().Get())
    
    min_bound = np.min(points, axis=0)
    size = np.max(points, axis=0) - min_bound
    
    # Let's correctly identify axes.
    # From previous output:
    # Size X: 121.3 (Depth)
    # Size Y: 21.1 (Height)
    # Size Z: 100.2 (Width)
    
    # 1. Front View (Width Z vs Height Y)
    print("\n[正视图] 宽度(Z) - 高度(Y) 截面")
    GRID_Z = 60 # Width
    GRID_Y = 15 # Height
    grid_zy = np.zeros((GRID_Y, GRID_Z))
    for p in points:
        z_norm = (p[2] - min_bound[2]) / size[2]
        y_norm = (p[1] - min_bound[1]) / size[1]
        iz = min(int(z_norm * GRID_Z), GRID_Z - 1)
        iy = min(int(y_norm * GRID_Y), GRID_Y - 1)
        grid_zy[iy, iz] += 1
        
    for iy in range(GRID_Y - 1, -1, -1): # Top to bottom height
        row_str = ""
        for iz in range(GRID_Z): # Left to right width
            count = grid_zy[iy, iz]
            row_str += "█" if count > 10 else "▒" if count > 0 else " "
        print(row_str)
        
    # 2. Top-Down View (Depth X vs Width Z) - THIS WILL SHOW THE BLOCKS!
    print("\n[俯视图] 深度(X) - 宽度(Z) 截面 (只看底部的垫块)")
    GRID_X = 30 # Depth
    GRID_Z2 = 60 # Width
    grid_xz = np.zeros((GRID_X, GRID_Z2))
    for p in points:
        y_norm = (p[1] - min_bound[1]) / size[1]
        if y_norm > 0.5: # Skip the top boards, only look at bottom half!
            continue
            
        x_norm = (p[0] - min_bound[0]) / size[0]
        z_norm = (p[2] - min_bound[2]) / size[2]
        ix = min(int(x_norm * GRID_X), GRID_X - 1)
        iz = min(int(z_norm * GRID_Z2), GRID_Z2 - 1)
        grid_xz[ix, iz] += 1
        
    for ix in range(GRID_X - 1, -1, -1): # Back to Front depth
        row_str = ""
        for iz in range(GRID_Z2): # Left to right width
            count = grid_xz[ix, iz]
            row_str += "█" if count > 10 else "▒" if count > 0 else " "
        print(row_str)

    simulation_app.close()

if __name__ == "__main__":
    main()
