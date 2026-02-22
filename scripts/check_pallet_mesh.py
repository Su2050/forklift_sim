import os
import sys
import numpy as np

# Set up Isaac Sim environment
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom

def main():
    usd_path = "/home/uniubi/projects/forklift_sim/assets/pallet.usd"
    if not os.path.exists(usd_path):
        print(f"Error: {usd_path} not found.")
        return
        
    stage = Usd.Stage.Open(usd_path)
    
    # Find the mesh
    mesh_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = prim
            break
            
    if not mesh_prim:
        print("No mesh found in USD.")
        return
        
    print(f"Found Mesh: {mesh_prim.GetPath()}")
    
    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.array(mesh.GetPointsAttr().Get())
    
    # Analyze the geometry
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    size = max_bound - min_bound
    
    print(f"Bounding Box Min: {min_bound}")
    print(f"Bounding Box Max: {max_bound}")
    print(f"Size: {size}")
    
    # To check if there's a middle block, we can look at the X-Y distribution of vertices.
    # We will create a simple 2D ASCII map.
    # The pallet's insertion face is usually along the X or Y axis.
    # Let's map X to rows and Y to columns (or vice versa).
    # Since we want to see the gaps for the forks, let's slice horizontally.
    
    print("\n--- 托盘 Y-Z 横截面 ASCII 图 (观察孔洞) ---")
    # Z is usually UP. Y is width. X is depth.
    # We want to look at the front face, so we project onto the Y-Z plane.
    # Let's take points in the front half (X < min_x + size_x/2)
    # Actually, let's just do a 2D histogram of Y and Z to see the solid parts.
    
    GRID_W = 60 # Y axis
    GRID_H = 15 # Z axis
    
    grid = np.zeros((GRID_H, GRID_W))
    
    for p in points:
        x, y, z = p
        # Normalize to 0-1
        ny = (y - min_bound[1]) / size[1] if size[1] > 0 else 0
        nz = (z - min_bound[2]) / size[2] if size[2] > 0 else 0
        
        # Clamp to grid indices
        iy = min(int(ny * GRID_W), GRID_W - 1)
        iz = min(int(nz * GRID_H), GRID_H - 1)
        
        grid[iz, iy] += 1
        
    # Print from top (max Z) to bottom (min Z)
    for iz in range(GRID_H - 1, -1, -1):
        row_str = ""
        for iy in range(GRID_W):
            count = grid[iz, iy]
            if count > 20:
                row_str += "█"
            elif count > 5:
                row_str += "▓"
            elif count > 0:
                row_str += "░"
            else:
                row_str += " "
        print(row_str)
        
    print("\n--- 托盘 X-Y 俯视 ASCII 图 (观察内部立柱结构) ---")
    grid_xy = np.zeros((30, 60)) # X is 30, Y is 60
    for p in points:
        x, y, z = p
        nx = (x - min_bound[0]) / size[0] if size[0] > 0 else 0
        ny = (y - min_bound[1]) / size[1] if size[1] > 0 else 0
        ix = min(int(nx * 30), 29)
        iy = min(int(ny * 60), 59)
        grid_xy[ix, iy] += 1
        
    for ix in range(30):
        row_str = ""
        for iy in range(60):
            count = grid_xy[ix, iy]
            if count > 20:
                row_str += "█"
            elif count > 5:
                row_str += "▓"
            elif count > 0:
                row_str += "░"
            else:
                row_str += " "
        print(row_str)

    simulation_app.close()

if __name__ == "__main__":
    main()
