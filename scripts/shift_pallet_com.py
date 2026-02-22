import os

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, UsdGeom, Gf

def shift_center_of_mass(usd_path, output_path, shift_x=-20.0):
    stage = Usd.Stage.Open(usd_path)
    
    mesh_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh_prim = prim
            break
            
    if not mesh_prim:
        print("Mesh not found")
        return
        
    if not mesh_prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(mesh_prim)
    else:
        mass_api = UsdPhysics.MassAPI(mesh_prim)
        
    current_com = mass_api.GetCenterOfMassAttr().Get()
    if current_com is None:
        new_com = Gf.Vec3f(shift_x, 0.0, 0.0)
    else:
        new_com = Gf.Vec3f(current_com[0] + shift_x, current_com[1], current_com[2])
        
    mass_api.CreateCenterOfMassAttr().Set(new_com)
    
    stage.GetRootLayer().Export(output_path)
    print(f"✅ Successfully shifted CoM by X={shift_x} to {output_path}")

if __name__ == "__main__":
    usd_path = "/home/uniubi/projects/forklift_sim/assets/pallet.usd"
    output_path = "/home/uniubi/projects/forklift_sim/assets/pallet_com_shifted.usd"
    
    # 托盘的长是120cm。我们想让重心往前移动20cm。
    # 它的单位可能是厘米，之前看到X范围是 -60 到 +60。
    # 所以移动20cm，就是 -20.0 (假设负方向是朝向货叉)。
    shift_center_of_mass(usd_path, output_path, shift_x=-20.0)
    simulation_app.close()
