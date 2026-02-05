#!/usr/bin/env python3
"""
扫描 Isaac Sim Nucleus 服务器上可用的托盘资产

运行方式：
cd /home/uniubi/projects/forklift_sim/IsaacLab
./isaaclab.sh -p ../scripts/scan_nucleus_pallets.py --headless
"""

import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("请通过 isaaclab.sh 运行此脚本")
    sys.exit(1)

isaaclab_path = Path(__file__).resolve().parent.parent / "IsaacLab"
sys.path.insert(0, str(isaaclab_path / "source"))

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="扫描Nucleus托盘资产")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.client
from pxr import Usd, UsdGeom, Gf
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def list_directory(path: str) -> list:
    """列出 Nucleus 目录内容"""
    result, entries = omni.client.list(path)
    if result != omni.client.Result.OK:
        return []
    return [(e.relative_path, e.flags) for e in entries]

def get_usd_size(usd_path: str) -> dict:
    """获取 USD 文件的边界框尺寸"""
    try:
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            return None
        
        # 获取根边界框
        root_prim = stage.GetPseudoRoot()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        
        total_bbox = Gf.BBox3d()
        for prim in stage.Traverse():
            if UsdGeom.Boundable(prim):
                bbox = bbox_cache.ComputeWorldBound(prim)
                total_bbox = Gf.BBox3d.Combine(total_bbox, bbox)
        
        bbox_range = total_bbox.ComputeAlignedBox()
        if bbox_range.IsEmpty():
            return None
        
        box_min = bbox_range.GetMin()
        box_max = bbox_range.GetMax()
        
        return {
            "min": (box_min[0], box_min[1], box_min[2]),
            "max": (box_max[0], box_max[1], box_max[2]),
            "size_m": (box_max[0] - box_min[0], box_max[1] - box_min[1], box_max[2] - box_min[2]),
            "size_mm": ((box_max[0] - box_min[0]) * 1000, 
                        (box_max[1] - box_min[1]) * 1000, 
                        (box_max[2] - box_min[2]) * 1000),
        }
    except Exception as e:
        print(f"  [错误] 无法读取 {usd_path}: {e}")
        return None

def scan_pallets():
    """扫描可用的托盘资产"""
    print("=" * 80)
    print("Isaac Sim Nucleus 托盘资产扫描")
    print("=" * 80)
    print(f"\nISAAC_NUCLEUS_DIR: {ISAAC_NUCLEUS_DIR}")
    
    # 搜索路径
    search_paths = [
        f"{ISAAC_NUCLEUS_DIR}/Props/Pallet",
        f"{ISAAC_NUCLEUS_DIR}/Props",
        f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props",
        f"{ISAAC_NUCLEUS_DIR}/Environments/Warehouse/Props",
    ]
    
    pallets_found = []
    
    for base_path in search_paths:
        print(f"\n扫描: {base_path}")
        entries = list_directory(base_path)
        
        if not entries:
            print("  (目录不存在或为空)")
            continue
        
        for name, flags in entries:
            full_path = f"{base_path}/{name}"
            
            # 检查是否是 pallet 相关
            if "pallet" in name.lower() or "Pallet" in name:
                print(f"  找到: {name}")
                
                # 如果是目录，扫描其中的 USD 文件
                if flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
                    sub_entries = list_directory(full_path)
                    for sub_name, _ in sub_entries:
                        if sub_name.endswith(".usd") or sub_name.endswith(".usda"):
                            usd_path = f"{full_path}/{sub_name}"
                            print(f"    USD: {sub_name}")
                            size_info = get_usd_size(usd_path)
                            if size_info:
                                pallets_found.append({
                                    "path": usd_path,
                                    "name": sub_name,
                                    "size": size_info
                                })
                                print(f"      尺寸(mm): X={size_info['size_mm'][0]:.1f}, "
                                      f"Y={size_info['size_mm'][1]:.1f}, Z={size_info['size_mm'][2]:.1f}")
                elif name.endswith(".usd") or name.endswith(".usda"):
                    size_info = get_usd_size(full_path)
                    if size_info:
                        pallets_found.append({
                            "path": full_path,
                            "name": name,
                            "size": size_info
                        })
                        print(f"    尺寸(mm): X={size_info['size_mm'][0]:.1f}, "
                              f"Y={size_info['size_mm'][1]:.1f}, Z={size_info['size_mm'][2]:.1f}")
    
    # 也检查当前使用的托盘
    print("\n" + "=" * 80)
    print("当前使用的托盘资产")
    print("=" * 80)
    
    current_pallet = f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd"
    print(f"\n路径: {current_pallet}")
    size_info = get_usd_size(current_pallet)
    if size_info:
        print(f"尺寸(mm): X={size_info['size_mm'][0]:.1f}, Y={size_info['size_mm'][1]:.1f}, Z={size_info['size_mm'][2]:.1f}")
        print(f"尺寸(m): X={size_info['size_m'][0]:.3f}, Y={size_info['size_m'][1]:.3f}, Z={size_info['size_m'][2]:.3f}")
    
    # 检查叉车资产
    print("\n" + "=" * 80)
    print("叉车资产信息")
    print("=" * 80)
    
    forklift_path = f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/ForkliftC/forklift_c.usd"
    print(f"\n路径: {forklift_path}")
    size_info = get_usd_size(forklift_path)
    if size_info:
        print(f"整体尺寸(mm): X={size_info['size_mm'][0]:.1f}, Y={size_info['size_mm'][1]:.1f}, Z={size_info['size_mm'][2]:.1f}")
    
    # 扫描 Props 目录中是否有其他叉车/托盘
    print("\n" + "=" * 80)
    print("Props 目录内容")
    print("=" * 80)
    
    props_path = f"{ISAAC_NUCLEUS_DIR}/Props"
    entries = list_directory(props_path)
    print(f"\n{props_path}:")
    for name, flags in entries[:30]:  # 只显示前30个
        print(f"  - {name}")
    
    # 扫描 Robots/IsaacSim 目录
    print("\n" + "=" * 80)
    print("Robots/IsaacSim 目录内容")
    print("=" * 80)
    
    robots_path = f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim"
    entries = list_directory(robots_path)
    print(f"\n{robots_path}:")
    for name, flags in entries[:30]:
        print(f"  - {name}")
    
    print("\n" + "=" * 80)
    print("扫描完成")
    print("=" * 80)
    
    simulation_app.close()

if __name__ == "__main__":
    scan_pallets()
