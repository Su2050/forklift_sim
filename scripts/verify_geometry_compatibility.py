#!/usr/bin/env python3
"""
验证叉车货叉与托盘插入孔的几何兼容性

检查内容：
1. 货叉几何尺寸（宽度、高度、间距、长度）
2. 托盘插入孔几何尺寸（宽度、高度、间距、深度）
3. 碰撞形状类型和尺寸
4. 几何兼容性分析
5. 实际碰撞测试
"""

import sys
from pathlib import Path

# 环境检查：确保在正确的Python环境中运行
try:
    import torch
except ImportError:
    print("=" * 80)
    print("错误：无法导入 torch 模块")
    print("=" * 80)
    print("\n原因：脚本需要通过 isaaclab.sh 运行，以使用IsaacLab的Python环境。")
    print("\n正确的运行方式：")
    print("  cd /home/uniubi/projects/forklift_sim/IsaacLab")
    print("  ./isaaclab.sh -p ../scripts/verify_geometry_compatibility.py --headless")
    print("=" * 80)
    sys.exit(1)

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# 添加 IsaacLab 路径
isaaclab_path = Path(__file__).resolve().parent.parent / "IsaacLab"
sys.path.insert(0, str(isaaclab_path / "source"))

# 首先初始化 Isaac Sim
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="验证叉车托盘几何兼容性")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 在 Isaac Sim 初始化后导入
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env_cfg import ForkliftPalletInsertLiftEnvCfg
from isaaclab_tasks.direct.forklift_pallet_insert_lift.env import ForkliftPalletInsertLiftEnv


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_info(label: str, value):
    """打印信息"""
    print(f"  {label}: {value}")


def print_usd_hierarchy(stage, root_path: str, max_depth: int = 5, show_types: bool = True):
    """打印USD层级结构用于调试
    
    Args:
        stage: USD stage
        root_path: 根路径
        max_depth: 最大遍历深度
        show_types: 是否显示prim类型
    """
    prim = stage.GetPrimAtPath(root_path)
    if not prim.IsValid():
        print(f"[警告] 无效的路径: {root_path}")
        return
    
    def traverse(prim, depth=0):
        if depth > max_depth:
            return
        indent = "  " * depth
        type_str = f" ({prim.GetTypeName()})" if show_types and prim.GetTypeName() else ""
        print(f"{indent}{prim.GetName()}{type_str}")
        for child in prim.GetChildren():
            traverse(child, depth + 1)
    
    print(f"\nUSD层级结构 (根路径: {root_path}, 最大深度: {max_depth}):")
    traverse(prim)


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """从四元数提取偏航角"""
    if q.dim() == 1:
        w, x, y, z = q.unbind(-1)
    else:
        w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


@dataclass
class GeometryDimensions:
    """几何尺寸数据类"""
    width: float = 0.0  # 宽度（mm）
    height: float = 0.0  # 高度（mm）
    length: float = 0.0  # 长度（mm）
    spacing: float = 0.0  # 间距（mm，用于两根货叉或两个插入孔）
    depth: float = 0.0  # 深度（mm，用于插入孔）
    collision_type: str = "Unknown"  # 碰撞形状类型


@dataclass
class CompatibilityResult:
    """兼容性检查结果"""
    width_compatible: bool = False
    height_compatible: bool = False
    spacing_compatible: bool = False
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class GeometryAnalyzer:
    """几何分析器"""
    
    def __init__(self):
        self.stage = None
        
    def load_usd(self, usd_path: str) -> bool:
        """加载USD文件"""
        self.stage = Usd.Stage.Open(usd_path)
        if not self.stage:
            print(f"[错误] 无法打开 USD 文件: {usd_path}")
            return False
        return True
    
    def get_bounding_box(self, prim_path: str) -> Optional[Dict]:
        """获取prim及其子节点的边界框（AABB）"""
        if not self.stage:
            return None
            
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        
        bbox_min = None
        bbox_max = None
        
        def traverse_prim(prim_path_str):
            nonlocal bbox_min, bbox_max
            prim = self.stage.GetPrimAtPath(prim_path_str)
            if not prim.IsValid():
                return
            
            # 检查是否有几何体
            mesh = UsdGeom.Mesh(prim)
            if mesh:
                # 获取边界框
                boundable = UsdGeom.Boundable(prim)
                if boundable:
                    bbox = boundable.ComputeWorldBound(Usd.TimeCode.Default(), "default")
                    if bbox:
                        bbox_range = bbox.ComputeAlignedBox()
                        if bbox_range:
                            box_min = bbox_range.GetMin()
                            box_max = bbox_range.GetMax()
                            
                            if bbox_min is None:
                                bbox_min = box_min
                                bbox_max = box_max
                            else:
                                bbox_min = Gf.Vec3d(
                                    min(bbox_min[0], box_min[0]),
                                    min(bbox_min[1], box_min[1]),
                                    min(bbox_min[2], box_min[2])
                                )
                                bbox_max = Gf.Vec3d(
                                    max(bbox_max[0], box_max[0]),
                                    max(bbox_max[1], box_max[1]),
                                    max(bbox_max[2], box_max[2])
                                )
            
            # 递归遍历子节点
            for child in prim.GetChildren():
                traverse_prim(str(child.GetPath()))
        
        traverse_prim(prim_path)
        
        if bbox_min is None or bbox_max is None:
            return None
        
        return {
            "min": np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
            "max": np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
            "size": np.array([bbox_max[0] - bbox_min[0], 
                            bbox_max[1] - bbox_min[1], 
                            bbox_max[2] - bbox_min[2]])
        }
    
    def find_prims_by_name(self, root_path: str, name_patterns: List[str]) -> List[str]:
        """根据名称模式查找prim路径"""
        if not self.stage:
            return []
        
        prim = self.stage.GetPrimAtPath(root_path)
        if not prim.IsValid():
            return []
        
        found_paths = []
        
        def traverse_prim(prim_path_str):
            prim = self.stage.GetPrimAtPath(prim_path_str)
            if not prim.IsValid():
                return
            
            prim_name = prim.GetName().lower()
            for pattern in name_patterns:
                if pattern.lower() in prim_name:
                    found_paths.append(prim_path_str)
                    break
            
            for child in prim.GetChildren():
                traverse_prim(str(child.GetPath()))
        
        traverse_prim(root_path)
        return found_paths
    
    def get_collision_shape_info(self, prim_path: str) -> Dict:
        """获取碰撞形状信息"""
        if not self.stage:
            return {"type": "Unknown", "size": None}
        
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"type": "Unknown", "size": None}
        
        # 检查是否有碰撞API
        collision_api = PhysxSchema.PhysxCollisionAPI(prim)
        if collision_api:
            # 检查碰撞形状类型
            collision_shape = collision_api.GetCollisionShapeAttr().Get()
            if collision_shape:
                return {"type": str(collision_shape), "size": None}
        
        # 检查是否有PhysicsCollisionAPI
        physics_collision = UsdPhysics.CollisionAPI(prim)
        if physics_collision:
            # 尝试获取碰撞mesh
            collision_mesh = physics_collision.GetCollisionMeshRel()
            if collision_mesh:
                targets = collision_mesh.GetTargets()
                if targets:
                    return {"type": "Mesh", "mesh_path": str(targets[0]), "size": None}
        
        # 检查是否有Box碰撞
        if prim.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim)
            size_attr = cube.GetSizeAttr()
            if size_attr:
                size = size_attr.Get()
                return {"type": "Box", "size": size}
        
        return {"type": "Unknown", "size": None}


class ForkliftGeometryAnalyzer(GeometryAnalyzer):
    """叉车几何分析器"""
    
    def __init__(self):
        super().__init__()
        self.env = None  # 环境实例，用于body投影法
    
    def extract_fork_dimensions(self, use_body_projection: bool = True) -> Optional[GeometryDimensions]:
        """提取货叉尺寸
        
        Args:
            use_body_projection: 是否使用body投影法（更准确）
        """
        print_section("提取货叉几何尺寸")
        
        # 优先使用body投影法
        if use_body_projection:
            result = self.extract_fork_via_body_projection()
            if result:
                return result
            print("[警告] body投影法失败，回退到名称匹配法...")
        
        # 回退：名称匹配法（带排除模式）
        return self._extract_fork_via_name_matching()
    
    def extract_fork_via_body_projection(self) -> Optional[GeometryDimensions]:
        """使用body投影法提取货叉尺寸（更准确）
        
        通过环境实例获取body位置，找到前向投影最大的body作为货叉
        """
        print("\n[方法] 使用body投影法识别货叉...")
        
        try:
            # 创建环境实例
            cfg = ForkliftPalletInsertLiftEnvCfg()
            cfg.scene.num_envs = 1
            self.env = ForkliftPalletInsertLiftEnv(cfg)
            self.env.reset()
            
            # 获取body信息
            body_names = self.env.robot.body_names
            body_pos = self.env.robot.data.body_pos_w[0]  # (B, 3)
            root_pos = self.env.robot.data.root_pos_w[0]  # (3,)
            root_quat = self.env.robot.data.root_quat_w[0]  # (4,)
            
            print_info("总body数量", len(body_names))
            
            # 计算前向投影
            yaw = _quat_to_yaw(root_quat)
            fwd = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)])
            
            rel = body_pos - root_pos  # (B, 3)
            proj = (rel * fwd).sum(-1)  # (B,)
            
            # 找到投影最大的几个body
            k = min(5, len(body_names))
            top_values, top_indices = torch.topk(proj, k=k)
            
            print("\n投影最大的body（可能是货叉）:")
            fork_bodies = []
            for i, (val, idx) in enumerate(zip(top_values, top_indices)):
                body_name = body_names[idx]
                body_position = body_pos[idx]
                print_info(f"  {i+1}. {body_name}", 
                          f"投影={val.item():.4f}m, 位置=({body_position[0]:.3f}, {body_position[1]:.3f}, {body_position[2]:.3f})")
                fork_bodies.append((body_name, idx.item(), body_position))
            
            # 在USD中查找对应的prim并计算边界框
            fork_bboxes = []
            for body_name, body_idx, body_position in fork_bodies:
                # 尝试多种路径模式查找
                prim_paths_to_try = [
                    f"/SM_Forklift_C01_01/{body_name}",
                    f"/ForkliftC/{body_name}",
                    f"/Robot/{body_name}",
                ]
                
                # 也尝试在整个stage中搜索精确匹配
                found_path = None
                for try_path in prim_paths_to_try:
                    prim = self.stage.GetPrimAtPath(try_path)
                    if prim.IsValid():
                        found_path = try_path
                        break
                
                # 如果直接路径找不到，搜索包含该名称的prim
                if not found_path:
                    found_paths = self._find_prims_exact_name(body_name)
                    if found_paths:
                        found_path = found_paths[0]
                
                if found_path:
                    bbox = self.get_bounding_box(found_path)
                    if bbox:
                        fork_bboxes.append((found_path, bbox, body_position))
                        print_info(f"    找到USD prim", found_path)
            
            # 如果通过USD找不到边界框，使用body位置估算
            if not fork_bboxes:
                print("[警告] 无法在USD中找到货叉prim，使用body位置估算...")
                return self._estimate_fork_from_body_positions(fork_bodies)
            
            # 计算尺寸
            if len(fork_bboxes) >= 2:
                return self._compute_fork_dimensions_from_bodies(fork_bboxes)
            else:
                return self._compute_fork_dimensions_single((fork_bboxes[0][0], fork_bboxes[0][1]))
            
        except Exception as e:
            print(f"[错误] body投影法失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if self.env:
                self.env.close()
                self.env = None
    
    def _find_prims_exact_name(self, name: str, root_path: str = "/") -> List[str]:
        """精确匹配名称查找prim（区分大小写）"""
        if not self.stage:
            return []
        
        prim = self.stage.GetPrimAtPath(root_path)
        if not prim.IsValid():
            return []
        
        found_paths = []
        
        def traverse(p):
            if p.GetName() == name:
                found_paths.append(str(p.GetPath()))
            for child in p.GetChildren():
                traverse(child)
        
        traverse(prim)
        return found_paths
    
    def _estimate_fork_from_body_positions(self, fork_bodies: List) -> Optional[GeometryDimensions]:
        """从body位置估算货叉尺寸"""
        if not fork_bodies:
            return None
        
        # 假设货叉是前向投影最大的body
        # 使用标准货叉尺寸估算
        print("[警告] 使用标准货叉尺寸估算")
        return GeometryDimensions(
            width=100.0,  # 标准货叉宽度约100mm
            height=50.0,  # 标准货叉高度约50mm
            length=1000.0,  # 标准货叉长度约1000mm
            spacing=400.0,  # 标准间距约400mm
            collision_type="Estimated_from_body"
        )
    
    def _compute_fork_dimensions_from_bodies(self, fork_bboxes: List) -> GeometryDimensions:
        """从多个body的边界框计算货叉尺寸"""
        if len(fork_bboxes) < 2:
            return self._compute_fork_dimensions_single((fork_bboxes[0][0], fork_bboxes[0][1]))
        
        # 计算尺寸
        widths = []
        heights = []
        lengths = []
        centers_y = []
        
        for path, bbox, body_pos in fork_bboxes:
            size = bbox["size"] * 1000  # mm
            center = (bbox["min"] + bbox["max"]) / 2 * 1000  # mm
            
            widths.append(size[1])
            heights.append(size[2])
            lengths.append(size[0])
            centers_y.append(center[1])
        
        # 计算间距
        centers_y.sort()
        spacing = abs(centers_y[-1] - centers_y[0]) if len(centers_y) >= 2 else 0.0
        
        return GeometryDimensions(
            width=np.mean(widths),
            height=np.mean(heights),
            length=np.mean(lengths),
            spacing=spacing,
            collision_type="Body_Projection"
        )
    
    def _extract_fork_via_name_matching(self) -> Optional[GeometryDimensions]:
        """使用名称匹配法提取货叉尺寸（带排除模式）"""
        print("\n[方法] 使用名称匹配法识别货叉...")
        
        if not self.stage:
            print("[错误] USD stage 未加载")
            return None
        
        # 查找货叉相关的prim，排除包含"forklift"的
        fork_patterns = ["fork", "tine", "blade", "prong"]
        exclude_patterns = ["forklift", "sm_forklift"]  # 排除整车
        root_paths = ["/SM_Forklift_C01_01", "/ForkliftC", "/Robot", "/"]
        
        fork_paths = []
        for root_path in root_paths:
            prim = self.stage.GetPrimAtPath(root_path)
            if prim.IsValid():
                fork_paths = self._find_prims_with_exclude(root_path, fork_patterns, exclude_patterns)
                if fork_paths:
                    break
        
        if not fork_paths:
            print("[警告] 未找到货叉相关的prim，尝试分析整个模型...")
            root_paths = ["/SM_Forklift_C01_01", "/ForkliftC", "/Robot"]
            for root_path in root_paths:
                prim = self.stage.GetPrimAtPath(root_path)
                if prim.IsValid():
                    bbox = self.get_bounding_box(root_path)
                    if bbox:
                        print_info("使用整个模型边界框", "（可能不准确）")
                        return self._estimate_fork_from_bbox(bbox)
            return None
        
        print_info("找到货叉prim数量", len(fork_paths))
        for i, path in enumerate(fork_paths[:5]):
            print_info(f"货叉prim {i+1}", path)
        
        # 分析每个货叉的尺寸
        fork_bboxes = []
        for path in fork_paths:
            bbox = self.get_bounding_box(path)
            if bbox:
                fork_bboxes.append((path, bbox))
        
        if not fork_bboxes:
            print("[错误] 无法获取货叉边界框")
            return None
        
        if len(fork_bboxes) >= 2:
            return self._compute_fork_dimensions_multiple(fork_bboxes)
        else:
            return self._compute_fork_dimensions_single(fork_bboxes[0])
    
    def _find_prims_with_exclude(self, root_path: str, include_patterns: List[str], 
                                  exclude_patterns: List[str]) -> List[str]:
        """查找prim，支持排除模式"""
        if not self.stage:
            return []
        
        prim = self.stage.GetPrimAtPath(root_path)
        if not prim.IsValid():
            return []
        
        found_paths = []
        
        def traverse(prim_path_str):
            prim = self.stage.GetPrimAtPath(prim_path_str)
            if not prim.IsValid():
                return
            
            prim_name = prim.GetName().lower()
            full_path = prim_path_str.lower()
            
            # 检查是否在排除列表中
            excluded = False
            for exclude in exclude_patterns:
                if exclude in prim_name or exclude in full_path:
                    # 只排除顶层匹配，不排除子节点
                    if exclude in prim_name:
                        excluded = True
                        break
            
            if not excluded:
                # 检查是否匹配包含模式
                for pattern in include_patterns:
                    if pattern in prim_name:
                        found_paths.append(prim_path_str)
                        break
            
            for child in prim.GetChildren():
                traverse(str(child.GetPath()))
        
        traverse(root_path)
        return found_paths
    
    def _estimate_fork_from_bbox(self, bbox: Dict) -> GeometryDimensions:
        """从整个模型的边界框估算货叉尺寸（不准确，仅供参考）"""
        size = bbox["size"] * 1000  # 转换为mm
        print("[警告] 使用估算值，可能不准确")
        return GeometryDimensions(
            width=size[1] * 0.1,  # 假设货叉宽度为模型宽度的10%
            height=size[2] * 0.05,  # 假设货叉高度为模型高度的5%
            length=size[0] * 0.3,  # 假设货叉长度为模型长度的30%
            spacing=size[1] * 0.5,  # 假设间距为模型宽度的一半
            collision_type="Estimated"
        )
    
    def _compute_fork_dimensions_single(self, fork_data: Tuple[str, Dict]) -> GeometryDimensions:
        """计算单根货叉的尺寸"""
        path, bbox = fork_data
        size = bbox["size"] * 1000  # 转换为mm
        
        # 获取碰撞形状信息
        collision_info = self.get_collision_shape_info(path)
        
        return GeometryDimensions(
            width=size[1],  # Y方向为宽度
            height=size[2],  # Z方向为高度
            length=size[0],  # X方向为长度
            spacing=0.0,  # 单根货叉无法计算间距
            collision_type=collision_info.get("type", "Unknown")
        )
    
    def _compute_fork_dimensions_multiple(self, fork_bboxes: List[Tuple[str, Dict]]) -> GeometryDimensions:
        """计算多根货叉的尺寸和间距"""
        if len(fork_bboxes) < 2:
            return self._compute_fork_dimensions_single(fork_bboxes[0])
        
        # 计算每根货叉的尺寸（使用平均值）
        widths = []
        heights = []
        lengths = []
        centers_y = []  # Y坐标（横向）
        
        for path, bbox in fork_bboxes:
            size = bbox["size"] * 1000  # mm
            center = (bbox["min"] + bbox["max"]) / 2 * 1000  # mm
            
            widths.append(size[1])
            heights.append(size[2])
            lengths.append(size[0])
            centers_y.append(center[1])
        
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        avg_length = np.mean(lengths)
        
        # 计算间距（中心到中心）
        centers_y.sort()
        spacing = 0.0
        if len(centers_y) >= 2:
            spacing = abs(centers_y[-1] - centers_y[0])  # 最外侧两根的间距
        
        # 获取碰撞形状（使用第一个货叉的）
        collision_info = self.get_collision_shape_info(fork_bboxes[0][0])
        
        return GeometryDimensions(
            width=avg_width,
            height=avg_height,
            length=avg_length,
            spacing=spacing,
            collision_type=collision_info.get("type", "Unknown")
        )


class PalletGeometryAnalyzer(GeometryAnalyzer):
    """托盘几何分析器"""
    
    # 欧标托盘标准尺寸 (mm)
    EUROPALLET_WIDTH = 800
    EUROPALLET_LENGTH = 1200
    EUROPALLET_HEIGHT = 144
    EUROPALLET_POCKET_HEIGHT = 100  # 标准插入孔高度
    EUROPALLET_POCKET_WIDTH = 227  # 标准插入孔宽度（每个孔）
    
    # 扫描控制参数
    MAX_SCAN_DEPTH = 10
    MAX_PRIMS_TO_PROCESS = 100
    
    def extract_pocket_dimensions(self) -> Optional[GeometryDimensions]:
        """提取托盘插入孔尺寸"""
        print_section("提取托盘插入孔几何尺寸")
        
        if not self.stage:
            print("[错误] USD stage 未加载")
            return None
        
        # 方法1：USD几何解析
        result = self._extract_via_geometry_analysis()
        if result:
            return result
        
        # 方法2：名称匹配法（回退）
        print("\n[警告] 几何分析未找到插入孔，尝试名称匹配法...")
        result = self._extract_via_name_matching()
        if result:
            return result
        
        # 方法3：基于整体尺寸估算（最终回退）
        print("\n[警告] 所有方法均失败，使用欧标托盘估算...")
        return self._estimate_from_europallet_ratio()
    
    def _extract_via_geometry_analysis(self) -> Optional[GeometryDimensions]:
        """使用USD几何解析提取插入孔尺寸
        
        策略：
        1. 扫描所有Mesh，按Z高度分组
        2. 识别底板（低Z）和顶板（高Z）
        3. 计算中间空腔区域
        4. 推导插入孔尺寸
        """
        print("\n[方法] 使用USD几何解析...")
        
        # 查找托盘根路径
        root_paths = ["/Pallet", "/World/Pallet", "/SM_Pallet", "/"]
        pallet_root = None
        for root_path in root_paths:
            prim = self.stage.GetPrimAtPath(root_path)
            if prim.IsValid():
                # 检查是否包含几何体
                bbox = self.get_bounding_box(root_path)
                if bbox:
                    pallet_root = root_path
                    print_info("托盘根路径", pallet_root)
                    break
        
        if not pallet_root:
            print("[警告] 未找到托盘根路径")
            return None
        
        # 获取托盘整体边界框
        pallet_bbox = self.get_bounding_box(pallet_root)
        if not pallet_bbox:
            return None
        
        pallet_size = pallet_bbox["size"] * 1000  # mm
        print_info("托盘整体尺寸(mm)", f"X={pallet_size[0]:.1f}, Y={pallet_size[1]:.1f}, Z={pallet_size[2]:.1f}")
        
        # 扫描所有Mesh prim
        meshes = self._scan_meshes(pallet_root)
        print_info("扫描到的Mesh数量", len(meshes))
        
        if len(meshes) == 0:
            print("[警告] 未找到任何Mesh")
            return None
        
        # 如果只有一个Mesh，使用单一Mesh处理
        if len(meshes) == 1:
            print("[信息] 托盘只有一个整体Mesh，使用比例估算...")
            return self._estimate_from_single_mesh(pallet_bbox)
        
        # 分析Mesh的Z高度分布
        return self._analyze_mesh_layers(meshes, pallet_bbox)
    
    def _scan_meshes(self, root_path: str, max_depth: int = None, 
                     max_prims: int = None) -> List[Dict]:
        """扫描指定路径下的所有Mesh prim
        
        Args:
            root_path: 根路径
            max_depth: 最大扫描深度
            max_prims: 最大扫描prim数量
        
        Returns:
            Mesh信息列表 [{path, bbox, center, size}, ...]
        """
        if max_depth is None:
            max_depth = self.MAX_SCAN_DEPTH
        if max_prims is None:
            max_prims = self.MAX_PRIMS_TO_PROCESS
        
        meshes = []
        count = [0]  # 使用列表以便在闭包中修改
        
        def traverse(prim, depth=0):
            if depth > max_depth or count[0] >= max_prims:
                return
            
            count[0] += 1
            
            # 检查是否是Mesh或Boundable
            is_mesh = UsdGeom.Mesh(prim)
            if is_mesh:
                bbox = self._get_single_prim_bbox(prim)
                if bbox:
                    meshes.append({
                        "path": str(prim.GetPath()),
                        "name": prim.GetName(),
                        "bbox": bbox,
                        "center": (bbox["min"] + bbox["max"]) / 2,
                        "size": bbox["size"]
                    })
            
            # 递归子节点
            for child in prim.GetChildren():
                traverse(child, depth + 1)
        
        prim = self.stage.GetPrimAtPath(root_path)
        if prim.IsValid():
            traverse(prim)
        
        return meshes
    
    def _get_single_prim_bbox(self, prim) -> Optional[Dict]:
        """获取单个prim的边界框（不递归子节点）"""
        boundable = UsdGeom.Boundable(prim)
        if not boundable:
            return None
        
        bbox = boundable.ComputeWorldBound(Usd.TimeCode.Default(), "default")
        if not bbox:
            return None
        
        bbox_range = bbox.ComputeAlignedBox()
        if not bbox_range or bbox_range.IsEmpty():
            return None
        
        box_min = bbox_range.GetMin()
        box_max = bbox_range.GetMax()
        
        return {
            "min": np.array([box_min[0], box_min[1], box_min[2]]),
            "max": np.array([box_max[0], box_max[1], box_max[2]]),
            "size": np.array([box_max[0] - box_min[0], 
                            box_max[1] - box_min[1], 
                            box_max[2] - box_min[2]])
        }
    
    def _analyze_mesh_layers(self, meshes: List[Dict], pallet_bbox: Dict) -> Optional[GeometryDimensions]:
        """分析Mesh的Z高度分布，识别底板、顶板和空腔
        
        标准托盘结构：
        - 顶板（deck boards）：位于顶部
        - 纵梁/隔板（stringers/blocks）：中间支撑
        - 底板（bottom boards）：位于底部
        - 插入孔（pockets）：顶板和底板之间的空间
        """
        print("\n[分析] Mesh层级分布...")
        
        pallet_z_min = pallet_bbox["min"][2]
        pallet_z_max = pallet_bbox["max"][2]
        pallet_height = pallet_bbox["size"][2] * 1000  # mm
        
        # 按Z中心位置分组
        z_centers = []
        for mesh in meshes:
            z_center = mesh["center"][2]
            z_min = mesh["bbox"]["min"][2]
            z_max = mesh["bbox"]["max"][2]
            height = mesh["size"][2] * 1000  # mm
            
            z_centers.append({
                "mesh": mesh,
                "z_center": z_center,
                "z_min": z_min,
                "z_max": z_max,
                "height_mm": height,
                # 相对位置（0=底部, 1=顶部）
                "rel_z": (z_center - pallet_z_min) / (pallet_z_max - pallet_z_min) if pallet_z_max > pallet_z_min else 0.5
            })
        
        # 按相对Z位置排序
        z_centers.sort(key=lambda x: x["rel_z"])
        
        # 打印分布
        print(f"\nMesh Z分布（托盘高度={pallet_height:.1f}mm）:")
        for i, item in enumerate(z_centers[:10]):  # 只显示前10个
            mesh = item["mesh"]
            print(f"  {i+1}. {mesh['name']}: rel_z={item['rel_z']:.2f}, "
                  f"z_range=[{item['z_min']*1000:.1f}, {item['z_max']*1000:.1f}]mm, "
                  f"height={item['height_mm']:.1f}mm")
        
        # 识别底板和顶板
        # 策略：rel_z < 0.3 的是底板，rel_z > 0.7 的是顶板
        bottom_meshes = [x for x in z_centers if x["rel_z"] < 0.3]
        top_meshes = [x for x in z_centers if x["rel_z"] > 0.7]
        middle_meshes = [x for x in z_centers if 0.3 <= x["rel_z"] <= 0.7]
        
        print(f"\n层级分类: 底板={len(bottom_meshes)}, 中间={len(middle_meshes)}, 顶板={len(top_meshes)}")
        
        # 计算插入孔尺寸
        if bottom_meshes and top_meshes:
            # 底板顶部
            bottom_top = max(x["z_max"] for x in bottom_meshes)
            # 顶板底部
            top_bottom = min(x["z_min"] for x in top_meshes)
            
            pocket_height = (top_bottom - bottom_top) * 1000  # mm
            print_info("计算得到的插入孔高度", f"{pocket_height:.1f} mm")
            
            if pocket_height > 0:
                # 计算深度（X方向）
                pallet_depth = pallet_bbox["size"][0] * 1000  # mm
                
                # 计算宽度和间距
                # 假设托盘宽度方向（Y）有多个插入孔
                pallet_width = pallet_bbox["size"][1] * 1000  # mm
                
                # 标准欧标托盘：3个插入孔，宽度约227mm，间距约400mm
                # 这里使用比例估算
                pocket_width = pallet_width / 3.5  # 每个孔约占宽度的1/3.5
                pocket_spacing = pallet_width / 2  # 两侧孔中心距
                
                return GeometryDimensions(
                    width=pocket_width,
                    height=pocket_height,
                    length=pallet_depth * 0.95,  # 深度约为托盘长度的95%
                    spacing=pocket_spacing,
                    depth=pallet_depth * 0.95,
                    collision_type="Geometry_Analysis"
                )
        
        # 如果无法明确区分底板和顶板，使用比例估算
        print("[警告] 无法明确区分底板和顶板，使用比例估算...")
        return self._estimate_from_single_mesh(pallet_bbox)
    
    def _estimate_from_single_mesh(self, pallet_bbox: Dict) -> GeometryDimensions:
        """从单一Mesh的边界框估算插入孔尺寸
        
        基于欧标托盘比例：
        - 插入孔高度 ≈ 总高度 × 0.69 (100/144)
        - 插入孔宽度 ≈ 总宽度 × 0.28 (227/800)
        """
        size = pallet_bbox["size"] * 1000  # mm
        
        # 使用欧标比例估算
        ratio_height = self.EUROPALLET_POCKET_HEIGHT / self.EUROPALLET_HEIGHT  # 约0.69
        ratio_width = self.EUROPALLET_POCKET_WIDTH / self.EUROPALLET_WIDTH  # 约0.28
        
        pocket_height = size[2] * ratio_height
        pocket_width = size[1] * ratio_width
        pocket_depth = size[0] * 0.95  # 深度约为长度的95%
        pocket_spacing = size[1] / 2  # 两侧孔中心距约为宽度的一半
        
        print(f"[估算] 基于欧标比例: 高度比={ratio_height:.2f}, 宽度比={ratio_width:.2f}")
        print_info("估算插入孔高度", f"{pocket_height:.1f} mm")
        print_info("估算插入孔宽度", f"{pocket_width:.1f} mm")
        
        return GeometryDimensions(
            width=pocket_width,
            height=pocket_height,
            length=pocket_depth,
            spacing=pocket_spacing,
            depth=pocket_depth,
            collision_type="Single_Mesh_Estimate"
        )
    
    def _estimate_from_europallet_ratio(self) -> GeometryDimensions:
        """使用欧标托盘固定估算值"""
        print("[估算] 使用欧标托盘标准尺寸")
        return GeometryDimensions(
            width=self.EUROPALLET_POCKET_WIDTH,
            height=self.EUROPALLET_POCKET_HEIGHT,
            length=self.EUROPALLET_LENGTH * 0.95,
            spacing=self.EUROPALLET_WIDTH / 2,
            depth=self.EUROPALLET_LENGTH * 0.95,
            collision_type="Europallet_Standard"
        )
    
    def _extract_via_name_matching(self) -> Optional[GeometryDimensions]:
        """使用名称匹配法提取插入孔尺寸"""
        print("\n[方法] 使用名称匹配法...")
        
        pocket_patterns = ["pocket", "gap", "opening", "slot", "hole", "void", "cavity"]
        root_paths = ["/Pallet", "/World/Pallet", "/SM_Pallet", "/"]
        
        pocket_paths = []
        for root_path in root_paths:
            prim = self.stage.GetPrimAtPath(root_path)
            if prim.IsValid():
                pocket_paths = self.find_prims_by_name(root_path, pocket_patterns)
                if pocket_paths:
                    break
        
        if not pocket_paths:
            return None
        
        print_info("找到插入孔prim数量", len(pocket_paths))
        for i, path in enumerate(pocket_paths[:5]):
            print_info(f"插入孔prim {i+1}", path)
        
        pocket_bboxes = []
        for path in pocket_paths:
            bbox = self.get_bounding_box(path)
            if bbox:
                pocket_bboxes.append((path, bbox))
        
        if not pocket_bboxes:
            return None
        
        if len(pocket_bboxes) >= 2:
            return self._compute_pocket_dimensions_multiple(pocket_bboxes)
        else:
            return self._compute_pocket_dimensions_single(pocket_bboxes[0])
    
    def _compute_pocket_dimensions_single(self, pocket_data: Tuple[str, Dict]) -> GeometryDimensions:
        """计算单个插入孔的尺寸"""
        path, bbox = pocket_data
        size = bbox["size"] * 1000  # mm
        
        collision_info = self.get_collision_shape_info(path)
        
        return GeometryDimensions(
            width=size[1],
            height=size[2],
            length=size[0],
            spacing=0.0,
            depth=size[0],
            collision_type=collision_info.get("type", "Unknown")
        )
    
    def _compute_pocket_dimensions_multiple(self, pocket_bboxes: List[Tuple[str, Dict]]) -> GeometryDimensions:
        """计算多个插入孔的尺寸和间距"""
        if len(pocket_bboxes) < 2:
            return self._compute_pocket_dimensions_single(pocket_bboxes[0])
        
        widths = []
        heights = []
        depths = []
        centers_y = []
        
        for path, bbox in pocket_bboxes:
            size = bbox["size"] * 1000  # mm
            center = (bbox["min"] + bbox["max"]) / 2 * 1000  # mm
            
            widths.append(size[1])
            heights.append(size[2])
            depths.append(size[0])
            centers_y.append(center[1])
        
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        avg_depth = np.mean(depths)
        
        centers_y.sort()
        spacing = abs(centers_y[-1] - centers_y[0]) if len(centers_y) >= 2 else 0.0
        
        collision_info = self.get_collision_shape_info(pocket_bboxes[0][0])
        
        return GeometryDimensions(
            width=avg_width,
            height=avg_height,
            length=avg_depth,
            spacing=spacing,
            depth=avg_depth,
            collision_type=collision_info.get("type", "Unknown")
        )


def check_compatibility(fork_dims: GeometryDimensions, pallet_dims: GeometryDimensions, 
                       clearance_mm: float = 20.0, spacing_tolerance_mm: float = 50.0) -> CompatibilityResult:
    """检查几何兼容性"""
    result = CompatibilityResult()
    
    # 检查宽度兼容性（需要留有余量）
    if fork_dims.width + clearance_mm <= pallet_dims.width:
        result.width_compatible = True
    else:
        result.width_compatible = False
        result.issues.append(
            f"宽度不兼容: 货叉宽度({fork_dims.width:.2f}mm) + 间隙({clearance_mm}mm) = "
            f"{fork_dims.width + clearance_mm:.2f}mm > 插入孔宽度({pallet_dims.width:.2f}mm)"
        )
    
    # 检查高度兼容性
    if fork_dims.height <= pallet_dims.height:
        result.height_compatible = True
    else:
        result.height_compatible = False
        result.issues.append(
            f"高度不兼容: 货叉高度({fork_dims.height:.2f}mm) > 插入孔高度({pallet_dims.height:.2f}mm)"
        )
    
    # 检查间距兼容性
    if fork_dims.spacing > 0 and pallet_dims.spacing > 0:
        spacing_diff = abs(fork_dims.spacing - pallet_dims.spacing)
        if spacing_diff <= spacing_tolerance_mm:
            result.spacing_compatible = True
        else:
            result.spacing_compatible = False
            result.issues.append(
                f"间距不匹配: 货叉间距({fork_dims.spacing:.2f}mm) vs 插入孔间距({pallet_dims.spacing:.2f}mm), "
                f"差异={spacing_diff:.2f}mm > 容差({spacing_tolerance_mm}mm)"
            )
    else:
        result.spacing_compatible = True  # 如果无法计算间距，假设兼容
        if fork_dims.spacing == 0:
            result.issues.append("警告: 无法计算货叉间距")
        if pallet_dims.spacing == 0:
            result.issues.append("警告: 无法计算插入孔间距")
    
    return result


class CollisionTester:
    """碰撞测试器"""
    
    def __init__(self):
        self.env = None
        self.cfg = None
        
    def initialize_environment(self):
        """初始化环境"""
        self.cfg = ForkliftPalletInsertLiftEnvCfg()
        self.cfg.scene.num_envs = 1
        self.env = ForkliftPalletInsertLiftEnv(self.cfg)
        self.env.reset()
        return True
    
    def _quat_to_yaw(self, q: torch.Tensor) -> torch.Tensor:
        """从四元数提取偏航角"""
        w, x, y, z = q.unbind(-1)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)
    
    def _yaw_to_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        """从偏航角转换为四元数 (w, x, y, z)"""
        half = yaw * 0.5
        return torch.stack([
            torch.cos(half),
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half)
        ], dim=-1)
    
    def set_robot_ideal_position(self, distance_from_front=0.5):
        """设置叉车到理想对齐位置"""
        pallet_pos = self.env.pallet.data.root_pos_w[0]
        pallet_yaw = self._quat_to_yaw(self.env.pallet.data.root_quat_w[0])
        
        pallet_front_x = pallet_pos[0] - self.cfg.pallet_depth_m * 0.5
        
        # 计算货叉尖端相对于root的偏移
        current_tip = self.env._compute_fork_tip()[0]
        current_root = self.env.robot.data.root_pos_w[0]
        fork_offset_x = current_tip[0] - current_root[0]
        
        ideal_tip_x = pallet_front_x - distance_from_front
        ideal_root_x = ideal_tip_x - fork_offset_x
        
        ideal_pos = torch.tensor([ideal_root_x, pallet_pos[1], 0.1], device=self.env.device)
        ideal_quat = self._yaw_to_quat(pallet_yaw)
        
        env_ids = torch.tensor([0], device=self.env.device)
        self.env._write_root_pose(self.env.robot, ideal_pos.unsqueeze(0), ideal_quat.unsqueeze(0), env_ids)
        
        zeros3 = torch.zeros((1, 3), device=self.env.device)
        self.env._write_root_vel(self.env.robot, zeros3, zeros3, env_ids)
        
        self.env.scene.write_data_to_sim()
        self.env.scene.update(self.env.cfg.sim.dt)
        
        # 更新缓存
        tip = self.env._compute_fork_tip()
        insert_depth = torch.clamp(tip[:, 0] - self.env._pallet_front_x, min=0.0)
        self.env._last_insert_depth[0] = insert_depth[0]
        
        root_pos = self.env.robot.data.root_pos_w[0:1]
        pallet_pos = self.env.pallet.data.root_pos_w[0:1]
        lateral_err = torch.abs(pallet_pos[:, 1] - root_pos[:, 1])
        yaw_robot = self._quat_to_yaw(self.env.robot.data.root_quat_w[0:1])
        yaw_pallet = self._quat_to_yaw(self.env.pallet.data.root_quat_w[0:1])
        yaw_err = torch.abs((yaw_pallet - yaw_robot + math.pi) % (2 * math.pi) - math.pi)
        E_align = lateral_err + yaw_err * 0.1
        self.env._last_E_align[0] = E_align[0]
        
        dist_front = tip[:, 0] - self.env._pallet_front_x
        self.env._last_dist_front[0] = dist_front[0]
        self.env._last_lift_pos[0] = self.env._joint_pos[0, self.env._lift_id]
        self.env.actions[0] = 0.0
    
    def test_collision_insertion(self) -> Dict:
        """测试理想对齐条件下的碰撞插入"""
        print_section("碰撞测试：理想对齐插入测试")
        
        # 设置理想对齐位置
        self.set_robot_ideal_position(distance_from_front=0.5)
        
        initial_pos = self.env.robot.data.root_pos_w[0].clone()
        initial_tip = self.env._compute_fork_tip()[0].clone()
        
        pallet_pos = self.env.pallet.data.root_pos_w[0]
        pallet_front_x = pallet_pos[0] - self.cfg.pallet_depth_m * 0.5
        
        print(f"\n初始状态:")
        print_info("叉车位置", f"({initial_pos[0]:.4f}, {initial_pos[1]:.4f}, {initial_pos[2]:.4f})")
        print_info("货叉尖端位置", f"({initial_tip[0]:.4f}, {initial_tip[1]:.4f}, {initial_tip[2]:.4f})")
        print_info("托盘前部x", f"{pallet_front_x:.4f}")
        print_info("初始距离托盘前部", f"{initial_tip[0] - pallet_front_x:.4f}m")
        
        # 逐步向前移动，检测碰撞
        step_size = 0.01  # 每步1cm
        max_steps = 100
        collision_detected = False
        collision_step = -1
        collision_pos = None
        
        print(f"\n开始逐步推进测试（步长={step_size*100:.1f}cm，最大步数={max_steps}）...")
        
        for step in range(max_steps):
            # 记录移动前的位置
            before_pos = self.env.robot.data.root_pos_w[0].clone()
            before_tip = self.env._compute_fork_tip()[0].clone()
            
            # 向前移动
            new_pos = before_pos.clone()
            new_pos[0] += step_size  # X方向向前
            
            env_ids = torch.tensor([0], device=self.env.device)
            self.env._write_root_pose(self.env.robot, new_pos.unsqueeze(0), 
                                     self.env.robot.data.root_quat_w[0:1], env_ids)
            
            # 同步到sim
            self.env.scene.write_data_to_sim()
            self.env.scene.update(self.env.cfg.sim.dt)
            
            # 检查实际位置变化
            after_pos = self.env.robot.data.root_pos_w[0]
            after_tip = self.env._compute_fork_tip()[0]
            
            pos_delta = after_pos - before_pos
            tip_delta = after_tip - before_tip
            
            # 如果位置变化远小于预期，可能发生碰撞
            expected_delta_x = step_size
            actual_delta_x = pos_delta[0].item()
            
            if abs(actual_delta_x) < expected_delta_x * 0.5:  # 实际位移小于预期的50%
                collision_detected = True
                collision_step = step
                collision_pos = after_pos.clone()
                print(f"\n[碰撞检测] 步数 {step}: 预期位移={expected_delta_x:.4f}m, 实际位移={actual_delta_x:.4f}m")
                print_info("碰撞位置", f"({after_pos[0]:.4f}, {after_pos[1]:.4f}, {after_pos[2]:.4f})")
                print_info("货叉尖端位置", f"({after_tip[0]:.4f}, {after_tip[1]:.4f}, {after_tip[2]:.4f})")
                break
            
            # 每10步打印一次进度
            if step % 10 == 0:
                dist_to_front = after_tip[0].item() - pallet_front_x
                print(f"  步数 {step}: 距离托盘前部={dist_to_front:.4f}m, 位移={actual_delta_x:.4f}m")
        
        # 最终状态
        final_pos = self.env.robot.data.root_pos_w[0]
        final_tip = self.env._compute_fork_tip()[0]
        final_dist_to_front = final_tip[0].item() - pallet_front_x
        
        print(f"\n最终状态:")
        print_info("叉车位置", f"({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
        print_info("货叉尖端位置", f"({final_tip[0]:.4f}, {final_tip[1]:.4f}, {final_tip[2]:.4f})")
        print_info("距离托盘前部", f"{final_dist_to_front:.4f}m")
        
        total_displacement = torch.norm(final_pos[:2] - initial_pos[:2]).item()
        print_info("总位移", f"{total_displacement:.4f}m")
        
        result = {
            "collision_detected": collision_detected,
            "collision_step": collision_step,
            "collision_pos": collision_pos.cpu().numpy() if collision_pos is not None else None,
            "final_dist_to_front": final_dist_to_front,
            "total_displacement": total_displacement,
            "insertion_successful": final_dist_to_front > 0.01  # 插入深度超过1cm
        }
        
        return result


def diagnose_usd_scale_and_units(stage, usd_path: str, asset_name: str):
    """诊断 USD 文件的单位和缩放信息"""
    print_section(f"{asset_name} USD 单位与缩放诊断")
    
    if not stage:
        print("[错误] Stage 未加载")
        return None
    
    # 获取 metersPerUnit
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    print_info("metersPerUnit", f"{meters_per_unit}")
    print_info("单位解释", f"1 USD 单位 = {meters_per_unit} 米 ({meters_per_unit * 100:.2f} cm)")
    
    # 获取根 prim 的缩放信息
    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        # 尝试获取第一个子 prim
        root_prim = stage.GetPseudoRoot()
        for child in root_prim.GetChildren():
            if child.IsValid():
                root_prim = child
                break
    
    if root_prim and root_prim.IsValid():
        print_info("根 Prim", root_prim.GetPath())
        
        # 检查 xformOp:scale
        xformable = UsdGeom.Xformable(root_prim)
        if xformable:
            xform_ops = xformable.GetOrderedXformOps()
            scale_found = False
            for op in xform_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_value = op.Get()
                    print_info("xformOp:scale", f"({scale_value[0]:.4f}, {scale_value[1]:.4f}, {scale_value[2]:.4f})")
                    scale_found = True
            if not scale_found:
                print_info("xformOp:scale", "无（使用默认缩放 1.0）")
    
    # 计算整体边界框（以米为单位）
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    root_prim = stage.GetPseudoRoot()
    bbox = bbox_cache.ComputeWorldBound(root_prim)
    bbox_range = bbox.ComputeAlignedBox()
    
    if bbox_range and not bbox_range.IsEmpty():
        size = bbox_range.GetSize()
        min_pt = bbox_range.GetMin()
        max_pt = bbox_range.GetMax()
        
        # 转换为米
        size_m = [s * meters_per_unit for s in size]
        min_m = [p * meters_per_unit for p in min_pt]
        max_m = [p * meters_per_unit for p in max_pt]
        
        print(f"\n{asset_name} 整体边界框（已转换为米）:")
        print_info("尺寸 (X,Y,Z)", f"({size_m[0]:.4f}, {size_m[1]:.4f}, {size_m[2]:.4f}) m")
        print_info("尺寸 (mm)", f"({size_m[0]*1000:.1f}, {size_m[1]*1000:.1f}, {size_m[2]*1000:.1f}) mm")
        print_info("最小点", f"({min_m[0]:.4f}, {min_m[1]:.4f}, {min_m[2]:.4f}) m")
        print_info("最大点", f"({max_m[0]:.4f}, {max_m[1]:.4f}, {max_m[2]:.4f}) m")
        print_info("Z 范围 (高度)", f"{min_m[2]:.4f} ~ {max_m[2]:.4f} m")
        
        return {
            "meters_per_unit": meters_per_unit,
            "size_m": size_m,
            "min_m": min_m,
            "max_m": max_m
        }
    else:
        print("[警告] 无法计算边界框")
        return None


def main():
    """主函数"""
    print("=" * 80)
    print("叉车托盘几何兼容性验证")
    print("=" * 80)
    
    # 加载USD文件
    forklift_usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/ForkliftC/forklift_c.usd"
    pallet_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Pallet/pallet.usd"
    
    print(f"\n叉车USD路径: {forklift_usd_path}")
    print(f"托盘USD路径: {pallet_usd_path}")
    
    # 分析叉车几何
    forklift_analyzer = ForkliftGeometryAnalyzer()
    if not forklift_analyzer.load_usd(forklift_usd_path):
        print("[错误] 无法加载叉车USD文件")
        simulation_app.close()
        return
    
    # 诊断叉车 USD 单位与缩放
    forklift_diag = diagnose_usd_scale_and_units(forklift_analyzer.stage, forklift_usd_path, "叉车")
    
    fork_dims = forklift_analyzer.extract_fork_dimensions()
    
    # 分析托盘几何
    pallet_analyzer = PalletGeometryAnalyzer()
    if not pallet_analyzer.load_usd(pallet_usd_path):
        print("[错误] 无法加载托盘USD文件")
        simulation_app.close()
        return
    
    # 诊断托盘 USD 单位与缩放
    pallet_diag = diagnose_usd_scale_and_units(pallet_analyzer.stage, pallet_usd_path, "托盘")
    
    pallet_dims = pallet_analyzer.extract_pocket_dimensions()
    
    # 关键尺寸对比诊断
    print_section("关键尺寸对比诊断")
    if forklift_diag and pallet_diag:
        print("\n尺寸比例分析:")
        forklift_x = forklift_diag["size_m"][0]
        pallet_x = pallet_diag["size_m"][0]
        print_info("叉车长度 (X)", f"{forklift_x:.4f} m ({forklift_x*1000:.1f} mm)")
        print_info("托盘深度 (X)", f"{pallet_x:.4f} m ({pallet_x*1000:.1f} mm)")
        print_info("叉车/托盘比例", f"{forklift_x/pallet_x:.2f}x" if pallet_x > 0 else "无法计算")
        
        forklift_y = forklift_diag["size_m"][1]
        pallet_y = pallet_diag["size_m"][1]
        print_info("叉车宽度 (Y)", f"{forklift_y:.4f} m ({forklift_y*1000:.1f} mm)")
        print_info("托盘宽度 (Y)", f"{pallet_y:.4f} m ({pallet_y*1000:.1f} mm)")
        print_info("叉车/托盘比例", f"{forklift_y/pallet_y:.2f}x" if pallet_y > 0 else "无法计算")
        
        print("\n高度对齐分析:")
        forklift_z_min = forklift_diag["min_m"][2]
        forklift_z_max = forklift_diag["max_m"][2]
        pallet_z_min = pallet_diag["min_m"][2]
        pallet_z_max = pallet_diag["max_m"][2]
        
        # 托盘 pocket 高度估算（约为托盘高度的下半部分）
        pallet_height = pallet_z_max - pallet_z_min
        pocket_z_bottom = pallet_z_min
        pocket_z_top = pallet_z_min + pallet_height * 0.5  # 假设 pocket 在下半部分
        
        print_info("叉车 Z 范围", f"{forklift_z_min:.4f} ~ {forklift_z_max:.4f} m")
        print_info("托盘 Z 范围", f"{pallet_z_min:.4f} ~ {pallet_z_max:.4f} m")
        print_info("托盘 pocket 估算高度", f"{pocket_z_bottom:.4f} ~ {pocket_z_top:.4f} m")
        
        # 检查是否需要调整
        if pallet_y < forklift_y * 0.5:
            print("\n⚠️  警告：托盘宽度远小于叉车宽度，货叉可能无法插入！")
            suggested_scale = forklift_y / pallet_y * 1.2  # 建议放大到叉车的 1.2 倍
            print_info("建议托盘缩放", f"约 {suggested_scale:.2f}x")
    
    # 检查兼容性
    compatibility = None
    if fork_dims and pallet_dims:
        print_section("几何兼容性检查")
        compatibility = check_compatibility(fork_dims, pallet_dims)
        
        # 打印结果
        print("\n货叉尺寸:")
        print_info("单根宽度", f"{fork_dims.width:.2f} mm")
        print_info("货叉间距", f"{fork_dims.spacing:.2f} mm" if fork_dims.spacing > 0 else "无法计算")
        print_info("货叉高度", f"{fork_dims.height:.2f} mm")
        print_info("货叉长度", f"{fork_dims.length:.2f} mm")
        print_info("碰撞形状", fork_dims.collision_type)
        
        print("\n托盘插入孔尺寸:")
        print_info("单孔宽度", f"{pallet_dims.width:.2f} mm")
        print_info("插入孔间距", f"{pallet_dims.spacing:.2f} mm" if pallet_dims.spacing > 0 else "无法计算")
        print_info("插入孔高度", f"{pallet_dims.height:.2f} mm")
        print_info("插入孔深度", f"{pallet_dims.depth:.2f} mm")
        print_info("碰撞形状", pallet_dims.collision_type)
        
        print("\n兼容性检查:")
        status_width = "✓" if compatibility.width_compatible else "✗"
        status_height = "✓" if compatibility.height_compatible else "✗"
        status_spacing = "✓" if compatibility.spacing_compatible else "✗"
        
        print_info(f"[{status_width}] 宽度兼容", 
                  f"货叉宽度({fork_dims.width:.2f}mm) < 插入孔宽度({pallet_dims.width:.2f}mm)")
        print_info(f"[{status_height}] 高度兼容", 
                  f"货叉高度({fork_dims.height:.2f}mm) < 插入孔高度({pallet_dims.height:.2f}mm)")
        if fork_dims.spacing > 0 and pallet_dims.spacing > 0:
            print_info(f"[{status_spacing}] 间距匹配", 
                      f"货叉间距({fork_dims.spacing:.2f}mm) ≈ 插入孔间距({pallet_dims.spacing:.2f}mm)")
        
        if compatibility.issues:
            print("\n发现的问题:")
            for issue in compatibility.issues:
                print(f"  ⚠️  {issue}")
        else:
            print("\n✓ 所有兼容性检查通过！")
    
    # 碰撞测试
    print_section("实际碰撞测试")
    collision_tester = CollisionTester()
    if collision_tester.initialize_environment():
        collision_result = collision_tester.test_collision_insertion()
        
        print("\n碰撞测试结果:")
        status_collision = "✗" if collision_result["collision_detected"] else "✓"
        status_insertion = "✓" if collision_result["insertion_successful"] else "✗"
        
        print_info(f"[{status_collision}] 碰撞检测", 
                  "检测到碰撞" if collision_result["collision_detected"] else "未检测到碰撞")
        if collision_result["collision_detected"]:
            print_info("碰撞步数", collision_result["collision_step"])
            if collision_result["collision_pos"] is not None:
                pos = collision_result["collision_pos"]
                print_info("碰撞位置", f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        
        print_info(f"[{status_insertion}] 插入成功", 
                  "成功插入" if collision_result["insertion_successful"] else "插入失败")
        print_info("最终距离托盘前部", f"{collision_result['final_dist_to_front']:.4f}m")
        print_info("总位移", f"{collision_result['total_displacement']:.4f}m")
        
        collision_tester.env.close()
    
    # 生成最终报告和建议
    print_section("诊断报告和建议")
    
    all_compatible = compatibility and compatibility.width_compatible and compatibility.height_compatible and compatibility.spacing_compatible
    insertion_ok = collision_result.get("insertion_successful", False) if 'collision_result' in locals() else False
    
    if all_compatible and insertion_ok:
        print("\n✓ 几何兼容性验证通过！")
        print("  - 货叉尺寸与托盘插入孔兼容")
        print("  - 实际碰撞测试中成功插入")
        print("\n建议:")
        print("  - 可以继续进行RL训练")
        print("  - 如果训练中仍有问题，可能是控制策略或奖励函数的问题")
    else:
        print("\n⚠️  发现问题：")
        if not all_compatible:
            print("  - 几何尺寸不兼容")
            if compatibility:
                for issue in compatibility.issues:
                    print(f"    • {issue}")
        if not insertion_ok:
            print("  - 实际碰撞测试中插入失败")
            if collision_result.get("collision_detected"):
                print("    • 检测到碰撞阻止插入")
        
        print("\n可能的修复方案:")
        print("  1. 调整碰撞形状：使用更简化的碰撞形状（Box vs Mesh）")
        print("  2. 缩放资产：调整叉车或托盘的缩放比例")
        print("  3. 修改USD文件：如果有权限，修改碰撞mesh或几何尺寸")
        print("  4. 禁用特定碰撞：使用碰撞过滤排除货叉-托盘碰撞")
        print("  5. 检查USD文件中的实际几何尺寸，确认是否与预期一致")
    
    # 关闭 Isaac Sim
    simulation_app.close()


if __name__ == "__main__":
    main()
