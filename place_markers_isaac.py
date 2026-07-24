#!/usr/bin/env python3
"""
place_markers_isaac.py — 在 Isaac 场景里叠加 Bullet 场景的"幽灵标记", 用于对齐树

读取 scene_layout.json(example_grasp_rrt 导出, 机器人基座 frame 坐标),
按 Isaac 里机器人基座 prim 的当前世界位姿变换到世界坐标, 生成纯视觉
(无碰撞)标记球:
  红色  = 硬障碍 (树干轴线采样点 + 苹果上方遮挡球) —— 碰到判 FAIL 的东西
  绿色  = 可穿透枝条 (每根枝条的 p0 / 中点 / p1)
  黄色  = 苹果

用法(Isaac Sim python, 不要加 --headless, 需要在 GUI 里手动对齐):
  ./python.sh place_markers_isaac.py --usd scene.usd --robot-prim /World/panda

流程:
  1. 脚本打开场景、生成标记后保持 GUI 运行;
  2. 在视口里平移/旋转你的树 prim, 直到树干/枝条/苹果与标记重合
     (优先级: 苹果 > 树干轴 > 遮挡球 > 枝条);
  3. File > Save 保存场景;
  4. 删除 /World/calib_markers 这个 prim(或保留, 它无碰撞, 不影响回放判定,
     但会挡视线);关掉窗口。

注意: 标记随机器人基座位姿走 —— 机器人在地面还是悬空都无所谓,
      对齐的是树相对基座的位姿, 这正是回放实验需要的。
"""

import argparse
import json
import sys

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--robot-prim", required=True)
ap.add_argument("--layout", default="scene_layout.json")
ap.add_argument("--marker-root", default="/World/calib_markers")
args = ap.parse_args()

simulation_app = SimulationApp({"headless": False})

import numpy as np

try:                                            # 4.5+
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleXFormPrim as XFormPrim
    from isaacsim.core.objects import VisualSphere
    from isaacsim.core.utils.stage import open_stage
except ImportError:                             # 2022.2 – 4.2
    from omni.isaac.core import World
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.objects import VisualSphere
    from omni.isaac.core.utils.stage import open_stage

RED = np.array([0.9, 0.1, 0.1])
GREEN = np.array([0.1, 0.8, 0.1])
YELLOW = np.array([0.95, 0.85, 0.1])


def quat_wxyz_to_rot(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def main():
    with open(args.layout) as f:
        layout = json.load(f)

    open_stage(args.usd)
    world = World(stage_units_in_meters=1.0)

    base = XFormPrim(args.robot_prim)
    world.reset()
    base_pos, base_quat = base.get_world_pose()
    R = quat_wxyz_to_rot(base_quat)
    base_pos = np.asarray(base_pos, dtype=float)
    print(f"[Info] 机器人基座世界位姿: pos={base_pos}, quat(wxyz)={base_quat}")

    def to_world(p_local):
        return base_pos + R @ np.asarray(p_local, dtype=float)

    idx = [0]

    def add_marker(p_local, radius, color, name):
        idx[0] += 1
        world.scene.add(VisualSphere(
            prim_path=f"{args.marker_root}/{name}_{idx[0]:03d}",
            name=f"marker_{idx[0]:03d}",
            position=to_world(p_local),
            radius=float(radius),
            color=color))

    def add_cylinder_markers(p0, p1, radius, color, name, spacing=0.25):
        """沿圆柱轴线撒标记球; p0==p1 视为球体, 只放一个"""
        p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
        length = np.linalg.norm(p1 - p0)
        if length < 1e-6:
            add_marker(p0, radius, color, name)
            return
        n = max(2, int(length / spacing) + 1)
        for k in range(n):
            add_marker(p0 + (p1 - p0) * k / (n - 1), radius, color, name)

    if "cylinders" in layout:
        # benchmark_grasp_rrt 导出的通用格式 (含 --tree6 等所有场景变体)
        for i, c in enumerate(layout["cylinders"]):
            hard = c["class"] == "hard"
            # 细枝半径太小看不见, 标记球半径下限 1.5cm
            r_vis = max(c["radius"], 0.015)
            add_cylinder_markers(c["p0"], c["p1"], r_vis,
                                 RED if hard else GREEN,
                                 f"{'hard' if hard else 'soft'}{i:03d}",
                                 spacing=0.10)
    else:
        # example_grasp_rrt 导出的命名格式
        tr = layout["trunk"]
        add_cylinder_markers(tr["p0"], tr["p1"], tr["radius"], RED, "trunk")
        ov = layout["overhead_obstacle"]
        add_marker(ov["center"], ov["radius"], RED, "overhead")
        for i, br in enumerate(layout["branches"]):
            b0, b1 = np.array(br["p0"]), np.array(br["p1"])
            for p in (b0, (b0 + b1) / 2, b1):
                add_marker(p, br["radius"], GREEN, f"branch{i:02d}")

    # 苹果
    ap_ = layout["apple"]
    add_marker(ap_["center"], ap_["radius"], YELLOW, "apple")

    print(f"[Info] 已生成 {idx[0]} 个标记 (root: {args.marker_root})")
    print("[Info] 现在在 GUI 里移动/旋转你的树, 使其与标记重合;")
    print("       对齐优先级: 黄(苹果) > 红(树干轴/遮挡球) > 绿(枝条)。")
    print("       完成后 File > Save, 然后删除标记 prim, 关闭窗口即可。")

    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
