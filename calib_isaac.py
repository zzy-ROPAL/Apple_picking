#!/usr/bin/env python3
"""
calib_isaac.py — Isaac Sim 侧 FK 标定脚本

用途:
  在你已有的 USD 场景中, 把 Panda 置于与 calib_bullet.py 完全相同的
  一组关节角(运动学置位, 零重力), 读出 panda_hand 的世界位姿, 写 CSV。

用法(必须用 Isaac Sim 自带的 python 跑):
  ./python.sh calib_isaac.py --usd /path/to/scene.usd --robot-prim /World/panda
  ./python.sh calib_isaac.py --usd scene.usd --robot-prim /World/panda \
      --set "0,-0.785,0,-2.356,0,1.571,0.785"        # 单个自定义构型

之后回到 Bullet 侧对比:
  python3 calib_bullet.py --compare calib_isaac_fk.csv

注意:
  - 输出四元数统一转成 xyzw(Isaac 内部是 wxyz), 与 Bullet CSV 同格式。
  - 若你的机器人基座不在世界原点, 位姿会以基座 frame 重新表达后输出,
    这样与 Bullet(基座在原点)可直接比较。
"""

import argparse
import csv
import sys

# ---------- SimulationApp 必须在其它 isaac import 之前创建 ----------
try:
    from isaacsim import SimulationApp          # Isaac Sim 4.x
except ImportError:
    from omni.isaac.kit import SimulationApp    # Isaac Sim 2022/2023

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True, help="包含机器人(和树)的 USD 场景")
ap.add_argument("--robot-prim", required=True, help="机器人 articulation 根 prim, 如 /World/panda")
ap.add_argument("--ee-prim", default=None,
                help="末端 prim 路径; 默认 <robot-prim>/panda_hand")
ap.add_argument("--out", default="calib_isaac_fk.csv")
ap.add_argument("--set", dest="single", default=None, help='单个构型 "q1,...,q7"')
ap.add_argument("--headless", action="store_true")
args = ap.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np

# isaac core 的 import 路径随版本变化, 逐个尝试
try:                                            # 4.5+
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.prims import SingleXFormPrim as XFormPrim
    from isaacsim.core.utils.stage import open_stage
except ImportError:                             # 2022.2 – 4.2
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.utils.stage import open_stage
try:
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    from omni.isaac.core.utils.types import ArticulationAction

EE_LINK_NAME = "panda_hand"   # 与 C++ 规划器的 TCP frame 一致 (panda_pinocchio.h)
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

# 必须与 calib_bullet.py 中的 TEST_CONFIGS 保持逐字一致!
TEST_CONFIGS = {
    "ready": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    "c2":    [0.5, -0.3, 0.2, -1.5, 0.3, 2.0, 0.0],
    "c3":    [-1.0, 0.5, -0.5, -2.0, 1.0, 3.0, 1.5],
    "c4":    [2.0, -1.5, 2.0, -0.5, -2.0, 0.5, -2.0],
    "c5":    [0.0, 1.0, 1.0, -2.8, -1.0, 1.0, 2.5],
}


def quat_wxyz_to_xyzw(q):
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def pose_in_base(base_pos, base_quat_xyzw, pos, quat_xyzw):
    """把世界位姿转换到机器人基座 frame(与 Bullet 基座在原点可比)"""
    def to_rot(q):
        x, y, z, w = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ])
    Rb = to_rot(base_quat_xyzw)
    Re = to_rot(quat_xyzw)
    p_rel = Rb.T @ (np.asarray(pos) - np.asarray(base_pos))
    R_rel = Rb.T @ Re
    # 旋转矩阵 → xyzw 四元数
    tr = np.trace(R_rel)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R_rel[2, 1] - R_rel[1, 2]) / s
        y = (R_rel[0, 2] - R_rel[2, 0]) / s
        z = (R_rel[1, 0] - R_rel[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R_rel)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R_rel[i, i] - R_rel[j, j] - R_rel[k, k])
        q = [0.0] * 4
        q[i] = 0.25 * s
        q[3] = (R_rel[k, j] - R_rel[j, k]) / s
        q[j] = (R_rel[j, i] + R_rel[i, j]) / s
        q[k] = (R_rel[k, i] + R_rel[i, k]) / s
        x, y, z, w = q[0], q[1], q[2], q[3]
    return p_rel.tolist(), [float(x), float(y), float(z), float(w)]


def main():
    open_stage(args.usd)
    world = World(stage_units_in_meters=1.0)
    # 标定阶段关掉重力, 保证纯运动学置位不下坠
    world.get_physics_context().set_gravity(0.0)

    robot = Articulation(args.robot_prim)
    world.scene.add(robot)
    world.reset()

    dof_names = list(robot.dof_names)
    print(f"[Info] articulation DOF 顺序: {dof_names}")
    try:
        arm_idx = [dof_names.index(n) for n in ARM_JOINT_NAMES]
    except ValueError as e:
        sys.exit(f"[Error] 找不到关节: {e}; 检查 --robot-prim 与 URDF 导入设置")

    ee_path = args.ee_prim or f"{args.robot_prim}/{EE_LINK_NAME}"
    ee = XFormPrim(ee_path)
    base = XFormPrim(args.robot_prim)

    def fk(q):
        full = robot.get_joint_positions()
        for i, qi in zip(arm_idx, q):
            full[i] = qi
        robot.set_joint_positions(full)
        # 同时把 PD 目标设成同一位置, 防止控制器往回拉
        robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=full))
        for _ in range(5):
            world.step(render=not args.headless)
        pos, quat = ee.get_world_pose()          # quat 为 wxyz
        bpos, bquat = base.get_world_pose()
        return pose_in_base(bpos, quat_wxyz_to_xyzw(bquat),
                            pos, quat_wxyz_to_xyzw(quat))

    if args.single:
        q = [float(v) for v in args.single.split(",")]
        assert len(q) == 7, "需要 7 个关节值"
        pos, orn = fk(q)
        print(f"q   = {q}")
        print(f"pos = {pos}  (m, 基座 frame)")
        print(f"orn = {orn}  (xyzw)")
    else:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name"] + [f"q{i}" for i in range(1, 8)]
                       + ["x", "y", "z", "qx", "qy", "qz", "qw"])
            for name, q in TEST_CONFIGS.items():
                pos, orn = fk(q)
                w.writerow([name] + q + pos + orn)
                print(f"[{name}] pos={np.round(pos,4).tolist()} "
                      f"orn={np.round(orn,4).tolist()}")
        print(f"\n已写入 {args.out}; 拷回 Bullet 侧后运行:\n"
              f"  python3 calib_bullet.py --compare {args.out}")

    simulation_app.close()


if __name__ == "__main__":
    main()
