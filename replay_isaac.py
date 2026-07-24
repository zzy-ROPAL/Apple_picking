#!/usr/bin/env python3
"""
replay_isaac.py — 在 Isaac Sim 中回放 GraspRRT 路径 (grasp_path.csv)

功能:
  1. 时间参数化: 对 RRT 路径点做线性插值, 按 Panda 关节速度限幅分配每段
     时长, 再按物理步长采样成逐步的位置目标 (几何路径不变, 只加时间)。
  2. 回放执行: 每个物理步给 articulation controller 发一个位置目标。
  3. 接触分类: 树干(rigid trunk)接触 = 失败; 绿色可穿透枝条接触 = 允许,
     只记录。用 prim 路径关键字区分, 见 --trunk-pattern / --branch-pattern。
  4. 关节跟踪误差: 每步记录 q_ref / q_actual / 误差范数, 写 tracking_log.csv。

用法(Isaac Sim 自带 python):
  ./python.sh replay_isaac.py --usd scene.usd --robot-prim /World/panda \
      --path grasp_path.csv --trunk-pattern trunk --branch-pattern branch

前提: calib_bullet.py / calib_isaac.py 标定已通过, 否则结果不可解释。

输出(默认写到 ./replay_out/):
  tracking_log.csv  — t, q_ref1..7, q_act1..7, err_norm
  contacts.csv      — t, robot_prim, tree_prim, class(trunk/branch/other), impulse
  终端打印 SUCCESS / FAIL 判定:
    FAIL 条件 = 出现 trunk 接触, 或最终关节误差 > --final-err-thresh (默认 0.05 rad)
"""

import argparse
import csv
import os
import sys

try:
    from isaacsim import SimulationApp          # Isaac Sim 4.x
except ImportError:
    from omni.isaac.kit import SimulationApp    # Isaac Sim 2022/2023

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True, help="包含机器人和树的 USD 场景")
ap.add_argument("--robot-prim", required=True, help="机器人 articulation 根 prim")
ap.add_argument("--path", default="grasp_path.csv", help="RRT 路径 CSV (每行 7 个关节值)")
ap.add_argument("--dt", type=float, default=1.0 / 60.0, help="物理步长 (s)")
ap.add_argument("--vel-scale", type=float, default=0.3,
                help="关节速度限幅的缩放系数 (0~1], 越小走得越慢越稳")
ap.add_argument("--trunk-pattern", default="trunk",
                help="树干 prim 路径关键字, 逗号分隔; 命中即判 FAIL")
ap.add_argument("--branch-pattern", default="branch",
                help="可穿透绿枝 prim 路径关键字, 逗号分隔; 命中只记录不判失败")
ap.add_argument("--hold-time", type=float, default=1.0,
                help="到达终点后继续保持的时间 (s), 用于观察稳态误差")
ap.add_argument("--final-err-thresh", type=float, default=0.05,
                help="最终关节误差范数阈值 (rad), 超过判 FAIL")
ap.add_argument("--out-dir", default="replay_out")
ap.add_argument("--headless", action="store_true")
args = ap.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np

try:                                            # 4.5+
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:                             # 2022.2 – 4.2
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.stage import open_stage
    from omni.isaac.core.utils.types import ArticulationAction

import omni.usd
from omni.physx import get_physx_simulation_interface
from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics

ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
# Franka 官方关节速度上限 (rad/s)
PANDA_VMAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])


# ---------------------------------------------------------------- 时间参数化
def load_waypoints(path):
    wps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split(",")]
            if len(vals) != 7:
                sys.exit(f"[Error] {path} 中某行不是 7 个值: {line}")
            wps.append(vals)
    if len(wps) < 2:
        sys.exit(f"[Error] {path} 少于 2 个路径点")
    return np.array(wps)


def time_parameterize(wps, dt, vel_scale):
    """线性插值 + 速度限幅定时: 返回按 dt 采样的参考轨迹 (M x 7)。
    几何路径不变 —— 机器人仍精确经过每个 RRT 路径点。"""
    vmax = PANDA_VMAX * vel_scale
    refs = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        seg_t = float(np.max(np.abs(b - a) / vmax))     # 该段时长由最慢关节决定
        n = max(1, int(np.ceil(seg_t / dt)))
        for k in range(1, n + 1):
            refs.append(a + (b - a) * (k / n))
    return np.array(refs)


# ---------------------------------------------------------------- 接触监听
class ContactMonitor:
    """订阅 PhysX contact report, 把机器人与树的接触按 prim 路径分类"""

    def __init__(self, robot_root, trunk_keys, branch_keys):
        self.robot_root = robot_root
        self.trunk_keys = [k.strip() for k in trunk_keys.split(",") if k.strip()]
        self.branch_keys = [k.strip() for k in branch_keys.split(",") if k.strip()]
        self.events = []          # (t, robot_prim, other_prim, cls, impulse)
        self.trunk_contact = False
        self.t = 0.0
        self._sub = get_physx_simulation_interface() \
            .subscribe_contact_report_events(self._on_contact)

    def enable_reports_on_robot(self, stage):
        """给机器人所有刚体 link 打开 contact report(阈值 0 = 全部上报)"""
        n = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(self.robot_root) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                api.CreateThresholdAttr().Set(0.0)
                n += 1
        print(f"[Info] 已在 {n} 个机器人 link 上启用 contact report")

    def _classify(self, path):
        low = path.lower()
        if any(k.lower() in low for k in self.trunk_keys):
            return "trunk"
        if any(k.lower() in low for k in self.branch_keys):
            return "branch"
        return "other"

    def _on_contact(self, contact_headers, contact_data):
        for h in contact_headers:
            a = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            a_is_robot = a.startswith(self.robot_root)
            b_is_robot = b.startswith(self.robot_root)
            if a_is_robot == b_is_robot:        # 机器人自碰撞或树内部接触, 跳过
                continue
            robot_prim, other = (a, b) if a_is_robot else (b, a)
            cls = self._classify(other)
            impulse = 0.0
            for i in range(h.contact_data_offset,
                           h.contact_data_offset + h.num_contact_data):
                d = contact_data[i]
                impulse += float(np.linalg.norm(
                    [d.impulse.x, d.impulse.y, d.impulse.z]))
            self.events.append((self.t, robot_prim, other, cls, impulse))
            if cls == "trunk":
                self.trunk_contact = True
                print(f"[FAIL-CONTACT] t={self.t:.3f}s  {robot_prim}  撞到树干  "
                      f"{other}  impulse={impulse:.4f}")


# ---------------------------------------------------------------- 主流程
def main():
    wps = load_waypoints(args.path)
    refs = time_parameterize(wps, args.dt, args.vel_scale)
    total_t = len(refs) * args.dt
    print(f"[Info] {len(wps)} 个路径点 → {len(refs)} 个参考目标, "
          f"执行时长 {total_t:.2f}s (vel_scale={args.vel_scale})")

    open_stage(args.usd)
    world = World(stage_units_in_meters=1.0, physics_dt=args.dt,
                  rendering_dt=args.dt)
    robot = Articulation(args.robot_prim)
    world.scene.add(robot)

    stage = omni.usd.get_context().get_stage()
    monitor = ContactMonitor(args.robot_prim, args.trunk_pattern,
                             args.branch_pattern)
    monitor.enable_reports_on_robot(stage)

    world.reset()
    dof_names = list(robot.dof_names)
    try:
        arm_idx = np.array([dof_names.index(n) for n in ARM_JOINT_NAMES])
    except ValueError as e:
        sys.exit(f"[Error] 找不到关节: {e}")
    ctrl = robot.get_articulation_controller()

    def action_from(q7):
        # 只驱动 7 个臂关节, 手指等其余 DOF 不动
        return ArticulationAction(joint_positions=np.asarray(q7),
                                  joint_indices=arm_idx)

    # 初始构型: 运动学置位到路径起点, 再稳定几步
    q0_full = robot.get_joint_positions()
    q0_full[arm_idx] = wps[0]
    robot.set_joint_positions(q0_full)
    ctrl.apply_action(action_from(wps[0]))
    for _ in range(30):
        world.step(render=not args.headless)

    # 回放 + 终点保持
    hold_steps = int(args.hold_time / args.dt)
    log = []
    t = 0.0
    for step in range(len(refs) + hold_steps):
        q_ref = refs[min(step, len(refs) - 1)]
        monitor.t = t
        ctrl.apply_action(action_from(q_ref))
        world.step(render=not args.headless)
        q_act = np.asarray(robot.get_joint_positions())[arm_idx]
        err = float(np.linalg.norm(q_act - q_ref))
        log.append([t] + q_ref.tolist() + q_act.tolist() + [err])
        t += args.dt

    # ------------------------------------------------------------ 写日志
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "tracking_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"q_ref{i}" for i in range(1, 8)]
                   + [f"q_act{i}" for i in range(1, 8)] + ["err_norm"])
        w.writerows(log)
    with open(os.path.join(args.out_dir, "contacts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "robot_prim", "tree_prim", "class", "impulse"])
        w.writerows(monitor.events)

    # ------------------------------------------------------------ 判定
    errs = np.array([r[-1] for r in log])
    final_err = errs[-1]
    n_trunk = sum(1 for e in monitor.events if e[3] == "trunk")
    n_branch = sum(1 for e in monitor.events if e[3] == "branch")
    n_other = sum(1 for e in monitor.events if e[3] == "other")

    print("\n================ 回放结果 ================")
    print(f"  跟踪误差: max={errs.max():.4f} rad, mean={errs.mean():.4f} rad, "
          f"final={final_err:.4f} rad")
    print(f"  接触事件: trunk={n_trunk}  branch(允许)={n_branch}  other={n_other}")
    fail_reasons = []
    if monitor.trunk_contact:
        fail_reasons.append("撞到树干")
    if final_err > args.final_err_thresh:
        fail_reasons.append(f"最终关节误差 {final_err:.4f} > {args.final_err_thresh}")
    if fail_reasons:
        print(f"  ==> FAIL: {'; '.join(fail_reasons)}")
    else:
        print("  ==> SUCCESS: 无树干接触, 终点误差在阈值内"
              + (f" (推开了绿枝 {n_branch} 次接触)" if n_branch else ""))
    print(f"  日志已写入 {args.out_dir}/")

    simulation_app.close()
    sys.exit(1 if fail_reasons else 0)


if __name__ == "__main__":
    main()
