#!/usr/bin/env python3
"""
calib_bullet.py — PyBullet 侧 FK 标定脚本

用途:
  把 Panda 机器人置于一组指定关节角(纯运动学, 不走物理),
  读出 panda_hand 的世界位姿, 写入 CSV。
  与 calib_isaac.py 输出的 CSV 对比, 验证两个仿真器的
  关节定义 / 基座位姿 / 末端 frame 完全一致。

用法:
  python3 calib_bullet.py [--urdf panda.urdf] [--out calib_bullet_fk.csv]
  python3 calib_bullet.py --set "0,-0.785,0,-2.356,0,1.571,0.785"   # 单个自定义构型
  python3 calib_bullet.py --compare calib_isaac_fk.csv              # 与 Isaac 结果对比

判定标准: 位置误差 < 1 mm, 姿态误差 < 0.1 deg。
CSV 格式: name,q1..q7,x,y,z,qx,qy,qz,qw   (位置米, 四元数 xyzw)
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import pybullet as p

# 与 C++ 规划器一致: TCP/GCP 使用 panda_hand frame (见 panda_pinocchio.h)
EE_LINK_NAME = "panda_hand"
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

# 标定测试构型 —— 必须与 calib_isaac.py 中的 TEST_CONFIGS 保持逐字一致!
TEST_CONFIGS = {
    "ready": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    "c2":    [0.5, -0.3, 0.2, -1.5, 0.3, 2.0, 0.0],
    "c3":    [-1.0, 0.5, -0.5, -2.0, 1.0, 3.0, 1.5],
    "c4":    [2.0, -1.5, 2.0, -0.5, -2.0, 0.5, -2.0],
    "c5":    [0.0, 1.0, 1.0, -2.8, -1.0, 1.0, 2.5],
}


def find_urdf(cli_path):
    """URDF 查找顺序与 C++ 一致: 命令行 → PANDA_URDF → pybullet_data"""
    if cli_path:
        return cli_path
    env = os.environ.get("PANDA_URDF")
    if env and os.path.exists(env):
        return env
    try:
        import pybullet_data
        cand = os.path.join(pybullet_data.getDataPath(), "franka_panda", "panda.urdf")
        if os.path.exists(cand):
            return cand
    except ImportError:
        pass
    sys.exit("[Error] 找不到 panda.urdf, 请用 --urdf 或设置 PANDA_URDF")


def load_robot(urdf):
    p.connect(p.DIRECT)
    robot = p.loadURDF(urdf, basePosition=[0, 0, 0], useFixedBase=True)
    joint_idx = {}   # 关节名 → pybullet joint index
    ee_link = -1
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        jname = info[1].decode()
        lname = info[12].decode()
        joint_idx[jname] = j
        if lname == EE_LINK_NAME:
            ee_link = j
    missing = [n for n in ARM_JOINT_NAMES if n not in joint_idx]
    if missing or ee_link < 0:
        sys.exit(f"[Error] URDF 中缺少关节 {missing} 或 link '{EE_LINK_NAME}'")
    return robot, [joint_idx[n] for n in ARM_JOINT_NAMES], ee_link


def fk(robot, arm_indices, ee_link, q):
    """纯运动学置位 + 读 panda_hand 的 URDF link frame 世界位姿"""
    for idx, qi in zip(arm_indices, q):
        p.resetJointState(robot, idx, qi)
    ls = p.getLinkState(robot, ee_link, computeForwardKinematics=1)
    pos = ls[4]    # worldLinkFramePosition (URDF frame, 不是质心 frame!)
    orn = ls[5]    # worldLinkFrameOrientation, xyzw
    return list(pos), list(orn)


def quat_angle_deg(qa, qb):
    """两个 xyzw 四元数之间的夹角(度), 数值稳定版 (见 calib_compare.py)"""
    s = 1.0 if sum(a * b for a, b in zip(qa, qb)) >= 0 else -1.0
    d = math.sqrt(sum((a - s * b) ** 2 for a, b in zip(qa, qb)))
    return math.degrees(4.0 * math.asin(min(1.0, d / 2.0)))


def compare(csv_a, csv_b):
    def load(path):
        rows = {}
        with open(path) as f:
            for r in csv.reader(f):
                if not r or r[0] == "name":
                    continue
                rows[r[0]] = [float(v) for v in r[1:]]
        return rows

    a, b = load(csv_a), load(csv_b)
    ok = True
    print(f"{'config':<10} {'pos_err(mm)':>12} {'rot_err(deg)':>13}  verdict")
    for name in a:
        if name not in b:
            print(f"{name:<10}  —— 在 {csv_b} 中缺失")
            ok = False
            continue
        pa, pb = np.array(a[name][7:10]), np.array(b[name][7:10])
        qa, qb = a[name][10:14], b[name][10:14]
        pe = np.linalg.norm(pa - pb) * 1000.0
        re = quat_angle_deg(qa, qb)
        passed = pe < 1.0 and re < 0.1
        ok = ok and passed
        print(f"{name:<10} {pe:>12.4f} {re:>13.5f}  {'PASS' if passed else 'FAIL'}")
    print("\n==> 标定" + ("通过: 两个仿真器 FK 一致, 可以开始回放实验"
                          if ok else "失败: 检查基座位姿 / EE frame / 关节顺序 / URDF 版本"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--out", default="calib_bullet_fk.csv")
    ap.add_argument("--set", dest="single", default=None,
                    help='单个构型 "q1,...,q7", 只打印不写文件')
    ap.add_argument("--compare", default=None,
                    help="与另一份 FK CSV(如 calib_isaac_fk.csv)对比")
    args = ap.parse_args()

    if args.compare:
        sys.exit(0 if compare(args.out, args.compare) else 1)

    robot, arm_indices, ee_link = load_robot(find_urdf(args.urdf))

    if args.single:
        q = [float(v) for v in args.single.split(",")]
        assert len(q) == 7, "需要 7 个关节值"
        pos, orn = fk(robot, arm_indices, ee_link, q)
        print(f"q   = {q}")
        print(f"pos = {pos}  (m)")
        print(f"orn = {orn}  (xyzw)")
        return

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name"] + [f"q{i}" for i in range(1, 8)]
                   + ["x", "y", "z", "qx", "qy", "qz", "qw"])
        for name, q in TEST_CONFIGS.items():
            pos, orn = fk(robot, arm_indices, ee_link, q)
            w.writerow([name] + q + pos + orn)
            print(f"[{name}] pos={np.round(pos,4).tolist()} orn={np.round(orn,4).tolist()}")
    print(f"\n已写入 {args.out}; 在 Isaac 侧跑 calib_isaac.py 后, 用:\n"
          f"  python3 calib_bullet.py --out {args.out} --compare calib_isaac_fk.csv")


if __name__ == "__main__":
    main()
