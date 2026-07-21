"""Simplified tree6: 按弯折角合并枝条小段 + 统一枝条半径,pybullet GUI 可视化。

需求:
  1. 每根枝条 5 段 → 若相邻段夹角小于阈值(默认 20°)则合并为长直段;
  2. 枝条半径统一放大到 10-branch 场景的 0.03 m;
  3. 树干不动(硬障碍,弯折明显,保持原样)。

Usage:
    python3 visualize_tree6_simple.py                  # 合并(20°) + r=0.03
    python3 visualize_tree6_simple.py --angle 30       # 更激进的合并
    python3 visualize_tree6_simple.py --radius 0.02    # 自定义统一半径
    python3 visualize_tree6_simple.py --report-only    # 只打印合并报告,不开 GUI
"""
import numpy as np
import math
import argparse
import time

from visualize_tree6 import TRUNK_NODES, TRUNK_RAD, BRANCH_NODES, BRANCH_RAD

BR_R_10BRANCH = 0.03   # 10-branch (sparse-long/standard) 场景的枝条半径
APPLE_R = 0.04
HOME = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]


def simplify_polyline(nodes, angle_deg):
    """按转折角合并:仅当在节点处的方向变化 >= 阈值时保留该节点。"""
    pts = [np.asarray(q, float) for q in nodes]
    keep = [pts[0]]
    for i in range(1, len(pts) - 1):
        d0 = pts[i] - keep[-1]          # 当前直段方向(相对上一个保留点)
        d1 = pts[i + 1] - pts[i]
        n0, n1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if n0 < 1e-9 or n1 < 1e-9:
            continue
        cosang = float(np.clip(np.dot(d0 / n0, d1 / n1), -1.0, 1.0))
        if math.degrees(math.acos(cosang)) >= angle_deg:
            keep.append(pts[i])
    keep.append(pts[-1])
    return keep


def build_simplified(angle_deg):
    simplified = []
    for b in range(len(BRANCH_NODES)):
        simplified.append(simplify_polyline(BRANCH_NODES[b], angle_deg))
    return simplified


def report(simplified):
    total_before = sum(len(b) - 1 for b in BRANCH_NODES)
    total_after = sum(len(b) - 1 for b in simplified)
    rads = np.array(BRANCH_RAD, float)
    print(f"branches: {len(BRANCH_NODES)}")
    print(f"segments: {total_before} -> {total_after}")
    per = {}
    for b, s in enumerate(simplified):
        per.setdefault(len(s) - 1, []).append(b)
    for k in sorted(per):
        print(f"  {k} seg(s): {len(per[k])} branches  {per[k]}")
    print(f"original branch radius: min {rads.min()*1000:.1f} mm, "
          f"max {rads.max()*1000:.1f} mm, mean {rads.mean()*1000:.1f} mm (taper root->tip)")
    print(f"new uniform radius:     {BR_R_10BRANCH*1000:.0f} mm (= 10-branch scene)")


def apple_position(radius_override, frac=0.95):
    """与 benchmark 相同逻辑:第 5 枝(index 4)弧长 frac 处,枝面下方。"""
    nodes = [np.asarray(q, float) for q in BRANCH_NODES[4]]
    seg = [np.linalg.norm(nodes[i + 1] - nodes[i]) for i in range(5)]
    total = sum(seg)
    target = frac * total
    acc = 0.0
    attach = nodes[5]
    for i in range(5):
        if acc + seg[i] >= target:
            t = (target - acc) / seg[i]
            attach = (1 - t) * nodes[i] + t * nodes[i + 1]
            break
        acc += seg[i]
    return [float(attach[0]), float(attach[1]),
            float(attach[2] - radius_override - APPLE_R)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angle", type=float, default=30.0,
                    help="merge threshold: keep a node only if bend >= this many degrees")
    ap.add_argument("--radius", type=float, default=BR_R_10BRANCH)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--path", type=str, default=None)
    ap.add_argument("--start", type=str, default=None,
                    help="7 comma-separated joint values: pose the robot there")
    ap.add_argument("--apple-frac", type=float, default=0.95,
                    help="apple position as fraction along branch 5 arclength (0.25 = near root)")
    args, _ = ap.parse_known_args()

    simplified = build_simplified(args.angle)
    print(f"== simplify report (angle threshold {args.angle:.0f} deg) ==")
    report(simplified)
    if args.report_only:
        return

    import pybullet as p
    import pybullet_data
    from visualize_tree6 import main as _unused  # noqa: F401 (确保同目录)

    def cylinder_pose(p0, p1):
        a, b = np.asarray(p0, float), np.asarray(p1, float)
        v = b - a
        L = float(np.linalg.norm(v))
        mid = 0.5 * (a + b)
        if L < 1e-9:
            return mid.tolist(), [0, 0, 0, 1], L
        ax = v / L
        z = np.array([0.0, 0.0, 1.0])
        c = np.cross(z, ax)
        s = float(np.linalg.norm(c))
        d = float(np.dot(z, ax))
        if s < 1e-9:
            return mid.tolist(), ([0, 0, 0, 1] if d > 0 else [1, 0, 0, 0]), L
        c /= s
        h = 0.5 * math.atan2(s, d)
        sh = math.sin(h)
        return mid.tolist(), [c[0]*sh, c[1]*sh, c[2]*sh, math.cos(h)], L

    def add_cyl(p0, p1, r, rgba):
        pos, orn, L = cylinder_pose(p0, p1)
        if L < 1e-6:
            return
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=L, rgbaColor=rgba)
        p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=pos, baseOrientation=orn)

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.loadURDF("plane.urdf")

    # 树干:保持原样(硬障碍,棕色)
    for i in range(len(TRUNK_NODES) - 1):
        r = 0.5 * (TRUNK_RAD[i] + TRUNK_RAD[i + 1])
        add_cyl(TRUNK_NODES[i], TRUNK_NODES[i + 1], r, [0.45, 0.25, 0.08, 1.0])

    # 简化后的枝条:统一半径(软障碍,绿色;苹果枝黄绿)
    for b, poly in enumerate(simplified):
        col = [0.55, 0.75, 0.10, 1.0] if b == 4 else [0.15, 0.60, 0.15, 1.0]
        for i in range(len(poly) - 1):
            add_cyl(poly[i], poly[i + 1], args.radius, col)

    apple = apple_position(args.radius, args.apple_frac)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=APPLE_R, rgbaColor=[0.85, 0.10, 0.10, 1.0])
    p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=apple)
    print(f"[tree6-simple viz] apple at {np.round(apple,3)}")

    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.4], useFixedBase=True)
    for i, q in enumerate(HOME):
        p.resetJointState(robot_id, i, q)
    if args.start:
        vals = [float(x) for x in args.start.split(",")]
        for i in range(7):
            p.resetJointState(robot_id, i, vals[i])
        print("[tree6-simple viz] robot posed at --start config")
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=-55.0,
                                 cameraPitch=-12.0, cameraTargetPosition=[0.6, 0.0, 1.0])

    # ---- 轨迹回放 + EE 蓝→红描线(注意:历史路径是在"细枝原版"
    # 场景里规划的,放到加粗场景里只作直观参考,穿枝属正常现象) ----
    waypoints = []
    if args.path and __import__("os").path.exists(args.path):
        with open(args.path) as f:
            for line in f:
                if line.strip():
                    waypoints.append([float(x) for x in line.strip().split(",")])
        print(f"[tree6-simple viz] loaded {len(waypoints)} waypoints")
        hand_link = None
        for j in range(p.getNumJoints(robot_id)):
            if p.getJointInfo(robot_id, j)[12].decode() == "panda_hand":
                hand_link = j
                break
        if hand_link is not None:
            ee = []
            for w in waypoints:
                for i in range(7):
                    p.resetJointState(robot_id, i, w[i])
                ee.append(p.getLinkState(robot_id, hand_link)[0])
            n = len(ee)
            for i in range(n - 1):
                t = i / max(1, n - 2)
                p.addUserDebugLine(ee[i], ee[i + 1],
                                   lineColorRGB=[t, 0.1, 1.0 - t], lineWidth=3)
            for i in range(7):
                p.resetJointState(robot_id, i, waypoints[0][i])

    print("[tree6-simple viz] close window to exit")
    try:
        if waypoints:
            while p.isConnected():
                for w in range(len(waypoints) - 1):
                    a = np.array(waypoints[w][:7])
                    b = np.array(waypoints[w + 1][:7])
                    for k in range(6):
                        q = (1 - k / 6.0) * a + (k / 6.0) * b
                        for i in range(7):
                            p.resetJointState(robot_id, i, q[i])
                        time.sleep(0.01)
                time.sleep(1.5)
        else:
            while p.isConnected():
                time.sleep(1.0 / 60.0)
    except Exception:
        pass


if __name__ == "__main__":
    main()
