"""Interactive pybullet view of the --sparse-long 10-branch scene.

Mirrors the geometry in benchmark_grasp_rrt.cpp (sparse_long_scene block).
Usage:
    python3 visualize_sparse_long.py                 # static scene
    python3 visualize_sparse_long.py --standard      # original 10-branch scene instead
    python3 visualize_sparse_long.py --path X.csv    # loop a trajectory replay
"""
import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import os
import argparse

TRUNK_C, TRUNK_R, TRUNK_H = (0.7, 0.0), 0.08, 1.5
BR_R = 0.03
APPLE, APPLE_R = (0.45, 0.0, 0.80), 0.04
OVERHEAD, OVERHEAD_R = (0.45, 0.0, 0.91), 0.08
HOME = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]

STANDARD = [  # (h, angle, length)
    (0.65, 3.1415, 0.35), (0.65, 2.60, 0.30), (0.65, 3.68, 0.30),
    (0.80, 2.70, 0.35),   (0.80, 3.58, 0.35),
    (0.80, 2.10, 0.30),   (0.80, 4.18, 0.30),
    (0.95, 3.1415, 0.40), (0.95, 2.60, 0.30), (0.95, 3.68, 0.30)]
SPARSE_LONG = [
    (0.60, 3.1415, 0.60), (0.60, 2.40, 0.55), (0.60, 3.88, 0.55),
    (0.80, 2.55, 0.60),   (0.80, 3.73, 0.60),
    (0.80, 1.90, 0.55),   (0.80, 4.38, 0.55),
    (1.00, 3.1415, 0.60), (1.00, 2.40, 0.55), (1.00, 3.88, 0.55)]


def cylinder_pose(p0, p1):
    a = np.array(p0, dtype=float)
    b = np.array(p1, dtype=float)
    v = b - a
    length = float(np.linalg.norm(v))
    mid = 0.5 * (a + b)
    if length < 1e-9:
        return mid.tolist(), [0, 0, 0, 1], length
    axis = v / length
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, axis)
    sin_t = float(np.linalg.norm(cross))
    cos_t = float(np.dot(z, axis))
    if sin_t < 1e-9:
        return mid.tolist(), ([0, 0, 0, 1] if cos_t > 0 else [1, 0, 0, 0]), length
    cross /= sin_t
    half = 0.5 * math.atan2(sin_t, cos_t)
    sh = math.sin(half)
    return mid.tolist(), [cross[0]*sh, cross[1]*sh, cross[2]*sh, math.cos(half)], length


def add_cylinder(p0, p1, radius, rgba):
    pos, orn, length = cylinder_pose(p0, p1)
    if length < 1e-6:
        return
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=rgba)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                      basePosition=pos, baseOrientation=orn)


def add_sphere(pos, radius, rgba):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard", action="store_true", help="original scene instead of sparse-long")
    parser.add_argument("--path", type=str, default=None, help="CSV trajectory to loop")
    args, _ = parser.parse_known_args()

    branches = STANDARD if args.standard else SPARSE_LONG

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.loadURDF("plane.urdf")

    # trunk (rigid, brown)
    add_cylinder((TRUNK_C[0], TRUNK_C[1], 0.0), (TRUNK_C[0], TRUNK_C[1], TRUNK_H),
                 TRUNK_R, [0.45, 0.25, 0.08, 1.0])
    # branches (soft, green)
    for h, ang, ln in branches:
        ca, sa = math.cos(ang), math.sin(ang)
        p0 = (TRUNK_C[0] + TRUNK_R * ca, TRUNK_C[1] + TRUNK_R * sa, h)
        p1 = (p0[0] + ca * ln, p0[1] + sa * ln, h)
        add_cylinder(p0, p1, BR_R, [0.15, 0.60, 0.15, 1.0])
    # apple (red), overhead obstacle (gray, rigid)
    add_sphere(list(APPLE), APPLE_R, [0.85, 0.10, 0.10, 1.0])
    add_sphere(list(OVERHEAD), OVERHEAD_R, [0.40, 0.40, 0.40, 1.0])

    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.4], useFixedBase=True)
    for i, q in enumerate(HOME):
        p.resetJointState(robot_id, i, q)

    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=-50.0,
                                 cameraPitch=-18.0, cameraTargetPosition=[0.5, 0.0, 0.85])

    waypoints = []
    if args.path and os.path.exists(args.path):
        with open(args.path) as f:
            for line in f:
                if line.strip():
                    waypoints.append([float(x) for x in line.strip().split(",")])
        print(f"[sparse-long viz] loaded {len(waypoints)} waypoints")

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
            print("[sparse-long viz] static scene — close window to exit")
            while p.isConnected():
                time.sleep(1.0 / 60.0)
    except p.error:
        pass


if __name__ == "__main__":
    main()
