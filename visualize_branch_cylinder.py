"""
Live PyBullet GUI playback for the trunk-attached anisotropy test scene.

Geometry mirrors --aniso-branch-scene in benchmark_grasp_rrt.cpp:
the green cylinder grows from the trunk surface at apple height, along
the trunk → apple direction, length 0.20 m, radius 0.12 m. The apple
sits inside the cylinder near the outer tip.

Usage:
    python3 visualize_branch_cylinder.py
    python3 visualize_branch_cylinder.py --path <CSV>
    python3 visualize_branch_cylinder.py --path <CSV> --speed 0.5 --loop
"""
import argparse, csv, math, os, time
import numpy as np
import pybullet as p
import pybullet_data

# Scene constants (must match benchmark_grasp_rrt.cpp)
APPLE        = np.array([0.45, 0.0, 0.80])
APPLE_RADIUS = 0.04
TRUNK_CENTER = np.array([0.70, 0.0, 0.75])
TRUNK_R, TRUNK_H = 0.08, 1.5
CYL_R, CYL_LENGTH = 0.12, 0.40    # length doubled from 0.20


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path",
                    default="benchmark_results_with_A_sampbias_branch/grasp_path_01.csv",
                    help="Trajectory CSV (one row = one waypoint, 7 joint values).")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    return ap.parse_args()


def build_scene():
    p.loadURDF("plane.urdf")

    # Trunk (brown vertical cylinder)
    tcol = p.createCollisionShape(p.GEOM_CYLINDER, radius=TRUNK_R, height=TRUNK_H)
    tvis = p.createVisualShape(p.GEOM_CYLINDER, radius=TRUNK_R, length=TRUNK_H,
                                rgbaColor=[0.55, 0.27, 0.07, 1])
    p.createMultiBody(0, tcol, tvis, TRUNK_CENTER.tolist())

    # Trunk-attached cylinder along the trunk → apple direction
    trunk_axis_proj = np.array([TRUNK_CENTER[0], TRUNK_CENTER[1], APPLE[2]])
    direction = APPLE - trunk_axis_proj
    if np.linalg.norm(direction) < 1e-6:
        direction = np.array([-1.0, 0.0, 0.0])
    direction = direction / np.linalg.norm(direction)
    p0 = trunk_axis_proj + TRUNK_R * direction
    p1 = p0 + CYL_LENGTH * direction
    center = 0.5 * (p0 + p1)

    # Rotate the cylinder so its native z-axis matches `direction`.
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-6:
        # antiparallel or parallel — handle separately
        orn = [0, 0, 0, 1] if dot > 0 else [1, 0, 0, 0]
    else:
        axis_n = cross / np.linalg.norm(cross)
        angle = math.acos(dot)
        s = math.sin(angle / 2.0)
        c = math.cos(angle / 2.0)
        orn = [axis_n[0] * s, axis_n[1] * s, axis_n[2] * s, c]

    ccol = p.createCollisionShape(p.GEOM_CYLINDER, radius=CYL_R, height=CYL_LENGTH)
    cvis = p.createVisualShape(p.GEOM_CYLINDER, radius=CYL_R, length=CYL_LENGTH,
                                rgbaColor=[0.20, 0.65, 0.20, 0.55])
    p.createMultiBody(0, ccol, cvis, center.tolist(), orn)

    # Axis-end markers: red at trunk end, blue at outer tip
    marker_r = 0.018
    s_red  = p.createVisualShape(p.GEOM_SPHERE, radius=marker_r,
                                  rgbaColor=[1.0, 0.0, 0.0, 1.0])
    s_blue = p.createVisualShape(p.GEOM_SPHERE, radius=marker_r,
                                  rgbaColor=[0.0, 0.4, 1.0, 1.0])
    p.createMultiBody(0, -1, s_red,  p0.tolist())
    p.createMultiBody(0, -1, s_blue, p1.tolist())

    # Apple (red)
    acol = p.createCollisionShape(p.GEOM_SPHERE, radius=APPLE_RADIUS)
    avis = p.createVisualShape(p.GEOM_SPHERE, radius=APPLE_RADIUS,
                                rgbaColor=[1.0, 0.05, 0.05, 1.0])
    p.createMultiBody(0, acol, avis, APPLE.tolist())


def load_waypoints(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"trajectory not found: {path}")
    qs = []
    with open(path) as f:
        for row in csv.reader(f):
            if row and len(row) >= 7:
                qs.append([float(x) for x in row[:7]])
    return np.asarray(qs)


def main():
    args = parse_args()
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    build_scene()
    robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.4], useFixedBase=True)
    REV = [0, 1, 2, 3, 4, 5, 6]

    qs = load_waypoints(args.path)
    print(f"Loaded {len(qs)} waypoints from {args.path}")

    p.resetDebugVisualizerCamera(cameraDistance=1.4, cameraYaw=35, cameraPitch=-18,
                                  cameraTargetPosition=[0.45, 0.0, 0.80])

    for j, qv in zip(REV, qs[0]):
        p.resetJointState(robot, j, qv)
    time.sleep(0.8)

    base_dt = 1.0 / 60.0
    dt = base_dt / max(args.speed, 1e-3)

    while True:
        for i in range(len(qs) - 1):
            for step in range(args.steps):
                a = step / float(args.steps)
                q = qs[i] * (1 - a) + qs[i + 1] * a
                for j, qv in zip(REV, q):
                    p.resetJointState(robot, j, qv)
                p.stepSimulation()
                time.sleep(dt)
        for j, qv in zip(REV, qs[-1]):
            p.resetJointState(robot, j, qv)
        time.sleep(1.5)
        if not args.loop:
            break

    print("Playback done. Close the window to exit.")
    while p.isConnected():
        time.sleep(0.05)


if __name__ == "__main__":
    main()
