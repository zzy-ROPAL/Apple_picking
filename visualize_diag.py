import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import os
import argparse


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
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=rgba)
    p.createMultiBody(baseMass=0,
                      baseCollisionShapeIndex=col,
                      baseVisualShapeIndex=vis,
                      basePosition=pos,
                      baseOrientation=orn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="CSV trajectory file")
    parser.add_argument("--xgate", action="store_true",
                        help="show the X-gate multi-branch scene (2 crossed soft branches)")
    args, _ = parser.parse_known_args()

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    p.loadURDF("plane.urdf")

    # --- Diagnostic scene geometry (v6: apple moved to y=-0.20 to force y-motion) ---
    # Branch A: along x̂ above apple, y=-0.20 (follows the apple), RIGID (BROWN)
    A_p0 = (0.40, -0.20, 1.05)
    A_p1 = (0.55, -0.20, 1.05)
    add_cylinder(A_p0, A_p1, 0.02, rgba=[0.55, 0.27, 0.07, 1.0])

    if not args.xgate:
        # Branch B: along ŷ, extended to shield the apple from the robot's y-range,
        # y from -0.30 to +0.30 (60 cm), r=0.04, SOFT (BLUE)
        B_p0 = (0.30, -0.30, 1.00)
        B_p1 = (0.30,  0.30, 1.00)
        add_cylinder(B_p0, B_p1, 0.04, rgba=[0.20, 0.35, 0.85, 0.75])
    else:
        # X-gate: two CROSSED soft branches with perpendicular axes, crossing
        # near (0.33, 0, 1.00) on the approach corridor.
        # B1 along ŷ (BLUE), x=0.33, y in [-0.25, 0.25]
        B1_p0 = (0.33, -0.25, 1.00)
        B1_p1 = (0.33,  0.25, 1.00)
        add_cylinder(B1_p0, B1_p1, 0.04, rgba=[0.20, 0.35, 0.85, 0.75])
        # B2 along x̂ (GREEN), y=0.00, x in [0.24, 0.46]
        B2_p0 = (0.24, 0.00, 1.00)
        B2_p1 = (0.46, 0.00, 1.00)
        add_cylinder(B2_p0, B2_p1, 0.04, rgba=[0.20, 0.75, 0.35, 0.75])

    # Apple (red) at (0.42, -0.20, 0.95)
    apple_pos = [0.42, -0.20, 0.95]
    apple_r = 0.04
    apple_col = p.createCollisionShape(p.GEOM_SPHERE, radius=apple_r)
    apple_vis = p.createVisualShape(p.GEOM_SPHERE, radius=apple_r, rgbaColor=[0.85, 0.10, 0.10, 1.0])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=apple_col,
                      baseVisualShapeIndex=apple_vis, basePosition=apple_pos)

    # Panda — displayed at Start 1 config so you can see where the EE actually is
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.4], useFixedBase=True)
    start1_q = [0.00412202, -0.944803, 0.620311, -2.04541, -0.48732, 1.33057, 1.50796]
    for i, q in enumerate(start1_q):
        p.resetJointState(robot_id, i, q)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=-60.0,
        cameraPitch=-15.0,
        cameraTargetPosition=[0.3, 0.0, 1.0],
    )

    # ---- Optional trajectory playback ----
    waypoints = []
    if args.path:
        candidates = [args.path, os.path.join("build", args.path), os.path.join("..", args.path)]
        path_file = next((c for c in candidates if os.path.exists(c)), None)
        if path_file is None:
            print(f"[diag viz] Could not find trajectory file: {args.path}")
        else:
            print(f"[diag viz] Loading trajectory from {path_file}")
            with open(path_file, "r") as f:
                for line in f:
                    if line.strip():
                        waypoints.append([float(x) for x in line.strip().split(",")])
            print(f"[diag viz] Loaded {len(waypoints)} waypoints.")

    revolute_joints = [0, 1, 2, 3, 4, 5, 6]

    if waypoints:
        try:
            while p.isConnected():
                for i, q in enumerate(waypoints[0][:7]):
                    p.resetJointState(robot_id, revolute_joints[i], q)
                time.sleep(0.8)
                for w in range(len(waypoints) - 1):
                    a = np.array(waypoints[w][:7])
                    b = np.array(waypoints[w + 1][:7])
                    steps = 6
                    for k in range(steps):
                        alpha = k / float(steps)
                        q = (1 - alpha) * a + alpha * b
                        for i, joint_idx in enumerate(revolute_joints):
                            p.resetJointState(robot_id, joint_idx, q[i])
                        p.stepSimulation()
                        time.sleep(0.01)
                time.sleep(1.5)
        except p.error:
            pass
    else:
        print("[diag viz] No trajectory provided — static scene only. Close window to exit.")
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1.0 / 120.0)
        except p.error:
            pass


if __name__ == "__main__":
    main()
