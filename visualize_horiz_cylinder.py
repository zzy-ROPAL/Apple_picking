"""
Live PyBullet GUI playback for the horizontal-cylinder anisotropy test scene.

Usage:
    python3 visualize_horiz_cylinder.py
    python3 visualize_horiz_cylinder.py --path benchmark_results_with_A_horizfine_s9_K200/grasp_path_01.csv
    python3 visualize_horiz_cylinder.py --path <CSV> --speed 0.5    # slower playback
    python3 visualize_horiz_cylinder.py --path <CSV> --steps 8      # more interpolation per segment
"""
import argparse, csv, math, os, time
import numpy as np
import pybullet as p
import pybullet_data

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="benchmark_results_with_A_horizfine_s1_K200/grasp_path_01.csv",
                    help="Trajectory CSV (one row = one waypoint, 7 joint values).")
    ap.add_argument("--steps", type=int, default=6,
                    help="Interpolation steps per waypoint segment.")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier (1.0 = ~real time).")
    ap.add_argument("--loop", action="store_true", help="Loop playback after finishing.")
    return ap.parse_args()

def build_scene():
    p.loadURDF("plane.urdf")

    # Trunk
    tcol = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.08, height=1.5)
    tvis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.08, length=1.5,
                                rgbaColor=[0.55, 0.27, 0.07, 1])
    p.createMultiBody(0, tcol, tvis, [0.7, 0.0, 0.75])

    # Horizontal test cylinder (axis y), wrapping apple.
    # NOTE: this cylinder is BY DESIGN independent from the trunk — it is a
    # controlled test obstacle for the anisotropy ablation, not a real branch.
    apple = [0.45, 0.0, 0.80]
    cyl_r, cyl_half = 0.12, 0.10
    ccol = p.createCollisionShape(p.GEOM_CYLINDER, radius=cyl_r, height=2*cyl_half)
    cvis = p.createVisualShape(p.GEOM_CYLINDER, radius=cyl_r, length=2*cyl_half,
                                rgbaColor=[0.2, 0.65, 0.2, 0.55])
    # Quaternion for a 90-degree rotation around +X axis: aligns cylinder axis with world +Y.
    # Using explicit quaternion form avoids any Euler-convention ambiguity.
    orn_xy90 = [math.sin(math.pi/4), 0.0, 0.0, math.cos(math.pi/4)]
    p.createMultiBody(0, ccol, cvis, apple, orn_xy90)

    # Axis-end markers so the cylinder's orientation is visually unambiguous:
    #   small red sphere at apple − ŷ · cyl_half (the −y end of the cylinder)
    #   small blue sphere at apple + ŷ · cyl_half (the +y end)
    marker_r = 0.018
    sphere_neg = p.createVisualShape(p.GEOM_SPHERE, radius=marker_r,
                                      rgbaColor=[1.0, 0.0, 0.0, 1.0])
    sphere_pos = p.createVisualShape(p.GEOM_SPHERE, radius=marker_r,
                                      rgbaColor=[0.0, 0.4, 1.0, 1.0])
    p.createMultiBody(0, -1, sphere_neg, [apple[0], apple[1] - cyl_half, apple[2]])
    p.createMultiBody(0, -1, sphere_pos, [apple[0], apple[1] + cyl_half, apple[2]])

    # Apple
    acol = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
    avis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04,
                                rgbaColor=[1.0, 0.05, 0.05, 1.0])
    p.createMultiBody(0, acol, avis, apple)

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
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # hide side panels

    build_scene()
    robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.4], useFixedBase=True)
    REV = [0, 1, 2, 3, 4, 5, 6]

    qs = load_waypoints(args.path)
    print(f"Loaded {len(qs)} waypoints from {args.path}")

    # Camera pointing from the +x side, slightly above. With yaw=0 we look from +x
    # straight at the apple, so the y-axis cylinder appears stretched left/right
    # across the view (red/blue axis markers visible on either side).
    p.resetDebugVisualizerCamera(cameraDistance=1.4, cameraYaw=0, cameraPitch=-15,
                                  cameraTargetPosition=[0.45, 0.0, 0.8])

    # Initial pose
    for j, qv in zip(REV, qs[0]):
        p.resetJointState(robot, j, qv)
    time.sleep(0.8)

    base_dt = 1.0 / 60.0
    dt = base_dt / max(args.speed, 1e-3)

    while True:
        for i in range(len(qs) - 1):
            for step in range(args.steps):
                a = step / float(args.steps)
                q = qs[i]*(1-a) + qs[i+1]*a
                for j, qv in zip(REV, q):
                    p.resetJointState(robot, j, qv)
                p.stepSimulation()
                time.sleep(dt)
        # Final waypoint
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
