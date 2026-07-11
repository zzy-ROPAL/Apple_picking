"""
Interactive PyBullet view of the V6-like tree6 scene:
  --tree6 --apple-frac 0.5 --branch-rscale 2, with the robot started on the
  -y side so soft branch 3 (BLUE) lies between the robot and the apple.

Usage:
  python3 visualize_v6_tree6.py                      # static scene + robot at V6 start
  python3 visualize_v6_tree6.py --path build/benchmark_results_with_A_XXX/grasp_path_01.csv
"""
import pybullet as p, pybullet_data, numpy as np, math, time, os, argparse
import visualize_tree6 as vt   # reuse TRUNK/BRANCH node + radius arrays

RSCALE = 4.0    # branches doubled twice (×4 of original tree6 radii)
APPLE_FRAC = 0.5
APPLE_R = 0.04
APPLE_BRANCH = 4
SHIELD = 3
V6_START = [1.058296, 0.508329, -1.578781, -2.553546, 0.210120, 3.104303, 1.329544]
BASE = [0, 0, 0.4]

BN = [np.array(b, float) for b in vt.BRANCH_NODES]
BR = vt.BRANCH_RAD
TN = [np.array(t, float) for t in vt.TRUNK_NODES]
TR = vt.TRUNK_RAD


def apple_pos():
    b5, r5 = BN[APPLE_BRANCH], BR[APPLE_BRANCH]
    sl = [np.linalg.norm(b5[i + 1] - b5[i]) for i in range(5)]
    tot = sum(sl); tgt = APPLE_FRAC * tot; acc = 0.0
    attach = b5[5]; lr = 0.5 * (r5[4] + r5[5])
    for i in range(5):
        if acc + sl[i] >= tgt:
            t = (tgt - acc) / sl[i]
            attach = (1 - t) * b5[i] + t * b5[i + 1]
            lr = 0.5 * (r5[i] + r5[i + 1]); break
        acc += sl[i]
    lr *= RSCALE
    return np.array([attach[0], attach[1], attach[2] - lr - APPLE_R])


def add_cyl(p0, p1, r, rgba):
    v = p1 - p0; L = float(np.linalg.norm(v))
    if L < 1e-6:
        return
    mid = 0.5 * (p0 + p1); z = np.array([0, 0, 1.]); ax = v / L
    c = np.cross(z, ax); s = float(np.linalg.norm(c)); d = float(np.dot(z, ax))
    if s < 1e-9:
        orn = [0, 0, 0, 1] if d > 0 else [1, 0, 0, 0]
    else:
        h = 0.5 * math.atan2(s, d); c /= s
        orn = [c[0] * math.sin(h), c[1] * math.sin(h), c[2] * math.sin(h), math.cos(h)]
    vs = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=L, rgbaColor=rgba)
    p.createMultiBody(0, baseVisualShapeIndex=vs, basePosition=mid.tolist(), baseOrientation=orn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default=None, help="trajectory CSV to play")
    args, _ = ap.parse_known_args()

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.loadURDF("plane.urdf")

    for i in range(8):
        add_cyl(TN[i], TN[i + 1], 0.5 * (TR[i] + TR[i + 1]), [0.4, 0.26, 0.13, 1])
    for b in range(len(BN)):
        for i in range(5):
            r = 0.5 * (BR[b][i] + BR[b][i + 1]) * RSCALE
            if b == APPLE_BRANCH:
                col = [0.55, 0.27, 0.07, 1]      # apple branch: brown
            elif b == SHIELD:
                col = [0.15, 0.4, 0.95, 1]        # shield branch 3: BLUE
            else:
                col = [0.2, 0.7, 0.25, 1]         # other soft branches: green
            add_cyl(BN[b][i], BN[b][i + 1], r, col)

    ap_pos = apple_pos()
    vs = p.createVisualShape(p.GEOM_SPHERE, radius=APPLE_R, rgbaColor=[0.9, 0.05, 0.05, 1])
    p.createMultiBody(0, baseVisualShapeIndex=vs, basePosition=ap_pos.tolist())

    rid = p.loadURDF("franka_panda/panda.urdf", BASE, useFixedBase=True)
    for i in range(7):
        p.resetJointState(rid, i, V6_START[i])

    p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=50, cameraPitch=-25,
                                 cameraTargetPosition=ap_pos.tolist())
    print(f"[v6 viz] apple at {ap_pos.round(3)}; blue = shield branch 3; robot at V6 start")

    waypoints = []
    if args.path:
        cands = [args.path, os.path.join("build", args.path), os.path.join("..", args.path)]
        f = next((c for c in cands if os.path.exists(c)), None)
        if f:
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        waypoints.append([float(x) for x in line.strip().split(",")])
            print(f"[v6 viz] loaded {len(waypoints)} waypoints from {f}")
        else:
            print(f"[v6 viz] path not found: {args.path}")

    try:
        if waypoints:
            while p.isConnected():
                for i in range(7):
                    p.resetJointState(rid, i, waypoints[0][i])
                time.sleep(0.8)
                for w in range(len(waypoints) - 1):
                    a = np.array(waypoints[w][:7]); b = np.array(waypoints[w + 1][:7])
                    for k in range(6):
                        q = a + (b - a) * k / 6.0
                        for i in range(7):
                            p.resetJointState(rid, i, q[i])
                        time.sleep(0.02)
                time.sleep(1.2)
        else:
            while p.isConnected():
                time.sleep(1 / 120.)
    except p.error:
        pass


if __name__ == "__main__":
    main()
