import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import os
import argparse

# ===========================================================================
# tree6 reconstructed scene - CONTINUOUS version
#   trunk : 8 segments from 9 nodes, hard + soft SDF (rigid), brown
#   branches: 23 branches x 5 segments (from 6 nodes each), soft SDF only, green
# Mirrors tree6_scene_continuous.cpp
# ===========================================================================

TRUNK_NODES = [
    (0.7526, -0.0624, 0.0011),
    (0.7454, -0.0440, 0.2991),
    (0.7334, -0.0345, 0.5977),
    (0.7319, -0.0335, 0.9226),
    (0.6726, -0.0063, 1.1675),
    (0.6255,  0.0163, 1.3831),
    (0.7273,  0.0267, 1.5939),
    (0.6326,  0.0541, 1.7930),
    (0.6561,  0.1058, 2.0094),
]
TRUNK_RAD = [0.0160, 0.0142, 0.0124, 0.0104, 0.0087, 0.0072, 0.0056, 0.0041, 0.0026]

BRANCH_NODES = [
    [(0.7338,-0.0364,0.6730),(0.6963, 0.0236,0.6883),(0.6504, 0.0901,0.6923),(0.6114, 0.1617,0.6865),(0.5648, 0.2300,0.7012),(0.5171, 0.2968,0.7215)],
    [(0.7341,-0.0374,0.6980),(0.6641,-0.0716,0.6774),(0.6178,-0.1627,0.6953),(0.5486,-0.2075,0.6671),(0.4574,-0.2419,0.6056),(0.3546,-0.2674,0.5607)],
    [(0.7335,-0.0382,0.7231),(0.7821, 0.0132,0.7384),(0.8349, 0.0744,0.7424),(0.8943, 0.1302,0.7365),(0.9486, 0.1925,0.7513),(1.0013, 0.2554,0.7716)],
    [(0.7307,-0.0393,0.7733),(0.6198,-0.0544,0.7767),(0.5131,-0.0478,0.7733),(0.4022,-0.0703,0.7646),(0.2957,-0.1365,0.7597),(0.2079,-0.2281,0.7597)],
    [(0.7328,-0.0420,0.8233),(0.6355,-0.0015,0.8288),(0.5440, 0.0577,0.8254),(0.4381, 0.0931,0.8176),(0.3128, 0.0886,0.8117),(0.1912, 0.0524,0.8118)],
    [(0.7328,-0.0353,0.8973),(0.7047,-0.1514,0.8297),(0.7241,-0.2402,0.7218),(0.7175,-0.1885,0.7896),(0.7677,-0.1361,0.7888),(0.8158,-0.0801,0.8041)],
    [(0.7308,-0.0326,0.9480),(0.7244,-0.0413,0.9473),(0.7179,-0.0500,0.9470),(0.7115,-0.0579,0.9476),(0.7043,-0.0630,0.9475),(0.6975,-0.0660,0.9466)],
    [(0.7296,-0.0307,0.9984),(0.7401,-0.0973,0.9858),(0.7248,-0.1196,0.9603),(0.7172,-0.1549,0.9210),(0.7123,-0.1910,0.8752),(0.7027,-0.2282,0.8309)],
    [(0.7292,-0.0291,1.0235),(0.7921,-0.0049,1.0507),(0.8722, 0.0098,1.0371),(0.9751, 0.0207,1.0386),(1.0928,-0.0137,1.0707),(1.1828,-0.0535,1.1150)],
    [(0.7290,-0.0249,1.0736),(0.6835, 0.0485,1.0521),(0.6157, 0.1149,1.0971),(0.5880, 0.1374,1.1261),(0.5669, 0.0906,1.0594),(0.4825, 0.1128,1.0263)],
    [(0.7140,-0.0168,1.1326),(0.7063,-0.0522,1.1151),(0.6832,-0.0998,1.0719),(0.6702,-0.1373,1.0469),(0.6446,-0.2074,1.0120),(0.6318,-0.2767,0.9944)],
    [(0.6302, 0.0038,1.1993),(0.6715,-0.0195,1.2540),(0.7072,-0.0558,1.2990),(0.7502,-0.0902,1.3270),(0.7862,-0.1337,1.3534),(0.8044,-0.1540,1.4100)],
    [(0.6075, 0.0127,1.2480),(0.6125,-0.0232,1.2305),(0.6075,-0.0759,1.1874),(0.6083,-0.1155,1.1623),(0.6088,-0.1901,1.1275),(0.6209,-0.2595,1.1099)],
    [(0.6106, 0.0135,1.2959),(0.7221, 0.0449,1.2733),(0.8309, 0.0882,1.2315),(0.6704, 0.0552,1.2685),(0.5943, 0.0756,1.2557),(0.5111, 0.1015,1.2659)],
    [(0.6168, 0.0154,1.3664),(0.5907, 0.0009,1.3847),(0.5715,-0.0166,1.3821),(0.5521,-0.0261,1.3871),(0.5183,-0.0402,1.3994),(0.4936,-0.0564,1.4108)],
    [(0.7362, 0.0169,1.4796),(0.7435, 0.0112,1.4844),(0.7497, 0.0056,1.4866),(0.7618,-0.0086,1.4872),(0.7768,-0.0270,1.4866),(0.7887,-0.0363,1.4830)],
    [(0.7381, 0.0208,1.5245),(0.7604, 0.0407,1.5428),(0.7753, 0.0620,1.5402),(0.7922, 0.0755,1.5452),(0.8220, 0.0966,1.5576),(0.8425, 0.1178,1.5689)],
    [(0.7372, 0.0236,1.5489),(0.7464, 0.0242,1.5537),(0.7548, 0.0243,1.5559),(0.7733, 0.0219,1.5565),(0.7968, 0.0183,1.5559),(0.8118, 0.0194,1.5523)],
    [(0.7273, 0.0267,1.5939),(0.7264, 0.0359,1.5986),(0.7262, 0.0443,1.6009),(0.7281, 0.0628,1.6015),(0.7311, 0.0864,1.6009),(0.7296, 0.1014,1.5973)],
    [(0.7033, 0.0290,1.6251),(0.5796, 0.0899,1.5926),(0.3903, 0.0973,1.5967),(0.6138, 0.0567,1.5920),(0.6650,-0.0006,1.5778),(0.7238,-0.0655,1.5875)],
    [(0.6445, 0.0335,1.6729),(0.6474, 0.0423,1.6776),(0.6506, 0.0500,1.6799),(0.6599, 0.0662,1.6805),(0.6722, 0.0865,1.6799),(0.6769, 0.1008,1.6763)],
    [(0.6348, 0.0468,1.7469),(0.6327, 0.0466,1.7580),(0.6287, 0.0453,1.7690),(0.6256, 0.0443,1.7803),(0.6220, 0.0421,1.7913),(0.6180, 0.0392,1.8021)],
    [(0.6317, 0.0577,1.8163),(0.6305, 0.0596,1.8166),(0.6271, 0.0652,1.8177),(0.6237, 0.0712,1.8179),(0.6201, 0.0770,1.8179),(0.6171, 0.0832,1.8178)],
]
BRANCH_RAD = [
    [0.0061,0.0059,0.0053,0.0049,0.0042,0.0036],
    [0.0081,0.0071,0.0059,0.0048,0.0036,0.0025],
    [0.0057,0.0059,0.0053,0.0049,0.0042,0.0036],
    [0.0058,0.0058,0.0053,0.0046,0.0040,0.0033],
    [0.0077,0.0069,0.0055,0.0057,0.0051,0.0046],
    [0.0085,0.0066,0.0041,0.0037,0.0031,0.0023],
    [0.0050,0.0053,0.0044,0.0036,0.0028,0.0023],
    [0.0039,0.0046,0.0039,0.0032,0.0023,0.0016],
    [0.0069,0.0067,0.0061,0.0056,0.0049,0.0043],
    [0.0040,0.0054,0.0048,0.0045,0.0040,0.0035],
    [0.0067,0.0061,0.0052,0.0042,0.0032,0.0022],
    [0.0052,0.0050,0.0046,0.0042,0.0037,0.0033],
    [0.0065,0.0061,0.0052,0.0042,0.0032,0.0022],
    [0.0057,0.0046,0.0033,0.0022,0.0019,0.0016],
    [0.0068,0.0061,0.0053,0.0046,0.0036,0.0029],
    [0.0051,0.0052,0.0045,0.0036,0.0027,0.0020],
    [0.0053,0.0050,0.0043,0.0037,0.0029,0.0023],
    [0.0049,0.0052,0.0045,0.0036,0.0027,0.0020],
    [0.0052,0.0047,0.0041,0.0033,0.0024,0.0018],
    [0.0112,0.0105,0.0096,0.0037,0.0049,0.0047],
    [0.0047,0.0042,0.0037,0.0030,0.0022,0.0016],
    [0.0044,0.0039,0.0035,0.0030,0.0026,0.0021],
    [0.0035,0.0038,0.0035,0.0032,0.0029,0.0027],
]


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
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    p.loadURDF("plane.urdf")

    # Trunk: 8 segments, per-segment radius = average of two endpoint radii
    n_trunk_segs = 0
    for i in range(len(TRUNK_NODES) - 1):
        p0 = TRUNK_NODES[i]
        p1 = TRUNK_NODES[i + 1]
        r = 0.5 * (TRUNK_RAD[i] + TRUNK_RAD[i + 1])
        add_cylinder(p0, p1, r, rgba=[0.55, 0.27, 0.07, 1.0])
        n_trunk_segs += 1

    # --rigid5: render the apple-attached branch (index 4) in brown to indicate
    # it is now a hard obstacle, matching the benchmark's tree6 + rigid-5 mode.
    import sys as _sys
    rigid5 = "--rigid5" in _sys.argv

    # Branches: 23 polylines x 5 segments, radius averaged between adjacent nodes
    n_branch_segs = 0
    for b in range(len(BRANCH_NODES)):
        nodes = BRANCH_NODES[b]
        radii = BRANCH_RAD[b]
        is_apple_branch = (b == 4 and rigid5)
        if is_apple_branch:
            color = [0.55, 0.27, 0.07, 1.0]  # brown, opaque, like trunk
        else:
            color = [0.20, 0.60, 0.20, 0.85]  # green, semi-transparent
        for i in range(len(nodes) - 1):
            p0 = nodes[i]
            p1 = nodes[i + 1]
            r = 0.5 * (radii[i] + radii[i + 1])
            add_cylinder(p0, p1, r, rgba=color)
            n_branch_segs += 1

    # ---- Apple hung on the 5th branch from the bottom (index 4), 25% along it,
    #      i.e. closer to the trunk side than the tip ----
    branch5_nodes = BRANCH_NODES[4]
    branch5_radii = BRANCH_RAD[4]
    # Compute cumulative arclength to find the 25% point.
    seg_lens = []
    for i in range(len(branch5_nodes) - 1):
        a = np.array(branch5_nodes[i])
        b = np.array(branch5_nodes[i + 1])
        seg_lens.append(float(np.linalg.norm(b - a)))
    total_len = sum(seg_lens)
    import sys as _sys
    # --apple-frac F overrides; --apple-root is shorthand for --apple-frac 0.25
    apple_frac = 0.95
    if "--apple-root" in _sys.argv:
        apple_frac = 0.25
    if "--apple-frac" in _sys.argv:
        idx = _sys.argv.index("--apple-frac")
        if idx + 1 < len(_sys.argv):
            apple_frac = max(0.0, min(1.0, float(_sys.argv[idx + 1])))
    target_s = apple_frac * total_len
    acc = 0.0
    apple_attach = None
    local_branch_r = branch5_radii[0]
    for i, seg_len in enumerate(seg_lens):
        if acc + seg_len >= target_s:
            t = (target_s - acc) / seg_len
            a = np.array(branch5_nodes[i])
            b = np.array(branch5_nodes[i + 1])
            apple_attach = (1 - t) * a + t * b
            local_branch_r = 0.5 * (branch5_radii[i] + branch5_radii[i + 1])
            break
        acc += seg_len

    # Apple hangs just below the branch (gravity-down)
    apple_radius = 0.04
    apple_pos = [
        float(apple_attach[0]),
        float(apple_attach[1]),
        float(apple_attach[2]) - local_branch_r - apple_radius,
    ]

    # Red apple sphere
    apple_col = p.createCollisionShape(p.GEOM_SPHERE, radius=apple_radius)
    apple_vis = p.createVisualShape(p.GEOM_SPHERE, radius=apple_radius,
                                     rgbaColor=[0.85, 0.10, 0.10, 1.0])
    p.createMultiBody(baseMass=0,
                      baseCollisionShapeIndex=apple_col,
                      baseVisualShapeIndex=apple_vis,
                      basePosition=apple_pos)

    # Small brown stem connecting branch to apple
    stem_p0 = [float(apple_attach[0]), float(apple_attach[1]), float(apple_attach[2])]
    stem_p1 = [float(apple_pos[0]), float(apple_pos[1]), float(apple_pos[2]) + apple_radius]
    add_cylinder(stem_p0, stem_p1, 0.003, rgba=[0.40, 0.22, 0.06, 1.0])

    print(f"[tree6 continuous] Apple hung at {apple_pos} on branch #5 (25% along, "
          f"total branch length {total_len:.3f} m, local branch r {local_branch_r:.4f} m).")

    # Panda at home pose (matches benchmark)
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.4], useFixedBase=True)
    home_pose = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
    for i, q in enumerate(home_pose):
        p.resetJointState(robot_id, i, q)
    import sys as _sys2
    if "--start" in _sys2.argv:
        _si = _sys2.argv.index("--start")
        if _si + 1 < len(_sys2.argv):
            vals = [float(x) for x in _sys2.argv[_si + 1].split(",")]
            for i in range(7):
                p.resetJointState(robot_id, i, vals[i])
            print("[tree6 viz] robot posed at --start config")

    # Camera framing the tree from the front
    p.resetDebugVisualizerCamera(
        cameraDistance=2.2,
        cameraYaw=-90.0,
        cameraPitch=-15.0,
        cameraTargetPosition=[0.6, 0.0, 1.0],
    )

    print(f"[tree6 continuous] {n_trunk_segs} trunk segments + "
          f"{n_branch_segs} branch segments loaded.")

    # ---- Optional trajectory playback ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="CSV trajectory file")
    args, _ = parser.parse_known_args()

    waypoints = []
    if args.path:
        candidates = [args.path, os.path.join("build", args.path), os.path.join("..", args.path)]
        path_file = next((c for c in candidates if os.path.exists(c)), None)
        if path_file is None:
            print(f"[tree6 viz] Could not find trajectory file: {args.path}")
        else:
            print(f"[tree6 viz] Loading trajectory from {path_file}")
            with open(path_file, "r") as f:
                for line in f:
                    if line.strip():
                        waypoints.append([float(x) for x in line.strip().split(",")])
            print(f"[tree6 viz] Loaded {len(waypoints)} waypoints.")

    revolute_joints = [0, 1, 2, 3, 4, 5, 6]

    if waypoints:
        try:
            while p.isConnected():
                # Snap to start
                for i, q in enumerate(waypoints[0][:7]):
                    p.resetJointState(robot_id, revolute_joints[i], q)
                time.sleep(0.8)
                # Play
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
        print("[tree6 viz] No trajectory provided — static scene only. Close window to exit.")
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1.0 / 120.0)
        except p.error:
            pass


if __name__ == "__main__":
    main()
