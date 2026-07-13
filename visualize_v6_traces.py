"""Offscreen V6 renders: scene + EE traces (baseline vs tensor-v3)."""
import pybullet as p
import pybullet_data
import numpy as np
import math, os

REPO = "/home/zhongyanzhang/Zhongyan_Zhang/May/5-14/grasp_rrt_framework_vf"
OUT  = os.path.dirname(os.path.abspath(__file__))
BASE_PATH   = os.path.join(REPO, "benchmark_results_with_A_v6new_base/grasp_path_01.csv")
TENSOR_PATH = os.path.join(REPO, "benchmark_results_with_A_v6v3_tensor/grasp_path_02.csv")
START1 = [0.00412202, -0.944803, 0.620311, -2.04541, -0.48732, 1.33057, 1.50796]


def cylinder_pose(p0, p1):
    a, b = np.array(p0, float), np.array(p1, float)
    v = b - a; L = float(np.linalg.norm(v)); mid = 0.5 * (a + b)
    axis = v / L; z = np.array([0., 0., 1.])
    c = np.cross(z, axis); s = float(np.linalg.norm(c)); d = float(np.dot(z, axis))
    if s < 1e-9:
        return mid.tolist(), ([0, 0, 0, 1] if d > 0 else [1, 0, 0, 0]), L
    c /= s; h = 0.5 * math.atan2(s, d); sh = math.sin(h)
    return mid.tolist(), [c[0]*sh, c[1]*sh, c[2]*sh, math.cos(h)], L


def add_cyl(p0, p1, r, rgba):
    pos, orn, L = cylinder_pose(p0, p1)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=L, rgbaColor=rgba)
    p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=pos, baseOrientation=orn)


def ee_trace(robot, csv_path):
    pts = []
    with open(csv_path) as f:
        for line in f:
            if not line.strip():
                continue
            q = [float(x) for x in line.split(",")][:7]
            for i, qi in enumerate(q):
                p.resetJointState(robot, i, qi)
            pts.append(p.getLinkState(robot, 8)[0])  # link 8 = panda_hand
    return pts


def draw_trace(pts, rgb, r=0.008):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=rgb + [1.0])
    for i, a in enumerate(pts):
        p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=a)
        # densify: midpoint between consecutive waypoints
        if i + 1 < len(pts):
            b = pts[i + 1]
            mid = [(a[j] + b[j]) / 2 for j in range(3)]
            p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=mid)


p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# V6 geometry (mirrors benchmark --diag-scene)
add_cyl((0.40, -0.20, 1.05), (0.55, -0.20, 1.05), 0.02, [0.55, 0.27, 0.07, 1.0])  # A rigid, brown
add_cyl((0.30, -0.30, 1.00), (0.30, 0.30, 1.00), 0.04, [0.20, 0.35, 0.85, 0.75])  # B soft, blue
vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[0.85, 0.1, 0.1, 1.0])
p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[0.42, -0.20, 0.95])   # apple

robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.4], useFixedBase=True)

# EE traces: baseline (orange) vs tensor v3 (green)
draw_trace(ee_trace(robot, BASE_PATH),   [1.0, 0.55, 0.0])
draw_trace(ee_trace(robot, TENSOR_PATH), [0.1, 0.8, 0.1])

# park robot at start1 for the stills
for i, q in enumerate(START1):
    p.resetJointState(robot, i, q)

W, H = 1280, 960
views = {
    "v6_top":   dict(dist=1.1, yaw=0,   pitch=-89, tgt=[0.33, -0.02, 1.0]),
    "v6_side":  dict(dist=1.4, yaw=-55, pitch=-20, tgt=[0.33, 0.0, 1.0]),
    "v6_front": dict(dist=1.2, yaw=90,  pitch=-12, tgt=[0.33, -0.05, 1.0]),
}
for name, v in views.items():
    vm = p.computeViewMatrixFromYawPitchRoll(v["tgt"], v["dist"], v["yaw"], v["pitch"], 0, 2)
    pm = p.computeProjectionMatrixFOV(60, W / H, 0.05, 5.0)
    img = p.getCameraImage(W, H, vm, pm, renderer=p.ER_TINY_RENDERER)
    rgba = np.reshape(np.array(img[2], dtype=np.uint8), (H, W, 4))
    try:
        from PIL import Image
        Image.fromarray(rgba[:, :, :3]).save(os.path.join(OUT, f"{name}.png"))
        print("saved", name)
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(os.path.join(OUT, f"{name}.png"), rgba[:, :, :3])
        print("saved (mpl)", name)
p.disconnect()
