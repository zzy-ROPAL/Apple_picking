"""
visualize_vector_field_3d.py

Show the GraspRRT workspace vector field as 3D arrows in PyBullet,
overlaid on the actual scene (Panda, apple, trunk, branches, overhead obstacle).

Usage:
    python3 visualize_vector_field_3d.py
    python3 visualize_vector_field_3d.py --component total
    python3 visualize_vector_field_3d.py --component overhead --grid 8
    python3 visualize_vector_field_3d.py --no-robot   # skip Panda load
"""

import pybullet as p
import pybullet_data
import numpy as np
import math
import argparse
import time

# ---------- scene parameters (mirror example_grasp_rrt.cpp) ----------
APPLE_POS    = np.array([0.45, 0.0, 0.80])
APPLE_RADIUS = 0.04

TRUNK_AXIS_XY = np.array([0.7, 0.0])
TRUNK_RADIUS  = 0.08
TRUNK_HEIGHT  = 1.5
TRUNK_TRIGGER = 0.20

OVERHEAD_POS     = np.array([0.45, 0.0, 0.91])
OVERHEAD_RADIUS  = 0.08
OVERHEAD_TRIGGER = 0.15

W_ATTRACT, W_TRUNK, W_VORTEX, W_OVERHEAD, W_LEAVES = 4.0, 2.5, 2.0, 2.5, 1.2
VIRTUAL_TARGET = APPLE_POS - np.array([0, 0, 0.08])
LEAF_TRIGGER = 0.15  # branch (soft) repulsion radius

BRANCHES = [
    (0.65, 3.1415, 0.35, 0.03), (0.65, 2.6, 0.30, 0.03), (0.65, 3.68, 0.30, 0.03),
    (0.80, 2.7,    0.35, 0.03), (0.80, 3.58, 0.35, 0.03),
    (0.80, 2.1,    0.30, 0.03), (0.80, 4.18, 0.30, 0.03),
    (0.95, 3.1415, 0.40, 0.03), (0.95, 2.6,  0.30, 0.03), (0.95, 3.68, 0.30, 0.03),
]

# ---------- field functions (same math as grasp_rrt.cpp computeVectorField) ----------
def attractive(p_):
    d = VIRTUAL_TARGET - p_
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else np.zeros(3)

def trunk_force(p_):
    diff_xy = p_[:2] - TRUNK_AXIS_XY
    d = np.linalg.norm(diff_xy)
    if d > TRUNK_TRIGGER or d < 0.01:
        return np.zeros(3), np.zeros(3)
    n_xy = diff_xy / d
    mag = 1.0 / (d * d)
    rep = np.array([n_xy[0], n_xy[1], 0.0]) * mag
    v_xy = np.array([-n_xy[1], n_xy[0]])
    vortex = np.array([v_xy[0], v_xy[1], 0.0]) * mag * 1.5
    return rep, vortex

def overhead_force(p_):
    diff = p_ - OVERHEAD_POS
    d = np.linalg.norm(diff)
    if d > OVERHEAD_TRIGGER or d < 0.01:
        return np.zeros(3)
    return (diff / d) * (1.0 / (d * d))

def _branch_endpoints():
    """Return (p0, p1, radius) for each branch, in world coords."""
    out = []
    trunk_pos = np.array([TRUNK_AXIS_XY[0], TRUNK_AXIS_XY[1], 0.75])
    for h, ang, length, rad in BRANCHES:
        # branch starts at trunk surface and extends along (cos, sin) at height h
        c = math.cos(ang); s = math.sin(ang)
        p0 = np.array([trunk_pos[0] + TRUNK_RADIUS * c,
                       trunk_pos[1] + TRUNK_RADIUS * s, h])
        p1 = np.array([trunk_pos[0] + (TRUNK_RADIUS + length) * c,
                       trunk_pos[1] + (TRUNK_RADIUS + length) * s, h])
        out.append((p0, p1, rad))
    return out

_BRANCH_SEGMENTS = _branch_endpoints()

def _dist_point_to_segment(pt, a, b):
    """Closest point on segment ab to pt; returns (closest, distance)."""
    ab = b - a
    t = np.dot(pt - a, ab) / max(np.dot(ab, ab), 1e-9)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return closest, np.linalg.norm(pt - closest)

def leaves_force(p_):
    """Approx branch (leaves) repulsion: 1/d^2 from each branch surface within trigger."""
    total = np.zeros(3)
    for (a, b, rad) in _BRANCH_SEGMENTS:
        closest, d_axis = _dist_point_to_segment(p_, a, b)
        d_surf = max(d_axis - rad, 0.005)
        if d_surf > LEAF_TRIGGER:
            continue
        n = (p_ - closest)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        total += n * (1.0 / (d_surf * d_surf))
    return total

def total_field(p_):
    def safe(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v
    att = attractive(p_)
    tr, vx = trunk_force(p_)
    ov = overhead_force(p_)
    lv = leaves_force(p_)
    return (W_ATTRACT  * safe(att) +
            W_TRUNK    * safe(tr)  +
            W_VORTEX   * safe(vx)  +
            W_OVERHEAD * safe(ov)  +
            W_LEAVES   * safe(lv))

def field_at(component, p_):
    if component == 'attractive':
        return attractive(p_)
    if component == 'trunk':
        return trunk_force(p_)[0]
    if component == 'vortex':
        return trunk_force(p_)[1]
    if component == 'overhead':
        return overhead_force(p_)
    if component == 'leaves':
        return leaves_force(p_)
    if component == 'total':
        return total_field(p_)
    raise ValueError(component)

# ---------- pybullet helpers ----------
def color_for(mag, mag_max):
    """viridis-ish: low=blue, mid=green, high=yellow."""
    t = min(mag / mag_max, 1.0) if mag_max > 0 else 0.0
    if t < 0.5:
        s = t * 2
        return [0.1, s, 1.0 - s]   # blue -> green
    s = (t - 0.5) * 2
    return [s, 1.0, 0.0]            # green -> yellow

def draw_arrow(start, vec, color, length=0.05, lifetime=0):
    """Draw an arrow from `start` along normalized `vec` of given length."""
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return
    direction = vec / n
    end = np.array(start) + direction * length
    p.addUserDebugLine(start.tolist(), end.tolist(), color,
                       lineWidth=2, lifeTime=lifetime)
    # arrow head: two short barbs at 30 deg
    # build a perpendicular axis
    if abs(direction[2]) < 0.9:
        perp = np.cross(direction, [0, 0, 1])
    else:
        perp = np.cross(direction, [1, 0, 0])
    perp /= (np.linalg.norm(perp) + 1e-9)
    barb_len = length * 0.3
    barb_dir = (-direction * math.cos(math.radians(30))
                + perp * math.sin(math.radians(30)))
    p.addUserDebugLine(end.tolist(),
                       (end + barb_dir * barb_len).tolist(),
                       color, lineWidth=2, lifeTime=lifetime)
    barb_dir2 = (-direction * math.cos(math.radians(30))
                 - perp * math.sin(math.radians(30)))
    p.addUserDebugLine(end.tolist(),
                       (end + barb_dir2 * barb_len).tolist(),
                       color, lineWidth=2, lifeTime=lifetime)

# ---------- scene builder ----------
def build_scene(load_robot=True):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # Trunk
    trunk_pos = [TRUNK_AXIS_XY[0], TRUNK_AXIS_XY[1], 0.75]
    tc = p.createCollisionShape(p.GEOM_CYLINDER, radius=TRUNK_RADIUS, height=TRUNK_HEIGHT)
    tv = p.createVisualShape(p.GEOM_CYLINDER, radius=TRUNK_RADIUS,
                             length=TRUNK_HEIGHT, rgbaColor=[0.55, 0.27, 0.07, 1])
    p.createMultiBody(0, tc, tv, basePosition=trunk_pos)

    # Branches
    for h, ang, length, rad in BRANCHES:
        cx = trunk_pos[0] + (TRUNK_RADIUS + length / 2) * math.cos(ang)
        cy = trunk_pos[1] + (TRUNK_RADIUS + length / 2) * math.sin(ang)
        bc = p.createCollisionShape(p.GEOM_CYLINDER, radius=rad, height=length)
        bv = p.createVisualShape(p.GEOM_CYLINDER, radius=rad, length=length,
                                 rgbaColor=[0.2, 0.6, 0.2, 0.5])
        qy = p.getQuaternionFromEuler([0, math.pi / 2, 0])
        qz = p.getQuaternionFromEuler([0, 0, ang])
        _, orn = p.multiplyTransforms([0, 0, 0], qz, [0, 0, 0], qy)
        p.createMultiBody(0, bc, bv, basePosition=[cx, cy, h], baseOrientation=orn)

    # Overhead obstacle
    oc = p.createCollisionShape(p.GEOM_SPHERE, radius=OVERHEAD_RADIUS)
    ov_ = p.createVisualShape(p.GEOM_SPHERE, radius=OVERHEAD_RADIUS,
                              rgbaColor=[0.55, 0.27, 0.07, 0.85])
    p.createMultiBody(0, oc, ov_, basePosition=OVERHEAD_POS.tolist())

    # Apple
    ac = p.createCollisionShape(p.GEOM_SPHERE, radius=APPLE_RADIUS)
    av = p.createVisualShape(p.GEOM_SPHERE, radius=APPLE_RADIUS,
                             rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(0, ac, av, basePosition=APPLE_POS.tolist())

    # Virtual target marker
    vc = p.createVisualShape(p.GEOM_SPHERE, radius=0.012,
                             rgbaColor=[1, 1, 0, 1])
    p.createMultiBody(0, -1, vc, basePosition=VIRTUAL_TARGET.tolist())

    if load_robot:
        try:
            p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.4], useFixedBase=True)
        except Exception:
            pass

    p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=60,
                                 cameraPitch=-25,
                                 cameraTargetPosition=[0.5, 0.0, 0.7])

# ---------- field drawing ----------
def draw_field(component, grid=6, arrow_len=0.05, mag_floor=1e-3):
    xs = np.linspace(0.20, 0.95, grid)
    ys = np.linspace(-0.30, 0.30, max(2, grid // 2))
    zs = np.linspace(0.45, 1.20, grid)

    # First pass: collect magnitudes for color scaling
    pts, vecs, mags = [], [], []
    for x in xs:
        for y in ys:
            for z in zs:
                p_ = np.array([x, y, z])
                v = field_at(component, p_)
                m = np.linalg.norm(v)
                if m < mag_floor:
                    continue
                pts.append(p_)
                vecs.append(v)
                mags.append(m)

    if not mags:
        print(f"[warn] no arrows above magnitude {mag_floor} for {component}")
        return

    mag_max = max(mags)
    print(f"[info] drawing {len(pts)} arrows ({component}), mag range "
          f"[{min(mags):.3f}, {mag_max:.3f}]")

    for pt, vec, m in zip(pts, vecs, mags):
        col = color_for(m, mag_max)
        draw_arrow(pt, vec, col, length=arrow_len)

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', default='total',
                        choices=['attractive', 'trunk', 'vortex',
                                 'overhead', 'leaves', 'total'])
    parser.add_argument('--w-attract', type=float, default=None,
                        help='override W_ATTRACT (default 4.0)')
    parser.add_argument('--grid', type=int, default=6,
                        help='grid resolution per axis (default 6)')
    parser.add_argument('--len', type=float, default=0.05,
                        help='arrow length in meters (default 0.05)')
    parser.add_argument('--no-robot', action='store_true',
                        help='skip loading the Panda for cleaner view')
    args = parser.parse_args()

    if args.w_attract is not None:
        global W_ATTRACT
        W_ATTRACT = args.w_attract
        print(f"[info] using W_ATTRACT = {W_ATTRACT}")

    build_scene(load_robot=not args.no_robot)
    draw_field(args.component, grid=args.grid, arrow_len=args.len)

    print("Visualization ready. Close the PyBullet window to exit.")
    while p.isConnected():
        time.sleep(0.1)


if __name__ == '__main__':
    main()
