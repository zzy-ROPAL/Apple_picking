"""
Visualize RRT tree growth from tree_dump_XX.csv.

Usage:
    python3 visualize_tree.py --tree build/.../tree_dump_09.csv
    python3 visualize_tree.py --tree ... --view path     # only solution path
    python3 visualize_tree.py --tree ... --view near     # only nodes near apple

Color encoding:
    Nodes are colored by node_id (= creation order):
      blue  → early
      red   → late
    Yellow circles overlaid = node was within try_grasps_distance² of apple
    Bold red line          = the final solution path (start → grasp)
"""

import argparse
import csv
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ---------------------------------------------------------------------------
# Scene constants (must match C++ SceneConfig)
# ---------------------------------------------------------------------------
TRUNK_POS    = np.array([0.7, 0.0, 0.75])
TRUNK_R      = 0.08
TRUNK_H      = 1.5

# (height, angle, length, radius)
BRANCHES = [
    (0.65, 3.1415, 0.35, 0.03), (0.65, 2.6, 0.3, 0.03),  (0.65, 3.68, 0.3, 0.03),
    (0.80, 2.7, 0.35, 0.03),    (0.80, 3.58, 0.35, 0.03),
    (0.80, 2.1, 0.3, 0.03),     (0.80, 4.18, 0.3, 0.03),
    (0.95, 3.1415, 0.4, 0.03),  (0.95, 2.6, 0.3, 0.03),  (0.95, 3.68, 0.3, 0.03),
]
OVERHEAD_POS = np.array([0.45, 0.0, 0.91])
OVERHEAD_R   = 0.08
APPLE_POS    = np.array([0.45, 0.0, 0.80])
APPLE_R      = 0.04


# ---------------------------------------------------------------------------
# Scene rendering helpers
# ---------------------------------------------------------------------------
def draw_cylinder(ax, base, height, axis, radius, color, alpha=0.3, n=20):
    """Draw a cylinder as a wireframe surface."""
    axis = axis / np.linalg.norm(axis)
    # Find two perpendicular vectors
    if abs(axis[2]) < 0.9:
        perp = np.cross(axis, [0, 0, 1])
    else:
        perp = np.cross(axis, [1, 0, 0])
    perp = perp / np.linalg.norm(perp)
    perp2 = np.cross(axis, perp)

    theta = np.linspace(0, 2 * np.pi, n)
    z = np.linspace(0, height, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)

    pts = (base[None, None, :]
           + z_grid[:, :, None] * axis[None, None, :]
           + x[:, :, None]      * perp[None, None, :]
           + y[:, :, None]      * perp2[None, None, :])

    ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2],
                    color=color, alpha=alpha, linewidth=0, antialiased=False)


def draw_sphere(ax, center, radius, color, alpha=0.3, n=15):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha,
                    linewidth=0, antialiased=False)


def draw_scene(ax):
    # Trunk
    draw_cylinder(ax, TRUNK_POS - np.array([0, 0, TRUNK_H/2]),
                  TRUNK_H, np.array([0, 0, 1]), TRUNK_R,
                  color="saddlebrown", alpha=0.20)

    # Branches (oriented horizontally from trunk)
    for h, angle, length, radius in BRANCHES:
        base = TRUNK_POS.copy()
        base[2] = h
        axis = np.array([math.cos(angle), math.sin(angle), 0])
        draw_cylinder(ax, base, length, axis, radius,
                      color="saddlebrown", alpha=0.25)

    # Overhead sphere
    draw_sphere(ax, OVERHEAD_POS, OVERHEAD_R, color="gray", alpha=0.30)

    # Apple
    draw_sphere(ax, APPLE_POS, APPLE_R, color="red", alpha=0.95)

    # Engulfment cage (vertical twigs) — must match C++ --dense-twigs config:
    #   3 z-rings × 8 angles, radius_to_apple=0.10, half-h=0.03, twig_r=0.02
    if True:
        ring_r = 0.10
        twig_r = 0.02
        half_h = 0.03
        for z in (0.76, 0.80, 0.84):
            for k in range(8):
                ang = k * (math.pi / 4)
                cx = APPLE_POS[0] + ring_r * math.cos(ang)
                cy = APPLE_POS[1] + ring_r * math.sin(ang)
                base = np.array([cx, cy, z - half_h])
                axis = np.array([0.0, 0.0, 1.0])
                draw_cylinder(ax, base, 2 * half_h, axis, twig_r,
                              color="green", alpha=0.55)


# ---------------------------------------------------------------------------
# Tree loading
# ---------------------------------------------------------------------------
def load_tree(csv_path):
    nodes = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row["node_id"])
            nodes[nid] = {
                "node_id": nid,
                "parent_id": int(row["parent_id"]),
                "tcp": np.array([float(row["tcp_x"]),
                                 float(row["tcp_y"]),
                                 float(row["tcp_z"])]),
                "in_pose_sphere": int(row["in_pose_sphere"]) == 1,
                "on_solution_path": int(row["on_solution_path"]) == 1,
            }
    return nodes


def load_face_stats(csv_path):
    """Load pose_sphere face statistics CSV (if present)."""
    if not os.path.exists(csv_path):
        return None
    faces = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            faces.append({
                "face_id":   int(row["face_id"]),
                "attempts":  int(row["attempts"]),
                "failures":  int(row["failures"]),
                "disabled":  int(row["disabled"]) == 1,
                "node_count": int(row["node_count"]),
                "dir":       np.array([float(row["dir_x"]),
                                       float(row["dir_y"]),
                                       float(row["dir_z"])]),
            })
    return faces


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(nodes, view="all", title=""):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Filter nodes per view
    keys = sorted(nodes.keys())
    if view == "path":
        keys = [k for k in keys if nodes[k]["on_solution_path"]]
    elif view == "near":
        keys = [k for k in keys
                if np.linalg.norm(nodes[k]["tcp"] - APPLE_POS) < 0.20]

    if not keys:
        print(f"No nodes match view='{view}'")
        return

    # Color by node_id (creation order)
    ids = np.array(keys)
    norm_ids = (ids - ids.min()) / max(1, ids.max() - ids.min())
    colors = plt.cm.viridis(norm_ids)

    tcps = np.array([nodes[k]["tcp"] for k in keys])

    # Draw scene first
    draw_scene(ax)

    # Edges (lines from child to parent) -- only for nodes in current view
    keys_set = set(keys)
    edges = []
    for k in keys:
        pid = nodes[k]["parent_id"]
        if pid >= 0 and pid in keys_set:
            edges.append([nodes[pid]["tcp"], nodes[k]["tcp"]])
    if edges:
        lc = Line3DCollection(edges, colors="lightgray", linewidths=0.4, alpha=0.4)
        ax.add_collection3d(lc)

    # All nodes (scatter, colored by creation order)
    ax.scatter(tcps[:, 0], tcps[:, 1], tcps[:, 2],
               c=colors, s=4, alpha=0.6, label="tree nodes")

    # pose_sphere nodes (yellow ring overlay)
    ps_keys = [k for k in keys if nodes[k]["in_pose_sphere"]]
    if ps_keys:
        ps_tcps = np.array([nodes[k]["tcp"] for k in ps_keys])
        ax.scatter(ps_tcps[:, 0], ps_tcps[:, 1], ps_tcps[:, 2],
                   facecolors="none", edgecolors="gold",
                   s=80, linewidths=1.2,
                   label=f"in pose_sphere ({len(ps_keys)})")

    # Solution path (bold red line + white dots)
    path_keys_sorted = []
    if any(nodes[k]["on_solution_path"] for k in nodes):
        # Walk solution path from leaf to root, then reverse
        leaf = None
        for k in sorted(nodes.keys(), reverse=True):
            if nodes[k]["on_solution_path"]:
                leaf = k
                break
        if leaf is not None:
            path_keys_sorted = [leaf]
            while True:
                pid = nodes[path_keys_sorted[-1]]["parent_id"]
                if pid < 0 or pid not in nodes:
                    break
                path_keys_sorted.append(pid)
            path_keys_sorted.reverse()

    if path_keys_sorted:
        path_tcps = np.array([nodes[k]["tcp"] for k in path_keys_sorted])
        ax.plot(path_tcps[:, 0], path_tcps[:, 1], path_tcps[:, 2],
                color="red", linewidth=2.5, alpha=0.85,
                label=f"solution path ({len(path_keys_sorted)} nodes)")

        # Start (green star)
        ax.scatter(*path_tcps[0], color="lime", marker="*", s=240,
                   edgecolors="black", linewidths=1, label="start TCP")
        # Grasp (red star)
        ax.scatter(*path_tcps[-1], color="red", marker="*", s=240,
                   edgecolors="black", linewidths=1, label="grasp TCP")

    # Apple position marker
    ax.scatter(*APPLE_POS, color="red", marker="o", s=120,
               edgecolors="black", linewidths=1, label="apple center")

    # Face stats (disabled faces highlighted) — only if face_stats CSV was loaded
    if hasattr(visualize, '_face_stats') and visualize._face_stats:
        FACE_ARROW_LEN = 0.10  # 10 cm 长度的箭头
        disabled_count = 0
        attempted_count = 0
        for fs in visualize._face_stats:
            if fs["attempts"] == 0:
                continue   # 没试过的面不画 (会让图太乱)
            origin = APPLE_POS
            tip = APPLE_POS + fs["dir"] * FACE_ARROW_LEN
            if fs["disabled"]:
                color = "red"
                lw = 3.0
                disabled_count += 1
            else:
                color = "darkorange"
                lw = 1.5
                attempted_count += 1
            ax.plot([origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]],
                    color=color, linewidth=lw, alpha=0.8)
            # 在箭头末端标 face_id + failures
            label_text = f"F{fs['face_id']}({fs['failures']})"
            ax.text(tip[0], tip[1], tip[2], label_text, fontsize=7, color=color)

        # Legend entries
        if disabled_count > 0:
            ax.plot([], [], [], color="red", linewidth=3.0,
                    label=f"disabled faces ({disabled_count})")
        if attempted_count > 0:
            ax.plot([], [], [], color="darkorange", linewidth=1.5,
                    label=f"attempted faces ({attempted_count})")

    # Labels & view
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title or f"RRT tree ({len(keys)} nodes, view='{view}')")
    ax.legend(loc="upper left", fontsize=9)

    # Reasonable initial viewpoint (apple-facing, slight elevation)
    ax.view_init(elev=20, azim=-50)

    # Crop view to scene region
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(0.0, 1.6)
    ax.set_box_aspect([1.4, 1.4, 1.6])

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True,
                    help="Path to tree_dump_XX.csv")
    ap.add_argument("--view", choices=["all", "path", "near"],
                    default="all",
                    help="all: full tree; path: only solution path; "
                         "near: nodes with TCP within 20cm of apple")
    args = ap.parse_args()

    nodes = load_tree(args.tree)
    print(f"Loaded {len(nodes)} tree nodes from {args.tree}")
    n_ps    = sum(1 for n in nodes.values() if n["in_pose_sphere"])
    n_path  = sum(1 for n in nodes.values() if n["on_solution_path"])
    print(f"  In pose_sphere range: {n_ps}")
    print(f"  On solution path:     {n_path}")

    # Try loading sibling face_stats CSV (same dir, similar name)
    face_csv = args.tree.replace("tree_dump_", "face_stats_")
    face_stats = load_face_stats(face_csv)
    if face_stats:
        n_disabled = sum(1 for fs in face_stats if fs["disabled"])
        n_attempted = sum(1 for fs in face_stats if fs["attempts"] > 0)
        print(f"Loaded face stats: {n_attempted} attempted, {n_disabled} disabled")
        visualize._face_stats = face_stats
    else:
        print(f"(no face stats file found at {face_csv})")
        visualize._face_stats = None

    visualize(nodes, view=args.view, title=f"{args.tree} (view={args.view})")


if __name__ == "__main__":
    main()
