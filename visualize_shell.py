"""
Visualize the apple + spherical cap attractor geometry.

Shows:
    - The apple (red solid sphere, radius = 4 cm at z=0.80)
    - The shell cap (green semi-transparent surface, radius = 5 cm,
      covering the 60° cone below apple = bottom 1/4 of shell sphere)
    - 200 randomly sampled cap points (cyan dots) to verify uniformity
    - Apple center marker (black cross)

Usage:
    python3 visualize_shell.py
    python3 visualize_shell.py --samples 500
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# Scene constants (must match C++ benchmark_grasp_rrt.cpp)
# -------------------------------------------------------------------------
APPLE_POS    = np.array([0.45, 0.0, 0.80])
APPLE_R      = 0.04                                # 4 cm

SHELL_OFFSET = 0.01                                # 1 cm beyond apple surface
SHELL_R      = APPLE_R + SHELL_OFFSET              # 5 cm
CAP_HEIGHT   = 0.015                               # 1.5 cm cap height (from south pole)
CAP_OPEN_DEG = math.degrees(math.acos((SHELL_R - CAP_HEIGHT) / SHELL_R))
                                                   # ≈ 45.6° half-angle from south pole

# Optional context (set False to hide the rest of the scene)
SHOW_SCENE_CONTEXT = True
TRUNK_POS    = np.array([0.7, 0.0, 0.75])
TRUNK_R      = 0.08
TRUNK_H      = 1.5
OVERHEAD_POS = np.array([0.45, 0.0, 0.91])
OVERHEAD_R   = 0.08


# -------------------------------------------------------------------------
def draw_full_sphere(ax, center, radius, color, alpha, n=30):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha,
                    linewidth=0, antialiased=True)


def draw_cap(ax, center, radius, half_angle_deg, color, alpha, n=40):
    """Draw a spherical cap below the center, opening downward.

    half_angle_deg is measured from the -z axis (south pole at the center).
    """
    theta_max = math.radians(half_angle_deg)
    theta = np.linspace(0, theta_max, n)         # angle from south pole
    phi   = np.linspace(0, 2 * np.pi, n)
    T, P  = np.meshgrid(theta, phi)
    x = center[0] + radius * np.sin(T) * np.cos(P)
    y = center[1] + radius * np.sin(T) * np.sin(P)
    z = center[2] - radius * np.cos(T)            # downward (south pole at -z)
    ax.plot_surface(x, y, z, color=color, alpha=alpha,
                    linewidth=0, antialiased=True)
    # Outline of the cap rim
    rim_theta = theta_max
    rim_phi   = np.linspace(0, 2 * np.pi, 100)
    rx = center[0] + radius * math.sin(rim_theta) * np.cos(rim_phi)
    ry = center[1] + radius * math.sin(rim_theta) * np.sin(rim_phi)
    rz = center[2] - radius * math.cos(rim_theta)
    ax.plot(rx, ry, rz, color="darkgreen", linewidth=1.5, alpha=0.8)


def sample_cap_points(n, apple_pos, shell_radius, half_angle_deg):
    """Uniformly sample n points on the spherical cap (same logic as C++)."""
    cos_max = math.cos(math.radians(half_angle_deg))
    # u = cos(theta) where theta from south pole, u ∈ [cos(60°), 1] = [0.5, 1]
    u = np.random.uniform(cos_max, 1.0, n)
    phi = np.random.uniform(0.0, 2.0 * np.pi, n)
    sin_theta = np.sqrt(1.0 - u * u)
    x = apple_pos[0] + shell_radius * sin_theta * np.cos(phi)
    y = apple_pos[1] + shell_radius * sin_theta * np.sin(phi)
    z = apple_pos[2] - shell_radius * u
    return np.column_stack([x, y, z])


def draw_cylinder_simple(ax, base, height, radius, color, alpha=0.25, n=20):
    theta = np.linspace(0, 2 * np.pi, n)
    z = np.linspace(0, height, 2)
    T, Z = np.meshgrid(theta, z)
    X = base[0] + radius * np.cos(T)
    Y = base[1] + radius * np.sin(T)
    Z = base[2] + Z
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200,
                    help="number of cap points to scatter for verification")
    ap.add_argument("--no-context", action="store_true",
                    help="hide trunk + overhead sphere (only apple + cap)")
    args = ap.parse_args()

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Optional scene context (trunk + overhead)
    if SHOW_SCENE_CONTEXT and not args.no_context:
        # Trunk
        trunk_base = TRUNK_POS - np.array([0, 0, TRUNK_H / 2.0])
        draw_cylinder_simple(ax, trunk_base, TRUNK_H, TRUNK_R,
                             color="saddlebrown", alpha=0.15)
        # Overhead obstacle
        draw_full_sphere(ax, OVERHEAD_POS, OVERHEAD_R,
                         color="gray", alpha=0.20)

    # Apple (solid red)
    draw_full_sphere(ax, APPLE_POS, APPLE_R, color="red", alpha=0.85)

    # Shell cap (semi-transparent green)
    draw_cap(ax, APPLE_POS, SHELL_R, CAP_OPEN_DEG,
             color="lime", alpha=0.30)

    # Sample points on the cap
    samples = sample_cap_points(args.samples, APPLE_POS,
                                SHELL_R, CAP_OPEN_DEG)
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
               color="cyan", s=12, edgecolors="darkcyan",
               linewidths=0.4, alpha=0.9,
               label=f"{args.samples} cap samples")

    # Apple center marker
    ax.scatter(*APPLE_POS, color="black", marker="x", s=80, linewidths=2,
               label="apple center")

    # Annotations
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(
        f"Shell cap attractor — apple R={APPLE_R*100:.0f}cm, "
        f"shell R={SHELL_R*100:.0f}cm (1cm offset),\n"
        f"cap half-angle={CAP_OPEN_DEG:.0f}° "
        f"(z_rel ∈ [{-SHELL_R*100:.1f}, {-SHELL_R*math.cos(math.radians(CAP_OPEN_DEG))*100:.1f}] cm)"
    )
    ax.legend(loc="upper left", fontsize=10)

    # Zoom into apple region
    margin = 0.10
    ax.set_xlim(APPLE_POS[0] - margin, APPLE_POS[0] + margin)
    ax.set_ylim(APPLE_POS[1] - margin, APPLE_POS[1] + margin)
    ax.set_zlim(APPLE_POS[2] - margin, APPLE_POS[2] + margin)
    ax.set_box_aspect([1, 1, 1])

    # View from below-front to clearly show the cap
    ax.view_init(elev=15, azim=-60)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
