"""
visualize_vector_field.py

Visualize the workspace vector field used by GraspRRT (computeVectorField in grasp_rrt.cpp).
Shows attractive (toward apple), repulsive (trunk + overhead), and vortex components
as 2D quiver slices through y=0 and z=0.8 planes.

Usage:
    python3 visualize_vector_field.py
    python3 visualize_vector_field.py --component attractive
    python3 visualize_vector_field.py --component repulsive
    python3 visualize_vector_field.py --component total
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

# ---- Scene parameters (must match example_grasp_rrt.cpp SceneConfig) ----
APPLE_POS    = np.array([0.45, 0.0, 0.80])
APPLE_RADIUS = 0.04

TRUNK_AXIS_XY = np.array([0.7, 0.0])  # trunk is vertical along z
TRUNK_RADIUS  = 0.08
TRUNK_TRIGGER = 0.20  # repulsion radius

OVERHEAD_POS     = np.array([0.45, 0.0, 0.91])
OVERHEAD_RADIUS  = 0.08
OVERHEAD_TRIGGER = 0.15

# ---- Weights from grasp_rrt.cpp:379-382 ----
W_ATTRACT  = 4.0
W_TRUNK    = 2.5
W_VORTEX   = 2.0
W_OVERHEAD = 2.5

VIRTUAL_TARGET = APPLE_POS - np.array([0, 0, 0.08])


def attractive_force(p):
    """Pull toward virtual_target (apple - 8cm z). Unit-normalized direction."""
    d = VIRTUAL_TARGET - p
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.zeros(3)
    return d / n  # normalized; magnitude is constant 1


def trunk_repulsion(p):
    """Trunk hard repulsion: 1/d^2 within trigger radius. Vortex tangent included."""
    p_xy = p[:2]
    diff = p_xy - TRUNK_AXIS_XY
    d = np.linalg.norm(diff)
    if d > TRUNK_TRIGGER or d < 0.01:
        return np.zeros(3), np.zeros(3)
    n_xy = diff / d
    mag = 1.0 / (d * d)
    rep = np.array([n_xy[0], n_xy[1], 0.0]) * mag
    # Vortex: z × r_hat (tangential swirl in xy)
    v_xy = np.array([-n_xy[1], n_xy[0]])
    vortex = np.array([v_xy[0], v_xy[1], 0.0]) * mag * 1.5
    return rep, vortex


def overhead_repulsion(p):
    """Overhead spherical hard repulsion: 1/d^2 within trigger radius."""
    diff = p - OVERHEAD_POS
    d = np.linalg.norm(diff)
    if d > OVERHEAD_TRIGGER or d < 0.01:
        return np.zeros(3)
    n = diff / d
    mag = 1.0 / (d * d)
    return n * mag


def total_field(p):
    """Weighted sum mirroring grasp_rrt.cpp:379-383."""
    att = attractive_force(p)
    trunk_rep, vortex = trunk_repulsion(p)
    overhead_rep = overhead_repulsion(p)

    # Each component is independently normalized in the C++ code before weighting.
    def safe_norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    return (W_ATTRACT  * safe_norm(att) +
            W_TRUNK    * safe_norm(trunk_rep) +
            W_VORTEX   * safe_norm(vortex) +
            W_OVERHEAD * safe_norm(overhead_rep))


def field_at_grid_xz(component, y_slice=0.0, n=30):
    """Evaluate a field component on a grid in x-z plane at given y."""
    xs = np.linspace(0.1, 1.0, n)
    zs = np.linspace(0.4, 1.3, n)
    X, Z = np.meshgrid(xs, zs)
    U = np.zeros_like(X)
    W = np.zeros_like(Z)
    M = np.zeros_like(X)  # magnitude

    for i in range(n):
        for j in range(n):
            p = np.array([X[i, j], y_slice, Z[i, j]])
            if component == 'attractive':
                v = attractive_force(p)
            elif component == 'trunk':
                v, _ = trunk_repulsion(p)
            elif component == 'vortex':
                _, v = trunk_repulsion(p)
            elif component == 'overhead':
                v = overhead_repulsion(p)
            elif component == 'total':
                v = total_field(p)
            else:
                raise ValueError(f"Unknown component: {component}")
            U[i, j] = v[0]
            W[i, j] = v[2]
            M[i, j] = np.linalg.norm(v)
    return X, Z, U, W, M


def draw_scene_xz(ax):
    """Draw apple, trunk, overhead obstacle on x-z plane (y=0 slice)."""
    # Apple (red)
    ax.add_patch(plt.Circle(APPLE_POS[[0, 2]], APPLE_RADIUS, color='red',
                            label='apple', zorder=5))
    # Overhead obstacle (brown)
    ax.add_patch(plt.Circle(OVERHEAD_POS[[0, 2]], OVERHEAD_RADIUS,
                            color='saddlebrown', alpha=0.7,
                            label='overhead obstacle', zorder=4))
    # Trunk vertical band
    trunk_x = TRUNK_AXIS_XY[0]
    ax.axvspan(trunk_x - TRUNK_RADIUS, trunk_x + TRUNK_RADIUS,
               color='peru', alpha=0.4, label='trunk', zorder=2)
    # Trigger radii (dashed circles)
    ax.add_patch(plt.Circle(OVERHEAD_POS[[0, 2]], OVERHEAD_TRIGGER,
                            fill=False, edgecolor='brown', linestyle='--',
                            alpha=0.5, label='overhead trigger'))
    # Trunk trigger band
    ax.axvspan(trunk_x - TRUNK_TRIGGER, trunk_x + TRUNK_TRIGGER,
               color='peru', alpha=0.1, label='trunk trigger')
    # Virtual target (where the attractor pulls)
    ax.plot(*VIRTUAL_TARGET[[0, 2]], 'k+', markersize=12,
            markeredgewidth=2, label='virtual target')


def plot_component(ax, component, y_slice=0.0, n=30, color_by_mag=True):
    X, Z, U, W, M = field_at_grid_xz(component, y_slice=y_slice, n=n)
    draw_scene_xz(ax)
    if color_by_mag and M.max() > 1e-6:
        q = ax.quiver(X, Z, U, W, M, cmap='viridis',
                      scale=30, width=0.003, alpha=0.85)
        plt.colorbar(q, ax=ax, label='|F|', shrink=0.8)
    else:
        ax.quiver(X, Z, U, W, scale=30, width=0.003, color='steelblue', alpha=0.7)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.4, 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title(f'{component} field  (y={y_slice:.2f} slice)')
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', default='all',
                        choices=['attractive', 'trunk', 'vortex',
                                 'overhead', 'total', 'all'])
    parser.add_argument('--n', type=int, default=30, help='grid resolution')
    parser.add_argument('--y', type=float, default=0.0, help='y-slice for x-z plot')
    args = parser.parse_args()

    if args.component == 'all':
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        for ax, comp in zip(axes.flat,
                            ['attractive', 'trunk', 'vortex',
                             'overhead', 'total', 'total']):
            plot_component(ax, comp, y_slice=args.y, n=args.n)
        # Hide the duplicate
        axes.flat[5].set_title('total field (duplicate hidden)')
        axes.flat[5].set_visible(False)
        # Add a single legend on the first plot
        axes[0, 0].legend(loc='upper right', fontsize=8)
        fig.suptitle(f'GraspRRT Vector Field Components  '
                     f'(y={args.y:.2f} slice through scene)',
                     fontsize=14)
    else:
        fig, ax = plt.subplots(figsize=(8, 9))
        plot_component(ax, args.component, y_slice=args.y, n=args.n)
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
