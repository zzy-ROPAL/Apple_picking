#!/usr/bin/env python3
"""Stage A/B: can branch stiffness be recovered from joint torques in Isaac?

Stage A (LOAD)   replay a planned path slowly into the canopy. At every physics step
                 record the contact force (PhysX), the TRUE branch deflection (each
                 segment's world displacement from its rest pose), the joint torques,
                 and q/q_ref.
Stage B (UNLOAD) stop at the deepest pose, then step BACK along the executed path in
                 increments, holding at each, so df/dx can be compared against the
                 loading slope -- the probe estimator's core assumption.

Why both in one run: the retreat is already part of the repair loop, so if df/dx on
the way out matches k on the way in, the probe measurement is free.

What this answers that no simulation of ours could:
  A1  is f = k*x linear at all?            -> est_static's entire premise
  A2  does the slope match the USD's declared K/L^2?
  A3  how does the planner's pen_model compare to the TRUE deflection x?
      (the planner models branches at a uniform 15 mm while the USD median radius is
       4.6 mm, so pen_model is expected to OVERSTATE -- by how much decides whether
       est_static's denominator is usable on hardware)
  A4  friction mu = Jt/Jn -- never measured, only ever estimated from literature
  B1  is unloading elastic, or is there hysteresis / a friction-reversal dead zone?

Ground truth for deflection is the segment's world displacement rather than the hinge
angle: it is what the contact actually sees, it needs no joint introspection, and it
is directly comparable to the planner's pen.

Outputs (--out-dir):
  steps.csv     t, phase, q0..6, qref0..6, tau0..6, arc
  contacts.csv  t, robot_prim, other_prim, px,py,pz, nx,ny,nz, Jn, Jt, sep
  defl.csv      t, seg_prim, dx, dy, dz, |d|      (only segments that moved > 0.1 mm)
  meta.json
"""

import argparse
import csv
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--path", required=True, help="planned path CSV for the SAME scene")
ap.add_argument("--robot-prim", default="/Franka")
ap.add_argument("--tree-root", default="/World/Tree6")
ap.add_argument("--out-dir", default="stageab_out")
ap.add_argument("--dt", type=float, default=1.0 / 240.0)
ap.add_argument("--vel-scale", type=float, default=0.05,
                help="slow on purpose: quasi-static loading, and Delta-tau needs a ramp")
ap.add_argument("--max-force", type=float, default=40.0)
ap.add_argument("--stiffness", type=float, default=600.0)
ap.add_argument("--damping", type=float, default=50.0)
ap.add_argument("--settle", type=int, default=120)
ap.add_argument("--unload-mm", default="1,2,5,10,20",
                help="unload increments, measured as C-space arc retraced")
ap.add_argument("--hold", type=int, default=120, help="steps held at each unload stop")
ap.add_argument("--headless", action="store_true", default=True)
args = ap.parse_args()

from isaacsim import SimulationApp                                   # noqa: E402
app = SimulationApp({"headless": args.headless})

import numpy as np                                                   # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.prims import SingleArticulation as Articulation   # noqa: E402
from isaacsim.core.utils.stage import open_stage                     # noqa: E402
from isaacsim.core.utils.types import ArticulationAction             # noqa: E402
import omni.usd                                                      # noqa: E402
from omni.physx import get_physx_simulation_interface                # noqa: E402
from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics, UsdGeom, Gf  # noqa: E402

ARM = [f"panda_joint{i}" for i in range(1, 8)]
VMAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])


def load_path(p):
    out = []
    with open(p) as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                out.append([float(x) for x in ln.split(",")[:7]])
    return out


def densify(wps, dt, vs):
    refs = [np.asarray(wps[0], float)]
    for a, b in zip(wps[:-1], wps[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        n = max(1, int(np.ceil(np.max(np.abs(d) / (VMAX * vs * dt)))))
        for i in range(1, n + 1):
            refs.append(a + d * (i / n))
    return refs


class Contacts:
    def __init__(self, robot_root):
        self.root = robot_root
        self.rows = []
        self.t = 0.0
        self._sub = get_physx_simulation_interface() \
            .subscribe_contact_report_events(self._cb)

    def enable(self, stage):
        n = 0
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if p.startswith(self.root) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim) \
                    .CreateThresholdAttr().Set(0.0)
                n += 1
        print(f"[Info] contact reports on {n} robot links", flush=True)

    def _cb(self, headers, data):
        for h in headers:
            a = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            ra, rb = a.startswith(self.root), b.startswith(self.root)
            if ra == rb:
                continue
            robot, other = (a, b) if ra else (b, a)
            for i in range(h.contact_data_offset,
                           h.contact_data_offset + h.num_contact_data):
                d = data[i]
                imp = np.array([d.impulse.x, d.impulse.y, d.impulse.z], float)
                nr = np.array([d.normal.x, d.normal.y, d.normal.z], float)
                nn = np.linalg.norm(nr)
                if nn > 1e-9:
                    nr = nr / nn
                jn = float(abs(imp @ nr))
                jt = float(np.linalg.norm(imp - (imp @ nr) * nr))
                self.rows.append([self.t, robot, other,
                                  float(d.position.x), float(d.position.y),
                                  float(d.position.z),
                                  float(nr[0]), float(nr[1]), float(nr[2]),
                                  jn, jt, float(getattr(d, "separation", 0.0))])


def tame(stage, root, k, d, mf):
    n = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root):
            continue
        for ax in ("angular", "linear", "rotX", "rotY", "rotZ"):
            drv = UsdPhysics.DriveAPI.Get(prim, ax)
            if drv:
                drv.CreateStiffnessAttr().Set(k)
                drv.CreateDampingAttr().Set(d)
                drv.CreateMaxForceAttr().Set(mf)
                n += 1
    print(f"[Info] tamed {n} robot drives k={k} d={d} maxF={mf}", flush=True)


def seg_prims(stage, tree_root):
    """Every branch segment body -- these are what deflect."""
    out = []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith(tree_root) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            out.append((p, UsdGeom.Xformable(prim)))
    return out


def world_pos(xf):
    m = xf.ComputeLocalToWorldTransform(0)
    t = m.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], float)


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    wps = load_path(args.path)
    refs = densify(wps, args.dt, args.vel_scale)
    print(f"[Info] {len(wps)} wps -> {len(refs)} refs "
          f"({len(refs)*args.dt:.1f}s load phase)", flush=True)

    open_stage(args.usd)
    world = World(stage_units_in_meters=1.0, physics_dt=args.dt, rendering_dt=args.dt)
    robot = Articulation(args.robot_prim)
    world.scene.add(robot)
    stage = omni.usd.get_context().get_stage()
    tame(stage, args.robot_prim, args.stiffness, args.damping, args.max_force)
    con = Contacts(args.robot_prim)
    con.enable(stage)

    world.reset()
    dof = list(robot.dof_names)
    arm = np.array([dof.index(n) for n in ARM])
    ctrl = robot.get_articulation_controller()
    segs = seg_prims(stage, args.tree_root)
    print(f"[Info] {len(segs)} tree segment bodies", flush=True)

    def act(q7):
        return ArticulationAction(joint_positions=np.asarray(q7), joint_indices=arm)

    # teleport to wp0 so the approach itself does not pre-bend branches
    qf = robot.get_joint_positions()
    qf[arm] = wps[0]
    robot.set_joint_positions(qf)
    ctrl.apply_action(act(wps[0]))
    for _ in range(args.settle):
        world.step(render=False)

    rest = {p: world_pos(xf) for p, xf in segs}        # rest pose AFTER settling

    steps, defl = [], []
    t = 0.0
    arc = 0.0
    qp = np.asarray(wps[0], float)

    def record(phase, q_ref):
        nonlocal t, arc, qp
        con.t = t
        ctrl.apply_action(act(q_ref))
        world.step(render=False)
        q = np.asarray(robot.get_joint_positions())[arm]
        try:
            tau = np.asarray(robot.get_measured_joint_efforts())[arm]
        except Exception:
            tau = np.zeros(7)
        arc += float(np.linalg.norm(q - qp))
        qp = q
        steps.append([t, phase] + list(q) + list(q_ref) + list(tau) + [arc])
        for p, xf in segs:
            d = world_pos(xf) - rest[p]
            nd = float(np.linalg.norm(d))
            if nd > 1e-4:
                defl.append([t, p, float(d[0]), float(d[1]), float(d[2]), nd])
        t += args.dt

    # ---- Stage A: load ----
    for i, r in enumerate(refs):
        record("load", r)
        if i % 2000 == 0:
            print(f"  load {i}/{len(refs)}  t={t:.1f}s  contacts={len(con.rows)}",
                  flush=True)
    for _ in range(args.hold):
        record("hold", refs[-1])

    # ---- Stage B: unload, stepping back along the executed reference path ----
    # Increments are in C-space arc (rad), retraced along the same refs -- which is
    # exactly what the repair loop's retreat does, so this measures the free probe.
    incs = [float(x) * 1e-3 for x in args.unload_mm.split(",")]
    idx = len(refs) - 1
    for inc in incs:
        target = 0.0
        while idx > 0 and target < inc:
            target += float(np.linalg.norm(refs[idx] - refs[idx - 1]))
            idx -= 1
            record("unload", refs[idx])
        for _ in range(args.hold):
            record("unload_hold", refs[idx])
        print(f"  unloaded {inc*1000:.0f} mm-arc -> ref {idx}, t={t:.1f}s", flush=True)

    with open(os.path.join(args.out_dir, "steps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase"] + [f"q{i}" for i in range(7)]
                   + [f"qref{i}" for i in range(7)]
                   + [f"tau{i}" for i in range(7)] + ["arc"])
        w.writerows(steps)
    with open(os.path.join(args.out_dir, "contacts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "robot_prim", "other_prim", "px", "py", "pz",
                    "nx", "ny", "nz", "Jn", "Jt", "sep"])
        w.writerows(con.rows)
    with open(os.path.join(args.out_dir, "defl.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "seg_prim", "dx", "dy", "dz", "norm"])
        w.writerows(defl)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"usd": args.usd, "path": args.path, "dt": args.dt,
                   "vel_scale": args.vel_scale, "n_steps": len(steps),
                   "n_contacts": len(con.rows), "n_defl": len(defl),
                   "unload_mm": args.unload_mm, "n_segments": len(segs)}, f, indent=2)
    print(f"[Done] steps={len(steps)} contacts={len(con.rows)} defl={len(defl)}",
          flush=True)
    app.close()


if __name__ == "__main__":
    main()
