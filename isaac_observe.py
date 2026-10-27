#!/usr/bin/env python3
"""Observe-only Isaac run: log every signal the attribution estimators need, plus
PhysX ground truth, so the estimators can be scored OFFLINE against the truth.

STATUS: WRITTEN, NEVER RUN. Isaac is not installed on the dev box (no `omni`), so
this has not been executed even once. Treat every line as a draft until it has.

Why observe-only, and why this is the gating experiment. Every attribution number in
CLAUDE.md comes from a synthetic forward model whose relationship to reality is
unverified -- and four separate conclusions reversed during development purely because
the test setup changed. The single run this script performs resolves:

  1. is  f = k * pen  (the planner's penalty-spring model) anything like reality?
     -> compare PhysX normal force against k*pen for the branch in contact
  2. how large is friction, i.e. how far from radial is the real contact force?
     -> |tangential impulse| / |normal impulse| per contact
  3. what is the sign of the torque signal?
     -> compare measured efforts against J^T(+r_hat)*f using PhysX's own f
  4. what is StallDetector's false-positive rate on a NOMINAL scene?
     -> it should never fire; count how often it does
  5. which estimator is actually best?
     -> score force-ranking / static-stiffness / probe against the PhysX truth

Deliberately NOT a closed loop: nothing is repaired, no scene is edited, no decision
acts on the robot. Replay the nominal planned path, record, stop.

This file imports Isaac. `contact_monitor.py` must stay Isaac-free, so the scoring
lives in `score_observe.py`, which reads these CSVs in the pybullet venv.

Outputs (--out-dir):
  steps.csv     t, q0..q6, qref0..qref6, tau0..tau6, arc           (per physics step)
  contacts.csv  t, robot_prim, other_prim, cls, px,py,pz, nx,ny,nz, Jn, Jt, sep
                  one row per PhysX contact point; Jn/Jt are the normal/tangential
                  impulse magnitudes, from which force = J/dt and mu = Jt/Jn
  meta.json     dt, vel_scale, path, usd, joint order, effort caps

Usage (from an Isaac python):
  ./python.sh isaac_observe.py --usd scene.usd --robot-prim /World/panda \
      --path grasp_path_01.csv --out-dir obs_run01 --headless
"""

import argparse
import csv
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--robot-prim", required=True)
ap.add_argument("--path", required=True, help="planned path CSV, 7 joint values/row")
ap.add_argument("--dt", type=float, default=1.0 / 240.0,
                help="physics step. 240 Hz by default: the >=5 N load ramp that dtau "
                     "needs, and the averaging that the probe estimator needs, both "
                     "want a high sample rate")
ap.add_argument("--vel-scale", type=float, default=0.10,
                help="fraction of joint vmax. LOW on purpose: Delta-tau needs the "
                     "load to rise >=5 N inside the stall window, which a fast "
                     "approach does not produce")
ap.add_argument("--max-force", type=float, default=40.0,
                help="per-joint drive force cap (N*m). MUST be finite: an infinite "
                     "maxForce plus the 6 g branch segments makes the replay explode "
                     "(see the Isaac branch-explosion note)")
ap.add_argument("--damping", type=float, default=50.0)
ap.add_argument("--stiffness", type=float, default=600.0,
                help="drive stiffness. Compliant on purpose -- a stiff position servo "
                     "drives the tracking error to ~0 and destroys the K_p*e channel")
ap.add_argument("--trunk-pattern", default="trunk")
ap.add_argument("--branch-pattern", default="branch")
ap.add_argument("--hold-time", type=float, default=2.0)
ap.add_argument("--out-dir", default="obs_out")
ap.add_argument("--headless", action="store_true")
args = ap.parse_args()

from isaacsim import SimulationApp                                 # noqa: E402
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np                                                 # noqa: E402

try:                                            # 4.5+
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:                             # 2022.2 - 4.2
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.stage import open_stage
    from omni.isaac.core.utils.types import ArticulationAction

import omni.usd                                                    # noqa: E402
from omni.physx import get_physx_simulation_interface               # noqa: E402
from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics         # noqa: E402

ARM = [f"panda_joint{i}" for i in range(1, 8)]
PANDA_VMAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])


def load_waypoints(path):
    wps = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                wps.append([float(x) for x in ln.split(",")[:7]])
    if not wps:
        sys.exit(f"[Error] no waypoints in {path}")
    return wps


def time_parameterize(wps, dt, vel_scale):
    """Dense reference trajectory at the control rate, capped at vel_scale*vmax."""
    refs = [np.asarray(wps[0], dtype=float)]
    for a, b in zip(wps[:-1], wps[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        d = b - a
        n = max(1, int(np.ceil(np.max(np.abs(d) / (PANDA_VMAX * vel_scale * dt)))))
        for i in range(1, n + 1):
            refs.append(a + d * (i / n))
    return refs


class TruthLogger:
    """PhysX contact reports, logged per contact POINT with normal and impulse split.

    The normal is what makes this ground truth rather than just a flag: it lets the
    offline scorer check the radial-force assumption directly, and the tangential
    impulse gives the friction coefficient -- the number the residual's reliability
    hinges on and which has never been measured.
    """

    def __init__(self, robot_root, trunk_keys, branch_keys):
        self.robot_root = robot_root
        self.trunk_keys = [k.strip().lower() for k in trunk_keys.split(",") if k.strip()]
        self.branch_keys = [k.strip().lower() for k in branch_keys.split(",") if k.strip()]
        self.rows = []
        self.t = 0.0
        self.trunk_hit = False
        self._sub = get_physx_simulation_interface() \
            .subscribe_contact_report_events(self._on_contact)

    def enable_on_robot(self, stage):
        n = 0
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if p.startswith(self.robot_root) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim) \
                    .CreateThresholdAttr().Set(0.0)
                n += 1
        print(f"[Info] contact reports enabled on {n} robot links")

    def _cls(self, path):
        low = path.lower()
        if any(k in low for k in self.trunk_keys):
            return "trunk"
        if any(k in low for k in self.branch_keys):
            return "branch"
        return "other"

    def _on_contact(self, headers, data):
        for h in headers:
            a = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            ra, rb = a.startswith(self.robot_root), b.startswith(self.robot_root)
            if ra == rb:
                continue                      # self-collision or tree-internal
            robot, other = (a, b) if ra else (b, a)
            cls = self._cls(other)
            if cls == "trunk":
                self.trunk_hit = True
            for i in range(h.contact_data_offset,
                           h.contact_data_offset + h.num_contact_data):
                d = data[i]
                imp = np.array([d.impulse.x, d.impulse.y, d.impulse.z], dtype=float)
                nrm = np.array([d.normal.x, d.normal.y, d.normal.z], dtype=float)
                nn = float(np.linalg.norm(nrm))
                if nn > 1e-9:
                    nrm = nrm / nn
                jn = float(abs(np.dot(imp, nrm)))
                jt = float(np.linalg.norm(imp - np.dot(imp, nrm) * nrm))
                self.rows.append([
                    self.t, robot, other, cls,
                    float(d.position.x), float(d.position.y), float(d.position.z),
                    float(nrm[0]), float(nrm[1]), float(nrm[2]),
                    jn, jt, float(getattr(d, "separation", 0.0)),
                ])


def tame_drives(stage, robot_root, stiffness, damping, max_force):
    """Finite force cap + compliant gains on every arm drive.

    Two separate reasons, both load-bearing:
      - an infinite maxForce with the heavy branch segments makes the replay explode
      - a stiff servo drives q_ref - q to zero, which destroys the K_p*e channel and
        makes the stall trigger's tracking-error term meaningless
    """
    n = 0
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith(robot_root):
            continue
        for api_name in ("angular", "linear"):
            drv = UsdPhysics.DriveAPI.Get(prim, api_name)
            if not drv:
                continue
            drv.CreateStiffnessAttr().Set(stiffness)
            drv.CreateDampingAttr().Set(damping)
            drv.CreateMaxForceAttr().Set(max_force)
            n += 1
    print(f"[Info] tamed {n} drives: k={stiffness} d={damping} maxF={max_force}")
    return n


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    wps = load_waypoints(args.path)
    refs = time_parameterize(wps, args.dt, args.vel_scale)
    print(f"[Info] {len(wps)} waypoints -> {len(refs)} refs, "
          f"{len(refs)*args.dt:.1f}s at vel_scale={args.vel_scale}")

    open_stage(args.usd)
    world = World(stage_units_in_meters=1.0, physics_dt=args.dt, rendering_dt=args.dt)
    robot = Articulation(args.robot_prim)
    world.scene.add(robot)

    stage = omni.usd.get_context().get_stage()
    tame_drives(stage, args.robot_prim, args.stiffness, args.damping, args.max_force)
    truth = TruthLogger(args.robot_prim, args.trunk_pattern, args.branch_pattern)
    truth.enable_on_robot(stage)

    world.reset()
    dof = list(robot.dof_names)
    try:
        arm = np.array([dof.index(n) for n in ARM])
    except ValueError as e:
        sys.exit(f"[Error] joint not found: {e}")
    ctrl = robot.get_articulation_controller()

    def act(q7):
        return ArticulationAction(joint_positions=np.asarray(q7), joint_indices=arm)

    q_full = robot.get_joint_positions()
    q_full[arm] = wps[0]
    robot.set_joint_positions(q_full)
    ctrl.apply_action(act(wps[0]))
    for _ in range(60):
        world.step(render=not args.headless)

    steps = []
    arc = 0.0
    q_prev = np.asarray(wps[0], dtype=float)
    hold = int(args.hold_time / args.dt)
    t = 0.0
    for i in range(len(refs) + hold):
        q_ref = refs[min(i, len(refs) - 1)]
        truth.t = t
        ctrl.apply_action(act(q_ref))
        world.step(render=not args.headless)

        q = np.asarray(robot.get_joint_positions())[arm]
        # measured efforts: what the drives actually applied. The offline scorer
        # subtracts the dynamics model, or uses K_p*(q_ref - q) instead -- which is
        # why q_ref is logged next to it.
        try:
            tau = np.asarray(robot.get_measured_joint_efforts())[arm]
        except Exception:
            tau = np.zeros(7)
        arc += float(np.linalg.norm(q - q_prev))
        q_prev = q
        steps.append([t] + list(q) + list(q_ref) + list(tau) + [arc])
        t += args.dt

    with open(os.path.join(args.out_dir, "steps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"q{i}" for i in range(7)]
                   + [f"qref{i}" for i in range(7)]
                   + [f"tau{i}" for i in range(7)] + ["arc"])
        w.writerows(steps)

    with open(os.path.join(args.out_dir, "contacts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "robot_prim", "other_prim", "cls",
                    "px", "py", "pz", "nx", "ny", "nz", "Jn", "Jt", "sep"])
        w.writerows(truth.rows)

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"dt": args.dt, "vel_scale": args.vel_scale, "path": args.path,
                   "usd": args.usd, "arm_joint_names": ARM,
                   "max_force": args.max_force, "stiffness": args.stiffness,
                   "damping": args.damping, "trunk_hit": truth.trunk_hit,
                   "n_steps": len(steps), "n_contact_rows": len(truth.rows)}, f,
                  indent=2)

    print(f"[Done] {len(steps)} steps, {len(truth.rows)} contact rows -> {args.out_dir}")
    if truth.trunk_hit:
        print("[WARN] the arm touched the TRUNK -- that is a hard failure, not data")
    simulation_app.close()


if __name__ == "__main__":
    main()
