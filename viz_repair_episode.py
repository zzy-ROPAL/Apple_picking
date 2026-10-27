#!/usr/bin/env python3
"""
viz_repair_episode.py — play back one sim-to-real repair episode in pybullet.

The episode has three phases, exported by `benchmark_grasp_rrt --harden-dump`:

  1. APPROACH  follow the nominal path until the stall waypoint      (white trail)
  2. RETREAT   reverse along the path already executed, until the    (yellow trail)
               pose is valid under the edited scene
  3. REPAIR    the replanned path to the apple                        (cyan trail)

The inserted hard obstacle (subclass "hard_branch") is drawn in RED. That is the
thing the robot hit and that the planner did not know about; everything the
repaired path does differently is a consequence of it.

Usage
  python3 viz_repair_episode.py --dir <results_dir> --run 7            # GUI
  python3 viz_repair_episode.py --dir <results_dir> --run 7 --render out.png
  python3 viz_repair_episode.py --dir <results_dir> --list

NOTE the robot base sits at z = 0.4 m (panda_pinocchio.h base_offset_). Forgetting
that puts every FK 40 cm off.
"""

import argparse
import glob
import json
import math
import os
import sys

BASE_Z = 0.40
URDF_DEFAULT = ("/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.venv/lib/"
                "python3.12/site-packages/pybullet_data/franka_panda/panda.urdf")

PHASE_COLOR = {
    "approach": (0.95, 0.95, 0.95),
    "retreat":  (1.00, 0.80, 0.10),
    "repair":   (0.10, 0.85, 0.95),
}


def load_csv(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split(",")])
    return rows


def load_scene(path):
    with open(path) as f:
        d = json.load(f)
    return d["apple"], d["cylinders"]


def quat_from_axis(axis):
    """Rotation taking +z onto `axis` (pybullet cylinders are z-aligned)."""
    L = math.sqrt(sum(a * a for a in axis))
    if L < 1e-9:
        return [0, 0, 0, 1], 0.0
    z = [a / L for a in axis]
    ref = (0.0, 0.0, 1.0)
    v = (ref[1] * z[2] - ref[2] * z[1],
         ref[2] * z[0] - ref[0] * z[2],
         ref[0] * z[1] - ref[1] * z[0])
    s = math.sqrt(sum(c * c for c in v))
    if s < 1e-8:
        return ([0, 0, 0, 1] if z[2] > 0 else [1, 0, 0, 0]), L
    ang = math.atan2(s, ref[0] * z[0] + ref[1] * z[1] + ref[2] * z[2])
    v = [c / s for c in v]
    sa = math.sin(ang / 2)
    return [v[0] * sa, v[1] * sa, v[2] * sa, math.cos(ang / 2)], L


def build_scene(p, cid, apple, cylinders):
    """Soft = green, trunk/rigid = brown, INSERTED obstacle = red."""
    colors = {
        "soft":        [0.25, 0.65, 0.20, 0.85],
        "trunk":       [0.55, 0.27, 0.07, 1.00],
        "hard_branch": [0.95, 0.10, 0.10, 1.00],
    }
    n_inserted = 0
    for c in cylinders:
        sub = c.get("subclass") or ("soft" if c["class"] == "soft" else "trunk")
        rgba = colors.get(sub, colors["trunk"])
        if sub == "hard_branch":
            n_inserted += 1
        p0, p1, R = c["p0"], c["p1"], float(c["radius"])
        mid = [(a + b) / 2 for a, b in zip(p0, p1)]
        quat, L = quat_from_axis([b - a for a, b in zip(p0, p1)])
        if L < 1e-6:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=R, rgbaColor=rgba,
                                      physicsClientId=cid)
            p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=mid,
                              physicsClientId=cid)
        else:
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=R, length=L,
                                      rgbaColor=rgba, physicsClientId=cid)
            p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=mid,
                              baseOrientation=quat, physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(apple["radius"]),
                              rgbaColor=[0.85, 0.10, 0.10, 1], physicsClientId=cid)
    p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=apple["center"],
                      physicsClientId=cid)
    return n_inserted


def phase_of(i, ph):
    if i <= ph["approach_end"]:
        return "approach"
    if i <= ph["retreat_end"]:
        return "retreat"
    return "repair"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="benchmark results directory")
    ap.add_argument("--run", type=int, help="run index (as in repair_XX_episode.csv)")
    ap.add_argument("--list", action="store_true", help="list available episodes")
    ap.add_argument("--render", metavar="PNG", help="offscreen render instead of GUI")
    ap.add_argument("--gif", metavar="GIF",
                    help="offscreen animation instead of GUI. Uses TinyRenderer, so it "
                         "needs no OpenGL context and works on a headless or locked "
                         "display (where the GUI cannot get a GLX context at all).")
    ap.add_argument("--gif-every", type=int, default=2,
                    help="capture one frame every N waypoints (default 2)")
    ap.add_argument("--gif-size", type=int, nargs=2, default=[800, 600],
                    metavar=("W", "H"))
    ap.add_argument("--gif-ms", type=int, default=90, help="ms per frame")
    ap.add_argument("--frames", metavar="DIR",
                    help="write individual PNG frames into DIR instead of (or as well "
                         "as) a GIF. Same capture path as --gif.")
    ap.add_argument("--hw", action="store_true",
                    help="capture through Bullet's OpenGL renderer instead of the "
                         "software TinyRenderer. Needs a working GL context on $DISPLAY "
                         "and briefly opens the Bullet window; better shading/shadows.")
    ap.add_argument("--interp", type=int, default=1,
                    help="linear sub-steps inserted between consecutive waypoints. "
                         "The path is a SHORTCUT path, so neighbouring waypoints can be "
                         "far apart in joint space and raw playback looks jumpy. 4-6 "
                         "gives smooth motion.")
    ap.add_argument("--original", action="store_true",
                    help="render the NOMINAL path (grasp_path_XX.csv) against the "
                         "unedited scene_layout.json — i.e. the plan as it was before "
                         "any contact, with no inserted obstacle. Use it as the "
                         "before/after counterpart to the repair episode.")
    ap.add_argument("--spin", type=float, default=0.0,
                    help="degrees of camera yaw to sweep across the episode")
    ap.add_argument("--urdf", default=os.environ.get("PANDA_URDF", URDF_DEFAULT))
    ap.add_argument("--hold", type=float, default=0.06, help="seconds per waypoint")
    ap.add_argument("--loop", action="store_true", help="repeat until Ctrl-C")
    args = ap.parse_args()

    eps = sorted(glob.glob(os.path.join(args.dir, "repair_*_episode.csv")))
    if args.list or args.run is None:
        if not eps:
            print(f"no episodes in {args.dir} — rerun the benchmark with --harden-dump")
            return 1
        print(f"{len(eps)} episode(s) in {args.dir}:")
        for e in eps:
            idx = os.path.basename(e).split("_")[1]
            ph = json.load(open(e.replace("_episode.csv", "_phases.json")))
            print(f"  run {idx}: {ph['total']:4d} wp | contact wp {ph['contact_wp']}"
                  f" pen {ph['contact_pen_mm']:.1f} mm | retreat {ph['retreat_steps']}"
                  f" steps | winner={ph['winner']} reused={ph['reused_wp']} wp")
        if args.run is None:
            print("\npick one with --run <idx>")
        return 0

    stem = os.path.join(args.dir, f"repair_{args.run:02d}")
    if args.original:
        # The plan as it stood before any contact: nominal path, unedited scene.
        wps = load_csv(os.path.join(args.dir, f"grasp_path_{args.run:02d}.csv"))
        apple, cylinders = load_scene(os.path.join(args.dir, "scene_layout.json"))
        ph = {"approach_end": len(wps) - 1, "retreat_end": len(wps) - 1,
              "total": len(wps), "contact_wp": -1, "retreat_wp": -1,
              "retreat_steps": 0, "contact_point": None, "contact_pen_mm": 0.0,
              "hard_idx": -1, "winner": "n/a", "reused_wp": 0}
    else:
        wps = load_csv(stem + "_episode.csv")
        apple, cylinders = load_scene(stem + "_scene.json")
        ph = json.load(open(stem + "_phases.json"))

    # Sub-sample for smooth playback. Phase boundaries scale with it.
    if args.interp > 1:
        n = args.interp
        dense = []
        for a, b in zip(wps[:-1], wps[1:]):
            for k in range(n):
                t = k / n
                dense.append([x + (y - x) * t for x, y in zip(a, b)])
        dense.append(wps[-1])
        ph = dict(ph)
        ph["approach_end"] = ph["approach_end"] * n
        ph["retreat_end"] = ph["retreat_end"] * n
        wps = dense
        ph["total"] = len(wps)

    capture = bool(args.render or args.gif or args.frames)
    headless = capture
    # --hw captures from the real OpenGL renderer, which needs a GL context, which
    # means a GUI connection even when we are only grabbing frames.
    use_gui = (not capture) or args.hw

    import pybullet as p
    import pybullet_data
    cid = p.connect(p.GUI if use_gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setGravity(0, 0, 0, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)

    n_ins = build_scene(p, cid, apple, cylinders)
    rid = p.loadURDF(args.urdf, [0, 0, BASE_Z], useFixedBase=True,
                     physicsClientId=cid)
    arm = [j for j in range(p.getNumJoints(rid, physicsClientId=cid))
           if p.getJointInfo(rid, j, physicsClientId=cid)[2] == p.JOINT_REVOLUTE][:7]
    tcp = max(arm) + 1

    # Mark the contact that triggered the repair (absent in --original mode).
    cp = ph.get("contact_point")
    if cp is not None:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.012,
                                  rgbaColor=[1, 0.3, 0, 1], physicsClientId=cid)
        p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=cp,
                          physicsClientId=cid)

    print(f"run {args.run}: {len(wps)} waypoints | "
          f"approach 0..{ph['approach_end']}, "
          f"retreat ..{ph['retreat_end']} ({ph['retreat_steps']} steps), "
          f"repair ..{len(wps)-1}")
    if args.original:
        print(f"  NOMINAL plan, unedited scene | {n_ins} inserted obstacle(s)")
    else:
        print(f"  contact pen {ph['contact_pen_mm']:.1f} mm | "
              f"{n_ins} inserted obstacle(s) drawn RED | winner={ph['winner']}")

    if not headless:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
        p.resetDebugVisualizerCamera(1.6, 55, -20, apple["center"],
                                     physicsClientId=cid)

    def set_q(q):
        for j, v in zip(arm, q):
            p.resetJointState(rid, j, v, physicsClientId=cid)

    def play(grab=None):
        prev_tcp, prev_phase, label = None, None, None
        for i, q in enumerate(wps):
            set_q(q)
            pos = p.getLinkState(rid, tcp, physicsClientId=cid)[0]
            phase = phase_of(i, ph)
            if prev_tcp is not None and phase == prev_phase:
                if headless:
                    # Debug lines are GUI-only and never appear in getCameraImage,
                    # so a still needs real geometry for the trail.
                    if i % 2 == 0:
                        sph = p.createVisualShape(
                            p.GEOM_SPHERE, radius=0.009,
                            rgbaColor=list(PHASE_COLOR[phase]) + [1.0],
                            physicsClientId=cid)
                        p.createMultiBody(0, baseVisualShapeIndex=sph,
                                          basePosition=list(pos),
                                          physicsClientId=cid)
                else:
                    p.addUserDebugLine(prev_tcp, pos, PHASE_COLOR[phase], 3.0,
                                       physicsClientId=cid)
            if phase != prev_phase and not headless:
                if label is not None:
                    p.removeUserDebugItem(label, physicsClientId=cid)
                txt = {"approach": "1/3  APPROACH - following the planned path",
                       "retreat":  "2/3  STALL -> RETREAT along the executed path",
                       "repair":   "3/3  REPAIR - replanned around the red obstacle"}[phase]
                label = p.addUserDebugText(txt, [0, 0, 1.75], PHASE_COLOR[phase],
                                           textSize=1.4, physicsClientId=cid)
                print(f"  wp {i:4d}  -> {phase}")
            prev_tcp, prev_phase = pos, phase
            if grab is not None and i % args.gif_every == 0:
                grab(i, phase)
            if not headless:
                time.sleep(args.hold * (3.0 if phase == "retreat" else 1.0))

    if args.gif or args.frames:
        from PIL import Image, ImageDraw
        W, H = args.gif_size
        if args.frames:
            os.makedirs(args.frames, exist_ok=True)
        renderer = (p.ER_BULLET_HARDWARE_OPENGL if args.hw else p.ER_TINY_RENDERER)
        frames = []
        n_wp = len(wps)
        band = {"approach": (240, 240, 240), "retreat": (255, 205, 25),
                "repair": (25, 215, 240)}
        if args.original:
            text = {"approach": "NOMINAL PLAN   no contact, no inserted obstacle",
                    "retreat": "", "repair": ""}
        else:
            text = {"approach": "1/3  APPROACH   following the planned path",
                    "retreat":  "2/3  RETREAT    stalled - backing out along the executed path",
                    "repair":   "3/3  REPAIR     replanned around the red obstacle"}

        def grab(i, phase):
            yaw = 55.0 + args.spin * (i / max(1, n_wp - 1))
            view = p.computeViewMatrixFromYawPitchRoll(
                apple["center"], 1.15, yaw, -18, 0, 2, physicsClientId=cid)
            proj = p.computeProjectionMatrixFOV(60, W / H, 0.1, 10,
                                                physicsClientId=cid)
            img = p.getCameraImage(W, H, view, proj, renderer=renderer,
                                   physicsClientId=cid)
            im = Image.frombytes("RGBA", (W, H), bytes(img[2])).convert("RGB")
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, W, 22], fill=band[phase])
            d.text((8, 6), f"{text[phase]}    wp {i}/{n_wp - 1}", fill=(20, 20, 20))
            if args.frames:
                im.save(os.path.join(args.frames,
                                     f"frame_{len(frames):04d}_{phase}.png"))
            frames.append(im)

        play(grab=grab)
        if args.frames:
            print(f"wrote {len(frames)} PNG frames to {args.frames}/  ({W}x{H}, "
                  f"{'OpenGL' if args.hw else 'TinyRenderer'})")
        if args.gif:
            for _ in range(8):                   # hold on the final pose
                frames.append(frames[-1])
            frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                           duration=args.gif_ms, loop=0, optimize=True)
            print(f"wrote {args.gif}  ({len(frames)} frames, {W}x{H})")
    elif args.render:
        play()
        set_q(wps[-1])
        w, h = 1280, 960
        view = p.computeViewMatrixFromYawPitchRoll(apple["center"], 1.15, 55, -18, 0, 2)
        proj = p.computeProjectionMatrixFOV(60, w / h, 0.1, 10)
        img = p.getCameraImage(w, h, view, proj,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL,
                               physicsClientId=cid)
        try:
            from PIL import Image
            Image.frombytes("RGBA", (w, h), bytes(img[2])).convert("RGB").save(args.render)
        except ImportError:
            import struct
            with open(args.render.replace(".png", ".ppm"), "wb") as f:
                f.write(b"P6\n%d %d\n255\n" % (w, h))
                px = bytes(img[2])
                f.write(bytes(b for k in range(0, len(px), 4) for b in px[k:k+3]))
            print("PIL missing — wrote .ppm instead")
        print(f"wrote {args.render}")
    else:
        import time
        while True:
            play()
            if not args.loop:
                break
            time.sleep(1.0)
            p.removeAllUserDebugItems(physicsClientId=cid)
        print("playback done — close the window or Ctrl-C to exit")
        while p.isConnected(physicsClientId=cid):
            time.sleep(0.2)
    return 0


if __name__ == "__main__":
    import time
    sys.exit(main())
