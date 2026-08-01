#!/usr/bin/env python3
# viz_cut_scene.py — 软枝穿透场景可视化 / 排名
#
# 用 scene_layout.json(规划器导出的精确几何)重建场景,对 grasp_path_XX.csv
# 逐路点做与 C++ 穿透闸门同款的探针检测(9 探针, allow(t)=15mm+60mm·t,
# 苹果 12cm 豁免),两种模式:
#
#   分析:  python3 viz_cut_scene.py --rank <结果目录>
#          对目录下全部 grasp_path_*.csv 输出每条路径的最深穿透 / 最险比值
#   回放:  python3 viz_cut_scene.py --show <path.csv> --scene <scene_layout.json> \
#              [--title 文本] [--snapshot out.png]
#          pybullet GUI 循环回放, 接触的软枝染橙, 全程最深处画标记与文字

import json
import math
import glob
import os
import sys
import numpy as np

# 与 C++ 穿透闸门一致的探针 (link 名, link 系偏移, 体半径)
PROBES = [
    ("panda_link3", [0, 0,     0.15], 0.07),
    ("panda_link4", [0, 0,     0.05], 0.07),
    ("panda_link5", [0, 0,     0.15], 0.06),
    ("panda_link5", [0, 0,     0.30], 0.06),
    ("panda_link6", [0, 0,     0.00], 0.06),
    ("panda_link7", [0, 0,     0.05], 0.06),
    ("panda_hand",  [0, 0,     0.08], 0.06),
    ("panda_hand",  [0, 0.04,  0.05], 0.06),
    ("panda_hand",  [0, -0.04, 0.05], 0.06),
]
ALLOW_ROOT = 0.015   # allow(t=0), m
ALLOW_TIP  = 0.075   # allow(t=1), m
EXEMPT_R   = 0.12    # 苹果豁免半径, m


def load_scene(path):
    with open(path) as f:
        d = json.load(f)
    apple = np.array(d["apple"]["center"], dtype=float)
    soft, hard = [], []
    for c in d["cylinders"]:
        p0 = np.array(c["p0"], dtype=float)
        p1 = np.array(c["p1"], dtype=float)
        # 多段枝 (tree6 等) 每段带自己的弧长子区间 [t0,t1]; 旧 JSON 无此
        # 字段时退化为单段 (0,1)
        t0, t1 = float(c.get("t0", 0.0)), float(c.get("t1", 1.0))
        (soft if c["class"] == "soft" else hard).append(
            (p0, p1, float(c["radius"]), t0, t1))
    return apple, soft, hard


def seg_point_dist(p, p0, p1):
    seg = p1 - p0
    sq = float(seg.dot(seg))
    u = 0.0 if sq < 1e-9 else float(np.clip((p - p0).dot(seg) / sq, 0.0, 1.0))
    return float(np.linalg.norm(p - (p0 + u * seg))), u


class PandaFK:
    """pybullet DIRECT 实例做 FK, 与 GUI 回放共用同一 URDF frame 定义。"""

    def __init__(self, client, robot_id):
        import pybullet as p
        self.p, self.cid, self.rid = p, client, robot_id
        self.link_idx = {}
        for j in range(p.getNumJoints(robot_id, physicsClientId=client)):
            info = p.getJointInfo(robot_id, j, physicsClientId=client)
            self.link_idx[info[12].decode()] = j
        self.arm_joints = [self.link_idx[f"panda_link{i}"] for i in range(1, 8)]

    def set_q(self, q):
        for jid, qi in zip(self.arm_joints, q):
            self.p.resetJointState(self.rid, jid, float(qi), physicsClientId=self.cid)
        for name in ("panda_leftfinger", "panda_rightfinger"):
            if name in self.link_idx:
                self.p.resetJointState(self.rid, self.link_idx[name], 0.04,
                                       physicsClientId=self.cid)

    def probe_positions(self):
        out = []
        for name, off, r in PROBES:
            st = self.p.getLinkState(self.rid, self.link_idx[name],
                                     computeForwardKinematics=1,
                                     physicsClientId=self.cid)
            pos, orn = np.array(st[4]), st[5]
            R = np.array(self.p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            out.append((pos + R @ np.array(off), r))
        return out


def analyze_path(fk, waypoints, apple, soft):
    """返回 (max_pen_m, max_ratio, argmax_wp, argmax_cyl, per_wp_contacts)。
    per_wp_contacts[i] = {软枝下标: 该路点上的最大 pen/allow 比值}。"""
    max_pen, max_ratio, arg_wp, arg_cyl = 0.0, 0.0, -1, -1
    contacts = []
    for i, q in enumerate(waypoints):
        fk.set_q(q)
        touched = {}
        for pp, pr in fk.probe_positions():
            if np.linalg.norm(pp - apple) < EXEMPT_R:
                continue
            for ci, (p0, p1, R, t0, t1) in enumerate(soft):
                dist, u = seg_point_dist(pp, p0, p1)
                pen = (R + pr) - dist
                if pen <= 0:
                    continue
                t_arc = t0 + u * (t1 - t0)
                allow = ALLOW_ROOT + (ALLOW_TIP - ALLOW_ROOT) * t_arc
                ratio = pen / allow
                touched[ci] = max(touched.get(ci, 0.0), ratio)
                if pen > max_pen:
                    max_pen = pen
                if ratio > max_ratio:
                    max_ratio, arg_wp, arg_cyl = ratio, i, ci
        contacts.append(touched)
    return max_pen, max_ratio, arg_wp, arg_cyl, contacts


def make_fk(gui, urdf_hint=None):
    import pybullet as p
    import pybullet_data
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdf = urdf_hint or "franka_panda/panda.urdf"
    # 与 C++ 一致: PandaPinocchioRobot::base_offset_ = (0, 0, 0.4)
    rid = p.loadURDF(urdf, basePosition=[0, 0, 0.4], useFixedBase=1,
                     physicsClientId=cid)
    return p, cid, rid


def cmd_rank(result_dir):
    apple, soft, hard = load_scene(os.path.join(result_dir, "scene_layout.json"))
    p, cid, rid = make_fk(gui=False)
    fk = PandaFK(cid, rid)
    rows = []
    for f in sorted(glob.glob(os.path.join(result_dir, "grasp_path_*.csv"))):
        wps = np.loadtxt(f, delimiter=",")
        max_pen, max_ratio, wp, ci, _ = analyze_path(fk, wps, apple, soft)
        run = int(os.path.basename(f)[len("grasp_path_"):-4])
        rows.append((run, max_pen, max_ratio, wp, ci, len(wps)))
        print(f"run{run:02d}: max_pen={max_pen*1000:6.1f}mm  "
              f"max_ratio={max_ratio:5.2f}  @wp{wp}/{len(wps)}  branch={ci}")
    rows.sort(key=lambda r: r[2], reverse=True)
    ratios = sorted(r[2] for r in rows)
    med = ratios[len(ratios) // 2]
    avg_row = min(rows, key=lambda r: abs(r[2] - med))
    print(f"\nWORST : run{rows[0][0]:02d} (ratio {rows[0][2]:.2f}, "
          f"pen {rows[0][1]*1000:.1f}mm)")
    print(f"MEDIAN: run{avg_row[0]:02d} (ratio {avg_row[2]:.2f}, "
          f"pen {avg_row[1]*1000:.1f}mm)")


def build_scene_bodies(p, cid, apple, soft, hard):
    """静态可视场景: 硬=棕, 软=绿, 苹果=红。返回软枝 body id 列表。"""
    p.loadURDF("plane.urdf", physicsClientId=cid)
    soft_ids = []
    for cls, cyls in (("hard", hard), ("soft", soft)):
        rgba = [0.55, 0.27, 0.07, 1] if cls == "hard" else [0.25, 0.65, 0.2, 0.9]
        for p0, p1, R, _t0, _t1 in cyls:
            mid = (p0 + p1) / 2
            axis = p1 - p0
            L = float(np.linalg.norm(axis))
            if L < 1e-6:      # 球体 (p0==p1)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=R, rgbaColor=rgba,
                                          physicsClientId=cid)
                bid = p.createMultiBody(0, baseVisualShapeIndex=vis,
                                        basePosition=mid.tolist(),
                                        physicsClientId=cid)
            else:
                z = axis / L
                ref = np.array([0, 0, 1.0])
                v = np.cross(ref, z)
                s = np.linalg.norm(v)
                if s < 1e-8:
                    quat = [0, 0, 0, 1]
                else:
                    ang = math.atan2(s, float(ref.dot(z)))
                    v = v / s
                    quat = [v[0] * math.sin(ang / 2), v[1] * math.sin(ang / 2),
                            v[2] * math.sin(ang / 2), math.cos(ang / 2)]
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=R, length=L,
                                          rgbaColor=rgba, physicsClientId=cid)
                bid = p.createMultiBody(0, baseVisualShapeIndex=vis,
                                        basePosition=mid.tolist(),
                                        baseOrientation=quat, physicsClientId=cid)
            if cls == "soft":
                soft_ids.append(bid)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04,
                              rgbaColor=[0.85, 0.1, 0.1, 1], physicsClientId=cid)
    p.createMultiBody(0, baseVisualShapeIndex=vis,
                      basePosition=apple.tolist(), physicsClientId=cid)
    return soft_ids


def cmd_show(path_csv, scene_json, title, snapshot):
    apple, soft, hard = load_scene(scene_json)
    p, cid, rid = make_fk(gui=True)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=48,
                                 cameraPitch=-22,
                                 cameraTargetPosition=[0.45, 0, 0.75],
                                 physicsClientId=cid)
    soft_ids = build_scene_bodies(p, cid, apple, soft, hard)
    fk = PandaFK(cid, rid)
    wps = np.loadtxt(path_csv, delimiter=",")

    max_pen, max_ratio, arg_wp, arg_cyl, contacts = \
        analyze_path(fk, wps, apple, soft)

    # TCP 轨迹线 + 最深接触标记
    tcp = []
    for q in wps:
        fk.set_q(q)
        st = p.getLinkState(rid, fk.link_idx["panda_hand"],
                            computeForwardKinematics=1, physicsClientId=cid)
        tcp.append(st[4])
    for a, b in zip(tcp[:-1], tcp[1:]):
        p.addUserDebugLine(a, b, [0.1, 0.1, 0.9], 2, physicsClientId=cid)
    hdr = (f"{title}   max_pen={max_pen*1000:.1f}mm   "
           f"worst pen/allow={max_ratio:.2f}  (>1 = broken)")
    p.addUserDebugText(hdr, [0.0, 0.0, 1.55], [0, 0, 0], 1.2,
                       physicsClientId=cid)
    if arg_wp >= 0:
        fk.set_q(wps[arg_wp])
        st = p.getLinkState(rid, fk.link_idx["panda_hand"],
                            computeForwardKinematics=1, physicsClientId=cid)
        p.addUserDebugText("deepest", st[4], [0.9, 0.4, 0], 1.0,
                           physicsClientId=cid)

    # 按预算占用染色: 绿=无接触, 黄=轻推(<50% 预算), 橙=深推但许可
    # (50-100%), 红=超预算(会折断; 门控 run 中不应出现)
    green = [0.25, 0.65, 0.2, 0.9]
    def bucket(r):
        return 0 if r < 0.5 else (1 if r < 1.0 else 2)
    bucket_colors = [[0.95, 0.85, 0.15, 1.0],   # 黄
                     [1.0, 0.55, 0.05, 1.0],    # 橙
                     [0.9, 0.08, 0.08, 1.0]]    # 红
    import time
    loop = 0
    prev_bucket = {}
    while p.isConnected(cid):
        for i, q in enumerate(wps):
            if not p.isConnected(cid):
                break
            fk.set_q(q)
            # 只更新染色桶变化的枝 (changeVisualShape 较贵)
            cur = {ci: bucket(r) for ci, r in contacts[i].items()}
            for ci, b in cur.items():
                if prev_bucket.get(ci) != b:
                    p.changeVisualShape(soft_ids[ci], -1,
                                        rgbaColor=bucket_colors[b],
                                        physicsClientId=cid)
            for ci in prev_bucket:
                if ci not in cur:
                    p.changeVisualShape(soft_ids[ci], -1, rgbaColor=green,
                                        physicsClientId=cid)
            prev_bucket = cur
            # 全程最深路点上停留放大
            time.sleep(0.55 if i == arg_wp else 0.045)
            if snapshot and loop == 0 and i == arg_wp:
                w, h, rgb, _, _ = p.getCameraImage(1280, 960,
                                                   physicsClientId=cid)
                img = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
                try:
                    from PIL import Image
                    Image.fromarray(img).save(snapshot)
                    print(f"[viz] snapshot -> {snapshot}")
                except ImportError:
                    np.save(snapshot + ".npy", img)
        loop += 1
        time.sleep(0.8)


def cmd_render(path_csv, scene_json, title, out_png):
    """离屏渲染 (DIRECT + tiny renderer): 最深接触路点的静帧。"""
    apple, soft, hard = load_scene(scene_json)
    p, cid, rid = make_fk(gui=False)
    build_scene_bodies(p, cid, apple, soft, hard)
    fk = PandaFK(cid, rid)
    wps = np.loadtxt(path_csv, delimiter=",")
    max_pen, max_ratio, arg_wp, arg_cyl, contacts = \
        analyze_path(fk, wps, apple, soft)
    fk.set_q(wps[max(arg_wp, 0)])
    view = p.computeViewMatrixFromYawPitchRoll([0.45, 0, 0.85], 1.25, 48, -20, 0, 2)
    proj = p.computeProjectionMatrixFOV(58, 4/3, 0.05, 4.0)
    w, h, rgb, _, _ = p.getCameraImage(1280, 960, view, proj,
                                       renderer=p.ER_TINY_RENDERER,
                                       physicsClientId=cid)
    img = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
    from PIL import Image
    Image.fromarray(img).save(out_png)
    print(f"{title}: max_pen={max_pen*1000:.1f}mm ratio={max_ratio:.2f} "
          f"@wp{arg_wp}/{len(wps)} -> {out_png}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--rank" in args:
        cmd_rank(args[args.index("--rank") + 1])
    elif "--render" in args:
        path_csv = args[args.index("--render") + 1]
        scene = args[args.index("--scene") + 1]
        title = args[args.index("--title") + 1] if "--title" in args else ""
        out = args[args.index("--out") + 1]
        cmd_render(path_csv, scene, title, out)
    elif "--show" in args:
        path_csv = args[args.index("--show") + 1]
        scene = args[args.index("--scene") + 1]
        title = args[args.index("--title") + 1] if "--title" in args else ""
        snap = args[args.index("--snapshot") + 1] if "--snapshot" in args else None
        cmd_show(path_csv, scene, title, snap)
    else:
        print(__doc__)
        sys.exit(1)
