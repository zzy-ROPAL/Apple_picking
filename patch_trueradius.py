"""真截面半径重标定: bbox 半径把弯曲段的"扫掠宽度"误当截面 (r⁴ → K 高估 16-80x)。
改用 C++ BRANCH_RAD 地面真值: 每个关节的 child 段中心 → 该枝 C++ 折线最近弧长位置
→ 插值截面半径 → K=E·πr⁴/4/l, D=0.1K, maxF=σ·πr³/4。"""
import re, math, shutil
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

SRC = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_calibrated_ab12.usd"
DST = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_calibrated_ab12_truer.usd"
CPP = "/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.claude/worktrees/experiment-scene-visualization-e0aaf2/benchmark_grasp_rrt.cpp"
E_WOOD, SIGMA = 5e9, 40e6

# ---- 解析 C++ 节点/半径数组 ----
src = open(CPP).read()
mn = re.search(r"TREE6_BRANCH_NODES\[23\]\[6\] = \{(.*?)\n\};", src, re.S).group(1)
rows = re.findall(r"\{\{(.*?)\}\}", mn)
NODES = []
for row in rows:
    vals = [float(x.rstrip('f')) for x in re.findall(r"-?[\d.]+f", row)]
    NODES.append(np.array(vals).reshape(6,3))
mr = re.search(r"TREE6_BRANCH_RAD\[23\]\[6\] = \{(.*?)\n\};", src, re.S).group(1)
rvals = [float(x.rstrip('f')) for x in re.findall(r"-?[\d.]+f", mr)]
RADS = np.array(rvals).reshape(23,6)
assert len(NODES)==23

# ---- 施加与实验相同的树变换: yaw -90 @ 树根, dist 0.5, z -0.35 ----
ROOT = np.array([0.7526,-0.0624,0.0011])
cy, sy = math.cos(math.radians(-90)), math.sin(math.radians(-90))
sh = (0.5-1.0)*ROOT
def xform(p):
    d = p - ROOT
    q = np.array([ROOT[0]+cy*d[0]-sy*d[1]+sh[0], ROOT[1]+sy*d[0]+cy*d[1]+sh[1], p[2]-0.35])
    return q
TN = [np.array([xform(p) for p in b]) for b in NODES]

def radius_at(bi, pt):
    """pt 到 C++ 枝 bi 折线最近点的插值截面半径"""
    best = (1e9, 0.0, 0)
    nodes = TN[bi]
    for i in range(5):
        a, b = nodes[i], nodes[i+1]
        v = b-a; t = float(np.clip(np.dot(pt-a, v)/max(1e-12, np.dot(v,v)), 0, 1))
        d = float(np.linalg.norm(pt-(a+t*v)))
        if d < best[0]: best = (d, t, i)
    d, t, i = best
    return (1-t)*RADS[bi][i] + t*RADS[bi][i+1], d

shutil.copy(SRC, DST)
stage = Usd.Stage.Open(DST)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default","render","proxy","guide"])
out = []
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if "/World/Tree6/branch" not in p or prim.GetTypeName() != "PhysicsJoint":
        continue
    drives = [ax for ax in ("rotX","rotY") if UsdPhysics.DriveAPI.Get(prim, ax)]
    if not drives: continue
    tgts = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
    child = stage.GetPrimAtPath(tgts[0])
    rng = cache.ComputeWorldBound(child).ComputeAlignedRange()
    c = np.array((rng.GetMin()+rng.GetMax()))/2.0
    dims = sorted(rng.GetSize()); L = max(0.02, dims[2])   # 段长仍用 bbox 长维
    bi = int(child.GetParent().GetName().replace("branch","")) - 1   # USD 1-based -> C++ 0-based
    r, dist = radius_at(bi, c)
    K  = E_WOOD*math.pi*r**4/4/L
    MF = SIGMA*math.pi*r**3/4
    for ax in drives:
        d = UsdPhysics.DriveAPI.Get(prim, ax)
        d.CreateStiffnessAttr().Set(float(K)); d.CreateDampingAttr().Set(float(0.1*K))
        d.CreateMaxForceAttr().Set(float(MF))
    out.append((K, MF, r, p, dist))
stage.GetRootLayer().Save()
out.sort()
import statistics
Ks = [o[0] for o in out]
print(f"[truer] {len(out)} joints  K: min {min(Ks):.1f}  median {statistics.median(Ks):.0f}  max {max(Ks):.0f} N·m/rad")
print(f"[truer] softest : K={out[0][0]:.1f} r={out[0][2]*1000:.1f}mm {out[0][3]}")
print(f"[truer] stiffest: K={out[-1][0]:.0f} r={out[-1][2]*1000:.1f}mm {out[-1][3]}")
bad = [o for o in out if o[4] > 0.10]
print(f"[truer] polyline match >10cm off: {len(bad)} (应为 0)")
print("[truer] wrote", DST)
