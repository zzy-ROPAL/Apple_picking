"""物理标定刚度: 在 tamed USD 副本上, 把统一 STIFF=150/DAMP=30/MAXF=1.5 改为
逐关节 Euler-Bernoulli 值: K=E·πr⁴/4/l, D=0.1K, maxF=断裂弯矩 σ·πr³/4。
r/l 取该关节 child 段(body1)的实测包围盒几何。其余配置(柔顺接触/果柄弹簧/
关重力/质量/armature)全部继承 tamed, 不动。"""
import shutil, math
from pxr import Usd, UsdPhysics, UsdGeom, Sdf

SRC = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_tamed.usd"
DST = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_calibrated.usd"
E_WOOD = 5e9    # Pa 活枝杨氏模量 (用户指定 5 GPa; 参考 Cannell&Morgan 1987 活枝 0.7-4.6 GPa, 干材清材 ~7GPa)
SIGMA  = 40e6   # Pa 活枝断裂应力 (绿材, 与 E=3GPa 一致下调)
CD     = 0.1    # D = CD·K (OrchardBench 阻尼惯例)

shutil.copy(SRC, DST)
stage = Usd.Stage.Open(DST)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])

def seg_geom(prim):
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    s = rng.GetSize(); dims = sorted([s[0], s[1], s[2]])
    r = 0.25*(dims[0]+dims[1])          # 两短维 ≈ 直径
    L = max(1e-3, dims[2])
    return r, L

rows = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "/World/Tree6/branch" not in path or prim.GetTypeName() != "PhysicsJoint":
        continue
    drives = [ax for ax in ("rotX","rotY") if UsdPhysics.DriveAPI.Get(prim, ax)]
    if not drives:
        continue
    j = UsdPhysics.Joint(prim)
    tgts = j.GetBody1Rel().GetTargets()
    child = stage.GetPrimAtPath(tgts[0]) if tgts else None
    if child is None or not child.IsValid():
        print("!! no body1 for", path); continue
    r, L = seg_geom(child)
    I = math.pi*r**4/4
    K  = E_WOOD*I/L
    D  = CD*K
    MF = SIGMA*math.pi*r**3/4
    for ax in drives:
        d = UsdPhysics.DriveAPI.Get(prim, ax)
        d.CreateStiffnessAttr().Set(float(K))
        d.CreateDampingAttr().Set(float(D))
        d.CreateMaxForceAttr().Set(float(MF))
    rows.append((path, r*1000, L*100, K, MF))

rows.sort(key=lambda x: x[3])
print(f"[calib] patched {len(rows)} joints ({sum(1 for _ in rows)*2} drives)")
print(f"[calib] K range: {rows[0][3]:.0f} .. {rows[-1][3]:.0f} N·m/rad")
print(f"[calib] softest : {rows[0][0]}  r={rows[0][1]:.1f}mm l={rows[0][2]:.1f}cm  maxF={rows[0][4]:.1f} N·m")
print(f"[calib] stiffest: {rows[-1][0]}  r={rows[-1][1]:.1f}mm l={rows[-1][2]:.1f}cm  maxF={rows[-1][4]:.1f} N·m")
import statistics
print(f"[calib] K median: {statistics.median(x[3] for x in rows):.0f} N·m/rad")
stage.GetRootLayer().Save()
print("[calib] wrote", DST)
