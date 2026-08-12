"""ab12 场景对齐: calibrated USD 副本上 (1) Tree6 下移 0.35m (= --base-raise 0.35),
(2) AppleTarget 移到 bullet 苹果位 (0.3234,0.113104,0.757427),
(3) 果柄弹簧重新挂到 branch13 (bullet idx12, USD 1-based) 最近段."""
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import shutil

SRC = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_calibrated.usd"
DST = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_calibrated_ab12.usd"
ZSHIFT = -0.35
APPLE_WORLD = Gf.Vec3d(0.3234, 0.113104, 0.757427)

shutil.copy(SRC, DST)
stage = Usd.Stage.Open(DST)

# 1) Tree6 下移
tree6 = stage.GetPrimAtPath("/World/Tree6")
xf = UsdGeom.Xformable(tree6)
M_cur = xf.GetOrderedXformOps()[0].GetOpTransform(Usd.TimeCode.Default())
M_new = M_cur * Gf.Matrix4d().SetTranslate(Gf.Vec3d(0,0,ZSHIFT))
xf.MakeMatrixXform().Set(M_new)

# 2) 苹果移动
ap = stage.GetPrimAtPath("/World/AppleTarget")
assert ap and ap.IsValid(), "no /World/AppleTarget (需在 calibrated/tamed USD 上运行)"
apx = UsdGeom.Xformable(ap)
ops = apx.GetOrderedXformOps()
assert ops, "AppleTarget has no xform op"
ops[0].Set(APPLE_WORLD)

# 3) 果柄重挂到 branch13 最近段
# 段 xform 原点在树根, 不在几何上 —— 必须用 bbox 中心找最近段
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default","render","proxy","guide"])
best = None
for prim in stage.Traverse():
    if prim.GetTypeName()=="Xform" and prim.GetName().startswith("seg") \
       and prim.GetParent().GetName()=="branch13":
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        c = Gf.Vec3d((r.GetMin()+r.GetMax())/2.0)
        d = (c - APPLE_WORLD).GetLength()
        M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if best is None or d < best[0]: best = (d, prim, M)
assert best, "branch13 segs not found"
dist, seg, M = best
stem = stage.GetPrimAtPath("/World/AppleStem")
assert stem and stem.IsValid(), "no /World/AppleStem"
j = UsdPhysics.Joint(stem)
j.GetBody0Rel().SetTargets([seg.GetPath()])
local = M.GetInverse().Transform(APPLE_WORLD)
j.GetLocalPos0Attr().Set(Gf.Vec3f(local))
j.GetLocalPos1Attr().Set(Gf.Vec3f(0,0,0))

stage.GetRootLayer().Save()

# 验证
def wt(p):
    M = UsdGeom.Xformable(stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return tuple(round(x,4) for x in M.ExtractTranslation())
print("[repose] Tree6 root:", wt("/World/Tree6"))
print("[repose] apple     :", wt("/World/AppleTarget"), "expect", tuple(APPLE_WORLD))
print("[repose] stem body0:", seg.GetPath(), f"(dist {dist*100:.1f} cm)")
print("[repose] wrote", DST)
