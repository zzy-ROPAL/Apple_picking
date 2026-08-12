"""y120root 场景对齐: ab12_truer USD 再绕 (0.3763,-0.0312) 竖直轴转 -30°
(等效 bullet --tree-yaw -90 → -120), 苹果移到 branch12(C++) 根部位置,
果柄重挂 branch13(USD) 最近段 (bbox 中心)."""
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import shutil, math

SRC="/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_calibrated_ab12_truer.usd"
DST="/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_calibrated_y120root.usd"
PIVOT = Gf.Vec3d(0.3763, -0.0312, 0.0)   # = 0.5 * 原树根 xy (dist-scale 后的树根竖直轴)
APPLE_WORLD = Gf.Vec3d(0.456705, 0.0873095, 0.801928)

shutil.copy(SRC, DST)
stage = Usd.Stage.Open(DST)
tree6 = stage.GetPrimAtPath("/World/Tree6")
xf = UsdGeom.Xformable(tree6)
M_cur = xf.GetOrderedXformOps()[0].GetOpTransform(Usd.TimeCode.Default())
M_T = (Gf.Matrix4d().SetTranslate(-PIVOT)
       * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1), -30.0))
       * Gf.Matrix4d().SetTranslate(PIVOT))
xf.MakeMatrixXform().Set(M_cur * M_T)

ap = stage.GetPrimAtPath("/World/AppleTarget")
UsdGeom.Xformable(ap).GetOrderedXformOps()[0].Set(APPLE_WORLD)

cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default","render","proxy","guide"])
best=None
for prim in stage.Traverse():
    if prim.GetTypeName()=="Xform" and prim.GetName().startswith("seg") \
       and prim.GetParent().GetName()=="branch13":
        r=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        c=Gf.Vec3d((r.GetMin()+r.GetMax())/2.0)
        d=(c-APPLE_WORLD).GetLength()
        M=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if best is None or d<best[0]: best=(d,prim,M)
dist,seg,M=best
stem=stage.GetPrimAtPath("/World/AppleStem")
j=UsdPhysics.Joint(stem)
j.GetBody0Rel().SetTargets([seg.GetPath()])
j.GetLocalPos0Attr().Set(Gf.Vec3f(M.GetInverse().Transform(APPLE_WORLD)))
j.GetLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
stage.GetRootLayer().Save()
print("[repose] stem ->", seg.GetPath(), f"dist {dist*100:.1f} cm")
# 验证: branch13 根段中心应贴近 bullet branch12 根 (0.457,0.087,~0.90)
for prim in stage.Traverse():
    if str(prim.GetPath())=="/World/Tree6/branch13/seg0":
        r=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        c=(r.GetMin()+r.GetMax())/2.0
        print("[verify] branch13/seg0 bbox center:", tuple(round(x,3) for x in c))
print("[repose] wrote", DST)
