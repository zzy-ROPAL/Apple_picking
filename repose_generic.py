"""通用苹果重挂: y120root USD (树姿态固定) + 指定苹果位置/USD 枝名 → 新 USD.
argv: usd_dst  apple_x apple_y apple_z  usd_branch_name"""
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import shutil, sys
SRC="/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_calibrated_y120root.usd"
DST=sys.argv[1]; APPLE_WORLD=Gf.Vec3d(*[float(x) for x in sys.argv[2:5]]); BR=sys.argv[5]
shutil.copy(SRC,DST)
stage=Usd.Stage.Open(DST)
UsdGeom.Xformable(stage.GetPrimAtPath("/World/AppleTarget")).GetOrderedXformOps()[0].Set(APPLE_WORLD)
cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),["default","render","proxy","guide"])
best=None
for prim in stage.Traverse():
    if prim.GetTypeName()=="Xform" and prim.GetName().startswith("seg") \
       and prim.GetParent().GetName()==BR:
        r=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        c=Gf.Vec3d((r.GetMin()+r.GetMax())/2.0); d=(c-APPLE_WORLD).GetLength()
        M=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if best is None or d<best[0]: best=(d,prim,M)
dist,seg,M=best
j=UsdPhysics.Joint(stage.GetPrimAtPath("/World/AppleStem"))
j.GetBody0Rel().SetTargets([seg.GetPath()])
j.GetLocalPos0Attr().Set(Gf.Vec3f(M.GetInverse().Transform(APPLE_WORLD)))
j.GetLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
stage.GetRootLayer().Save()
print(f"[repose] {DST.split('/')[-1]}: stem -> {seg.GetPath()} ({dist*100:.1f} cm)")
