"""检查 Isaac USD 场景: 打印层级 / prim 类型 / 网格与几何 / 变换 / 物理属性。"""
import sys
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

path = sys.argv[1]
stage = Usd.Stage.Open(path)
print("="*70)
print("STAGE:", path)
up = UsdGeom.GetStageUpAxis(stage)
mpu = UsdGeom.GetStageMetersPerUnit(stage)
dp = stage.GetDefaultPrim()
dp_name = dp.GetName() if dp and dp.IsValid() else "(none set)"
print(f"upAxis={up}  metersPerUnit={mpu}  defaultPrim={dp_name!r}")
tr = stage.GetRootLayer()
print(f"start/end timecode: {stage.GetStartTimeCode()} .. {stage.GetEndTimeCode()}")
print("="*70)

def xform_world(prim):
    try:
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    except Exception:
        return None

n_by_type = {}
meshes = []
apples = []
physics = []
for prim in stage.Traverse():
    t = prim.GetTypeName()
    n_by_type[t] = n_by_type.get(t, 0) + 1
    path_s = str(prim.GetPath())
    low = path_s.lower()
    if t == "Mesh":
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        fvc = m.GetFaceVertexCountsAttr().Get()
        npts = len(pts) if pts else 0
        nfaces = len(fvc) if fvc else 0
        # world-space bbox from points
        wb = None
        M = xform_world(prim)
        if pts and M is not None:
            import math
            xs=[];ys=[];zs=[]
            for p in pts:
                wp = M.Transform(Gf.Vec3d(p[0],p[1],p[2]))
                xs.append(wp[0]);ys.append(wp[1]);zs.append(wp[2])
            wb=(min(xs),min(ys),min(zs),max(xs),max(ys),max(zs))
        meshes.append((path_s, npts, nfaces, wb))
    if "apple" in low:
        M = xform_world(prim)
        pos = None
        if M is not None:
            tr = M.ExtractTranslation(); pos=(tr[0],tr[1],tr[2])
        apples.append((path_s, t, pos))
    # physics APIs
    applied = prim.GetAppliedSchemas()
    phys = [a for a in applied if "Physics" in a or "Collision" in a or "RigidBody" in a]
    if phys:
        physics.append((path_s, t, phys))

print("PRIM TYPE COUNTS:")
for k,v in sorted(n_by_type.items(), key=lambda x:-x[1]):
    print(f"  {str(k) or '(none)':<22} {v}")

print("\nFULL PRIM TREE (path : type):")
for prim in stage.Traverse():
    depth = str(prim.GetPath()).count("/") - 1
    print("  " + "  "*depth + prim.GetName() + f"  [{prim.GetTypeName()}]")

print(f"\nMESHES ({len(meshes)}):  path | #points | #faces | world-bbox")
for p,npt,nf,wb in meshes:
    bs = "-" if wb is None else "x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]"%(wb[0],wb[3],wb[1],wb[4],wb[2],wb[5])
    print(f"  {p}\n     pts={npt} faces={nf}  bbox(world,m)= {bs}")

print(f"\nAPPLE-NAMED PRIMS ({len(apples)}):")
for p,t,pos in apples:
    ps = "-" if pos is None else "(%.4f, %.4f, %.4f)"%pos
    print(f"  {p}  [{t}]  world_pos(m)= {ps}")

print(f"\nPHYSICS-TAGGED PRIMS ({len(physics)}):")
for p,t,ph in physics:
    print(f"  {p}  [{t}]  {ph}")

# references / payloads
print("\nREFERENCES / PAYLOADS / VARIANTS:")
found=False
for prim in stage.Traverse():
    md = prim.GetMetadata("references")
    pl = prim.GetMetadata("payload")
    if prim.HasAuthoredReferences() or (pl is not None):
        found=True
        print(f"  {prim.GetPath()}  refs={prim.HasAuthoredReferences()} payload={pl is not None}")
if not found:
    print("  (none — self-contained stage)")
