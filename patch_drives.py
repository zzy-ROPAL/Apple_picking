"""Method 2 — 驯服软枝驱动: 在 USD 副本上给所有枝条 hinge 驱动加"有限 maxForce"、
降低刚度、加大段质量/惯量, 使被推太狠时枝条屈服而非爆炸。原文件不动。"""
import shutil
from pxr import Usd, UsdPhysics, UsdGeom, UsdShade, Sdf, Gf

SRC = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_fixed.usd"
DST = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_tamed.usd"

# 目标: 保持"刚性"(重力下不下垂) + 不爆炸。
# 关键手法: 关掉枝条刚体的重力(disableGravity)→ 枝条保持建模姿态、零下垂,
#   于是质量可以留得重一点(接触稳定), 与"下垂"彻底解耦。armature 再加一层
#   关节稳定, maxForce 有限保证被推太狠时屈服不爆。
DISABLE_GRAVITY = True   # 枝条不受重力 → 零下垂 (无论刚度多少; 仍会被推弯并回弹)
STIFF   = 150.0   # 已验证稳定的软刚度 (重力已关, 不靠它撑姿态)
DAMP    = 30.0
MAXF    = 1.5     # 已验证稳定
MASS    = 0.5     # 接触稳定
ARMATURE= 0.08    # 反射惯量: 额外稳定
MAXDEPEN= 0.15    # m/s 分离速度上限 (调低: 支持"直接传送到嵌枝起点", 重叠枝条缓慢滑出而非弹出)
POSITER = 128
# 柔顺接触 (compliant contact): 枝条接触改为弹簧-阻尼 F = k·穿深, 模拟树皮/木质局部弹性。
# 这是消除"楔在腕部 C 形钳里的 kN 级刚体干涉力"的物理正解 —— 力被材料弹性限幅:
# 5mm 穿深 × 3000 N/m ≈ 15 N, 与真实细枝受力量级一致。
# 标定: 细枝(15mm)局部受 ~20N 会让出数厘米 → k ≈ 20N / 0.06m ≈ 350 N/m。
# 使"深重叠(≈60mm, 出现在刚性根段/腕部楔点)"的接触力落在 ~20N 量级 (用户指定)。
COMP_K  = 350.0   # N/m 接触刚度
COMP_D  = 30.0    # 接触阻尼
# 苹果-枝 软连接 (果柄弹簧): 苹果为独立刚体, 用各向同性平移弹簧挂在枝上,
# 被碰会摆开并回弹, 不再是"焊死在刚性根段上"的硬连接。
APPLE_MASS   = 0.15   # kg
APPLE_STEM_K = 150.0  # N/m  (~推 5N 摆开 3cm, 真实果柄横向刚度量级)
APPLE_STEM_D = 5.0
APPLE_STEM_F = 30.0   # N  弹簧最大力 (真实苹果脱落力 ~10-40N; 不做脱落, 只设上限)

shutil.copy(SRC, DST)
stage = Usd.Stage.Open(DST)
n_drv = n_mass = n_dep = n_art = n_bind = 0

# 柔顺接触材质 (绑到全部软枝碰撞体)
_mat = UsdShade.Material.Define(stage, "/World/Tree6/BranchCompliantMaterial")
_mp = _mat.GetPrim()
UsdPhysics.MaterialAPI.Apply(_mp)
_mp.AddAppliedSchema("PhysxMaterialAPI")   # 正式挂 PhysX 材质 schema (离线无 PhysxSchema, 用元数据)
_mp.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(COMP_K)
_mp.CreateAttribute("physxMaterial:compliantContactDamping",  Sdf.ValueTypeNames.Float).Set(COMP_D)
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "/World/Tree6/branch" not in path:
        continue
    if prim.GetTypeName() == "PhysicsJoint":
        has_drive = False
        for axis in ("rotX", "rotY"):
            d = UsdPhysics.DriveAPI.Get(prim, axis)
            if d:
                # force 型驱动: stiffness/damping/maxForce 都是真实力矩单位 (N·m/rad 等),
                # 是标定过的材料属性。枝条重力已关, 不存在"force 型下垂过多"问题 (用户确认)。
                d.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
                d.CreateStiffnessAttr().Set(STIFF)
                d.CreateDampingAttr().Set(DAMP)
                d.CreateMaxForceAttr().Set(MAXF)
                n_drv += 1
                has_drive = True
        if has_drive:  # 反射惯量: 稳定高刚度驱动而不增重力
            prim.CreateAttribute("physxJoint:armature",
                                 Sdf.ValueTypeNames.Float).Set(ARMATURE)
    if prim.HasAPI(UsdPhysics.MassAPI):
        UsdPhysics.MassAPI(prim).CreateMassAttr().Set(MASS)
        n_mass += 1
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        # 原始属性名 (PhysxSchema 在离线 pxr 不可用): 限制分离速度 + 关重力
        prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity",
                             Sdf.ValueTypeNames.Float).Set(MAXDEPEN)
        if DISABLE_GRAVITY:
            prim.CreateAttribute("physxRigidBody:disableGravity",
                                 Sdf.ValueTypeNames.Bool).Set(True)
        n_dep += 1
        # 绑定柔顺接触材质 (physics purpose); strongerThanDescendants 压过引用资产自带材质
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            _mat, UsdShade.Tokens.strongerThanDescendants, "physics")
        n_bind += 1
    if prim.GetAttribute("physxArticulation:solverPositionIterationCount"):
        prim.CreateAttribute("physxArticulation:solverPositionIterationCount",
                             Sdf.ValueTypeNames.Int).Set(POSITER)
        n_art += 1
# ---- 苹果对齐 bullet3: 移到枝5弧长中点 (与 C++ --apple-frac 0.5 的世界坐标一致) ----
# 该坐标来自 benchmark setup log: "apple at 0.520081 0.248243 0.765182" (旋转+半距场景)
APPLE_WORLD = Gf.Vec3d(0.520081, 0.248243, 0.765182)
APPLE_R = 0.04
# 找 branch5 里离目标最近的 seg 作为新父级 (苹果跟随该段运动)
best = None
for prim in stage.Traverse():
    if prim.GetTypeName() == "Xform" and prim.GetName().startswith("seg") \
       and prim.GetParent().GetName() == "branch5":
        M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        d = (M.ExtractTranslation() - APPLE_WORLD).GetLength()
        if best is None or d < best[0]:
            best = (d, prim, M)
_, seg, M = best
old = None
for prim in stage.Traverse():
    if prim.GetName() == "AppleTarget": old = prim.GetPath(); break
if old: stage.RemovePrim(old)
# 苹果 = /World 下的独立刚体 (不嵌套在枝刚体里), 世界坐标直接定位
ap = stage.DefinePrim(Sdf.Path("/World/AppleTarget"), "Sphere")
UsdGeom.Sphere(ap).CreateRadiusAttr().Set(APPLE_R)
UsdGeom.Gprim(ap).CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.10, 0.10)])
UsdGeom.Xformable(ap).AddTranslateOp().Set(Gf.Vec3d(APPLE_WORLD))
UsdPhysics.CollisionAPI.Apply(ap)
UsdPhysics.RigidBodyAPI.Apply(ap)
UsdPhysics.MassAPI.Apply(ap).CreateMassAttr().Set(APPLE_MASS)
ap.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)
ap.CreateAttribute("physxRigidBody:maxDepenetrationVelocity",
                   Sdf.ValueTypeNames.Float).Set(MAXDEPEN)
# 苹果也用柔顺接触材质
UsdShade.MaterialBindingAPI.Apply(ap).Bind(_mat, UsdShade.Tokens.strongerThanDescendants, "physics")
# 果柄弹簧: D6 joint, 平移三轴 force 型弹簧驱动 (各向同性 → 与关节坐标系朝向无关),
# 旋转全自由 (球体朝向无意义), 关节自动禁用 苹果↔挂载枝段 的碰撞
local = M.GetInverse().Transform(APPLE_WORLD)   # 挂点在枝段坐标系中的位置
stem = UsdPhysics.Joint.Define(stage, Sdf.Path("/World/AppleStem"))
stem.CreateBody0Rel().SetTargets([seg.GetPath()])
stem.CreateBody1Rel().SetTargets([ap.GetPath()])
stem.CreateLocalPos0Attr().Set(Gf.Vec3f(local))
stem.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
for axis in ("transX", "transY", "transZ"):
    dr = UsdPhysics.DriveAPI.Apply(stem.GetPrim(), axis)
    dr.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
    dr.CreateStiffnessAttr().Set(APPLE_STEM_K)
    dr.CreateDampingAttr().Set(APPLE_STEM_D)
    dr.CreateMaxForceAttr().Set(APPLE_STEM_F)
    dr.CreateTargetPositionAttr().Set(0.0)
print(f"[apple] independent body at /World/AppleTarget world={tuple(APPLE_WORLD)}; "
      f"stem spring K={APPLE_STEM_K} D={APPLE_STEM_D} maxF={APPLE_STEM_F} -> {seg.GetPath()}")

stage.GetRootLayer().Save()
print(f"[patch] drives={n_drv} re-massed={n_mass} depenCap={n_dep} artIters={n_art} compliantBind={n_bind} (K={COMP_K} D={COMP_D})")
print(f"[patch] STIFF={STIFF} DAMP={DAMP} MAXF={MAXF} MASS={MASS} MAXDEPEN={MAXDEPEN} POSITER={POSITER}")
print(f"[patch] wrote {DST}")
