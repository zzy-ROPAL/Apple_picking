"""1) 把 tree6 引用路径从 east118(旧机器)改到本机 7-18/zzy 副本 → 树几何可正确加载。
   2) 把 /World/Tree6 重新摆位, 使 树-机器人 相对位姿 == 我们刚在 Bullet3 做的改造场景
      (绕树根竖直轴顺时针 90° + 树根到基座水平距离减半)。
   原文件不动, 输出 scene_with_tree_apple_fixed.usd。
"""
import os
from pxr import Sdf, Usd, UsdGeom, Gf

SRC = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple.usd"
DST = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_fixed.usd"
OLD = "/home/east118/zhongyan_zhang/July/7-3/"
NEW = "/home/zhongyanzhang/zhongyan_zhang/July/7-18/zzy/tree/"

# ---- 与 Bullet3 改造实验完全一致的刚体变换参数 ----
C     = Gf.Vec3d(0.7526, -0.0624, 0.0011)   # 旋转中心 = C++ TREE_ROOT
YAW   = -90.0                                # 顺时针 90° (绕 +Z 从顶视)
SCALE = 0.5                                  # tree_dist_scale: 树根→基座距离减半
SHIFT = Gf.Vec3d((SCALE-1.0)*C[0], (SCALE-1.0)*C[1], 0.0)   # = -0.5 * root_xy

# ---------- 1) 路径修复 (在 layer 文本层做子串替换, 引用保持为引用) ----------
src_layer = Sdf.Layer.FindOrOpen(SRC)
text = src_layer.ExportToString()
n_old = text.count(OLD)
text = text.replace(OLD, NEW)
assert OLD not in text, "still has east118 paths!"
with open(DST, "w") as f:
    f.write(text)
print(f"[paths] replaced {n_old} occurrences of east118 -> local 7-18/zzy")

# ---------- 2) 重新摆位 Tree6 ----------
stage = Usd.Stage.Open(DST)
tree6 = stage.GetPrimAtPath("/World/Tree6")
xf = UsdGeom.Xformable(tree6)
# 现有 Tree6 本地矩阵 (行向量约定: worldPt = localPt * M)
M_cur = xf.GetOrderedXformOps()[0].GetOpTransform(Usd.TimeCode.Default())

# 构造实验变换 T(p) = R_z(YAW)*(p - C) + C + SHIFT  (行向量约定)
Tneg   = Gf.Matrix4d().SetTranslate(-C)
Rz     = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1), YAW))
Tpos   = Gf.Matrix4d().SetTranslate(C + SHIFT)
M_T    = Tneg * Rz * Tpos          # 行向量: 先 -C, 再旋转, 再 +C+SHIFT

# 叠加到现有变换之上: worldPt_new = (localPt * M_cur) * M_T = localPt * (M_cur * M_T)
M_new = M_cur * M_T
xf.MakeMatrixXform().Set(M_new)     # 清空并设为单一 matrix op
stage.GetRootLayer().Save()

# ---------- 验证 ----------
def wtrans(p):
    pr = stage.GetPrimAtPath(p)
    if not (pr and pr.IsValid()): return None
    M = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return tuple(round(x,4) for x in M.ExtractTranslation())
print("[repose] verification (world translations, m):")
print("  Franka base        :", wtrans("/Franka"), " (robot fixed at origin, z=0.4)")
print("  Tree6 root (new)    :", wtrans("/World/Tree6"), " expect ~(0.369,-0.034)")
print("  trunk               :", wtrans("/World/Tree6/trunk"))
ap = stage.GetPrimAtPath("/World/Tree6/branch5/seg2/AppleTarget")
if ap and ap.IsValid():
    M = UsdGeom.Xformable(ap).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = M.ExtractTranslation()
    import math
    print("  AppleTarget (new)   :", tuple(round(x,4) for x in t),
          f" dist_from_base_xy={math.hypot(t[0],t[1]):.3f}m")
print("[out]", DST)
