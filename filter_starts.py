"""起点过滤: 加载 panda(基座 z=0.4, 对齐 planner base_offset) + 场景圆柱,
计算每个候选起点全臂与所有枝条(软+硬+树干)的最小间隙, 只保留 >= 3cm 的,
再按关节空间分散度选 5 个."""
import json, math, numpy as np, pybullet as p

W = "/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.claude/worktrees/experiment-scene-visualization-e0aaf2"
SCENE = W + "/benchmark_results_with_A_pengate_uni_nav_ab12_zr0.35_genstarts/scene_layout.json"
STARTS = W + "/benchmark_results_with_A_pengate_uni_nav_ab12_zr0.35_genstarts/starts_used.txt"
URDF = "/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.venv/lib/python3.12/site-packages/pybullet_data/franka_panda/panda.urdf"
CLEAR_MIN = 0.03

p.connect(p.DIRECT)
robot = p.loadURDF(URDF, [0,0,0.4], useFixedBase=True)
nj = p.getNumJoints(robot)
arm = [j for j in range(nj) if p.getJointInfo(robot,j)[2] == p.JOINT_REVOLUTE][:7]

d = json.load(open(SCENE))
bodies = []
def add_cyl(p0, p1, r):
    p0, p1 = np.array(p0), np.array(p1)
    v = p1-p0; L = float(np.linalg.norm(v))
    if L < 1e-6: return
    a = v/L; mid = 0.5*(p0+p1); z = np.array([0,0,1.0]); c = float(np.dot(z,a))
    if c > 0.9999: q=[0,0,0,1]
    elif c < -0.9999: q=[1,0,0,0]
    else:
        ax=np.cross(z,a); ax/=np.linalg.norm(ax); th=math.acos(c)
        q=[ax[0]*math.sin(th/2),ax[1]*math.sin(th/2),ax[2]*math.sin(th/2),math.cos(th/2)]
    cs = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=L)
    bodies.append(p.createMultiBody(baseCollisionShapeIndex=cs, basePosition=mid.tolist(), baseOrientation=q))
for c in d['cylinders']:
    if c['p0'] == c['p1']:
        cs = p.createCollisionShape(p.GEOM_SPHERE, radius=c['radius'])
        bodies.append(p.createMultiBody(baseCollisionShapeIndex=cs, basePosition=c['p0']))
    else:
        add_cyl(c['p0'], c['p1'], c['radius'])

starts = [tuple(float(x) for x in l.split(',')) for l in open(STARTS) if l.strip()]
starts = list(dict.fromkeys(starts))
res = []
for q in starts:
    for j,v in zip(arm, q): p.resetJointState(robot, j, v)
    dmin = 1e9
    for b in bodies:
        pts = p.getClosestPoints(robot, b, distance=1.0)
        for pt in pts: dmin = min(dmin, pt[8])
    res.append((dmin, q))
res.sort(key=lambda x: -x[0])
ok = [r for r in res if r[0] >= CLEAR_MIN]
print(f"candidates {len(starts)}, clearance>= {CLEAR_MIN*100:.0f}cm: {len(ok)}")
for dmin,q in res[:12]: print(f"  clear={dmin*100:5.1f}cm  {','.join(f'{v:.4f}' for v in q)}")
# 关节空间贪心分散选 5
sel = [ok[0]]
while len(sel) < 5 and len(sel) < len(ok):
    best = max((r for r in ok if r not in sel),
               key=lambda r: min(np.linalg.norm(np.array(r[1])-np.array(s[1])) for s in sel))
    sel.append(best)
out = W + "/saved_starts_ab12_5x10.txt"
with open(out,"w") as f:
    f.write("# 5 gate-valid starts, full-arm clearance >= 3cm from ALL branches (scene: ab12 zr0.35)\n")
    for dmin,q in sel:
        f.write(f"# clearance {dmin*100:.1f} cm\n")
        for _ in range(10): f.write(",".join(f"{v:.6f}" for v in q)+"\n")
print("\nselected 5 (clearance cm):", [round(s[0]*100,1) for s in sel])
print("wrote", out)
