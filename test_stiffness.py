"""验证驯服后: (1)重力下树是否保持姿态(不下垂)  (2)回放是否仍不爆炸。
Phase A: 树独处 settle 3s, 量各枝关节角偏移(下垂代理), 抓帧 settled。
Phase B: 软起步回放 path_02, 查发散, 抓帧 grasp。"""
import os, numpy as np
from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.types import ArticulationAction
import omni.replicator.core as rep
from PIL import Image

USD="/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_tamed.usd"
P2="/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.claude/worktrees/experiment-scene-visualization-e0aaf2/benchmark_results_with_A_pengate_uni_nav_pen_metric/grasp_path_02.csv"
OUT="/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.claude/worktrees/experiment-scene-visualization-e0aaf2/isaac_stiff"; os.makedirs(OUT,exist_ok=True)
DT=1.0/480.0; ARM=[f"panda_joint{i}" for i in range(1,8)]
VMAX=np.array([2.175,2.175,2.175,2.175,2.61,2.61,2.61])
wps=np.array([[float(x) for x in l.split(",")] for l in open(P2) if l.strip()])

open_stage(USD)
world=World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
robot=Articulation("/Franka",name="franka"); world.scene.add(robot)
probes={b:Articulation(f"/World/Tree6/branch{b}",name=f"b{b}") for b in (1,3,5,10,13)}
for a in probes.values(): world.scene.add(a)
cam=rep.create.camera(position=(2.3,-2.0,1.7),look_at=(0.28,0.02,0.80))
rp=rep.create.render_product(cam,(1400,900)); rgb=rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach(rp)
world.reset()
dof=list(robot.dof_names); arm=np.array([dof.index(n) for n in ARM])
ctrl=robot.get_articulation_controller(); HOME=np.asarray(robot.get_joint_positions())[arm].copy()
def act(q): return ArticulationAction(joint_positions=np.asarray(q),joint_indices=arm)
def snap(t):
    for _ in range(3): world.step(render=True)
    d=rgb.get_data()
    if d is not None and len(d): Image.fromarray(np.asarray(d)[:,:,:3].astype(np.uint8)).save(f"{OUT}/{t}.png"); print("  snap",t,flush=True)
def treespeed():
    m=0.0
    for a in probes.values():
        v=a.get_joint_velocities()
        if v is None: continue
        v=np.asarray(v)
        if v.size and not np.all(np.isfinite(v)): return 1e9
        if v.size: m=max(m,float(np.max(np.abs(v))))
    return m
def sag():  # 各枝关节角最大绝对值 (rad) —— 下垂代理
    m=0.0
    for a in probes.values():
        q=a.get_joint_positions()
        if q is not None and np.asarray(q).size: m=max(m,float(np.max(np.abs(np.asarray(q)))))
    return m

# robot 停 home
q=robot.get_joint_positions(); q[arm]=HOME; robot.set_joint_positions(q)
ctrl.apply_action(act(HOME))
print("\n== PHASE A: tree settle under gravity (robot at home) ==",flush=True)
for s in range(int(3.0/DT)):
    world.step(render=False)
    if s%60==0: print(f"  t={s*DT:4.2f}s  max_branch_joint_angle={sag()*57.3:6.2f} deg  tree_speed={treespeed():.2f}",flush=True)
print(f"  SETTLED max branch joint angle = {sag()*57.3:.2f} deg (small = holds shape, no sag)",flush=True)
snap("settled")

print("\n== PHASE B: soft-start replay path_02 — still stable? ==",flush=True)
world.reset(); q=robot.get_joint_positions(); q[arm]=HOME; robot.set_joint_positions(q)
for _ in range(10): world.step(render=False)
# approach home->wp0
N=int(3.0/DT); div=None
for k in range(N):
    ctrl.apply_action(act(HOME+(wps[0]-HOME)*((k+1)/N))); world.step(render=False)
    if treespeed()>30: div=("approach",k*DT); break
# follow
if div is None:
    refs=[wps[0]]
    for a,b in zip(wps[:-1],wps[1:]):
        n=max(1,int(np.ceil(float(np.max(np.abs(b-a)/(VMAX*0.15)))/DT)))
        for kk in range(1,n+1): refs.append(a+(b-a)*(kk/n))
    refs=np.array(refs)
    for s in range(len(refs)):
        ctrl.apply_action(act(refs[min(s,len(refs)-1)])); world.step(render=False)
        if treespeed()>30: div=("path",s); break
    for _ in range(int(1.0/DT)): world.step(render=False)
snap("grasp")
print(f"  diverged: {div}   peak handled OK={div is None}",flush=True)
print("[test] done ->",OUT,flush=True)
sim.close()
