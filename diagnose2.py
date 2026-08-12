"""区分"树本身抖爆(刚度参数)" vs "起点嵌枝触发":
 Phase 1: 机器人停在 HOME(远离枝冠), 步进物理, 看树是否自爆。
 Phase 2: 机器人置于路径起点 wp0, 先读起点接触(是否嵌枝), 再步进看是否触发。
只读, 不改场景。每步 flush, 崩溃也能看到最后一步。"""
import numpy as np, sys
from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.types import ArticulationAction
import omni.usd
from omni.physx import get_physx_simulation_interface
from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools

USD  = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_fixed.usd"
ROOT = "/Franka"
P2   = "/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.claude/worktrees/experiment-scene-visualization-e0aaf2/benchmark_results_with_A_pengate_uni_nav_pen_metric/grasp_path_02.csv"
DT   = 1.0/60.0
ARM  = [f"panda_joint{i}" for i in range(1,8)]

wps = np.array([[float(x) for x in l.split(",")] for l in open(P2) if l.strip()])

# 起点接触监听
events=[]
def on_contact(headers, data):
    for h in headers:
        a=str(PhysicsSchemaTools.intToSdfPath(h.actor0)); b=str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        ar=a.startswith(ROOT); br=b.startswith(ROOT)
        if ar==br: continue
        other=b if ar else a; rp=a if ar else b
        imp=0.0
        for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
            d=data[i]; imp+=float((d.impulse.x**2+d.impulse.y**2+d.impulse.z**2)**0.5)
        events.append((rp.split("/")[-1], other.split("/World/Tree6/")[-1], round(imp,4)))
sub=get_physx_simulation_interface().subscribe_contact_report_events(on_contact)

open_stage(USD)
stage=omni.usd.get_context().get_stage()
# 打开机器人 link 接触上报
for p in stage.Traverse():
    if str(p.GetPath()).startswith(ROOT) and p.HasAPI(UsdPhysics.RigidBodyAPI):
        PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr().Set(0.0)

world=World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
robot=Articulation(ROOT, name="franka"); world.scene.add(robot)
probes={b:Articulation(f"/World/Tree6/branch{b}", name=f"branch{b}") for b in (1,5,10)}
for a in probes.values(): world.scene.add(a)
world.reset()
dof=list(robot.dof_names); arm=np.array([dof.index(n) for n in ARM])
ctrl=robot.get_articulation_controller()

def tree_speed():
    m=0.0
    for a in probes.values():
        v=a.get_joint_velocities()
        if v is None: continue
        v=np.asarray(v)
        if v.size and not np.all(np.isfinite(v)): return float('inf')
        if v.size: m=max(m,float(np.max(np.abs(v))))
    return m
def robot_speed():
    v=robot.get_joint_velocities()
    if v is None: return 0.0
    v=np.asarray(v)[arm]
    return float('inf') if not np.all(np.isfinite(v)) else float(np.max(np.abs(v)))

def step_and_watch(tag, nmax, apply_q=None):
    onset=None
    for s in range(nmax):
        if apply_q is not None:
            ctrl.apply_action(ArticulationAction(joint_positions=apply_q, joint_indices=arm))
        try:
            world.step(render=False)
        except Exception as e:
            print(f"  [{tag}] world.step CRASHED at step {s}: {e}", flush=True); return s
        ts, rs = tree_speed(), robot_speed()
        if s%10==0 or ts>30 or rs>30:
            print(f"  [{tag}] step {s:3d} t={s*DT:5.2f}s  tree_max_jointspeed={ts:8.1f}  robot_max={rs:8.1f} rad/s", flush=True)
        if onset is None and (ts>30 or rs>30):
            print(f"  [{tag}] >>> DIVERGENCE at step {s} (t={s*DT:.2f}s)  tree={ts:.1f} robot={rs:.1f}", flush=True)
            onset=s; break
    if onset is None:
        print(f"  [{tag}] STABLE over {nmax} steps ({nmax*DT:.1f}s)", flush=True)
    return onset

print("\n===== PHASE 1: robot at HOME (folded, away from crown) — is the TREE stable alone? =====", flush=True)
q_home=robot.get_joint_positions()  # USD 默认 = home
print("  robot home joints(arm) =", np.round(np.asarray(q_home)[arm],3), flush=True)
events.clear()
step_and_watch("tree-alone@home", 180, apply_q=None)
print("  contacts during Phase1 (robot-tree):", events[:8], "..." if len(events)>8 else "", " total=",len(events), flush=True)

print("\n===== PHASE 2: robot at START pose wp0 — does the start overlap a branch? =====", flush=True)
world.reset()
q0=robot.get_joint_positions(); q0[arm]=wps[0]; robot.set_joint_positions(q0)
events.clear()
# 先只 settle 1 步读起点接触
try: world.step(render=False)
except Exception as e: print("  step0 crash:", e, flush=True)
print("  START-POSE contacts (robot link, tree prim, impulse):", flush=True)
for e in events[:15]: print("     ", e, flush=True)
print("     total start contacts =", len(events),
      " | branch=", sum('branch' in e[1] for e in events),
      " trunk=", sum('trunk' in e[1] for e in events), flush=True)
events.clear()
step_and_watch("hold@start", 120, apply_q=wps[0])

print("\n[diag2] done", flush=True)
sim.close()
