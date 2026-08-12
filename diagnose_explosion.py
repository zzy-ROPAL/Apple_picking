"""诊断 Isaac 回放"爆炸": (A) 静态物理属性 dump  (B) 树独处稳定性(机器人不动)
(C) 路径回放发散检测。全程只读, 不改场景。"""
import numpy as np
from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.types import ArticulationAction
import omni.usd
from omni.physx import get_physx_simulation_interface
from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools, Usd

USD  = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_fixed.usd"
ROOT = "/Franka"
P2   = "/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.claude/worktrees/experiment-scene-visualization-e0aaf2/benchmark_results_with_A_pengate_uni_nav_pen_metric/grasp_path_02.csv"
DT, VEL = 1.0/60.0, 0.3
ARM = [f"panda_joint{i}" for i in range(1,8)]
VMAX = np.array([2.175,2.175,2.175,2.175,2.61,2.61,2.61])

open_stage(USD)
stage = omni.usd.get_context().get_stage()

# ============================ (A) 静态物理属性 ============================
print("\n================= (A) STATIC PHYSICS DUMP =================", flush=True)
def dump_branch(bp):
    prim = stage.GetPrimAtPath(bp)
    if not (prim and prim.IsValid()): print(f"  {bp}: invalid"); return
    art = PhysxSchema.PhysxArticulationAPI(prim) if prim.HasAPI(PhysxSchema.PhysxArticulationAPI) else None
    selfcol = art.GetEnabledSelfCollisionsAttr().Get() if art else "no PhysxArticulationAPI"
    posit = art.GetSolverPositionIterationCountAttr().Get() if art else None
    velit = art.GetSolverVelocityIterationCountAttr().Get() if art else None
    print(f"  {bp}:  enabledSelfCollisions={selfcol}  solverPosIters={posit} velIters={velit}", flush=True)
    # segment masses
    masses=[]
    for c in prim.GetChildren():
        if c.HasAPI(UsdPhysics.MassAPI):
            m=UsdPhysics.MassAPI(c).GetMassAttr().Get()
            masses.append((c.GetName(), m))
    print(f"      segment masses: {masses}", flush=True)
    # one hinge's drive stiffness/damping
    for c in prim.GetChildren():
        if c.GetTypeName()=="PhysicsJoint" and "hinge" in c.GetName():
            for axis in ("rotX","rotY"):
                if UsdPhysics.DriveAPI.Get(c, axis):
                    d=UsdPhysics.DriveAPI.Get(c, axis)
                    print(f"      {c.GetName()}.{axis}: stiffness={d.GetStiffnessAttr().Get()} "
                          f"damping={d.GetDampingAttr().Get()} maxForce={d.GetMaxForceAttr().Get()}", flush=True)
            break
for bp in ["/World/Tree6/branch1","/World/Tree6/branch5"]:
    dump_branch(bp)
# collision groups
cg = stage.GetPrimAtPath("/World/Tree6/collisionGroups")
if cg and cg.IsValid():
    print("  collisionGroups children:", [c.GetName() for c in cg.GetChildren()], flush=True)
    for c in cg.GetChildren():
        col = UsdPhysics.CollisionGroup(c)
        filt = col.GetFilteredGroupsRel().GetTargets() if col else []
        inc = Usd.CollectionAPI.Get(c, "colliders")
        n_inc = len(inc.GetIncludesRel().GetTargets()) if inc else "?"
        print(f"      {c.GetName()}: filteredGroups={[str(f) for f in filt]}  #includes={n_inc}", flush=True)
else:
    print("  NO collisionGroups prim found", flush=True)

# ============================ helpers ============================
def load(p): return np.array([[float(x) for x in l.split(",")] for l in open(p) if l.strip()])
def timeparam(wps):
    refs=[wps[0]]
    for a,b in zip(wps[:-1],wps[1:]):
        n=max(1,int(np.ceil(float(np.max(np.abs(b-a)/(VMAX*VEL)))/DT)))
        for k in range(1,n+1): refs.append(a+(b-a)*(k/n))
    return np.array(refs)

world=World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
robot=Articulation(ROOT); world.scene.add(robot)
probes={b:Articulation(f"/World/Tree6/branch{b}") for b in (1,5,10)}
for a in probes.values(): world.scene.add(a)
world.reset()
dof=list(robot.dof_names); arm=np.array([dof.index(n) for n in ARM])
ctrl=robot.get_articulation_controller()
wps=load(P2); refs=timeparam(wps)

def maxspeed():
    """所有探针分支 + 机器人的最大关节速度 (rad/s); NaN→inf"""
    vals=[]
    for a in list(probes.values())+[robot]:
        v=a.get_joint_velocities()
        if v is None: continue
        v=np.asarray(v);
        if np.any(~np.isfinite(v)): return float('inf')
        vals.append(np.max(np.abs(v)) if v.size else 0.0)
    return max(vals) if vals else 0.0

def run_phase(name, drive_path, nmax):
    world.reset()
    q0=robot.get_joint_positions(); q0[arm]=wps[0]; robot.set_joint_positions(q0)
    ctrl.apply_action(ArticulationAction(joint_positions=wps[0], joint_indices=arm))
    for _ in range(5): world.step(render=False)
    onset=None; peak=0.0
    for step in range(nmax):
        if drive_path is not None:
            q=refs[min(step,len(refs)-1)]
            ctrl.apply_action(ArticulationAction(joint_positions=q, joint_indices=arm))
        # else: keep holding start pose (no new target) — tree alone
        world.step(render=False)
        ms=maxspeed(); peak=max(peak,ms)
        if onset is None and ms>30.0:
            onset=step
            print(f"  [{name}] DIVERGENCE at step {step} (t={step*DT:.2f}s), max_joint_speed={ms:.1f} rad/s", flush=True)
            # who exploded?
            for b,a in probes.items():
                v=a.get_joint_velocities()
                if v is not None and np.asarray(v).size and (not np.all(np.isfinite(v)) or np.max(np.abs(v))>30):
                    print(f"      -> branch{b} joint speeds: {np.round(np.asarray(v),1)}", flush=True)
            rv=robot.get_joint_velocities()
            print(f"      -> robot arm speeds: {np.round(np.asarray(rv)[arm],1) if rv is not None else None}", flush=True)
            break
    if onset is None:
        print(f"  [{name}] STABLE over {nmax} steps ({nmax*DT:.1f}s), peak_joint_speed={peak:.2f} rad/s", flush=True)
    return onset, peak

print("\n========== (B) TREE ALONE — robot frozen at start pose ==========", flush=True)
run_phase("tree-alone", None, 240)

print("\n========== (C) PATH REPLAY — grasp_path_02 ==========", flush=True)
run_phase("replay", refs, min(len(refs)+60, 2000))

print("\n[diag] done", flush=True)
sim.close()
