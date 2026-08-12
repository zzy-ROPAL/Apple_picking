"""软起步回放: 机器人先停在无碰撞 home 姿态(Phase1 已验证稳定), 再缓慢过渡到
路径起点 wp0(枝条被逐步推开而非瞬间嵌入), 然后低速跟随规划路径。
只读场景; 输出发散检测 + 接触日志 + 截图。"""
import os, csv, numpy as np
from isaacsim import SimulationApp
sim = SimulationApp({"headless": False})   # GUI 版: 窗口内可旋转视角观看
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.types import ArticulationAction
import omni.usd, omni.replicator.core as rep
from omni.physx import get_physx_simulation_interface
from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools
from PIL import Image

import sys
USD  = sys.argv[1] if len(sys.argv)>1 else "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_calibrated_ab12_truer.usd"
ROOT = "/Franka"
_DFLT = "/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.claude/worktrees/experiment-scene-visualization-e0aaf2/benchmark_results_with_A_pengate_uni_nav_ab12_zr0.35_ab12_5x10/grasp_path_13.csv"
P2   = sys.argv[2] if len(sys.argv)>2 else _DFLT
_tag = os.path.splitext(os.path.basename(P2))[0]
OUT  = "/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.claude/worktrees/experiment-scene-visualization-e0aaf2/isaac_ab12_gui_"+_tag; os.makedirs(OUT, exist_ok=True)
DT   = 1.0/480.0   # 更细步长, 配合驯服后的驱动
ARM  = [f"panda_joint{i}" for i in range(1,8)]
VMAX = np.array([2.175,2.175,2.175,2.175,2.61,2.61,2.61])
TELEPORT   = True    # True: 直接把机器人放到 wp0(不经过 home->wp0 运动, 避免路上压弯枝条);
                     # 依赖驯服后的 maxDepenVel/有限扭矩 —— 嵌入的枝条会被缓缓挤出而非弹飞
SETTLE_S   = 2.0     # 传送后原地稳定时长 (s), 让被嵌入的枝条温和滑出
APPROACH_S = 4.0     # (TELEPORT=False 时) home->wp0 缓慢过渡时长 (s)
PATH_VEL   = 0.08    # 路径跟随速度缩放 (更慢; 轴向拖行路径需要更轻柔)

wps=np.array([[float(x) for x in l.split(",")] for l in open(P2) if l.strip()])
def timeparam(a0, wpts, vel):
    seq=[wpts[0]]; full=np.vstack([wpts])
    refs=[full[0]]
    for a,b in zip(full[:-1],full[1:]):
        n=max(1,int(np.ceil(float(np.max(np.abs(b-a)/(VMAX*vel)))/DT)))
        for k in range(1,n+1): refs.append(a+(b-a)*(k/n))
    return np.array(refs)

events=[]
def on_contact(headers,data):
    for h in headers:
        a=str(PhysicsSchemaTools.intToSdfPath(h.actor0)); b=str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        ar=a.startswith(ROOT); br=b.startswith(ROOT)
        if ar==br: continue
        other=b if ar else a; rp=a if ar else b
        imp=0.0
        for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
            d=data[i]; imp+=float((d.impulse.x**2+d.impulse.y**2+d.impulse.z**2)**0.5)
        c="trunk" if "trunk" in other.lower() else ("branch" if "branch" in other.lower() else "other")
        events.append((round(_T[0],3), rp.split("/")[-1], other.split("/World/Tree6/")[-1], c, round(imp,4)))
sub=get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
_T=[0.0]
_GSTEP=[0]
def gstep():
    _GSTEP[0]+=1
    world.step(render=(_GSTEP[0]%16==0))  # 每 16 物理步渲染一帧

open_stage(USD)
stage=omni.usd.get_context().get_stage()
for p in stage.Traverse():
    if str(p.GetPath()).startswith(ROOT) and p.HasAPI(UsdPhysics.RigidBodyAPI):
        PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr().Set(0.0)

world=World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
robot=Articulation(ROOT, name="franka"); world.scene.add(robot)
# 指标采样: TCP(panda_hand 系 +z 0.1034m, Franka 官方 TCP 偏置) 与 苹果心
from isaacsim.core.prims import SingleXFormPrim
import omni.usd as _ou
from pxr import UsdPhysics as _UP

# ---- 真实 Franka 执行器: 把关节驱动 maxForce 压到官方力矩上限 ----
# 官方: joint1-4 = 87 N·m, joint5-7 = 12 N·m。此后机器人无法输出超过真实电机
# 的力/力矩 → 接触力回到真实量级 (被楔住时会停滞/偏差, 与真机一致)。
REAL_TORQUE = True
TAU_MAX = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
if REAL_TORQUE:
    _st = _ou.get_context().get_stage()
    n_cap = 0
    for prim in _st.Traverse():
        nm = prim.GetName()
        if nm.startswith("panda_joint") and nm[-1].isdigit() and len(nm) == len("panda_jointX"):
            idx = int(nm[-1]) - 1
            if 0 <= idx < 7:
                d = _UP.DriveAPI.Get(prim, "angular")
                if d:
                    d.CreateMaxForceAttr().Set(TAU_MAX[idx]); n_cap += 1
    print(f"[real-torque] capped {n_cap} joint drives to Franka limits {TAU_MAX}", flush=True)

hand_x  = SingleXFormPrim("/Franka/panda_hand", name="handx")
# AppleTarget 的父级可能已改 (苹果移到枝中点), 按名字搜索
_apath = next(str(p.GetPath()) for p in _ou.get_context().get_stage().Traverse()
              if p.GetName()=="AppleTarget")
print("[metrics] AppleTarget at", _apath, flush=True)
apple_x = SingleXFormPrim(_apath, name="applex")
TCP_OFF = np.array([0.0, 0.0, 0.1034])
def quat_rot(q, v):
    w,x,y,z=q; Q=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                           [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                           [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
    return Q@v
def tcp_state():
    p,q = hand_x.get_world_pose()          # q = wxyz
    p=np.asarray(p); q=np.asarray(q)
    return p + quat_rot(q, TCP_OFF), q
def apple_pos():
    p,_ = apple_x.get_world_pose()
    return np.asarray(p)
metrics=dict(min_dist=1e9, min_dist_t=0.0, min_dist_quat=None, min_dist_zaxis=None,
             min_dist_align_deg=None)
def sample_metrics():
    tcp,q = tcp_state(); ap = apple_pos()
    d=float(np.linalg.norm(tcp-ap))
    if d < metrics["min_dist"]:
        z=quat_rot(q, np.array([0,0,1.0]))          # 手爪接近轴 (hand +z)
        u=(ap-tcp)/max(1e-9,d)
        metrics.update(min_dist=d, min_dist_t=_T[0], min_dist_quat=q.copy(),
                       min_dist_zaxis=z,
                       min_dist_align_deg=float(np.degrees(np.arccos(np.clip(np.dot(z,u),-1,1)))))
# 全部 23 根枝: 发散监测 + 最大枝条变形(最大铰链弯角)统计
probes={b:Articulation(f"/World/Tree6/branch{b}", name=f"branch{b}") for b in range(1,24)}
for a in probes.values(): world.scene.add(a)
deform=dict(max_rad=0.0, branch=None, t=0.0)
def sample_deform():
    for b,a in probes.items():
        qb=a.get_joint_positions()
        if qb is None: continue
        qb=np.asarray(qb)
        if qb.size and np.all(np.isfinite(qb)):
            m=float(np.max(np.abs(qb)))
            if m>deform["max_rad"]: deform.update(max_rad=m, branch=b, t=_T[0])
cam=rep.create.camera(position=(2.3,-2.0,1.7), look_at=(0.28,0.02,0.80))
rp=rep.create.render_product(cam,(1400,900)); rgb=rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach(rp)
world.reset()
dof=list(robot.dof_names); arm=np.array([dof.index(n) for n in ARM])
ctrl=robot.get_articulation_controller()
HOME=np.asarray(robot.get_joint_positions())[arm].copy()   # 无碰撞 home (Phase1 验证)
print("[softstart] collision-free HOME arm =", np.round(HOME,3), flush=True)

def speeds():
    tm=0.0
    for a in probes.values():
        v=a.get_joint_velocities()
        if v is None: continue
        v=np.asarray(v)
        if v.size and not np.all(np.isfinite(v)): tm=float('inf'); break
        if v.size: tm=max(tm,float(np.max(np.abs(v))))
    rv=robot.get_joint_velocities(); rv=np.asarray(rv)[arm] if rv is not None else np.zeros(7)
    rm=float('inf') if not np.all(np.isfinite(rv)) else float(np.max(np.abs(rv)))
    return tm,rm
def snap(tag):
    for _ in range(3): world.step(render=True)
    d=rgb.get_data()
    if d is not None and len(d): Image.fromarray(np.asarray(d)[:,:,:3].astype(np.uint8)).save(f"{OUT}/{tag}.png"); print("  snap",tag,flush=True)

# 置 home 并稳定
q=robot.get_joint_positions(); q[arm]=HOME; robot.set_joint_positions(q)
ctrl.apply_action(ArticulationAction(joint_positions=HOME, joint_indices=arm))
for _ in range(20): gstep()
snap("00_home")

diverged=None
def watch(tag,step):
    global diverged
    tm,rm=speeds()
    if step%20==0: print(f"  [{tag}] t={_T[0]:5.2f}s tree={tm:8.1f} robot={rm:6.1f} rad/s",flush=True)
    # 200 rad/s = 真失控(持续增长/NaN); 30-40 的短暂尖峰是枝条滑脱回弹, 属正常
    if diverged is None and (tm>200 or rm>200):
        diverged=(tag,_T[0],tm,rm); print(f"  >>> DIVERGENCE [{tag}] t={_T[0]:.2f}s tree={tm:.1f} robot={rm:.1f}",flush=True)
        return True
    return False

# ---- 阶段1: 到达 wp0 —— 直接传送(默认) 或 home->wp0 缓慢过渡 ----
import time as _time
_wall0=_time.time()
if TELEPORT:
    print("\n== teleport: place robot AT wp0 directly, settle %.1fs =="%SETTLE_S, flush=True)
    q=robot.get_joint_positions(); q[arm]=wps[0]; robot.set_joint_positions(q)
    ctrl.apply_action(ArticulationAction(joint_positions=wps[0], joint_indices=arm))
    for k in range(int(SETTLE_S/DT)):
        gstep(); _T[0]+=DT
        sample_metrics()
        if watch("settle",k): break
else:
    print("\n== approach: HOME -> wp0 slowly (%.1fs) =="%APPROACH_S, flush=True)
    N=int(APPROACH_S/DT)
    for k in range(N):
        q=HOME+(wps[0]-HOME)*((k+1)/N)
        ctrl.apply_action(ArticulationAction(joint_positions=q, joint_indices=arm))
        gstep(); _T[0]+=DT
        if watch("approach",k): break
snap("01_at_wp0")
print("  phase-1 contacts:", sum(1 for e in events if e[3]=='branch'), "branch,", sum(1 for e in events if e[3]=='trunk'),"trunk", flush=True)

# ---- 阶段2: 低速跟随规划路径 ----
path_sim_s=0.0; ev0=len(events)
if diverged is None:
    print("\n== follow planned path slowly (vel=%.2f) =="%PATH_VEL, flush=True)
    refs=timeparam(wps[0], wps, PATH_VEL); ev0=len(events)
    shots={int(len(refs)*0.5):"02_mid"}
    max_track_err=0.0
    for step in range(len(refs)):
        qr=refs[min(step,len(refs)-1)]
        ctrl.apply_action(ArticulationAction(joint_positions=qr, joint_indices=arm))
        gstep(); _T[0]+=DT; path_sim_s+=DT
        sample_metrics()
        if step%4==0:
            sample_deform()
            qa=np.asarray(robot.get_joint_positions())[arm]
            max_track_err=max(max_track_err, float(np.linalg.norm(qa-qr)))
        if step in shots: snap(shots[step])
        if watch("path",step): break
    # 到终点后停住(hold), 稳定后取"停止时"指标
    for _ in range(60): gstep(); _T[0]+=DT; sample_metrics(); sample_deform()
    snap("03_grasp")
# 停止时 TCP-苹果距离 (机器人已停在路径终点)
_tcp_f,_q_f = tcp_state(); final_dist = float(np.linalg.norm(_tcp_f - apple_pos()))
wall_s=_time.time()-_wall0

# ---- 结果 + 需求指标 ----
nt=sum(1 for e in events if e[3]=='trunk'); nb=sum(1 for e in events if e[3]=='branch')
ubr=sorted({e[2].split('/')[0] for e in events if e[3]=='branch'})
with open(f"{OUT}/contacts.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["t","robot_link","tree_prim","class","impulse"]); w.writerows(events)
# 路径阶段接触力 (只统计 path 段): 用 50ms 滑窗"持续力"替代单步冲量/dt ——
# force = 窗内全部接触冲量之和 / 窗长, 即力传感器意义上的平均接触力峰值,
# 不再被单个求解步的分离冲量尖峰污染。
FORCE_WIN = 0.05   # s
path_events = events[ev0:]
if path_events:
    t0p = min(e[0] for e in path_events)
    # 按"接触对"(机器人 link ↔ 具体枝段)分别开窗 —— 单个接触面的力, 不再全链求和
    pair_bins = {}                       # (link, branch_prim) -> {win_idx: impulse_sum}
    for e in path_events:
        k = (e[1], e[2])
        pair_bins.setdefault(k, {})
        w = int((e[0]-t0p)/FORCE_WIN)
        pair_bins[k][w] = pair_bins[k].get(w, 0.0) + e[4]
    pair_peak = {k: max(b.values())/FORCE_WIN for k, b in pair_bins.items()}   # N
    top_pairs = sorted(pair_peak.items(), key=lambda kv: -kv[1])[:3]
    path_max_force, max_pair = top_pairs[0][1], top_pairs[0][0]
    # 参考: 全链求和口径 (旧) 与全程平均
    sum_bins = {}
    for e in path_events:
        w = int((e[0]-t0p)/FORCE_WIN)
        sum_bins[w] = sum_bins.get(w, 0.0) + e[4]
    all_links_max = max(sum_bins.values())/FORCE_WIN
    path_mean_force = sum(e[4] for e in path_events)/max(1e-9, path_sim_s)
else:
    path_max_force = path_mean_force = all_links_max = 0.0
    max_pair, top_pairs = ("-","-"), []
q=metrics["min_dist_quat"]; z=metrics["min_dist_zaxis"]
print("\n==== RESULT ====", flush=True)
print("  diverged:", diverged, flush=True)
print(f"  contacts: trunk={nt} branch={nb}  branches_touched={ubr}", flush=True)
print("\n==== PATH-FINISH METRICS ====", flush=True)
print(f"  Max contact force (path) : {path_max_force:.2f} N  (单接触对 50ms 持续峰值: {max_pair[0]} ↔ {max_pair[1]})", flush=True)
for (lk,br),fv in top_pairs[1:]:
    print(f"      next pairs           : {fv:.2f} N  ({lk} ↔ {br})", flush=True)
print(f"      (reference)          : all-links-sum peak {all_links_max:.2f} N; mean over path {path_mean_force:.2f} N", flush=True)
print(f"  Min TCP-apple distance   : {metrics['min_dist']*1000:.1f} mm  (at sim t={metrics['min_dist_t']:.2f}s)", flush=True)
print(f"  Final TCP-apple distance : {final_dist*1000:.1f} mm  (robot stopped at path end)", flush=True)
print(f"  Max branch deformation   : {np.degrees(deform['max_rad']):.1f} deg  (branch{deform['branch']}, at sim t={deform['t']:.2f}s)", flush=True)
print(f"  Execution time           : path {path_sim_s:.2f} s (sim)  |  total sim {_T[0]:.2f} s  |  wall {wall_s:.1f} s", flush=True)
print(f"  Hard-obstacle (trunk) hit: {nt}  ({'FAIL' if nt else 'none — SUCCESS'})", flush=True)
try: print(f"  Max joint tracking error : {max_track_err:.4f} rad  (大 = 真实力矩下被枝条顶住/停滞)", flush=True)
except NameError: pass
if q is not None:
    print(f"  TCP orientation @closest : quat(wxyz)=[{q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}]", flush=True)
    print(f"                             approach axis(hand +z)=[{z[0]:+.3f} {z[1]:+.3f} {z[2]:+.3f}]"
          f"  align-to-apple={metrics['min_dist_align_deg']:.1f} deg", flush=True)
with open(f"{OUT}/metrics.txt","w") as f:
    f.write(f"path_max_contact_force_N={path_max_force:.3f}\n"
            f"final_tcp_apple_mm={final_dist*1000:.2f}\n"
            f"max_branch_deform_deg={np.degrees(deform['max_rad']):.2f}\nmax_deform_branch={deform['branch']}\n"
            f"min_tcp_apple_mm={metrics['min_dist']*1000:.2f}\n"
            f"path_sim_s={path_sim_s:.2f}\ntotal_sim_s={_T[0]:.2f}\nwall_s={wall_s:.1f}\ntrunk_contacts={nt}\n"
            f"tcp_quat_wxyz={list(map(float,q)) if q is not None else None}\n"
            f"approach_axis={list(map(float,z)) if z is not None else None}\n"
            f"align_deg={metrics['min_dist_align_deg']}\n")
print("  [softstart] done ->", OUT, flush=True)
print("  [gui] 进入循环回放 — 关闭窗口退出。", flush=True)
try:
    while sim.is_running() and diverged is None:
        # 回到路径起点 (瞬移 + 0.5s 稳定, 枝条回弹)
        q = robot.get_joint_positions(); q[arm] = wps[0]; robot.set_joint_positions(q)
        ctrl.apply_action(ArticulationAction(joint_positions=wps[0], joint_indices=arm))
        for _ in range(int(0.5/DT)):
            if not sim.is_running(): break
            gstep()
        # 重走路径
        for step in range(len(refs)):
            if not sim.is_running(): break
            qr = refs[min(step, len(refs)-1)]
            ctrl.apply_action(ArticulationAction(joint_positions=qr, joint_indices=arm))
            gstep()
        # 抓取姿态停留 ~1s
        for _ in range(int(1.0/DT)):
            if not sim.is_running(): break
            gstep()
except KeyboardInterrupt:
    pass
sim.close()
