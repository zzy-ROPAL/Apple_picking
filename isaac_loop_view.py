"""在 Isaac Sim 里循环回放 GraspRRT 路径 (软起步 + 驯服驱动)。
默认 GUI 窗口, 循环遍历给定的多条路径; 每条之间 world.reset() 让树复位。
用法(在自己的终端里, 带 GUI):
  cd ~/isaacsim && ./python.sh <此脚本> \
      --usd  ~/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_tamed.usd \
      --paths <path_02.csv> <path_01.csv>            # 循环这两条
  加 --loops 3 只循环 3 轮; 加 --headless 无窗口(仅测试)。Ctrl-C 或关窗退出。
"""
import argparse, numpy as np
ap = argparse.ArgumentParser()
DEF_USD = "/home/zhongyanzhang/Desktop/zhongyan_zhang/7-23/scene_with_tree_apple_tamed.usd"
DEF_DIR = "/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.claude/worktrees/experiment-scene-visualization-e0aaf2/benchmark_results_with_A_pengate_uni_nav_pen_metric"
ap.add_argument("--usd", default=DEF_USD)
ap.add_argument("--paths", nargs="+",
                default=[DEF_DIR+"/grasp_path_02.csv", DEF_DIR+"/grasp_path_01.csv"])
ap.add_argument("--robot-prim", default="/Franka")
ap.add_argument("--loops", type=int, default=0, help="0 = 无限循环")
ap.add_argument("--dt", type=float, default=1.0/480.0)
ap.add_argument("--approach", type=float, default=3.0, help="(--no-teleport 时) home->wp0 过渡时长 s")
ap.add_argument("--no-teleport", action="store_true",
                help="改回 home->wp0 运动式到位 (默认: 直接传送到 wp0 + settle, 不压弯沿途枝条)")
ap.add_argument("--settle", type=float, default=2.0, help="传送后稳定时长 s")
ap.add_argument("--path-vel", type=float, default=0.15)
ap.add_argument("--hold", type=float, default=1.0)
ap.add_argument("--render-every", type=int, default=8, help="每 N 物理步渲染一次(GUI 流畅度)")
ap.add_argument("--headless", action="store_true")
args = ap.parse_args()

from isaacsim import SimulationApp
sim = SimulationApp({"headless": args.headless})
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.types import ArticulationAction

ARM = [f"panda_joint{i}" for i in range(1,8)]
VMAX = np.array([2.175,2.175,2.175,2.175,2.61,2.61,2.61])
def load(p): return np.array([[float(x) for x in l.split(",")] for l in open(p) if l.strip()])
def timeparam(wpts, vel, dt):
    refs=[wpts[0]]
    for a,b in zip(wpts[:-1],wpts[1:]):
        n=max(1,int(np.ceil(float(np.max(np.abs(b-a)/(VMAX*vel)))/dt)))
        for k in range(1,n+1): refs.append(a+(b-a)*(k/n))
    return np.array(refs)

open_stage(args.usd)
world=World(stage_units_in_meters=1.0, physics_dt=args.dt, rendering_dt=args.dt)
robot=Articulation(args.robot_prim, name="franka"); world.scene.add(robot)
world.reset()
dof=list(robot.dof_names); arm=np.array([dof.index(n) for n in ARM])
ctrl=robot.get_articulation_controller()
HOME=np.asarray(robot.get_joint_positions())[arm].copy()
paths=[load(p) for p in args.paths]
names=[p.split("/")[-1] for p in args.paths]
print(f"[loop] USD={args.usd}\n[loop] paths={names}  dt={args.dt:.5f}  loops={'inf' if args.loops==0 else args.loops}", flush=True)

def act(q): return ArticulationAction(joint_positions=np.asarray(q), joint_indices=arm)
def go(qA, qB, secs):
    N=max(1,int(secs/args.dt))
    for k in range(N):
        if not sim.is_running(): return False
        ctrl.apply_action(act(qA+(qB-qA)*((k+1)/N)))
        world.step(render=(not args.headless and k%args.render_every==0))
    return True
def follow(refs):
    for s in range(len(refs)):
        if not sim.is_running(): return False
        ctrl.apply_action(act(refs[min(s,len(refs)-1)]))
        world.step(render=(not args.headless and s%args.render_every==0))
    for _ in range(int(args.hold/args.dt)):
        if not sim.is_running(): return False
        world.step(render=(not args.headless))
    return True

loop=0
while sim.is_running():
    for wps,name in zip(paths,names):
        if not sim.is_running(): break
        print(f"[loop {loop+1}] replay {name}", flush=True)
        world.reset()
        if args.no_teleport:
            q=robot.get_joint_positions(); q[arm]=HOME; robot.set_joint_positions(q)
            for _ in range(10): world.step(render=not args.headless)
            if not go(HOME, wps[0], args.approach): break      # home -> wp0 缓慢
        else:
            # 直接传送到 wp0(不经过途中运动, 不压弯枝条), settle 让重叠枝条缓缓滑出
            q=robot.get_joint_positions(); q[arm]=wps[0]; robot.set_joint_positions(q)
            ctrl.apply_action(act(wps[0]))
            for k in range(int(args.settle/args.dt)):
                if not sim.is_running(): break
                world.step(render=(not args.headless and k%args.render_every==0))
        if not follow(timeparam(wps, args.path_vel, args.dt)): break  # 跟随路径
    loop+=1
    if args.loops and loop>=args.loops: break
print("[loop] done", flush=True)
sim.close()
