"""Bullet3 场景 + 软枝编号可视化。
默认离屏渲染 PNG (标签用投影+PIL 叠加); --gui 开交互窗口 (标签 = debug text, 可旋转)。"""
import json, math, sys, numpy as np
import pybullet as p

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--scene", default="/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.claude/worktrees/experiment-scene-visualization-e0aaf2/benchmark_results_with_A_pengate_uni_nav_pen_metric/scene_layout.json")
_ap.add_argument("--apple-idx", type=int, default=4); _ap.add_argument("--out-prefix", default="bullet_branches"); _ap.add_argument("--gui", action="store_true")
_args,_ = _ap.parse_known_args()
SCENE = _args.scene
GUI = _args.gui
d = json.load(open(SCENE))
ap = np.array(d['apple']['center'])

# 链重建 (t0==0 开新枝)
hard, soft, cur = [], [], None
for c in d['cylinders']:
    if c['p0'] == c['p1']:
        hard.append(('sphere', np.array(c['p0']), c['radius'])); continue
    seg = (np.array(c['p0']), np.array(c['p1']), c['radius'])
    if c['class'] == 'hard': hard.append(('cyl',)+seg); continue
    if c.get('t0',0)==0: cur=[]; soft.append(cur)
    cur.append(seg)
APPLE_IDX = _args.apple_idx

p.connect(p.GUI if GUI else p.DIRECT)
if GUI: p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

def add_cyl(p0, p1, r, rgba):
    v = p1-p0; L = float(np.linalg.norm(v)); a = v/L
    mid = 0.5*(p0+p1)
    z = np.array([0,0,1.0]); c = float(np.dot(z,a))
    if c > 0.9999: q = [0,0,0,1]
    elif c < -0.9999: q = [1,0,0,0]
    else:
        ax = np.cross(z,a); ax/=np.linalg.norm(ax); th = math.acos(c)
        q = [ax[0]*math.sin(th/2), ax[1]*math.sin(th/2), ax[2]*math.sin(th/2), math.cos(th/2)]
    vs = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=L, rgbaColor=rgba)
    p.createMultiBody(baseVisualShapeIndex=vs, basePosition=mid.tolist(), baseOrientation=q)

for h in hard:
    if h[0]=='sphere':
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=h[2], rgbaColor=[0.35,0.35,0.35,1])
        p.createMultiBody(baseVisualShapeIndex=vs, basePosition=h[1].tolist())
    else: add_cyl(h[1],h[2],h[3],[0.45,0.28,0.12,1])
labels=[]
for i,b in enumerate(soft):
    col = [0.85,0.1,0.1,1] if i==APPLE_IDX else [0.15,0.55,0.15,1]
    for p0,p1,r in b: add_cyl(p0,p1,r,col)
    tip=b[-1][1]; root=b[0][0]; dirv=(tip-root)/max(1e-9,np.linalg.norm(tip-root))
    labels.append((i, tip + 0.05*dirv + np.array([0,0,0.03])))
vs = p.createVisualShape(p.GEOM_SPHERE, radius=d['apple']['radius'], rgbaColor=[1,0,0,1])
p.createMultiBody(baseVisualShapeIndex=vs, basePosition=ap.tolist())
vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05,0.05,0.02], rgbaColor=[0.2,0.3,0.9,1])
p.createMultiBody(baseVisualShapeIndex=vs, basePosition=[0,0,0.01])

if GUI:
    for i,pos in labels:
        p.addUserDebugText(str(i), pos.tolist(), textColorRGB=[0.6,0,0] if i==APPLE_IDX else [0,0.35,0],
                           textSize=1.6)
    p.addUserDebugText("apple", (ap+[0,0,0.08]).tolist(), textColorRGB=[1,0,0], textSize=1.4)
    p.addUserDebugText("robot base", [0,0,0.1], textColorRGB=[0.2,0.3,0.9], textSize=1.2)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-20,
                                 cameraTargetPosition=[0.42,0.05,1.0])
    print("GUI up — 关窗口退出")
    import time
    while p.isConnected(): time.sleep(0.2)
    sys.exit()

# ---- 离屏: 渲染 + PIL 标签叠加 ----
from PIL import Image, ImageDraw, ImageFont
W,H = 1600,1200
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
except: font = ImageFont.load_default()
views = [("side",  dict(cameraTargetPosition=[0.42,0.05,1.0], distance=1.9, yaw=140, pitch=-12, roll=0, upAxisIndex=2)),
         ("robotview", dict(cameraTargetPosition=[0.42,0.05,1.0], distance=1.9, yaw=230, pitch=-12, roll=0, upAxisIndex=2)),
         ("top",   dict(cameraTargetPosition=[0.42,0.05,1.1], distance=1.6, yaw=180, pitch=-85, roll=0, upAxisIndex=2))]
for name, v in views:
    vm = p.computeViewMatrixFromYawPitchRoll(**v)
    pm = p.computeProjectionMatrixFOV(fov=55, aspect=W/H, nearVal=0.05, farVal=10)
    img = p.getCameraImage(W,H, viewMatrix=vm, projectionMatrix=pm, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(img[2], (H,W,4))[:,:,:3].astype(np.uint8)
    im = Image.fromarray(rgb); dr = ImageDraw.Draw(im)
    V = np.array(vm).reshape(4,4,order='F'); P = np.array(pm).reshape(4,4,order='F')
    def proj(x):
        h = P@V@np.array([x[0],x[1],x[2],1.0])
        if h[3]<=0: return None
        ndc = h[:3]/h[3]
        return ((ndc[0]+1)*0.5*W, (1-ndc[1])*0.5*H)
    for i,pos in labels:
        s = proj(pos)
        if s is None: continue
        x,y = s; col = (150,0,0) if i==APPLE_IDX else (0,90,0)
        r0 = 24
        dr.ellipse([x-r0,y-r0,x+r0,y+r0], fill=(255,255,255), outline=col, width=3)
        tb = dr.textbbox((0,0), str(i), font=font)
        dr.text((x-(tb[2]-tb[0])/2, y-(tb[3]-tb[1])/2-tb[1]), str(i), fill=col, font=font)
    s = proj(ap+[0,0,0.09])
    if s: dr.text((s[0]-35, s[1]), "apple", fill=(255,30,30), font=font)
    s = proj(np.array([0,0,0.12]))
    if s: dr.text((s[0]-70, s[1]), "robot base", fill=(40,70,230), font=font)
    im.save(f"/tmp/claude-1000/-home-zhongyanzhang-zhongyan-zhang-Apple-picking--claude-worktrees-experiment-scene-visualization-102f76/203e51d4-57dc-4433-a3de-888b5ec35c61/scratchpad/{_args.out_prefix}_{name}.png")
    print("saved", name)
