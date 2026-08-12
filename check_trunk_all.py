import json, math, glob, numpy as np, pybullet as p
W="/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.claude/worktrees/experiment-scene-visualization-e0aaf2"
D=W+"/benchmark_results_with_A_pengate_uni_nav_ab12_zr0.35_y120root_capfix_5x10"
URDF="/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.venv/lib/python3.12/site-packages/pybullet_data/franka_panda/panda.urdf"
p.connect(p.DIRECT)
robot=p.loadURDF(URDF,[0,0,0.4],useFixedBase=True)
arm=[j for j in range(p.getNumJoints(robot)) if p.getJointInfo(robot,j)[2]==p.JOINT_REVOLUTE][:7]
d=json.load(open(D+"/scene_layout.json"))
trunks=[]
def add_cyl(p0,p1,r):
    p0,p1=np.array(p0),np.array(p1); v=p1-p0; L=float(np.linalg.norm(v)); a=v/L
    mid=0.5*(p0+p1); z=np.array([0,0,1.0]); c=float(np.dot(z,a))
    if c>0.9999: q=[0,0,0,1]
    elif c<-0.9999: q=[1,0,0,0]
    else:
        ax=np.cross(z,a); ax/=np.linalg.norm(ax); th=math.acos(c)
        q=[ax[0]*math.sin(th/2),ax[1]*math.sin(th/2),ax[2]*math.sin(th/2),math.cos(th/2)]
    cs=p.createCollisionShape(p.GEOM_CYLINDER,radius=r,height=L)
    trunks.append(p.createMultiBody(baseCollisionShapeIndex=cs,basePosition=mid.tolist(),baseOrientation=q))
for c in d['cylinders']:
    if c['class']=='hard' and c['p0']!=c['p1']: add_cyl(c['p0'],c['p1'],c['radius'])
gmin=1e9; bad=0
for f in sorted(glob.glob(D+"/grasp_path_*.csv")):
    wps=np.array([[float(x) for x in l.split(',')] for l in open(f) if l.strip()])
    dmin=1e9
    for i in range(len(wps)-1):
        for s in range(8):
            q=wps[i]+(wps[i+1]-wps[i])*(s/8)
            for j,v in zip(arm,q): p.resetJointState(robot,j,v)
            for tb in trunks:
                for pt in p.getClosestPoints(robot,tb,distance=0.03):
                    dmin=min(dmin,pt[8])
    tag="OK" if dmin>0 else "PENETRATION!"
    if dmin<=0: bad+=1
    gmin=min(gmin,dmin)
    print(f"{f.split('/')[-1]}: min trunk clearance {dmin*1000:6.1f} mm  {tag}")
print(f"\npaths with true trunk penetration: {bad}   global min clearance: {gmin*1000:.1f} mm")
