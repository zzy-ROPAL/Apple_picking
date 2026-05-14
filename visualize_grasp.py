import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import math

def main():
    # 1. 连接 PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 2. 场景布局 (与 C++ SceneConfig 保持一致)
    p.loadURDF("plane.urdf")
    
    # 树干 (Trunk) - 退到后面
    trunk_radius = 0.08
    trunk_height = 1.5
    trunk_pos = [0.7, 0.0, 0.75]
    trunk_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=trunk_radius, height=trunk_height)
    trunk_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=trunk_radius, length=trunk_height, rgbaColor=[0.55, 0.27, 0.07, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=trunk_col, baseVisualShapeIndex=trunk_vis, basePosition=trunk_pos)

    # 树冠 (Branches) 
    branches = [
        {"h": 0.65, "angle": 3.1415, "length": 0.35, "radius": 0.03},
        {"h": 0.65, "angle": 2.6, "length": 0.3, "radius": 0.03},
        {"h": 0.65, "angle": 3.68, "length": 0.3, "radius": 0.03},
        {"h": 0.80, "angle": 2.7, "length": 0.35, "radius": 0.03},
        {"h": 0.80, "angle": 3.58, "length": 0.35, "radius": 0.03},
        {"h": 0.80, "angle": 2.1, "length": 0.3, "radius": 0.03},
        {"h": 0.80, "angle": 4.18, "length": 0.3, "radius": 0.03},
        {"h": 0.95, "angle": 3.1415, "length": 0.4, "radius": 0.03},
        {"h": 0.95, "angle": 2.6, "length": 0.3, "radius": 0.03},
        {"h": 0.95, "angle": 3.68, "length": 0.3, "radius": 0.03}
    ]
    
    for br in branches:
        angle = br["angle"]
        branch_length = br["length"]
        branch_radius = br["radius"]
        
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        pos = [
            trunk_pos[0] + (trunk_radius + branch_length/2.0) * cos_a,
            trunk_pos[1] + (trunk_radius + branch_length/2.0) * sin_a,
            br["h"] 
        ]
        
        q_y = p.getQuaternionFromEuler([0, math.pi/2, 0])
        q_z = p.getQuaternionFromEuler([0, 0, angle])
        _, orn = p.multiplyTransforms([0,0,0], q_z, [0,0,0], q_y)
        
        b_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=branch_radius, height=branch_length)
        b_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=branch_radius, length=branch_length, rgbaColor=[0.2, 0.6, 0.2, 0.6])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=b_col, baseVisualShapeIndex=b_vis, basePosition=pos, baseOrientation=orn)

    # 苹果正上方的硬质障碍物
    # 与 C++ 保持一致：addObstacleSphere(radius=0.08, center=(0.45,0,0.91))
    overhead_center = [0.45, 0.0, 0.91]
    overhead_radius = 0.08
    overhead_col = p.createCollisionShape(p.GEOM_SPHERE, radius=overhead_radius)
    overhead_vis = p.createVisualShape(p.GEOM_SPHERE, radius=overhead_radius,
                                       rgbaColor=[0.55, 0.27, 0.07, 0.90])  # 树干色
    p.createMultiBody(baseMass=0,
                      baseCollisionShapeIndex=overhead_col,
                      baseVisualShapeIndex=overhead_vis,
                      basePosition=overhead_center)

    # 苹果 (Apple)
    apple_pos = [0.45, 0.0, 0.80]
    apple_radius = 0.04
    apple_col = p.createCollisionShape(p.GEOM_SPHERE, radius=apple_radius)
    apple_vis = p.createVisualShape(p.GEOM_SPHERE, radius=apple_radius, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=apple_col, baseVisualShapeIndex=apple_vis, basePosition=apple_pos)

    # 3. 导入 Panda 机器人并垫高 0.4m
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.4], useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    revolute_joint_indices = [0, 1, 2, 3, 4, 5, 6]

    # 4. 读取路径 (支持命令行参数 --path)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="grasp_path.csv", help="CSV trajectory file to load")
    # 忽略不属于 argparse 的未知参数 (在 ExampleBrowser 环境中运行)
    args, _ = parser.parse_known_args()

    path_file = args.path
    
    # 路径解析加强：如果在当前目录找不到，尝试在同名文件夹的 ../ 寻找，或者 build 子目录寻寻找
    candidates = [
        path_file,
        os.path.join("build", path_file),
        os.path.join("..", path_file)
    ]
    
    found = False
    for cand in candidates:
        if os.path.exists(cand):
            path_file = cand
            found = True
            break
            
    if not found:
        print(f"Error: Could not find trajectory file matching: {args.path}")
        print(f"Searched: {candidates}")
        return

    print(f"Loading path from {path_file}...")
    waypoints = []
    try:
        with open(path_file, 'r') as f:
            for line in f:
                if line.strip():
                    wp = [float(x) for x in line.strip().split(',')]
                    waypoints.append(wp)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    if not waypoints:
        print("Error: Waypoints list is empty.")
        return
        
    print(f"Loaded {len(waypoints)} waypoints.")

    # 5. 播放动画
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0.5, 0.1, 0.5])
    
    # 记录我们关心的连杆名（Panda URDF 中的 link 名称）
    num_joints_total = p.getNumJoints(robot_id)
    link_name_to_idx = {}
    for i in range(num_joints_total):
        info = p.getJointInfo(robot_id, i)
        link_name_to_idx[info[12].decode()] = i  # index 12 = child link name

    # 找到上方障碍物的 id (它是第3个 createMultiBody，编号从0开始)
    # 更稳健的方法：记录我们创建时的 id
    # 这里用我们上面创建的 overhead_body_id
    # （如果上面没保存，就扫描所有 body 找最近的）
    overhead_body = None
    for body_id in range(p.getNumBodies()):
        pos, _ = p.getBasePositionAndOrientation(body_id)
        if abs(pos[0]-0.45) < 0.01 and abs(pos[2]-0.91) < 0.01:
            overhead_body = body_id
            print(f"  Found overhead obstacle at body_id={overhead_body}")
            break

    played_once = False
    while True:
        print("Starting playback...")
        # Reset to start position first
        if waypoints:
            for i, joint_idx in enumerate(revolute_joint_indices):
                p.resetJointState(robot_id, joint_idx, waypoints[0][i])
            time.sleep(0.5)

        # Interpolate between waypoints
        min_dist_overall = float('inf')
        worst_link = "?"
        for w_idx in range(len(waypoints) - 1):
            start_wp = np.array(waypoints[w_idx])
            end_wp   = np.array(waypoints[w_idx+1])
            steps = 4 
            for step in range(steps):
                alpha = step / float(steps)
                interp_wp = (1 - alpha) * start_wp + alpha * end_wp
                for i, joint_idx in enumerate(revolute_joint_indices):
                    p.resetJointState(robot_id, joint_idx, interp_wp[i])
                p.stepSimulation()

                # --- 碰撞诊断 ---
                if overhead_body is not None and not played_once:
                    for link_idx in range(-1, num_joints_total):
                        closest = p.getClosestPoints(
                            bodyA=robot_id, bodyB=overhead_body,
                            distance=0.20,            # 查询 20cm 内的最近点
                            linkIndexA=link_idx,
                            linkIndexB=-1)
                        for pt in closest:
                            dist = pt[8]              # contactDistance
                            if dist < min_dist_overall:
                                min_dist_overall = dist
                                jinfo = p.getJointInfo(robot_id, link_idx) if link_idx >= 0 else None
                                worst_link = jinfo[12].decode() if jinfo else "base"
                # ----------------

                time.sleep(0.01)

        if not played_once:
            print(f"\n[诊断] 与上方障碍物最近距离: {min_dist_overall*100:.1f} cm")
            print(f"[诊断] 最近连杆: {worst_link}")
            if min_dist_overall < 0:
                print(f"[诊断] *** 几何穿透! 深度 {-min_dist_overall*100:.1f} cm ***")
            elif min_dist_overall < 0.02:
                print(f"[诊断] *** 极近距离, 视觉几乎接触 ***")
            played_once = True

        time.sleep(2)

if __name__ == "__main__":
    main()
