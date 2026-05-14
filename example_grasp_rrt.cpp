/**
 * example_grasp_rrt.cpp — GraspRRT 示例: Panda 机器人抓取苹果 (穿过树枝)
 *
 * 场景 (来自 tree_model.py):
 *   - Panda 7-DOF 机器人手臂 (Pinocchio FK/Jacobian)
 *   - 树干 (不可穿透圆柱体)
 *   - 树枝 (可穿透, 但在此 example 中简化为障碍物)
 *   - 苹果 (球形抓取目标)
 *
 * 编译:
 *   cd grasp_rrt_framework/build
 *   cmake .. && make -j$(nproc)
 *   ./example_grasp_rrt [path_to_panda.urdf]
 *
 * 默认 URDF 路径搜索:
 *   1. 命令行参数
 *   2. 环境变量 PANDA_URDF
 *   3. pybullet_data 常见路径
 */

#include <iostream>
#include <memory>
#include <cmath>
#include <string>
#include <fstream>

// GraspRRT 框架
#include "types.h"
#include "cspace_sampled.h"
#include "cspace_tree.h"
#include "grasp_rrt.h"

// 具体实现
#include "panda_pinocchio.h"
#include "bullet_collision_checker.h"
#include "simple_end_effector.h"
#include "simple_grasp_quality.h"
#include "simple_target_object.h"
#include "soft_cost_evaluator.h"
#include "sdf_cost_evaluator.h"
#include "sdf_traj_optimizer.h"
#include <chrono>
#include <random>

using namespace grasp_rrt;

// ============================================================================
// URDF 路径查找
// ============================================================================

std::string findPandaUrdf(int argc, char* argv[]) {
    // 1. 命令行参数
    if (argc > 1) {
        std::string path = argv[1];
        std::ifstream f(path);
        if (f.good()) return path;
        std::cerr << "WARNING: URDF not found at: " << path << "\n";
    }

    // 2. 环境变量
    const char* env = std::getenv("PANDA_URDF");
    if (env) {
        std::ifstream f(env);
        if (f.good()) return std::string(env);
    }

    // 3. 常见路径
    std::vector<std::string> search_paths = {
        // pybullet_data (pip install)
        "/usr/local/lib/python3.10/dist-packages/pybullet_data/franka_panda/panda.urdf",
        "/usr/local/lib/python3.8/dist-packages/pybullet_data/franka_panda/panda.urdf",
        // conda
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/.local/lib/python3.10/site-packages/pybullet_data/franka_panda/panda.urdf",
        // ROS
        "/opt/ros/humble/share/franka_description/robots/panda/panda.urdf",
    };

    for (const auto& p : search_paths) {
        std::ifstream f(p);
        if (f.good()) {
            std::cout << "Found URDF at: " << p << "\n";
            return p;
        }
    }

    std::cerr << "ERROR: Could not find panda.urdf!\n"
              << "Usage: ./example_grasp_rrt <path_to_panda.urdf>\n"
              << "Or set PANDA_URDF environment variable.\n"
              << "Tip: run `python3 -c \"import pybullet_data; print(pybullet_data.getDataPath())\"`\n";
    return "";
}

// ============================================================================
// 场景参数 (对应 tree_model.py)
// ============================================================================

struct SceneConfig {
    // 树干中心位置 (往后退一点，给树冠留出空间)
    Vector3f trunk_center = Vector3f(0.7f, 0.0f, 0.75f);
    float trunk_height = 1.5f;
    float trunk_radius = 0.08f;

    // 苹果正上方的硬质障碍物 (树干属性, 不可穿过)
    struct OverheadObstacle {
        Vector3f center  = Vector3f(0.45f, 0.0f, 0.91f); // 苹果(0.45,0,0.80)正上方
        float    radius  = 0.08f;   // 遮挡半径 8cm，略大于苹果(4cm)，强制机器人从侧方进入
        float    half_h  = 0.04f;   // 半高 4cm，总高 8cm (底部 z=0.87，距苹果顶部 ~3cm)
    } overhead;

    // 树枝(叶子)参数: 添加密集的树冠
    // 树枝长度缩短，方向朝向机器人 (angle ≈ PI)
    struct BranchInfo {
        float h;
        float angle;
        float length = 0.35f;
        float radius = 0.03f;
    };
    std::vector<BranchInfo> branches = {
        // 下层树叶 (阻挡手腕向下)
        {0.65f, 3.1415f, 0.35f, 0.03f}, {0.65f, 2.6f, 0.3f, 0.03f}, {0.65f, 3.68f, 0.3f, 0.03f},
        // 中层树叶，将苹果包裹在中间
        {0.80f, 2.7f, 0.35f, 0.03f}, {0.80f, 3.58f, 0.35f, 0.03f}, {0.80f, 2.1f, 0.3f, 0.03f}, {0.80f, 4.18f, 0.3f, 0.03f},
        // 上层树叶 (压迫上方空间)
        {0.95f, 3.1415f, 0.4f, 0.03f}, {0.95f, 2.6f, 0.3f, 0.03f}, {0.95f, 3.68f, 0.3f, 0.03f}
    };

    // 苹果: 放在密集的树枝中间
    float apple_radius = 0.04f;
    Vector3f apple_pos = Vector3f(0.45f, 0.0f, 0.80f);

    // Panda home pose
    Config home_pose = Config(7);
    SceneConfig() {
        home_pose << 0.0f, -M_PI/4, 0.0f, -3*M_PI/4, 0.0f, M_PI/2, M_PI/4;
    }
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "======================================================\n"
              << "  GraspRRT Example: Panda Picking Apple Through Tree\n"
              << "======================================================\n\n";

    // ---- 1. 查找 URDF ----
    std::string urdf_path = findPandaUrdf(argc, argv);
    if (urdf_path.empty()) return 1;

    // ---- 2. 场景配置 ----
    SceneConfig scene;

    // ---- 3. 加载 Panda 机器人 (Pinocchio) ----
    std::cout << "[Setup] Loading Panda robot...\n";
    auto robot = std::make_shared<PandaPinocchioRobot>(urdf_path, "panda_hand");

    // DEBUG: 打印连杆之间的相对位置，确定偏移方向
    robot->setJointValues(Config::Zero(7));
    std::vector<std::string> debug_links = {"panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_hand"};
    std::cout << "\n[Debug] Link Relative Offsets (Checking for sphere placement):\n";
    for(size_t i=1; i<debug_links.size(); ++i) {
        Matrix4f p1 = robot->getFramePose(debug_links[i-1]);
        Matrix4f p2 = robot->getFramePose(debug_links[i]);
        Vector3f rel = p1.block<3,3>(0,0).transpose() * (p2.block<3,1>(0,3) - p1.block<3,1>(0,3));
        std::cout << "  " << debug_links[i-1] << " -> " << debug_links[i] << ": [" << rel.transpose() << "]\n";
    }
    std::cout << "\n";

    // 获取关节极限
    auto [joint_lower, joint_upper] = robot->getJointLimits();
    std::cout << "[Setup] Joint limits:\n"
              << "  Lower: " << joint_lower.transpose() << "\n"
              << "  Upper: " << joint_upper.transpose() << "\n";

    // ---- 4. 创建配置空间 ----
    std::cout << "[Setup] Creating C-Space (7-DOF)...\n";
    auto collision_checker_ptr = std::make_shared<BulletCollisionChecker>(robot);
    auto cspace = std::make_shared<CSpaceSampled>(7, collision_checker_ptr);
    cspace->setBoundaries(joint_lower, joint_upper);

    // ---- 5. 设置碰撞场景 (Bullet3) ----
    std::cout << "[Setup] Building collision scene...\n";

    // 5a. 树干 (硬障碍物 - 必须避开)
    Matrix4f trunk_pose = Matrix4f::Identity();
    trunk_pose.block<3,1>(0, 3) = scene.trunk_center;
    collision_checker_ptr->addObstacleCylinder(scene.trunk_radius, scene.trunk_height / 2.0f, trunk_pose);

    // 5a-2. 苹果正上方的硬质障碍物 (不可穿透球体)
    collision_checker_ptr->addObstacleSphere(scene.overhead.radius, scene.overhead.center);
    std::cout << "  Overhead hard obstacle (SPHERE): center=["
              << scene.overhead.center.transpose() << "]"
              << "  r=" << scene.overhead.radius << "m\n";

    // 5b. 树冠/树叶 (柔性/可穿透区域 - 不加入硬碰撞检测)
    std::cout << "  (Note: " << scene.branches.size() 
              << " canopy branches exist in scene but act as permeable leaves, so they are not added to hard collision checker.)\n";

    // 5c. 机器人碰撞体 —— 胶囊体填充法 (Capsule Approximation)
    // 胶囊体 = 圆柱 + 两端半球，完美贴合连杆形状，无间隙死角。
    // 每条连杆只需 1 个胶囊体，碰撞速度与球体相当。
    //
    // 参数说明:
    //   center_z: 胶囊中心在连杆局部坐标系中的 Z 偏移
    //   half_h:   胶囊圆柱部分的半高 (两端半球不算在内)
    //   radius:   胶囊半径
    struct LinkCapsule {
        std::string name;
        float radius;
        float center_z;   // 胶囊中心相对于连杆原点的 Z 偏移
        float half_h;     // 圆柱半高
    };

    // 基于 Panda URDF 几何的胶囊参数 (+Z 为连杆伸展方向)
    // 注意：panda_link7（法兰）由下方的锥台体单独建模，不在此列表中
    std::vector<LinkCapsule> capsule_structure = {
        {"panda_link0", 0.10f, 0.05f,  0.02f},
        {"panda_link1", 0.08f, 0.15f,  0.12f},
        {"panda_link2", 0.07f, 0.00f,  0.02f},
        {"panda_link3", 0.06f, 0.15f,  0.12f},
        {"panda_link4", 0.06f, 0.05f,  0.03f},
        // Link5 在 5cm 设定下受到 3cm 穿透，说明电机外壳的真实等效半径约为 8cm
        {"panda_link5", 0.08f, 0.15f,  0.13f},
        {"panda_link6", 0.06f, 0.00f,  0.02f},
        // panda_link7 单独建模（见下方锥台体）
        // panda_hand 单独建模（由于夹爪有手指且很宽，用包围盒建模）
    };

    for (const auto& lc : capsule_structure) {
        float cz = lc.center_z;
        auto fk_func = [robot, lc, cz](const Config& config) -> Matrix4f {
            robot->setJointValues(config);
            Matrix4f pose = robot->getFramePose(lc.name);
            pose.block<3,1>(0,3) += pose.block<3,3>(0,0) * Vector3f(0, 0, cz);
            return pose;
        };
        collision_checker_ptr->addRobotLinkCapsule(lc.radius, lc.half_h, fk_func, lc.name);
    }

    // panda_link7 (法兰) 特殊建模：梯形截面锥台体
    //   底面(靠 link6 侧, z=0)  : r = 0.040m  (Panda 法兰直径约 80mm)
    //   顶面(靠 hand 侧, z=0.035): r = 0.030m  (收窄端约 55-60mm)
    //   锥台中心在 z = 0.0175m 处
    {
        float flange_center_z = 0.0175f;
        auto fk_flange = [robot, flange_center_z](const Config& config) -> Matrix4f {
            robot->setJointValues(config);
            Matrix4f pose = robot->getFramePose("panda_link7");
            pose.block<3,1>(0,3) += pose.block<3,3>(0,0) * Vector3f(0, 0, flange_center_z);
            return pose;
        };
        // 真实的 Panda 法兰：底端直径约 88mm (r=44mm), 顶端直径约 60mm (r=30mm), 总厚度约 36mm
        collision_checker_ptr->addRobotLinkFrustum(0.044f, 0.030f, 0.0180f,
                                                    fk_flange, "panda_link7_frustum");
    }

    // panda_link7 过渡段胶囊：填补锥台顶端(z≈0.036m)与 panda_hand 胶囊底端(z≈0.057m)之间的空隙
    // 这段腕部连接轴没有任何碰撞体覆盖，是视觉碰撞的根本原因。
    {
        float shaft_center_z = 0.053f; // 胶囊中心：z = 0.036～0.070m 的中点
        auto fk_shaft = [robot, shaft_center_z](const Config& config) -> Matrix4f {
            robot->setJointValues(config);
            Matrix4f pose = robot->getFramePose("panda_link7");
            pose.block<3,1>(0,3) += pose.block<3,3>(0,0) * Vector3f(0, 0, shaft_center_z);
            return pose;
        };
        // 过渡轴半径约 25mm（细轴），覆盖从 z≈0.036 到 z≈0.070m
        collision_checker_ptr->addRobotLinkCapsule(0.025f, 0.017f, fk_shaft, "panda_link7_shaft");
    }

    // panda_hand (夹爪) 特殊建模：Box (长方体) 包围盒
    // 诊断表明手部严重穿透，原因是夹爪有宽大的基座和完全张开的指头，单一胶囊无法覆盖 Y 轴上的宽度。
    // 我们用一个 8cm x 22cm x 12cm 的 Box 把整个夹爪连同两根指头完全包住。
    {
        float hand_center_z = 0.05f; // Box 中心
        auto fk_hand = [robot, hand_center_z](const Config& config) -> Matrix4f {
            robot->setJointValues(config);
            Matrix4f pose = robot->getFramePose("panda_hand");
            pose.block<3,1>(0,3) += pose.block<3,3>(0,0) * Vector3f(0, 0, hand_center_z);
            return pose;
        };
        // Box 的 half_extents=(x, y, z)=(0.04, 0.11, 0.06) 
        // 对应总尺寸: X(厚度)=8cm, Y(宽度)=22cm, Z(长度)=12cm
        collision_checker_ptr->addRobotLinkBox(Vector3f(0.04f, 0.11f, 0.06f), fk_hand, "panda_hand");
    }
    std::cout << "  Robot links: 7 capsules + 1 frustum + 1 shaft + 1 hand box (total 10 parts).\n";

    // 球体定义仍保留，供 SDF 软代价评估器使用 (SDF 只支持球体查询)
    struct LinkSphere {
        std::string name;
        float radius;
        std::vector<Vector3f> offsets;
    };
    std::vector<LinkSphere> robot_structure = {
        {"panda_link0", 0.12f, {Vector3f(0,0,0.05f)}},
        {"panda_link1", 0.10f, {Vector3f(0,0,0.1f), Vector3f(0,0,0.2f), Vector3f(0,0,0.3f)}},
        {"panda_link2", 0.08f, {Vector3f(0,0,0)}},
        {"panda_link3", 0.07f, {Vector3f(0,0,0.05f), Vector3f(0,0,0.15f), Vector3f(0,0,0.25f)}},
        {"panda_link4", 0.07f, {Vector3f(0,0,0.05f)}},
        {"panda_link5", 0.06f, {Vector3f(0,0,0.1f), Vector3f(0,0,0.2f), Vector3f(0,0,0.3f)}},
        {"panda_link6", 0.06f, {Vector3f(0,0,0)}},
        {"panda_link7", 0.06f, {Vector3f(0,0,0), Vector3f(0,0,0.05f)}},
        {"panda_hand",  0.06f, {Vector3f(0,0,0), Vector3f(0,0,0.08f), Vector3f(0,0.04f,0.05f), Vector3f(0,-0.04f,0.05f)}}
    };

    // ---- 5d. 配置软碰撞代价评估器 (RRT* 各向异性代价) ----
    auto cost_evaluator = std::make_shared<SdfCostEvaluator>(robot, /*voxel_size=*/0.01f);

    // 树干也加入 SDF，使 VF 软斥力场在 0.20m 以外就能感知到树干，
    // 而不只依赖 grasp_rrt.cpp 里硬编码的 0.20m 触发范围。
    // btCylinderShapeZ 轴沿 Z，转换为线段: p0/p1 = trunk_center ± half_height * Z
    {
        Vector3f p0 = scene.trunk_center - Vector3f(0, 0, scene.trunk_height / 2.0f);
        Vector3f p1 = scene.trunk_center + Vector3f(0, 0, scene.trunk_height / 2.0f);
        cost_evaluator->addBranchCylinder(p0, p1, scene.trunk_radius);
    }

    // 导入树冠叶子
    for (const auto& br : scene.branches) {
        float cos_a = std::cos(br.angle);
        float sin_a = std::sin(br.angle);

        Vector3f p0 = scene.trunk_center;
        p0.z() = br.h; // 高度为绝对高度
        p0.x() += scene.trunk_radius * cos_a;
        p0.y() += scene.trunk_radius * sin_a;

        Vector3f p1 = p0 + Vector3f(cos_a, sin_a, 0) * br.length;
        cost_evaluator->addBranchCylinder(p0, p1, br.radius);
    }

    cost_evaluator->addBranchCylinder(
        scene.overhead.center, scene.overhead.center, scene.overhead.radius);

    // 导入机器人节点
    for (const auto& ls : robot_structure) {
        for (const auto& off : ls.offsets) {
            cost_evaluator->addRobotSphere(ls.name, ls.radius, off);
        }
    }
    
    // 一次性建表：根据所有圆柱几何自动计算包围盒并预计算 SDF 网格
    auto t_voxel_start = std::chrono::high_resolution_clock::now();
    cost_evaluator->buildSdf(/*padding=*/0.40f);
    auto t_voxel_end = std::chrono::high_resolution_clock::now();
    float voxel_build_ms =
        std::chrono::duration<float, std::milli>(t_voxel_end - t_voxel_start).count();
    std::cout << "  SDF voxel build: " << voxel_build_ms << " ms\n";

    cspace->setCostEvaluator(cost_evaluator);
    std::cout << "  SDF configured: " << scene.branches.size()
              << " branches + overhead obstacle (VF soft repulsion + TrajOpt).\n";

    // ---- 6. 创建目标物体 (苹果) ----
    std::cout << "[Setup] Apple target: pos=" << scene.apple_pos.transpose()
              << ", r=" << scene.apple_radius << "m\n";
    auto target = std::make_shared<SimpleTargetObject>(scene.apple_pos, scene.apple_radius);

    // 5e. 苹果 (Class 3: Goal Obstacle) — 大臂不可穿透，夹爪特赦
    collision_checker_ptr->addGoalObstacleSphere(scene.apple_radius, scene.apple_pos);
    // 豁免夹爪连杆 (panda_hand 及两根手指) 对苹果的碰撞检测
    collision_checker_ptr->addGoalObstacleExemptLink("panda_hand");
    collision_checker_ptr->addGoalObstacleExemptLink("panda_leftfinger");
    collision_checker_ptr->addGoalObstacleExemptLink("panda_rightfinger");
    std::cout << "  Apple: Registered as CLASS-3 Goal Obstacle (ACM: panda_hand/fingers exempt)\n";

    // ---- 7. 创建末端执行器 (简化平行夹爪) ----
    auto eef = std::make_shared<SimpleEndEffector>(robot, target, 0.08f, 0.05f);

    // ---- 8. 创建抓取质量评估器 ----
    auto quality = std::make_shared<SimpleGraspQuality>(2);

    // ---- 9. 配置 GraspRRT 参数 ----
    GraspRrtConfig config;
    config.max_cycles = std::numeric_limits<int>::max();
    config.extend_step_size = 0.05f;
    config.probab_grasp_hypothesis = 0.25f;       // 提高到 25% 概率，增加尝试次数
    config.cart_sampling_pos_step_size = 0.01f;    // 10mm 笛卡尔步长
    config.cart_sampling_ori_step_size = M_PI / 18; // 10° 旋转步长
    config.grasp_quality_min_score = 0.01f;
    config.min_grasp_contacts = 2;
    config.move_towards_goal_max_loops = 200;
    config.max_dist_goal_reached = 0.02f;          // 2cm 到达阈值
    config.planning_timeout_ms = 20000.0f;         // 20s 超时
    // [ADJUST] 扩大抓取感应圈从 10cm 到 15cm，使机器人更容易进入局部 Jacobian 引导阶段
    config.try_grasps_distance2 = 0.15f * 0.15f;
    // [BUG FIX] 进一步缩小 DCD 步长至 0.01rad，确保即使是极小障碍物也能在第一级递归中检测到点
    config.sampling_size_dcd = 0.01f;

    // ---- 10. 创建 GraspRRT 规划器 ----
    std::cout << "\n[Setup] Creating GraspRRT Planner...\n";
    auto planner = std::make_unique<GraspRrt>(
        cspace, robot, eef, target, quality, config);
    planner->setVerbose(true);
    planner->setHardObstacle(scene.overhead.center, /*trigger_dist=*/0.15f);
    planner->addExtraSetupTimeMs(voxel_build_ms);  // include SDF voxel build in Setup

    // ---- 11. 设置起始配置 (在 home_pose ± π/4 范围内随机无碰撞) ----
    auto [q_min, q_max] = robot->getJointLimits();
    Config home = scene.home_pose;
    Config delta = Config::Constant(7, static_cast<float>(M_PI / 4.0));
    Config sample_min = (home - delta).cwiseMax(q_min);
    Config sample_max = (home + delta).cwiseMin(q_max);

    std::mt19937 rng(std::random_device{}());
    Config start_config(7);
    int attempts = 0;
    std::cout << "[Setup] Sampling random collision-free start config "
              << "(within home_pose ± π/4)...\n";
    do {
        for (int i = 0; i < 7; i++)
            start_config[i] = std::uniform_real_distribution<float>(
                sample_min[i], sample_max[i])(rng);
        attempts++;
    } while (!planner->setStart(start_config));
    std::cout << "[Setup] Start config (random, attempts=" << attempts << "): "
              << start_config.transpose() << "\n";

    // 验证起始配置的 FK
    robot->setJointValues(start_config);
    Matrix4f start_tcp = robot->getGCPGlobalPose();
    std::cout << "[Setup] Start TCP position: "
              << start_tcp.block<3,1>(0,3).transpose() << "\n";
    std::cout << "[Setup] Distance to apple: "
              << (start_tcp.block<3,1>(0,3) - scene.apple_pos).norm() << "m\n\n";

    // ---- 12. 执行规划! ----
    std::cout << "============================================\n"
              << "  PLANNING START\n"
              << "============================================\n\n";

    auto t_start_rrt = std::chrono::high_resolution_clock::now();
    bool success = planner->plan(/*quiet=*/false);
    auto t_end_rrt = std::chrono::high_resolution_clock::now();
    double rrt_ms = std::chrono::duration<double, std::milli>(t_end_rrt - t_start_rrt).count();

    // ---- 13. 输出结果 ----
    std::cout << "\n============================================\n";
    if (success) {
        std::cout << "  PLANNING SUCCEEDED!\n"
                  << "============================================\n";

        auto solution = planner->getSolution();
        std::cout << "\n  Path: " << solution.size() << " waypoints\n";

        // ---- 保存路径到文件用于可视化 ----
        std::string out_path = "grasp_path.csv";
        std::ofstream out_file(out_path);
        if (out_file.is_open()) {
            for (const auto& wp : solution) {
                for (int i = 0; i < wp.size(); ++i) {
                    out_file << wp[i] << (i == wp.size() - 1 ? "" : ",");
                }
                out_file << "\n";
            }
            out_file.close();
            std::cout << "\n[Success] Path saved to: " << out_path << "\n";
        } else {
            std::cerr << "\n[Error] Could not save path to: " << out_path << "\n";
        }

        // 打印前几个和最后几个路径点
        int show = std::min(5, (int)solution.size());
        std::cout << "\n  First " << show << " waypoints:\n";
        for (int i = 0; i < show; ++i) {
            std::cout << "    [" << i << "] " << solution[i].transpose() << "\n";
        }
        if (solution.size() > 10) {
            std::cout << "    ... (" << solution.size() - 10 << " more) ...\n";
        }
        show = std::min(5, (int)solution.size());
        if (solution.size() > 5) {
            std::cout << "  Last " << show << " waypoints:\n";
            for (int i = solution.size() - show; i < (int)solution.size(); ++i) {
                std::cout << "    [" << i << "] " << solution[i].transpose() << "\n";
            }
        }

        // 打印最终 TCP 位置
        robot->setJointValues(solution.back());
        Matrix4f final_tcp = robot->getGCPGlobalPose();
        std::cout << "\n  Final TCP position: "
                  << final_tcp.block<3,1>(0,3).transpose() << "\n";
        std::cout << "  Distance to apple: "
                  << (final_tcp.block<3,1>(0,3) - scene.apple_pos).norm() << "m\n";

        // 打印抓取信息
        auto grasps = planner->getGraspResults();
        if (!grasps.empty()) {
            std::cout << "\n  Best grasp score: " << grasps.back().grasp_score << "\n";
            std::cout << "  Contacts: " << grasps.back().contacts.size() << "\n";
        }

        // ---- 14. 执行 TrajOpt 优化并计时 ----
        std::cout << "\n[TrajOpt] Starting SDF-based trajectory optimization...\n";
        auto t_start_trajopt = std::chrono::high_resolution_clock::now();
        
        auto optimized_solution = SdfTrajOptimizer::optimize(solution, robot, cost_evaluator);

        auto t_end_trajopt = std::chrono::high_resolution_clock::now();
        double trajopt_ms = std::chrono::duration<double, std::milli>(t_end_trajopt - t_start_trajopt).count();

        std::cout << "============================================\n"
                  << "  PERFORMANCE SUMMARY\n"
                  << "============================================\n"
                  << "  RRT Search Time       : " << rrt_ms << " ms\n"
                  << "  TrajOpt Opt Time      : " << trajopt_ms << " ms\n"
                  << "  Total Planning Time   : " << rrt_ms + trajopt_ms << " ms\n"
                  << "============================================\n";

        // 保存优化后的路径
        std::string opt_path = "grasp_path_trajopt.csv";
        std::ofstream opt_file(opt_path);
        if (opt_file.is_open()) {
            for (const auto& wp : optimized_solution) {
                for (int i = 0; i < wp.size(); ++i) {
                    opt_file << wp[i] << (i == wp.size() - 1 ? "" : ",");
                }
                opt_file << "\n";
            }
            opt_file.close();
            std::cout << "[Success] Optimized path saved to: " << opt_path << "\n";
        }
    } else {
        std::cout << "  PLANNING FAILED\n"
                  << "============================================\n";
        std::cout << "  Cycles completed: " << planner->getCycleCount() << "\n";
        std::cout << "  Tree nodes: " << planner->getTree()->getNrOfNodes() << "\n";
        std::cout << "\n  Tips:\n"
                  << "  - Increase max_cycles\n"
                  << "  - Increase probab_grasp_hypothesis\n"
                  << "  - Check apple position is reachable\n"
                  << "  - Adjust robot base position\n";
    }

    std::cout << "\nDone.\n";
    return success ? 0 : 1;
}
