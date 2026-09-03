/**
 * example_grasp_rrt.cpp — GraspRRT example: a Panda robot grasping an apple (through branches)
 *
 * Scene (from tree_model.py):
 *   - Panda 7-DOF robot arm (Pinocchio FK/Jacobian)
 *   - trunk (impenetrable cylinder)
 *   - branches (penetrable, but simplified to obstacles in this example)
 *   - apple (spherical grasp target)
 *
 * Build:
 *   cd grasp_rrt_framework/build
 *   cmake .. && make -j$(nproc)
 *   ./example_grasp_rrt [path_to_panda.urdf]
 *
 * Default URDF search order:
 *   1. command-line argument
 *   2. environment variable PANDA_URDF
 *   3. common pybullet_data paths
 */

#include <iostream>
#include <memory>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>

// GraspRRT framework
#include "types.h"
#include "cspace_sampled.h"
#include "cspace_tree.h"
#include "grasp_rrt.h"

// Concrete implementations
#include "panda_pinocchio.h"
#include "bullet_collision_checker.h"
#include "simple_end_effector.h"
#include "simple_grasp_quality.h"
#include "simple_target_object.h"
#include "sdf_cost_evaluator.h"
#include "sdf_traj_optimizer.h"
#include <chrono>
#include <random>

using namespace grasp_rrt;

// ============================================================================
// URDF path lookup
// ============================================================================

std::string findPandaUrdf(int argc, char* argv[]) {
    // 1. command-line argument
    if (argc > 1) {
        std::string path = argv[1];
        std::ifstream f(path);
        if (f.good()) return path;
        std::cerr << "WARNING: URDF not found at: " << path << "\n";
    }

    // 2. environment variable
    const char* env = std::getenv("PANDA_URDF");
    if (env) {
        std::ifstream f(env);
        if (f.good()) return std::string(env);
    }

    // 3. common paths
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
// Scene parameters (mirroring tree_model.py)
// ============================================================================

struct SceneConfig {
    // Trunk center position (pushed back a little to leave room for the canopy)
    Vector3f trunk_center = Vector3f(0.7f, 0.0f, 0.75f);
    float trunk_height = 1.5f;
    float trunk_radius = 0.08f;

    // Rigid obstacle directly above the apple (trunk-like, impenetrable)
    struct OverheadObstacle {
        Vector3f center  = Vector3f(0.45f, 0.0f, 0.91f); // directly above the apple (0.45,0,0.80)
        float    radius  = 0.08f;   // 8 cm occluder radius, slightly larger than the apple (4 cm), forcing a side approach
        float    half_h  = 0.04f;   // half-height 4 cm, total 8 cm (bottom at z=0.87, ~3 cm above the apple top)
    } overhead;

    // Branch (foliage) parameters: build a dense canopy
    // Branches are shortened and oriented toward the robot (angle ~ PI)
    struct BranchInfo {
        float h;
        float angle;
        float length = 0.35f;
        float radius = 0.03f;
    };
    std::vector<BranchInfo> branches = {
        // Lower foliage layer (blocks the wrist from below)
        {0.65f, 3.1415f, 0.35f, 0.03f}, {0.65f, 2.6f, 0.3f, 0.03f}, {0.65f, 3.68f, 0.3f, 0.03f},
        // Middle foliage layer, wrapping the apple
        {0.80f, 2.7f, 0.35f, 0.03f}, {0.80f, 3.58f, 0.35f, 0.03f}, {0.80f, 2.1f, 0.3f, 0.03f}, {0.80f, 4.18f, 0.3f, 0.03f},
        // Upper foliage layer (squeezes the space above)
        {0.95f, 3.1415f, 0.4f, 0.03f}, {0.95f, 2.6f, 0.3f, 0.03f}, {0.95f, 3.68f, 0.3f, 0.03f}
    };

    // Apple: placed in the middle of the dense branches
    float apple_radius = 0.04f;
    Vector3f apple_pos = Vector3f(0.45f, 0.0f, 0.80f);

    // Panda home pose
    Config home_pose = Config(7);
    SceneConfig() {
        home_pose << 0.0f, -M_PI/4, 0.0f, -3*M_PI/4, 0.0f, M_PI/2, M_PI/4;
    }
};

// ============================================================================
// Scene-layout export (for aligning with Isaac Sim)
// ============================================================================
// The Bullet side is the reference scene; export all geometry to scene_layout.json.
// All coordinates are in the ROBOT BASE frame (which is the world origin in this scene):
// on the Isaac side do not move the robot (FK calibration validated it), only place the tree/apple at these relative poses.
static void exportSceneLayout(const SceneConfig& scene,
                              const Config& start_config,
                              const std::string& out_path = "scene_layout.json") {
    std::ofstream f(out_path);
    if (!f.is_open()) {
        std::cerr << "[Warn] cannot write scene layout: " << out_path << "\n";
        return;
    }
    auto vec3 = [](const Vector3f& v) {
        std::ostringstream ss;
        ss << "[" << v.x() << ", " << v.y() << ", " << v.z() << "]";
        return ss.str();
    };

    f << "{\n";
    f << "  \"comment\": \"all coordinates are in the robot base frame, units meters/radians; cylinders are given as endpoints p0/p1 plus radius\",\n";
    f << "  \"robot_base_note\": \"the robot base is at the origin; on the Isaac side keep the robot fixed and move the tree to align\",\n";
    f << "  \"start_config\": [";
    for (int i = 0; i < start_config.size(); ++i)
        f << start_config[i] << (i + 1 < start_config.size() ? ", " : "");
    f << "],\n";

    // Trunk (rigid obstacle, contact = FAIL)
    Vector3f t0 = scene.trunk_center - Vector3f(0, 0, scene.trunk_height / 2.0f);
    Vector3f t1 = scene.trunk_center + Vector3f(0, 0, scene.trunk_height / 2.0f);
    f << "  \"trunk\": {\"p0\": " << vec3(t0) << ", \"p1\": " << vec3(t1)
      << ", \"radius\": " << scene.trunk_radius << ", \"class\": \"hard\"},\n";

    // Rigid obstacle above the apple (a sphere in the hard collision check; half_h unused)
    f << "  \"overhead_obstacle\": {\"center\": " << vec3(scene.overhead.center)
      << ", \"radius\": " << scene.overhead.radius << ", \"shape\": \"sphere\""
      << ", \"class\": \"hard\"},\n";

    // Canopy branches (penetrable soft obstacles, may be pushed aside) — endpoints computed exactly as for the SDF import
    f << "  \"branches\": [\n";
    for (size_t i = 0; i < scene.branches.size(); ++i) {
        const auto& br = scene.branches[i];
        float cos_a = std::cos(br.angle);
        float sin_a = std::sin(br.angle);
        Vector3f p0 = scene.trunk_center;
        p0.z() = br.h;
        p0.x() += scene.trunk_radius * cos_a;
        p0.y() += scene.trunk_radius * sin_a;
        Vector3f p1 = p0 + Vector3f(cos_a, sin_a, 0) * br.length;
        f << "    {\"p0\": " << vec3(p0) << ", \"p1\": " << vec3(p1)
          << ", \"radius\": " << br.radius << ", \"class\": \"soft\"}"
          << (i + 1 < scene.branches.size() ? "," : "") << "\n";
    }
    f << "  ],\n";

    // Apple (target)
    f << "  \"apple\": {\"center\": " << vec3(scene.apple_pos)
      << ", \"radius\": " << scene.apple_radius << "}\n";
    f << "}\n";
    std::cout << "[Setup] Scene layout exported to: " << out_path << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "======================================================\n"
              << "  GraspRRT Example: Panda Picking Apple Through Tree\n"
              << "======================================================\n\n";

    // ---- 1. Locate the URDF ----
    std::string urdf_path = findPandaUrdf(argc, argv);
    if (urdf_path.empty()) return 1;

    // ---- 2. Scene configuration ----
    SceneConfig scene;

    // ---- 3. Load the Panda robot (Pinocchio) ----
    std::cout << "[Setup] Loading Panda robot...\n";
    auto robot = std::make_shared<PandaPinocchioRobot>(urdf_path, "panda_hand");

    // DEBUG: print relative link positions to confirm the offset direction
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

    // Fetch joint limits
    auto [joint_lower, joint_upper] = robot->getJointLimits();
    std::cout << "[Setup] Joint limits:\n"
              << "  Lower: " << joint_lower.transpose() << "\n"
              << "  Upper: " << joint_upper.transpose() << "\n";

    // ---- 4. Create the configuration space ----
    std::cout << "[Setup] Creating C-Space (7-DOF)...\n";
    auto collision_checker_ptr = std::make_shared<BulletCollisionChecker>(robot);
    auto cspace = std::make_shared<CSpaceSampled>(7, collision_checker_ptr);
    cspace->setBoundaries(joint_lower, joint_upper);

    // ---- 5. Set up the collision scene (Bullet3) ----
    std::cout << "[Setup] Building collision scene...\n";

    // 5a. Trunk (rigid obstacle - must be avoided)
    Matrix4f trunk_pose = Matrix4f::Identity();
    trunk_pose.block<3,1>(0, 3) = scene.trunk_center;
    collision_checker_ptr->addObstacleCylinder(scene.trunk_radius, scene.trunk_height / 2.0f, trunk_pose);

    // 5a-2. Rigid obstacle directly above the apple (impenetrable sphere)
    collision_checker_ptr->addObstacleSphere(scene.overhead.radius, scene.overhead.center);
    std::cout << "  Overhead hard obstacle (SPHERE): center=["
              << scene.overhead.center.transpose() << "]"
              << "  r=" << scene.overhead.radius << "m\n";

    // 5b. Canopy/foliage (compliant/penetrable region - not added to hard collision checking)
    std::cout << "  (Note: " << scene.branches.size() 
              << " canopy branches exist in scene but act as permeable leaves, so they are not added to hard collision checker.)\n";

    // 5c. Robot collision bodies — capsule approximation
    // A capsule = cylinder + two hemispherical caps: hugs the link shape with no gaps.
    // One capsule per link suffices, and it is about as fast as spheres.
    //
    // Parameters:
    //   center_z: Z offset of the capsule center in the link's local frame
    //   half_h:   half-height of the capsule's cylindrical part (caps excluded)
    //   radius:   capsule radius
    struct LinkCapsule {
        std::string name;
        float radius;
        float center_z;   // Z offset of the capsule center relative to the link origin
        float half_h;     // cylinder half-height
    };

    // Capsule parameters derived from the Panda URDF geometry (+Z is the link extension direction)
    // Note: panda_link7 (the flange) is modeled separately below as a truncated cone and is not in this list
    std::vector<LinkCapsule> capsule_structure = {
        {"panda_link0", 0.10f, 0.05f,  0.02f},
        {"panda_link1", 0.08f, 0.15f,  0.12f},
        {"panda_link2", 0.07f, 0.00f,  0.02f},
        {"panda_link3", 0.06f, 0.15f,  0.12f},
        {"panda_link4", 0.06f, 0.05f,  0.03f},
        // Link5 still showed ~5.7 cm penetration at an 8 cm radius (vs the overhead sphere);
        // the real motor housing is wider, so it is enlarged to 13 cm for safety margin
        {"panda_link5", 0.13f, 0.15f,  0.13f},
        {"panda_link6", 0.06f, 0.00f,  0.02f},
        // panda_link7 modeled separately (truncated cone below)
        // panda_hand modeled separately (the gripper has fingers and is very wide, so a bounding box is used)
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

    // panda_link7 (flange) special model: trapezoidal-section truncated cone
    //   bottom face (link6 side, z=0)   : r = 0.040 m  (the Panda flange is about 80 mm across)
    //   top face (hand side, z=0.035)   : r = 0.030 m  (the narrowed end is about 55-60 mm)
    //   the cone center sits at z = 0.0175 m
    {
        float flange_center_z = 0.0175f;
        auto fk_flange = [robot, flange_center_z](const Config& config) -> Matrix4f {
            robot->setJointValues(config);
            Matrix4f pose = robot->getFramePose("panda_link7");
            pose.block<3,1>(0,3) += pose.block<3,3>(0,0) * Vector3f(0, 0, flange_center_z);
            return pose;
        };
        // The real Panda flange: bottom about 88 mm across (r=44 mm), top about 60 mm (r=30 mm), total thickness about 36 mm
        collision_checker_ptr->addRobotLinkFrustum(0.044f, 0.030f, 0.0180f,
                                                    fk_flange, "panda_link7_frustum");
    }

    // panda_link7 transition capsule: fills the gap between the cone top (z~0.036 m) and the panda_hand capsule bottom (z~0.057 m)
    // This wrist connecting shaft had no collision body at all, which was the root cause of the visual interpenetration.
    {
        float shaft_center_z = 0.053f; // capsule center: midpoint of z = 0.036 to 0.070 m
        auto fk_shaft = [robot, shaft_center_z](const Config& config) -> Matrix4f {
            robot->setJointValues(config);
            Matrix4f pose = robot->getFramePose("panda_link7");
            pose.block<3,1>(0,3) += pose.block<3,3>(0,0) * Vector3f(0, 0, shaft_center_z);
            return pose;
        };
        // The transition shaft is about 25 mm in radius (a thin shaft), covering z~0.036 to z~0.070 m
        collision_checker_ptr->addRobotLinkCapsule(0.025f, 0.017f, fk_shaft, "panda_link7_shaft");
    }

    // panda_hand (gripper) special model: an axis-aligned box
    // Diagnostics showed severe hand interpenetration: the gripper has a wide base and fully opened fingers, which a single capsule cannot cover in Y.
    // An 8 cm x 22 cm x 12 cm box wraps the whole gripper including both fingers.
    {
        float hand_center_z = 0.05f; // box center
        auto fk_hand = [robot, hand_center_z](const Config& config) -> Matrix4f {
            robot->setJointValues(config);
            Matrix4f pose = robot->getFramePose("panda_hand");
            pose.block<3,1>(0,3) += pose.block<3,3>(0,0) * Vector3f(0, 0, hand_center_z);
            return pose;
        };
        // Box half_extents=(x, y, z)=(0.04, 0.11, 0.06)
        // Total size: X (thickness)=8 cm, Y (width)=22 cm, Z (length)=12 cm
        collision_checker_ptr->addRobotLinkBox(Vector3f(0.04f, 0.11f, 0.06f), fk_hand, "panda_hand");
    }
    std::cout << "  Robot links: 7 capsules + 1 frustum + 1 shaft + 1 hand box (total 10 parts).\n";

    // The sphere definitions are kept for the SDF soft-cost evaluator (the SDF only supports sphere queries)
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

    // ---- 5d. Configure the soft-collision cost evaluator (RRT* anisotropic cost) ----
    auto cost_evaluator = std::make_shared<SdfCostEvaluator>(robot, /*voxel_size=*/0.01f);

    // The trunk is added to the SDF as well, so the VF soft repulsion field senses it
    // beyond 0.20 m instead of relying only on the 0.20 m trigger hardcoded in grasp_rrt.cpp.
    // btCylinderShapeZ has its axis along Z; as a segment: p0/p1 = trunk_center +- half_height * Z
    {
        Vector3f p0 = scene.trunk_center - Vector3f(0, 0, scene.trunk_height / 2.0f);
        Vector3f p1 = scene.trunk_center + Vector3f(0, 0, scene.trunk_height / 2.0f);
        cost_evaluator->addBranchCylinder(p0, p1, scene.trunk_radius);
    }

    // Import the canopy foliage
    for (const auto& br : scene.branches) {
        float cos_a = std::cos(br.angle);
        float sin_a = std::sin(br.angle);

        Vector3f p0 = scene.trunk_center;
        p0.z() = br.h; // height is absolute
        p0.x() += scene.trunk_radius * cos_a;
        p0.y() += scene.trunk_radius * sin_a;

        Vector3f p1 = p0 + Vector3f(cos_a, sin_a, 0) * br.length;
        cost_evaluator->addBranchCylinder(p0, p1, br.radius);
    }

    cost_evaluator->addBranchCylinder(
        scene.overhead.center, scene.overhead.center, scene.overhead.radius);

    // Import the robot spheres
    for (const auto& ls : robot_structure) {
        for (const auto& off : ls.offsets) {
            cost_evaluator->addRobotSphere(ls.name, ls.radius, off);
        }
    }
    
    // One-shot build: derive the bounding box from all cylinders and precompute the SDF grid
    auto t_voxel_start = std::chrono::high_resolution_clock::now();
    cost_evaluator->buildSdf(/*padding=*/0.40f);
    auto t_voxel_end = std::chrono::high_resolution_clock::now();
    float voxel_build_ms =
        std::chrono::duration<float, std::milli>(t_voxel_end - t_voxel_start).count();
    std::cout << "  SDF voxel build: " << voxel_build_ms << " ms\n";

    cspace->setCostEvaluator(cost_evaluator);
    std::cout << "  SDF configured: " << scene.branches.size()
              << " branches + overhead obstacle (VF soft repulsion + TrajOpt).\n";

    // ---- 6. Create the target object (apple) ----
    std::cout << "[Setup] Apple target: pos=" << scene.apple_pos.transpose()
              << ", r=" << scene.apple_radius << "m\n";
    auto target = std::make_shared<SimpleTargetObject>(scene.apple_pos, scene.apple_radius);

    // 5e. Apple (Class 3: goal obstacle) — impenetrable for the arm, exempt for the gripper
    collision_checker_ptr->addGoalObstacleSphere(scene.apple_radius, scene.apple_pos);
    // Exempt the gripper links (panda_hand and both fingers) from apple collision checking
    collision_checker_ptr->addGoalObstacleExemptLink("panda_hand");
    collision_checker_ptr->addGoalObstacleExemptLink("panda_leftfinger");
    collision_checker_ptr->addGoalObstacleExemptLink("panda_rightfinger");
    std::cout << "  Apple: Registered as CLASS-3 Goal Obstacle (ACM: panda_hand/fingers exempt)\n";

    // ---- 7. Create the end effector (simplified parallel gripper) ----
    auto eef = std::make_shared<SimpleEndEffector>(robot, target, 0.08f, 0.05f);

    // ---- 8. Create the grasp-quality evaluator ----
    auto quality = std::make_shared<SimpleGraspQuality>(2);

    // ---- 9. Configure GraspRRT parameters ----
    GraspRrtConfig config;
    config.max_cycles = std::numeric_limits<int>::max();
    config.extend_step_size = 0.05f;
    config.probab_grasp_hypothesis = 0.25f;       // raised to 25% to increase the number of attempts
    config.cart_sampling_pos_step_size = 0.01f;    // 10 mm Cartesian step
    config.cart_sampling_ori_step_size = M_PI / 18; // 10 deg rotation step
    config.grasp_quality_min_score = 0.01f;
    config.min_grasp_contacts = 2;
    config.move_towards_goal_max_loops = 200;
    config.max_dist_goal_reached = 0.02f;          // 2 cm arrival threshold
    config.planning_timeout_ms = 20000.0f;         // 20 s timeout
    // [ADJUST] widen the grasp sensing radius from 10 cm to 15 cm so the robot enters the local Jacobian-guided phase more easily
    config.try_grasps_distance2 = 0.15f * 0.15f;
    // [BUG FIX] shrink the DCD step further to 0.01 rad so even tiny obstacles are caught at the first recursion level
    config.sampling_size_dcd = 0.01f;

    // ---- 10. Create the GraspRRT planner ----
    std::cout << "\n[Setup] Creating GraspRRT Planner...\n";
    auto planner = std::make_unique<GraspRrt>(
        cspace, robot, eef, target, quality, config);
    planner->setVerbose(true);
    planner->setHardObstacle(scene.overhead.center, /*trigger_dist=*/0.15f);
    planner->addExtraSetupTimeMs(voxel_build_ms);  // include SDF voxel build in Setup

    // ---- 11. Set the start configuration (random collision-free within home_pose +- pi/4) ----
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

    // Validate the FK of the start configuration
    robot->setJointValues(start_config);
    Matrix4f start_tcp = robot->getGCPGlobalPose();
    std::cout << "[Setup] Start TCP position: "
              << start_tcp.block<3,1>(0,3).transpose() << "\n";
    std::cout << "[Setup] Distance to apple: "
              << (start_tcp.block<3,1>(0,3) - scene.apple_pos).norm() << "m\n\n";

    // Export the scene layout (tree/apple/start configuration) for aligning the Isaac Sim scene
    exportSceneLayout(scene, start_config);

    // ---- 12. Run the planner ----
    std::cout << "============================================\n"
              << "  PLANNING START\n"
              << "============================================\n\n";

    auto t_start_rrt = std::chrono::high_resolution_clock::now();
    bool success = planner->plan(/*quiet=*/false);
    auto t_end_rrt = std::chrono::high_resolution_clock::now();
    double rrt_ms = std::chrono::duration<double, std::milli>(t_end_rrt - t_start_rrt).count();

    // ---- 13. Report results ----
    std::cout << "\n============================================\n";
    if (success) {
        std::cout << "  PLANNING SUCCEEDED!\n"
                  << "============================================\n";

        auto solution = planner->getSolution();
        std::cout << "\n  Path: " << solution.size() << " waypoints\n";

        // ---- Save the path to a file for visualization ----
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

        // Print the first and last few waypoints
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

        // Print the final TCP position
        robot->setJointValues(solution.back());
        Matrix4f final_tcp = robot->getGCPGlobalPose();
        std::cout << "\n  Final TCP position: "
                  << final_tcp.block<3,1>(0,3).transpose() << "\n";
        std::cout << "  Distance to apple: "
                  << (final_tcp.block<3,1>(0,3) - scene.apple_pos).norm() << "m\n";

        // Print grasp information
        auto grasps = planner->getGraspResults();
        if (!grasps.empty()) {
            std::cout << "\n  Best grasp score: " << grasps.back().grasp_score << "\n";
            std::cout << "  Contacts: " << grasps.back().contacts.size() << "\n";
        }

        // ---- 14. Run TrajOpt optimization and time it ----
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

        // Save the optimized path
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
