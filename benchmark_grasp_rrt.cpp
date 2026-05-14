/**
 * benchmark_grasp_rrt.cpp — Run GraspRRT path planning N times and record results.
 *
 * Each run uses a fresh random initial posture (within home_pose ± π/4).
 * Results are written to:
 *   benchmark_results/benchmark_results.csv   - summary table (one row per run)
 *   benchmark_results/run01_log.txt           - RESULTS block per run
 *   benchmark_results/grasp_path_01.csv       - successful trajectory (only if run succeeded)
 *
 * Usage:
 *   ./benchmark_grasp_rrt                  # 10 runs (default)
 *   ./benchmark_grasp_rrt 20               # 20 runs
 *   ./benchmark_grasp_rrt 10 <urdf_path>   # 10 runs with explicit URDF
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>

// GraspRRT framework
#include "types.h"
#include "cspace_sampled.h"
#include "cspace_tree.h"
#include "grasp_rrt.h"

// Implementations
#include "panda_pinocchio.h"
#include "bullet_collision_checker.h"
#include "simple_end_effector.h"
#include "simple_grasp_quality.h"
#include "simple_target_object.h"
#include "soft_cost_evaluator.h"
#include "sdf_cost_evaluator.h"
#include "sdf_traj_optimizer.h"

using namespace grasp_rrt;

// ============================================================================
// URDF discovery
// ============================================================================

std::string findPandaUrdf(const std::string& cli_path) {
    if (!cli_path.empty()) {
        std::ifstream f(cli_path);
        if (f.good()) return cli_path;
        std::cerr << "WARNING: URDF not found at: " << cli_path << "\n";
    }
    const char* env = std::getenv("PANDA_URDF");
    if (env) {
        std::ifstream f(env);
        if (f.good()) return std::string(env);
    }
    std::vector<std::string> search_paths = {
        "/usr/local/lib/python3.10/dist-packages/pybullet_data/franka_panda/panda.urdf",
        "/usr/local/lib/python3.8/dist-packages/pybullet_data/franka_panda/panda.urdf",
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/.local/lib/python3.10/site-packages/pybullet_data/franka_panda/panda.urdf",
        "/opt/ros/humble/share/franka_description/robots/panda/panda.urdf",
    };
    for (const auto& p : search_paths) {
        std::ifstream f(p);
        if (f.good()) {
            std::cout << "Found URDF at: " << p << "\n";
            return p;
        }
    }
    return "";
}

// ============================================================================
// Scene config (mirrors example_grasp_rrt.cpp::SceneConfig)
// ============================================================================

struct SceneConfig {
    Vector3f trunk_center = Vector3f(0.7f, 0.0f, 0.75f);
    float trunk_height = 1.5f;
    float trunk_radius = 0.08f;

    struct OverheadObstacle {
        Vector3f center = Vector3f(0.45f, 0.0f, 0.91f);
        float    radius = 0.08f;
        float    half_h = 0.04f;
    } overhead;

    struct BranchInfo { float h, angle, length, radius; };
    std::vector<BranchInfo> branches = {
        {0.65f, 3.1415f, 0.35f, 0.03f}, {0.65f, 2.6f, 0.3f, 0.03f}, {0.65f, 3.68f, 0.3f, 0.03f},
        {0.80f, 2.7f, 0.35f, 0.03f}, {0.80f, 3.58f, 0.35f, 0.03f},
        {0.80f, 2.1f, 0.3f, 0.03f}, {0.80f, 4.18f, 0.3f, 0.03f},
        {0.95f, 3.1415f, 0.4f, 0.03f}, {0.95f, 2.6f, 0.3f, 0.03f}, {0.95f, 3.68f, 0.3f, 0.03f}
    };

    float apple_radius = 0.04f;
    Vector3f apple_pos = Vector3f(0.45f, 0.0f, 0.80f);

    Config home_pose = Config(7);
    SceneConfig() {
        home_pose << 0.0f, -M_PI/4, 0.0f, -3*M_PI/4, 0.0f, M_PI/2, M_PI/4;
    }
};

// ============================================================================
// Per-run result
// ============================================================================

struct RunResult {
    int     run_idx;
    bool    success = false;
    Config  start_config;
    float   tcp_to_apple_start = std::nanf("");

    // GraspInfo
    float   grasp_distance = std::nanf("");
    float   grasp_score    = std::nanf("");
    int     grasp_node_id  = -1;
    int     contacts       = 0;

    // RESULTS block
    float   setup_ms          = std::nanf("");
    float   planning_ms       = std::nanf("");
    float   rrt_build_ms      = std::nanf("");
    float   nn_search_ms      = std::nanf("");
    float   vf_grad_ms        = std::nanf("");
    float   collision_ms      = std::nanf("");
    float   move_to_grasp_ms  = std::nanf("");
    int     move_to_grasp_tries = 0;
    float   grasp_scoring_ms  = std::nanf("");
    int     grasp_scorings    = 0;
    int     nodes             = 0;

    // Solution info
    int     cycles            = 0;
    int     waypoints         = 0;
    float   final_tcp_to_apple = std::nanf("");
    float   trajopt_ms        = std::nanf("");

    // Group A: tree growth efficiency
    int     collision_check_count = 0;
    int     rejected_configs   = 0;     // cycles - nodes
    float   cycles_per_node    = std::nanf("");

    // Group B: grasp attempt result breakdown
    int     grasp_attempts_total = 0;
    int     grasp_goal_reached   = 0;
    int     grasp_pose_reached   = 0;
    int     grasp_trapped        = 0;
    int     grasp_error          = 0;

    // Group C: face disable stats
    int     faces_total          = 0;
    int     faces_disabled       = 0;
    int     faces_active_final   = 0;
    int     faces_with_attempts  = 0;
    int     max_face_failures    = 0;
};

// Helper: format float (NaN → "NaN")
std::string fmt(float v) {
    if (std::isnan(v)) return "NaN";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << v;
    return oss.str();
}

void writeRunLog(const std::string& path, const RunResult& r) {
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "Run " << std::setfill('0') << std::setw(2) << r.run_idx
      << " - " << (r.success ? "SUCCESS" : "FAILED") << "\n\n";
    f << "Start config:        " << r.start_config.transpose() << "\n";
    f << "Start TCP→apple:     " << fmt(r.tcp_to_apple_start) << " m\n\n";

    if (!r.success) {
        f << "PLANNING FAILED\n";
        return;
    }

    f << "Grasp Info:\n"
      << "    Distance to object: " << fmt(r.grasp_distance) << "\n"
      << "    Grasp score:        " << fmt(r.grasp_score) << "\n"
      << "    RRT Node ID:        " << r.grasp_node_id << "\n"
      << "    Contacts:           " << r.contacts << "\n\n";

    f << "=================== RESULTS ===================\n"
      << "Setup:         " << fmt(r.setup_ms) << " ms\n"
      << "Planning:      " << fmt(r.planning_ms) << " ms\n"
      << "RRT build:     " << fmt(r.rrt_build_ms) << " ms\n"
      << " ├─ NN Search: " << fmt(r.nn_search_ms) << " ms\n"
      << " ├─ VF Grad:   " << fmt(r.vf_grad_ms) << " ms\n"
      << " ├─ Collision: " << fmt(r.collision_ms)
      << " ms (count: " << r.collision_check_count << ")\n"
      << "Move to grasp: " << fmt(r.move_to_grasp_ms)
      << " ms (" << r.move_to_grasp_tries << " tries)\n"
      << "Grasp scoring: " << fmt(r.grasp_scoring_ms)
      << " ms (" << r.grasp_scorings << " scorings)\n"
      << "Nodes:         " << r.nodes << "\n"
      << "Cycles:        " << r.cycles << "\n"
      << "Cycles/Node:   " << fmt(r.cycles_per_node) << "\n"
      << "Rejected:      " << r.rejected_configs << "\n"
      << "=================== RESULTS ===================\n\n";

    f << "Grasp Attempts:\n"
      << "    Total:         " << r.grasp_attempts_total << "\n"
      << "    Goal reached:  " << r.grasp_goal_reached << "\n"
      << "    Pose reached:  " << r.grasp_pose_reached << "\n"
      << "    Trapped:       " << r.grasp_trapped << "\n"
      << "    Errors:        " << r.grasp_error << "\n\n";

    f << "Face Disable:\n"
      << "    Faces disabled:      " << r.faces_disabled << " / " << r.faces_total << "\n"
      << "    Faces active final:  " << r.faces_active_final << "\n"
      << "    Faces with attempts: " << r.faces_with_attempts << "\n"
      << "    Max failures/face:   " << r.max_face_failures << "\n\n";

    f << "Solution waypoints:  " << r.waypoints << "\n"
      << "Final TCP→apple:     " << fmt(r.final_tcp_to_apple) << " m\n"
      << "TrajOpt time:        " << fmt(r.trajopt_ms) << " ms\n";
}

void writeCsvHeader(std::ofstream& f) {
    f << "run,success,setup_ms,planning_ms,rrt_build_ms,nn_search_ms,"
      << "vf_grad_ms,collision_ms,move_to_grasp_ms,move_to_grasp_tries,"
      << "grasp_scoring_ms,grasp_scorings,nodes,cycles,waypoints,"
      << "grasp_distance,grasp_score,grasp_node_id,contacts,"
      << "tcp_to_apple_start,final_tcp_to_apple,trajopt_ms,"
      // Group A
      << "collision_check_count,rejected_configs,cycles_per_node,"
      // Group B
      << "grasp_attempts_total,grasp_goal_reached,grasp_pose_reached,"
      << "grasp_trapped,grasp_error,"
      // Group C
      << "faces_total,faces_disabled,faces_active_final,"
      << "faces_with_attempts,max_face_failures\n";
}

void writeCsvRow(std::ofstream& f, const RunResult& r) {
    f << r.run_idx << "," << (r.success ? 1 : 0) << ",";
    f << fmt(r.setup_ms) << "," << fmt(r.planning_ms) << ","
      << fmt(r.rrt_build_ms) << "," << fmt(r.nn_search_ms) << ","
      << fmt(r.vf_grad_ms) << "," << fmt(r.collision_ms) << ","
      << fmt(r.move_to_grasp_ms) << ",";
    if (r.success) f << r.move_to_grasp_tries; else f << "NaN";
    f << "," << fmt(r.grasp_scoring_ms) << ",";
    if (r.success) f << r.grasp_scorings; else f << "NaN";
    f << ",";
    if (r.success) f << r.nodes; else f << "NaN";
    f << ",";
    if (r.success) f << r.cycles; else f << "NaN";
    f << ",";
    if (r.success) f << r.waypoints; else f << "NaN";
    f << ",";
    f << fmt(r.grasp_distance) << "," << fmt(r.grasp_score) << ",";
    if (r.success) f << r.grasp_node_id; else f << "NaN";
    f << ",";
    if (r.success) f << r.contacts; else f << "NaN";
    f << ",";
    f << fmt(r.tcp_to_apple_start) << "," << fmt(r.final_tcp_to_apple) << ","
      << fmt(r.trajopt_ms) << ",";
    // Group A
    f << r.collision_check_count << "," << r.rejected_configs << ","
      << fmt(r.cycles_per_node) << ",";
    // Group B
    f << r.grasp_attempts_total << "," << r.grasp_goal_reached << ","
      << r.grasp_pose_reached << "," << r.grasp_trapped << ","
      << r.grasp_error << ",";
    // Group C
    f << r.faces_total << "," << r.faces_disabled << ","
      << r.faces_active_final << "," << r.faces_with_attempts << ","
      << r.max_face_failures << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    int num_runs = 10;
    std::string urdf_cli;
    bool method_a_enabled = true;
    bool face_disable_enabled = true;
    std::string fixed_start_str;
    std::string starts_file;
    std::string label_suffix;

    // Parse args
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--no-method-a") {
            method_a_enabled = false;
        } else if (a == "--no-face-disable") {
            face_disable_enabled = false;
        } else if (a == "--start" && i + 1 < argc) {
            fixed_start_str = argv[++i];
        } else if (a == "--starts-file" && i + 1 < argc) {
            starts_file = argv[++i];
        } else if (a == "--label" && i + 1 < argc) {
            label_suffix = argv[++i];
        } else {
            positional.push_back(a);
        }
    }
    if (positional.size() > 0) num_runs = std::atoi(positional[0].c_str());
    if (positional.size() > 1) urdf_cli = positional[1];

    // Parse fixed start if provided
    Config fixed_start;
    bool use_fixed_start = false;
    if (!fixed_start_str.empty()) {
        std::vector<float> vals;
        std::stringstream ss(fixed_start_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) vals.push_back(std::stof(tok));
        if (vals.size() != 7) {
            std::cerr << "ERROR: --start expects 7 comma-separated values, got "
                      << vals.size() << "\n";
            return 1;
        }
        fixed_start = Config(7);
        for (int i = 0; i < 7; i++) fixed_start[i] = vals[i];
        use_fixed_start = true;
    }

    // Parse starts-file if provided (each line = one config of 7 values)
    std::vector<Config> starts_from_file;
    if (!starts_file.empty()) {
        std::ifstream sf(starts_file);
        if (!sf.is_open()) {
            std::cerr << "ERROR: cannot open --starts-file " << starts_file << "\n";
            return 1;
        }
        std::string line;
        while (std::getline(sf, line)) {
            // Strip whitespace; skip empty / comment lines
            size_t s = line.find_first_not_of(" \t\r\n");
            if (s == std::string::npos || line[s] == '#') continue;
            std::vector<float> vals;
            std::stringstream ls(line);
            std::string tok;
            while (std::getline(ls, tok, ',')) vals.push_back(std::stof(tok));
            if (vals.size() != 7) {
                std::cerr << "ERROR: line in starts-file has " << vals.size()
                          << " values, expected 7: " << line << "\n";
                return 1;
            }
            Config q(7);
            for (int i = 0; i < 7; i++) q[i] = vals[i];
            starts_from_file.push_back(q);
        }
        if (starts_from_file.empty()) {
            std::cerr << "ERROR: no valid starts in " << starts_file << "\n";
            return 1;
        }
        num_runs = static_cast<int>(starts_from_file.size());
        std::cout << "[Info] Loaded " << num_runs << " starts from " << starts_file << "\n";
    }

    std::cout << "================================================================\n"
              << "  GraspRRT Benchmark — " << num_runs << " trials, "
              << "Method A: " << (method_a_enabled ? "ENABLED" : "DISABLED")
              << ", Face-disable: " << (face_disable_enabled ? "ENABLED" : "DISABLED")
              << (use_fixed_start ? ", FIXED START"
                  : (!starts_from_file.empty() ? ", STARTS FROM FILE" : ", random starts"))
              << "\n"
              << "================================================================\n\n";

    // ---- 1. URDF ----
    std::string urdf_path = findPandaUrdf(urdf_cli);
    if (urdf_path.empty()) {
        std::cerr << "ERROR: Could not find panda.urdf!\n";
        return 1;
    }

    // ---- 2. Scene ----
    SceneConfig scene;

    // ---- 3. Robot ----
    auto robot = std::make_shared<PandaPinocchioRobot>(urdf_path, "panda_hand");
    auto [joint_lower, joint_upper] = robot->getJointLimits();

    // ---- 4. C-space ----
    auto collision_checker_ptr = std::make_shared<BulletCollisionChecker>(robot);
    auto cspace = std::make_shared<CSpaceSampled>(7, collision_checker_ptr);
    cspace->setBoundaries(joint_lower, joint_upper);

    // ---- 5. Hard collision scene (trunk + overhead) ----
    Matrix4f trunk_pose = Matrix4f::Identity();
    trunk_pose.block<3,1>(0, 3) = scene.trunk_center;
    collision_checker_ptr->addObstacleCylinder(scene.trunk_radius, scene.trunk_height / 2.0f, trunk_pose);
    collision_checker_ptr->addObstacleSphere(scene.overhead.radius, scene.overhead.center);

    // 5c-d. Robot collision capsules (mirrors example)
    struct LinkCapsule { std::string name; float radius, center_z, half_h; };
    std::vector<LinkCapsule> capsule_structure = {
        {"panda_link0", 0.10f, 0.05f, 0.02f}, {"panda_link1", 0.08f, 0.15f, 0.12f},
        {"panda_link2", 0.07f, 0.00f, 0.02f}, {"panda_link3", 0.06f, 0.15f, 0.12f},
        {"panda_link4", 0.06f, 0.05f, 0.03f}, {"panda_link5", 0.08f, 0.15f, 0.13f},
        {"panda_link6", 0.06f, 0.00f, 0.02f},
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
    {
        float flange_center_z = 0.0175f;
        auto fk_flange = [robot, flange_center_z](const Config& c) -> Matrix4f {
            robot->setJointValues(c);
            Matrix4f pose = robot->getFramePose("panda_link7");
            pose.block<3,1>(0,3) += pose.block<3,3>(0,0) * Vector3f(0, 0, flange_center_z);
            return pose;
        };
        collision_checker_ptr->addRobotLinkFrustum(0.044f, 0.030f, 0.0180f, fk_flange, "panda_link7_frustum");
    }
    {
        float shaft_center_z = 0.053f;
        auto fk_shaft = [robot, shaft_center_z](const Config& c) -> Matrix4f {
            robot->setJointValues(c);
            Matrix4f pose = robot->getFramePose("panda_link7");
            pose.block<3,1>(0,3) += pose.block<3,3>(0,0) * Vector3f(0, 0, shaft_center_z);
            return pose;
        };
        collision_checker_ptr->addRobotLinkCapsule(0.025f, 0.017f, fk_shaft, "panda_link7_shaft");
    }
    {
        float hand_center_z = 0.05f;
        auto fk_hand = [robot, hand_center_z](const Config& c) -> Matrix4f {
            robot->setJointValues(c);
            Matrix4f pose = robot->getFramePose("panda_hand");
            pose.block<3,1>(0,3) += pose.block<3,3>(0,0) * Vector3f(0, 0, hand_center_z);
            return pose;
        };
        collision_checker_ptr->addRobotLinkBox(Vector3f(0.04f, 0.11f, 0.06f), fk_hand, "panda_hand");
    }

    // ---- 6. SDF cost evaluator (soft repulsion) ----
    struct LinkSphere { std::string name; float radius; std::vector<Vector3f> offsets; };
    std::vector<LinkSphere> robot_structure = {
        {"panda_link0", 0.12f, {Vector3f(0,0,0.05f)}},
        {"panda_link1", 0.10f, {Vector3f(0,0,0.1f), Vector3f(0,0,0.2f), Vector3f(0,0,0.3f)}},
        {"panda_link2", 0.08f, {Vector3f(0,0,0)}},
        {"panda_link3", 0.07f, {Vector3f(0,0,0.05f), Vector3f(0,0,0.15f), Vector3f(0,0,0.25f)}},
        {"panda_link4", 0.07f, {Vector3f(0,0,0.05f)}},
        {"panda_link5", 0.06f, {Vector3f(0,0,0.1f), Vector3f(0,0,0.2f), Vector3f(0,0,0.3f)}},
        {"panda_link6", 0.06f, {Vector3f(0,0,0)}},
        {"panda_link7", 0.06f, {Vector3f(0,0,0), Vector3f(0,0,0.05f)}},
        {"panda_hand",  0.06f, {Vector3f(0,0,0), Vector3f(0,0,0.08f),
                                Vector3f(0,0.04f,0.05f), Vector3f(0,-0.04f,0.05f)}}
    };

    auto cost_evaluator = std::make_shared<SdfCostEvaluator>(robot, /*voxel_size=*/0.01f);
    {
        Vector3f p0 = scene.trunk_center - Vector3f(0, 0, scene.trunk_height / 2.0f);
        Vector3f p1 = scene.trunk_center + Vector3f(0, 0, scene.trunk_height / 2.0f);
        cost_evaluator->addBranchCylinder(p0, p1, scene.trunk_radius);
    }
    for (const auto& br : scene.branches) {
        float ca = std::cos(br.angle), sa = std::sin(br.angle);
        Vector3f p0 = scene.trunk_center;
        p0.z() = br.h;
        p0.x() += scene.trunk_radius * ca;
        p0.y() += scene.trunk_radius * sa;
        Vector3f p1 = p0 + Vector3f(ca, sa, 0) * br.length;
        cost_evaluator->addBranchCylinder(p0, p1, br.radius);
    }
    cost_evaluator->addBranchCylinder(scene.overhead.center, scene.overhead.center, scene.overhead.radius);
    for (const auto& ls : robot_structure)
        for (const auto& off : ls.offsets)
            cost_evaluator->addRobotSphere(ls.name, ls.radius, off);

    auto t_voxel_start = std::chrono::high_resolution_clock::now();
    cost_evaluator->buildSdf(/*padding=*/0.40f);
    auto t_voxel_end = std::chrono::high_resolution_clock::now();
    float voxel_build_ms =
        std::chrono::duration<float, std::milli>(t_voxel_end - t_voxel_start).count();
    std::cout << "[Setup] SDF voxel build: " << voxel_build_ms << " ms (one-time)\n";

    cspace->setCostEvaluator(cost_evaluator);

    // ---- 7. Target object (apple) ----
    auto target = std::make_shared<SimpleTargetObject>(scene.apple_pos, scene.apple_radius);
    collision_checker_ptr->addGoalObstacleSphere(scene.apple_radius, scene.apple_pos);
    collision_checker_ptr->addGoalObstacleExemptLink("panda_hand");
    collision_checker_ptr->addGoalObstacleExemptLink("panda_leftfinger");
    collision_checker_ptr->addGoalObstacleExemptLink("panda_rightfinger");

    auto eef = std::make_shared<SimpleEndEffector>(robot, target, 0.08f, 0.05f);
    auto quality = std::make_shared<SimpleGraspQuality>(2);

    // ---- 8. Planner config ----
    GraspRrtConfig config;
    config.max_cycles = std::numeric_limits<int>::max();
    config.extend_step_size = 0.05f;
    config.probab_grasp_hypothesis = 0.25f;
    config.cart_sampling_pos_step_size = 0.01f;
    config.cart_sampling_ori_step_size = M_PI / 18;
    config.grasp_quality_min_score = 0.01f;
    config.min_grasp_contacts = 2;
    config.move_towards_goal_max_loops = 200;
    config.max_dist_goal_reached = 0.02f;
    config.planning_timeout_ms = 20000.0f;
    config.try_grasps_distance2 = 0.15f * 0.15f;
    config.sampling_size_dcd = 0.01f;
    config.face_disable_enabled = face_disable_enabled;

    // ---- 9. Output directory ----
    std::string out_dir = method_a_enabled
        ? "benchmark_results_with_A"
        : "benchmark_results_no_A";
    if (!face_disable_enabled) out_dir += "_noFD";
    if (!label_suffix.empty()) out_dir += "_" + label_suffix;
    mkdir(out_dir.c_str(), 0755);  // ignore EEXIST
    std::cout << "[Setup] Output directory: " << out_dir << "/\n";

    std::ofstream csv_file(out_dir + "/benchmark_results.csv");
    if (!csv_file.is_open()) {
        std::cerr << "ERROR: Cannot open benchmark_results.csv\n";
        return 1;
    }
    writeCsvHeader(csv_file);

    // ---- 10. Random start sampling range (home_pose ± π/4) ----
    Config delta = Config::Constant(7, static_cast<float>(M_PI / 4.0));
    Config sample_min = (scene.home_pose - delta).cwiseMax(joint_lower);
    Config sample_max = (scene.home_pose + delta).cwiseMin(joint_upper);
    std::mt19937 rng(std::random_device{}());

    std::vector<RunResult> all_results;

    // ---- 11. Run loop ----
    for (int run = 0; run < num_runs; run++) {
        std::cout << "\n┌──────────────────────────────────────────────────────────────┐\n"
                  << "│  Run " << std::setw(2) << (run + 1) << " / " << num_runs
                  << std::setw(50) << " │\n"
                  << "└──────────────────────────────────────────────────────────────┘\n";

        RunResult result;
        result.run_idx = run + 1;

        // Fresh planner per run (clean state)
        auto planner = std::make_unique<GraspRrt>(
            cspace, robot, eef, target, quality, config);
        planner->setVerbose(false);
        planner->setHardObstacle(scene.overhead.center, /*trigger_dist=*/0.15f);
        planner->addExtraSetupTimeMs(voxel_build_ms);
        planner->setGoalBiasProb(method_a_enabled ? 0.15f : 0.0f);  // Method A on/off

        // Decide start config: priority is starts-file > fixed-start > random
        Config start_config(7);
        if (!starts_from_file.empty()) {
            start_config = starts_from_file[run];
            if (!planner->setStart(start_config)) {
                std::cerr << "ERROR: Start " << (run + 1)
                          << " from file is invalid (collision/OOB), skipping.\n";
                result.success = false;
                writeRunLog((std::string(out_dir) + "/run" +
                             (run < 9 ? "0" : "") + std::to_string(run + 1) +
                             "_log.txt").c_str(), result);
                writeCsvRow(csv_file, result);
                csv_file.flush();
                all_results.push_back(result);
                continue;
            }
        } else if (use_fixed_start) {
            start_config = fixed_start;
            if (!planner->setStart(start_config)) {
                std::cerr << "ERROR: Fixed start config is invalid (collision/OOB)\n";
                return 1;
            }
        } else {
            int attempts = 0;
            do {
                for (int i = 0; i < 7; i++)
                    start_config[i] = std::uniform_real_distribution<float>(
                        sample_min[i], sample_max[i])(rng);
                attempts++;
                if (attempts > 100) break;
            } while (!planner->setStart(start_config));
        }

        result.start_config = start_config;
        robot->setJointValues(start_config);
        Matrix4f start_tcp = robot->getGCPGlobalPose();
        result.tcp_to_apple_start = (start_tcp.block<3,1>(0,3) - scene.apple_pos).norm();

        std::cout << "  Start config: " << start_config.transpose() << "\n";
        std::cout << "  TCP→apple:    " << result.tcp_to_apple_start << " m\n";
        std::cout << "  Planning...   " << std::flush;

        // Plan
        bool success = planner->plan(/*quiet=*/true);
        result.success = success;

        if (success) {
            const auto& perf = planner->getPerformance();
            result.setup_ms          = perf.setup_time_ms;
            result.planning_ms       = perf.planning_time_ms;
            result.rrt_build_ms      = perf.rrt_time_ms;
            result.nn_search_ms      = perf.nearest_neighbor_time_ms;
            result.vf_grad_ms        = perf.vf_gradient_time_ms;
            result.collision_ms      = perf.collision_check_time_ms;
            result.move_to_grasp_ms  = perf.move_toward_grasp_poses_time_ms;
            result.move_to_grasp_tries = perf.number_of_movements_toward_grasp_pose;
            result.grasp_scoring_ms  = perf.score_grasp_time_ms;
            result.grasp_scorings    = perf.number_of_grasp_scorings;
            result.nodes             = planner->getTree()->getNrOfNodes();
            result.cycles            = planner->getCycleCount();

            // Group A
            result.collision_check_count = planner->getCollisionCheckCount();
            result.rejected_configs = result.cycles - result.nodes;
            if (result.nodes > 0) {
                result.cycles_per_node = static_cast<float>(result.cycles) / result.nodes;
            }

            // Group B
            result.grasp_attempts_total = planner->getGraspAttemptTotal();
            result.grasp_goal_reached   = planner->getGraspGoalReachedCount();
            result.grasp_pose_reached   = planner->getGraspPoseReachedCount();
            result.grasp_trapped        = planner->getGraspTrappedCount();
            result.grasp_error          = planner->getGraspErrorCount();

            // Group C
            if (auto* ps = planner->getPoseSphere()) {
                result.faces_total          = ps->getTotalFaceCount();
                result.faces_disabled       = ps->getDisabledFaceCount();
                result.faces_active_final   = ps->getActiveFaceCount();
                result.faces_with_attempts  = ps->getFacesWithAttempts();
                result.max_face_failures    = ps->getMaxFaceFailures();
            }

            auto solution = planner->getSolution();
            result.waypoints = static_cast<int>(solution.size());

            if (!solution.empty()) {
                robot->setJointValues(solution.back());
                Matrix4f final_tcp = robot->getGCPGlobalPose();
                result.final_tcp_to_apple = (final_tcp.block<3,1>(0,3) - scene.apple_pos).norm();
            }

            auto grasps = planner->getGraspResults();
            if (!grasps.empty()) {
                const auto& g = grasps.back();
                result.grasp_distance = g.distance_to_object;
                result.grasp_score    = g.grasp_score;
                result.grasp_node_id  = g.rrt_node_id;
                result.contacts       = static_cast<int>(g.contacts.size());
            }

            // Save trajectory CSV
            char path_filename[128];
            std::snprintf(path_filename, sizeof(path_filename),
                          "%s/grasp_path_%02d.csv", out_dir.c_str(), result.run_idx);
            std::ofstream path_file(path_filename);
            if (path_file.is_open()) {
                for (const auto& wp : solution) {
                    for (int i = 0; i < wp.size(); ++i) {
                        path_file << wp[i] << (i == wp.size() - 1 ? "" : ",");
                    }
                    path_file << "\n";
                }
            }

            // Run TrajOpt (same as example)
            auto t_to_start = std::chrono::high_resolution_clock::now();
            auto optimized = SdfTrajOptimizer::optimize(solution, robot, cost_evaluator);
            auto t_to_end = std::chrono::high_resolution_clock::now();
            result.trajopt_ms =
                std::chrono::duration<float, std::milli>(t_to_end - t_to_start).count();

            std::cout << "✓ SUCCESS  ("
                      << "Plan=" << fmt(result.planning_ms) << "ms, "
                      << "Cycles=" << result.cycles << ", "
                      << "Nodes=" << result.nodes << ", "
                      << "Tries=" << result.move_to_grasp_tries << ")\n";
        } else {
            std::cout << "✗ FAILED   (Cycles=" << planner->getCycleCount()
                      << ", Nodes=" << planner->getTree()->getNrOfNodes() << ")\n";
        }

        // Per-run log
        char log_filename[128];
        std::snprintf(log_filename, sizeof(log_filename),
                      "%s/run%02d_log.txt", out_dir.c_str(), result.run_idx);
        writeRunLog(log_filename, result);

        // Append CSV
        writeCsvRow(csv_file, result);
        csv_file.flush();

        all_results.push_back(result);
    }

    csv_file.close();

    // ---- 12. Final summary ----
    std::cout << "\n┌─────────────────────────────────────────────────────────────────────────┐\n"
              << "│  Benchmark complete                                                      │\n"
              << "└─────────────────────────────────────────────────────────────────────────┘\n";

    std::cout << "\n  Run | Status  | Planning | RRT     | Move→G  | Tries  | Cycles | Nodes\n";
    std::cout << "  ----+---------+----------+---------+---------+--------+--------+------\n";
    int success_count = 0;
    float total_plan_ms = 0;
    float min_plan_ms = std::numeric_limits<float>::max();
    float max_plan_ms = 0;
    for (const auto& r : all_results) {
        std::cout << "   " << std::setw(2) << r.run_idx << " | ";
        if (r.success) {
            std::cout << "SUCCESS | ";
            std::cout << std::setw(7) << std::fixed << std::setprecision(1) << r.planning_ms << "ms | ";
            std::cout << std::setw(7) << r.rrt_build_ms << "ms | ";
            std::cout << std::setw(7) << r.move_to_grasp_ms << "ms | ";
            std::cout << std::setw(6) << r.move_to_grasp_tries << " | ";
            std::cout << std::setw(6) << r.cycles << " | ";
            std::cout << std::setw(5) << r.nodes;
            success_count++;
            total_plan_ms += r.planning_ms;
            min_plan_ms = std::min(min_plan_ms, r.planning_ms);
            max_plan_ms = std::max(max_plan_ms, r.planning_ms);
        } else {
            std::cout << "FAILED  |     NaN  |     NaN |     NaN |    NaN |    NaN |   NaN";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    std::cout << "  Success rate:    " << success_count << "/" << num_runs
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * success_count / num_runs) << "%)\n";
    if (success_count > 0) {
        std::cout << "  Planning (mean): " << (total_plan_ms / success_count) << " ms\n";
        std::cout << "  Planning (min):  " << min_plan_ms << " ms\n";
        std::cout << "  Planning (max):  " << max_plan_ms << " ms\n";
    }
    std::cout << "\n  Results written to: " << out_dir << "/\n"
              << "    benchmark_results.csv  (summary)\n"
              << "    runXX_log.txt          (per-run RESULTS block)\n"
              << "    grasp_path_XX.csv      (successful trajectories)\n";

    return 0;
}
