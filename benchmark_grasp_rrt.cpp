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
#include <atomic>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

    // Group D: timing sub-breakdown
    float   collision_extend_ms  = 0;
    float   collision_move_ms    = 0;
    float   collision_validate_ms = 0;
    float   pose_calc_ms         = 0;
    // Derived helpers (computed at fill time)
    float   rrt_other_ms         = 0;   // rrt_build - NN - VF - collision_extend
    float   grasp_hyp_total_ms   = 0;   // move_to_grasp + grasp_scoring
    float   move_other_ms        = 0;   // move_to_grasp - collision_move - pose_calc

    // Group E: path anisotropy (post-hoc instrumentation, ~ms cost, no effect on planning)
    //   axial_contact_fraction = Σ_waypoints Σ_spheres pen · |v·â|  /  Σ pen · |v|
    //   Range [0, 1]. Lower = more perpendicular crossings (preferred by Option A).
    //   -1.0 = no contact along the path (clean path, fraction undefined).
    float   axial_contact_fraction = -1.0f;
    float   total_contact_weight   = 0.0f;  // = Σ pen · |v|  (denominator; 0 = no contact)

    // Group E2: depth-based decomposition (L1, position-only, no velocity)
    //   For each consecutive waypoint pair (i, i+1), each penetrating sphere s:
    //     Δp = p_s(i+1) − p_s(i)
    //     axial_depth_sum  += pen(i+1) · |Δp·â|
    //     radial_depth_sum += pen(i+1) · ‖Δp − (Δp·â)â‖
    //   radial_depth_fraction = radial_depth_sum / (axial_depth_sum + radial_depth_sum)
    //   Satisfies radial + axial = 1. Higher = more radial penetration.
    float   axial_depth_sum         = 0.0f;
    float   radial_depth_sum        = 0.0f;
    float   radial_depth_fraction   = -1.0f;  // -1 = no contact

    // Group E3: winner-switch counter (post-hoc oscillation instrumentation)
    //   Walk the solution path; at each edge compute the MAX-of-K most-
    //   dangerous branch id (maxWinnerBranchId). Count edges that HAVE a
    //   winner, and how many times the winner CHANGES from the previous edge
    //   that had one. switch_rate = winner_switches / max(1, winner_segments-1).
    //   High switch_rate = rejection axis chatters (edge refuse oscillation).
    int     winner_segments  = 0;   // edges with a valid (in-envelope) winner
    int     winner_switches  = 0;   // # of winner-id changes between such edges
    float   winner_switch_rate = -1.0f;  // switches / (segments-1); -1 = n/a

    // Group E4: EE-path smoothness (raw RRT path, pre-trajopt) — the DIRECT
    // wiggle metric. Walk EE workspace positions p_i = FK(waypoint_i):
    //   segment s_i = p_{i+1} - p_i ; turn θ_i = angle(s_{i-1}, s_i)
    //   total_turning = Σ θ_i (deg) ; mean = total / #turns
    //   reversals = # of turns > 90° (sharp back-and-forth)
    // Higher total_turning / more reversals = wigglier (oscillating) path.
    float   path_total_turning_deg = 0.0f;
    float   path_mean_turning_deg  = -1.0f;
    int     path_reversals         = 0;
    // Same, restricted to waypoints in-envelope of a branch (where rejection
    // is active) — isolates wiggle caused by branch interaction near the apple.
    float   path_env_turning_deg   = 0.0f;
    int     path_env_turns         = 0;
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

    auto pct = [&](float t) {
        if (r.planning_ms <= 0 || std::isnan(t)) return std::string("");
        std::ostringstream oss;
        oss << "  (" << std::fixed << std::setprecision(1)
            << (100.0f * t / r.planning_ms) << "%)";
        return oss.str();
    };

    f << "=================== RESULTS ===================\n"
      << "Setup:         " << fmt(r.setup_ms) << " ms\n"
      << "Planning:      " << fmt(r.planning_ms) << " ms\n"
      << "├─ RRT build:           " << fmt(r.rrt_build_ms) << " ms" << pct(r.rrt_build_ms) << "\n"
      << "│   ├─ NN Search:         " << fmt(r.nn_search_ms) << " ms" << pct(r.nn_search_ms) << "\n"
      << "│   ├─ VF Gradient:       " << fmt(r.vf_grad_ms) << " ms" << pct(r.vf_grad_ms) << "\n"
      << "│   ├─ Collision (extend):" << fmt(r.collision_extend_ms) << " ms" << pct(r.collision_extend_ms) << "\n"
      << "│   └─ Other (bookkeeping):" << fmt(r.rrt_other_ms) << " ms" << pct(r.rrt_other_ms) << "\n"
      << "├─ Grasp hypothesis:    " << fmt(r.grasp_hyp_total_ms) << " ms" << pct(r.grasp_hyp_total_ms) << "\n"
      << "│   ├─ Pose calc:         " << fmt(r.pose_calc_ms) << " ms" << pct(r.pose_calc_ms) << "\n"
      << "│   ├─ Collision (move):  " << fmt(r.collision_move_ms) << " ms" << pct(r.collision_move_ms) << "\n"
      << "│   ├─ Grasp scoring:     " << fmt(r.grasp_scoring_ms)
      << " ms (" << r.grasp_scorings << " scorings)" << pct(r.grasp_scoring_ms) << "\n"
      << "│   └─ Jacobian math:     " << fmt(r.move_other_ms) << " ms" << pct(r.move_other_ms) << "\n"
      << "└─ Path validate:       " << fmt(r.collision_validate_ms) << " ms" << pct(r.collision_validate_ms) << "\n"
      << "\n"
      << "Move to grasp count:    " << r.move_to_grasp_tries << " tries\n"
      << "Nodes:                  " << r.nodes << "\n"
      << "Cycles:                 " << r.cycles << "\n"
      << "Cycles/Node:            " << fmt(r.cycles_per_node) << "\n"
      << "Rejected:               " << r.rejected_configs << "\n"
      << "Collision check count:  " << r.collision_check_count << " (total)\n"
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
      << "faces_with_attempts,max_face_failures,"
      // Group D: timing sub-breakdown
      << "collision_extend_ms,collision_move_ms,collision_validate_ms,"
      << "pose_calc_ms,rrt_other_ms,grasp_hyp_total_ms,move_other_ms,"
      // Group E: path anisotropy
      << "axial_contact_fraction,total_contact_weight,"
      // Group E2: depth-based decomposition
      << "radial_depth_fraction,axial_depth_sum,radial_depth_sum,"
      // Group E3: winner-switch oscillation counter
      << "winner_segments,winner_switches,winner_switch_rate,"
      // Group E4: EE-path smoothness (raw path)
      << "path_total_turning_deg,path_mean_turning_deg,path_reversals,"
      << "path_env_turning_deg,path_env_turns\n";
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
      << r.max_face_failures << ",";
    // Group D
    f << fmt(r.collision_extend_ms) << "," << fmt(r.collision_move_ms) << ","
      << fmt(r.collision_validate_ms) << "," << fmt(r.pose_calc_ms) << ","
      << fmt(r.rrt_other_ms) << "," << fmt(r.grasp_hyp_total_ms) << ","
      << fmt(r.move_other_ms) << ",";
    // Group E: path anisotropy
    f << fmt(r.axial_contact_fraction) << "," << fmt(r.total_contact_weight) << ",";
    // Group E2
    f << fmt(r.radial_depth_fraction) << "," << fmt(r.axial_depth_sum) << ","
      << fmt(r.radial_depth_sum) << ",";
    // Group E3: winner-switch oscillation counter
    f << r.winner_segments << "," << r.winner_switches << ","
      << fmt(r.winner_switch_rate) << ",";
    // Group E4: EE-path smoothness
    f << fmt(r.path_total_turning_deg) << "," << fmt(r.path_mean_turning_deg) << ","
      << r.path_reversals << "," << fmt(r.path_env_turning_deg) << ","
      << r.path_env_turns << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    int num_runs = 10;
    std::string urdf_cli;
    bool method_a_enabled = true;
    bool face_disable_enabled = true;
    bool shell_attractor_enabled = true;
    std::string fixed_start_str;
    std::string starts_file;
    std::string label_suffix;
    int num_threads = 1;
    int or_parallel_K = 0;  // 0 = disabled; >1 = OR-Parallel with K trees
    float branch_aniso_K_ax = 0.0f;  // 0 = isotropic legacy; >0 enables Option A axial drag
    bool trajopt_enabled = true;     // --no-trajopt skips the post-process step
    bool dense_twigs = false;        // --dense-twigs adds an engulfment cage around the apple
    bool extension_coin_flip = false; // --coin-flip switches the extension primitive to stochastic 50/50
    bool aniso_test_scene = false;    // --aniso-test-scene: minimal scene with one branch below apple
    bool aniso_horiz_scene = false;   // --aniso-horiz-scene: horizontal cylinder (axis y) wrapping apple
    bool aniso_branch_scene = false;  // --aniso-branch-scene: cylinder grown from trunk toward apple
    bool add_apple_branch   = false;  // --apple-branch: add the thick trunk→apple cylinder ON TOP of standard 10-branch apple-tree scene (apple embedded inside the new branch)
    bool tree6_scene        = false;  // --tree6: reconstructed tree6 continuous scene (8-seg trunk + 23 branches × 5 segs). Apple defaults to the tip of the 5th branch; --apple-root flips it back to 25% along (trunk side).
    bool tree6_apple_root   = false;  // --apple-root: shortcut for --apple-frac 0.25 (only meaningful with --tree6)
    float tree6_apple_frac  = 0.95f;  // --apple-frac F: fraction along the 5th branch arclength (0=root, 1=tip)
    float branch_rscale     = 1.0f;   // --branch-rscale S: multiply every tree6 branch radius by S (2=double). Apple offset tracks the scaled apple-branch radius.
    bool diag_scene         = false;  // --diag-scene: minimal two-orthogonal-branches diagnostic scene for testing sticky-winner. Branch A along x̂ above apple, branch B along ẑ in front of apple.
    bool xgate_scene        = false;  // --xgate: multi-branch scene = V6 rigid A + two CROSSED soft branches (B1 ŷ, B2 x̂) whose 5cm envelopes overlap on the approach corridor. Tests MAX-vs-winner oscillation.
    bool sticky_edge_refuse_enabled = false;  // --sticky-edge-refuse: enable sticky-winner edge refuse (hysteresis + probabilistic edge refuse using single MAX-scored winner)
    bool sticky_sample_steering_enabled   = false;  // --sticky-sample-steering: enable sticky-winner F (hysteresis + deterministic direction bending using single MAX-scored winner)
    float sticky_hysteresis = 1.25f;  // --sticky-hysteresis M: challenger must beat incumbent by this factor to switch
    int   sticky_k          = 3;      // --sticky-k K: how many nearest non-rigid branches to score at each cycle
    float sticky_cap_deg    = 30.0f;  // --sticky-cap DEG: sticky-sample-steering min angle off branch axis (deg). 30=cos0.866, 45=0.707, 60=0.5
    bool edge_refuse_enabled = false;   // --edge-refuse: enable anisotropic edge refuse (Layer 1)
    bool edge_refuse_tensor  = false;   // --edge-refuse-tensor: structure-tensor aggregation instead of MAX (implies --edge-refuse)
    float edge_refuse_envelope  = 0.05f;
    float edge_refuse_threshold = 0.9063f;  // cos(25°): sigmoid midpoint
    float edge_refuse_gain      = 20.0f;    // sigmoid steepness
    int   edge_refuse_k         = 3;        // K nearest non-rigid branches per sphere (MAX mode only)

    // sample steering — angle-range cap on extension direction.
    bool  sample_steering_enabled         = false;
    float sample_steering_envelope        = 0.05f;
    float sample_steering_cos_cap         = 0.866f;   // cos(30°)
    int   sample_steering_k               = 5;        // structure tensor: K=5 nearest
    float sample_steering_min_anisotropy  = 0.3f;     // skip if too isotropic

    // Parse args
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--no-method-a") {
            method_a_enabled = false;
        } else if (a == "--no-face-disable") {
            face_disable_enabled = false;
        } else if (a == "--no-shell") {
            shell_attractor_enabled = false;
        } else if (a == "--start" && i + 1 < argc) {
            fixed_start_str = argv[++i];
        } else if (a == "--starts-file" && i + 1 < argc) {
            starts_file = argv[++i];
        } else if (a == "--label" && i + 1 < argc) {
            label_suffix = argv[++i];
        } else if (a == "--threads" && i + 1 < argc) {
            num_threads = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--or-parallel" && i + 1 < argc) {
            or_parallel_K = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--branch-aniso") {
            // Option A (axial drag force) is disabled: dormant in all current
            // experiments (K_ax = 0), and its axis source is still the
            // single-nearest-branch argmin — NOT retargeted onto the
            // structure tensor. Reviving it without that retarget would
            // reintroduce a chattering force field. The radial repulsion
            // (same function, K_ax = 0 path) is unaffected.
            std::cerr << "ERROR: --branch-aniso is DISABLED (Option A dormant).\n"
                      << "       Anisotropy is handled by --edge-refuse-tensor.\n"
                      << "       If Option A is ever revived, retarget its axis\n"
                      << "       onto buildLocalAxisTensor first.\n";
            return 1;
        } else if (a == "--no-trajopt") {
            trajopt_enabled = false;
        } else if (a == "--dense-twigs") {
            dense_twigs = true;
        } else if (a == "--coin-flip") {
            extension_coin_flip = true;
        } else if (a == "--aniso-test-scene") {
            aniso_test_scene = true;
        } else if (a == "--aniso-horiz-scene") {
            aniso_horiz_scene = true;
        } else if (a == "--aniso-branch-scene") {
            aniso_branch_scene = true;
        } else if (a == "--apple-branch") {
            add_apple_branch = true;
        } else if (a == "--tree6") {
            tree6_scene = true;
        } else if (a == "--apple-root") {
            tree6_apple_root = true;
        } else if (a == "--apple-frac" && i + 1 < argc) {
            tree6_apple_frac = std::max(0.0f, std::min(1.0f,
                                  static_cast<float>(std::atof(argv[++i]))));
        } else if (a == "--branch-rscale" && i + 1 < argc) {
            branch_rscale = std::max(0.1f, static_cast<float>(std::atof(argv[++i])));
        } else if (a == "--diag-scene") {
            diag_scene = true;
        } else if (a == "--xgate") {
            xgate_scene = true;
            diag_scene  = true;   // reuse diag apple/overhead override + geometry gating
        } else if (a == "--sticky-edge-refuse" || a == "--sticky-sample-steering") {
            // Sticky winner is disabled: hysteresis only delays the argmax
            // winner flip, it cannot remove the discontinuity (oscillation
            // persisted in the 10-branch scene). Superseded by the smooth
            // structure-tensor aggregation, which has no winner at all.
            std::cerr << "ERROR: " << a << " is DISABLED (superseded).\n"
                      << "       Use --edge-refuse-tensor instead.\n";
            return 1;
        } else if (a == "--sticky-hysteresis" && i + 1 < argc) {
            sticky_hysteresis = std::max(1.0f,
                                  static_cast<float>(std::atof(argv[++i])));
        } else if (a == "--sticky-k" && i + 1 < argc) {
            sticky_k = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--sticky-cap" && i + 1 < argc) {
            sticky_cap_deg = std::max(0.0f, std::min(89.0f,
                                  static_cast<float>(std::atof(argv[++i]))));
        } else if (a == "--edge-refuse") {
            edge_refuse_enabled = true;
        } else if (a == "--edge-refuse-tensor") {
            edge_refuse_enabled = true;
            edge_refuse_tensor  = true;
        } else if (a == "--edge-refuse-envelope" && i + 1 < argc) {
            edge_refuse_envelope = std::max(0.001f, static_cast<float>(std::atof(argv[++i])));
        } else if (a == "--edge-refuse-threshold" && i + 1 < argc) {
            edge_refuse_threshold = std::max(0.0f, std::min(1.0f,
                                          static_cast<float>(std::atof(argv[++i]))));
        } else if (a == "--edge-refuse-gain" && i + 1 < argc) {
            edge_refuse_gain = std::max(0.1f,
                                          static_cast<float>(std::atof(argv[++i])));
        } else if (a == "--edge-refuse-k" && i + 1 < argc) {
            edge_refuse_k = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--sample-steering") {
            sample_steering_enabled = true;
        } else if (a == "--sample-steering-envelope" && i + 1 < argc) {
            sample_steering_envelope = std::max(0.001f,
                                              static_cast<float>(std::atof(argv[++i])));
        } else if (a == "--sample-steering-cap" && i + 1 < argc) {
            sample_steering_cos_cap = std::max(0.0f, std::min(1.0f,
                                              static_cast<float>(std::atof(argv[++i]))));
        } else if (a == "--sample-steering-k" && i + 1 < argc) {
            sample_steering_k = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--sample-steering-anisotropy" && i + 1 < argc) {
            sample_steering_min_anisotropy = std::max(0.0f, std::min(1.0f,
                                              static_cast<float>(std::atof(argv[++i]))));
        } else {
            positional.push_back(a);
        }
    }
    if (positional.size() > 0) num_runs = std::atoi(positional[0].c_str());
    if (positional.size() > 1) urdf_cli = positional[1];

    // OR-Parallel needs per-thread infrastructure sized for K outer threads.
    // We piggyback on the same per-thread state used by Batch RRT.
    if (or_parallel_K > 1) {
        if (num_threads > 1 && num_threads != or_parallel_K) {
            std::cerr << "WARN: --threads and --or-parallel both set; "
                      << "OR-Parallel overrides, using " << or_parallel_K << " threads.\n";
        }
        num_threads = or_parallel_K;
    }

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
    robot->setNumThreads(num_threads);
    auto [joint_lower, joint_upper] = robot->getJointLimits();

    // ---- 4. C-space ----
    auto collision_checker_ptr = std::make_shared<BulletCollisionChecker>(robot, num_threads);
    auto cspace = std::make_shared<CSpaceSampled>(7, collision_checker_ptr);
    cspace->setBoundaries(joint_lower, joint_upper);

    // ---- Diagnostic scene override (v6) ----
    //   Apple at (0.42, -0.20, 0.95). EE (Start 1) at (0.185, 0.194, 1.10).
    //   Robot must transit ~40 cm in -y direction to reach apple.
    //   Branch A: rigid nub above apple, follows apple to y=-0.20.
    //   Branch B: soft, along ŷ, extended 60 cm (y=-0.30 to 0.30) so it
    //             shields the apple — swinging around either end is not
    //             an option. Robot must push through B.
    // The question the benchmark answers: does baseline push through B
    // with y-dominant (axial) motion, while sticky-winner enforces
    // radial (xz-plane) motion through B?
    if (diag_scene) {
        scene.apple_pos = Vector3f(0.42f, -0.20f, 0.95f);
        scene.overhead.center = Vector3f(100.0f, 100.0f, 100.0f);
        scene.overhead.radius = 0.001f;
    }

    // ---- tree6 scene override: relocate apple, replace standard scene ----
    if (tree6_scene) {
        // --apple-root is a back-compat alias for --apple-frac 0.25.
        if (tree6_apple_root) tree6_apple_frac = 0.25f;

        // Compute apple position from the requested fraction along the 5th branch
        // (index 4) arclength. The branch nodes / radii here must match the
        // BRANCH_NODES / BRANCH_RAD arrays defined later in the tree6 setup.
        Vector3f b5_nodes[6] = {
            Vector3f(0.7328f,-0.0420f,0.8233f), Vector3f(0.6355f,-0.0015f,0.8288f),
            Vector3f(0.5440f, 0.0577f,0.8254f), Vector3f(0.4381f, 0.0931f,0.8176f),
            Vector3f(0.3128f, 0.0886f,0.8117f), Vector3f(0.1912f, 0.0524f,0.8118f),
        };
        float b5_rad[6] = {0.0077f,0.0069f,0.0055f,0.0057f,0.0051f,0.0046f};
        float seg_len[5], total = 0.0f;
        for (int i = 0; i < 5; ++i) {
            seg_len[i] = (b5_nodes[i+1] - b5_nodes[i]).norm();
            total += seg_len[i];
        }
        float target = tree6_apple_frac * total;
        float acc = 0.0f;
        Vector3f attach = b5_nodes[5];
        float local_r = 0.5f * (b5_rad[4] + b5_rad[5]);
        for (int i = 0; i < 5; ++i) {
            if (acc + seg_len[i] >= target) {
                float t = (target - acc) / seg_len[i];
                attach = (1.0f - t) * b5_nodes[i] + t * b5_nodes[i+1];
                local_r = 0.5f * (b5_rad[i] + b5_rad[i+1]);
                break;
            }
            acc += seg_len[i];
        }
        local_r *= branch_rscale;   // apple sits below the (possibly thickened) branch surface
        scene.apple_pos = Vector3f(attach.x(), attach.y(),
                                    attach.z() - local_r - scene.apple_radius);
        // No overhead obstacle in tree6 — push it far so any hint referencing it is a no-op
        scene.overhead.center = Vector3f(100.0f, 100.0f, 100.0f);
        scene.overhead.radius = 0.001f;
    }

    // ---- Helper for arbitrary cylinder pose (segment-aligned) ----
    auto makeCylinderPose = [](const Vector3f& p0, const Vector3f& p1) -> Matrix4f {
        Matrix4f T = Matrix4f::Identity();
        Vector3f mid  = 0.5f * (p0 + p1);
        Vector3f axis = (p1 - p0);
        float L = axis.norm();
        if (L > 1e-9f) {
            axis /= L;
            Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(Vector3f::UnitZ(), axis);
            T.block<3,3>(0,0) = q.toRotationMatrix();
        }
        T.block<3,1>(0,3) = mid;
        return T;
    };

    // ---- 5. Hard collision scene (trunk + overhead) ----
    if (!tree6_scene && !diag_scene) {
        Matrix4f trunk_pose = Matrix4f::Identity();
        trunk_pose.block<3,1>(0, 3) = scene.trunk_center;
        collision_checker_ptr->addObstacleCylinder(scene.trunk_radius, scene.trunk_height / 2.0f, trunk_pose);
        collision_checker_ptr->addObstacleSphere(scene.overhead.radius, scene.overhead.center);
    }

    // 5c-d. Robot collision capsules (mirrors example)
    struct LinkCapsule { std::string name; float radius, center_z, half_h; };
    std::vector<LinkCapsule> capsule_structure = {
        {"panda_link0", 0.10f, 0.05f, 0.02f}, {"panda_link1", 0.08f, 0.15f, 0.12f},
        {"panda_link2", 0.07f, 0.00f, 0.02f}, {"panda_link3", 0.06f, 0.15f, 0.12f},
        {"panda_link4", 0.06f, 0.05f, 0.03f}, {"panda_link5", 0.13f, 0.15f, 0.13f},
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
    if (!tree6_scene && !diag_scene) {
        Vector3f p0 = scene.trunk_center - Vector3f(0, 0, scene.trunk_height / 2.0f);
        Vector3f p1 = scene.trunk_center + Vector3f(0, 0, scene.trunk_height / 2.0f);
        // Trunk is RIGID — anisotropy mechanisms skip it. Robot must avoid,
        // not push through.
        cost_evaluator->addRigidCylinder(p0, p1, scene.trunk_radius);
    }
    // Standard horizontal branches from the trunk — SKIPPED in aniso-test mode
    // to isolate a single branch for anisotropy verification.
    if (!tree6_scene && !diag_scene && !aniso_test_scene && !aniso_horiz_scene && !aniso_branch_scene) {
        for (const auto& br : scene.branches) {
            float ca = std::cos(br.angle), sa = std::sin(br.angle);
            Vector3f p0 = scene.trunk_center;
            p0.z() = br.h;
            p0.x() += scene.trunk_radius * ca;
            p0.y() += scene.trunk_radius * sa;
            Vector3f p1 = p0 + Vector3f(ca, sa, 0) * br.length;
            cost_evaluator->addBranchCylinder(p0, p1, br.radius);
        }
    }
    // --apple-branch: add a thick horizontal cylinder grown from the trunk
    // through the apple, on top of the 10 standard branches. Mirrors the
    // geometry of --aniso-branch-scene (r=0.12 m, length=0.40 m).
    if (add_apple_branch) {
        const Vector3f& apple = scene.apple_pos;
        const Vector3f& trunk_c = scene.trunk_center;
        const float trunk_r = scene.trunk_radius;
        Vector3f trunk_axis_proj(trunk_c.x(), trunk_c.y(), apple.z());
        Vector3f dir = apple - trunk_axis_proj;
        if (dir.norm() < 1e-6f) dir = Vector3f(-1.0f, 0.0f, 0.0f);
        dir.normalize();
        const float cyl_r      = 0.12f;
        const float cyl_length = 0.40f;
        Vector3f p0 = trunk_axis_proj + trunk_r * dir;
        Vector3f p1 = p0 + cyl_length * dir;
        cost_evaluator->addBranchCylinder(p0, p1, cyl_r);
        std::cout << "[Setup] --apple-branch: thick cylinder added through apple\n"
                  << "        p0=" << p0.transpose() << "  p1=" << p1.transpose()
                  << "  r=" << cyl_r << "\n";
    }

    // Overhead obstacle is RIGID (degenerate cylinder = sphere; already skipped
    // by axis-length guard but mark explicitly for clarity).
    if (!tree6_scene && !diag_scene) {
        cost_evaluator->addRigidCylinder(scene.overhead.center, scene.overhead.center, scene.overhead.radius);
    }

    // ---- Diagnostic scene (v6): shield-B forces push-through ----
    //   A: rigid nub above apple, follows apple to y = -0.20
    //   B: SOFT wide shield along ŷ, y ∈ [-0.30, 0.30] — 60 cm long
    if (diag_scene) {
        // Branch A: rigid nub, x ∈ [0.40, 0.55], y = -0.20, z = 1.05
        Vector3f a_p0(0.40f, -0.20f, 1.05f);
        Vector3f a_p1(0.55f, -0.20f, 1.05f);
        float a_r = 0.02f;
        {
            float a_halfLen = 0.5f * (a_p1 - a_p0).norm();
            collision_checker_ptr->addObstacleCylinder(a_r, a_halfLen, makeCylinderPose(a_p0, a_p1));
            cost_evaluator->addBranchCylinder(a_p0, a_p1, a_r);
        }

        if (!xgate_scene) {
            // Branch B: soft shield, x = 0.30, y ∈ [-0.30, 0.30], z = 1.00, r = 0.04
            Vector3f b_p0(0.30f, -0.30f, 1.00f);
            Vector3f b_p1(0.30f,  0.30f, 1.00f);
            float b_r = 0.04f;
            cost_evaluator->addBranchCylinder(b_p0, b_p1, b_r);

            std::cout << "[Setup] --diag-scene v6: A x̂ RIGID (y=-0.20), B ŷ SOFT shield (60 cm), "
                      << "apple at " << scene.apple_pos.transpose() << "\n";
        } else {
            // ---- X-gate: two CROSSED soft branches, non-parallel axes ----
            // B1 along ŷ and B2 along x̂ cross near (0.33, 0, 1.00). Their 5cm
            // envelopes overlap in a ~10cm blob on the EE approach corridor
            // (Start1 y=+0.19 → apple y=-0.20 must sweep through y≈0). The
            // perpendicular axes make the MAX-winner flip B1↔B2 as the EE
            // moves — the oscillation sticky-edge-refuse's hysteresis should suppress.
            float b_r = 0.04f;
            // B1: soft shield along ŷ, x = 0.33, y ∈ [-0.25, 0.25], z = 1.00
            Vector3f b1_p0(0.33f, -0.25f, 1.00f);
            Vector3f b1_p1(0.33f,  0.25f, 1.00f);
            cost_evaluator->addBranchCylinder(b1_p0, b1_p1, b_r);
            // B2: soft branch along x̂, y = 0.00, x ∈ [0.24, 0.46], z = 1.00
            Vector3f b2_p0(0.24f, 0.00f, 1.00f);
            Vector3f b2_p1(0.46f, 0.00f, 1.00f);
            cost_evaluator->addBranchCylinder(b2_p0, b2_p1, b_r);

            std::cout << "[Setup] --xgate: A x̂ RIGID (y=-0.20) + B1 ŷ SOFT + B2 x̂ SOFT "
                      << "crossed at (0.33,0,1.00), apple at "
                      << scene.apple_pos.transpose() << "\n";
        }
    }

    // ---- tree6 reconstructed scene: 8-segment trunk + 23 branches × 5 segments ----
    if (tree6_scene) {
        static const Vector3f TRUNK_NODES[9] = {
            Vector3f(0.7526f,-0.0624f,0.0011f), Vector3f(0.7454f,-0.0440f,0.2991f),
            Vector3f(0.7334f,-0.0345f,0.5977f), Vector3f(0.7319f,-0.0335f,0.9226f),
            Vector3f(0.6726f,-0.0063f,1.1675f), Vector3f(0.6255f, 0.0163f,1.3831f),
            Vector3f(0.7273f, 0.0267f,1.5939f), Vector3f(0.6326f, 0.0541f,1.7930f),
            Vector3f(0.6561f, 0.1058f,2.0094f),
        };
        static const float TRUNK_RAD[9] = {0.0160f,0.0142f,0.0124f,0.0104f,0.0087f,0.0072f,0.0056f,0.0041f,0.0026f};

        for (int i = 0; i < 8; ++i) {
            Vector3f p0 = TRUNK_NODES[i], p1 = TRUNK_NODES[i+1];
            float r = 0.5f * (TRUNK_RAD[i] + TRUNK_RAD[i+1]);
            float halfLen = 0.5f * (p1 - p0).norm();
            collision_checker_ptr->addObstacleCylinder(r, halfLen, makeCylinderPose(p0, p1));
            cost_evaluator->addRigidCylinder(p0, p1, r);
        }

        static const Vector3f BRANCH_NODES[23][6] = {
            {{0.7338f,-0.0364f,0.6730f},{0.6963f, 0.0236f,0.6883f},{0.6504f, 0.0901f,0.6923f},{0.6114f, 0.1617f,0.6865f},{0.5648f, 0.2300f,0.7012f},{0.5171f, 0.2968f,0.7215f}},
            {{0.7341f,-0.0374f,0.6980f},{0.6641f,-0.0716f,0.6774f},{0.6178f,-0.1627f,0.6953f},{0.5486f,-0.2075f,0.6671f},{0.4574f,-0.2419f,0.6056f},{0.3546f,-0.2674f,0.5607f}},
            {{0.7335f,-0.0382f,0.7231f},{0.7821f, 0.0132f,0.7384f},{0.8349f, 0.0744f,0.7424f},{0.8943f, 0.1302f,0.7365f},{0.9486f, 0.1925f,0.7513f},{1.0013f, 0.2554f,0.7716f}},
            {{0.7307f,-0.0393f,0.7733f},{0.6198f,-0.0544f,0.7767f},{0.5131f,-0.0478f,0.7733f},{0.4022f,-0.0703f,0.7646f},{0.2957f,-0.1365f,0.7597f},{0.2079f,-0.2281f,0.7597f}},
            {{0.7328f,-0.0420f,0.8233f},{0.6355f,-0.0015f,0.8288f},{0.5440f, 0.0577f,0.8254f},{0.4381f, 0.0931f,0.8176f},{0.3128f, 0.0886f,0.8117f},{0.1912f, 0.0524f,0.8118f}},
            {{0.7328f,-0.0353f,0.8973f},{0.7047f,-0.1514f,0.8297f},{0.7241f,-0.2402f,0.7218f},{0.7175f,-0.1885f,0.7896f},{0.7677f,-0.1361f,0.7888f},{0.8158f,-0.0801f,0.8041f}},
            {{0.7308f,-0.0326f,0.9480f},{0.7244f,-0.0413f,0.9473f},{0.7179f,-0.0500f,0.9470f},{0.7115f,-0.0579f,0.9476f},{0.7043f,-0.0630f,0.9475f},{0.6975f,-0.0660f,0.9466f}},
            {{0.7296f,-0.0307f,0.9984f},{0.7401f,-0.0973f,0.9858f},{0.7248f,-0.1196f,0.9603f},{0.7172f,-0.1549f,0.9210f},{0.7123f,-0.1910f,0.8752f},{0.7027f,-0.2282f,0.8309f}},
            {{0.7292f,-0.0291f,1.0235f},{0.7921f,-0.0049f,1.0507f},{0.8722f, 0.0098f,1.0371f},{0.9751f, 0.0207f,1.0386f},{1.0928f,-0.0137f,1.0707f},{1.1828f,-0.0535f,1.1150f}},
            {{0.7290f,-0.0249f,1.0736f},{0.6835f, 0.0485f,1.0521f},{0.6157f, 0.1149f,1.0971f},{0.5880f, 0.1374f,1.1261f},{0.5669f, 0.0906f,1.0594f},{0.4825f, 0.1128f,1.0263f}},
            {{0.7140f,-0.0168f,1.1326f},{0.7063f,-0.0522f,1.1151f},{0.6832f,-0.0998f,1.0719f},{0.6702f,-0.1373f,1.0469f},{0.6446f,-0.2074f,1.0120f},{0.6318f,-0.2767f,0.9944f}},
            {{0.6302f, 0.0038f,1.1993f},{0.6715f,-0.0195f,1.2540f},{0.7072f,-0.0558f,1.2990f},{0.7502f,-0.0902f,1.3270f},{0.7862f,-0.1337f,1.3534f},{0.8044f,-0.1540f,1.4100f}},
            {{0.6075f, 0.0127f,1.2480f},{0.6125f,-0.0232f,1.2305f},{0.6075f,-0.0759f,1.1874f},{0.6083f,-0.1155f,1.1623f},{0.6088f,-0.1901f,1.1275f},{0.6209f,-0.2595f,1.1099f}},
            {{0.6106f, 0.0135f,1.2959f},{0.7221f, 0.0449f,1.2733f},{0.8309f, 0.0882f,1.2315f},{0.6704f, 0.0552f,1.2685f},{0.5943f, 0.0756f,1.2557f},{0.5111f, 0.1015f,1.2659f}},
            {{0.6168f, 0.0154f,1.3664f},{0.5907f, 0.0009f,1.3847f},{0.5715f,-0.0166f,1.3821f},{0.5521f,-0.0261f,1.3871f},{0.5183f,-0.0402f,1.3994f},{0.4936f,-0.0564f,1.4108f}},
            {{0.7362f, 0.0169f,1.4796f},{0.7435f, 0.0112f,1.4844f},{0.7497f, 0.0056f,1.4866f},{0.7618f,-0.0086f,1.4872f},{0.7768f,-0.0270f,1.4866f},{0.7887f,-0.0363f,1.4830f}},
            {{0.7381f, 0.0208f,1.5245f},{0.7604f, 0.0407f,1.5428f},{0.7753f, 0.0620f,1.5402f},{0.7922f, 0.0755f,1.5452f},{0.8220f, 0.0966f,1.5576f},{0.8425f, 0.1178f,1.5689f}},
            {{0.7372f, 0.0236f,1.5489f},{0.7464f, 0.0242f,1.5537f},{0.7548f, 0.0243f,1.5559f},{0.7733f, 0.0219f,1.5565f},{0.7968f, 0.0183f,1.5559f},{0.8118f, 0.0194f,1.5523f}},
            {{0.7273f, 0.0267f,1.5939f},{0.7264f, 0.0359f,1.5986f},{0.7262f, 0.0443f,1.6009f},{0.7281f, 0.0628f,1.6015f},{0.7311f, 0.0864f,1.6009f},{0.7296f, 0.1014f,1.5973f}},
            {{0.7033f, 0.0290f,1.6251f},{0.5796f, 0.0899f,1.5926f},{0.3903f, 0.0973f,1.5967f},{0.6138f, 0.0567f,1.5920f},{0.6650f,-0.0006f,1.5778f},{0.7238f,-0.0655f,1.5875f}},
            {{0.6445f, 0.0335f,1.6729f},{0.6474f, 0.0423f,1.6776f},{0.6506f, 0.0500f,1.6799f},{0.6599f, 0.0662f,1.6805f},{0.6722f, 0.0865f,1.6799f},{0.6769f, 0.1008f,1.6763f}},
            {{0.6348f, 0.0468f,1.7469f},{0.6327f, 0.0466f,1.7580f},{0.6287f, 0.0453f,1.7690f},{0.6256f, 0.0443f,1.7803f},{0.6220f, 0.0421f,1.7913f},{0.6180f, 0.0392f,1.8021f}},
            {{0.6317f, 0.0577f,1.8163f},{0.6305f, 0.0596f,1.8166f},{0.6271f, 0.0652f,1.8177f},{0.6237f, 0.0712f,1.8179f},{0.6201f, 0.0770f,1.8179f},{0.6171f, 0.0832f,1.8178f}},
        };
        static const float BRANCH_RAD[23][6] = {
            {0.0061f,0.0059f,0.0053f,0.0049f,0.0042f,0.0036f},{0.0081f,0.0071f,0.0059f,0.0048f,0.0036f,0.0025f},
            {0.0057f,0.0059f,0.0053f,0.0049f,0.0042f,0.0036f},{0.0058f,0.0058f,0.0053f,0.0046f,0.0040f,0.0033f},
            {0.0077f,0.0069f,0.0055f,0.0057f,0.0051f,0.0046f},{0.0085f,0.0066f,0.0041f,0.0037f,0.0031f,0.0023f},
            {0.0050f,0.0053f,0.0044f,0.0036f,0.0028f,0.0023f},{0.0039f,0.0046f,0.0039f,0.0032f,0.0023f,0.0016f},
            {0.0069f,0.0067f,0.0061f,0.0056f,0.0049f,0.0043f},{0.0040f,0.0054f,0.0048f,0.0045f,0.0040f,0.0035f},
            {0.0067f,0.0061f,0.0052f,0.0042f,0.0032f,0.0022f},{0.0052f,0.0050f,0.0046f,0.0042f,0.0037f,0.0033f},
            {0.0065f,0.0061f,0.0052f,0.0042f,0.0032f,0.0022f},{0.0057f,0.0046f,0.0033f,0.0022f,0.0019f,0.0016f},
            {0.0068f,0.0061f,0.0053f,0.0046f,0.0036f,0.0029f},{0.0051f,0.0052f,0.0045f,0.0036f,0.0027f,0.0020f},
            {0.0053f,0.0050f,0.0043f,0.0037f,0.0029f,0.0023f},{0.0049f,0.0052f,0.0045f,0.0036f,0.0027f,0.0020f},
            {0.0052f,0.0047f,0.0041f,0.0033f,0.0024f,0.0018f},{0.0112f,0.0105f,0.0096f,0.0037f,0.0049f,0.0047f},
            {0.0047f,0.0042f,0.0037f,0.0030f,0.0022f,0.0016f},{0.0044f,0.0039f,0.0035f,0.0030f,0.0026f,0.0021f},
            {0.0035f,0.0038f,0.0035f,0.0032f,0.0029f,0.0027f},
        };
        // Branch index 4 (5th from bottom) is the apple-attached branch.
        // Make it RIGID — hard collision in Bullet + rigid in SDF (anisotropy skips).
        const int APPLE_BRANCH_IDX = 4;
        int n_soft = 0, n_rigid = 0;
        for (int b = 0; b < 23; ++b) {
            bool is_apple_branch = (b == APPLE_BRANCH_IDX);
            for (int i = 0; i < 5; ++i) {
                Vector3f p0 = BRANCH_NODES[b][i], p1 = BRANCH_NODES[b][i+1];
                float r = 0.5f * (BRANCH_RAD[b][i] + BRANCH_RAD[b][i+1]) * branch_rscale;
                if (is_apple_branch) {
                    float halfLen = 0.5f * (p1 - p0).norm();
                    collision_checker_ptr->addObstacleCylinder(r, halfLen, makeCylinderPose(p0, p1));
                    cost_evaluator->addRigidCylinder(p0, p1, r);
                    ++n_rigid;
                } else {
                    cost_evaluator->addBranchCylinder(p0, p1, r);
                    ++n_soft;
                }
            }
        }
        std::cout << "[Setup] --tree6: 8 trunk segs (hard+SDF), "
                  << n_rigid << " rigid apple-branch segs (hard+SDF), "
                  << n_soft << " soft branch segs (SDF only), apple at "
                  << scene.apple_pos.transpose() << "\n";
    }

    // ---- Anisotropy test scene: single VERTICAL cylinder wrapping the
    //      apple (--aniso-test-scene) ----
    // Vertical axis along z. The robot's natural downward descent motion is
    // therefore axial relative to the cylinder axis, so the axial drag term
    // F_drag should activate strongly. Without anisotropy, the planner can
    // enter from the top opening; with anisotropy, it should prefer to push
    // through the side wall (radial direction).
    if (aniso_test_scene) {
        const Vector3f& apple = scene.apple_pos;
        const float cyl_r       = 0.12f;    // cylinder radius (12 cm)
        const float cyl_z_half  = 0.10f;    // ±10 cm along z; total length 20 cm
        Vector3f p0(apple.x(), apple.y(), apple.z() - cyl_z_half);
        Vector3f p1(apple.x(), apple.y(), apple.z() + cyl_z_half);
        cost_evaluator->addBranchCylinder(p0, p1, cyl_r);
        std::cout << "[Setup] Anisotropy test scene: VERTICAL cylinder wrapping apple\n"
                  << "        axis = z-direction, p0=" << p0.transpose()
                  << " p1=" << p1.transpose() << " r=" << cyl_r << "\n";
    }

    // ---- Anisotropy test scene: HORIZONTAL cylinder (axis y) wrapping apple ----
    if (aniso_horiz_scene) {
        const Vector3f& apple = scene.apple_pos;
        const float cyl_r       = 0.12f;
        const float cyl_y_half  = 0.10f;
        Vector3f p0(apple.x(), apple.y() - cyl_y_half, apple.z());
        Vector3f p1(apple.x(), apple.y() + cyl_y_half, apple.z());
        cost_evaluator->addBranchCylinder(p0, p1, cyl_r);
        std::cout << "[Setup] Anisotropy test scene: HORIZONTAL cylinder wrapping apple\n"
                  << "        axis = y-direction, p0=" << p0.transpose()
                  << " p1=" << p1.transpose() << " r=" << cyl_r << "\n";
    }

    // ---- Anisotropy test scene: branch grown FROM trunk toward (and past) apple ----
    // p0 sits on the trunk surface at the apple's height; p1 extends outward
    // along the trunk → apple direction, length 0.20 m. The apple ends up
    // inside the cylinder near the outer tip, so the robot must enter the
    // cylinder from outside (radially through the side wall) to grasp.
    if (aniso_branch_scene) {
        const Vector3f& apple = scene.apple_pos;
        const Vector3f& trunk_c = scene.trunk_center;
        const float trunk_r = scene.trunk_radius;

        Vector3f trunk_axis_proj(trunk_c.x(), trunk_c.y(), apple.z());
        Vector3f dir = apple - trunk_axis_proj;
        if (dir.norm() < 1e-6f) dir = Vector3f(-1.0f, 0.0f, 0.0f);
        dir.normalize();

        const float cyl_r      = 0.12f;
        const float cyl_length = 0.40f;     // doubled from 0.20 m
        Vector3f p0 = trunk_axis_proj + trunk_r * dir;     // on trunk surface
        Vector3f p1 = p0 + cyl_length * dir;               // outward, past apple
        cost_evaluator->addBranchCylinder(p0, p1, cyl_r);
        std::cout << "[Setup] Anisotropy test scene: branch GROWN FROM TRUNK\n"
                  << "        axis dir=" << dir.transpose() << "\n"
                  << "        p0(trunk surface) = " << p0.transpose() << "\n"
                  << "        p1(outer tip)     = " << p1.transpose() << "\n"
                  << "        radius=" << cyl_r << " length=" << cyl_length << "\n";
    }

    // ---- Optional engulfment cage around the apple (--dense-twigs) ----
    // 24 short vertical twigs arranged in 3 z-rings around the apple,
    // 8 twigs per ring at 45° spacing on a radius-0.10m circle in the XY plane.
    // Twigs are SOFT (cost-only); robot must displace them to reach the apple.
    if (dense_twigs) {
        int n_twigs = 0;
        const float ring_r = 0.10f;      // radial distance from apple to each twig
        const float twig_r = 0.02f;      // twig radius
        const float half_h = 0.03f;      // twig half-height (so total length = 0.06m)
        const Vector3f& apple = scene.apple_pos;
        const float rings[] = {0.76f, 0.80f, 0.84f};
        for (float z : rings) {
            for (int k = 0; k < 8; ++k) {
                float ang = k * (M_PI / 4.0f);   // 0, 45, 90, ..., 315 deg
                float cx = apple.x() + ring_r * std::cos(ang);
                float cy = apple.y() + ring_r * std::sin(ang);
                Vector3f p0(cx, cy, z - half_h);
                Vector3f p1(cx, cy, z + half_h);
                cost_evaluator->addBranchCylinder(p0, p1, twig_r);
                ++n_twigs;
            }
        }
        std::cout << "[Setup] Dense engulfment cage: " << n_twigs
                  << " twigs added around apple at " << apple.transpose() << "\n";
    }

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

    // ---- Anisotropic edge refuse (Layer 1, pre-contact directional filter) ----
    if (edge_refuse_enabled) {
        cspace->setEdgeRefuseEnabled(true);
        cspace->setEdgeRefuseEnvelope(edge_refuse_envelope);
        cspace->setEdgeRefuseThreshold(edge_refuse_threshold);
        cspace->setEdgeRefuseGain(edge_refuse_gain);
        cspace->setEdgeRefuseK(edge_refuse_k);
        cspace->setEdgeRefuseTensorMode(edge_refuse_tensor);
        std::cout << "[Setup] Edge refuse ENABLED ("
                  << (edge_refuse_tensor
                        ? "STRUCTURE-TENSOR aggregation, no winner selection"
                        : "legacy MAX aggregation")
                  << "):\n"
                  << "        envelope  = " << edge_refuse_envelope << " m\n"
                  << "        threshold = " << edge_refuse_threshold
                  << " (cos midpoint)\n"
                  << "        gain      = " << edge_refuse_gain
                  << " (sigmoid steepness)\n";
        if (!edge_refuse_tensor) {
            std::cout << "        K nearest = " << edge_refuse_k
                      << " (non-rigid branches per sphere)\n";
        }
        std::cout << "        spheres   = panda_hand + panda_link6\n";
    }

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
    config.use_shell_attractor = shell_attractor_enabled;
    config.branch_anisotropy_K_ax = branch_aniso_K_ax;   // Option A: axial drag on/off
    config.extension_coin_flip = extension_coin_flip;    // ablation: stochastic switch vs deterministic blend
    config.sample_steering_enabled         = sample_steering_enabled;   // sample steering
    config.sample_steering_envelope        = sample_steering_envelope;
    config.sample_steering_cos_cap         = sample_steering_cos_cap;
    config.sample_steering_k               = sample_steering_k;
    config.sample_steering_min_anisotropy  = sample_steering_min_anisotropy;
    // Sticky-winner variants
    config.sticky_sample_steering_enabled              = sticky_sample_steering_enabled;
    config.sticky_edge_refuse_enabled            = sticky_edge_refuse_enabled;
    config.sticky_hysteresis_margin      = sticky_hysteresis;
    config.sticky_k                      = sticky_k;
    config.sticky_cos_cap                = std::cos(sticky_cap_deg * static_cast<float>(M_PI) / 180.0f);
    if (sticky_sample_steering_enabled) {
        std::cout << "[Setup] STICKY-F ENABLED (hysteresis margin=" << sticky_hysteresis
                  << ", K=" << sticky_k << ", cos_cap=" << config.sticky_cos_cap
                  << ", envelope=" << config.sticky_envelope << ")\n";
    }
    if (sticky_edge_refuse_enabled) {
        std::cout << "[Setup] STICKY-edge refuse ENABLED (hysteresis margin=" << sticky_hysteresis
                  << ", K=" << sticky_k << ", threshold=" << config.sticky_edge_refuse_threshold
                  << ", gain=" << config.sticky_edge_refuse_gain << ")\n";
    }
    if (sticky_sample_steering_enabled && sticky_edge_refuse_enabled) {
        std::cerr << "WARN: both --sticky-sample-steering and --sticky-edge-refuse set; sticky-sample-steering takes precedence.\n";
    }
    if (sample_steering_enabled) {
        std::cout << "[Setup] Sample steering (structure tensor) ENABLED:\n"
                  << "        envelope        = " << sample_steering_envelope << " m\n"
                  << "        cos_cap         = " << sample_steering_cos_cap
                  << " (min angle ≈ "
                  << static_cast<int>(std::acos(sample_steering_cos_cap) * 180.0f / M_PI)
                  << "° off axis)\n"
                  << "        K nearest       = " << sample_steering_k << "\n"
                  << "        min anisotropy  = " << sample_steering_min_anisotropy
                  << " (skip if (λmax−λmin)/λmax below)\n";
    }

    // ---- 9. Output directory ----
    std::string out_dir = method_a_enabled
        ? "benchmark_results_with_A"
        : "benchmark_results_no_A";
    if (!face_disable_enabled) out_dir += "_noFD";
    if (!shell_attractor_enabled) out_dir += "_noShell";
    if (!label_suffix.empty()) out_dir += "_" + label_suffix;
    mkdir(out_dir.c_str(), 0755);  // ignore EEXIST
    std::cout << "[Setup] Output directory: " << out_dir << "/\n";

    std::ofstream csv_file(out_dir + "/benchmark_results.csv");
    if (!csv_file.is_open()) {
        std::cerr << "ERROR: Cannot open benchmark_results.csv\n";
        return 1;
    }
    writeCsvHeader(csv_file);

    // 自动保存每个 run 的起点 (方便后续用 --starts-file 复现)
    std::ofstream starts_out(out_dir + "/starts_used.txt");

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
        planner->setNumThreads(num_threads);  // Batch RRT (K=1 = serial)

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

        // 保存这个 run 的起点 (CSV 一行)
        if (starts_out.is_open()) {
            for (int j = 0; j < 7; ++j) {
                starts_out << start_config[j] << (j == 6 ? "\n" : ",");
            }
            starts_out.flush();
        }

        std::cout << "  Start config: " << start_config.transpose() << "\n";
        std::cout << "  TCP→apple:    " << result.tcp_to_apple_start << " m\n";
        std::cout << "  Planning...   " << std::flush;

        // ---- Plan ----
        bool success = false;
        if (or_parallel_K > 1) {
            // OR-Parallel: K independent GraspRrt instances race; first success wins.
            // Each instance has its own tree/face_state, but they share cspace/robot/
            // collision_checker through per-thread state (per_thread_ arrays indexed
            // by omp_get_thread_num()).
            const int K = or_parallel_K;
            std::vector<std::unique_ptr<GraspRrt>> planners(K);
            for (int k = 0; k < K; ++k) {
                planners[k] = std::make_unique<GraspRrt>(
                    cspace, robot, eef, target, quality, config);
                planners[k]->setVerbose(false);
                planners[k]->setHardObstacle(scene.overhead.center, /*trigger_dist=*/0.15f);
                planners[k]->addExtraSetupTimeMs(voxel_build_ms);
                planners[k]->setGoalBiasProb(method_a_enabled ? 0.15f : 0.0f);
                planners[k]->setNumThreads(1);  // each tree runs serial K=1 internally
                if (!planners[k]->setStart(start_config)) {
                    std::cerr << "ERROR: planner[" << k << "] setStart failed\n";
                }
            }

            std::atomic<bool> stop_flag{false};
            std::atomic<int>  winner_id{-1};
            std::vector<char> succ(K, 0);

            for (int k = 0; k < K; ++k) {
                planners[k]->setExternalStopFlag(&stop_flag);
            }

            auto t_race_start = std::chrono::high_resolution_clock::now();
            #ifdef _OPENMP
            #pragma omp parallel for num_threads(K) schedule(static)
            #endif
            for (int k = 0; k < K; ++k) {
                bool ok = planners[k]->plan(/*quiet=*/true);
                succ[k] = ok ? 1 : 0;
                if (ok) {
                    int expected = -1;
                    if (winner_id.compare_exchange_strong(expected, k)) {
                        // first one to find a solution
                        stop_flag.store(true, std::memory_order_relaxed);
                    }
                }
            }
            auto t_race_end = std::chrono::high_resolution_clock::now();
            float race_wall_ms = std::chrono::duration<float, std::milli>(
                t_race_end - t_race_start).count();

            int w = winner_id.load();
            if (w >= 0) {
                success = true;
                planner = std::move(planners[w]);
            } else {
                // nobody succeeded — keep planner[0] for stats reporting
                success = false;
                planner = std::move(planners[0]);
            }
            // Override the winner's reported planning time with race wall-clock
            // (= time until ALL K threads exited; = honest end-to-end latency).
            const_cast<GraspPlanningPerformance&>(planner->getPerformance())
                .planning_time_ms = race_wall_ms;
        } else {
            success = planner->plan(/*quiet=*/true);
        }
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

            // Group D: timing sub-breakdown
            result.collision_extend_ms   = perf.collision_in_extend_ms;
            result.collision_move_ms     = perf.collision_in_move_ms;
            result.collision_validate_ms = perf.collision_in_validate_ms;
            result.pose_calc_ms          = perf.pose_calc_ms;
            result.rrt_other_ms          = std::max(0.0f, result.rrt_build_ms
                                                          - result.nn_search_ms
                                                          - result.vf_grad_ms
                                                          - result.collision_extend_ms);
            result.grasp_hyp_total_ms    = result.move_to_grasp_ms + result.grasp_scoring_ms;
            result.move_other_ms         = std::max(0.0f, result.move_to_grasp_ms
                                                          - result.collision_move_ms
                                                          - result.pose_calc_ms);

            auto solution = planner->getSolution();
            result.waypoints = static_cast<int>(solution.size());

            if (!solution.empty()) {
                robot->setJointValues(solution.back());
                Matrix4f final_tcp = robot->getGCPGlobalPose();
                result.final_tcp_to_apple = (final_tcp.block<3,1>(0,3) - scene.apple_pos).norm();
            }

            // ---- Path anisotropy instrumentation (post-hoc, <1ms) ----
            // Measures axial vs radial branch-contact fraction along the
            // solution path. Lower = more perpendicular crossings (Option A goal).
            if (auto sdf_eval = std::dynamic_pointer_cast<SdfCostEvaluator>(cost_evaluator)) {
                double axial_sum = 0.0, total_sum = 0.0;
                for (size_t i = 1; i + 1 < solution.size(); ++i) {
                    Config v_joint = solution[i+1] - solution[i-1];
                    auto contrib = sdf_eval->measureWaypointAnisotropy(solution[i], v_joint);
                    axial_sum += contrib.first;
                    total_sum += contrib.second;
                }
                result.total_contact_weight = static_cast<float>(total_sum);
                if (total_sum > 1e-9) {
                    result.axial_contact_fraction =
                        static_cast<float>(axial_sum / total_sum);
                }
                // else: path has no branch contact → axial_contact_fraction = -1 (default)

                // ---- Group E2: depth-based decomposition (position-only) ----
                double ax_depth = 0.0, rad_depth = 0.0;
                for (size_t i = 0; i + 1 < solution.size(); ++i) {
                    auto d = sdf_eval->measureWaypointAnisotropyDisplacement(
                        solution[i], solution[i+1]);
                    ax_depth  += d.first;   // axial pen-weighted disp
                    rad_depth += d.second;  // radial pen-weighted disp
                }
                result.axial_depth_sum  = static_cast<float>(ax_depth);
                result.radial_depth_sum = static_cast<float>(rad_depth);
                if (ax_depth + rad_depth > 1e-12) {
                    result.radial_depth_fraction =
                        static_cast<float>(rad_depth / (ax_depth + rad_depth));
                }

                // ---- Group E3: winner-switch counter (oscillation, post-hoc) ----
                // For each path edge, find the MAX-of-K most-dangerous branch.
                // Count edges with a valid winner and how many times the winner
                // changes between consecutive in-envelope edges. This is the
                // de-facto rejection axis for edge refuse; a high change-rate through
                // the crossing is the direct signature of oscillation.
                int prev_winner = -1, segs = 0, switches = 0;
                for (size_t i = 0; i + 1 < solution.size(); ++i) {
                    int w = sdf_eval->maxWinnerBranchId(
                        solution[i], solution[i+1],
                        config.sticky_envelope, sticky_k);
                    if (w < 0) continue;   // no branch in envelope on this edge
                    if (prev_winner >= 0 && w != prev_winner) ++switches;
                    ++segs;
                    prev_winner = w;
                }
                result.winner_segments = segs;
                result.winner_switches = switches;
                if (segs > 1) {
                    result.winner_switch_rate =
                        static_cast<float>(switches) / static_cast<float>(segs - 1);
                }

                // ---- Group E4: EE-path smoothness (the direct wiggle metric) ----
                // Compute EE workspace positions, then integrated turning angle
                // and reversals along the raw RRT path. Also accumulate turning
                // restricted to waypoints in-envelope of a branch.
                std::vector<Vector3f> ee_pos(solution.size());
                for (size_t i = 0; i < solution.size(); ++i) {
                    robot->setJointValues(solution[i]);
                    ee_pos[i] = robot->getGCPGlobalPose().block<3,1>(0,3);
                }
                double total_turn = 0.0, env_turn = 0.0;
                int    n_turns = 0, reversals = 0, env_turns = 0;
                for (size_t i = 1; i + 1 < ee_pos.size(); ++i) {
                    Vector3f s0 = ee_pos[i]   - ee_pos[i-1];
                    Vector3f s1 = ee_pos[i+1] - ee_pos[i];
                    float n0 = s0.norm(), n1 = s1.norm();
                    if (n0 < 1e-6f || n1 < 1e-6f) continue;
                    float c = std::max(-1.0f, std::min(1.0f, s0.dot(s1) / (n0 * n1)));
                    float ang_deg = std::acos(c) * 180.0f / static_cast<float>(M_PI);
                    total_turn += ang_deg;
                    ++n_turns;
                    if (c < 0.0f) ++reversals;   // turn > 90°
                    // In-envelope? use the winner query as a cheap "near a branch" test
                    int w = sdf_eval->maxWinnerBranchId(
                        solution[i-1], solution[i],
                        config.sticky_envelope, sticky_k);
                    if (w >= 0) { env_turn += ang_deg; ++env_turns; }
                }
                result.path_total_turning_deg = static_cast<float>(total_turn);
                result.path_reversals         = reversals;
                if (n_turns > 0) {
                    result.path_mean_turning_deg =
                        static_cast<float>(total_turn / n_turns);
                }
                result.path_env_turning_deg = static_cast<float>(env_turn);
                result.path_env_turns       = env_turns;
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

            // Dump full RRT tree for offline visualization
            char tree_filename[128];
            std::snprintf(tree_filename, sizeof(tree_filename),
                          "%s/tree_dump_%02d.csv", out_dir.c_str(), result.run_idx);
            planner->dumpTree(tree_filename);

            // Dump pose_sphere face statistics (for visualizing disabled faces etc.)
            char face_filename[128];
            std::snprintf(face_filename, sizeof(face_filename),
                          "%s/face_stats_%02d.csv", out_dir.c_str(), result.run_idx);
            planner->dumpFaceStats(face_filename);

            // Run TrajOpt (same as example) — skipped when --no-trajopt is set.
            // Note: axial_contact_fraction is measured BEFORE trajopt (line ~820),
            // so toggling trajopt does not change the path-anisotropy metric.
            if (trajopt_enabled) {
                auto t_to_start = std::chrono::high_resolution_clock::now();
                auto optimized = SdfTrajOptimizer::optimize(solution, robot, cost_evaluator);
                auto t_to_end = std::chrono::high_resolution_clock::now();
                result.trajopt_ms =
                    std::chrono::duration<float, std::milli>(t_to_end - t_to_start).count();
            } else {
                result.trajopt_ms = 0.0f;
            }

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
        // 收集所有成功 run 的各项指标
        auto collect = [&](auto getter) {
            std::vector<float> v;
            v.reserve(all_results.size());
            for (const auto& r : all_results) if (r.success) v.push_back(getter(r));
            std::sort(v.begin(), v.end());
            return v;
        };
        auto stats = [](const std::vector<float>& v) {
            struct S { float mean, median, min, max; } s{};
            if (v.empty()) return s;
            float sum = 0;
            for (float x : v) sum += x;
            s.mean = sum / v.size();
            s.median = v[v.size() / 2];
            s.min = v.front();
            s.max = v.back();
            return s;
        };

        auto plan_v   = collect([](const RunResult& r){ return r.planning_ms; });
        auto rrt_v    = collect([](const RunResult& r){ return r.rrt_build_ms; });
        auto nn_v     = collect([](const RunResult& r){ return r.nn_search_ms; });
        auto vf_v     = collect([](const RunResult& r){ return r.vf_grad_ms; });
        auto colE_v   = collect([](const RunResult& r){ return r.collision_extend_ms; });
        auto rrtO_v   = collect([](const RunResult& r){ return r.rrt_other_ms; });
        auto gh_v     = collect([](const RunResult& r){ return r.grasp_hyp_total_ms; });
        auto pc_v     = collect([](const RunResult& r){ return r.pose_calc_ms; });
        auto colM_v   = collect([](const RunResult& r){ return r.collision_move_ms; });
        auto gs_v     = collect([](const RunResult& r){ return r.grasp_scoring_ms; });
        auto mvO_v    = collect([](const RunResult& r){ return r.move_other_ms; });
        auto colV_v   = collect([](const RunResult& r){ return r.collision_validate_ms; });
        auto colT_v   = collect([](const RunResult& r){ return r.collision_ms; });
        auto cycles_v = collect([](const RunResult& r){ return static_cast<float>(r.cycles); });
        auto nodes_v  = collect([](const RunResult& r){ return static_cast<float>(r.nodes); });
        auto tries_v  = collect([](const RunResult& r){ return static_cast<float>(r.move_to_grasp_tries); });

        auto p    = stats(plan_v);
        auto rt   = stats(rrt_v);
        auto nn   = stats(nn_v);
        auto vf   = stats(vf_v);
        auto colE = stats(colE_v);
        auto rrtO = stats(rrtO_v);
        auto gh   = stats(gh_v);
        auto pc   = stats(pc_v);
        auto colM = stats(colM_v);
        auto gs   = stats(gs_v);
        auto mvO  = stats(mvO_v);
        auto colV = stats(colV_v);
        auto colT = stats(colT_v);
        auto cy   = stats(cycles_v);
        auto nd   = stats(nodes_v);
        auto tr   = stats(tries_v);

        std::cout << "\n  Statistics over " << success_count << " successful runs:\n";
        std::cout << "  " << std::setfill('-') << std::setw(86) << "" << std::setfill(' ') << "\n";
        std::cout << "  " << std::setw(28) << std::left << "Metric"
                  << std::setw(13) << std::right << "Mean"
                  << std::setw(13) << "Median"
                  << std::setw(13) << "Min"
                  << std::setw(13) << "Max" << "\n";
        std::cout << "  " << std::setfill('-') << std::setw(86) << "" << std::setfill(' ') << "\n";
        std::cout << std::fixed << std::setprecision(2);
        auto row = [](const std::string& name, const auto& s, const std::string& unit) {
            std::cout << "  " << std::setw(28) << std::left << name
                      << std::setw(11) << std::right << s.mean << unit
                      << std::setw(11) << s.median << unit
                      << std::setw(11) << s.min << unit
                      << std::setw(11) << s.max << unit << "\n";
        };
        row("Planning",                p,    " ms");
        row("├─ RRT build",            rt,   " ms");
        row("│   ├─ NN Search",        nn,   " ms");
        row("│   ├─ VF Gradient",      vf,   " ms");
        row("│   ├─ Collision (extend)", colE, " ms");
        row("│   └─ Other (bookkeep.)", rrtO, " ms");
        row("├─ Grasp hypothesis",     gh,   " ms");
        row("│   ├─ Pose calc",        pc,   " ms");
        row("│   ├─ Collision (move)", colM, " ms");
        row("│   ├─ Grasp scoring",    gs,   " ms");
        row("│   └─ Jacobian math",    mvO,  " ms");
        row("└─ Path validate",        colV, " ms");
        row("Collision total",         colT, " ms");

        std::cout << std::setprecision(0);
        row("Cycles",          cy, "   ");
        row("Nodes",           nd, "   ");
        row("Grasp tries",     tr, "   ");
    }

    std::cout << "\n  Results written to: " << out_dir << "/\n"
              << "    benchmark_results.csv  (summary)\n"
              << "    runXX_log.txt          (per-run RESULTS block)\n"
              << "    grasp_path_XX.csv      (successful trajectories)\n"
              << "    starts_used.txt        (all start configs, for --starts-file)\n";

    return 0;
}
