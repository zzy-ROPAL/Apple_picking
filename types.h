#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <memory>
#include <cmath>
#include <string>

namespace grasp_rrt {

// ============================================================================
// Basic type aliases
// ============================================================================

using Config    = Eigen::VectorXf;           // joint configuration vector
using ConfigPath = std::vector<Config>;      // path of configurations
using Matrix4f  = Eigen::Matrix4f;           // 4x4 homogeneous transform
using Vector3f  = Eigen::Vector3f;           // 3D vector
using MatrixXf  = Eigen::MatrixXf;           // dynamically sized matrix (Jacobians etc.)

// ============================================================================
// Constants
// ============================================================================

constexpr float PI          = 3.14159265359f;
constexpr float EPSILON     = 1e-6f;
constexpr int   RECURSION_MAX_DEPTH = 24;    // maximum DCD recursion depth

// ============================================================================
// RRT extension mode — counterpart of the Python RrtMethod
// ============================================================================

enum class RrtMode {
    EXTEND = 0,            // extend by a single step
    CONNECT = 1,           // extend until collision
    CONNECT_COMPLETE = 2   // full connect: add only if the complete path is valid
};

// ============================================================================
// RRT extension result — counterpart of the Python ExtensionResult
// ============================================================================

enum class ExtensionResult {
    ERROR,     // an error occurred
    FAILED,    // extension failed (collision)
    SUCCESS,   // extension fully succeeded
    PARTIAL    // extension partially succeeded
};

// ============================================================================
// MoveArm result — counterpart of the Python MoveArmResult
// Describes the outcome of a workspace sampling step (differential IK)
// ============================================================================

enum class MoveArmResult {
    MOVED_ONE_STEP,            // moved one step successfully but did not reach the goal
    GOAL_REACHED,              // reached the grasp target pose
    COLLISION_ENVIRONMENT,     // collided with the environment
    GRASPABLE_POSE_REACHED,    // reached a graspable pose (score not high enough)
    JOINT_BOUNDARY_VIOLATION,  // joint limit exceeded
    TRAPPED,                   // stuck in a local minimum
    ERROR,                     // generic error
    FATAL_ERROR                // fatal error
};

// ============================================================================
// Contact-point information — used for grasp-quality evaluation
// ============================================================================

struct ContactInfo {
    Vector3f position  = Vector3f::Zero();   // contact position
    Vector3f normal    = Vector3f::Zero();   // contact normal
    float    force     = 0.0f;               // contact force
    int      finger_id = -1;                 // finger ID
};

// ============================================================================
// GraspInfo — counterpart of the Python GraspInfo
// Records the full information of one successful grasp
// ============================================================================

struct GraspInfo {
    Matrix4f hand_to_object_transform = Matrix4f::Identity();
    int      rrt_node_id              = -1;
    float    grasp_score              = 0.0f;
    float    distance_to_object       = 0.0f;
    std::vector<ContactInfo> contacts;
};

// ============================================================================
// GraspPlanningPerformance — counterpart of the Python PlanningPerformance
// ============================================================================

struct GraspPlanningPerformance {
    float setup_time_ms                          = 0.0f;
    float planning_time_ms                       = 0.0f;
    float move_toward_grasp_poses_time_ms        = 0.0f;
    int   number_of_movements_toward_grasp_pose  = 0;
    float score_grasp_time_ms                    = 0.0f;
    int   number_of_grasp_scorings               = 0;
    float rrt_time_ms                            = 0.0f;

    float collision_check_time_ms                = 0.0f;  // total (extend + move + validate)
    float vf_gradient_time_ms                    = 0.0f;
    float nearest_neighbor_time_ms               = 0.0f;

    // Breakdown: collision checks by origin
    float collision_in_extend_ms                 = 0.0f;  // from extendVFRRT
    float collision_in_move_ms                   = 0.0f;  // from moveTowardsGoal
    float collision_in_validate_ms               = 0.0f;  // from validateSolutionPath

    // Breakdown: inside the grasp hypothesis
    float pose_calc_ms                           = 0.0f;  // calculateGlobalGraspPose

    void reset() {
        setup_time_ms = planning_time_ms = 0.0f;
        move_toward_grasp_poses_time_ms = 0.0f;
        number_of_movements_toward_grasp_pose = 0;
        score_grasp_time_ms = 0.0f;
        number_of_grasp_scorings = 0;
        rrt_time_ms = 0.0f;
        collision_check_time_ms = 0.0f;
        vf_gradient_time_ms = 0.0f;
        nearest_neighbor_time_ms = 0.0f;
        collision_in_extend_ms = 0.0f;
        collision_in_move_ms = 0.0f;
        collision_in_validate_ms = 0.0f;
        pose_calc_ms = 0.0f;
    }
};

// ============================================================================
// GraspRrtConfig — GraspRRT algorithm parameters
// ============================================================================

struct GraspRrtConfig {
    // RRT parameters
    int   max_cycles                 = 50000;
    float extend_step_size           = 0.08f;
    float goal_bias_probability      = 0.1f;
    float planning_timeout_ms        = 60000.0f;
    RrtMode rrt_mode                 = RrtMode::CONNECT;

    // GraspRRT specific
    float probab_grasp_hypothesis    = 0.1f;    // probability of running a grasp hypothesis per round
    float grasp_quality_min_score    = 0.01f;   // minimum score for an acceptable grasp
    float cart_sampling_pos_step_size = 10.0f;  // Cartesian translation step (mm)
    float cart_sampling_ori_step_size = PI / 18; // Cartesian rotation step (rad, ~10 deg)
    float try_grasps_distance2       = 360000.0f; // squared distance threshold for attempting grasps (600^2)
    float max_dist_goal_reached      = 1.0f;    // pose distance below which the goal counts as reached
    int   min_grasp_contacts         = 3;       // minimum number of contact points
    int   move_towards_goal_max_loops = 300;    // maximum moveTowardsGoal iterations
    // DCD parameters
    float sampling_size_paths        = 0.08f;   // path sampling step
    float sampling_size_dcd          = 0.04f;   // DCD collision-checking step

    // Approach face disable mechanism
    bool  face_disable_enabled         = true;  // single switch, convenient for ablation runs
    int   face_disable_failure_threshold = 200; // disable after N failures
    int   face_disable_min_active      = 2;     // always keep at least N active faces (relaxed to 2)
    int   face_disable_min_tree_size   = 500;   // disabling allowed only once the tree has N nodes

    // Vector-field shell attractor (replaces the fixed single-point attraction)
    // The attractor is a spherical cap (a cone below the apple); every N cycles a new point is drawn on it
    bool  use_shell_attractor           = true;  // false -> fall back to a fixed point (8 cm below the apple)
    float shell_attractor_target_radius = 0.04f; // apple radius (shell center)
    float shell_attractor_offset        = 0.01f; // shell distance from the apple surface (= shell_radius - target_radius)
    float shell_attractor_cap_height    = 0.015f;// cap height (how far up from the south pole, default 1.5 cm)
    int   shell_attractor_resample_period = 1000;// resample once every N cycles

    // ---- Extension primitive mode (ablation) ----
    // false (default) : deterministic 50/50 weighted blend
    //                   q_step_dir = normalize(0.5*q_rand_dir + 0.5*vf_dir)
    // true            : stochastic coin flip
    //                   50% probability q_step_dir = q_rand_dir (pure random)
    //                   50% probability q_step_dir = vf_dir     (pure VF)
    bool extension_coin_flip = false;

    // ---- Branch anisotropy (Option A) — DISABLED (dormant) ----
    // Axial drag force on top of the radial push-out:
    //   F_drag = −K_ax · penetration · (v · â) â   (workspace, mapped via J^T)
    // where â comes from findNearestCylinderAxis — a single-nearest-branch
    // argmin that has NOT been retargeted onto the structure tensor. With a
    // discrete â the force field chatters at branch handoffs, the same
    // defect the tensor removed from edge refuse.
    //
    // Status: K_ax = 0 in every current experiment (drag path never runs;
    // the radial repulsion in evaluateAnisotropicForce is unaffected).
    // The --branch-aniso CLI flag errors out. If Option A is ever revived,
    // retarget it first: F_drag = −K_ax · pen · (M̂ · v) using
    // buildLocalAxisTensor. MUST stay 0.0 until then.
    float branch_anisotropy_K_ax        = 0.0f;

    // ---- Branch stiffness (soft-branch stiffness field, pure guidance, no rejection) ----
    // Physical model: a branch is a cantilever — stiff at the root (forcing it snaps),
    // soft at the tip (easy to sweep aside); deeper toward the axis needs more displacement.
    // Encoded as a stiffness weighting of the soft-branch repulsion:
    // force ~ stiffness(t)*depth, t = arc-length fraction (0 root, 1 tip). Axis dominance:
    // as the deepest contact's stiffness*depth activation rises, the repulsion weight in
    // the VF blend climbs from 0 to bs_weight_max (> attraction weight 4.0), so the net
    // field near the axis always points outward. The descent side has the same bias
    // (bs_descent_gain), faded by distance to the apple (bs_descent_fade_r) so a
    // branch-embedded apple stays reachable. Proposal distribution only: C_free and probabilistic completeness unchanged.
    bool  branch_stiffness_enabled = false;
    float bs_k_root        = 3.0f;   // stiffness coefficient at the branch root
    float bs_k_tip         = 0.5f;   // stiffness coefficient at the branch tip
    float bs_shell         = 0.10f;  // thickness of the active shell (m, outside the branch surface)
    float bs_weight_max    = 6.0f;   // upper bound of the axis-dominance weight (> the attraction weight 4.0)
    float bs_descent_gain  = 1.0f;   // strength of the descent-side bias
    float bs_descent_fade_r = 0.12f; // fade radius of the descent bias near the apple (m)

    // ---- Depth steering (proposal-layer companion to the penetration gate, pure guidance) ----
    // The pen gate is a hard rejection in the validity layer; depth steering projects the
    // extension step onto the tangent of the "allowed penetration boundary": for every
    // (probe, soft branch) pair in the contact neighborhood, predict this step's radial
    // deepening and subtract the excess above kappa*slack (slack = allow(t) - current pen)
    // — a step aimed at the root becomes a surface slide, while the tip barely changes.
    // A linearized approximation, not a guarantee: safety still falls back on the gate.
    bool  pen_steer_enabled = false;
    float pen_steer_safety  = 0.8f;   // kappa: max fraction of the remaining allowance consumed per step
    float pen_steer_margin  = 0.03f;  // activation shell: project only when the probe is within margin of the contact surface (m)

    // ---- Unified refuse (unified branch-interaction rule, proposal-layer switch) ----
    // Merges tensor edge refuse (axial) and depth steering (radial) into a single motion
    // cone around the branch axis: severity = inward pressure / depth budget + G * axial
    // slip / slip budget <= 1. G has two regimes: before contact it rises 0->eta with
    // approach (encoding "reach the apple from the side", the way people actually pick),
    // after contact eta->1 with depth (the deeper the press, the less sawing is allowed).
    // Over budget, the pressure and slip components shrink by 1/severity, leaving roughly
    // hoop motion. The positional hard core is still guaranteed by the pen gate.
    bool  uni_refuse_enabled = false;

    // ---- Nav field (plan B: geodesic attraction field, see sdf_cost_evaluator.h) ----
    // Far field (geodesic distance > switch+band): attraction direction = flood downhill
    // (around the canopy, into the funnel). Near field (< switch): fall back to straight
    // attraction toward a random spherical-cap point (uniform coverage of the apple's
    // underside; the attractor does not switch off on arrival). The switch is ALWAYS
    // geodesic, never Euclidean. Descent triggering uses the same geodesic threshold.
    bool  nav_field_enabled = false;
    float nav_switch_dist   = 0.15f;   // near/far switching geodesic threshold (m along the route)
    float nav_blend_band    = 0.10f;   // smoothing bandwidth of the switch (m)
    float nav_descent_dist  = 0.60f;   // geodesic threshold for triggering descent (m, was 600 mm)

    // ---- Finger-first steering (pure guidance, no rejection) ----
    // While descending toward the apple, softly steer so that the fingertips touch first:
    //   (a) rotation bias: turn the gripper axis z_g toward the apple direction u,
    //       d_omega = w(d)*gain*phi*normalize(z_g x u), phi = the included angle;
    //   (b) advance soft-brake: scale the component advancing toward the apple while misaligned
    //       scale = min_advance + (1-min_advance)*cos(phi) (floor 0.3, never blocking);
    //   w(d) = smoothstep, rising to 1 as d goes from ff_far to ff_near (no threshold surface).
    // Motivation: a sideways first touch sweeps a free apple away (measured in Isaac);
    // a hard constraint would add rejection pressure, so this is done purely by steering.
    bool  finger_first_steering = false;
    float ff_align_gain         = 1.0f;
    float ff_far                = 0.15f;   // outer activation radius (m)
    float ff_near               = 0.05f;   // inner radius where w=1 (m)
    float ff_min_advance        = 0.3f;    // minimum retained fraction of the advance component


    // ---- Winner-switch metric (post-hoc instrumentation only) ----
    // Envelope used by SdfCostEvaluator::maxWinnerBranchId, which answers
    // "which branch is the most dangerous here" for the Group E3 oscillation
    // counter (winner_segments / winner_switches / winner_switch_rate) and for
    // the in-envelope test of the axial metrics. It drives NO planner behavior.
    float winner_metric_envelope    = 0.05f;   // 5 cm
};

} // namespace grasp_rrt
