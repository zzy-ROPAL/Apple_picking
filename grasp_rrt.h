#pragma once

#include "types.h"
#include "cspace_node.h"
#include "cspace_sampled.h"
#include "cspace_tree.h"
#include "approach_discretization.h"
#include "collision_checker.h"
#include "robot_interface.h"

#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <iostream>
#include <limits>

namespace grasp_rrt {

// ============================================================================
// GraspRrt — counterpart of the Python GraspRrt (derived from Rrt)
// Core planner: generates and scores grasp hypotheses while the RRT explores
//
// Reference paper: "Integrated Grasp and Motion Planning"
//           N. Vahrenkamp, M. Do, T. Asfour, R. Dillmann, ICRA 2010
// ============================================================================

class GraspRrt {
public:
    /**
     * @param cspace        configuration space (with DCD collision checking)
     * @param robot         robot interface (FK/IK/Jacobian)
     * @param eef           end-effector interface (gripper control)
     * @param target        target-object interface
     * @param quality       grasp-quality evaluator
     * @param config        algorithm parameters
     */
    GraspRrt(std::shared_ptr<CSpaceSampled> cspace,
             std::shared_ptr<RobotInterface> robot,
             std::shared_ptr<EndEffectorInterface> eef,
             std::shared_ptr<TargetObjectInterface> target,
             std::shared_ptr<GraspQualityInterface> quality,
             const GraspRrtConfig& config = GraspRrtConfig());

    ~GraspRrt() = default;

    // ---- Configuration ----

    void setConfig(const GraspRrtConfig& cfg) { config_ = cfg; }
    const GraspRrtConfig& getConfig() const { return config_; }
    void setVerbose(bool v) { verbose_ = v; }

    // ---- Additional rigid obstacles ----

    void setHardObstacle(const Vector3f& pos, float trigger_dist = 0.20f) {
        hard_obstacle_pos_     = pos;
        hard_obstacle_trigger_ = trigger_dist;
        hard_obstacle_enabled_ = true;
    }
    void clearHardObstacle() { hard_obstacle_enabled_ = false; }

    // ---- Start configuration ----

    /**
     * Set the start configuration (counterpart of the Python GraspRrt.setStart)
     * Note: GraspRRT needs no externally set goal — goals are sampled during planning
     */
    bool setStart(const Config& start_config);

    // ---- Main planning interface ----

    /**
     * Run GraspRRT planning (counterpart of the Python GraspRrt.plan)
     * Blocking; returns true on success
     */
    bool plan(bool quiet = false);

    // ---- Result access ----

    ConfigPath getSolution() const { return solution_; }

    // ------------------------------------------------------------------
    // Method 1: score-aware shortcutting (path pruning, NOT TrajOpt)
    //
    // Repeatedly tries to "skip intermediate waypoints and connect directly" on the
    // solution path. A direct connection replaces the detour only if, split at planning
    // granularity (extend_step_size), every sub-segment passes exactly the same validity
    // checks used while growing the tree (DCD collision + edge-refuse rejection). No cost
    // function, no gradient, no deformation — pure pruning, so every edge of the final
    // path passed the planner's own checks and the method guarantee (monitored spheres
    // never travel along a branch axis) is preserved by construction.
    // ------------------------------------------------------------------
    std::vector<Config> shortcutSolutionPath(const std::vector<Config>& path,
                                             int max_rounds = 20);
    std::vector<GraspInfo> getGraspResults() const;
    int getGraspResultCount() const;
    const GraspPlanningPerformance& getPerformance() const { return performance_; }
    int getCycleCount() const { return cycles_; }
    int getCollisionCheckCount() const { return col_checks_overall_; }
    CSpaceTree* getTree() { return tree_.get(); }
    ApproachDiscretization* getPoseSphere() { return pose_sphere_.get(); }

    // Grasp attempt result breakdown (Group B metrics)
    int getGraspGoalReachedCount()  const { return n_grasp_goal_reached_; }
    int getGraspPoseReachedCount()  const { return n_grasp_pose_reached_; }
    int getGraspTrappedCount()      const { return n_grasp_trapped_; }
    int getGraspErrorCount()        const { return n_grasp_error_; }
    int getGraspAttemptTotal()      const {
        return n_grasp_goal_reached_ + n_grasp_pose_reached_ +
               n_grasp_trapped_ + n_grasp_error_;
    }

    /**
     * Dump the whole RRT tree to CSV for offline visualization of tree growth
     * Fields: node_id,parent_id,j0..j6,tcp_x,tcp_y,tcp_z,in_pose_sphere,on_solution_path
     */
    void dumpTree(const std::string& filepath);

    /**
     * Dump per-face statistics of pose_sphere to CSV
     * Fields: face_id,attempts,failures,disabled,node_count,dir_x,dir_y,dir_z
     */
    void dumpFaceStats(const std::string& filepath);

    // Add an extra setup-time contribution (e.g. SDF voxel build).
    // Will be added to setup_time_ms at the end of init().
    void addExtraSetupTimeMs(float ms) { extra_setup_time_ms_ += ms; }

    // Tune Method A (goal-biased sampling). Set p=0.0 to disable.
    void setGoalBiasProb(float p) { goal_bias_prob_ = p; }

    // ---- Repair mode (sim-to-real path repair) ----
    //
    // Turns the planner from "grow to a new grasp" into "reconnect to the path we
    // were already executing". Instead of a single goal, it accepts reaching ANY
    // member of `goals`, which must be the longest still-valid SUFFIX of the path
    // under repair — goals[k] is only usable because goals[k..end] has already been
    // re-validated against the edited scene, so joining at k and running to the end
    // is safe by construction.
    //
    // Preference is for the SMALLEST reachable k, which preserves the most of the
    // original path. connect_radius is in C-space; a candidate within it is accepted
    // only if the straight edge passes cspace_->isPathValid, so the splice inherits
    // the penetration gate exactly like a planned edge.
    //
    // In repair mode the grasp-hypothesis strategy is skipped: every cycle goes into
    // tree growth, because the goal is the suffix, not a new grasp. To keep the
    // "unrestricted replan" behaviour on some instances of an OR-parallel race,
    // simply do not call this on them.
    void setRepairGoals(const std::vector<Config>& goals,
                        float connect_radius,
                        float goal_sample_prob = 0.3f);

    // Index into the goal set that was reached, or -1. The caller splices
    // [connector solution] + [original waypoints from this index onward].
    int getRepairJoinedIndex() const { return repair_joined_; }
    bool isRepairMode() const { return repair_mode_; }

    // ---- Start-in-violation search ----
    //
    // Lets the tree be rooted at a pose that violates an INSERTED repair capsule (see
    // SdfCostEvaluator::penViolation). Motivation: that capsule's forbidden region is a
    // budget we created one cycle ago, at a place the arm physically occupies and reached
    // safely; refusing to start there is the model overruling reality, and it is the whole
    // of the no-valid-retreat failure class (first_invalid_wp == 0).
    //
    // Mechanics: every node carries v(q) = penViolation(q). A node with v = 0 extends under
    // the ordinary gate. A node with v > 0 extends with the HARD_BRANCH allowance relaxed by
    // exactly its own v, and the child is kept ONLY if v(child) < v(parent) — so the tree can
    // sit in the violation but may only move out of it, and reverts to the ordinary gate the
    // moment it reaches v = 0. Soft-branch allow(t) is never relaxed on any node.
    //
    // Naively relaxing just the root check does not work: one extension step cannot leave a
    // 20-30 mm violation, so the second node fails the strict gate and the tree never grows.
    // The grading is what makes it a search rather than a single hop.
    void setStartInViolation(bool on) { start_in_violation_ = on; }
    // v(root) in meters, 0 if the start was ordinarily valid. Telemetry for the harness.
    float getStartViolation() const { return start_violation_; }
    // Nodes whose violation was > 0 (i.e. how much of the tree lived inside the capsule).
    int getViolatingNodeCount() const { return violating_nodes_; }

    // ---- Batch RRT (parallel extend, serial commit) ----
    //
    // Set how many threads each planning round uses for extension. 1 = serial (default).
    // Precondition: the collision checker of robot_ / cspace_ must already have the
    // matching per-thread state (PandaPinocchioRobot::setNumThreads, BulletCollisionChecker(num_threads)).
    void setNumThreads(int n) { num_threads_ = (n > 1) ? n : 1; }
    int  getNumThreads() const { return num_threads_; }

    // Parallelism of shortcut pruning (independent of num_threads_: under OR-parallel each
    // tree runs serially with num_threads_=1, but once the race is over the other cores are
    // idle while the winner is pruned, so all per-thread slots of the robot / collision
    // checker can be used). n must not exceed the thread count passed at construction.
    void setShortcutThreads(int n) { shortcut_threads_ = (n > 1) ? n : 1; }

    // ---- OR-Parallel external stop signal ----
    // When set, the planning loop polls this flag and exits early when true.
    // Used by OR-Parallel mode to cancel losing trees once any tree finds a solution.
    void setExternalStopFlag(std::atomic<bool>* f) { external_stop_flag_ = f; }

    // ---- Reset ----

    void reset();

private:
    // ---- Initialization (counterpart of the Python GraspRrt.init) ----
    bool init();

    // ---- Single planning round (counterpart of the Python doPlanningCycle) ----
    bool doPlanningCycle();
    bool doPlanningCycleBatch(int K);

    // ---- Extension methods ----

    ExtensionResult extendVFRRT(const Config& target);

    // Batch-friendly: pure computation (NN/VF/collision), never writes tree/perf counters.
    // The returned candidate is committed serially by the main thread.
    struct ExtendCandidate {
        bool       valid = false;
        int        nn_id = -1;
        Config     new_config;
        float      edge_cost = 0.0f;
        float      nn_time_ms = 0.0f;
        float      vf_time_ms = 0.0f;
        float      collision_time_ms = 0.0f;
        int        col_check_count_delta = 0;
        // v(new_config) under start-in-violation; 0 in ordinary operation.
        float      violation = 0.0f;
    };
    ExtendCandidate extendVFRRTCompute(const Config& target);

    // motion_dir: intended extension direction in joint space (e.g., q_rand−q_near).
    // Used by Option A branch anisotropy to apply axial drag. Pass an empty Config()
    // (or omit) to disable the drag and recover the legacy isotropic behavior.
    Config computeVectorField(const Config& q, CSpaceNodePtr nn,
                              const Config& motion_dir = Config());

    // Shell attractor: the main thread decides from shell_resample_counter_/period whether
    // to resample cached_shell_target_. computeVectorField only reads it.
    void maybeResampleShellAttractor();

    // ---- Path validation (lazy collision check) ----
    bool validateSolutionPath(CSpaceNodePtr goal_node);

    // Per-segment validity check used by shortcutSolutionPath: split [a,b] into
    // extend_step_size pieces and call cspace_->isPathValid on each
    // (= DCD + edge refuse, same granularity and semantics as tree growth).
    bool segmentValidPiecewise(const Config& a, const Config& b);

    // ---- Grasp-direction guidance ----

    /**
     * Counterpart of the Python connectRandomGraspPositionJacobian
     * Pick a node from pose_sphere, compute the grasp pose, step under Jacobian guidance
     */
    MoveArmResult connectRandomGraspPositionJacobian();

    /**
     * Decide from the face's failure statistics plus the safeguard whether to disable it
     */
    void tryDisableFace(int face_id);

    /**
     * Counterpart of the Python calculateGlobalGraspPose
     * Generate candidate grasp poses on the object surface
     */
    Matrix4f calculateGlobalGraspPose(const Config& config, bool& success);

    // ---- Jacobian-guided stepping ----

    /**
     * Counterpart of the Python moveTowardsGoal
     * Iteratively step from startNode toward targetPose using DiffIK
     */
    MoveArmResult moveTowardsGoal(CSpaceNodePtr start_node,
                                   const Matrix4f& target_pose);

    /**
     * Counterpart of the Python createWorkSpaceSamplingStep
     * Compute the current -> goal delta, clamp the step, convert to joint increments via DiffIK
     */
    std::pair<MoveArmResult, Config> createWorkspaceSamplingStep(
        const Matrix4f& current_pose, const Matrix4f& target_pose);

    /**
     * Counterpart of the Python createWorkSpaceSamplingStepFromNode
     */
    std::pair<MoveArmResult, Config> createWorkspaceSamplingStepFromNode(
        const Matrix4f& goal_pose, CSpaceNodePtr extend_node);

    /**
     * Counterpart of the Python moveArmDiffKin
     * One differential IK step: J+ * e -> dtheta
     */
    std::pair<MoveArmResult, Config> moveArmDiffKin(const Matrix4f& delta_pose);

    /**
     * Counterpart of the Python limitWorkspaceStep
     * Clamp the Cartesian step to a preset maximum
     */
    Matrix4f limitWorkspaceStep(const Matrix4f& delta_pose);

    // ---- Grasp scoring ----

    /**
     * Counterpart of the Python calculateGraspScore
     * Close the gripper -> detect contacts -> force-closure scoring
     */
    float calculateGraspScore(const Config& config, int node_id,
                              bool store_on_success = false);

    // ---- Node processing ----

    /** Counterpart of the Python processNode: cache the TCP pose, register into pose_sphere */
    bool processNode(CSpaceNodePtr node);

    /** Method A: build a target from the closest_below node plus jitter */
    Config sampleGoalBiased();

    /** Counterpart of the Python processNodes: process every node on the path */
    bool processNodes(int start_id, int end_id);

    // ---- Solution-path construction ----

    bool createSolution(bool quiet = false);

    // ---- Jacobian pseudoinverse (damped) ----

    MatrixXf computePseudoInverseJacobian();

    // ---- Utility functions ----

    /** Cartesian pose distance (position + weighted angle) */
    static float cartesianPoseDiff(const Matrix4f& p1, const Matrix4f& p2);

    /** Axis-angle -> 4x4 homogeneous matrix */
    static Matrix4f axisAngleToMatrix4(const Vector3f& axis, float angle);

    /** Extract axis-angle from a rotation matrix */
    static void rotationToAxisAngle(const Eigen::Matrix3f& R,
                                     Vector3f& axis, float& angle);

    /** Printing */
    void printPerformanceResults() const;
    void printGraspInfo(const GraspInfo& info) const;

    // ======== Member variables ========

    // Core components
    std::shared_ptr<CSpaceSampled>         cspace_;
    std::shared_ptr<RobotInterface>        robot_;
    std::shared_ptr<EndEffectorInterface>  eef_;
    std::shared_ptr<TargetObjectInterface>  target_object_;
    std::shared_ptr<GraspQualityInterface>  quality_measure_;

    // RRT tree
    std::unique_ptr<CSpaceTree> tree_;
    CSpaceNodePtr start_node_;
    CSpaceNodePtr goal_node_;

    // Discretized approach-direction sphere
    std::unique_ptr<ApproachDiscretization> pose_sphere_;

    // Configuration
    GraspRrtConfig config_;
    int dimension_;

    // State
    bool planner_initialized_;
    bool found_solution_;
    bool stop_search_;
    bool verbose_;
    int  cycles_;
    int  last_added_id_;
    int  col_checks_overall_;

    // Additional rigid obstacles
    Vector3f hard_obstacle_pos_     = Vector3f::Zero();
    float    hard_obstacle_trigger_ = 0.20f;
    bool     hard_obstacle_enabled_ = false;

    // Cached J_T (7x6)
    MatrixXf cached_J_T_;
    bool     cached_J_T_valid_ = false;

    // RRT*
    float best_goal_cost_ = 1e9f;
    std::unordered_map<int, Matrix4f> map_config_tcp_;

    // Cached target-object position
    Vector3f target_object_position_;

    // Grasp results
    std::vector<GraspInfo> grasps_;
    std::mutex grasp_mutex_;

    // Solution path
    ConfigPath solution_;

    // Performance statistics
    GraspPlanningPerformance performance_;
    float extra_setup_time_ms_ = 0.0f;  // SDF voxel build, etc. (added in init)

    // Grasp attempt result counters (Group B metrics)
    int n_grasp_goal_reached_ = 0;
    int n_grasp_pose_reached_ = 0;
    int n_grasp_trapped_ = 0;
    int n_grasp_error_ = 0;

    // Shell attractor (vector field): cache + resample counter
    Vector3f cached_shell_target_      = Vector3f::Zero();
    int      shell_resample_counter_   = 0;

    // Goal-biased sampling (Method A): track best below-equator node
    int   closest_below_node_id_ = -1;
    float closest_below_dist2_   = std::numeric_limits<float>::max();
    float goal_bias_prob_        = 0.15f;   // 15% of cycles use goal-biased target

    // ---- Repair mode state (see setRepairGoals) ----
    bool                repair_mode_       = false;
    std::vector<Config> repair_goals_;
    float               repair_connect_r2_ = 0.0f;   // squared C-space connect radius
    float               repair_goal_prob_  = 0.3f;   // fraction of cycles aimed at the suffix
    int                 repair_joined_     = -1;

    // After each successful extension, try to close onto the goal set. Sets
    // goal_node_ / found_solution_ / repair_joined_ on success.
    bool tryConnectRepairGoal();

    // Sample a target near the early part of the goal set (a rejoin as close to the
    // retreat pose as possible preserves the most of the original path).
    Config sampleRepairGoalBiased();
    float goal_bias_jitter_      = 0.25f;   // joint-space jitter scale (rad)

    // ---- Start-in-violation state (see setStartInViolation) ----
    bool  start_in_violation_ = false;
    float start_violation_    = 0.0f;
    int   violating_nodes_    = 0;
    // v per node, indexed by node id (ids are sequential from CSpaceTree::appendNode).
    // Written only during the serial commit, read during the parallel extend compute, and
    // the two phases never overlap — so a plain vector needs no synchronization.
    std::vector<float> node_violation_;
    float nodeViolation(int id) const {
        return (id >= 0 && id < static_cast<int>(node_violation_.size()))
             ? node_violation_[static_cast<size_t>(id)] : 0.0f;
    }
    void setNodeViolation(int id, float v) {
        if (id < 0) return;
        if (static_cast<size_t>(id) >= node_violation_.size())
            node_violation_.resize(static_cast<size_t>(id) + 1, 0.0f);
        node_violation_[static_cast<size_t>(id)] = v;
        if (v > 0.0f) ++violating_nodes_;
    }
    // A child must beat its parent by at least this much (m) to count as progress out of the
    // violation. Guards against an infinite chain of ~0 improvements.
    static constexpr float kViolationDecrease = 1e-4f;   // 0.1 mm

    // Batch RRT
    int num_threads_ = 1;
    int shortcut_threads_ = 1;   // parallelism of shortcut pruning (see setShortcutThreads)

    // OR-Parallel external stop flag (nullable, owned by caller)
    std::atomic<bool>* external_stop_flag_ = nullptr;
};

} // namespace grasp_rrt
