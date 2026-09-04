#include "grasp_rrt.h"
#include "panda_pinocchio.h"
#include "sdf_cost_evaluator.h"
#include "rng_util.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <unordered_set>

namespace grasp_rrt {

// ============================================================================
// Constructor
// ============================================================================

GraspRrt::GraspRrt(std::shared_ptr<CSpaceSampled> cspace,
                   std::shared_ptr<RobotInterface> robot,
                   std::shared_ptr<EndEffectorInterface> eef,
                   std::shared_ptr<TargetObjectInterface> target,
                   std::shared_ptr<GraspQualityInterface> quality,
                   const GraspRrtConfig& config)
    : cspace_(cspace), robot_(robot), eef_(eef),
      target_object_(target), quality_measure_(quality),
      config_(config),
      dimension_(cspace->getDimension()),
      planner_initialized_(false),
      found_solution_(false),
      stop_search_(false),
      verbose_(false),
      cycles_(0),
      last_added_id_(-1),
      col_checks_overall_(0) {
    // Sync the DCD parameters into CSpaceSampled
    cspace_->setSamplingSize(config_.sampling_size_paths);
    cspace_->setSamplingSizeDCD(config_.sampling_size_dcd);

    target_object_position_ = target_object_->getPosition();
}

// ============================================================================
// setStart — counterpart of the Python GraspRrt.setStart
// ============================================================================

bool GraspRrt::setStart(const Config& start_config) {
    if (!cspace_->isConfigValid(start_config)) {
        std::cerr << "[GraspRrt] ERROR: Start config is invalid (collision or out of bounds)\n";
        return false;
    }

    // Initialize the tree
    tree_ = std::make_unique<CSpaceTree>(cspace_.get());
    start_node_ = tree_->appendNode(start_config, -1);
    start_node_->setStatus(0);

    // Process the start node (cache its TCP pose)
    processNode(start_node_);

    if (verbose_) {
        std::cout << "[GraspRrt] setStart: " << start_config.transpose() << "\n";
    }
    return true;
}

// ============================================================================
// reset — counterpart of the Python GraspRrt.reset
// ============================================================================

void GraspRrt::reset() {
    solution_.clear();
    start_node_ = nullptr;
    goal_node_ = nullptr;
    pose_sphere_.reset();
    performance_.reset();
    planner_initialized_ = false;
    found_solution_ = false;
    stop_search_ = false;
    cycles_ = 0;
    col_checks_overall_ = 0;
    map_config_tcp_.clear();
    grasps_.clear();
    if (tree_) tree_->reset();
}

// ============================================================================
// init — counterpart of the Python GraspRrt.init
// ============================================================================

bool GraspRrt::init() {
    performance_.reset();
    found_solution_ = false;
    col_checks_overall_ = 0;
    solution_.clear();

    auto t_start = std::chrono::high_resolution_clock::now();

    if (!start_node_) {
        std::cerr << "[GraspRrt] ERROR: start node not set\n";
        return false;
    }

    cycles_ = 0;
    stop_search_ = false;
    grasps_.clear();

    // Reset goal-biased sampling bookkeeping (Method A)
    closest_below_node_id_ = -1;
    closest_below_dist2_   = std::numeric_limits<float>::max();

    // Reset grasp attempt counters (Group B metrics)
    n_grasp_goal_reached_ = 0;
    n_grasp_pose_reached_ = 0;
    n_grasp_trapped_      = 0;
    n_grasp_error_        = 0;

    // Reset shell attractor cache
    cached_shell_target_     = Vector3f::Zero();
    shell_resample_counter_  = 0;

    // Initialize the discretized approach-direction sphere
    pose_sphere_ = std::make_unique<ApproachDiscretization>();
    pose_sphere_->setGlobalPose(target_object_->getGlobalPose());
    target_object_position_ = target_object_->getPosition();

    planner_initialized_ = true;

    auto t_end = std::chrono::high_resolution_clock::now();
    performance_.setup_time_ms =
        std::chrono::duration<float, std::milli>(t_end - t_start).count()
        + extra_setup_time_ms_;
    return true;
}

// ============================================================================
// plan — the core main loop (counterpart of the Python GraspRrt.plan)
// ============================================================================

bool GraspRrt::plan(bool quiet) {
    if (!planner_initialized_) {
        if (!init()) return false;
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    bool stop_loop = false;

    while (!stop_loop) {
        if (num_threads_ > 1) doPlanningCycleBatch(num_threads_);
        else                  doPlanningCycle();

        if (stop_search_)  stop_loop = true;
        if (found_solution_) stop_loop = true;
        if (cycles_ > config_.max_cycles) stop_loop = true;
        if (external_stop_flag_ && external_stop_flag_->load(std::memory_order_relaxed)) {
            stop_loop = true;
        }

        // Timeout check
        auto t_now = std::chrono::high_resolution_clock::now();
        float elapsed_ms = std::chrono::duration<float, std::milli>(t_now - t_start).count();
        if (elapsed_ms > config_.planning_timeout_ms) {
            std::cout << "[GraspRrt] Timeout (" << config_.planning_timeout_ms << " ms)\n";
            stop_loop = true;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    performance_.planning_time_ms =
        std::chrono::duration<float, std::milli>(t_end - t_start).count();

    // Compute the RRT tree-building time
    performance_.rrt_time_ms =
        performance_.planning_time_ms - performance_.move_toward_grasp_poses_time_ms;
    performance_.move_toward_grasp_poses_time_ms -= performance_.score_grasp_time_ms;

    if (!quiet) printPerformanceResults();

    if (found_solution_) {
        if (!quiet) {
            std::cout << "[GraspRrt] Found path in " << cycles_ << " cycles\n";
        }
        return createSolution(quiet);
    }

    if (cycles_ >= config_.max_cycles) {
        std::cout << "[GraspRrt] Max cycles exceeded\n";
    }
    return false;
}

// ============================================================================
// doPlanningCycle — a single planning round (counterpart of the Python doPlanningCycle)
// ============================================================================

bool GraspRrt::doPlanningCycle() {
    if (!planner_initialized_) return false;

    int col_checks_start = cspace_->getCollisionCheckCount();
    bool do_normal_connect = true;

    maybeResampleShellAttractor();

    float r = randFloat01();

    if (r <= config_.probab_grasp_hypothesis) {
        // ===== Strategy A: probabilistically triggered terminal grasp descent =====
        do_normal_connect = false;
        MoveArmResult grasp_status = connectRandomGraspPositionJacobian();

        if (grasp_status == MoveArmResult::FATAL_ERROR) {
            stop_search_ = true;
        } else if (grasp_status == MoveArmResult::GRASPABLE_POSE_REACHED) {
            std::cout << "[GraspRrt] -- FOUND GRASPABLE POSITION (score not high enough) --\n";
            if (!grasps_.empty()) printGraspInfo(grasps_.back());
        } else if (grasp_status == MoveArmResult::GOAL_REACHED) {
            if (grasps_.empty()) {
                std::cerr << "[GraspRrt] ERROR: no grasp results stored\n";
                stop_search_ = true;
            } else {
                const auto& grasp_info = grasps_.back();
                auto gn = tree_->getNode(grasp_info.rrt_node_id);
                if (gn) {
                    float current_cost = gn->getCostFromRoot();
                    if (!found_solution_ || current_cost < best_goal_cost_) {
                        best_goal_cost_ = current_cost;
                        goal_node_ = gn;
                        found_solution_ = true;
                        std::cout << "[GraspRRT] New Best Path Found! Cost: " << current_cost << "\n";
                        printGraspInfo(grasp_info);
                    }
                }
            }
        }
    }

    if (do_normal_connect) {
        // ===== Stage one: VF-RRT tree extension (with goal-biased sampling) =====
        Config rand_config;
        float r2 = randFloat01();
        if (r2 < goal_bias_prob_) {
            rand_config = sampleGoalBiased();   // Method A
        } else {
            rand_config = cspace_->getRandomConfig();
        }
        ExtensionResult ext_status = extendVFRRT(rand_config);
        if (ext_status == ExtensionResult::ERROR) {
            stop_search_ = true;
        }
    }

    cycles_++;
    col_checks_overall_ += (cspace_->getCollisionCheckCount() - col_checks_start);

    if (cycles_ % 50 == 0) {
        std::cout << "[GraspRrt] Cycle " << cycles_ << "/" << config_.max_cycles
                  << ", Nodes: " << tree_->getNrOfNodes()
                  << ", Grasps: " << grasps_.size() << "\n";
    }
    return true;
}

// ============================================================================
// maybeResampleShellAttractor — called serially from the main thread
// ============================================================================
void GraspRrt::maybeResampleShellAttractor() {
    if (!config_.use_shell_attractor) return;

    const float shell_radius = config_.shell_attractor_target_radius
                              + config_.shell_attractor_offset;
    const float cap_height   = config_.shell_attractor_cap_height;
    const float u_min = std::max(0.0f, (shell_radius - cap_height) / shell_radius);

    bool need_resample = (shell_resample_counter_ % config_.shell_attractor_resample_period == 0)
                       || cached_shell_target_.norm() < 1e-9f;
    if (need_resample) {
        Vector3f candidate = Vector3f::Zero();
        const int MAX_RETRIES = 10;
        for (int retry = 0; retry < MAX_RETRIES; ++retry) {
            float u   = u_min + (1.0f - u_min) * randFloat01();
            float phi = 2.0f * static_cast<float>(M_PI) * randFloat01();
            float sin_theta = std::sqrt(1.0f - u * u);
            candidate = target_object_position_ + Vector3f(
                shell_radius * sin_theta * std::cos(phi),
                shell_radius * sin_theta * std::sin(phi),
                -shell_radius * u
            );
            if (pose_sphere_) {
                int fid = pose_sphere_->getNearestFaceId(candidate);
                if (!pose_sphere_->isFaceDisabled(fid)) break;
            } else break;
        }
        cached_shell_target_ = candidate;
    }
    shell_resample_counter_++;
}

// ============================================================================
// doPlanningCycleBatch — Batch RRT main loop
// The main thread pre-samples K q_rand values -> K parallel extend computations -> serial commit on the main thread.
// The grasp-hypothesis branch stays on the serial path (outside the parallel section).
// ============================================================================
bool GraspRrt::doPlanningCycleBatch(int K) {
    if (!planner_initialized_) return false;
    if (K <= 1) return doPlanningCycle();

    int col_checks_start = cspace_->getCollisionCheckCount();
    maybeResampleShellAttractor();

    // 1) Grasp-hypothesis branch (serial) — consistent with doPlanningCycle.
    //    The main-thread RNG decides whether to trigger; if so, one grasp attempt runs alone with no batch extend.
    float r = randFloat01();
    if (r <= config_.probab_grasp_hypothesis) {
        MoveArmResult grasp_status = connectRandomGraspPositionJacobian();
        if (grasp_status == MoveArmResult::FATAL_ERROR) {
            stop_search_ = true;
        } else if (grasp_status == MoveArmResult::GRASPABLE_POSE_REACHED) {
            std::cout << "[GraspRrt] -- FOUND GRASPABLE POSITION (score not high enough) --\n";
            if (!grasps_.empty()) printGraspInfo(grasps_.back());
        } else if (grasp_status == MoveArmResult::GOAL_REACHED) {
            if (grasps_.empty()) {
                std::cerr << "[GraspRrt] ERROR: no grasp results stored\n";
                stop_search_ = true;
            } else {
                const auto& grasp_info = grasps_.back();
                auto gn = tree_->getNode(grasp_info.rrt_node_id);
                if (gn) {
                    float current_cost = gn->getCostFromRoot();
                    if (!found_solution_ || current_cost < best_goal_cost_) {
                        best_goal_cost_ = current_cost;
                        goal_node_ = gn;
                        found_solution_ = true;
                        std::cout << "[GraspRRT] New Best Path Found! Cost: " << current_cost << "\n";
                        printGraspInfo(grasp_info);
                    }
                }
            }
        }
        cycles_++;
        col_checks_overall_ += (cspace_->getCollisionCheckCount() - col_checks_start);
        return true;
    }

    // 2) Pre-sample K targets on main thread (serial, deterministic order).
    std::vector<Config> targets(K);
    for (int k = 0; k < K; ++k) {
        float r2 = randFloat01();
        if (r2 < goal_bias_prob_) targets[k] = sampleGoalBiased();
        else                      targets[k] = cspace_->getRandomConfig();
    }

    // 3) Parallel extend compute.
    std::vector<ExtendCandidate> cands(K);
#ifdef _OPENMP
    #pragma omp parallel for num_threads(K) schedule(static)
#endif
    for (int k = 0; k < K; ++k) {
        cands[k] = extendVFRRTCompute(targets[k]);
    }

    // 4) Serial commit + perf accumulation.
    for (int k = 0; k < K; ++k) {
        const auto& c = cands[k];
        performance_.nearest_neighbor_time_ms += c.nn_time_ms;
        performance_.vf_gradient_time_ms      += c.vf_time_ms;
        performance_.collision_check_time_ms  += c.collision_time_ms;
        performance_.collision_in_extend_ms   += c.collision_time_ms;

        if (!c.valid || c.nn_id < 0) continue;

        // nn_id was looked up against tree pre-batch; the tree only grew since, so
        // the node is still valid (ids are stable).
        auto nn = tree_->getNode(c.nn_id);
        if (!nn) continue;
        float cost = nn->getCostFromRoot() + c.edge_cost;
        auto new_node = tree_->appendNode(c.new_config, nn->getId());
        new_node->setCostFromRoot(cost);
        new_node->setEdgeCostToParent(c.edge_cost);
        last_added_id_ = new_node->getId();
        processNode(new_node);
    }

    cycles_ += K;  // K parallel extend attempts count as K cycles
    col_checks_overall_ += (cspace_->getCollisionCheckCount() - col_checks_start);

    if ((cycles_ / K) % (50 / K + 1) == 0) {
        std::cout << "[GraspRrt] Cycle " << cycles_ << "/" << config_.max_cycles
                  << " (batch K=" << K << "), Nodes: " << tree_->getNrOfNodes()
                  << ", Grasps: " << grasps_.size() << "\n";
    }
    return true;
}

// ============================================================================
// VF-RRT core: compute the vector force field (repulsion + attraction)
// ============================================================================
Config GraspRrt::computeVectorField(const Config& q, CSpaceNodePtr nn,
                                     const Config& motion_dir) {
    Config vf = Eigen::VectorXf::Zero(dimension_);

    // 1. Semi-rigid foliage as a workspace force field (paper Sec. 3.3.1).
    //    For each robot collision sphere with penetration > 0 the field is
    //        F_i = +K_rad · pen · ∇φ_SDF                       (radial)
    //              − K_ax  · pen · (v_i · â_i) â_i             (axial drag,
    //                                                            non-conservative)
    //    mapped to joint space by J_i^T and summed over spheres. Setting
    //    K_ax = 0 (or v_joint = 0) yields the pure radial component.
    Config rep_vector = Eigen::VectorXf::Zero(dimension_);
    auto evaluator = cspace_->getCostEvaluator();
    if (evaluator) {
        auto tg0 = std::chrono::high_resolution_clock::now();
        auto sdf_eval = std::dynamic_pointer_cast<SdfCostEvaluator>(evaluator);
        if (sdf_eval) {
            rep_vector = sdf_eval->evaluateAnisotropicForce(
                q, motion_dir, config_.branch_anisotropy_K_ax);
        }
        auto tg1 = std::chrono::high_resolution_clock::now();
        static int vf_call_count = 0;
        static double vf_acc_us = 0;
        vf_acc_us += std::chrono::duration<double, std::micro>(tg1 - tg0).count();
        vf_call_count++;
        if (vf_call_count % 500 == 0) {
            std::cerr << "[VF-PROFILE @" << vf_call_count << "] "
                      << "field-force avg=" << (vf_acc_us / vf_call_count) << " us/call\n";
        }
    }
    if (rep_vector.norm() > 1e-6f) rep_vector.normalize();

    // 2. Rigid-obstacle repulsion + vortex (trunk + additional obstacles)
    // Note: under batch RRT, sharing cached_J_T_ across threads is a race; use a per-call local instead.
    auto panda_robot = dynamic_cast<PandaPinocchioRobot*>(robot_.get());
    if (panda_robot && !panda_robot->isJacobiansReady()) {
        robot_->prepareJacobians();
    } else if (!panda_robot) {
        robot_->prepareJacobians();
    }
    // Mapping (a): each workspace force is converted with the Jacobian evaluated at
    // ITS OWN application point. Reusing the end-effector Jacobian (the previous
    // behaviour) is only valid for forces acting at the end effector: for a force on
    // link3 it commands joints 4-7, which cannot move link3 at all, and in 3-8% of
    // configurations the resulting motion pushes the link further into the obstacle.
    // Frames repeat across the three loops below, so results are cached per call.
    struct JacEntry { int fid; Vector3f off; MatrixXf J; };
    std::vector<JacEntry> jac_cache;
    jac_cache.reserve(12);
    auto jacobianAt = [&](int fid, const Vector3f& off) -> const MatrixXf& {
        for (const auto& e : jac_cache) {
            if (e.fid == fid && (e.off - off).squaredNorm() < 1e-12f) return e.J;
        }
        MatrixXf J = panda_robot
            ? panda_robot->computeSphereJacobianDirect(fid, off)
            : robot_->computeSphereJacobianById(fid, off);
        jac_cache.push_back({fid, off, std::move(J)});
        return jac_cache.back().J;
    };

    Matrix4f tcp_pose = robot_->getGCPGlobalPose();
    Vector3f tcp_pos = tcp_pose.block<3,1>(0,3);

    Config hard_trunk_rep_vector = Eigen::VectorXf::Zero(dimension_);
    Config vortex_vector = Eigen::VectorXf::Zero(dimension_);
    Config soft_stiff_vector = Eigen::VectorXf::Zero(dimension_);  // soft-branch stiffness field
    float  bs_act = 0.0f;   // activation of the deepest contact, in [0,1] (used for axis dominance)

    // Per-link trunk repulsion: check every major link, not just TCP
    {
        static int fid_l2 = robot_->getFrameId("panda_link2");
        static int fid_l3 = robot_->getFrameId("panda_link3");
        static int fid_l4 = robot_->getFrameId("panda_link4");
        static int fid_l5 = robot_->getFrameId("panda_link5");
        static int fid_l6 = robot_->getFrameId("panda_link6");
        static int fid_l7 = robot_->getFrameId("panda_link7");
        static int fid_lh = robot_->getFrameId("panda_hand");

        // radius = the approximate body radius here (consistent with the SDF sphere model).
        // The stiffness field measures depth by SURFACE distance (dist - radius): with the
        // centroid 6 cm from the axis and a body radius of 6 cm the body is already crushing
        struct TrunkPoint { int fid; Vector3f offset; float radius; };
        TrunkPoint trunk_points[] = {
            {fid_l2, Vector3f(0,0,0),     0.08f},
            {fid_l3, Vector3f(0,0,0.15f), 0.07f},
            {fid_l4, Vector3f(0,0,0.05f), 0.07f},
            {fid_l5, Vector3f(0,0,0.15f), 0.06f},
            {fid_l6, Vector3f(0,0,0),     0.06f},
            {fid_l7, Vector3f(0,0,0.05f), 0.06f},
            {fid_lh, Vector3f(0,0,0.08f), 0.06f}
        };

        // ---- Generalization: iterate over all rigid cylinders, each with its true axis ----
        // The original implementation handled only the trunk with a hardcoded vertical axis
        // (0.7,0) — merely an approximation for tree6's tilted trunk, and no coverage at all
        // for the rigid (horizontal) apple branch: the attraction field dragged the search
        // straight into it with no tangential detour (one cause of start 3 getting stuck).
        // Now every rigid cylinder in the SDF is traversed (exclude_from_aniso=true:
        // trunk segments + tree6's rigid apple branch):
        //   repulsion = radial (toward the closest point on the segment), 1/d^2
        //   vortex    = axis_unit x radial (tangential around the true axis), weight 1.5x
        // Degenerate cylinders (p0==p1, e.g. the overhead sphere) are skipped — their repulsion
        // still comes from setHardObstacle. For a vertical trunk the new formula matches the old.
        constexpr float rigid_trigger = 0.20f;
        auto sdf_eval_vf = std::dynamic_pointer_cast<SdfCostEvaluator>(
            cspace_->getCostEvaluator());

        if (sdf_eval_vf) {
            const auto& cyls = sdf_eval_vf->getCylinders();
            for (const auto& pt : trunk_points) {
                Matrix4f pose = robot_->getFramePoseById(pt.fid);
                Vector3f p = pose.block<3,1>(0,3)
                           + pose.block<3,3>(0,0) * pt.offset;

                for (const auto& cyl : cyls) {
                    if (cyl.axis_unit.squaredNorm() < 0.25f) continue; // skip degenerate

                    // ---- Soft-branch stiffness field (branch stiffness, optional) ----
                    // force ~ stiffness(t)*depth: push hard at the root / deep inside, softly at the tip.
                    // The depth activation bs_act_ raises this term's weight in the VF blend
                    // until the axis dominates (> attraction); see the note at the blending site.
                    if (!cyl.exclude_from_aniso) {
                        if (!config_.branch_stiffness_enabled) continue;
                        Vector3f seg = cyl.p1 - cyl.p0;
                        float seg_sq = seg.squaredNorm();
                        float u = (seg_sq > 1e-9f)
                            ? std::max(0.0f, std::min(1.0f,
                                  (p - cyl.p0).dot(seg) / seg_sq))
                            : 0.0f;
                        Vector3f radial = p - (cyl.p0 + u * seg);
                        float dist = radial.norm();
                        // Volumetric: measured from the body surface (dist - body radius), not the centroid
                        float dist_eff = std::max(0.0f, dist - pt.radius);
                        float shell = cyl.radius + config_.bs_shell;
                        if (dist_eff >= shell || dist <= 1e-4f) continue;

                        float t_arc = cyl.t0 + u * (cyl.t1 - cyl.t0);
                        float stiff = SdfCostEvaluator::branchStiffness(
                            t_arc, config_.bs_k_root, config_.bs_k_tip);
                        // depth in (0,1]: 0 when the surface is at the shell's outer edge, 1 when it touches the axis
                        float depth = (shell - dist_eff) / shell;
                        float act = std::min(1.0f,
                            stiff * depth / std::max(1e-3f, config_.bs_k_root));
                        if (act > bs_act) bs_act = act;

                        radial /= dist;
                        Vector3f F_stiff = radial * (stiff * depth);
                        // Sum the real forces; the term is normalized once after the
                        // loop. Magnitudes therefore survive within the sum, so a deep
                        // contact near a stiff root outvotes a shallow one at the tip.
                        soft_stiff_vector += jacobianAt(pt.fid, pt.offset).transpose() * F_stiff;
                        continue;
                    }

                    // ---- Rigid cylinder: repulsion + vortex (true axis) ----
                    Vector3f seg = cyl.p1 - cyl.p0;
                    float seg_sq = seg.squaredNorm();
                    float t = (seg_sq > 1e-9f)
                        ? std::max(0.0f, std::min(1.0f,
                              (p - cyl.p0).dot(seg) / seg_sq))
                        : 0.0f;
                    Vector3f radial = p - (cyl.p0 + t * seg);
                    float dist = radial.norm();
                    if (dist >= rigid_trigger || dist <= 0.01f) continue;

                    radial /= dist;
                    float force_mag = 1.0f / (dist * dist);

                    const MatrixXf& J_pt = jacobianAt(pt.fid, pt.offset);
                    hard_trunk_rep_vector += J_pt.transpose() * (radial * force_mag);

                    Vector3f vortex_dir = cyl.axis_unit.cross(radial);
                    if (vortex_dir.norm() > 1e-6f) {
                        vortex_dir.normalize();
                        // The former x1.5 factor had no effect while each contribution
                        // was normalized individually; it is dropped rather than
                        // silently revived, keeping the previous vortex:repulsion 1:1.
                        vortex_vector += J_pt.transpose() * (vortex_dir * force_mag);
                    }
                }
            }
        }
        if (hard_trunk_rep_vector.norm() > 1e-6f) hard_trunk_rep_vector.normalize();
        if (vortex_vector.norm() > 1e-6f) vortex_vector.normalize();
        if (soft_stiff_vector.norm() > 1e-6f) soft_stiff_vector.normalize();
    }

    Config total_hard_obs_rep = Eigen::VectorXf::Zero(dimension_);
    if (hard_obstacle_enabled_) {
        static int fid_link4 = robot_->getFrameId("panda_link4");
        static int fid_link5 = robot_->getFrameId("panda_link5");
        static int fid_link6 = robot_->getFrameId("panda_link6");
        static int fid_link7 = robot_->getFrameId("panda_link7");
        static int fid_hand  = robot_->getFrameId("panda_hand");
        struct ProtectionPoint { int fid; Vector3f offset; };
        ProtectionPoint points[] = {
            {fid_link4, Vector3f(0,0,0.05f)},
            {fid_link5, Vector3f(0,0,0.15f)},
            {fid_link6, Vector3f(0,0,0)},
            {fid_link7, Vector3f(0,0,0.05f)},
            {fid_hand,  Vector3f(0,0,0.08f)}
        };
        for (const auto& pt : points) {
            Matrix4f pose = robot_->getFramePoseById(pt.fid);
            Vector3f p = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * pt.offset;
            Vector3f p_to_obs = p - hard_obstacle_pos_;
            float dist = p_to_obs.norm();
            if (dist < hard_obstacle_trigger_ && dist > 0.01f) {
                p_to_obs.normalize();
                float force_mag = 1.0f / (dist * dist);
                const MatrixXf& J_pt = jacobianAt(pt.fid, pt.offset);
                total_hard_obs_rep += J_pt.transpose() * (p_to_obs * force_mag);
                Vector3f v_dir = Vector3f(0,0,1).cross(p_to_obs);
                if (v_dir.norm() > 1e-6f) {
                    v_dir.normalize();
                    // vortex at half the repulsion, as before (the x1.5 was dead)
                    total_hard_obs_rep += J_pt.transpose() * (v_dir * force_mag * 0.5f);
                }
            }
        }
        // Normalize like every other component: without this the term's magnitude
        // grew with the number of protection points in range (up to ~7.5), silently
        // overriding the blend weights. Dormant in tree6 scenes (no overhead obstacle).
        if (total_hard_obs_rep.norm() > 1e-6f) total_hard_obs_rep.normalize();
    }

    // 3. Attraction field — applied to wrist region (link6, link7, hand) + TCP
    //
    // Two modes (selected by config_.use_shell_attractor):
    //   A. Random sampling on a spherical cap (use_shell_attractor=true):
    //      shell radius = target_radius + offset, cap height = cap_height
    //      resampled uniformly on the cap every shell_attractor_resample_period cycles
    //   B. Fixed single point (use_shell_attractor=false):
    //      the attractor sits 8 cm below the apple
    Vector3f virtual_target;
    if (config_.use_shell_attractor) {
        // Note: the shell resampling logic moved out. At each cycle/batch entry the main
        // thread decides via maybeResampleShellAttractor() whether to update cached_shell_target_.
        // Here it is only read (parallel-safe).
        if (cached_shell_target_.norm() < 1e-9f) {
            // Fallback: first call and main has not sampled yet — use the fixed point
            virtual_target = target_object_position_;
            virtual_target.z() -= 0.08f;
        } else {
            virtual_target = cached_shell_target_;
        }
    } else {
        // Fixed single point (original design): 8 cm below the apple
        virtual_target = target_object_position_;
        virtual_target.z() -= 0.08f;
    }

    Config att_vector = Eigen::VectorXf::Zero(dimension_);
    {
        static int fid_link6_a = robot_->getFrameId("panda_link6");
        static int fid_link7_a = robot_->getFrameId("panda_link7");
        static int fid_hand_a  = robot_->getFrameId("panda_hand");
        struct AttractPoint { int fid; Vector3f offset; float weight; };
        AttractPoint apts[] = {
            {fid_link6_a, Vector3f(0,0,0),     0.4f},
            {fid_link7_a, Vector3f(0,0,0.05f), 0.6f},
            {fid_hand_a,  Vector3f(0,0,0.08f), 1.0f},  // ~TCP
        };
        // Nav-field blended attraction (see the note in types.h): the far field follows the
        // flood downhill, the near field falls back to a straight line toward a random cap point;
        auto sdf_nav = config_.nav_field_enabled
            ? std::dynamic_pointer_cast<SdfCostEvaluator>(
                  cspace_->getCostEvaluator())
            : nullptr;
        const float NAV_FAR = std::numeric_limits<float>::max() / 4.0f;
        for (const auto& ap : apts) {
            Matrix4f pose = robot_->getFramePoseById(ap.fid);
            Vector3f link_pos = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * ap.offset;
            Vector3f dir = virtual_target - link_pos;
            if (dir.norm() > 1e-6f) dir.normalize();
            if (sdf_nav && sdf_nav->isNavBuilt()) {
                float dnav = sdf_nav->queryNavDistance(link_pos);
                if (dnav < NAV_FAR) {
                    Vector3f ndir = sdf_nav->queryNavDirection(link_pos);
                    if (ndir.norm() > 1e-6f) {
                        float s = (dnav - config_.nav_switch_dist)
                                / std::max(1e-3f, config_.nav_blend_band);
                        s = std::max(0.0f, std::min(1.0f, s));
                        float w = s * s * (3.0f - 2.0f * s);   // 1 = far-field flood
                        Vector3f blend = w * ndir + (1.0f - w) * dir;
                        if (blend.norm() > 1e-6f) dir = blend.normalized();
                    }
                }
                // dnav = infinity (outside the field / unreachable) -> keep the straight-line fallback
            }
            // ap.weight was previously applied twice (once in the force, once after
            // normalizing), making the effective point weights 0.16/0.36/1.0. It is
            // now applied once, so they are the stated 0.4/0.6/1.0.
            att_vector += jacobianAt(ap.fid, ap.offset).transpose() * (dir * ap.weight);
        }
        if (att_vector.norm() > 1e-6f) att_vector.normalize();
    }

    // 4. Weighted blend
    // Weight of the soft-branch stiffness term: rises with the deepest contact's activation
    // from 0 to bs_weight_max (default 6.0, above the attraction weight 4.0). At shallow
    // contact it merely supplements the repulsion; near the branch axis the net field
    float w_stiff = bs_act * config_.bs_weight_max;
    vf = config_.vf_w_rep     * rep_vector +
         w_stiff              * soft_stiff_vector +
         config_.vf_w_rigid   * hard_trunk_rep_vector +
         config_.vf_w_vortex  * vortex_vector +
         config_.vf_w_hardobs * total_hard_obs_rep +
         config_.vf_w_att     * att_vector;
    if (vf.norm() > 1e-6f) vf.normalize();
    return vf;
}

// ============================================================================
// computeAdaptiveStep — step size adapted to the TCP distance to obstacles
//   far -> long steps (fast coverage), near -> short steps (avoid overshooting)
// ============================================================================
static float computeAdaptiveStep(float dist_to_obstacle, float base_step) {
    // Negative inside the SDF -> already inside an obstacle (abnormal); use the smallest step
    if (dist_to_obstacle <= 0.0f) return base_step * 0.4f;

    // Piecewise map: dist (m) -> step ratio
    // > 0.20 m : 2.4x (far, move boldly)
    // 0.10-0.20: 1.6x
    // 0.05-0.10: 1.0x (default)
    // 0.02-0.05: 0.6x
    // < 0.02 m : 0.4x (very close, be careful)
    float ratio;
    if      (dist_to_obstacle > 0.20f) ratio = 2.4f;
    else if (dist_to_obstacle > 0.10f) ratio = 1.6f;
    else if (dist_to_obstacle > 0.05f) ratio = 1.0f;
    else if (dist_to_obstacle > 0.02f) ratio = 0.6f;
    else                               ratio = 0.4f;
    return base_step * ratio;
}


// ============================================================================
// extendVFRRT — VF-RRT extension primitive (eager: every edge is validated on the spot)
// ============================================================================
// Pure-compute part of extendVFRRT.
// Reads tree_/cspace_/robot_ only; never writes tree_/performance_/last_added_id_.
// Safe to call from multiple threads (precondition: phases 1+2 are ready).
GraspRrt::ExtendCandidate GraspRrt::extendVFRRTCompute(const Config& target) {
    ExtendCandidate cand;

    auto t_start_nn = std::chrono::high_resolution_clock::now();
    auto nn = tree_->getNearestNeighbor(target);
    auto t_end_nn = std::chrono::high_resolution_clock::now();
    cand.nn_time_ms = std::chrono::duration<float, std::milli>(t_end_nn - t_start_nn).count();

    if (!nn) return cand;  // valid stays false

    Config q_near = nn->getConfiguration();
    cand.nn_id = nn->getId();

    Config q_rand_dir = target - q_near;
    if (q_rand_dir.norm() > 1e-6f) q_rand_dir.normalize();

    auto t_start_vf = std::chrono::high_resolution_clock::now();
    // q_rand_dir is the already-normalized extension direction, passed to the VF as motion_dir
    // (Option A's axial drag needs it; the parameter is ignored when K_ax=0).
    Config vf_dir = computeVectorField(q_near, nn, q_rand_dir);
    auto t_end_vf = std::chrono::high_resolution_clock::now();
    cand.vf_time_ms = std::chrono::duration<float, std::milli>(t_end_vf - t_start_vf).count();

    // Extension primitive: deterministic blend (default) or coin-flip ablation.
    Config q_step_dir;
    if (config_.extension_coin_flip) {
        // 50% pure q_rand_dir, 50% pure vf_dir — discrete switch
        q_step_dir = (randFloat01() < 0.5f) ? q_rand_dir : vf_dir;
    } else {
        // Standard VF-RRT: 50/50 deterministic weighted blend
        q_step_dir = 0.50f * q_rand_dir + 0.50f * vf_dir;
    }
    if (q_step_dir.norm() > 1e-6f) q_step_dir.normalize();

    // Note: the sample steering that used to sit here (angle-cone cap) has been removed
    // entirely, implementation included. Its role is fully covered by the unified steering
    // below (proposal-layer projection driven by the damage budget rather than by angle):
    // 9 whole-arm probes x all soft branches, pairwise max, no argmax and no K-nearest,


    float step_size = config_.extend_step_size;
    auto evaluator = cspace_->getCostEvaluator();
    if (evaluator) {
        Vector3f tcp_q_near = robot_->getGCPGlobalPose().block<3,1>(0,3);
        float dist = evaluator->queryDistance(tcp_q_near);
        if (dist < std::numeric_limits<float>::max()) {
            step_size = computeAdaptiveStep(dist, config_.extend_step_size);
        }
    }

    // ---- Depth steering / Unified steering (proposal layer, see the note in types.h) ----
    // Project the proposed direction once the real step size is known and before the edge is
    // formed: unified handles inward pressure and axial slip together (leaving ~ hoop motion);
    if ((config_.uni_refuse_enabled || config_.pen_steer_enabled) &&
        q_step_dir.norm() > 1e-6f) {
        auto sdf_eval_ds = std::dynamic_pointer_cast<SdfCostEvaluator>(
            cspace_->getCostEvaluator());
        if (sdf_eval_ds) {
            bool bent = config_.uni_refuse_enabled
                ? sdf_eval_ds->applyUnifiedSteering(q_near, q_step_dir, step_size)
                : sdf_eval_ds->applyDepthSteering(q_near, q_step_dir, step_size,
                                                  config_.pen_steer_safety,
                                                  config_.pen_steer_margin);
            if (bent && q_step_dir.norm() > 1e-6f) q_step_dir.normalize();
        }
    }

    Config new_config = q_near + q_step_dir * step_size;


    int col_before = cspace_->getCollisionCheckCount();
    auto t_start_col = std::chrono::high_resolution_clock::now();
    bool is_valid = cspace_->isPathValid(q_near, new_config);
    auto t_end_col = std::chrono::high_resolution_clock::now();
    cand.collision_time_ms = std::chrono::duration<float, std::milli>(t_end_col - t_start_col).count();
    cand.col_check_count_delta = cspace_->getCollisionCheckCount() - col_before;

    if (!is_valid) return cand;

    cand.new_config = new_config;
    cand.edge_cost  = cspace_->calcDist(q_near, new_config);
    cand.valid      = true;
    return cand;
}

// Serial entry point — same semantics as before.
ExtensionResult GraspRrt::extendVFRRT(const Config& target) {
    ExtendCandidate c = extendVFRRTCompute(target);

    performance_.nearest_neighbor_time_ms += c.nn_time_ms;
    performance_.vf_gradient_time_ms      += c.vf_time_ms;
    performance_.collision_check_time_ms  += c.collision_time_ms;
    performance_.collision_in_extend_ms   += c.collision_time_ms;

    if (c.nn_id < 0) return ExtensionResult::ERROR;
    if (!c.valid)   return ExtensionResult::FAILED;

    auto nn = tree_->getNode(c.nn_id);
    if (!nn) return ExtensionResult::ERROR;

    float cost = nn->getCostFromRoot() + c.edge_cost;
    auto new_node = tree_->appendNode(c.new_config, nn->getId());
    new_node->setCostFromRoot(cost);
    new_node->setEdgeCostToParent(c.edge_cost);
    last_added_id_ = new_node->getId();

    processNode(new_node);
    return ExtensionResult::SUCCESS;
}

// ============================================================================
// shortcutSolutionPath — Method 1: score-aware shortcutting (see grasp_rrt.h)
// ============================================================================

bool GraspRrt::segmentValidPiecewise(const Config& a, const Config& b) {
    float dist = (b - a).norm();
    int n_seg = std::max(1, static_cast<int>(
        std::ceil(dist / std::max(1e-4f, config_.extend_step_size))));
    Config prev = a;
    for (int s = 1; s <= n_seg; ++s) {
        Config next = a + (b - a) * (static_cast<float>(s) / n_seg);
        if (!cspace_->isPathValid(prev, next)) return false;
        prev = next;
    }
    return true;
}

std::vector<Config> GraspRrt::shortcutSolutionPath(
    const std::vector<Config>& in, int max_rounds) {
    std::vector<Config> path = in;
    if (path.size() < 3) return path;

    // Greedy "farthest direct connection" scan: for each i, search backward from the end of
    // the path for the farthest j whose direct (i,j) connection is valid; on success drop the
    //
    // Parallelization: for a fixed i, the per-segment checks of the candidate j values are
    // independent. Starting from the end, evaluate in chunks (chunk = 2 x thread count) and
    // take the largest valid j in the chunk — identical to the serial "first valid j scanning
    // backward from the end", while the failing prefix is spread across idle cores.
    const int nt = shortcut_threads_;
    bool improved = true;
    int round = 0;
    while (improved && round++ < max_rounds) {
        improved = false;
        size_t i = 0;
        while (i + 2 < path.size()) {
            size_t found = i;
            const size_t j_lo = i + 2;            // candidates j > i+1
            const size_t j_hi = path.size() - 1;
            if (nt <= 1) {
                for (size_t j = j_hi; j > i + 1; --j) {
                    if (segmentValidPiecewise(path[i], path[j])) {
                        found = j;
                        break;
                    }
                }
            } else {
                const size_t chunk = static_cast<size_t>(2 * nt);
                size_t hi = j_hi;
                while (found == i) {
                    const size_t lo = (hi >= j_lo + chunk - 1) ? hi - chunk + 1
                                                               : j_lo;
                    const int n = static_cast<int>(hi - lo + 1);
                    std::vector<char> ok(n, 0);
#ifdef _OPENMP
                    #pragma omp parallel for num_threads(nt) schedule(dynamic)
#endif
                    for (int k = 0; k < n; ++k) {
                        ok[k] = segmentValidPiecewise(path[i], path[lo + k])
                                    ? 1 : 0;
                    }
                    for (int k = n - 1; k >= 0; --k) {
                        if (ok[k]) { found = lo + k; break; }
                    }
                    if (lo == j_lo) break;
                    hi = lo - 1;
                }
            }
            if (found > i + 1) {
                path.erase(path.begin() + i + 1, path.begin() + found);
                improved = true;
            }
            ++i;
        }
    }

    // Re-densify at planning granularity: after pruning only a few waypoints may remain,
    // while the anisotropy metrics sample penetration depth per waypoint — too sparse and
    // contacts are missed, understating the metric. Densifying does not change the geometry
    std::vector<Config> dense;
    dense.push_back(path.front());
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        float dist = (path[i + 1] - path[i]).norm();
        int n_seg = std::max(1, static_cast<int>(
            std::ceil(dist / std::max(1e-4f, config_.extend_step_size))));
        for (int s = 1; s <= n_seg; ++s)
            dense.push_back(path[i] +
                (path[i + 1] - path[i]) * (static_cast<float>(s) / n_seg));
    }
    return dense;
}

// ============================================================================
// validateSolutionPath — lazy collision validation: check the full path only after a grasp is found
// ============================================================================
bool GraspRrt::validateSolutionPath(CSpaceNodePtr goal_node) {
    std::vector<Config> path;
    auto current = goal_node;
    while (current) {
        path.push_back(current->getConfiguration());
        if (current->getId() == start_node_->getId()) break;
        if (current->getParentId() < 0) return false;
        current = tree_->getNode(current->getParentId());
    }
    std::reverse(path.begin(), path.end());

    for (size_t i = 0; i + 1 < path.size(); ++i) {
        auto t_start_col = std::chrono::high_resolution_clock::now();
        bool valid = cspace_->isPathValid(path[i], path[i + 1]);
        auto t_end_col = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float, std::milli>(t_end_col - t_start_col).count();
        performance_.collision_check_time_ms   += dt;
        performance_.collision_in_validate_ms  += dt;
        if (!valid) return false;
    }
    return true;
}

// ============================================================================
// connectRandomGraspPositionJacobian
// Counterpart of the Python GraspRrt.connectRandomGraspPositionJacobian
// ============================================================================

MoveArmResult GraspRrt::connectRandomGraspPositionJacobian() {
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Get a node with uniformly sampled direction from the discretized sphere
    CSpaceNodePtr nn_node = pose_sphere_->getGoodRatedNode();
    if (!nn_node) {
        if (verbose_) std::cout << "[GraspRrt] No good rated node in pose sphere\n";
        // No node to pick: TRAPPED immediately, not counted in the face stats (face_id == -1)
        n_grasp_trapped_++;
        return MoveArmResult::TRAPPED;
    }
    int face_id = pose_sphere_->getLastSelectedFace();

    // 2. Compute the grasp target pose on the object surface
    auto t_pose_start = std::chrono::high_resolution_clock::now();
    bool pose_ok = false;
    Matrix4f goal_pose = calculateGlobalGraspPose(nn_node->getConfiguration(), pose_ok);
    auto t_pose_end = std::chrono::high_resolution_clock::now();
    performance_.pose_calc_ms += std::chrono::duration<float, std::milli>(t_pose_end - t_pose_start).count();
    if (!pose_ok) {
        // A failed pose computation also counts as a failure for that face
        pose_sphere_->recordGraspAttempt(face_id, /*success=*/false);
        tryDisableFace(face_id);
        n_grasp_error_++;
        return MoveArmResult::ERROR;
    }

    // 3. Step from that node toward the grasp pose
    MoveArmResult res = moveTowardsGoal(nn_node, goal_pose);

    // 4. Accumulate result counters (Group B metrics)
    switch (res) {
        case MoveArmResult::GOAL_REACHED:           n_grasp_goal_reached_++; break;
        case MoveArmResult::GRASPABLE_POSE_REACHED: n_grasp_pose_reached_++; break;
        case MoveArmResult::TRAPPED:                n_grasp_trapped_++;      break;
        case MoveArmResult::ERROR:                  n_grasp_error_++;        break;
        default: break;  // FATAL_ERROR is not counted
    }

    // 5. Update face statistics and trigger disabling
    bool attempt_success = (res == MoveArmResult::GOAL_REACHED) ||
                           (res == MoveArmResult::GRASPABLE_POSE_REACHED);
    pose_sphere_->recordGraspAttempt(face_id, attempt_success);
    if (!attempt_success && res != MoveArmResult::FATAL_ERROR) {
        tryDisableFace(face_id);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    float move_time_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    performance_.move_toward_grasp_poses_time_ms += move_time_ms;
    performance_.number_of_movements_toward_grasp_pose++;

    if (verbose_) {
        std::cout << "[GraspRrt] Connect to grasp: " << move_time_ms << " ms\n";
    }
    return res;
}

void GraspRrt::dumpTree(const std::string& filepath) {
    if (!tree_) return;

    // Collect the node ids on the solution path (walk parents up to the root)
    std::unordered_set<int> path_ids;
    if (goal_node_) {
        auto cur = goal_node_;
        while (cur) {
            path_ids.insert(cur->getId());
            int pid = cur->getParentId();
            if (pid < 0) break;
            cur = tree_->getNode(pid);
        }
    }

    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "[GraspRrt] WARNING: cannot open tree dump file: " << filepath << "\n";
        return;
    }

    f << "node_id,parent_id,j0,j1,j2,j3,j4,j5,j6,tcp_x,tcp_y,tcp_z,in_pose_sphere,on_solution_path\n";

    const auto& all_nodes = tree_->getNodes();
    for (const auto& node : all_nodes) {
        const Config& cfg = node->getConfiguration();
        robot_->setJointValues(cfg);
        Matrix4f tcp_pose = robot_->getGCPGlobalPose();
        Vector3f tcp = tcp_pose.block<3,1>(0, 3);

        Vector3f diff = tcp - target_object_position_;
        bool in_ps   = diff.squaredNorm() < config_.try_grasps_distance2;
        bool on_path = path_ids.count(node->getId()) > 0;

        f << node->getId() << "," << node->getParentId();
        for (int d = 0; d < 7; ++d) f << "," << cfg[d];
        f << "," << tcp.x() << "," << tcp.y() << "," << tcp.z();
        f << "," << (in_ps ? 1 : 0)
          << "," << (on_path ? 1 : 0) << "\n";
    }
}

void GraspRrt::dumpFaceStats(const std::string& filepath) {
    if (!pose_sphere_) return;

    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "[GraspRrt] WARNING: cannot open face stats file: " << filepath << "\n";
        return;
    }

    f << "face_id,attempts,failures,disabled,node_count,dir_x,dir_y,dir_z\n";
    int n = pose_sphere_->getTotalFaceCount();
    for (int fid = 0; fid < n; ++fid) {
        int attempts  = pose_sphere_->getAttemptCount(fid);
        int failures  = pose_sphere_->getFailureCount(fid);
        bool disabled = pose_sphere_->isFaceDisabled(fid);
        int n_nodes   = pose_sphere_->getNodeCount(fid);
        Vector3f dir  = pose_sphere_->getFaceDirection(fid);

        f << fid << "," << attempts << "," << failures << ","
          << (disabled ? 1 : 0) << "," << n_nodes << ","
          << dir.x() << "," << dir.y() << "," << dir.z() << "\n";
    }
}

void GraspRrt::tryDisableFace(int face_id) {
    if (!config_.face_disable_enabled || face_id < 0) return;
    if (pose_sphere_->isFaceDisabled(face_id)) return;

    int failures = pose_sphere_->getFailureCount(face_id);
    int tree_size = tree_ ? tree_->getNrOfNodes() : 0;
    int active_count = pose_sphere_->getActiveFaceCount();

    if (failures >= config_.face_disable_failure_threshold &&
        tree_size >= config_.face_disable_min_tree_size &&
        active_count > config_.face_disable_min_active) {
        pose_sphere_->disableFace(face_id);
        std::cout << "[GraspRrt] Disabled face " << face_id
                  << " (failures=" << failures
                  << ", tree=" << tree_size
                  << ", remaining active=" << (active_count - 1) << ")\n";
    }
}

// ============================================================================
// calculateGlobalGraspPose — counterpart of the Python calculateGlobalGraspPose
// ============================================================================

Matrix4f GraspRrt::calculateGlobalGraspPose(const Config& config, bool& success) {
    success = false;

    robot_->setJointValues(config);
    Matrix4f gcp_pose = robot_->getGCPGlobalPose();

    // Get the closest point on the object surface
    Vector3f gcp_pos = gcp_pose.block<3,1>(0, 3);
    Vector3f P1 = target_object_->getClosestPoint(gcp_pos);

    // ======== MODIFICATION: Bottom Hemisphere Check ========
    Matrix4f obj_pose = target_object_->getGlobalPose();
    Eigen::Vector4f p1_h;
    p1_h << P1, 1.0f;
    Eigen::Vector4f p1_local = obj_pose.inverse() * p1_h;
    if (p1_local.z() > 0.0f) {
        return Matrix4f::Identity(); // success remains false
    }

    // Compute the direction of P1 in the GCP local frame
    Matrix4f gcp_inv = gcp_pose.inverse();
    Eigen::Vector4f P1_h;
    P1_h << P1, 1.0f;
    Eigen::Vector4f P1_local_h = gcp_inv * P1_h;
    Vector3f vec_target_local = P1_local_h.head<3>();
    float length = vec_target_local.norm();

    if (length < 1e-8f) {
        std::cerr << "[GraspRrt] WARNING: distance to target too small\n";
        return Matrix4f::Identity();
    }
    vec_target_local /= length;

    // Z axis = the GCP's default approach direction
    Vector3f vec_z(0.0f, 0.0f, 1.0f);

    // Find the rotation that turns the Z axis toward the target direction
    float dot = std::max(-1.0f, std::min(1.0f, vec_z.dot(vec_target_local)));
    float angle = std::acos(dot);
    Vector3f rot_axis = vec_z.cross(vec_target_local);
    float axis_norm = rot_axis.norm();

    Matrix4f rot_matrix = Matrix4f::Identity();
    if (axis_norm > 1e-10f) {
        rot_axis /= axis_norm;
        rot_matrix = axisAngleToMatrix4(rot_axis, angle);
    }

    // Convert to global coordinates
    Matrix4f goal_pose = gcp_pose * rot_matrix;
    goal_pose.block<3,1>(0, 3) = P1; // target position set to the point on the object surface

    success = true;
    return goal_pose;
}

// ============================================================================
// moveTowardsGoal — counterpart of the Python moveTowardsGoal
// ============================================================================

MoveArmResult GraspRrt::moveTowardsGoal(CSpaceNodePtr start_node,
                                         const Matrix4f& target_pose) {
    if (!start_node) return MoveArmResult::ERROR;

    auto last_extend_node = start_node;

    // Current Cartesian distance from the TCP to the goal
    Matrix4f current_tcp = Matrix4f::Identity();
    auto it = map_config_tcp_.find(last_extend_node->getId());
    if (it != map_config_tcp_.end()) current_tcp = it->second;

    float dist = cartesianPoseDiff(current_tcp, target_pose);
    float last_dist = 0.0f;

    for (int loop = 0; loop < config_.move_towards_goal_max_loops; ++loop) {
        // Compute the workspace sampling step
        auto [res, tmp_config] = createWorkspaceSamplingStepFromNode(
            target_pose, last_extend_node);

        // ---- Handle the step result ----

        if (res == MoveArmResult::ERROR || res == MoveArmResult::FATAL_ERROR) {
            return res;
        }

        if (res == MoveArmResult::COLLISION_ENVIRONMENT) {
            // Collision -> try to grasp at the last valid configuration
            float score = calculateGraspScore(
                last_extend_node->getConfiguration(),
                last_extend_node->getId(), true);
            if (score >= config_.grasp_quality_min_score) {
                std::cout << "[GraspRrt] Valid grasp (collision, score=" << score << ")\n";
                return MoveArmResult::GOAL_REACHED;
            } else if (score > 0.0f) {
                return MoveArmResult::GRASPABLE_POSE_REACHED;
            } else {
                if (verbose_) std::cout << "[GraspRrt] Collision, no graspable config\n";
                return MoveArmResult::COLLISION_ENVIRONMENT;
            }
        }

        if (res == MoveArmResult::MOVED_ONE_STEP) {
            // Check path validity
            auto t_start_col = std::chrono::high_resolution_clock::now();
            bool is_valid = cspace_->isPathValid(last_extend_node->getConfiguration(), tmp_config);
            auto t_end_col = std::chrono::high_resolution_clock::now();
            float dt_mv = std::chrono::duration<float, std::milli>(t_end_col - t_start_col).count();
            performance_.collision_check_time_ms += dt_mv;
            performance_.collision_in_move_ms    += dt_mv;
            
            if (!is_valid) {
                if (verbose_) std::cout << "[GraspRrt] Path check failed after IK step\n";
                return MoveArmResult::COLLISION_ENVIRONMENT;
            }

            // Add to the RRT tree
            auto [success, new_id] = tree_->appendPath(last_extend_node, tmp_config);
            if (!success) return MoveArmResult::ERROR;
            last_added_id_ = new_id;

            processNodes(last_extend_node->getId(), last_added_id_);

            last_extend_node = tree_->getNode(last_added_id_);
            last_extend_node->setStatus(2); // mark as created by moveArm

            last_dist = dist;
            auto tcp_it = map_config_tcp_.find(last_extend_node->getId());
            Matrix4f tcp_pose = (tcp_it != map_config_tcp_.end())
                                ? tcp_it->second : Matrix4f::Identity();
            dist = cartesianPoseDiff(tcp_pose, target_pose);

            // Check whether we are trapped
            if ((dist - last_dist) > 2.0f * config_.cart_sampling_pos_step_size) {
                if (verbose_) {
                    std::cout << "[GraspRrt] Trapped: last=" << last_dist
                              << ", now=" << dist << "\n";
                }
                return MoveArmResult::TRAPPED;
            }
        }

        if (res == MoveArmResult::JOINT_BOUNDARY_VIOLATION) {
            if (verbose_) std::cout << "[GraspRrt] Joint limits reached\n";
            return MoveArmResult::TRAPPED;
        }

        // Check whether the goal has been reached
        if (dist <= config_.max_dist_goal_reached) {
            if (verbose_) {
                std::cout << "[GraspRrt] Grasp goal reached, dist=" << dist << "\n";
            }
            float score = calculateGraspScore(
                last_extend_node->getConfiguration(),
                last_extend_node->getId(), true);
            if (score >= config_.grasp_quality_min_score) {
                std::cout << "[GraspRrt] Valid grasp (score=" << score << ")\n";
                return MoveArmResult::GOAL_REACHED;
            } else if (score > 0.0f) {
                return MoveArmResult::GRASPABLE_POSE_REACHED;
            } else {
                if (verbose_) std::cout << "[GraspRrt] Goal reached but zero grasp score\n";
                return MoveArmResult::TRAPPED;
            }
        }
    }

    if (verbose_) std::cout << "[GraspRrt] Max loops reached, dist=" << dist << "\n";
    return MoveArmResult::TRAPPED;
}

// ============================================================================
// createWorkspaceSamplingStep — counterpart of the Python createWorkSpaceSamplingStep
// ============================================================================

std::pair<MoveArmResult, Config> GraspRrt::createWorkspaceSamplingStep(
    const Matrix4f& current_pose, const Matrix4f& goal_pose) {
    // Compute the delta
    Matrix4f delta_pose = goal_pose * current_pose.inverse();
    delta_pose.block<3,1>(0, 3) = goal_pose.block<3,1>(0, 3) - current_pose.block<3,1>(0, 3);

    // ---- Finger-first steering (pure guidance, see the note in types.h) ----
    // A sideways first touch sweeps a free apple away; at every descent step this:
    //   (a) turns the gripper axis toward the apple (aligning early: rotate first, then arrive);
    //   (b) softly slows the component advancing toward the apple while misaligned (giving (a) time to turn).
    // Smooth weights throughout and no rejection — the worst case is a few extra steps, never a failure.
    if (config_.finger_first_steering) {
        Vector3f p_palm = current_pose.block<3,1>(0, 3);
        Vector3f z_g    = current_pose.block<3,1>(0, 2);   // gripper approach axis
        Vector3f to_apple = target_object_position_ - p_palm;
        float d = to_apple.norm();
        if (d > 1e-6f && d < config_.ff_far && z_g.norm() > 1e-6f) {
            // Smooth activation weight w(d): rises from 0 to 1 between ff_far and ff_near
            float s = (config_.ff_far - d)
                    / std::max(1e-6f, config_.ff_far - config_.ff_near);
            s = std::max(0.0f, std::min(1.0f, s));
            float w = s * s * (3.0f - 2.0f * s);

            Vector3f u = to_apple / d;
            float cos_phi = std::max(-1.0f, std::min(1.0f, z_g.dot(u)));
            float phi = std::acos(cos_phi);

            // (a) rotation bias: d_omega = w*gain*phi, about (z_g x u)
            Vector3f rot_axis = z_g.cross(u);
            float axis_norm = rot_axis.norm();
            if (axis_norm > 1e-6f && phi > 1e-3f) {
                float ang = w * config_.ff_align_gain * phi;
                Eigen::AngleAxisf bias(ang, rot_axis / axis_norm);
                delta_pose.block<3,3>(0, 0) =
                    bias.toRotationMatrix() * delta_pose.block<3,3>(0, 0);
            }

            // (b) advance soft-brake: scale the component of the position increment toward the apple (only when close)
            Vector3f dp = delta_pose.block<3,1>(0, 3);
            float adv = dp.dot(u);
            if (adv > 0.0f) {
                float scale = config_.ff_min_advance
                            + (1.0f - config_.ff_min_advance)
                              * std::max(0.0f, cos_phi);
                float scale_eff = 1.0f - w * (1.0f - scale);
                dp += (scale_eff - 1.0f) * adv * u;
                delta_pose.block<3,1>(0, 3) = dp;
            }
        }
    }

    // ---- Branch-stiffness descent bias (same stiffness model as the VF) ----
    // The descent does not go through the VF, so the bias is added separately: while the hand
    // point is inside a soft-branch shell, bend the position increment toward the
    // "stiffness-weighted radial" direction — push hard at the root / deep, softly at the tip.
    // Faded by distance to the apple (bs_descent_fade_r). Pure guidance, no rejection.
    if (config_.branch_stiffness_enabled) {
        auto sdf_eval_bs = std::dynamic_pointer_cast<SdfCostEvaluator>(
            cspace_->getCostEvaluator());
        if (sdf_eval_bs) {
            Vector3f p_ee = current_pose.block<3,1>(0, 3);
            float d_apple = (target_object_position_ - p_ee).norm();
            float fade = std::max(0.0f, std::min(1.0f,
                d_apple / std::max(1e-3f, config_.bs_descent_fade_r)));
            if (fade > 0.01f) {
                Vector3f push = Vector3f::Zero();
                float act = 0.0f;
                for (const auto& cyl : sdf_eval_bs->getCylinders()) {
                    if (cyl.exclude_from_aniso) continue;               // soft branches only
                    if (cyl.axis_unit.squaredNorm() < 0.25f) continue;
                    Vector3f seg = cyl.p1 - cyl.p0;
                    float seg_sq = seg.squaredNorm();
                    float u = (seg_sq > 1e-9f)
                        ? std::max(0.0f, std::min(1.0f,
                              (p_ee - cyl.p0).dot(seg) / seg_sq))
                        : 0.0f;
                    Vector3f radial = p_ee - (cyl.p0 + u * seg);
                    float dist = radial.norm();
                    // Volumetric: hand body radius 0.06, depth measured from the body surface
                    float dist_eff = std::max(0.0f, dist - 0.06f);
                    float shell = cyl.radius + config_.bs_shell;
                    if (dist_eff >= shell || dist <= 1e-4f) continue;
                    float t_arc = cyl.t0 + u * (cyl.t1 - cyl.t0);
                    float stiff = SdfCostEvaluator::branchStiffness(
                        t_arc, config_.bs_k_root, config_.bs_k_tip);
                    float depth = (shell - dist_eff) / shell;
                    push += (radial / dist) * (stiff * depth);
                    act = std::max(act, std::min(1.0f,
                        stiff * depth / std::max(1e-3f, config_.bs_k_root)));
                }
                float pn = push.norm();
                Vector3f dp = delta_pose.block<3,1>(0, 3);
                float dpn = dp.norm();
                if (pn > 1e-6f && dpn > 1e-9f && act > 0.0f) {
                    // Bend the direction, keep the magnitude: the worst case is a detour, never a stall
                    Vector3f bent = dp + fade * config_.bs_descent_gain
                                       * act * dpn * (push / pn);
                    float bn = bent.norm();
                    if (bn > 1e-9f)
                        delta_pose.block<3,1>(0, 3) = bent / bn * dpn;
                }
            }
        }
    }

    // ---- Depth / Unified steering, descent variant (proposal layer) ----
    // Budget clamping on the hand probes, replacing blind pressing down; the arm probes are
    // still backed by the gate inside isPathValid. Contacts inside the apple exemption sphere are ignored.
    if (config_.uni_refuse_enabled || config_.pen_steer_enabled) {
        auto sdf_eval_ps = std::dynamic_pointer_cast<SdfCostEvaluator>(
            cspace_->getCostEvaluator());
        if (sdf_eval_ps) {
            Vector3f dp = delta_pose.block<3,1>(0, 3);
            bool bent = config_.uni_refuse_enabled
                ? sdf_eval_ps->applyUnifiedSteeringDescent(dp)
                : sdf_eval_ps->applyDepthSteeringDescent(
                      dp, config_.pen_steer_safety, config_.pen_steer_margin);
            if (bent) delta_pose.block<3,1>(0, 3) = dp;
        }
    }

    // Clamp the step size
    delta_pose = limitWorkspaceStep(delta_pose);

    return moveArmDiffKin(delta_pose);
}

std::pair<MoveArmResult, Config> GraspRrt::createWorkspaceSamplingStepFromNode(
    const Matrix4f& goal_pose, CSpaceNodePtr extend_node) {
    robot_->setJointValues(extend_node->getConfiguration());

    auto it = map_config_tcp_.find(extend_node->getId());
    Matrix4f current_pose = (it != map_config_tcp_.end()) ? it->second : Matrix4f::Identity();

    return createWorkspaceSamplingStep(current_pose, goal_pose);
}

// ============================================================================
// moveArmDiffKin — counterpart of the Python moveArmDiffKin
// One differential IK step: J+ * e -> dtheta
// ============================================================================

std::pair<MoveArmResult, Config> GraspRrt::moveArmDiffKin(const Matrix4f& delta_pose) {
    // Get the current joint values
    Config start_config = robot_->getJointValues();

    // Build the 6D error vector: [translation(3) | rotation(3)]
    Eigen::VectorXf e(6);
    e.head<3>() = delta_pose.block<3,1>(0, 3);

    // Extract axis-angle from the rotation matrix as the rotation error
    Vector3f axis;
    float angle;
    rotationToAxisAngle(delta_pose.block<3,3>(0, 0), axis, angle);
    e.tail<3>() = axis * angle;

    // Jacobian pseudoinverse
    MatrixXf J_pinv = computePseudoInverseJacobian();
    Eigen::VectorXf d_theta = J_pinv * e;

    // The new joint configuration
    Config new_config = start_config + d_theta.head(dimension_).cast<float>();

    // Joint-limit check
    if (!cspace_->isInBoundary(new_config)) {
        return {MoveArmResult::JOINT_BOUNDARY_VIOLATION, new_config};
    }

    // Collision check
    auto t_start_col = std::chrono::high_resolution_clock::now();
    bool is_free = cspace_->isCollisionFree(new_config);
    auto t_end_col = std::chrono::high_resolution_clock::now();
    float dt_step = std::chrono::duration<float, std::milli>(t_end_col - t_start_col).count();
    performance_.collision_check_time_ms += dt_step;
    performance_.collision_in_move_ms    += dt_step;

    if (!is_free) {
        return {MoveArmResult::COLLISION_ENVIRONMENT, new_config};
    }

    return {MoveArmResult::MOVED_ONE_STEP, new_config};
}

// ============================================================================
// limitWorkspaceStep — counterpart of the Python limitWorkspaceStep
// ============================================================================

Matrix4f GraspRrt::limitWorkspaceStep(const Matrix4f& delta_pose) {
    Vector3f pos_part = delta_pose.block<3,1>(0, 3);
    float dist_pos = pos_part.norm();

    // Extract the rotation angle from the rotation matrix
    Vector3f axis;
    float angle;
    rotationToAxisAngle(delta_pose.block<3,3>(0, 0), axis, angle);
    float dist_ori = std::abs(angle);

    float factor_pos = 1.0f;
    float factor_ori = 1.0f;

    if (dist_pos > config_.cart_sampling_pos_step_size ||
        dist_ori > config_.cart_sampling_ori_step_size) {

        if (dist_pos > config_.cart_sampling_pos_step_size) {
            factor_pos = config_.cart_sampling_pos_step_size / dist_pos;
        }
        if (dist_ori > config_.cart_sampling_ori_step_size) {
            factor_ori = config_.cart_sampling_ori_step_size / dist_ori;
        }
        // Take the smaller scale factor (keeps position and orientation in sync)
        float factor = std::min(factor_pos, factor_ori);
        factor_pos = factor;
        factor_ori = factor;
    }

    Matrix4f result = axisAngleToMatrix4(axis, angle * factor_ori);
    result.block<3,1>(0, 3) = pos_part * factor_pos;
    return result;
}

// ============================================================================
// calculateGraspScore — counterpart of the Python calculateGraspScore
// ============================================================================

float GraspRrt::calculateGraspScore(const Config& config, int node_id,
                                     bool store_on_success) {
    auto t_start = std::chrono::high_resolution_clock::now();
    performance_.number_of_grasp_scorings++;

    robot_->setJointValues(config);

    // Close the fingers and collect the contact points
    auto contacts = eef_->closeActors();
    eef_->openActors();

    // Contact-count check
    if (static_cast<int>(contacts.size()) < config_.min_grasp_contacts) {
        if (verbose_) {
            std::cout << "[GraspRrt] Low contacts (" << contacts.size()
                      << ") -> zero score\n";
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        performance_.score_grasp_time_ms +=
            std::chrono::duration<float, std::milli>(t_end - t_start).count();
        return 0.0f;
    }

    // Compute the TCP-to-object distance
    Vector3f gcp_pos = robot_->getGCPGlobalPose().block<3,1>(0, 3);
    float dist_to_target = target_object_->calculateDistance(gcp_pos);

    // Score with quality_measure
    quality_measure_->setContactPoints(contacts);
    quality_measure_->calculateGraspQuality();
    float score = quality_measure_->getGraspQuality();
    bool is_force_closure = quality_measure_->isValid();

    std::cout << "[GraspRrt] Grasp score: " << score
              << ", FC: " << (is_force_closure ? "true" : "false") << "\n";

    if (!is_force_closure) score = 0.0f;

    // Store the result
    if (store_on_success && score > 0.0f) {
        GraspInfo info;
        info.distance_to_object = dist_to_target;
        info.grasp_score = score;
        info.rrt_node_id = node_id;
        info.contacts = contacts;

        Matrix4f tcp_pose = robot_->getTCPGlobalPose();
        Matrix4f obj_pose = target_object_->getGlobalPose();
        info.hand_to_object_transform = tcp_pose.inverse() * obj_pose;

        std::lock_guard<std::mutex> lock(grasp_mutex_);
        grasps_.push_back(info);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    performance_.score_grasp_time_ms +=
        std::chrono::duration<float, std::milli>(t_end - t_start).count();
    return score;
}

// ============================================================================
// processNode / processNodes — counterparts of the Python processNode / processNodes
// ============================================================================

bool GraspRrt::processNode(CSpaceNodePtr node) {
    if (!node) return false;

    robot_->setJointValues(node->getConfiguration());
    Matrix4f tcp_pose = robot_->getGCPGlobalPose();
    map_config_tcp_[node->getId()] = tcp_pose;

    Vector3f tcp_pos = tcp_pose.block<3,1>(0, 3);

    // Bottom hemisphere check: only allow grasping from below the apple
    Matrix4f obj_pose = target_object_->getGlobalPose();
    Eigen::Vector4f tcp_h;
    tcp_h << tcp_pos, 1.0f;
    Eigen::Vector4f tcp_local = obj_pose.inverse() * tcp_h;
    if (tcp_local.z() > 0.0f) {
        return true;
    }

    Vector3f diff = tcp_pos - target_object_position_;
    float dist2 = diff.squaredNorm();

    // Method A: track the below-equator node whose TCP is closest to the apple
    if (dist2 < closest_below_dist2_) {
        closest_below_dist2_ = dist2;
        closest_below_node_id_ = node->getId();
    }

    // Descent candidate gate: Euclidean threshold by default; with the nav field enabled it
    // switches to geodesic travel — a hovering node straight above the apple is Euclidean-near
    bool near_enough = dist2 < config_.try_grasps_distance2;
    if (config_.nav_field_enabled) {
        auto sdf_nav_pn = std::dynamic_pointer_cast<SdfCostEvaluator>(
            cspace_->getCostEvaluator());
        if (sdf_nav_pn && sdf_nav_pn->isNavBuilt()) {
            near_enough = sdf_nav_pn->queryNavDistance(tcp_pos)
                        < config_.nav_descent_dist;
        }
    }
    if (pose_sphere_ && near_enough) {
        pose_sphere_->addCSpaceNode(tcp_pos, node);
    } else if (pose_sphere_) {
        if (verbose_) {
            static int print_count_dist = 0;
            if (print_count_dist++ % 50 == 0) {
                std::cout << "[GraspRrt] Node " << node->getId() << " rejected: Below equator but too far (dist2=" << dist2 << " > " << config_.try_grasps_distance2 << ")\n";
            }
        }
    }
    return true;
}

// ============================================================================
// sampleGoalBiased — Method A: closest below-equator node + jitter
// ============================================================================
Config GraspRrt::sampleGoalBiased() {
    // Fallback: if no node below apple equator yet, use uniform random
    if (closest_below_node_id_ < 0) {
        return cspace_->getRandomConfig();
    }
    auto node = tree_->getNode(closest_below_node_id_);
    if (!node) {
        return cspace_->getRandomConfig();
    }
    Config base = node->getConfiguration();

    // Add c-space jitter (uniform in each dimension)
    Config jitter(dimension_);
    for (int i = 0; i < dimension_; i++) {
        float r = 2.0f * randFloat01() - 1.0f;  // [-1, 1]
        jitter[i] = r * goal_bias_jitter_;
    }
    return base + jitter;
}

bool GraspRrt::processNodes(int start_id, int end_id) {
    if (!tree_) return false;

    int act_id = end_id;
    while (act_id >= 0 && act_id != start_id) {
        auto node = tree_->getNode(act_id);
        if (!node) break;
        processNode(node);
        act_id = node->getParentId();
    }
    return true;
}

// ============================================================================
// createSolution — counterpart of the Python Rrt.createSolution
// ============================================================================

bool GraspRrt::createSolution(bool quiet) {
    if (!goal_node_ || goal_node_->getParentId() < 0) {
        std::cerr << "[GraspRrt] WARNING: no path to goal\n";
        return false;
    }

    std::vector<int> path_ids;
    auto current = goal_node_;
    bool failure = false;

    while (current) {
        path_ids.push_back(current->getId());
        if (current->getId() == start_node_->getId()) break;
        if (current->getParentId() < 0) { failure = true; break; }
        current = tree_->getNode(current->getParentId());
    }

    if (failure) return false;

    std::reverse(path_ids.begin(), path_ids.end());
    solution_.clear();
    for (int nid : path_ids) {
        auto node = tree_->getNode(nid);
        if (node) solution_.push_back(node->getConfiguration());
    }

    if (!quiet) {
        std::cout << "[GraspRrt] Solution: " << solution_.size() << " waypoints\n";
    }
    return true;
}

// ============================================================================
// computePseudoInverseJacobian — Jacobian pseudoinverse with DLS damping
// Counterpart of the Python _compute_pseudo_inverse_jacobian
// ============================================================================

MatrixXf GraspRrt::computePseudoInverseJacobian() {
    MatrixXf J = robot_->computeJacobian();

    float damping_factor = 0.03f;
    float svd_threshold = 1e-4f;

    // SVD decomposition: J = U S V^T
    Eigen::JacobiSVD<MatrixXf> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXf S = svd.singularValues();
    Eigen::VectorXf S_inv = Eigen::VectorXf::Zero(S.size());

    for (int i = 0; i < S.size(); ++i) {
        if (S(i) > svd_threshold) {
            S_inv(i) = 1.0f / S(i);
        } else {
            // Levenberg-Marquardt damping
            S_inv(i) = S(i) / (S(i) * S(i) + damping_factor * damping_factor);
        }
    }

    // J⁺ = V Σ⁻¹ U^T
    MatrixXf J_pinv = svd.matrixV() * S_inv.asDiagonal() * svd.matrixU().transpose();
    return J_pinv;
}

// ============================================================================
// Result access
// ============================================================================

std::vector<GraspInfo> GraspRrt::getGraspResults() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(grasp_mutex_));
    return grasps_;
}

int GraspRrt::getGraspResultCount() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(grasp_mutex_));
    return static_cast<int>(grasps_.size());
}

// ============================================================================
// Utility functions
// ============================================================================

float GraspRrt::cartesianPoseDiff(const Matrix4f& p1, const Matrix4f& p2) {
    float pos_diff = (p1.block<3,1>(0, 3) - p2.block<3,1>(0, 3)).norm();

    Eigen::Matrix3f R_diff = p1.block<3,3>(0, 0).transpose() * p2.block<3,3>(0, 0);
    Vector3f axis;
    float angle;
    rotationToAxisAngle(R_diff, axis, angle);

    return pos_diff + std::abs(angle) * 3.0f; // position + weighted angle
}

Matrix4f GraspRrt::axisAngleToMatrix4(const Vector3f& axis, float angle) {
    Matrix4f M = Matrix4f::Identity();
    if (std::abs(angle) < 1e-10f) return M;

    Eigen::AngleAxisf aa(angle, axis.normalized());
    M.block<3,3>(0, 0) = aa.toRotationMatrix();
    return M;
}

void GraspRrt::rotationToAxisAngle(const Eigen::Matrix3f& R,
                                    Vector3f& axis, float& angle) {
    Eigen::AngleAxisf aa(R);
    angle = aa.angle();
    axis = aa.axis();
    if (angle > PI) {
        angle = 2.0f * PI - angle;
        axis = -axis;
    }
}

void GraspRrt::printPerformanceResults() const {
    std::cout << "\n=================== RESULTS ===================\n"
              << "Setup:         " << performance_.setup_time_ms << " ms\n"
              << "Planning:      " << performance_.planning_time_ms << " ms\n"
              << "RRT build:     " << performance_.rrt_time_ms << " ms\n"
              << " ├─ NN Search: " << performance_.nearest_neighbor_time_ms << " ms\n"
              << " ├─ VF Grad:   " << performance_.vf_gradient_time_ms << " ms\n"
              << " ├─ Collision: " << performance_.collision_check_time_ms << " ms\n"
              << "Move to grasp: " << performance_.move_toward_grasp_poses_time_ms
              << " ms (" << performance_.number_of_movements_toward_grasp_pose << " tries)\n"
              << "Grasp scoring: " << performance_.score_grasp_time_ms
              << " ms (" << performance_.number_of_grasp_scorings << " scorings)\n"
              << "Nodes:         " << tree_->getNrOfNodes() << "\n"
              << "Col checks:    " << col_checks_overall_ << "\n"
              << "=================== RESULTS ===================\n\n";
}

void GraspRrt::printGraspInfo(const GraspInfo& info) const {
    std::cout << "  Grasp Info:\n"
              << "    Distance to object: " << info.distance_to_object << "\n"
              << "    Grasp score: " << info.grasp_score << "\n"
              << "    RRT Node ID: " << info.rrt_node_id << "\n"
              << "    Contacts: " << info.contacts.size() << "\n";
}

} // namespace grasp_rrt
