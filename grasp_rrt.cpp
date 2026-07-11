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
// 构造函数
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
    // 同步 CSpaceSampled 的 DCD 参数
    cspace_->setSamplingSize(config_.sampling_size_paths);
    cspace_->setSamplingSizeDCD(config_.sampling_size_dcd);

    target_object_position_ = target_object_->getPosition();
}

// ============================================================================
// setStart — 对应 Python GraspRrt.setStart
// ============================================================================

bool GraspRrt::setStart(const Config& start_config) {
    if (!cspace_->isConfigValid(start_config)) {
        std::cerr << "[GraspRrt] ERROR: Start config is invalid (collision or out of bounds)\n";
        return false;
    }

    // 初始化树
    tree_ = std::make_unique<CSpaceTree>(cspace_.get());
    start_node_ = tree_->appendNode(start_config, -1);
    start_node_->setStatus(0);

    // 处理起始节点 (缓存 TCP 位姿)
    processNode(start_node_);

    if (verbose_) {
        std::cout << "[GraspRrt] setStart: " << start_config.transpose() << "\n";
    }
    return true;
}

// ============================================================================
// reset — 对应 Python GraspRrt.reset
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
    {
        std::lock_guard<std::mutex> lk(ee_jacobian_cache_mutex_);
        ee_jacobian_cache_.clear();
    }
    if (tree_) tree_->reset();
}

// ============================================================================
// init — 对应 Python GraspRrt.init
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

    // 初始化接近方向离散球面
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
// plan — 核心主循环 (对应 Python GraspRrt.plan)
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

        // 超时检查
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

    // 计算 RRT 建树时间
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
// doPlanningCycle — 单轮规划 (对应 Python doPlanningCycle)
// ============================================================================

bool GraspRrt::doPlanningCycle() {
    if (!planner_initialized_) return false;

    int col_checks_start = cspace_->getCollisionCheckCount();
    bool do_normal_connect = true;

    maybeResampleShellAttractor();

    float r = randFloat01();

    if (r <= config_.probab_grasp_hypothesis) {
        // ===== 策略 A: 概率触发终端抓取下探 (Terminal Grasp) =====
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
        // ===== 阶段一：VF-RRT 树扩展（含 goal-biased sampling） =====
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
// maybeResampleShellAttractor — 主线程串行调用
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
// doPlanningCycleBatch — Batch RRT 主循环
// 主线程预采 K 个 q_rand → K 路并行做 extend compute → 主线程串行 commit.
// Grasp hypothesis 分支沿用串行路径 (不在并行段内).
// ============================================================================
bool GraspRrt::doPlanningCycleBatch(int K) {
    if (!planner_initialized_) return false;
    if (K <= 1) return doPlanningCycle();

    int col_checks_start = cspace_->getCollisionCheckCount();
    maybeResampleShellAttractor();

    // 1) Grasp hypothesis 分支 (串行) - 与 doPlanningCycle 保持一致.
    //    用主线程 RNG 决定是否触发, 触发则单独走一次抓取尝试, 不做 batch extend.
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
        // Sticky-winner: propagate chosen winner branch id (batch RRT path).
        new_node->setWinnerBranchId(c.chosen_winner_id);
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
// VF-RRT 核心: 计算矢量力场 (斥力 + 引力)
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

    // 2. 硬质障碍物斥力 + 旋涡 (树干 + 额外障碍物)
    // 注: 在 batch RRT 下, 跨线程共享 cached_J_T_ 会产生 race; 改为 per-call 局部变量.
    auto panda_robot = dynamic_cast<PandaPinocchioRobot*>(robot_.get());
    if (panda_robot && !panda_robot->isJacobiansReady()) {
        robot_->prepareJacobians();
    } else if (!panda_robot) {
        robot_->prepareJacobians();
    }
    MatrixXf J_T = robot_->computeJacobian().transpose();

    Matrix4f tcp_pose = robot_->getGCPGlobalPose();
    Vector3f tcp_pos = tcp_pose.block<3,1>(0,3);

    Config hard_trunk_rep_vector = Eigen::VectorXf::Zero(dimension_);
    Config vortex_vector = Eigen::VectorXf::Zero(dimension_);

    // Per-link trunk repulsion: check every major link, not just TCP
    {
        static int fid_l2 = robot_->getFrameId("panda_link2");
        static int fid_l3 = robot_->getFrameId("panda_link3");
        static int fid_l4 = robot_->getFrameId("panda_link4");
        static int fid_l5 = robot_->getFrameId("panda_link5");
        static int fid_l6 = robot_->getFrameId("panda_link6");
        static int fid_l7 = robot_->getFrameId("panda_link7");
        static int fid_lh = robot_->getFrameId("panda_hand");

        struct TrunkPoint { int fid; Vector3f offset; };
        TrunkPoint trunk_points[] = {
            {fid_l2, Vector3f(0,0,0)},
            {fid_l3, Vector3f(0,0,0.15f)},
            {fid_l4, Vector3f(0,0,0.05f)},
            {fid_l5, Vector3f(0,0,0.15f)},
            {fid_l6, Vector3f(0,0,0)},
            {fid_l7, Vector3f(0,0,0.05f)},
            {fid_lh, Vector3f(0,0,0.08f)}
        };

        Vector3f trunk_axis(0.7f, 0.0f, 0.0f);
        constexpr float trunk_trigger = 0.20f;

        for (const auto& pt : trunk_points) {
            Matrix4f pose = robot_->getFramePoseById(pt.fid);
            Vector3f p = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * pt.offset;
            Vector3f p_to_trunk = p - trunk_axis;
            p_to_trunk.z() = 0.0f;
            float dist = p_to_trunk.norm();

            if (dist < trunk_trigger && dist > 0.01f) {
                p_to_trunk.normalize();
                float force_mag = 1.0f / (dist * dist);

                Eigen::VectorXf e_rep = Eigen::VectorXf::Zero(6);
                e_rep.head<3>() = p_to_trunk * force_mag;
                Config link_rep = J_T * e_rep;
                if (link_rep.norm() > 1e-6f) hard_trunk_rep_vector += link_rep.normalized();

                Vector3f vortex_dir = Vector3f(0,0,1).cross(p_to_trunk);
                if (vortex_dir.norm() > 1e-6f) {
                    vortex_dir.normalize();
                    Eigen::VectorXf e_v = Eigen::VectorXf::Zero(6);
                    e_v.head<3>() = vortex_dir * force_mag * 1.5f;
                    Config link_vortex = J_T * e_v;
                    if (link_vortex.norm() > 1e-6f) vortex_vector += link_vortex.normalized();
                }
            }
        }
        if (hard_trunk_rep_vector.norm() > 1e-6f) hard_trunk_rep_vector.normalize();
        if (vortex_vector.norm() > 1e-6f) vortex_vector.normalize();
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
                Eigen::VectorXf e_rep = Eigen::VectorXf::Zero(6);
                e_rep.head<3>() = p_to_obs * force_mag;
                Config node_rep = J_T * e_rep;
                if (node_rep.norm() > 1e-6f) total_hard_obs_rep += node_rep.normalized();
                Vector3f v_dir = Vector3f(0,0,1).cross(p_to_obs);
                if (v_dir.norm() > 1e-6f) {
                    v_dir.normalize();
                    Eigen::VectorXf e_v = Eigen::VectorXf::Zero(6);
                    e_v.head<3>() = v_dir * force_mag * 1.5f;
                    Config bin_v = J_T * e_v;
                    if (bin_v.norm() > 1e-6f) total_hard_obs_rep += bin_v.normalized() * 0.5f;
                }
            }
        }
    }

    // 3. 引力场 — applied to wrist region (link6, link7, hand) + TCP
    //
    // 两种模式 (由 config_.use_shell_attractor 切换):
    //   A. 球冠随机采样 (use_shell_attractor=true):
    //      球壳半径 = target_radius + offset, 球冠高度 = cap_height
    //      每 shell_attractor_resample_period 个 cycle 在球冠上均匀重新采样
    //   B. 固定单点 (use_shell_attractor=false):
    //      引力源固定在苹果下方 8cm
    Vector3f virtual_target;
    if (config_.use_shell_attractor) {
        // 注: shell 重采样逻辑已移出. 主线程在每个 cycle/batch 入口处通过
        // maybeResampleShellAttractor() 决定是否更新 cached_shell_target_.
        // 这里只做只读使用 (并行安全).
        if (cached_shell_target_.norm() < 1e-9f) {
            // Fallback: 第一次调用且 main 还没采过, 退到固定单点
            virtual_target = target_object_position_;
            virtual_target.z() -= 0.08f;
        } else {
            virtual_target = cached_shell_target_;
        }
    } else {
        // 固定单点 (原始设计): 苹果下方 8cm
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
        for (const auto& ap : apts) {
            Matrix4f pose = robot_->getFramePoseById(ap.fid);
            Vector3f link_pos = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * ap.offset;
            Vector3f dir = virtual_target - link_pos;
            if (dir.norm() > 1e-6f) dir.normalize();
            Eigen::VectorXf e_a = Eigen::VectorXf::Zero(6);
            e_a.head<3>() = dir * ap.weight;
            Config link_att = J_T * e_a;
            if (link_att.norm() > 1e-6f) att_vector += link_att.normalized() * ap.weight;
        }
        if (att_vector.norm() > 1e-6f) att_vector.normalize();
    }

    // 4. 加权融合
    vf = 1.2f * rep_vector +
         2.5f * hard_trunk_rep_vector + 2.0f * vortex_vector +
         2.5f * total_hard_obs_rep +
         4.0f * att_vector;
    if (vf.norm() > 1e-6f) vf.normalize();
    return vf;
}

// ============================================================================
// computeAdaptiveStep — 根据 TCP 到障碍物的距离自适应步长
//   远 → 大步（快速覆盖），近 → 小步（避免冲过去）
// ============================================================================
static float computeAdaptiveStep(float dist_to_obstacle, float base_step) {
    // SDF 内部为负 → 已在障碍内（异常情况），用最小步
    if (dist_to_obstacle <= 0.0f) return base_step * 0.4f;

    // 分段映射：dist (m) → step ratio
    // > 0.20m  : 2.4 倍（远，大胆走）
    // 0.10-0.20: 1.6 倍
    // 0.05-0.10: 1.0 倍（默认）
    // 0.02-0.05: 0.6 倍
    // < 0.02m  : 0.4 倍（极近，谨慎）
    float ratio;
    if      (dist_to_obstacle > 0.20f) ratio = 2.4f;
    else if (dist_to_obstacle > 0.10f) ratio = 1.6f;
    else if (dist_to_obstacle > 0.05f) ratio = 1.0f;
    else if (dist_to_obstacle > 0.02f) ratio = 0.6f;
    else                               ratio = 0.4f;
    return base_step * ratio;
}

// ============================================================================
// getOrComputeEEJacobian — Method F (Optimization 1)
// ============================================================================
// Returns the 3×7 EE Jacobian at the given node configuration, cached by
// node_id. On cache miss, computes via PandaPinocchioRobot::computeSphere
// JacobianDirect (or fallback). Robot state assumed already at q on first
// computation (caller guarantees this).
MatrixXf GraspRrt::getOrComputeEEJacobian(int node_id, const Config& q) {
    {
        std::lock_guard<std::mutex> lk(ee_jacobian_cache_mutex_);
        auto it = ee_jacobian_cache_.find(node_id);
        if (it != ee_jacobian_cache_.end()) return it->second;
    }

    // Cache miss: compute. Robot must be at q for FK-derived Jacobian.
    robot_->setJointValues(q);
    MatrixXf J;
    auto panda = dynamic_cast<PandaPinocchioRobot*>(robot_.get());
    Eigen::Vector3f zero_off = Eigen::Vector3f::Zero();
    if (panda) {
        int frame_id = robot_->getFrameId("panda_hand");
        J = panda->computeSphereJacobianDirect(frame_id, zero_off);
    } else {
        robot_->prepareJacobians();
        J = robot_->computeSphereJacobian("panda_hand", zero_off);
    }

    {
        std::lock_guard<std::mutex> lk(ee_jacobian_cache_mutex_);
        ee_jacobian_cache_[node_id] = J;
    }
    return J;
}

// ============================================================================
// extendVFRRT — Lazy VF-RRT: 无碰撞检测扩展，路径验证延迟到找到抓取后
// ============================================================================
// Pure-compute part of extendVFRRT.
// 只读 tree_/cspace_/robot_, 不写 tree_/performance_/last_added_id_.
// 可在多线程并行调用 (前提: Phase 1+2 已就绪).
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
    // q_rand_dir 是已经 normalize 过的扩展方向, 作为 motion_dir 传给 VF
    // (Option A 的轴向 drag 需要它; K_ax=0 时此参数被忽略).
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

    // ---- Method F: angle-range sampling bias (Optimizations 1-4) ----
    // Cap the axial-vs-branch-axis component of q_step_dir when the EE is
    // within envelope of a branch. Uses cached J_EE keyed by node id.
    if (config_.sampling_bias_enabled) {
        auto sdf_eval = std::dynamic_pointer_cast<SdfCostEvaluator>(
            cspace_->getCostEvaluator());
        if (sdf_eval) {
            const MatrixXf& J_EE = getOrComputeEEJacobian(nn->getId(), q_near);
            // Ensure robot is at q_near so SDF / axis lookup matches.
            robot_->setJointValues(q_near);
            bool applied = sdf_eval->applySamplingBias(
                q_near, q_step_dir, J_EE,
                config_.sampling_bias_envelope,
                config_.sampling_bias_cos_cap,
                config_.sampling_bias_k,
                config_.sampling_bias_min_anisotropy);
            if (applied && q_step_dir.norm() > 1e-6f) {
                q_step_dir.normalize();
            }
        }
    }

    // ---- Sticky-winner selection (shared by sticky-F and sticky-BCD) ----
    // Selection + F bending happen HERE (they shape q_step_dir before the
    // edge is formed). BCD rejection is deferred until after new_config is
    // known, so it can sub-sample the real edge (see below).
    int   sticky_winner_id = -1;   // committed winner branch (for BCD rejection)
    if (config_.sticky_f_enabled || config_.sticky_bcd_enabled) {
        auto sdf_eval = std::dynamic_pointer_cast<SdfCostEvaluator>(
            cspace_->getCostEvaluator());
        if (sdf_eval) {
            const MatrixXf& J_EE = getOrComputeEEJacobian(nn->getId(), q_near);
            robot_->setJointValues(q_near);
            int prev_winner = nn->getWinnerBranchId();
            auto sw = sdf_eval->stickyWinnerSelect(
                q_near, q_step_dir, J_EE,
                prev_winner,
                config_.sticky_envelope,
                config_.sticky_k,
                config_.sticky_hysteresis_margin);
            cand.chosen_winner_id = sw.chosen_id;
            sticky_winner_id      = sw.chosen_id;

            if (sw.chosen_id >= 0 && config_.sticky_f_enabled) {
                // F-style deterministic bending (before edge is formed).
                sdf_eval->bendAgainstAxis(
                    q_step_dir, J_EE, sw.chosen_axis,
                    config_.sticky_cos_cap);
                if (q_step_dir.norm() > 1e-6f) q_step_dir.normalize();
            }
        }
    }

    float step_size = config_.extend_step_size;
    auto evaluator = cspace_->getCostEvaluator();
    if (evaluator) {
        Vector3f tcp_q_near = robot_->getGCPGlobalPose().block<3,1>(0,3);
        float dist = evaluator->queryDistance(tcp_q_near);
        if (dist < std::numeric_limits<float>::max()) {
            step_size = computeAdaptiveStep(dist, config_.extend_step_size);
        }
    }
    Config new_config = q_near + q_step_dir * step_size;

    // ---- Sticky-BCD rejection (deferred): score the REAL edge -----------
    // Now that new_config is known, evaluate the winner branch's axial score
    // over the full edge q_near→new_config using both engagement spheres and
    // DCD-granularity sub-sampling (catches envelope-entry / mid-edge-axial
    // that a single q_near Jacobian prediction misses).
    if (config_.sticky_bcd_enabled && sticky_winner_id >= 0) {
        auto sdf_eval = std::dynamic_pointer_cast<SdfCostEvaluator>(
            cspace_->getCostEvaluator());
        if (sdf_eval) {
            float score = sdf_eval->stickyWinnerEdgeScore(
                q_near, new_config, sticky_winner_id,
                config_.sticky_envelope,
                config_.sampling_size_dcd);
            float x = config_.sticky_bcd_gain *
                      (score - config_.sticky_bcd_threshold);
            float p_reject = (x >= 0.0f)
                ? 1.0f / (1.0f + std::exp(-x))
                : std::exp(x) / (1.0f + std::exp(x));
            if (randFloat01() < p_reject) {
                // Rejected — return invalid candidate, do not extend.
                return cand;
            }
        }
    }

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
    // Sticky-winner: propagate the chosen winner branch id onto the new node.
    // If sticky mode was off, chosen_winner_id is -1 (harmless default).
    new_node->setWinnerBranchId(c.chosen_winner_id);
    last_added_id_ = new_node->getId();

    processNode(new_node);
    return ExtensionResult::SUCCESS;
}

// ============================================================================
// validateSolutionPath — Lazy 碰撞验证: 找到抓取后才检查完整路径
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
// 对应 Python GraspRrt.connectRandomGraspPositionJacobian
// ============================================================================

MoveArmResult GraspRrt::connectRandomGraspPositionJacobian() {
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. 从离散球面中获取采样方向均匀的节点
    CSpaceNodePtr nn_node = pose_sphere_->getGoodRatedNode();
    if (!nn_node) {
        if (verbose_) std::cout << "[GraspRrt] No good rated node in pose sphere\n";
        // 没节点可挑就直接 TRAPPED，不计入 face stats（face_id == -1）
        n_grasp_trapped_++;
        return MoveArmResult::TRAPPED;
    }
    int face_id = pose_sphere_->getLastSelectedFace();

    // 2. 在物体表面计算抓取目标位姿
    auto t_pose_start = std::chrono::high_resolution_clock::now();
    bool pose_ok = false;
    Matrix4f goal_pose = calculateGlobalGraspPose(nn_node->getConfiguration(), pose_ok);
    auto t_pose_end = std::chrono::high_resolution_clock::now();
    performance_.pose_calc_ms += std::chrono::duration<float, std::milli>(t_pose_end - t_pose_start).count();
    if (!pose_ok) {
        // 把 pose 计算失败也算作该面的一次失败
        pose_sphere_->recordGraspAttempt(face_id, /*success=*/false);
        tryDisableFace(face_id);
        n_grasp_error_++;
        return MoveArmResult::ERROR;
    }

    // 3. 从该节点出发, 向抓取位姿迈步
    MoveArmResult res = moveTowardsGoal(nn_node, goal_pose);

    // 4. 累加结果计数 (Group B metrics)
    switch (res) {
        case MoveArmResult::GOAL_REACHED:           n_grasp_goal_reached_++; break;
        case MoveArmResult::GRASPABLE_POSE_REACHED: n_grasp_pose_reached_++; break;
        case MoveArmResult::TRAPPED:                n_grasp_trapped_++;      break;
        case MoveArmResult::ERROR:                  n_grasp_error_++;        break;
        default: break;  // FATAL_ERROR 不计入
    }

    // 5. 更新面统计 + 触发禁用
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

    // 收集解路径上的节点 id (沿 parent 走到根)
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
// calculateGlobalGraspPose — 对应 Python calculateGlobalGraspPose
// ============================================================================

Matrix4f GraspRrt::calculateGlobalGraspPose(const Config& config, bool& success) {
    success = false;

    robot_->setJointValues(config);
    Matrix4f gcp_pose = robot_->getGCPGlobalPose();

    // 获取物体表面最近点
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

    // 在 GCP 局部坐标系中计算 P1 的方向
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

    // Z 轴 = GCP 的默认接近方向
    Vector3f vec_z(0.0f, 0.0f, 1.0f);

    // 求旋转: 把 Z 轴旋转到指向目标的方向
    float dot = std::max(-1.0f, std::min(1.0f, vec_z.dot(vec_target_local)));
    float angle = std::acos(dot);
    Vector3f rot_axis = vec_z.cross(vec_target_local);
    float axis_norm = rot_axis.norm();

    Matrix4f rot_matrix = Matrix4f::Identity();
    if (axis_norm > 1e-10f) {
        rot_axis /= axis_norm;
        rot_matrix = axisAngleToMatrix4(rot_axis, angle);
    }

    // 转换到全局坐标
    Matrix4f goal_pose = gcp_pose * rot_matrix;
    goal_pose.block<3,1>(0, 3) = P1; // 目标位置设为物体表面点

    success = true;
    return goal_pose;
}

// ============================================================================
// moveTowardsGoal — 对应 Python moveTowardsGoal
// ============================================================================

MoveArmResult GraspRrt::moveTowardsGoal(CSpaceNodePtr start_node,
                                         const Matrix4f& target_pose) {
    if (!start_node) return MoveArmResult::ERROR;

    auto last_extend_node = start_node;

    // 当前 TCP 到目标的笛卡尔距离
    Matrix4f current_tcp = Matrix4f::Identity();
    auto it = map_config_tcp_.find(last_extend_node->getId());
    if (it != map_config_tcp_.end()) current_tcp = it->second;

    float dist = cartesianPoseDiff(current_tcp, target_pose);
    float last_dist = 0.0f;

    for (int loop = 0; loop < config_.move_towards_goal_max_loops; ++loop) {
        // 计算工作空间采样步
        auto [res, tmp_config] = createWorkspaceSamplingStepFromNode(
            target_pose, last_extend_node);

        // ---- 处理步进结果 ----

        if (res == MoveArmResult::ERROR || res == MoveArmResult::FATAL_ERROR) {
            return res;
        }

        if (res == MoveArmResult::COLLISION_ENVIRONMENT) {
            // 碰撞 → 尝试在最后有效配置处抓取
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
            // 检查路径有效性
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

            // 添加到 RRT 树
            auto [success, new_id] = tree_->appendPath(last_extend_node, tmp_config);
            if (!success) return MoveArmResult::ERROR;
            last_added_id_ = new_id;

            processNodes(last_extend_node->getId(), last_added_id_);

            last_extend_node = tree_->getNode(last_added_id_);
            last_extend_node->setStatus(2); // 标记为 moveArm 创建

            last_dist = dist;
            auto tcp_it = map_config_tcp_.find(last_extend_node->getId());
            Matrix4f tcp_pose = (tcp_it != map_config_tcp_.end())
                                ? tcp_it->second : Matrix4f::Identity();
            dist = cartesianPoseDiff(tcp_pose, target_pose);

            // 检查是否陷入 (trapped)
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

        // 检查是否到达目标
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
// createWorkspaceSamplingStep — 对应 Python createWorkSpaceSamplingStep
// ============================================================================

std::pair<MoveArmResult, Config> GraspRrt::createWorkspaceSamplingStep(
    const Matrix4f& current_pose, const Matrix4f& goal_pose) {
    // 计算 delta
    Matrix4f delta_pose = goal_pose * current_pose.inverse();
    delta_pose.block<3,1>(0, 3) = goal_pose.block<3,1>(0, 3) - current_pose.block<3,1>(0, 3);

    // 限制步长
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
// moveArmDiffKin — 对应 Python moveArmDiffKin
// 单步微分逆运动学: J⁺ · e → dθ
// ============================================================================

std::pair<MoveArmResult, Config> GraspRrt::moveArmDiffKin(const Matrix4f& delta_pose) {
    // 获取当前关节值
    Config start_config = robot_->getJointValues();

    // 构建 6D 误差向量: [平移(3) | 旋转(3)]
    Eigen::VectorXf e(6);
    e.head<3>() = delta_pose.block<3,1>(0, 3);

    // 从旋转矩阵提取轴角作为旋转误差
    Vector3f axis;
    float angle;
    rotationToAxisAngle(delta_pose.block<3,3>(0, 0), axis, angle);
    e.tail<3>() = axis * angle;

    // Jacobian 伪逆
    MatrixXf J_pinv = computePseudoInverseJacobian();
    Eigen::VectorXf d_theta = J_pinv * e;

    // 新的关节配置
    Config new_config = start_config + d_theta.head(dimension_).cast<float>();

    // 关节边界检查
    if (!cspace_->isInBoundary(new_config)) {
        return {MoveArmResult::JOINT_BOUNDARY_VIOLATION, new_config};
    }

    // 碰撞检查
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
// limitWorkspaceStep — 对应 Python limitWorkspaceStep
// ============================================================================

Matrix4f GraspRrt::limitWorkspaceStep(const Matrix4f& delta_pose) {
    Vector3f pos_part = delta_pose.block<3,1>(0, 3);
    float dist_pos = pos_part.norm();

    // 从旋转矩阵提取旋转角
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
        // 取较小的缩放因子 (保持位姿同步)
        float factor = std::min(factor_pos, factor_ori);
        factor_pos = factor;
        factor_ori = factor;
    }

    Matrix4f result = axisAngleToMatrix4(axis, angle * factor_ori);
    result.block<3,1>(0, 3) = pos_part * factor_pos;
    return result;
}

// ============================================================================
// calculateGraspScore — 对应 Python calculateGraspScore
// ============================================================================

float GraspRrt::calculateGraspScore(const Config& config, int node_id,
                                     bool store_on_success) {
    auto t_start = std::chrono::high_resolution_clock::now();
    performance_.number_of_grasp_scorings++;

    robot_->setJointValues(config);

    // 闭合手指, 获取接触点
    auto contacts = eef_->closeActors();
    eef_->openActors();

    // 接触点数量检查
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

    // 计算 TCP 到物体的距离
    Vector3f gcp_pos = robot_->getGCPGlobalPose().block<3,1>(0, 3);
    float dist_to_target = target_object_->calculateDistance(gcp_pos);

    // 使用 quality_measure 打分
    quality_measure_->setContactPoints(contacts);
    quality_measure_->calculateGraspQuality();
    float score = quality_measure_->getGraspQuality();
    bool is_force_closure = quality_measure_->isValid();

    std::cout << "[GraspRrt] Grasp score: " << score
              << ", FC: " << (is_force_closure ? "true" : "false") << "\n";

    if (!is_force_closure) score = 0.0f;

    // 存储结果
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
// processNode / processNodes — 对应 Python processNode / processNodes
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

    if (pose_sphere_ && dist2 < config_.try_grasps_distance2) {
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
// createSolution — 对应 Python Rrt.createSolution
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
// computePseudoInverseJacobian — 带 DLS 阻尼的 Jacobian 伪逆
// 对应 Python _compute_pseudo_inverse_jacobian
// ============================================================================

MatrixXf GraspRrt::computePseudoInverseJacobian() {
    MatrixXf J = robot_->computeJacobian();

    float damping_factor = 0.03f;
    float svd_threshold = 1e-4f;

    // SVD 分解: J = U Σ V^T
    Eigen::JacobiSVD<MatrixXf> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXf S = svd.singularValues();
    Eigen::VectorXf S_inv = Eigen::VectorXf::Zero(S.size());

    for (int i = 0; i < S.size(); ++i) {
        if (S(i) > svd_threshold) {
            S_inv(i) = 1.0f / S(i);
        } else {
            // Levenberg-Marquardt 阻尼
            S_inv(i) = S(i) / (S(i) * S(i) + damping_factor * damping_factor);
        }
    }

    // J⁺ = V Σ⁻¹ U^T
    MatrixXf J_pinv = svd.matrixV() * S_inv.asDiagonal() * svd.matrixU().transpose();
    return J_pinv;
}

// ============================================================================
// 结果获取
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
// 工具函数
// ============================================================================

float GraspRrt::cartesianPoseDiff(const Matrix4f& p1, const Matrix4f& p2) {
    float pos_diff = (p1.block<3,1>(0, 3) - p2.block<3,1>(0, 3)).norm();

    Eigen::Matrix3f R_diff = p1.block<3,3>(0, 0).transpose() * p2.block<3,3>(0, 0);
    Vector3f axis;
    float angle;
    rotationToAxisAngle(R_diff, axis, angle);

    return pos_diff + std::abs(angle) * 3.0f; // 位置 + 加权角度
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
