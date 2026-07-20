#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <memory>
#include <cmath>
#include <string>

namespace grasp_rrt {

// ============================================================================
// 基础类型别名
// ============================================================================

using Config    = Eigen::VectorXf;           // 关节配置向量
using ConfigPath = std::vector<Config>;      // 配置路径
using Matrix4f  = Eigen::Matrix4f;           // 4x4 齐次变换矩阵
using Vector3f  = Eigen::Vector3f;           // 3D 向量
using MatrixXf  = Eigen::MatrixXf;           // 动态大小矩阵 (Jacobian 等)

// ============================================================================
// 常量
// ============================================================================

constexpr float PI          = 3.14159265359f;
constexpr float EPSILON     = 1e-6f;
constexpr int   RECURSION_MAX_DEPTH = 24;    // DCD 递归最大深度

// ============================================================================
// RRT 扩展模式 — 对应 Python RrtMethod
// ============================================================================

enum class RrtMode {
    EXTEND = 0,            // 每次只延伸一步
    CONNECT = 1,           // 延伸直到碰撞
    CONNECT_COMPLETE = 2   // 全连接: 只有完整路径有效时才添加
};

// ============================================================================
// RRT 扩展结果 — 对应 Python ExtensionResult
// ============================================================================

enum class ExtensionResult {
    ERROR,     // 发生错误
    FAILED,    // 延伸失败 (碰撞)
    SUCCESS,   // 延伸完全成功
    PARTIAL    // 部分延伸成功
};

// ============================================================================
// MoveArm 结果 — 对应 Python MoveArmResult
// 描述工作空间采样步 (DifferentialIK) 的结果
// ============================================================================

enum class MoveArmResult {
    MOVED_ONE_STEP,            // 成功移动了一步, 但未到达目标
    GOAL_REACHED,              // 到达了抓取目标位姿
    COLLISION_ENVIRONMENT,     // 与环境发生碰撞
    GRASPABLE_POSE_REACHED,    // 到达了可抓取位置 (分数不够高)
    JOINT_BOUNDARY_VIOLATION,  // 关节超限
    TRAPPED,                   // 陷入局部极小
    ERROR,                     // 一般错误
    FATAL_ERROR                // 致命错误
};

// ============================================================================
// 接触点信息 — 用于抓取质量评估
// ============================================================================

struct ContactInfo {
    Vector3f position  = Vector3f::Zero();   // 接触位置
    Vector3f normal    = Vector3f::Zero();   // 接触法线
    float    force     = 0.0f;               // 接触力
    int      finger_id = -1;                 // 手指 ID
};

// ============================================================================
// GraspInfo — 对应 Python GraspInfo
// 记录一次成功抓取的完整信息
// ============================================================================

struct GraspInfo {
    Matrix4f hand_to_object_transform = Matrix4f::Identity();
    int      rrt_node_id              = -1;
    float    grasp_score              = 0.0f;
    float    distance_to_object       = 0.0f;
    std::vector<ContactInfo> contacts;
};

// ============================================================================
// GraspPlanningPerformance — 对应 Python PlanningPerformance
// ============================================================================

struct GraspPlanningPerformance {
    float setup_time_ms                          = 0.0f;
    float planning_time_ms                       = 0.0f;
    float move_toward_grasp_poses_time_ms        = 0.0f;
    int   number_of_movements_toward_grasp_pose  = 0;
    float score_grasp_time_ms                    = 0.0f;
    int   number_of_grasp_scorings               = 0;
    float rrt_time_ms                            = 0.0f;

    float collision_check_time_ms                = 0.0f;  // 总和 (extend + move + validate)
    float vf_gradient_time_ms                    = 0.0f;
    float nearest_neighbor_time_ms               = 0.0f;

    // 细分: collision check 按来源
    float collision_in_extend_ms                 = 0.0f;  // 从 extendVFRRT
    float collision_in_move_ms                   = 0.0f;  // 从 moveTowardsGoal
    float collision_in_validate_ms               = 0.0f;  // 从 validateSolutionPath

    // 细分: grasp hypothesis 内部
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
// GraspRrtConfig — GraspRRT 算法参数
// ============================================================================

struct GraspRrtConfig {
    // RRT 参数
    int   max_cycles                 = 50000;
    float extend_step_size           = 0.08f;
    float goal_bias_probability      = 0.1f;
    float planning_timeout_ms        = 60000.0f;
    RrtMode rrt_mode                 = RrtMode::CONNECT;

    // GraspRRT 专有
    float probab_grasp_hypothesis    = 0.1f;    // 每轮执行抓取假设的概率
    float grasp_quality_min_score    = 0.01f;   // 合格抓取的最低分数
    float cart_sampling_pos_step_size = 10.0f;  // 笛卡尔位移步长 (mm)
    float cart_sampling_ori_step_size = PI / 18; // 笛卡尔旋转步长 (rad, ~10°)
    float try_grasps_distance2       = 360000.0f; // 尝试抓取的距离阈值² (600²)
    float max_dist_goal_reached      = 1.0f;    // 判定到达目标的位姿距离
    int   min_grasp_contacts         = 3;       // 最少接触点数
    int   move_towards_goal_max_loops = 300;    // moveTowardsGoal 最大循环
    // DCD 参数
    float sampling_size_paths        = 0.08f;   // 路径采样步长
    float sampling_size_dcd          = 0.04f;   // DCD 碰撞检测步长

    // Approach face disable mechanism
    bool  face_disable_enabled         = true;  // 一键开关，方便对照实验
    int   face_disable_failure_threshold = 200; // 失败 N 次触发禁用
    int   face_disable_min_active      = 2;     // 永远保留至少 N 个 active 面 (放宽到 2)
    int   face_disable_min_tree_size   = 500;   // 树长到 N 节点后才允许禁用

    // Vector field shell attractor (替代固定单点引力)
    // 引力源是一个球冠 (苹果下方某角度锥), 每 N 个 cycle 在球冠上随机重新采一个点
    bool  use_shell_attractor           = true;  // false → 退回固定单点 (苹果下方 8cm)
    float shell_attractor_target_radius = 0.04f; // 苹果半径 (球壳中心)
    float shell_attractor_offset        = 0.01f; // 球壳距苹果表面距离 (= shell_radius - target_radius)
    float shell_attractor_cap_height    = 0.015f;// 球冠高度 (从南极向上多高, 默认 1.5cm)
    int   shell_attractor_resample_period = 1000;// 每 N 个 cycle 重新采样一次

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

    // ---- Sample steering ( angle-range cap on extension direction) ----
    // When enabled, the RRT extension primitive applies an axial-cap to
    // q_step_dir whenever the EE (panda_hand reference point) is within
    // `sample_steering_envelope` of a branch surface. The cap reduces the
    // motion component along the branch axis so the resulting EE motion
    // stays at least `min_angle` away from the axis.
    //
    // Cheaper than edge refuse: each cycle produces a useful extension
    // (no retries). Combined with a per-node Jacobian cache, per-cycle
    // overhead is ~5-10 μs.
    //
    //   cos_cap = cos(min_angle) — maximum allowed |Δp̂·â|.
    //   Default 0.866 (cos 30°): motion must be ≥ 30° off the branch axis.
    bool  sample_steering_enabled       = false;
    float sample_steering_envelope      = 0.05f;   // 5 cm pre-contact shell
    float sample_steering_cos_cap       = 0.866f;  // cos(30°)
    // Tensor variant (recommended): order-8 soft-max axial score with the
    // unclamped depth-amplified weight, escape toward the MINIMUM
    // eigenvector, no penetration skip. Fixes the legacy dominant-
    // eigenvector degeneracy and the switch-off-at-contact defect. Also
    // enables improvements A (grasp-descent steering) and B (link6 as a
    // second steered sphere). When true, sample_steering_k /
    // sample_steering_min_anisotropy are unused.
    bool  sample_steering_tensor        = false;
    // Improvement C: bend target = cos_cap * overshoot. 1.0 = land exactly
    // on the allowed-cone boundary (old behavior, systematically the
    // least-radial acceptable motion); 0.7 = land inside the cone,
    // mimicking the interior-average that rejection resampling achieves.
    float sample_steering_overshoot     = 0.7f;
    // ---- Finger-first steering(纯引导,无拒绝) ----
    // 下探接近苹果时软性引导"指尖先接触":
    //   (a) 旋转偏置:把爪口轴 ẑ_g 转向苹果方向 u,提前对准
    //       Δω = w(d)·gain·φ·(ẑ_g×u)̂,φ = 夹角;
    //   (b) 前进软刹车:未对准时缩放朝苹果的前进分量
    //       scale = min_advance + (1−min_advance)·cosφ(下限 0.3,永不禁止);
    //   w(d) = smoothstep,d 从 ff_far 到 ff_near 平滑升到 1(无阈值面)。
    // 动机:侧向先触会把自由苹果拨走(Isaac 实测);硬约束会加拒绝
    // 压力,故全部用转向偏置实现。
    bool  finger_first_steering = false;
    float ff_align_gain         = 1.0f;
    float ff_far                = 0.15f;   // 激活外半径 (m)
    float ff_near               = 0.05f;   // w=1 内半径 (m)
    float ff_min_advance        = 0.3f;    // 前进分量最小保留比例

    // Change 1(sticky escape sign,仅 tensor 转向):±e_min 得分严格
    // 相等,旧规则用含噪提议方向定向导致逐周期符号抛硬币(之字形来源
    // 之一)。开启后箭头沿树父子链/下探步序继承,平局裁决保持一致。
    // 零质量代价 —— 与 sticky winner(不等分选项间的滞回)本质不同。
    bool  sample_steering_sticky_sign   = false;

    // Multi-branch handling: aggregate the K nearest branches inside envelope
    // into a structure tensor M = Σ wᵢ âᵢ âᵢᵀ, then bias against M's dominant
    // eigenvector. Falls back gracefully when fewer than K branches present.
    //   K = 5: balances coverage and noise resistance in dense foliage.
    //   min_anisotropy ∈ [0, 1]: skip bias when (λmax−λmin)/λmax < this. At
    //     ~0.3 a clearly dominant axis exists; near 0 the geometry is too
    //     isotropic to bias against meaningfully.
    int   sample_steering_k              = 5;       // structure tensor: K=5 nearest
    float sample_steering_min_anisotropy = 0.3f;    // skip if (λmax−λmin)/λmax below

    // ---- Sticky-winner (Upgrades A and B) ----
    // Two variants share the same axis-selection logic:
    //   * sticky-sample-steering   → deterministic sample-steering-style bending on the winner's axis
    //   * sticky-edge-refuse → probabilistic sigmoid rejection on the winner's score
    // Both read/write CSpaceNode::winner_branch_id_ for hysteresis.
    bool  sticky_sample_steering_enabled          = false;
    bool  sticky_edge_refuse_enabled        = false;
    float sticky_envelope           = 0.05f;   // 5 cm
    float sticky_cos_cap            = 0.866f;  // cos(30°) for sample-steering bending
    float sticky_hysteresis_margin  = 1.25f;   // challenger must beat incumbent by ×margin
    int   sticky_k                  = 3;       // K nearest branches
    float sticky_edge_refuse_threshold      = 0.9063f; // sigmoid midpoint (cos 25°)
    float sticky_edge_refuse_gain           = 20.0f;   // sigmoid steepness
};

} // namespace grasp_rrt
