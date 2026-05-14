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
    
    float collision_check_time_ms                = 0.0f;
    float vf_gradient_time_ms                    = 0.0f;
    float nearest_neighbor_time_ms               = 0.0f;

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
    int   face_disable_failure_threshold = 15;  // 失败 N 次触发禁用
    int   face_disable_min_active      = 5;     // 永远保留至少 N 个 active 面
    int   face_disable_min_tree_size   = 500;   // 树长到 N 节点后才允许禁用
};

} // namespace grasp_rrt
