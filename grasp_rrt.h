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
#include <iostream>
#include <limits>

namespace grasp_rrt {

// ============================================================================
// GraspRrt — 对应 Python GraspRrt (继承自 Rrt)
// 核心规划器: 在 RRT 探索的同时生成并评估抓取假设
//
// 参考论文: "Integrated Grasp and Motion Planning"
//           N. Vahrenkamp, M. Do, T. Asfour, R. Dillmann, ICRA 2010
// ============================================================================

class GraspRrt {
public:
    /**
     * @param cspace        配置空间 (带 DCD 碰撞检测)
     * @param robot         机器人接口 (FK/IK/Jacobian)
     * @param eef           末端执行器接口 (夹爪控制)
     * @param target        目标物体接口
     * @param quality       抓取质量评估器
     * @param config        算法参数
     */
    GraspRrt(std::shared_ptr<CSpaceSampled> cspace,
             std::shared_ptr<RobotInterface> robot,
             std::shared_ptr<EndEffectorInterface> eef,
             std::shared_ptr<TargetObjectInterface> target,
             std::shared_ptr<GraspQualityInterface> quality,
             const GraspRrtConfig& config = GraspRrtConfig());

    ~GraspRrt() = default;

    // ---- 配置 ----

    void setConfig(const GraspRrtConfig& cfg) { config_ = cfg; }
    const GraspRrtConfig& getConfig() const { return config_; }
    void setVerbose(bool v) { verbose_ = v; }

    // ---- 额外硬质障碍物 ----

    void setHardObstacle(const Vector3f& pos, float trigger_dist = 0.20f) {
        hard_obstacle_pos_     = pos;
        hard_obstacle_trigger_ = trigger_dist;
        hard_obstacle_enabled_ = true;
    }
    void clearHardObstacle() { hard_obstacle_enabled_ = false; }

    // ---- 起点设置 ----

    /**
     * 设置起始配置 (对应 Python GraspRrt.setStart)
     * 注意: GraspRRT 不需要外部设置 goal — goal 在规划过程中自动采样
     */
    bool setStart(const Config& start_config);

    // ---- 主规划接口 ----

    /**
     * 执行 GraspRRT 规划 (对应 Python GraspRrt.plan)
     * 阻塞式, 成功返回 true
     */
    bool plan(bool quiet = false);

    // ---- 结果获取 ----

    ConfigPath getSolution() const { return solution_; }
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

    // Add an extra setup-time contribution (e.g. SDF voxel build).
    // Will be added to setup_time_ms at the end of init().
    void addExtraSetupTimeMs(float ms) { extra_setup_time_ms_ += ms; }

    // Tune Method A (goal-biased sampling). Set p=0.0 to disable.
    void setGoalBiasProb(float p) { goal_bias_prob_ = p; }

    // ---- 重置 ----

    void reset();

private:
    // ---- 初始化 (对应 Python GraspRrt.init) ----
    bool init();

    // ---- 单轮规划循环 (对应 Python doPlanningCycle) ----
    bool doPlanningCycle();

    // ---- 扩展方法 ----

    ExtensionResult extendVFRRT(const Config& target);
    Config computeVectorField(const Config& q, CSpaceNodePtr nn);

    // ---- 路径验证 (Lazy collision check) ----
    bool validateSolutionPath(CSpaceNodePtr goal_node);

    // ---- 抓取方向引导 ----

    /**
     * 对应 Python connectRandomGraspPositionJacobian
     * 从 pose_sphere 选节点, 计算抓取位姿, Jacobian 引导步进
     */
    MoveArmResult connectRandomGraspPositionJacobian();

    /**
     * 根据当前面的失败统计 + 保护机制，决定是否禁用某个 face
     */
    void tryDisableFace(int face_id);

    /**
     * 对应 Python calculateGlobalGraspPose
     * 在物体表面生成候选抓取位姿
     */
    Matrix4f calculateGlobalGraspPose(const Config& config, bool& success);

    // ---- Jacobian 引导步进 ----

    /**
     * 对应 Python moveTowardsGoal
     * 从 startNode 迭代用 DiffIK 向 targetPose 迈步
     */
    MoveArmResult moveTowardsGoal(CSpaceNodePtr start_node,
                                   const Matrix4f& target_pose);

    /**
     * 对应 Python createWorkSpaceSamplingStep
     * 计算当前 → 目标的 delta, 限制步长, DiffIK 转关节增量
     */
    std::pair<MoveArmResult, Config> createWorkspaceSamplingStep(
        const Matrix4f& current_pose, const Matrix4f& target_pose);

    /**
     * 对应 Python createWorkSpaceSamplingStepFromNode
     */
    std::pair<MoveArmResult, Config> createWorkspaceSamplingStepFromNode(
        const Matrix4f& goal_pose, CSpaceNodePtr extend_node);

    /**
     * 对应 Python moveArmDiffKin
     * 单步微分逆运动学: J⁺ · e → dθ
     */
    std::pair<MoveArmResult, Config> moveArmDiffKin(const Matrix4f& delta_pose);

    /**
     * 对应 Python limitWorkspaceStep
     * 限制笛卡尔步长不超过预设最大值
     */
    Matrix4f limitWorkspaceStep(const Matrix4f& delta_pose);

    // ---- 抓取评分 ----

    /**
     * 对应 Python calculateGraspScore
     * 闭合夹爪 → 检测接触 → Force Closure 打分
     */
    float calculateGraspScore(const Config& config, int node_id,
                              bool store_on_success = false);

    // ---- 节点处理 ----

    /** 对应 Python processNode: 缓存 TCP 位姿, 注册到 pose_sphere */
    bool processNode(CSpaceNodePtr node);

    /** Method A: 用 closest_below 节点 + 抖动产生 target */
    Config sampleGoalBiased();

    /** 对应 Python processNodes: 处理路径上每个节点 */
    bool processNodes(int start_id, int end_id);

    // ---- 解路径构造 ----

    bool createSolution(bool quiet = false);

    // ---- Jacobian 伪逆 (带阻尼) ----

    MatrixXf computePseudoInverseJacobian();

    // ---- 工具函数 ----

    /** 笛卡尔位姿距离 (位置 + 加权角度) */
    static float cartesianPoseDiff(const Matrix4f& p1, const Matrix4f& p2);

    /** 轴角 → 4x4 齐次矩阵 */
    static Matrix4f axisAngleToMatrix4(const Vector3f& axis, float angle);

    /** 从旋转矩阵提取轴角 */
    static void rotationToAxisAngle(const Eigen::Matrix3f& R,
                                     Vector3f& axis, float& angle);

    /** 打印 */
    void printPerformanceResults() const;
    void printGraspInfo(const GraspInfo& info) const;

    // ======== 成员变量 ========

    // 核心组件
    std::shared_ptr<CSpaceSampled>         cspace_;
    std::shared_ptr<RobotInterface>        robot_;
    std::shared_ptr<EndEffectorInterface>  eef_;
    std::shared_ptr<TargetObjectInterface>  target_object_;
    std::shared_ptr<GraspQualityInterface>  quality_measure_;

    // RRT 树
    std::unique_ptr<CSpaceTree> tree_;
    CSpaceNodePtr start_node_;
    CSpaceNodePtr goal_node_;

    // 接近方向离散球面
    std::unique_ptr<ApproachDiscretization> pose_sphere_;

    // 配置
    GraspRrtConfig config_;
    int dimension_;

    // 状态
    bool planner_initialized_;
    bool found_solution_;
    bool stop_search_;
    bool verbose_;
    int  cycles_;
    int  last_added_id_;
    int  col_checks_overall_;

    // 额外硬质障碍物
    Vector3f hard_obstacle_pos_     = Vector3f::Zero();
    float    hard_obstacle_trigger_ = 0.20f;
    bool     hard_obstacle_enabled_ = false;

    // 缓存的 J_T (7×6)
    MatrixXf cached_J_T_;
    bool     cached_J_T_valid_ = false;

    // RRT*
    float best_goal_cost_ = 1e9f;
    std::unordered_map<int, Matrix4f> map_config_tcp_;

    // 目标物体位置缓存
    Vector3f target_object_position_;

    // 抓取结果
    std::vector<GraspInfo> grasps_;
    std::mutex grasp_mutex_;

    // 解路径
    ConfigPath solution_;

    // 性能统计
    GraspPlanningPerformance performance_;
    float extra_setup_time_ms_ = 0.0f;  // SDF voxel build, etc. (added in init)

    // Grasp attempt result counters (Group B metrics)
    int n_grasp_goal_reached_ = 0;
    int n_grasp_pose_reached_ = 0;
    int n_grasp_trapped_ = 0;
    int n_grasp_error_ = 0;

    // Goal-biased sampling (Method A): track best below-equator node
    int   closest_below_node_id_ = -1;
    float closest_below_dist2_   = std::numeric_limits<float>::max();
    float goal_bias_prob_        = 0.15f;   // 15% of cycles use goal-biased target
    float goal_bias_jitter_      = 0.25f;   // joint-space jitter scale (rad)

};

} // namespace grasp_rrt
