#pragma once

#include "types.h"

namespace grasp_rrt {

// ============================================================================
// RobotInterface — 替代 Simox VirtualRobot 的抽象接口
// 用户需要根据自己的机器人框架实现这个接口
// (例如: Pinocchio, KDL, PyBullet C++ API, 或自定义的 FK/IK/Jacobian)
// ============================================================================

class RobotInterface {
public:
    virtual ~RobotInterface() = default;

    // ---- 关节管理 ----

    /**
     * 将机器人设置到给定关节配置
     */
    virtual void setJointValues(const Config& config) = 0;

    /**
     * 获取当前关节配置
     */
    virtual Config getJointValues() const = 0;

    /**
     * 获取关节数 (自由度)
     */
    virtual int getDimension() const = 0;

    // ---- 正运动学 ----

    /**
     * 获取 GCP (Grasp Center Point) 全局位姿 (4x4 齐次矩阵)
     * GCP 是抓取中心点, 通常位于夹爪中心
     */
    virtual Matrix4f getGCPGlobalPose() const = 0;

    /**
     * 获取 TCP (Tool Center Point) 全局位姿 (4x4 齐次矩阵)
     * TCP 是末端执行器工具中心点
     */
    virtual Matrix4f getTCPGlobalPose() const = 0;

    /**
     * 获取指定连杆/框架的全局位姿 (4x4 齐次矩阵)
     */
    virtual Matrix4f getFramePose(const std::string& frame_name) const = 0;

    // ---- Jacobian (微分运动学) ----

    /**
     * 计算当前配置下的几何 Jacobian (6 × n_joints)
     * 前 3 行: 线速度部分
     * 后 3 行: 角速度部分
     */
    virtual MatrixXf computeJacobian() const = 0;

    /**
     * @brief 预计算所有关节的雅可比缓存。
     * 仅在需要 Jacobian 时调用（如梯度计算），避免污染普通 FK 路径。
     */
    virtual void prepareJacobians(float reuse_threshold_rad = 0.15f) {}

    /**
     * @brief 计算某个连杆上局部偏移点的 3xN 线性平移雅可比矩阵
     * @param frame_name 连杆名称
     * @param local_offset 该点在连杆局部坐标系下的偏移
     * @return 3xN 的雅可比矩阵 (把关节速度映射到该点的世界空间线速度)
     */
    virtual MatrixXf computeSphereJacobian(const std::string& frame_name, const Eigen::Vector3f& local_offset) const {
        return MatrixXf::Zero(3, getDimension());
    }

    /**
     * @brief O(1) 获取快速访问缓存的 frame_id 及其无哈希计算版本
     */
    virtual int getFrameId(const std::string& frame_name) const { return -1; }
    
    virtual Matrix4f getFramePoseById(int frame_id) const {
        return Matrix4f::Identity();
    }
    
    virtual MatrixXf computeSphereJacobianById(int frame_id, const Eigen::Vector3f& local_offset) const {
        return MatrixXf::Zero(3, getDimension());
    }
};

// ============================================================================
// EndEffectorInterface — 替代 Simox EndEffector 的抽象接口
// 负责夹爪/手指的动作和接触检测
// ============================================================================

class EndEffectorInterface {
public:
    virtual ~EndEffectorInterface() = default;

    /**
     * 关闭夹爪手指, 返回接触点列表
     * 对应 C++ eef->closeActors(graspCollisionObjects)
     */
    virtual std::vector<ContactInfo> closeActors() = 0;

    /**
     * 打开夹爪手指
     * 对应 C++ eef->openActors()
     */
    virtual void openActors() = 0;

    /**
     * 获取末端执行器名称 (可选)
     */
    virtual std::string getName() const { return "EndEffector"; }
};

// ============================================================================
// GraspQualityInterface — 替代 Simox GraspQualityMeasure 的抽象接口
// 负责根据接触点评估抓取质量 (力封闭打分)
// ============================================================================

class GraspQualityInterface {
public:
    virtual ~GraspQualityInterface() = default;

    /**
     * 设置接触点
     */
    virtual void setContactPoints(const std::vector<ContactInfo>& contacts) = 0;

    /**
     * 计算抓取质量
     */
    virtual void calculateGraspQuality() = 0;

    /**
     * 获取抓取质量分数 (越高越好)
     */
    virtual float getGraspQuality() const = 0;

    /**
     * 是否满足力封闭 (Force Closure) 条件
     */
    virtual bool isValid() const = 0;
};

// ============================================================================
// TargetObjectInterface — 目标物体的抽象接口
// 提供物体位姿和表面最近点查询
// ============================================================================

class TargetObjectInterface {
public:
    virtual ~TargetObjectInterface() = default;

    /**
     * 获取物体全局位姿
     */
    virtual Matrix4f getGlobalPose() const = 0;

    /**
     * 获取物体全局位置 (3D)
     */
    virtual Vector3f getPosition() const {
        return getGlobalPose().block<3,1>(0, 3);
    }

    /**
     * 获取物体表面离给定点最近的点
     * 默认实现: 返回物体中心 (用户应覆盖以获得更精确的结果)
     */
    virtual Vector3f getClosestPoint(const Vector3f& query_pos) const {
        return getPosition();
    }

    /**
     * 计算给定点到物体的距离
     */
    virtual float calculateDistance(const Vector3f& pos) const {
        return (pos - getPosition()).norm();
    }
};

} // namespace grasp_rrt
