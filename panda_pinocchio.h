#pragma once

#include "robot_interface.h"
#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <string>
#include <iostream>
#include <stdexcept>

namespace grasp_rrt {

// ============================================================================
// PandaPinocchioRobot — 基于 Pinocchio 的 Panda 机器人实现
// 提供 FK (正运动学) 和 Jacobian 计算
// ============================================================================

class PandaPinocchioRobot : public RobotInterface {
public:
    /**
     * @param urdf_path  Panda 机器人 URDF 文件路径
     * @param ee_frame_name  末端执行器 frame 名称
     *   常见值: "panda_hand", "panda_link8", "panda_hand_tcp"
     */
    PandaPinocchioRobot(const std::string& urdf_path,
                        const std::string& ee_frame_name = "panda_hand")
        : ee_frame_name_(ee_frame_name) {
        // 从 URDF 构建模型
        pinocchio::urdf::buildModel(urdf_path, model_);
        data_ = pinocchio::Data(model_);

        // 查找末端执行器 frame ID
        ee_frame_id_ = model_.getFrameId(ee_frame_name);
        if (ee_frame_id_ >= static_cast<int>(model_.nframes)) {
            std::cerr << "[PandaPinocchioRobot] WARNING: frame '" << ee_frame_name
                      << "' not found. Available frames:\n";
            for (size_t i = 0; i < model_.nframes; ++i) {
                std::cerr << "  [" << i << "] " << model_.frames[i].name << "\n";
            }
            throw std::runtime_error("EE frame not found: " + ee_frame_name);
        }

        // 初始化关节配置 (7-DOF for Panda arm)
        dimension_ = 7; // Panda arm 有 7 个关节
        current_config_ = Config::Zero(dimension_);

        std::cout << "[PandaPinocchioRobot] Loaded model: "
                  << model_.njoints << " joints, "
                  << model_.nframes << " frames, "
                  << "EE frame: " << ee_frame_name << " (id=" << ee_frame_id_ << ")\n";

        // 预分配 Jacobian 缓存 (6 × nv)
        jacobian_cache_ = Eigen::MatrixXd::Zero(6, model_.nv);
        tmp_J_6d_ = Eigen::MatrixXd::Zero(6, model_.nv);
    }

    // ---- RobotInterface 实现 ----

    void setJointValues(const Config& config) override {
        current_config_ = config;
        jacobians_ready_ = false;

        Eigen::VectorXd q = Eigen::VectorXd::Zero(model_.nq);
        int n = std::min(static_cast<int>(config.size()), model_.nq);
        for (int i = 0; i < n; ++i) {
            q(i) = static_cast<double>(config(i));
        }

        // 轻量级 FK：只算位姿，不算雅可比 (~10us vs ~50us)
        pinocchio::forwardKinematics(model_, data_, q);
        pinocchio::updateFramePlacements(model_, data_);
    }

    /**
     * @brief 预计算所有关节雅可比（仅在需要 Jacobian 时调用！）
     * 必须在 setJointValues 之后、computeSphereJacobianById 之前调用。
     * 基于已有的 FK 数据补算全关节雅可比缓存，开销约 40us。
     */
    void prepareJacobians(float /*unused*/ = 0.0f) override {
        if (jacobians_ready_) return;
        pinocchio::computeJointJacobians(model_, data_);
        jacobians_ready_ = true;
    }

    bool isJacobiansReady() const { return jacobians_ready_; }

    // 直接计算单个 frame 的 3×N 线速度 Jacobian，无需先调用 prepareJacobians。
    // 内部用 computeFrameJacobian（仅遍历从 frame 到根的关节路径），
    // 比 computeJointJacobians（全量）快 2-4 倍（当只需少数 frame 时）。
    MatrixXf computeSphereJacobianDirect(int frame_id, const Eigen::Vector3f& local_offset) const {
        if (frame_id < 0 || frame_id >= static_cast<int>(model_.nframes))
            return MatrixXf::Zero(3, dimension_);

        // computeFrameJacobian 需要当前关节配置 q（不依赖 prepareJacobians）
        Eigen::VectorXd q = Eigen::VectorXd::Zero(model_.nq);
        int n = std::min(dimension_, model_.nq);
        for (int i = 0; i < n; ++i) q(i) = static_cast<double>(current_config_(i));

        tmp_J_6d_.setZero();
        pinocchio::computeFrameJacobian(model_, const_cast<pinocchio::Data&>(data_),
                                        q, frame_id,
                                        pinocchio::LOCAL_WORLD_ALIGNED, tmp_J_6d_);

        const auto& se3 = data_.oMf[frame_id];
        Eigen::Vector3f p_world_offset = se3.rotation().cast<float>() * local_offset;

        Eigen::Matrix3f cross_mat;
        cross_mat << 0, -p_world_offset.z(), p_world_offset.y(),
                     p_world_offset.z(), 0, -p_world_offset.x(),
                    -p_world_offset.y(), p_world_offset.x(), 0;

        return tmp_J_6d_.topRows(3).leftCols(dimension_).cast<float>()
             - cross_mat * tmp_J_6d_.bottomRows(3).leftCols(dimension_).cast<float>();
    }

    Config getJointValues() const override {
        return current_config_;
    }

    int getDimension() const override {
        return dimension_;
    }

    Vector3f base_offset_ = Vector3f(0.0f, 0.0f, 0.4f);

    Matrix4f getGCPGlobalPose() const override {
        // GCP = 末端执行器 frame 位姿
        const auto& se3 = data_.oMf[ee_frame_id_];
        Matrix4f pose = Matrix4f::Identity();
        pose.block<3,3>(0, 0) = se3.rotation().cast<float>();
        pose.block<3,1>(0, 3) = se3.translation().cast<float>() + base_offset_;
        return pose;
    }

    Matrix4f getTCPGlobalPose() const override {
        // TCP 使用与 GCP 相同的 frame (用户可自定义偏移)
        return getGCPGlobalPose();
    }

    MatrixXf computeJacobian() const override {
        // 惰性计算：如果雅可比还没准备好，先补算
        const_cast<PandaPinocchioRobot*>(this)->prepareJacobians();
        jacobian_cache_.setZero();
        pinocchio::getFrameJacobian(model_, const_cast<pinocchio::Data&>(data_), ee_frame_id_,
                                    pinocchio::LOCAL_WORLD_ALIGNED, jacobian_cache_);
        MatrixXf J = jacobian_cache_.leftCols(dimension_).cast<float>();
        return J;
    }

    MatrixXf computeSphereJacobian(const std::string& frame_name, const Eigen::Vector3f& local_offset) const override {
        auto fid = model_.getFrameId(frame_name);
        if (fid >= static_cast<int>(model_.nframes)) {
            return MatrixXf::Zero(3, dimension_);
        }
        
        tmp_J_6d_.setZero();
        pinocchio::getFrameJacobian(model_, const_cast<pinocchio::Data&>(data_), fid, pinocchio::LOCAL_WORLD_ALIGNED, tmp_J_6d_);

        const auto& se3 = data_.oMf[fid];
        Eigen::Vector3f p_world_offset = se3.rotation().cast<float>() * local_offset;

        Eigen::Matrix3f cross_mat;
        cross_mat << 0, -p_world_offset.z(), p_world_offset.y(),
                     p_world_offset.z(), 0, -p_world_offset.x(),
                    -p_world_offset.y(), p_world_offset.x(), 0;

        return tmp_J_6d_.topRows(3).leftCols(dimension_).cast<float>()
             - cross_mat * tmp_J_6d_.bottomRows(3).leftCols(dimension_).cast<float>();
    }

    // ---- 额外方法 ----

    int getFrameId(const std::string& frame_name) const override {
        return static_cast<int>(model_.getFrameId(frame_name));
    }

    Matrix4f getFramePoseById(int frame_id) const override {
        if (frame_id < 0 || frame_id >= static_cast<int>(model_.nframes)) {
            return Matrix4f::Identity();
        }
        const auto& se3 = data_.oMf[frame_id];
        Matrix4f pose = Matrix4f::Identity();
        pose.block<3,3>(0, 0) = se3.rotation().cast<float>();
        pose.block<3,1>(0, 3) = se3.translation().cast<float>() + base_offset_;
        return pose;
    }

    MatrixXf computeSphereJacobianById(int frame_id, const Eigen::Vector3f& local_offset) const override {
        if (frame_id < 0 || frame_id >= static_cast<int>(model_.nframes)) {
            return MatrixXf::Zero(3, dimension_);
        }
        // Reuse pre-allocated tmp_J_6d_ to avoid heap allocation per call
        tmp_J_6d_.setZero();
        pinocchio::getFrameJacobian(model_, const_cast<pinocchio::Data&>(data_), frame_id, pinocchio::LOCAL_WORLD_ALIGNED, tmp_J_6d_);

        const auto& se3 = data_.oMf[frame_id];
        Eigen::Vector3f p_world_offset = se3.rotation().cast<float>() * local_offset;

        Eigen::Matrix3f cross_mat;
        cross_mat << 0, -p_world_offset.z(), p_world_offset.y(),
                     p_world_offset.z(), 0, -p_world_offset.x(),
                    -p_world_offset.y(), p_world_offset.x(), 0;

        // Single-expression return lets the compiler apply NRVO; no J_float temporary
        return tmp_J_6d_.topRows(3).leftCols(dimension_).cast<float>()
             - cross_mat * tmp_J_6d_.bottomRows(3).leftCols(dimension_).cast<float>();
    }
    
    const pinocchio::Model& getModel() const { return model_; }
    const pinocchio::Data& getData() const { return data_; }
    int getEEFrameId() const { return ee_frame_id_; }

    /**
     * 获取指定 frame 的全局位姿
     */
    Matrix4f getFramePose(const std::string& frame_name) const override {
        auto fid = model_.getFrameId(frame_name);
        if (fid >= static_cast<int>(model_.nframes)) {
            return Matrix4f::Identity();
        }
        const auto& se3 = data_.oMf[fid];
        Matrix4f pose = Matrix4f::Identity();
        pose.block<3,3>(0, 0) = se3.rotation().cast<float>();
        pose.block<3,1>(0, 3) = se3.translation().cast<float>() + base_offset_;
        return pose;
    }

    /**
     * 获取关节极限
     */
    std::pair<Eigen::VectorXf, Eigen::VectorXf> getJointLimits() const {
        Eigen::VectorXf lower = model_.lowerPositionLimit.head(dimension_).cast<float>();
        Eigen::VectorXf upper = model_.upperPositionLimit.head(dimension_).cast<float>();
        return {lower, upper};
    }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;
    int ee_frame_id_;
    int dimension_;
    std::string ee_frame_name_;
    Config current_config_;

    // Jacobian 缓存 (6 × model_.nv)
    mutable Eigen::MatrixXd jacobian_cache_ = Eigen::MatrixXd::Zero(6, 1);
    mutable Eigen::MatrixXd tmp_J_6d_;
    bool jacobians_ready_ = false;
};

} // namespace grasp_rrt
