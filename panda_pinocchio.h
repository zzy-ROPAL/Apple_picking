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
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace grasp_rrt {

// ============================================================================
// PandaPinocchioRobot — 基于 Pinocchio 的 Panda 机器人实现 (per-thread state)
// Model 共享只读, Data 等 mutable 状态按线程独立持有.
// ============================================================================

class PandaPinocchioRobot : public RobotInterface {
public:
    PandaPinocchioRobot(const std::string& urdf_path,
                        const std::string& ee_frame_name = "panda_hand")
        : ee_frame_name_(ee_frame_name) {
        pinocchio::urdf::buildModel(urdf_path, model_);

        ee_frame_id_ = model_.getFrameId(ee_frame_name);
        if (ee_frame_id_ >= static_cast<int>(model_.nframes)) {
            std::cerr << "[PandaPinocchioRobot] WARNING: frame '" << ee_frame_name
                      << "' not found. Available frames:\n";
            for (size_t i = 0; i < model_.nframes; ++i) {
                std::cerr << "  [" << i << "] " << model_.frames[i].name << "\n";
            }
            throw std::runtime_error("EE frame not found: " + ee_frame_name);
        }

        dimension_ = 7;

        std::cout << "[PandaPinocchioRobot] Loaded model: "
                  << model_.njoints << " joints, "
                  << model_.nframes << " frames, "
                  << "EE frame: " << ee_frame_name << " (id=" << ee_frame_id_ << ")\n";

        // 默认 1 个线程
        setNumThreads(1);
    }

    // ---- 并行支持 ----

    void setNumThreads(int N) override {
        N = std::max(1, N);
        per_thread_.clear();
        per_thread_.reserve(N);
        for (int i = 0; i < N; ++i) {
            PerThread pt;
            pt.data           = pinocchio::Data(model_);
            pt.current_config = Config::Zero(dimension_);
            pt.jacobian_cache = Eigen::MatrixXd::Zero(6, model_.nv);
            pt.tmp_J_6d       = Eigen::MatrixXd::Zero(6, model_.nv);
            pt.jacobians_ready = false;
            per_thread_.push_back(std::move(pt));
        }
        num_threads_ = N;
    }

    // ---- RobotInterface 实现 ----

    void setJointValues(const Config& config) override {
        auto& pt = perThread();
        pt.current_config = config;
        pt.jacobians_ready = false;

        Eigen::VectorXd q = Eigen::VectorXd::Zero(model_.nq);
        int n = std::min(static_cast<int>(config.size()), model_.nq);
        for (int i = 0; i < n; ++i) {
            q(i) = static_cast<double>(config(i));
        }

        pinocchio::forwardKinematics(model_, pt.data, q);
        pinocchio::updateFramePlacements(model_, pt.data);
    }

    void prepareJacobians(float /*unused*/ = 0.0f) override {
        auto& pt = perThread();
        if (pt.jacobians_ready) return;
        pinocchio::computeJointJacobians(model_, pt.data);
        pt.jacobians_ready = true;
    }

    bool isJacobiansReady() const { return perThread().jacobians_ready; }

    MatrixXf computeSphereJacobianDirect(int frame_id, const Eigen::Vector3f& local_offset) const {
        if (frame_id < 0 || frame_id >= static_cast<int>(model_.nframes))
            return MatrixXf::Zero(3, dimension_);

        auto& pt = perThread();
        Eigen::VectorXd q = Eigen::VectorXd::Zero(model_.nq);
        int n = std::min(dimension_, model_.nq);
        for (int i = 0; i < n; ++i) q(i) = static_cast<double>(pt.current_config(i));

        pt.tmp_J_6d.setZero();
        pinocchio::computeFrameJacobian(model_, pt.data,
                                        q, frame_id,
                                        pinocchio::LOCAL_WORLD_ALIGNED, pt.tmp_J_6d);

        const auto& se3 = pt.data.oMf[frame_id];
        Eigen::Vector3f p_world_offset = se3.rotation().cast<float>() * local_offset;

        Eigen::Matrix3f cross_mat;
        cross_mat << 0, -p_world_offset.z(), p_world_offset.y(),
                     p_world_offset.z(), 0, -p_world_offset.x(),
                    -p_world_offset.y(), p_world_offset.x(), 0;

        return pt.tmp_J_6d.topRows(3).leftCols(dimension_).cast<float>()
             - cross_mat * pt.tmp_J_6d.bottomRows(3).leftCols(dimension_).cast<float>();
    }

    Config getJointValues() const override {
        return perThread().current_config;
    }

    int getDimension() const override {
        return dimension_;
    }

    Vector3f base_offset_ = Vector3f(0.0f, 0.0f, 0.4f);

    Matrix4f getGCPGlobalPose() const override {
        const auto& se3 = perThread().data.oMf[ee_frame_id_];
        Matrix4f pose = Matrix4f::Identity();
        pose.block<3,3>(0, 0) = se3.rotation().cast<float>();
        pose.block<3,1>(0, 3) = se3.translation().cast<float>() + base_offset_;
        return pose;
    }

    Matrix4f getTCPGlobalPose() const override {
        return getGCPGlobalPose();
    }

    MatrixXf computeJacobian() const override {
        const_cast<PandaPinocchioRobot*>(this)->prepareJacobians();
        auto& pt = perThread();
        pt.jacobian_cache.setZero();
        pinocchio::getFrameJacobian(model_, pt.data, ee_frame_id_,
                                    pinocchio::LOCAL_WORLD_ALIGNED, pt.jacobian_cache);
        MatrixXf J = pt.jacobian_cache.leftCols(dimension_).cast<float>();
        return J;
    }

    MatrixXf computeSphereJacobian(const std::string& frame_name, const Eigen::Vector3f& local_offset) const override {
        auto fid = model_.getFrameId(frame_name);
        if (fid >= static_cast<int>(model_.nframes)) {
            return MatrixXf::Zero(3, dimension_);
        }

        auto& pt = perThread();
        pt.tmp_J_6d.setZero();
        pinocchio::getFrameJacobian(model_, pt.data, fid, pinocchio::LOCAL_WORLD_ALIGNED, pt.tmp_J_6d);

        const auto& se3 = pt.data.oMf[fid];
        Eigen::Vector3f p_world_offset = se3.rotation().cast<float>() * local_offset;

        Eigen::Matrix3f cross_mat;
        cross_mat << 0, -p_world_offset.z(), p_world_offset.y(),
                     p_world_offset.z(), 0, -p_world_offset.x(),
                    -p_world_offset.y(), p_world_offset.x(), 0;

        return pt.tmp_J_6d.topRows(3).leftCols(dimension_).cast<float>()
             - cross_mat * pt.tmp_J_6d.bottomRows(3).leftCols(dimension_).cast<float>();
    }

    // ---- 额外方法 ----

    int getFrameId(const std::string& frame_name) const override {
        return static_cast<int>(model_.getFrameId(frame_name));
    }

    Matrix4f getFramePoseById(int frame_id) const override {
        if (frame_id < 0 || frame_id >= static_cast<int>(model_.nframes)) {
            return Matrix4f::Identity();
        }
        const auto& se3 = perThread().data.oMf[frame_id];
        Matrix4f pose = Matrix4f::Identity();
        pose.block<3,3>(0, 0) = se3.rotation().cast<float>();
        pose.block<3,1>(0, 3) = se3.translation().cast<float>() + base_offset_;
        return pose;
    }

    MatrixXf computeSphereJacobianById(int frame_id, const Eigen::Vector3f& local_offset) const override {
        if (frame_id < 0 || frame_id >= static_cast<int>(model_.nframes)) {
            return MatrixXf::Zero(3, dimension_);
        }
        auto& pt = perThread();
        pt.tmp_J_6d.setZero();
        pinocchio::getFrameJacobian(model_, pt.data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, pt.tmp_J_6d);

        const auto& se3 = pt.data.oMf[frame_id];
        Eigen::Vector3f p_world_offset = se3.rotation().cast<float>() * local_offset;

        Eigen::Matrix3f cross_mat;
        cross_mat << 0, -p_world_offset.z(), p_world_offset.y(),
                     p_world_offset.z(), 0, -p_world_offset.x(),
                    -p_world_offset.y(), p_world_offset.x(), 0;

        return pt.tmp_J_6d.topRows(3).leftCols(dimension_).cast<float>()
             - cross_mat * pt.tmp_J_6d.bottomRows(3).leftCols(dimension_).cast<float>();
    }

    const pinocchio::Model& getModel() const { return model_; }
    const pinocchio::Data& getData() const { return perThread().data; }
    int getEEFrameId() const { return ee_frame_id_; }

    Matrix4f getFramePose(const std::string& frame_name) const override {
        auto fid = model_.getFrameId(frame_name);
        if (fid >= static_cast<int>(model_.nframes)) {
            return Matrix4f::Identity();
        }
        const auto& se3 = perThread().data.oMf[fid];
        Matrix4f pose = Matrix4f::Identity();
        pose.block<3,3>(0, 0) = se3.rotation().cast<float>();
        pose.block<3,1>(0, 3) = se3.translation().cast<float>() + base_offset_;
        return pose;
    }

    std::pair<Eigen::VectorXf, Eigen::VectorXf> getJointLimits() const {
        Eigen::VectorXf lower = model_.lowerPositionLimit.head(dimension_).cast<float>();
        Eigen::VectorXf upper = model_.upperPositionLimit.head(dimension_).cast<float>();
        return {lower, upper};
    }

private:
    struct PerThread {
        pinocchio::Data  data;
        Config           current_config;
        Eigen::MatrixXd  jacobian_cache;
        Eigen::MatrixXd  tmp_J_6d;
        bool             jacobians_ready = false;
    };

    pinocchio::Model model_;
    int              ee_frame_id_;
    int              dimension_;
    std::string      ee_frame_name_;

    // 必须 mutable: const 方法 (如 getFramePose) 也要拿到自己线程的 data
    mutable std::vector<PerThread> per_thread_;
    int num_threads_ = 1;

    int tid() const {
#ifdef _OPENMP
        int t = omp_get_thread_num();
        return (t < num_threads_) ? t : 0;
#else
        return 0;
#endif
    }
    PerThread& perThread() const { return per_thread_[tid()]; }
};

} // namespace grasp_rrt
