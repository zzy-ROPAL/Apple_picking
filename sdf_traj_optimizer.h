#pragma once
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <pinocchio/algorithm/jacobian.hpp>
#include "types.h"
#include "panda_pinocchio.h"
#include "sdf_cost_evaluator.h"

namespace grasp_rrt {

/**
 * SdfTrajOptimizer (Standalone Lite)
 * ---------------------------------
 * 基于 SDF 梯度的局部轨迹优化器。不依赖 OpenRAVE。
 * 直接利用项目的 Pinocchio 机器人模型和 SdfCostEvaluator。
 */
class SdfTrajOptimizer {
public:
    struct Config {
        int max_iter;
        float alpha;
        float smooth_weight;
        float safe_dist;
        Config() : max_iter(50), alpha(0.5f), smooth_weight(0.1f), safe_dist(0.05f) {}
    };

    /**
     * 针对 VFRRT 路径进行优化
     */
    static std::vector<grasp_rrt::Config> optimize(
        const std::vector<grasp_rrt::Config>& raw_path,
        std::shared_ptr<PandaPinocchioRobot> robot,
        std::shared_ptr<SdfCostEvaluator> sdf_eval,
        const Config& cfg = Config()) 
    {
        std::vector<grasp_rrt::Config> traj = raw_path;
        int n_steps = traj.size();
        if (n_steps < 3) return traj;

        // 获取机器人球体定义
        // 注意：我们需要访问 SdfCostEvaluator 内部的机器人球体信息
        // 由于 RobotSphere 是私有的，我们需要一个辅助结构或直接硬编码 Panda 的关键球
        struct SphereInfo { std::string link; Eigen::Vector3f offset; float radius; };
        std::vector<SphereInfo> spheres = {
            {"panda_link4", {0, 0, 0.05f}, 0.07f},
            {"panda_link5", {0, 0, 0.15f}, 0.08f},
            {"panda_link6", {0, 0, 0},     0.06f},
            {"panda_link7", {0, 0, 0.03f}, 0.05f},
            {"panda_hand",  {0, 0, 0.06f}, 0.08f}
        };

        for (int iter = 0; iter < cfg.max_iter; ++iter) {
            std::vector<grasp_rrt::Config> next_traj = traj;

            for (int i = 1; i < n_steps - 1; ++i) {
                grasp_rrt::Config q = traj[i];
                grasp_rrt::Config grad_q = grasp_rrt::Config::Zero(7);

                // 1. 设置关节角并更新 Pinocchio 数据
                robot->setJointValues(q);
                
                // 2. 针对每个碰撞球计算梯度
                for (const auto& sp : spheres) {
                    Eigen::Matrix4f T = robot->getFramePose(sp.link);
                    Eigen::Vector3f p = T.block<3,1>(0,3) + T.block<3,3>(0,0) * sp.offset;

                    float dist = sdf_eval->querySdf(p) - sp.radius;
                    if (dist < cfg.safe_dist) {
                        // 数值法计算 SDF 梯度 (指向远离障碍物)
                        Eigen::Vector3f sdf_grad = calculateSdfGradient(sdf_eval, p);
                        
                        // 获取该连杆对应位置的 Jacobian (3x7)
                        // 注意：Panda 模型有 9 个自由度（包含手指），所以我们需要补零
                        Eigen::VectorXd full_q = Eigen::VectorXd::Zero(robot->getModel().nq);
                        for (int k = 0; k < std::min((int)q.size(), (int)full_q.size()); ++k) {
                            full_q(k) = q(k);
                        }

                        Eigen::MatrixXd J_6xN = Eigen::MatrixXd::Zero(6, robot->getModel().nv);
                        pinocchio::computeFrameJacobian(robot->getModel(), 
                                                        const_cast<pinocchio::Data&>(robot->getData()), 
                                                        full_q, 
                                                        robot->getModel().getFrameId(sp.link),
                                                        pinocchio::LOCAL_WORLD_ALIGNED,
                                                        J_6xN);
                        
                        Eigen::Matrix<float, 3, 7> J_pos = J_6xN.topRows(3).leftCols(7).cast<float>();
                        
                        // 关节空间梯度更新: dq = alpha * (safe - dist) * J^T * sdf_grad
                        grad_q += J_pos.transpose() * sdf_grad * (cfg.safe_dist - dist);
                    }
                }

                // 3. 应用梯度更新 + 平滑项
                grasp_rrt::Config smooth = (traj[i-1] + traj[i+1] - 2.0f * traj[i]);
                next_traj[i] += cfg.alpha * grad_q + cfg.smooth_weight * smooth;
            }
            traj = next_traj;
        }

        return traj;
    }

private:
    static Eigen::Vector3f calculateSdfGradient(std::shared_ptr<SdfCostEvaluator> sdf, const Eigen::Vector3f& p, float eps = 0.01f) {
        float v0 = sdf->querySdf(p);
        float dx = sdf->querySdf(p + Eigen::Vector3f(eps, 0, 0)) - v0;
        float dy = sdf->querySdf(p + Eigen::Vector3f(0, eps, 0)) - v0;
        float dz = sdf->querySdf(p + Eigen::Vector3f(0, 0, eps)) - v0;
        Eigen::Vector3f g(dx, dy, dz);
        if (g.norm() > 1e-6) return g.normalized();
        return Eigen::Vector3f::Zero();
    }
};

} // namespace grasp_rrt
