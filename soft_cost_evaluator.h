#pragma once

#include "cspace_sampled.h"
#include "robot_interface.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>

namespace grasp_rrt {

// ============================================================================
// TreeCanopyCostEvaluator
// 用于评估机器人各个碰撞球与各向异性树冠(圆柱体组)之间的穿透代价。
// 不需要依赖 Bullet3，直接通过简单的几何运算计算最小距离，速度极快。
// ============================================================================

class TreeCanopyCostEvaluator : public CostEvaluator {
public:
    struct CylinderObstacle {
        Eigen::Vector3f p0;     // 圆柱体起点 (局部Z轴向外方向的起点)
        Eigen::Vector3f p1;     // 圆柱体终点
        float radius;
    };

    struct RobotSphere {
        std::string link_name;
        float radius;
        Eigen::Vector3f offset; // 相对于该连杆原点的局部坐标
    };

    TreeCanopyCostEvaluator(std::shared_ptr<RobotInterface> robot) : robot_(robot) {}

    // 添加一片“树叶”(一个圆柱体分支)
    void addBranchCylinder(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, float radius) {
        cylinders_.push_back({p0, p1, radius});
    }

    // 添加机器人的一个碰撞球
    void addRobotSphere(const std::string& link_name, float radius, const Eigen::Vector3f& offset) {
        spheres_.push_back({link_name, radius, offset});
    }

    // 实现 CostEvaluator 接口
    float evaluateCost(const Config& q) override {
        // 1. 设置机器人的关节状态
        robot_->setJointValues(q);

        float total_cost = 0.0f;

        // 2. 遍历每一个碰撞球，计算与所有树叶圆柱体的重叠程度
        for (const auto& rs : spheres_) {
            Eigen::Matrix4f pose = robot_->getFramePose(rs.link_name);
            Eigen::Vector3f p = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * rs.offset;

            for (const auto& cyl : cylinders_) {
                // 计算点 p 到线段 (p0, p1) 的距离
                Eigen::Vector3f v = cyl.p1 - cyl.p0;
                Eigen::Vector3f w = p - cyl.p0;
                float v_sq = v.squaredNorm();
                
                float t = 0.0f;
                if (v_sq > 1e-6f) {
                    t = w.dot(v) / v_sq;
                    t = std::max(0.0f, std::min(1.0f, t));
                }
                
                // 线段上最近的点
                Eigen::Vector3f closest = cyl.p0 + t * v;
                float dist = (p - closest).norm();
                
                // 穿透深度
                float penetration = cyl.radius + rs.radius - dist;
                
                if (penetration > 0.0f) {
                    // 各向异性代价：穿透得越深，代价呈二次方增长
                    total_cost += penetration * penetration * 100.0f; 
                }
            }
        }
        return total_cost;
    }

private:
    std::shared_ptr<RobotInterface> robot_;
    std::vector<CylinderObstacle> cylinders_;
    std::vector<RobotSphere> spheres_;
};

} // namespace grasp_rrt
