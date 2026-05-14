#pragma once

#include "cspace_sampled.h"
#include "robot_interface.h"
#include "panda_pinocchio.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <chrono>

namespace grasp_rrt {

// ============================================================================
// SdfCostEvaluator
// 用预计算的 3D 有符号距离场 (SDF) 替代 TreeCanopyCostEvaluator 中的
// 逐次球-圆柱距离计算，将 evaluateCost() 从 O(k×m) 降至 O(k)。
//
// 使用方法:
//   1. addBranchCylinder(p0, p1, radius)  — 添加所有树枝圆柱 (与原接口相同)
//   2. addRobotSphere(link, radius, offset) — 添加所有机器人碰撞球 (与原接口相同)
//   3. buildSdf()  — 规划开始前调用一次，耗时 << 1s，之后查询 O(1)
//   4. cspace->setCostEvaluator(cost_evaluator)
//
// 原理:
//   - 根据所有圆柱几何自动计算世界坐标包围盒，再扩展 padding
//   - 将包围盒划分为分辨率为 voxel_size 的体素网格
//   - 每个体素存储到最近圆柱表面的有符号距离 (内部为负)
//   - evaluateCost() 对每个机器人球通过三线性插值 O(1) 查询 SDF，
//     计算穿透深度并累加二次代价，与原始公式完全一致
// ============================================================================

class SdfCostEvaluator : public CostEvaluator {
public:
    struct CylinderObstacle {
        Eigen::Vector3f p0;
        Eigen::Vector3f p1;
        float radius;
    };

    struct RobotSphere {
        std::string link_name;
        float radius;
        Eigen::Vector3f offset;
        int frame_id;
    };

    // voxel_size: 体素分辨率 (m)，默认 1cm，精度和速度的较好平衡点
    explicit SdfCostEvaluator(std::shared_ptr<RobotInterface> robot,
                              float voxel_size = 0.01f)
        : robot_(robot), voxel_size_(voxel_size),
          nx_(0), ny_(0), nz_(0), sdf_built_(false) {}

    // ---- 与 TreeCanopyCostEvaluator 相同的构建接口 ----

    void addBranchCylinder(const Eigen::Vector3f& p0,
                           const Eigen::Vector3f& p1,
                           float radius) {
        cylinders_.push_back({p0, p1, radius});
        sdf_built_ = false;
    }

    void addRobotSphere(const std::string& link_name,
                        float radius,
                        const Eigen::Vector3f& offset) {
        spheres_.push_back({link_name, radius, offset, robot_->getFrameId(link_name)});
        // Pre-allocate Jacobian cache slot (3×7 for Panda); no reallocation during planning
        sphere_jac_cache_.emplace_back(3, robot_->getDimension());
    }

    // ---- 建表 ----

    // 根据所有圆柱的几何范围自动计算包围盒，再向外扩展 padding 米，
    // 然后对整个网格逐体素计算到最近圆柱的距离并缓存。
    // padding 应 >= 最大机器人球半径 + 一定余量，确保球在接近时能拿到正确梯度。
    void buildSdf(float padding = 0.40f) {
        if (cylinders_.empty()) {
            std::cerr << "[SDF] WARNING: No cylinders — SDF trivially empty.\n";
            sdf_built_ = true;
            return;
        }

        // 1. 从圆柱几何自动推导包围盒
        Eigen::Vector3f bb_min = Eigen::Vector3f::Constant( std::numeric_limits<float>::max());
        Eigen::Vector3f bb_max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::max());
        for (const auto& cyl : cylinders_) {
            Eigen::Vector3f r3 = Eigen::Vector3f::Constant(cyl.radius);
            bb_min = bb_min.cwiseMin(cyl.p0 - r3).cwiseMin(cyl.p1 - r3);
            bb_max = bb_max.cwiseMax(cyl.p0 + r3).cwiseMax(cyl.p1 + r3);
        }
        bbox_min_ = bb_min - Eigen::Vector3f::Constant(padding);
        bbox_max_ = bb_max + Eigen::Vector3f::Constant(padding);

        // 2. 计算网格尺寸
        Eigen::Vector3f extent = bbox_max_ - bbox_min_;
        nx_ = static_cast<int>(std::ceil(extent.x() / voxel_size_)) + 1;
        ny_ = static_cast<int>(std::ceil(extent.y() / voxel_size_)) + 1;
        nz_ = static_cast<int>(std::ceil(extent.z() / voxel_size_)) + 1;

        const long long total = (long long)nx_ * ny_ * nz_;
        std::cout << "[SDF] Bounding box : ["
                  << bbox_min_.transpose() << "] ~ ["
                  << bbox_max_.transpose() << "]\n"
                  << "[SDF] Grid size    : " << nx_ << " x " << ny_ << " x " << nz_
                  << "  =  " << total << " voxels  ("
                  << total * sizeof(float) / 1024.0f / 1024.0f << " MB)\n"
                  << "[SDF] Building...  (cylinders=" << cylinders_.size() << ")\n";

        // 3. 逐体素填充最小有符号距离（计时）
        sdf_grid_.resize(total);
        auto t_start = std::chrono::steady_clock::now();

        for (int iz = 0; iz < nz_; ++iz) {
            for (int iy = 0; iy < ny_; ++iy) {
                for (int ix = 0; ix < nx_; ++ix) {
                    Eigen::Vector3f p = voxelCenter(ix, iy, iz);
                    float min_dist = std::numeric_limits<float>::max();
                    for (const auto& cyl : cylinders_) {
                        min_dist = std::min(min_dist, distToCylinder(p, cyl));
                    }
                    sdf_grid_[linearIdx(ix, iy, iz)] = min_dist;
                }
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        float build_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

        sdf_built_ = true;
        std::cout << "[SDF] Build complete in " << build_ms << " ms. "
                  << "O(1) distance queries ready.\n";
    }

    // ---- CostEvaluator 接口 ----

    // 代价计算：与 TreeCanopyCostEvaluator 完全相同的公式，
    // 但用 SDF 查询替代逐圆柱循环，速度提升 ~m 倍 (m = 圆柱数)。
    float evaluateCost(const Config& q) override {
        if (!sdf_built_) {
            throw std::runtime_error(
                "[SdfCostEvaluator] buildSdf() must be called before planning!");
        }

        robot_->setJointValues(q);
        float total_cost = 0.0f;

        for (const auto& rs : spheres_) {
            // 正运动学：球心世界坐标 (O(1) 访问)
            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            Eigen::Vector3f p = pose.block<3,1>(0,3)
                              + pose.block<3,3>(0,0) * rs.offset;

            // SDF 查询：球心到最近圆柱表面的有符号距离
            float sdf_val = sampleSdf(p);

            // 穿透深度 = 球半径 - 表面距离（球心进入圆柱时 sdf_val < 0）
            float penetration = rs.radius - sdf_val;

            if (penetration > 0.0f) {
                // 与原始公式一致：二次代价
                total_cost += penetration * penetration * 100.0f;
            }
        }

        return total_cost;
    }

    // ---- 梯度计算 (解析雅可比法) ----
    Config evaluateGradient(const Config& q) override {
        if (!sdf_built_) {
            throw std::runtime_error("[SdfCostEvaluator] buildSdf() must be called!");
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        robot_->setJointValues(q);
        auto t1 = std::chrono::high_resolution_clock::now();

        Config total_gradient = Eigen::VectorXf::Zero(q.size());
        float delta = voxel_size_ * 2.0f;
        float inv_2d = 1.0f / (2.0f * delta);
        bool jacobians_prepared = false;
        int active_spheres = 0;

        for (const auto& rs : spheres_) {
            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            Eigen::Vector3f p = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * rs.offset;

            float sdf_val = sampleSdf(p);
            float penetration = rs.radius - sdf_val;

            if (penetration > 0.0f) {
                active_spheres++;
                jacobians_prepared = true;  // track that at least one sphere needed Jacobian

                Eigen::Vector3f grad_sdf;
                grad_sdf.x() = (sampleSdf(p + Eigen::Vector3f(delta, 0, 0)) - sampleSdf(p - Eigen::Vector3f(delta, 0, 0))) * inv_2d;
                grad_sdf.y() = (sampleSdf(p + Eigen::Vector3f(0, delta, 0)) - sampleSdf(p - Eigen::Vector3f(0, delta, 0))) * inv_2d;
                grad_sdf.z() = (sampleSdf(p + Eigen::Vector3f(0, 0, delta)) - sampleSdf(p - Eigen::Vector3f(0, 0, delta))) * inv_2d;

                Eigen::Vector3f grad_x_cost = -200.0f * penetration * grad_sdf;
                // Direct per-frame Jacobian: avoids computeJointJacobians (full-body, all joints)
                // Uses computeFrameJacobian internally (only traverses from frame to root)
                auto panda = dynamic_cast<PandaPinocchioRobot*>(robot_.get());
                if (panda) {
                    sphere_jac_cache_[&rs - spheres_.data()] = panda->computeSphereJacobianDirect(rs.frame_id, rs.offset);
                } else {
                    robot_->prepareJacobians();
                    sphere_jac_cache_[&rs - spheres_.data()] = robot_->computeSphereJacobianById(rs.frame_id, rs.offset);
                }
                total_gradient.noalias() += sphere_jac_cache_[&rs - spheres_.data()].transpose() * grad_x_cost;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        static int n = 0, n_active = 0;
        static double acc_fk = 0, acc_jac = 0, acc_loop = 0;
        acc_fk   += std::chrono::duration<double, std::micro>(t1 - t0).count();
        acc_loop += std::chrono::duration<double, std::micro>(t2 - t1).count();
        if (jacobians_prepared) { acc_jac += std::chrono::duration<double, std::micro>(t2 - t1).count(); n_active++; }
        n++;
        if (n % 500 == 0) {
            std::cout << "[GRAD @" << n << "] FK=" << acc_fk/n << "us  "
                      << "SphereLoop=" << acc_loop/n << "us  "
                      << "Jac-calls=" << n_active << "/" << n
                      << " (jac-avg=" << (n_active ? acc_jac/n_active : 0) << "us)\n";
        }

        return total_gradient;
    }

    // 查询某世界坐标点的原始 SDF 值（用于可视化/调试）
    float querySdf(const Eigen::Vector3f& p) const {
        if (!sdf_built_) return std::numeric_limits<float>::max();
        return sampleSdf(p);
    }

    // CostEvaluator interface: distance from p to nearest obstacle surface
    float queryDistance(const Eigen::Vector3f& p) const override {
        return querySdf(p);
    }

    bool isSdfBuilt() const { return sdf_built_; }

private:
    std::shared_ptr<RobotInterface> robot_;
    std::vector<CylinderObstacle>   cylinders_;
    std::vector<RobotSphere>        spheres_;
    // Pre-allocated per-sphere Jacobian buffers; sized in addRobotSphere, reused every gradient call
    mutable std::vector<Eigen::MatrixXf> sphere_jac_cache_;

    float           voxel_size_;
    Eigen::Vector3f bbox_min_, bbox_max_;
    int             nx_, ny_, nz_;
    std::vector<float> sdf_grid_;
    bool            sdf_built_;

    // ------------------------------------------------------------------
    // 几何辅助：点到圆柱的有符号距离（在圆柱内为负）
    // ------------------------------------------------------------------
    static float distToCylinder(const Eigen::Vector3f& p,
                                 const CylinderObstacle& cyl) {
        const Eigen::Vector3f v  = cyl.p1 - cyl.p0;
        const Eigen::Vector3f w  = p - cyl.p0;
        const float v_sq = v.squaredNorm();

        float t = (v_sq > 1e-6f)
                  ? std::max(0.0f, std::min(1.0f, w.dot(v) / v_sq))
                  : 0.0f;

        const Eigen::Vector3f closest = cyl.p0 + t * v;
        return (p - closest).norm() - cyl.radius;  // 负值 = 在圆柱内部
    }

    // ------------------------------------------------------------------
    // 网格辅助
    // ------------------------------------------------------------------

    // 体素中心的世界坐标
    inline Eigen::Vector3f voxelCenter(int ix, int iy, int iz) const {
        return bbox_min_ + Eigen::Vector3f(ix * voxel_size_,
                                           iy * voxel_size_,
                                           iz * voxel_size_);
    }

    // 平铺一维索引
    inline int linearIdx(int ix, int iy, int iz) const {
        return ix + nx_ * (iy + ny_ * iz);
    }

    // ------------------------------------------------------------------
    // 三线性插值 SDF 查询
    // 对网格内的点给出连续平滑的距离值；越界时夹取最近边界体素。
    // ------------------------------------------------------------------
    float sampleSdf(const Eigen::Vector3f& p) const {
        // 映射到连续网格坐标
        const float gx = (p.x() - bbox_min_.x()) / voxel_size_;
        const float gy = (p.y() - bbox_min_.y()) / voxel_size_;
        const float gz = (p.z() - bbox_min_.z()) / voxel_size_;

        // 越界：直接夹取最近体素（不插值，节省一次 if-else 分支）
        const int x0 = std::max(0, std::min(nx_ - 1, (int)gx));
        const int y0 = std::max(0, std::min(ny_ - 1, (int)gy));
        const int z0 = std::max(0, std::min(nz_ - 1, (int)gz));

        // 检查是否真的在可插值范围内
        if (gx < 0.0f || gx >= nx_ - 1 ||
            gy < 0.0f || gy >= ny_ - 1 ||
            gz < 0.0f || gz >= nz_ - 1) {
            return sdf_grid_[linearIdx(x0, y0, z0)];
        }

        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const int z1 = z0 + 1;

        const float fx = gx - x0;
        const float fy = gy - y0;
        const float fz = gz - z0;

        // 8 个角点
        const float c000 = sdf_grid_[linearIdx(x0, y0, z0)];
        const float c100 = sdf_grid_[linearIdx(x1, y0, z0)];
        const float c010 = sdf_grid_[linearIdx(x0, y1, z0)];
        const float c110 = sdf_grid_[linearIdx(x1, y1, z0)];
        const float c001 = sdf_grid_[linearIdx(x0, y0, z1)];
        const float c101 = sdf_grid_[linearIdx(x1, y0, z1)];
        const float c011 = sdf_grid_[linearIdx(x0, y1, z1)];
        const float c111 = sdf_grid_[linearIdx(x1, y1, z1)];

        // 沿 X 插值
        const float c00 = c000 * (1 - fx) + c100 * fx;
        const float c01 = c001 * (1 - fx) + c101 * fx;
        const float c10 = c010 * (1 - fx) + c110 * fx;
        const float c11 = c011 * (1 - fx) + c111 * fx;

        // 沿 Y 插值
        const float c0 = c00 * (1 - fy) + c10 * fy;
        const float c1 = c01 * (1 - fy) + c11 * fy;

        // 沿 Z 插值
        return c0 * (1 - fz) + c1 * fz;
    }
};

} // namespace grasp_rrt
