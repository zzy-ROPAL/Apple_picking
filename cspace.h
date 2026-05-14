#pragma once

#include "types.h"
#include "cspace_node.h"
#include "collision_checker.h"
#include <vector>
#include <memory>
#include <random>
#include <functional>

namespace grasp_rrt {

// ============================================================================
// CSpace — 对应 Python CSpace
// 配置空间基类: 管理关节边界、随机采样、碰撞检测
// ============================================================================

class CSpace {
public:
    /**
     * @param dimension 配置空间维度 (关节个数)
     * @param collision_checker 碰撞检测器 (可为 nullptr, 则默认全部无碰撞)
     * @param seed 随机种子 (0 = 系统随机)
     */
    CSpace(int dimension,
           std::shared_ptr<CollisionChecker> collision_checker = nullptr,
           unsigned int seed = 0)
        : dimension_(dimension),
          collision_checker_(collision_checker),
          boundary_min_(Eigen::VectorXf::Zero(dimension)),
          boundary_max_(Eigen::VectorXf::Ones(dimension)),
          metric_weights_(Eigen::VectorXf::Ones(dimension)),
          use_metric_weights_(false),
          performance_collision_checks_(0),
          performance_distance_checks_(0),
          rng_(seed > 0 ? seed : std::random_device{}()) {}

    virtual ~CSpace() = default;

    // ---- 维度 ----
    int getDimension() const { return dimension_; }

    // ---- 边界管理 (对应 Python setBoundaries / getBoundaryMin 等) ----

    void setBoundaries(const Eigen::VectorXf& bmin, const Eigen::VectorXf& bmax) {
        boundary_min_ = bmin;
        boundary_max_ = bmax;
    }

    float getBoundaryMin(int d) const { return boundary_min_(d); }
    float getBoundaryMax(int d) const { return boundary_max_(d); }
    float getBoundaryDist(int d) const { return boundary_max_(d) - boundary_min_(d); }
    const Eigen::VectorXf& getBoundaryMinVec() const { return boundary_min_; }
    const Eigen::VectorXf& getBoundaryMaxVec() const { return boundary_max_; }

    // ---- 随机采样 (对应 Python getRandomConfig) ----

    Config getRandomConfig() {
        Config config(dimension_);
        for (int i = 0; i < dimension_; ++i) {
            std::uniform_real_distribution<float> dist(boundary_min_(i), boundary_max_(i));
            config(i) = dist(rng_);
        }
        return config;
    }

    /**
     * 获取有效的随机配置 (无碰撞 + 满足约束)
     */
    Config getRandomConfigValid(int max_tries = 1000) {
        for (int i = 0; i < max_tries; ++i) {
            Config c = getRandomConfig();
            if (isConfigValid(c)) return c;
        }
        return getRandomConfig(); // fallback
    }

    /**
     * 将配置裁剪到边界内 (对应 Python respectBoundaries)
     */
    Config respectBoundaries(const Config& config) const {
        return config.cwiseMax(boundary_min_).cwiseMin(boundary_max_);
    }

    // ---- 边界检查 ----

    bool isInBoundary(const Config& config) const {
        for (int i = 0; i < dimension_; ++i) {
            if (config(i) < boundary_min_(i) || config(i) > boundary_max_(i))
                return false;
        }
        return true;
    }

    // ---- 碰撞检测 (对应 Python isCollisionFree) ----

    bool isCollisionFree(const Config& config) {
        performance_collision_checks_++;
        if (collision_checker_)
            return collision_checker_->isCollisionFree(config);
        return true; // 无碰撞检测器时默认通过
    }

    // ---- 约束检查 ----

    void addConstraint(std::function<bool(const Config&)> constraint) {
        constraints_.push_back(constraint);
    }

    bool isSatisfyingConstraints(const Config& config) const {
        for (const auto& c : constraints_) {
            if (!c(config)) return false;
        }
        return true;
    }

    // ---- 综合有效性检查 (对应 Python isConfigValid) ----

    bool isConfigValid(const Config& config,
                       bool check_borders = true,
                       bool check_collisions = true,
                       bool check_constraints = true) {
        if (check_borders && !isInBoundary(config)) return false;
        if (check_collisions && !isCollisionFree(config)) return false;
        if (check_constraints && !isSatisfyingConstraints(config)) return false;
        return true;
    }

    // ---- 路径有效性 (基类只检查终点; CSpaceSampled 覆盖做 DCD) ----

    virtual bool isPathValid(const Config& q1, const Config& q2) {
        return isConfigValid(q2);
    }

    // ---- 距离计算 (对应 Python calcDist) ----

    float calcDist(const Config& c1, const Config& c2,
                   bool force_disable_weights = false) const {
        Eigen::VectorXf diff = c2 - c1;
        if (use_metric_weights_ && !force_disable_weights) {
            diff = diff.cwiseProduct(metric_weights_);
        }
        return diff.norm();
    }

    // ---- 插值 (对应 Python interpolate) ----

    Config interpolate(const Config& q1, const Config& q2, float t) const {
        return q1 + (q2 - q1) * t;
    }

    // ---- 权重 ----

    void setMetricWeights(const Eigen::VectorXf& w) {
        metric_weights_ = w;
        use_metric_weights_ = true;
    }

    void enableWeights(bool enable) { use_metric_weights_ = enable; }
    bool getUseMetricWeights() const { return use_metric_weights_; }
    const Eigen::VectorXf& getMetricWeights() const { return metric_weights_; }

    // ---- 性能统计 ----

    int getCollisionCheckCount() const { return performance_collision_checks_; }
    void resetPerformance() {
        performance_collision_checks_ = 0;
        performance_distance_checks_ = 0;
    }

    // ---- 碰撞检测器访问 ----
    std::shared_ptr<CollisionChecker> getCollisionChecker() const {
        return collision_checker_;
    }

protected:
    int dimension_;
    std::shared_ptr<CollisionChecker> collision_checker_;
    Eigen::VectorXf boundary_min_;
    Eigen::VectorXf boundary_max_;
    Eigen::VectorXf metric_weights_;
    bool use_metric_weights_;
    int performance_collision_checks_;
    int performance_distance_checks_;
    std::mt19937 rng_;
    std::vector<std::function<bool(const Config&)>> constraints_;
};

} // namespace grasp_rrt
