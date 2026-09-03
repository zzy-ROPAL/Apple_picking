#pragma once

#include "types.h"
#include "cspace_node.h"
#include "collision_checker.h"
#include <vector>
#include <memory>
#include <random>
#include <atomic>
#include "rng_util.h"
#include <functional>

namespace grasp_rrt {

// ============================================================================
// CSpace — counterpart of the Python CSpace
// Configuration-space base class: joint limits, random sampling, collision checking
// ============================================================================

class CSpace {
public:
    /**
     * @param dimension configuration-space dimension (number of joints)
     * @param collision_checker collision checker (may be nullptr — then everything is collision-free)
     * @param seed random seed (0 = seed from the system)
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

    // ---- Dimension ----
    int getDimension() const { return dimension_; }

    // ---- Boundary management (Python setBoundaries / getBoundaryMin etc.) ----

    void setBoundaries(const Eigen::VectorXf& bmin, const Eigen::VectorXf& bmax) {
        boundary_min_ = bmin;
        boundary_max_ = bmax;
    }

    float getBoundaryMin(int d) const { return boundary_min_(d); }
    float getBoundaryMax(int d) const { return boundary_max_(d); }
    float getBoundaryDist(int d) const { return boundary_max_(d) - boundary_min_(d); }
    const Eigen::VectorXf& getBoundaryMinVec() const { return boundary_min_; }
    const Eigen::VectorXf& getBoundaryMaxVec() const { return boundary_max_; }

    // ---- Random sampling (Python getRandomConfig) ----

    Config getRandomConfig() {
        Config config(dimension_);
        auto& rng = threadRng();
        for (int i = 0; i < dimension_; ++i) {
            std::uniform_real_distribution<float> dist(boundary_min_(i), boundary_max_(i));
            config(i) = dist(rng);
        }
        return config;
    }

    /**
     * Get a valid random configuration (collision-free + constraints satisfied)
     */
    Config getRandomConfigValid(int max_tries = 1000) {
        for (int i = 0; i < max_tries; ++i) {
            Config c = getRandomConfig();
            if (isConfigValid(c)) return c;
        }
        return getRandomConfig(); // fallback
    }

    /**
     * Clamp a configuration into the boundaries (Python respectBoundaries)
     */
    Config respectBoundaries(const Config& config) const {
        return config.cwiseMax(boundary_min_).cwiseMin(boundary_max_);
    }

    // ---- Boundary check ----

    bool isInBoundary(const Config& config) const {
        for (int i = 0; i < dimension_; ++i) {
            if (config(i) < boundary_min_(i) || config(i) > boundary_max_(i))
                return false;
        }
        return true;
    }

    // ---- Collision checking (Python isCollisionFree) ----

    bool isCollisionFree(const Config& config) {
        performance_collision_checks_++;
        if (collision_checker_)
            return collision_checker_->isCollisionFree(config);
        return true; // no collision checker: pass by default
    }

    // ---- Constraint check ----

    void addConstraint(std::function<bool(const Config&)> constraint) {
        constraints_.push_back(constraint);
    }

    bool isSatisfyingConstraints(const Config& config) const {
        for (const auto& c : constraints_) {
            if (!c(config)) return false;
        }
        return true;
    }

    // ---- Combined validity check (Python isConfigValid) ----
    // virtual: CSpaceSampled overrides it to append the soft-branch penetration gate.

    virtual bool isConfigValid(const Config& config,
                       bool check_borders = true,
                       bool check_collisions = true,
                       bool check_constraints = true) {
        if (check_borders && !isInBoundary(config)) return false;
        if (check_collisions && !isCollisionFree(config)) return false;
        if (check_constraints && !isSatisfyingConstraints(config)) return false;
        return true;
    }

    // ---- Path validity (base class checks endpoints only; CSpaceSampled overrides with DCD) ----

    virtual bool isPathValid(const Config& q1, const Config& q2) {
        return isConfigValid(q2);
    }

    // ---- Distance computation (Python calcDist) ----

    float calcDist(const Config& c1, const Config& c2,
                   bool force_disable_weights = false) const {
        Eigen::VectorXf diff = c2 - c1;
        if (use_metric_weights_ && !force_disable_weights) {
            diff = diff.cwiseProduct(metric_weights_);
        }
        return diff.norm();
    }

    // ---- Interpolation (Python interpolate) ----

    Config interpolate(const Config& q1, const Config& q2, float t) const {
        return q1 + (q2 - q1) * t;
    }

    // ---- Weights ----

    void setMetricWeights(const Eigen::VectorXf& w) {
        metric_weights_ = w;
        use_metric_weights_ = true;
    }

    void enableWeights(bool enable) { use_metric_weights_ = enable; }
    bool getUseMetricWeights() const { return use_metric_weights_; }
    const Eigen::VectorXf& getMetricWeights() const { return metric_weights_; }

    // ---- Performance statistics ----

    int getCollisionCheckCount() const { return performance_collision_checks_; }
    void resetPerformance() {
        performance_collision_checks_ = 0;
        performance_distance_checks_ = 0;
    }

    // ---- Collision-checker access ----
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
    std::atomic<int> performance_collision_checks_;
    std::atomic<int> performance_distance_checks_;
    std::mt19937 rng_;  // kept for single-threaded use (e.g. serial doPlanningCycle); parallel sampling uses threadRng()
    std::vector<std::function<bool(const Config&)>> constraints_;
};

} // namespace grasp_rrt
