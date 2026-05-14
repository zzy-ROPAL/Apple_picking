#pragma once

#include "cspace.h"
#include <cmath>
#include <limits>

namespace grasp_rrt {

// ============================================================================
// CostEvaluator — 评估软代价 (如各向异性树冠穿透阻力)
// ============================================================================
class CostEvaluator {
public:
    virtual ~CostEvaluator() = default;
    virtual float evaluateCost(const Config& q) = 0;

    // 返回梯度向量 (大小与 q 一致)
    virtual Config evaluateGradient(const Config& q) {
        return Eigen::VectorXf::Zero(q.size());
    }

    // 查询某 3D 点到最近障碍物表面的距离 (单位：米)
    // 默认返回一个极大值，表示"没有距离信息"。
    // 子类（如 SdfCostEvaluator）应当 override 它。
    virtual float queryDistance(const Eigen::Vector3f& /*p*/) const {
        return std::numeric_limits<float>::max();
    }
};

// ============================================================================
// CSpaceSampled — 对应 Python CSpaceSampled
// 在 CSpace 基础上增加:
//   - 离散碰撞检测 (DCD): isPathValid 沿路径递归检测中间点
//   - 路径创建: 生成中间路径点
//   - 边缘代价计算: 计算联合空间距离和软代价积分
// ============================================================================

class CSpaceSampled : public CSpace {
public:
    CSpaceSampled(int dimension,
                  std::shared_ptr<CollisionChecker> collision_checker = nullptr,
                  unsigned int seed = 0)
        : CSpace(dimension, collision_checker, seed),
          sampling_size_paths_(0.08f),
          sampling_size_dcd_(0.04f),
          cost_weight_soft_(100.0f) {}

    // ---- 软代价评估 ----
    void setCostEvaluator(std::shared_ptr<CostEvaluator> evaluator) {
        cost_evaluator_ = evaluator;
    }

    void setSoftCostWeight(float w) { cost_weight_soft_ = w; }
    
    std::shared_ptr<CostEvaluator> getCostEvaluator() const {
        return cost_evaluator_;
    }

    float evaluateEdgeCost(const Config& q1, const Config& q2) {
        float dist = calcDist(q1, q2);
        if (!cost_evaluator_) return dist;

        float soft_cost_integral = 0.0f;
        int steps = std::max(2, static_cast<int>(std::ceil(dist / sampling_size_dcd_)));
        for (int i = 1; i <= steps; ++i) {
            float t = static_cast<float>(i) / steps;
            Config inter = interpolate(q1, q2, t);
            soft_cost_integral += cost_evaluator_->evaluateCost(inter);
        }
        
        float ds = dist / steps;
        return dist + soft_cost_integral * ds * cost_weight_soft_;
    }

    // ---- 步长配置 ----

    void setSamplingSize(float s) { sampling_size_paths_ = s; }
    void setSamplingSizeDCD(float s) { sampling_size_dcd_ = s; }
    float getSamplingSize() const { return sampling_size_paths_; }
    float getSamplingSizeDCD() const { return sampling_size_dcd_; }

    // ---- 离散碰撞检测 (覆盖基类) ----

    /**
     * 对应 Python CSpaceSampled.isPathValid
     * 递归中点检测: 先检查中点, 若有效则递归检查两半
     */
    bool isPathValid(const Config& q1, const Config& q2) override {
        return recursivePathCheck(q1, q2, 0);
    }

    // ---- 路径创建 (对应 Python createPath) ----

    /**
     * 在 start 和 goal 之间生成中间路径点 (不做碰撞检测)
     */
    ConfigPath createPath(const Config& start, const Config& goal) const {
        ConfigPath path;
        path.push_back(start);

        float dist = calcDist(start, goal);
        if (dist <= 0) return path;

        int n_steps = std::max(1, static_cast<int>(std::ceil(dist / sampling_size_paths_)));
        for (int i = 1; i <= n_steps; ++i) {
            float t = static_cast<float>(i) / n_steps;
            path.push_back(interpolate(start, goal, t));
        }
        return path;
    }

    /**
     * 对应 Python createPathUntilInvalid
     * 沿路径添加中间点, 遇到碰撞时停止
     * @return {有效路径点列表, 有效长度比例 0~1}
     */
    std::pair<ConfigPath, float> createPathUntilInvalid(const Config& start,
                                                        const Config& goal) {
        ConfigPath path;
        path.push_back(start);

        float dist = calcDist(start, goal);
        if (dist <= 0) return {path, 1.0f};

        int n_steps = std::max(1, static_cast<int>(std::ceil(dist / sampling_size_paths_)));
        for (int i = 1; i <= n_steps; ++i) {
            float t = static_cast<float>(i) / n_steps;
            Config q = interpolate(start, goal, t);
            if (!isConfigValid(q)) {
                return {path, static_cast<float>(i - 1) / n_steps};
            }
            path.push_back(q);
        }
        return {path, 1.0f};
    }

private:
    float sampling_size_paths_;  // 路径采样步长
    float sampling_size_dcd_;    // DCD 碰撞检测步长
    const int RECURSION_MAX_DEPTH = 24; // 极大提高递归深度，防止极细微路径被漏检

    std::shared_ptr<CostEvaluator> cost_evaluator_;
    float cost_weight_soft_;

    /**
     * 递归路径碰撞检测
     */
    bool recursivePathCheck(const Config& q1, const Config& q2, int depth) {
        float dist = calcDist(q1, q2, /*force_disable_weights=*/true);
        
        // 基本情况 1: 步长已经足够小
        if (dist <= sampling_size_dcd_) {
            return isConfigValid(q2); 
        }
        
        // 基本情况 2: 达到最大深度 (此时应强制检测终点)
        if (depth >= RECURSION_MAX_DEPTH) {
            return isConfigValid(q2);
        }

        Config q_mid = interpolate(q1, q2, 0.5f);
        if (!isConfigValid(q_mid)) return false;

        return recursivePathCheck(q1, q_mid, depth + 1) &&
               recursivePathCheck(q_mid, q2, depth + 1);
    }
};

} // namespace grasp_rrt
