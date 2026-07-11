#pragma once

#include "cspace.h"
#include <cmath>
#include <limits>
#include <random>

namespace grasp_rrt {

// ============================================================================
// RepulsiveFieldEvaluator — semi-rigid obstacles as a workspace force field
//
// Historical note: this interface was originally designed as a scalar cost
// function CostEvaluator for an RRT*-style approach (path-cost accumulation +
// rewiring). That direction was abandoned due to the sub-second compute budget
// of (R3) — RRT* rewiring did not converge within 1 s on this problem. The
// surviving role of the interface is purely as a force-field source: it owns
// the SDF and exposes a queryDistance() accessor used by the planner's vector
// field. All force computation lives in derived classes (see SdfCostEvaluator
// :: evaluateAnisotropicForce). The legacy class name "CostEvaluator" is
// retained only to avoid breaking downstream code that holds the type name.
// ============================================================================
class CostEvaluator {
public:
    virtual ~CostEvaluator() = default;

    // Signed distance from a workspace point to the nearest obstacle surface.
    // Used by the planner's VF to query branch / trunk proximity.
    // Default returns +inf for evaluators that carry no geometry.
    virtual float queryDistance(const Eigen::Vector3f& /*p*/) const {
        return std::numeric_limits<float>::max();
    }

    // ---- Anisotropic edge rejection (pre-contact directional filter) ----
    //
    // For an RRT extension (q_near → q_new), returns a score in [0, 1] that
    // characterizes how axially aligned the engagement-zone sphere motions
    // are with nearby branches inside `envelope`:
    //
    //   score = 0  → no engagement-zone sphere in envelope, OR all checked
    //                sphere/branch pairs move perpendicular to branch axes
    //   score = 1  → at least one (sphere, branch) pair has the sphere
    //                moving exactly along the branch axis at full envelope weight
    //
    // Multi-branch handling: for each engagement sphere, the K nearest
    // non-rigid (semi-rigid foliage) branches within envelope are considered;
    // the final score is the max across all (sphere, branch) pairs.
    //
    // The caller (CSpaceSampled) decides what to do with the score — hard
    // cutoff, soft probabilistic, or anything else.
    //
    // Default returns 0 (never reject) for evaluators without an SDF /
    // branch-axis representation. SdfCostEvaluator overrides this.
    virtual float computeEdgeAxialScore(
        const Eigen::VectorXf& /*q_near*/,
        const Eigen::VectorXf& /*q_new*/,
        float /*envelope*/,
        int /*K*/ = 3) const {
        return 0.0f;
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
          sampling_size_dcd_(0.04f) {}

    // ---- Repulsive-field evaluator (holds the SDF; see paper Sec. 3) ----
    void setCostEvaluator(std::shared_ptr<CostEvaluator> evaluator) {
        cost_evaluator_ = evaluator;
    }

    std::shared_ptr<CostEvaluator> getCostEvaluator() const {
        return cost_evaluator_;
    }

    // Edge cost = pure C-space distance.
    //
    // The scalar-penalty integral that used to be summed in here (for an
    // RRT*-style rewiring) was removed when we abandoned RRT* for VF-RRT
    // (paper Sec. 3.2). Semi-rigid obstacles now influence the planner only
    // through the workspace force field of Sec. 3.3, not through edge cost.
    // The node-level cost_from_root field thus reduces to "C-space path
    // length from root", retained only as a tie-breaker between candidate
    // grasps at termination.
    float evaluateEdgeCost(const Config& q1, const Config& q2) {
        return calcDist(q1, q2);
    }

    // ---- 步长配置 ----

    void setSamplingSize(float s) { sampling_size_paths_ = s; }
    void setSamplingSizeDCD(float s) { sampling_size_dcd_ = s; }
    float getSamplingSize() const { return sampling_size_paths_; }
    float getSamplingSizeDCD() const { return sampling_size_dcd_; }

    // ---- Edge-rejection configuration (anisotropic directional filter) ----
    //
    // When enabled, isPathValid runs a soft probabilistic directional filter
    // on each candidate extension. The CostEvaluator computes an "axial
    // alignment score" ∈ [0, 1] for the edge (engagement-zone sphere motion
    // vs nearest-branch axis within `envelope`), and we accept/reject by:
    //
    //   p_reject = sigmoid(gain · (score − threshold))
    //
    // - threshold = sigmoid midpoint. score = threshold → 50/50 accept/reject.
    //   Default cos(25°) ≈ 0.9063 → motions within 25° of the axis sit at
    //   p_reject = 0.5; motions ≥ 30° off-axis are mostly accepted.
    // - gain     = steepness. Higher gain → sharper boundary (closer to hard
    //   cutoff). gain = 20 gives a moderately smooth transition.
    // - envelope = pre-contact shell radius (default 5 cm).
    //
    // The soft sigmoid eliminates the "thrashing" near-boundary thrashing
    // that a hard cutoff would produce.
    void setEdgeRejectionEnabled(bool enabled)  { edge_rejection_enabled_ = enabled; }
    void setEdgeRejectionEnvelope(float e)      { edge_rejection_envelope_ = e; }
    void setEdgeRejectionThreshold(float t)     { edge_rejection_threshold_ = t; }
    void setEdgeRejectionGain(float g)          { edge_rejection_gain_ = g; }
    void setEdgeRejectionK(int k)               { edge_rejection_k_ = std::max(1, k); }
    bool  getEdgeRejectionEnabled() const       { return edge_rejection_enabled_; }
    float getEdgeRejectionEnvelope() const      { return edge_rejection_envelope_; }
    float getEdgeRejectionThreshold() const     { return edge_rejection_threshold_; }
    float getEdgeRejectionGain() const          { return edge_rejection_gain_; }
    int   getEdgeRejectionK() const             { return edge_rejection_k_; }

    // ---- 离散碰撞检测 (覆盖基类) ----

    /**
     * 对应 Python CSpaceSampled.isPathValid
     * 递归中点检测: 先检查中点, 若有效则递归检查两半
     *
     * SPEED OPTIMIZATION: when edge rejection is enabled, we run the cheap
     * directional check (~15 μs) BEFORE the expensive geometric collision
     * check (~300-700 μs). Edges rejected by the directional filter never
     * trigger the recursive midpoint check, saving substantial work in
     * scenes with high directional-rejection rates.
     *
     * Soft probabilistic filter:
     *     score = cost_evaluator_->computeEdgeAxialScore(q1, q2, envelope)
     *     p_reject = sigmoid(gain · (score − threshold))
     *     reject if uniform(0,1) < p_reject
     */
    bool isPathValid(const Config& q1, const Config& q2) override {
        // Step 1 (cheap, ~15 μs): directional check, if enabled.
        if (edge_rejection_enabled_ && cost_evaluator_) {
            float score = cost_evaluator_->computeEdgeAxialScore(
                q1, q2, edge_rejection_envelope_, edge_rejection_k_);
            if (score > 0.0f) {
                float x = edge_rejection_gain_
                          * (score - edge_rejection_threshold_);
                // Numerically stable sigmoid
                float p_reject = (x >= 0.0f)
                    ? 1.0f / (1.0f + std::exp(-x))
                    : std::exp(x) / (1.0f + std::exp(x));
                std::uniform_real_distribution<float> u(0.0f, 1.0f);
                if (u(edge_rng_) < p_reject) return false;
            }
        }
        // Step 2 (expensive, ~300-700 μs): geometric collision check.
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

    // Anisotropic edge-rejection (off by default to preserve baseline)
    // Defaults: 5 cm envelope, threshold cos(25°) ≈ 0.9063, sigmoid gain 20,
    //           K=5 nearest non-rigid branches considered per engagement sphere.
    bool  edge_rejection_enabled_   = false;
    float edge_rejection_envelope_  = 0.05f;
    float edge_rejection_threshold_ = 0.9063f;
    float edge_rejection_gain_      = 20.0f;
    int   edge_rejection_k_         = 3;

    // Private RNG for the probabilistic accept/reject decision. Seeded with
    // a fixed value so soft rejection is deterministic per CSpaceSampled
    // instance; if you want per-trial variability, reseed externally.
    mutable std::mt19937 edge_rng_{0xC0FFEEu};

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
