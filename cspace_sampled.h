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

    // ---- Anisotropic edge refuse (pre-contact directional filter) ----
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

    // ---- Tensor variant of the edge-refuse score ----
    //
    // Same contract as computeEdgeAxialScore (score in [0, 1], 0 = never
    // reject), but the branch aggregation is a smooth structure-tensor
    // quadratic form instead of a discrete MAX over (sphere, branch) pairs:
    //
    //   M(p)     = sum_i w_i(p) * a_i * a_i^T     (all non-rigid branches)
    //   score(d) = gate * sqrt( d^T M d / trace(M) )
    //
    // No branch is ever *selected*, so the score cannot jump when the
    // dominant branch changes — the argmax discontinuity (the oscillation
    // source) is removed at the root rather than merely damped by hysteresis.
    // Reduces to the single-branch score w * |d . a| when one branch
    // dominates. Default returns 0 for evaluators without branch axes.
    virtual float computeEdgeAxialScoreTensor(
        const Eigen::VectorXf& /*q_near*/,
        const Eigen::VectorXf& /*q_new*/,
        float /*envelope*/) const {
        return 0.0f;
    }

    // ---- Soft-branch penetration gate (hard validity constraint) ----
    //
    // Decide whether, at configuration q, every body-probe penetration into a soft branch
    //   pen = (r_branch + r_body) - dist_axis  <=  allow(t) = root + (tip-root)*t
    // stays within its allowance. t = arc-length fraction along the branch (0 root,
    // 1 tip): near the root almost nothing may be squeezed (it snaps), near the tip
    // the branch may be pushed aside. Same criterion as the benchmark's cut_violations
    // metric, but in the validity layer. Contacts inside the apple exemption sphere are ignored.
    //
    // true = passed (or gate disabled). Always true by default; SdfCostEvaluator overrides.
    virtual bool isPenetrationAllowed(const Eigen::VectorXf& /*q*/) const {
        return true;
    }

    // ---- Edge-level severity for unified refuse ----
    // Maximum severity over all (probe, soft branch) pairs along the whole edge q1->q2
    // (inward pressure / depth budget + G * axial slip / slip budget). >1 = over budget.
    // 0 = disabled / no contact neighborhood. CSpaceSampled rejects with probability
    // sigmoid(gain*(score-threshold)), threshold = 1 (50/50 exactly at full budget).
    virtual float computeEdgeUnifiedSeverity(
        const Eigen::VectorXf& /*q1*/,
        const Eigen::VectorXf& /*q2*/) const {
        return 0.0f;
    }
};

// ============================================================================
// CSpaceSampled — counterpart of the Python CSpaceSampled
// Adds on top of CSpace:
//   - discrete collision detection (DCD): isPathValid recursively checks midpoints
//   - path creation: generates intermediate waypoints
//   - edge cost computation: joint-space distance plus the soft-cost integral
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

    // ---- Step-size configuration ----

    void setSamplingSize(float s) { sampling_size_paths_ = s; }
    void setSamplingSizeDCD(float s) { sampling_size_dcd_ = s; }
    float getSamplingSize() const { return sampling_size_paths_; }
    float getSamplingSizeDCD() const { return sampling_size_dcd_; }

    // ---- Edge-refuse configuration (anisotropic directional filter) ----
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
    void setEdgeRefuseEnabled(bool enabled)  { edge_refuse_enabled_ = enabled; }
    void setEdgeRefuseEnvelope(float e)      { edge_refuse_envelope_ = e; }
    void setEdgeRefuseThreshold(float t)     { edge_refuse_threshold_ = t; }
    void setEdgeRefuseGain(float g)          { edge_refuse_gain_ = g; }
    void setEdgeRefuseK(int k)               { edge_refuse_k_ = std::max(1, k); }
    // Tensor mode: replace the discrete MAX branch aggregation with the
    // smooth structure-tensor quadratic form (see CostEvaluator::
    // computeEdgeAxialScoreTensor). Same sigmoid, same threshold/gain —
    // only the score function changes.
    void setEdgeRefuseTensorMode(bool on)    { edge_refuse_tensor_mode_ = on; }
    // Unified mode: score = computeEdgeUnifiedSeverity; threshold now means
    // "fraction of severity budget used" (default 1.0), the rest of the sigmoid pipeline is reused.
    void setEdgeRefuseUnifiedMode(bool on)   { edge_refuse_unified_mode_ = on; }
    bool  getEdgeRefuseEnabled() const       { return edge_refuse_enabled_; }
    float getEdgeRefuseEnvelope() const      { return edge_refuse_envelope_; }
    float getEdgeRefuseThreshold() const     { return edge_refuse_threshold_; }
    float getEdgeRefuseGain() const          { return edge_refuse_gain_; }
    int   getEdgeRefuseK() const             { return edge_refuse_k_; }
    bool  getEdgeRefuseTensorMode() const    { return edge_refuse_tensor_mode_; }

    // ---- Configuration validity (overrides the base class: appends the soft-branch penetration gate) ----
    //
    // The gate runs before Bullet's narrow phase (~10 us of FK probes vs ~300-700 us
    // of geometry checks), the same cheap-check-first pattern as edge refuse. Hanging
    // it on isConfigValid means every DCD sample, the descent phase (moveTowardsGoal
    // goes through the same isPathValid) and random start generation all inherit it
    bool isConfigValid(const Config& config,
                       bool check_borders = true,
                       bool check_collisions = true,
                       bool check_constraints = true) override {
        if (check_borders && !isInBoundary(config)) return false;
        if (check_collisions && cost_evaluator_) {
            pen_gate_checks_++;
            if (!cost_evaluator_->isPenetrationAllowed(config)) {
                pen_gate_rejects_++;
                return false;
            }
        }
        return CSpace::isConfigValid(config, /*check_borders=*/false,
                                     check_collisions, check_constraints);
    }

    // ---- Penetration-gate telemetry (cumulative; caller diffs across runs) ----
    unsigned long long getPenGateChecks()  const { return pen_gate_checks_; }
    unsigned long long getPenGateRejects() const { return pen_gate_rejects_; }

    // ---- Discrete collision detection (overrides the base class) ----

    /**
     * Counterpart of the Python CSpaceSampled.isPathValid
     * Recursive midpoint check: test the midpoint first, then recurse into both halves if valid
     *
     * SPEED OPTIMIZATION: when edge refuse is enabled, we run the cheap
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
        if (edge_refuse_enabled_ && cost_evaluator_) {
            float score = edge_refuse_unified_mode_
                ? cost_evaluator_->computeEdgeUnifiedSeverity(q1, q2)
                : (edge_refuse_tensor_mode_
                    ? cost_evaluator_->computeEdgeAxialScoreTensor(
                          q1, q2, edge_refuse_envelope_)
                    : cost_evaluator_->computeEdgeAxialScore(
                          q1, q2, edge_refuse_envelope_, edge_refuse_k_));
            er_checks_++;
            er_score_sum_ += score;
            if (score > 0.85f) er_hi_++;
            if (score > 0.0f) {
                float x = edge_refuse_gain_
                          * (score - edge_refuse_threshold_);
                // Numerically stable sigmoid
                float p_reject = (x >= 0.0f)
                    ? 1.0f / (1.0f + std::exp(-x))
                    : std::exp(x) / (1.0f + std::exp(x));
                // thread_local: isPathValid is called concurrently by the OR-parallel race
                // and by parallel shortcutting; a shared member RNG would be a data race
                static thread_local std::mt19937 rng{std::random_device{}()};
                std::uniform_real_distribution<float> u(0.0f, 1.0f);
                if (u(rng) < p_reject) {
                    er_rejects_++;
                    return false;
                }
            }
        }
        // Step 2 (expensive, ~300-700 μs): geometric collision check.
        // DCD deduplication: endpoint q2 is checked once here and each midpoint once during
        // recursion (q1 is guaranteed valid by the caller: a tree node or the previous
        // waypoint). The old recursion re-checked every midpoint as a sub-segment endpoint
        if (!isConfigValid(q2)) return false;
        return recursivePathCheck(q1, q2, 0);
    }

    // ---- Edge-refuse telemetry (cumulative; caller diffs across runs) ----
    unsigned long long getEdgeRefuseChecks()  const { return er_checks_; }
    unsigned long long getEdgeRefuseRejects() const { return er_rejects_; }
    unsigned long long getEdgeRefuseHiScores() const { return er_hi_; }
    double getEdgeRefuseScoreSum() const { return er_score_sum_; }

    // ---- Path creation (counterpart of the Python createPath) ----

    /**
     * Generate intermediate waypoints between start and goal (no collision checking)
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
     * Counterpart of the Python createPathUntilInvalid
     * Append intermediate points along the path, stopping at the first collision
     * @return {list of valid waypoints, valid length fraction 0~1}
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
    float sampling_size_paths_;  // path sampling step
    float sampling_size_dcd_;    // DCD collision-checking step
    const int RECURSION_MAX_DEPTH = 24; // deliberately deep, so that very thin path features are not missed

    std::shared_ptr<CostEvaluator> cost_evaluator_;

    // Anisotropic edge-refuse (off by default to preserve baseline)
    // Defaults: 5 cm envelope, threshold cos(25°) ≈ 0.9063, sigmoid gain 20,
    //           K=5 nearest non-rigid branches considered per engagement sphere.
    bool  edge_refuse_enabled_   = false;
    float edge_refuse_envelope_  = 0.05f;
    float edge_refuse_threshold_ = 0.9063f;
    float edge_refuse_gain_      = 20.0f;
    int   edge_refuse_k_         = 3;
    // false = legacy MAX aggregation; true = structure-tensor quadratic form
    bool  edge_refuse_tensor_mode_ = false;
    // true = unified severity score (unified rule), takes precedence over tensor/MAX
    bool  edge_refuse_unified_mode_ = false;

    // Edge-refuse telemetry
    mutable unsigned long long er_checks_  = 0;
    mutable unsigned long long er_rejects_ = 0;
    mutable unsigned long long er_hi_      = 0;
    mutable double er_score_sum_ = 0.0;

    // Penetration-gate telemetry (rejects stays 0 while the gate is disabled; same
    // convention as the er_ counters: non-atomic races under parallel batch are fine, telemetry only)
    mutable unsigned long long pen_gate_checks_  = 0;
    mutable unsigned long long pen_gate_rejects_ = 0;

    // (the probabilistic-rejection RNG moved to a thread_local inside isPathValid —
    //  a shared member RNG would be a data race under concurrent calls; seeding still follows the random_device convention)

    /**
     * Recursive path collision check
     * Precondition: q1 and q2 have both already been validated (isPathValid handles q2, the caller guarantees q1).
     * Each midpoint is checked exactly once where it is generated; the base case needs
     * no endpoint re-check — this removes the ~50% redundant collision checks the old
     */
    bool recursivePathCheck(const Config& q1, const Config& q2, int depth) {
        float dist = calcDist(q1, q2, /*force_disable_weights=*/true);

        // Base case: step small enough or max depth reached; both endpoints already checked
        if (dist <= sampling_size_dcd_) return true;
        if (depth >= RECURSION_MAX_DEPTH) return true;

        Config q_mid = interpolate(q1, q2, 0.5f);
        if (!isConfigValid(q_mid)) return false;

        return recursivePathCheck(q1, q_mid, depth + 1) &&
               recursivePathCheck(q_mid, q2, depth + 1);
    }
};

} // namespace grasp_rrt
