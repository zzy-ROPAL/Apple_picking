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
#include <queue>

namespace grasp_rrt {

// ============================================================================
// SdfCostEvaluator — owner of the precomputed signed-distance field (SDF)
// over the semi-rigid foliage and rigid trunk/canopy.
//
// Role in the paper:
//   - Implements the workspace force field of Sec. 3.3 via
//     evaluateAnisotropicForce(q, v_joint, K_ax), which returns the joint-
//     space force (radial repulsion + non-conservative axial drag, both
//     gated by per-sphere penetration > 0).
//   - Exposes queryDistance(p) for the planner's adaptive step size.
//   - Provides measureWaypointAnisotropy() as a post-hoc evaluation metric.
//
// Note: the legacy scalar-penalty methods evaluateCost() and evaluateGradient()
// were removed when the planner moved from RRT* to VF-RRT (paper Sec. 3.2);
// they had no remaining functional role beyond the conservative component of
// the force field, which evaluateAnisotropicForce computes directly.
//
// Usage:
//   1. addBranchCylinder(p0, p1, radius)
//   2. addRobotSphere(link, radius, offset)
//   3. buildSdf()          — call once before planning, O(1) queries afterwards
//   4. cspace->setCostEvaluator(...)
// ============================================================================

class SdfCostEvaluator : public CostEvaluator {
public:
    struct CylinderObstacle {
        Eigen::Vector3f p0;
        Eigen::Vector3f p1;
        float radius;
        // If true, this cylinder is excluded from anisotropy mechanisms
        // (sample steering, axial drag K_ax, edge refuse). Use for rigid
        // obstacles like the trunk where "push aside radially" is not a
        // valid behavior — the robot must avoid them entirely.
        bool exclude_from_aniso = false;
        // Pre-normalized axis (p1 − p0)/|p1 − p0|. Cached at build time
        // (Optimization B) so per-cycle anisotropy checks skip the
        // normalize() call. Zero for degenerate cylinders (p0 == p1).
        Eigen::Vector3f axis_unit = Eigen::Vector3f::Zero();
        // Arc-length fraction interval [t0, t1] of this segment along its branch (0 = root, 1 = tip).
        // Used by the branch-stiffness model (the root is "stiffer"). A single-segment branch defaults to 0..1.
        float t0 = 0.0f;
        float t1 = 1.0f;
    };

    struct RobotSphere {
        std::string link_name;
        float radius;
        Eigen::Vector3f offset;
        int frame_id;
    };

    // voxel_size: voxel resolution (m); default 1 cm, a good accuracy/speed balance
    explicit SdfCostEvaluator(std::shared_ptr<RobotInterface> robot,
                              float voxel_size = 0.01f)
        : robot_(robot), voxel_size_(voxel_size),
          nx_(0), ny_(0), nz_(0), sdf_built_(false) {}

    // ---- Same construction interface as TreeCanopyCostEvaluator ----

    void addBranchCylinder(const Eigen::Vector3f& p0,
                           const Eigen::Vector3f& p1,
                           float radius,
                           float t0 = 0.0f, float t1 = 1.0f) {
        Eigen::Vector3f a = p1 - p0;
        float len = a.norm();
        Eigen::Vector3f a_unit = (len > 1e-6f)
            ? Eigen::Vector3f(a / len)
            : Eigen::Vector3f::Zero();
        cylinders_.push_back({p0, p1, radius, /*exclude_from_aniso=*/false,
                              a_unit, t0, t1});
        sdf_built_ = false;
    }

    // branch-stiffness model: stiffness coefficient at position t on a soft branch.
    //   t = 0 (root, stiffest cantilever) -> k_root;  t = 1 (tip, bends easily) -> k_tip.
    //   Physical motivation: a real branch is a cantilever beam, so displacing the root costs
    //   far more than the tip; this encodes "cutting through the root/middle snaps the branch" as field strength.
    static float branchStiffness(float t, float k_root, float k_tip) {
        t = std::max(0.0f, std::min(1.0f, t));
        return k_root + (k_tip - k_root) * t;
    }

    // Same as addBranchCylinder but flags the cylinder as RIGID — the
    // anisotropy mechanisms (sample steering, K_ax, edge refuse) will
    // skip it. Use for the trunk and any other rigid obstacle the robot
    // is expected to avoid rather than push through.
    void addRigidCylinder(const Eigen::Vector3f& p0,
                          const Eigen::Vector3f& p1,
                          float radius) {
        Eigen::Vector3f a = p1 - p0;
        float len = a.norm();
        Eigen::Vector3f a_unit = (len > 1e-6f)
            ? Eigen::Vector3f(a / len)
            : Eigen::Vector3f::Zero();
        cylinders_.push_back({p0, p1, radius, /*exclude_from_aniso=*/true, a_unit});
        sdf_built_ = false;
    }

    // Read-only access to all cylinder obstacles (rigid flag included). Used to export the scene
    // geometry for Isaac Sim alignment (scene_layout.json): a single exit point covering every scene variant.
    const std::vector<CylinderObstacle>& getCylinders() const { return cylinders_; }

    // ---- Soft-branch penetration gate ----
    //
    // Lifts the criterion of the benchmark's post-hoc cut_violations metric into the validity layer:
    //   pen = (r_branch + r_probe) - dist_axis <= allow(t)  (see penAllowAt for the ramp)
    // The probe set matches that metric (link3...hand, 9 volumetric probes). Contacts inside the
    // exemption sphere (exempt_center, exempt_radius) are ignored — they are inherent to the terminal
    // grasp of a branch-embedded apple. allow is in meters.
    void enablePenetrationGate(float allow_root, float allow_tip,
                               const Eigen::Vector3f& exempt_center,
                               float exempt_radius) {
        pen_gate_enabled_      = true;
        pen_gate_allow_root_   = allow_root;
        pen_gate_allow_tip_    = allow_tip;
        pen_gate_exempt_center_ = exempt_center;
        pen_gate_exempt_r2_    = exempt_radius * exempt_radius;
        if (pen_probes_.empty()) {
            struct P { const char* frame; Eigen::Vector3f off; float r; };
            const P defs[] = {
                {"panda_link3", {0, 0,     0.15f}, 0.07f},
                {"panda_link4", {0, 0,     0.05f}, 0.07f},
                {"panda_link5", {0, 0,     0.15f}, 0.06f},
                {"panda_link5", {0, 0,     0.30f}, 0.06f},
                {"panda_link6", {0, 0,     0    }, 0.06f},
                {"panda_link7", {0, 0,     0.05f}, 0.06f},
                {"panda_hand",  {0, 0,     0.08f}, 0.06f},
                {"panda_hand",  {0, 0.04f, 0.05f}, 0.06f},
                {"panda_hand",  {0,-0.04f, 0.05f}, 0.06f},
            };
            for (const auto& d : defs)
                pen_probes_.push_back({robot_->getFrameId(d.frame), d.off, d.r});
            pen_gate_hand_fid_ = robot_->getFrameId("panda_hand");
        }
    }

    // allow(t): the permitted-penetration ramp. The root plateau [0, ramp_t0] always takes the root
    // value and the ramp covers only [ramp_t0, 1] — i.e. the first quarter of the branch (by default)
    // is treated at root stiffness and the remaining three quarters relax linearly toward the tip.
    // This is the ONLY definition of allow(t): the gate, steering, severity and the nav toll all share it,
    // so the layers cannot drift apart.
    float penAllowAt(float t_arc) const {
        float u = (t_arc - pen_gate_ramp_t0_)
                / std::max(1e-3f, 1.0f - pen_gate_ramp_t0_);
        if (u < 0.0f) u = 0.0f;
        if (u > 1.0f) u = 1.0f;
        return pen_gate_allow_root_
             + (pen_gate_allow_tip_ - pen_gate_allow_root_) * u;
    }

    // Ramp start (width of the root plateau, arc-length fraction). 0 = old behavior (linear along the whole branch).
    void setPenGateRampT0(float t0) {
        pen_gate_ramp_t0_ = std::min(0.95f, std::max(0.0f, t0));
    }

    // Gate decision. Always true while disabled. FK goes through the per-thread robot state,
    // the same parallel-safe pattern as computeEdgeAxialScore.
    bool isPenetrationAllowed(const Eigen::VectorXf& q) const override {
        if (!pen_gate_enabled_) return true;
        robot_->setJointValues(q);
        for (const auto& pr : pen_probes_) {
            Eigen::Matrix4f pose = robot_->getFramePoseById(pr.fid);
            Eigen::Vector3f pp = pose.block<3,1>(0,3)
                               + pose.block<3,3>(0,0) * pr.off;
            if ((pp - pen_gate_exempt_center_).squaredNorm() < pen_gate_exempt_r2_)
                continue;   // exempt near the apple
            for (const auto& cyl : cylinders_) {
                if (cyl.exclude_from_aniso) continue;              // soft branches only
                if (cyl.axis_unit.squaredNorm() < 0.25f) continue; // skip degenerate
                Eigen::Vector3f seg = cyl.p1 - cyl.p0;
                float seg_sq = seg.squaredNorm();
                float u = (seg_sq > 1e-9f)
                    ? std::max(0.0f, std::min(1.0f,
                          (pp - cyl.p0).dot(seg) / seg_sq))
                    : 0.0f;
                float dist = (pp - (cyl.p0 + u * seg)).norm();
                float pen = (cyl.radius + pr.r) - dist;
                if (pen <= 0.0f) continue;
                float t_arc = cyl.t0 + u * (cyl.t1 - cyl.t0);
                float allow = penAllowAt(t_arc);
                // Hard forbidden core around the axis: a probe center never enters within axis_clear of
                // the axis — allow is capped at (R+r-clear), so even the tip forbids "sliding along the axis".
                // A deterministic positional rule (acceptance probability exactly 0); probabilistic filtering cannot give that guarantee.
                float allow_cap = (cyl.radius + pr.r)
                    - pen_gate_axis_clear_ * (1.0f - t_arc);
                if (allow > allow_cap) allow = allow_cap;
                if (pen > allow) return false;
            }
        }
        return true;
    }

    bool isPenetrationGateEnabled() const { return pen_gate_enabled_; }

    // ---- Depth steering (proposal-layer companion to the gate, see the note in types.h) ----
    //
    // Projects the C-space proposal direction onto the tangent of the permitted-penetration boundary.
    // For every (probe, soft branch) pair in the contact neighborhood (body surface within margin):
    //   g  = J^T * (-r_hat)   — the joint direction that deepens this probe fastest (unnormalized)
    //   v  = step * dir . g   — predicted deepening this step (m, first-order linearization)
    //   when v > kappa*slack, subtract the excess along g_hat from dir (slack = allow(t) - pen)
    // Constraints are projected sequentially (an approximation with multiple contacts). On return dir is
    // not normalized; the caller renormalizes. Returns whether a bend happened (telemetry).
    bool applyDepthSteering(const Eigen::VectorXf& q,
                            Eigen::VectorXf& q_step_dir,
                            float step, float safety, float margin) const {
        if (!pen_gate_enabled_ || pen_probes_.empty() || step <= 1e-6f)
            return false;
        pen_steer_calls_++;
        robot_->setJointValues(q);
        bool bent = false, jac_ready = false;
        for (const auto& pr : pen_probes_) {
            Eigen::Matrix4f pose = robot_->getFramePoseById(pr.fid);
            Eigen::Vector3f pp = pose.block<3,1>(0,3)
                               + pose.block<3,3>(0,0) * pr.off;
            if ((pp - pen_gate_exempt_center_).squaredNorm() < pen_gate_exempt_r2_)
                continue;
            for (const auto& cyl : cylinders_) {
                if (cyl.exclude_from_aniso) continue;
                if (cyl.axis_unit.squaredNorm() < 0.25f) continue;
                Eigen::Vector3f seg = cyl.p1 - cyl.p0;
                float seg_sq = seg.squaredNorm();
                float u = (seg_sq > 1e-9f)
                    ? std::max(0.0f, std::min(1.0f,
                          (pp - cyl.p0).dot(seg) / seg_sq))
                    : 0.0f;
                Eigen::Vector3f radial = pp - (cyl.p0 + u * seg);
                float dist = radial.norm();
                if (dist <= 1e-4f) continue;               // degenerate on the axis, handled by the gate
                float surf = cyl.radius + pr.r;            // the contact surface where pen = 0
                if (dist - surf > margin) continue;        // outside the contact neighborhood
                float pen = surf - dist;                   // may be negative (no contact)
                float t_arc = cyl.t0 + u * (cyl.t1 - cyl.t0);
                float allow = penAllowAt(t_arc);
                float slack = allow - std::max(pen, 0.0f);
                if (slack < 0.0f) slack = 0.0f;

                if (!jac_ready) { robot_->prepareJacobians(); jac_ready = true; }
                Eigen::MatrixXf J = robot_->computeSphereJacobianById(pr.fid, pr.off);
                if (J.rows() != 3 || J.cols() != q_step_dir.size()) continue;
                Eigen::VectorXf g = J.transpose() * (-(radial / dist));
                float gn = g.norm();
                if (gn < 1e-6f) continue;

                float v   = step * q_step_dir.dot(g);      // predicted deepening (m)
                float lim = safety * slack;
                if (v > lim) {
                    q_step_dir -= ((v - lim) / (step * gn)) * (g / gn);
                    bent = true;
                }
            }
        }
        if (bent) pen_steer_bends_++;
        return bent;
    }

    // Workspace variant for the descent phase: only the hand probes are clamped (the hand leads into
    // the foliage; arm probes are still backed by the gate). robot must already be setJointValues to q.
    // dp is the EE position increment; after clipping the excess the step magnitude is preserved (tangential slide).
    bool applyDepthSteeringDescent(Eigen::Vector3f& dp,
                                   float safety, float margin) const {
        if (!pen_gate_enabled_ || pen_probes_.empty()) return false;
        float dpn = dp.norm();
        if (dpn < 1e-9f) return false;
        bool bent = false;
        for (const auto& pr : pen_probes_) {
            if (pr.fid != pen_gate_hand_fid_) continue;    // hand probes only
            Eigen::Matrix4f pose = robot_->getFramePoseById(pr.fid);
            Eigen::Vector3f pp = pose.block<3,1>(0,3)
                               + pose.block<3,3>(0,0) * pr.off;
            if ((pp - pen_gate_exempt_center_).squaredNorm() < pen_gate_exempt_r2_)
                continue;
            for (const auto& cyl : cylinders_) {
                if (cyl.exclude_from_aniso) continue;
                if (cyl.axis_unit.squaredNorm() < 0.25f) continue;
                Eigen::Vector3f seg = cyl.p1 - cyl.p0;
                float seg_sq = seg.squaredNorm();
                float u = (seg_sq > 1e-9f)
                    ? std::max(0.0f, std::min(1.0f,
                          (pp - cyl.p0).dot(seg) / seg_sq))
                    : 0.0f;
                Eigen::Vector3f radial = pp - (cyl.p0 + u * seg);
                float dist = radial.norm();
                if (dist <= 1e-4f) continue;
                float surf = cyl.radius + pr.r;
                if (dist - surf > margin) continue;
                float pen = surf - dist;
                float t_arc = cyl.t0 + u * (cyl.t1 - cyl.t0);
                float allow = penAllowAt(t_arc);
                float slack = allow - std::max(pen, 0.0f);
                if (slack < 0.0f) slack = 0.0f;

                Eigen::Vector3f deep = -(radial / dist);   // deepening direction
                float v   = dp.dot(deep);                  // deepening this step (m)
                float lim = safety * slack;
                if (v > lim) {
                    dp -= (v - lim) * deep;
                    bent = true;
                }
            }
        }
        if (bent) {
            float bn = dp.norm();
            if (bn > 1e-9f) dp *= dpn / bn;                // preserve magnitude: tangential slide
            pen_steer_bends_++;
        }
        return bent;
    }

    // Depth-steering telemetry (cumulative)
    unsigned long long getDepthSteerCalls() const { return pen_steer_calls_; }
    unsigned long long getDepthSteerBends() const { return pen_steer_bends_; }

    // ---- Unified refuse (unified branch-interaction rule, see the note in types.h) ----
    //
    // A single motion cone replacing {tensor edge refuse + sample steering + depth
    // steering}: for every (probe, soft branch) pair in the contact neighborhood
    //   severity = v_in / (κ·slack) + G · v_ax / ax_budget
    //   v_in = radial inward pressure this step (m), v_ax = axial slip this step (m)
    //   G    = before contact: eta*(1 - surf/env)  (approach shaping: come in from the side)
    //          after contact:  eta + (1-eta)*min(1, pen/allow)  (saw penalty grows with depth)
    // When severity > 1, the pressure and slip components are each multiplied by 1/severity (linear,
    // so after shrinking severity = 1), leaving roughly hoop motion. Requires enablePenetrationGate first.
    void enableUnifiedRefuse(float env, float eta,
                             float ax_root, float ax_tip,
                             float safety, float clamp_target = 0.7f) {
        uni_enabled_ = true;
        uni_env_     = env;
        uni_eta_     = eta;
        uni_ax_root_ = ax_root;
        uni_ax_tip_  = ax_tip;
        uni_safety_  = safety;
        uni_clamp_   = clamp_target;
    }
    bool isUnifiedRefuseEnabled() const { return uni_enabled_; }

    // C-space variant (the RRT extension primitive): dir is the normalized proposal direction, step is
    // the actual step size. Returns whether a bend happened; after bending dir is unnormalized, caller renormalizes.
    bool applyUnifiedSteering(const Eigen::VectorXf& q,
                              Eigen::VectorXf& q_step_dir,
                              float step) const {
        if (!uni_enabled_ || !pen_gate_enabled_ || pen_probes_.empty() ||
            step <= 1e-6f)
            return false;
        uni_calls_++;
        robot_->setJointValues(q);
        bool bent = false, jac_ready = false;
        for (const auto& pr : pen_probes_) {
            Eigen::Matrix4f pose = robot_->getFramePoseById(pr.fid);
            Eigen::Vector3f pp = pose.block<3,1>(0,3)
                               + pose.block<3,3>(0,0) * pr.off;
            if ((pp - pen_gate_exempt_center_).squaredNorm() < pen_gate_exempt_r2_)
                continue;
            for (const auto& cyl : cylinders_) {
                if (cyl.exclude_from_aniso) continue;
                if (cyl.axis_unit.squaredNorm() < 0.25f) continue;
                float G, slack, ax_b, h_in;
                Eigen::Vector3f r_hat;
                if (!uniPairGeometry(pp, pr.r, cyl, G, slack, r_hat, ax_b, h_in))
                    continue;

                if (!jac_ready) { robot_->prepareJacobians(); jac_ready = true; }
                Eigen::MatrixXf J = robot_->computeSphereJacobianById(pr.fid, pr.off);
                if (J.rows() != 3 || J.cols() != q_step_dir.size()) continue;
                Eigen::VectorXf g_in = J.transpose() * (-r_hat);
                Eigen::VectorXf g_ax = J.transpose() * cyl.axis_unit;
                float s_in = q_step_dir.dot(g_in);   // inward-pressure rate (m per unit step)
                float s_ax = q_step_dir.dot(g_ax);   // axial-slip rate (signed)
                float v_in = step * std::max(0.0f, s_in);
                float v_ax = step * std::fabs(s_ax);
                float b_in = std::max(uni_safety_ * slack, 1e-4f);
                float sev  = h_in * v_in / b_in + G * v_ax / ax_b;
                if (sev <= uni_clamp_) continue;

                // Shrink both over-budget components to the clamp target (severity is linear in both)
                float keep = uni_clamp_ / sev;
                float gin2 = g_in.squaredNorm();
                if (s_in > 0.0f && gin2 > 1e-12f)
                    q_step_dir -= (1.0f - keep) * s_in * g_in / gin2;
                float gax2 = g_ax.squaredNorm();
                if (gax2 > 1e-12f)
                    q_step_dir -= (1.0f - keep) * s_ax * g_ax / gax2;
                bent = true;
            }
        }
        if (bent) uni_bends_++;
        return bent;
    }

    // Descent variant (workspace): dp is the EE position increment; hand probes only (the hand leads
    // the way in, the arm is still backed by the gate). Magnitude is preserved after shrinking (hoop slide).
    bool applyUnifiedSteeringDescent(Eigen::Vector3f& dp) const {
        if (!uni_enabled_ || !pen_gate_enabled_ || pen_probes_.empty())
            return false;
        float dpn = dp.norm();
        if (dpn < 1e-9f) return false;
        bool bent = false;
        for (const auto& pr : pen_probes_) {
            if (pr.fid != pen_gate_hand_fid_) continue;
            Eigen::Matrix4f pose = robot_->getFramePoseById(pr.fid);
            Eigen::Vector3f pp = pose.block<3,1>(0,3)
                               + pose.block<3,3>(0,0) * pr.off;
            if ((pp - pen_gate_exempt_center_).squaredNorm() < pen_gate_exempt_r2_)
                continue;
            for (const auto& cyl : cylinders_) {
                if (cyl.exclude_from_aniso) continue;
                if (cyl.axis_unit.squaredNorm() < 0.25f) continue;
                float G, slack, ax_b, h_in;
                Eigen::Vector3f r_hat;
                if (!uniPairGeometry(pp, pr.r, cyl, G, slack, r_hat, ax_b, h_in))
                    continue;
                float s_in = dp.dot(-r_hat);
                float s_ax = dp.dot(cyl.axis_unit);
                float v_in = std::max(0.0f, s_in);
                float b_in = std::max(uni_safety_ * slack, 1e-4f);
                float sev  = h_in * v_in / b_in + G * std::fabs(s_ax) / ax_b;
                if (sev <= uni_clamp_) continue;
                float keep = uni_clamp_ / sev;
                if (s_in > 0.0f) dp += (1.0f - keep) * s_in * r_hat;
                dp -= (1.0f - keep) * s_ax * cyl.axis_unit;
                bent = true;
            }
        }
        if (bent) {
            float bn = dp.norm();
            if (bn > 1e-9f) dp *= dpn / bn;
            uni_bends_++;
        }
        return bent;
    }

    // Edge-level severity (companion to the validity layer): take the maximum pair severity over the
    // probe displacement along the whole edge q1->q2. CSpaceSampled rejects with probability
    // sigmoid(gain*(score-1)) (same pipeline as the old edge refuse, with the threshold moved to 1).
    float computeEdgeUnifiedSeverity(const Eigen::VectorXf& q1,
                                     const Eigen::VectorXf& q2) const override {
        if (!uni_enabled_ || !pen_gate_enabled_ || pen_probes_.empty())
            return 0.0f;
        // Probe displacement: two FK evaluations, q1 -> q2
        std::vector<Eigen::Vector3f> p1(pen_probes_.size()), p2(pen_probes_.size());
        robot_->setJointValues(q1);
        for (size_t i = 0; i < pen_probes_.size(); ++i) {
            Eigen::Matrix4f pose = robot_->getFramePoseById(pen_probes_[i].fid);
            p1[i] = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * pen_probes_[i].off;
        }
        robot_->setJointValues(q2);
        for (size_t i = 0; i < pen_probes_.size(); ++i) {
            Eigen::Matrix4f pose = robot_->getFramePoseById(pen_probes_[i].fid);
            p2[i] = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * pen_probes_[i].off;
        }
        float worst = 0.0f;
        for (size_t i = 0; i < pen_probes_.size(); ++i) {
            if ((p1[i] - pen_gate_exempt_center_).squaredNorm() < pen_gate_exempt_r2_)
                continue;
            Eigen::Vector3f dp = p2[i] - p1[i];
            for (const auto& cyl : cylinders_) {
                if (cyl.exclude_from_aniso) continue;
                if (cyl.axis_unit.squaredNorm() < 0.25f) continue;
                float G, slack, ax_b, h_in;
                Eigen::Vector3f r_hat;
                if (!uniPairGeometry(p1[i], pen_probes_[i].r, cyl,
                                     G, slack, r_hat, ax_b, h_in))
                    continue;
                float v_in = std::max(0.0f, dp.dot(-r_hat));
                float b_in = std::max(uni_safety_ * slack, 1e-4f);
                float sev  = h_in * v_in / b_in
                           + G * std::fabs(dp.dot(cyl.axis_unit)) / ax_b;
                if (sev > worst) worst = sev;
            }
        }
        return worst;
    }

    // Unified-refuse telemetry (cumulative)
    unsigned long long getUnifiedCalls() const { return uni_calls_; }
    unsigned long long getUnifiedBends() const { return uni_bends_; }

    void addRobotSphere(const std::string& link_name,
                        float radius,
                        const Eigen::Vector3f& offset) {
        spheres_.push_back({link_name, radius, offset, robot_->getFrameId(link_name)});
        // Pre-allocate Jacobian cache slot (3×7 for Panda); no reallocation during planning
        sphere_jac_cache_.emplace_back(3, robot_->getDimension());
    }

    // ---- Build ----

    // Derive the bounding box automatically from the extent of all cylinders, expand it outward by
    // padding meters, then compute and cache the distance to the nearest cylinder for every voxel.
    // padding should be >= the largest robot sphere radius plus margin, so spheres get correct gradients on approach.
    void buildSdf(float padding = 0.40f) {
        if (cylinders_.empty()) {
            std::cerr << "[SDF] WARNING: No cylinders — SDF trivially empty.\n";
            sdf_built_ = true;
            return;
        }

        // 1. Derive the bounding box from the cylinder geometry
        Eigen::Vector3f bb_min = Eigen::Vector3f::Constant( std::numeric_limits<float>::max());
        Eigen::Vector3f bb_max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::max());
        for (const auto& cyl : cylinders_) {
            Eigen::Vector3f r3 = Eigen::Vector3f::Constant(cyl.radius);
            bb_min = bb_min.cwiseMin(cyl.p0 - r3).cwiseMin(cyl.p1 - r3);
            bb_max = bb_max.cwiseMax(cyl.p0 + r3).cwiseMax(cyl.p1 + r3);
        }
        bbox_min_ = bb_min - Eigen::Vector3f::Constant(padding);
        bbox_max_ = bb_max + Eigen::Vector3f::Constant(padding);

        // 2. Compute the grid dimensions
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

        // 3. Fill in the minimum signed distance per voxel (OpenMP parallel, embarrassingly parallel)
        sdf_grid_.resize(total);
        auto t_start = std::chrono::steady_clock::now();

        #pragma omp parallel for collapse(3) schedule(static)
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

    // ------------------------------------------------------------------
    //
    // NOTE — Removed APIs:
    //   evaluateCost(q) and evaluateGradient(q) were the legacy scalar-
    //   penalty interfaces inherited from the abandoned RRT*-style design
    //   (paper Sec. 3.2). All force-field queries now go through
    //   evaluateAnisotropicForce() below, which directly returns the joint-
    //   space force (radial + axial drag). Setting K_ax = 0 recovers the
    //   pure radial component without computing any scalar potential.
    // ------------------------------------------------------------------
    // evaluateAnisotropicForce — Option A: radial repulsion + axial drag
    //
    // Returns the joint-space FORCE (already negated; caller adds directly
    // to the VF without further negation).
    //
    //   F_radial(ws) = +200 · penetration · grad_sdf   (existing, isotropic)
    //   F_drag  (ws) = −K_ax · penetration · (v_sphere_ws · â) â   (new)
    //   total joint force = Σ_spheres  J_sphere^T · (F_radial + F_drag)
    //
    // â is the unit axis of the nearest cylinder (branch or trunk segment)
    // to the sphere center. v_sphere_ws = J_sphere · v_joint is the workspace
    // velocity induced by the intended extension direction v_joint.
    //
    // F_drag is non-conservative and velocity-dependent — it cannot be
    // expressed as the gradient of any scalar potential. This is the
    // structural property that lets the field encode motion-direction
    // anisotropy (axial vs radial), unlike a conservative cost gradient.
    //
    // When K_ax = 0 (or v_joint = 0) this reduces EXACTLY to
    // -evaluateGradient(q), preserving legacy behavior bit-for-bit.
    // ------------------------------------------------------------------
    grasp_rrt::Config evaluateAnisotropicForce(const grasp_rrt::Config& q,
                                               const grasp_rrt::Config& v_joint,
                                               float K_ax) const {
        if (!sdf_built_) {
            throw std::runtime_error(
                "[SdfCostEvaluator] buildSdf() must be called before planning!");
        }
        robot_->setJointValues(q);
        grasp_rrt::Config total_force = grasp_rrt::Config::Zero(q.size());
        float delta = voxel_size_ * 2.0f;
        float inv_2d = 1.0f / (2.0f * delta);

        const bool use_drag = (K_ax > 0.0f)
                           && (v_joint.size() == q.size())
                           && (v_joint.norm() > 1e-9f);

        for (const auto& rs : spheres_) {
            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            Eigen::Vector3f p = pose.block<3,1>(0,3)
                              + pose.block<3,3>(0,0) * rs.offset;

            float sdf_val = sampleSdf(p);
            float penetration = rs.radius - sdf_val;
            if (penetration <= 0.0f) continue;

            // Workspace SDF gradient (~ unit, points radially out of obstacle)
            Eigen::Vector3f grad_sdf;
            grad_sdf.x() = (sampleSdf(p + Eigen::Vector3f(delta, 0, 0))
                          - sampleSdf(p - Eigen::Vector3f(delta, 0, 0))) * inv_2d;
            grad_sdf.y() = (sampleSdf(p + Eigen::Vector3f(0, delta, 0))
                          - sampleSdf(p - Eigen::Vector3f(0, delta, 0))) * inv_2d;
            grad_sdf.z() = (sampleSdf(p + Eigen::Vector3f(0, 0, delta))
                          - sampleSdf(p - Eigen::Vector3f(0, 0, delta))) * inv_2d;

            // Existing radial repulsion: F = +200·pen·∇sdf  (anti-gradient of U)
            Eigen::Vector3f F_total_ws = 200.0f * penetration * grad_sdf;

            // Per-sphere Jacobian (3xN) — needed regardless of drag for J^T mapping.
            auto panda = dynamic_cast<PandaPinocchioRobot*>(robot_.get());
            grasp_rrt::MatrixXf J_sphere;
            if (panda) {
                J_sphere = panda->computeSphereJacobianDirect(rs.frame_id, rs.offset);
            } else {
                robot_->prepareJacobians();
                J_sphere = robot_->computeSphereJacobianById(rs.frame_id, rs.offset);
            }

            // Axial drag: only when K_ax > 0, v_joint nonzero, and nearest
            // cylinder has a non-degenerate axis (skips overhead-sphere case).
            if (use_drag) {
                Eigen::Vector3f a_hat = findNearestCylinderAxis(p);
                if (a_hat.squaredNorm() > 0.25f) {   // valid unit axis
                    Eigen::Vector3f v_sphere_ws = J_sphere * v_joint;
                    float v_ax_scalar = v_sphere_ws.dot(a_hat);
                    F_total_ws.noalias() +=
                        (-K_ax * penetration * v_ax_scalar) * a_hat;
                }
            }

            total_force.noalias() += J_sphere.transpose() * F_total_ws;
        }
        return total_force;
    }

    // ------------------------------------------------------------------
    // measureWaypointAnisotropy — path-quality instrumentation
    //
    // Post-hoc measurement (NOT used during planning). For one waypoint q
    // with intended motion v_joint, returns (axial_weighted, total_weighted)
    // accumulated over all robot collision spheres that are CURRENTLY in
    // branch contact (penetration > 0):
    //
    //   axial += penetration · |v_sphere · â|     ← motion projected on branch axis
    //   total += penetration · |v_sphere|         ← full motion magnitude
    //
    // Caller accumulates across all waypoints and computes the fraction
    //   f = Σ axial / Σ total ∈ [0, 1]
    // Lower f means more perpendicular crossings (Option A's intended effect).
    //
    // Cost ≈ ~50 spheres · cheap ops per waypoint; <1ms per solution path.
    // Degenerate axis (overhead-sphere) and free-space (penetration ≤ 0) are
    // silently skipped (contribute nothing).
    // ------------------------------------------------------------------
    // engagement_only = true restricts the sum to the two engagement-zone
    // spheres (panda_hand, panda_link6) — the links the anisotropy methods
    // actually control. Secondary diagnostic; the primary (all-sphere)
    // definition is unchanged and stays comparable with all past results.
    std::pair<double, double> measureWaypointAnisotropy(
        const grasp_rrt::Config& q,
        const grasp_rrt::Config& v_joint,
        bool engagement_only = false) const {
        if (!sdf_built_) return {0.0, 0.0};
        if (v_joint.size() != q.size() || v_joint.norm() < 1e-9f) return {0.0, 0.0};

        robot_->setJointValues(q);
        double axial_acc = 0.0, total_acc = 0.0;

        for (size_t k = 0; k < spheres_.size(); ++k) {
            const auto& rs = spheres_[k];
            // engagement_only now follows the planner's target sphere set (erSphereActive): by default
            // hand+link6; with a custom --er-links it follows that set. Note this makes axial_ee a
            // self-assessment of "did the planner protect its own targets", so arms with different
            // sphere sets are no longer directly comparable.
            if (engagement_only && !erSphereActive(k)) continue;
            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            Eigen::Vector3f p = pose.block<3,1>(0,3)
                              + pose.block<3,3>(0,0) * rs.offset;

            float sdf_val = sampleSdf(p);
            float penetration = rs.radius - sdf_val;
            if (penetration <= 0.0f) continue;

            Eigen::Vector3f a_hat = findNearestCylinderAxis(p);
            if (a_hat.squaredNorm() < 0.25f) continue;  // degenerate axis (skip)

            // Per-sphere workspace velocity
            auto panda = dynamic_cast<PandaPinocchioRobot*>(robot_.get());
            grasp_rrt::MatrixXf J_sphere;
            if (panda) {
                J_sphere = panda->computeSphereJacobianDirect(rs.frame_id, rs.offset);
            } else {
                robot_->prepareJacobians();
                J_sphere = robot_->computeSphereJacobianById(rs.frame_id, rs.offset);
            }
            Eigen::Vector3f v_sphere_ws = J_sphere * v_joint;
            float v_norm = v_sphere_ws.norm();
            if (v_norm < 1e-9f) continue;

            float v_ax_abs = std::abs(v_sphere_ws.dot(a_hat));
            axial_acc += static_cast<double>(penetration) * v_ax_abs;
            total_acc += static_cast<double>(penetration) * v_norm;
        }
        return {axial_acc, total_acc};
    }

    // ------------------------------------------------------------------
    // measureWaypointAnisotropyDisplacement — depth-based (not velocity)
    //
    // For a path segment q_prev → q_curr, accumulates penetration-weighted
    // path-length contributions, decomposed into radial and axial directions
    // (L1 decomposition: radial + axial = 1).
    //
    //   For each sphere s with penetration > 0 at q_curr:
    //       Δp_s   = sphere_position(q_curr) − sphere_position(q_prev)
    //       Δp_ax  = |Δp_s · â|                       (axial component)
    //       Δp_rad = ‖Δp_s − (Δp_s · â)â‖             (radial component)
    //
    //       axial_acc  += penetration · Δp_ax
    //       radial_acc += penetration · Δp_rad
    //
    // Caller divides:
    //       radial_fraction_L1 = Σ radial_acc / Σ (radial_acc + axial_acc)
    //
    // Interpretation: "of the total penetration-weighted path inside the
    // obstacle, what fraction was traveled in the radial direction?"
    //   1.0 → robot pushed straight through the wall (pure radial)
    //   0.0 → robot slid along the cylinder axis (pure axial)
    //
    // No velocity, no Jacobian — pure position differences.
    // ------------------------------------------------------------------
    std::pair<double, double> measureWaypointAnisotropyDisplacement(
        const grasp_rrt::Config& q_prev,
        const grasp_rrt::Config& q_curr) const {
        if (!sdf_built_) return {0.0, 0.0};
        if (q_prev.size() != q_curr.size()) return {0.0, 0.0};

        // Cache sphere positions at q_prev
        robot_->setJointValues(q_prev);
        std::vector<Eigen::Vector3f> p_prev(spheres_.size());
        for (size_t k = 0; k < spheres_.size(); ++k) {
            const auto& rs = spheres_[k];
            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            p_prev[k] = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * rs.offset;
        }

        // Evaluate at q_curr: penetration + Δp + axis decomposition
        robot_->setJointValues(q_curr);
        double radial_acc = 0.0, axial_acc = 0.0;
        for (size_t k = 0; k < spheres_.size(); ++k) {
            const auto& rs = spheres_[k];
            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            Eigen::Vector3f p_curr = pose.block<3,1>(0,3)
                                   + pose.block<3,3>(0,0) * rs.offset;

            float sdf_val = sampleSdf(p_curr);
            float penetration = rs.radius - sdf_val;
            if (penetration <= 0.0f) continue;

            Eigen::Vector3f a_hat = findNearestCylinderAxis(p_curr);
            if (a_hat.squaredNorm() < 0.25f) continue;  // degenerate axis (skip)

            Eigen::Vector3f dp = p_curr - p_prev[k];
            float dp_axial = std::abs(dp.dot(a_hat));
            Eigen::Vector3f dp_radial_vec = dp - dp.dot(a_hat) * a_hat;
            float dp_radial = dp_radial_vec.norm();

            axial_acc  += static_cast<double>(penetration) * dp_axial;
            radial_acc += static_cast<double>(penetration) * dp_radial;
        }
        return {axial_acc, radial_acc};
    }

    // ------------------------------------------------------------------
    // computeEdgeAxialScore — Anisotropic edge refuse (edge refuse, Option A)
    //
    // Multi-branch axial-alignment score with K-nearest per-sphere max
    // aggregation:
    //
    //   For each engagement-zone sphere s (panda_hand, panda_link6):
    //       p_s     = sphere position at q_new
    //       Δp_s    = p_s − sphere_pos(q_near, s)
    //       For each non-rigid cylinder b within envelope of s, keep the
    //       K nearest:
    //           weight_sb = max(0, 1 − dist_sb / envelope)
    //           cos_sb    = |Δp̂_s · â_b|
    //           score_sb  = weight_sb · cos_sb
    //   max_score = max over all (sphere s, branch b) pairs
    //
    // Rigid cylinders (exclude_from_aniso = true, e.g., trunk) are skipped
    // entirely — anisotropy is a property of semi-rigid foliage.
    //
    // Sphere selection (D change): panda_hand (1st link from EE) and
    // panda_link6 (3rd from EE). Skips panda_link7 (2nd, kinematically
    // slaved to the hand) and panda_link5 (4th, slaved to link6).
    //
    // K = 5 default (controlled by caller). Bounded compute regardless
    // of foliage density: per call O(N + K) for N total cylinders.
    // ------------------------------------------------------------------
    float computeEdgeAxialScore(const Eigen::VectorXf& q_near,
                                 const Eigen::VectorXf& q_new,
                                 float envelope,
                                 int   K = 3) const override {
        if (!sdf_built_) return 0.0f;
        if (q_near.size() != q_new.size()) return 0.0f;
        if (envelope <= 0.0f) return 0.0f;
        if (cylinders_.empty()) return 0.0f;
        if (K < 1) K = 1;

        // ---- Optimization A: thread-local cache of q_near sphere positions
        // Each thread keeps the most-recent q_near it computed FK for. RRT
        // often re-uses the same q_near across consecutive cycles (nearest-
        // neighbor returns the same node), so this saves a setJointValues
        // call (~10-15 μs) on cache hits.
        struct ThreadFKCache {
            Eigen::VectorXf  q_cached;
            // We store positions for ALL spheres; only engagement ones are
            // read, but extra cost is negligible (~12 vec3).
            std::vector<Eigen::Vector3f> p_prev;
            bool valid = false;
        };
        static thread_local ThreadFKCache cache;

        const std::vector<Eigen::Vector3f>* p_prev_ptr = nullptr;

        bool reuse = cache.valid
                  && cache.q_cached.size() == q_near.size()
                  && (cache.q_cached - q_near).cwiseAbs().maxCoeff() < 1e-7f
                  && cache.p_prev.size() == spheres_.size();

        if (reuse) {
            p_prev_ptr = &cache.p_prev;
        } else {
            robot_->setJointValues(q_near);
            cache.p_prev.assign(spheres_.size(), Eigen::Vector3f::Zero());
            for (size_t k = 0; k < spheres_.size(); ++k) {
                const auto& rs = spheres_[k];
                if (rs.link_name != "panda_hand" &&
                    rs.link_name != "panda_link6") {
                    continue;
                }
                Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
                cache.p_prev[k] = pose.block<3,1>(0,3)
                                + pose.block<3,3>(0,0) * rs.offset;
            }
            cache.q_cached = q_near;
            cache.valid    = true;
            p_prev_ptr     = &cache.p_prev;
        }
        const auto& p_prev = *p_prev_ptr;

        // ---- Evaluate at q_new -----------------------------------------
        robot_->setJointValues(q_new);
        float max_score = 0.0f;
        struct BranchHit { float dist; Eigen::Vector3f axis; };
        std::vector<BranchHit> nearby;
        nearby.reserve(cylinders_.size());

        for (size_t k = 0; k < spheres_.size(); ++k) {
            const auto& rs = spheres_[k];
            if (rs.link_name != "panda_hand" &&
                rs.link_name != "panda_link6") {
                continue;
            }

            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            Eigen::Vector3f p_curr = pose.block<3,1>(0,3)
                                   + pose.block<3,3>(0,0) * rs.offset;

            // ---- Optimization C: fast SDF early-exit
            // Single trilinear SDF lookup; if even the closest obstacle is
            // further than envelope, no cylinder can match. Skips the
            // O(N_cyl) loop for spheres in clear free space.
            float sdf_val = sampleSdf(p_curr);
            if (sdf_val - rs.radius > envelope) continue;

            // Displacement of this sphere
            Eigen::Vector3f dp = p_curr - p_prev[k];
            float dp_norm = dp.norm();
            if (dp_norm < 1e-4f) continue;
            Eigen::Vector3f dp_hat = dp / dp_norm;

            // Collect non-rigid branches with body-to-surface dist < envelope.
            // Uses cached axis_unit (Optimization B) — no normalize() inside.
            nearby.clear();
            for (const auto& cyl : cylinders_) {
                if (cyl.exclude_from_aniso) continue;
                if (cyl.axis_unit.squaredNorm() < 0.25f) continue;  // degenerate
                float d = distToCylinder(p_curr, cyl) - rs.radius;
                if (d > envelope) continue;
                nearby.push_back({d, cyl.axis_unit});
            }
            if (nearby.empty()) continue;

            if (static_cast<int>(nearby.size()) > K) {
                std::partial_sort(
                    nearby.begin(), nearby.begin() + K, nearby.end(),
                    [](const BranchHit& a, const BranchHit& b) {
                        return a.dist < b.dist;
                    });
                nearby.resize(K);
            }

            for (const auto& br : nearby) {
                float weight = std::max(0.0f, 1.0f - br.dist / envelope);
                float cos_ax = std::abs(dp_hat.dot(br.axis));
                float score  = weight * cos_ax;
                if (score > max_score) max_score = score;
            }
        }
        return max_score;
    }

    // ==================================================================
    // STRUCTURE TENSOR — the shared anisotropy primitive
    //
    // buildLocalAxisTensor(p, body_radius, envelope) returns
    //
    //     M(p) = sum over non-rigid branches i of  w_i(p) * a_i * a_i^T
    //
    // where a_i is branch i's unit axis and w_i is a smoothstep weight on
    // the body-to-surface distance d_i = distToCylinder(p) − body_radius:
    //
    //     w_i = 1                              if d_i <= 0   (contact)
    //     w_i = smoothstep(1 − d_i/envelope)   if 0 < d_i < envelope
    //     w_i = 0                              if d_i >= envelope
    //
    // Properties that make M the right object (vs a single winner axis):
    //   * SMOOTH: w_i are C1 in p and each a_i*a_i^T is constant, so M(p)
    //     is C1. There is NO selection step — nothing that can flip when
    //     the dominant branch changes. This removes the argmax/argmin
    //     discontinuity (the oscillation source) at the root.
    //   * SIGN-INVARIANT: a*a^T == (−a)*(−a)^T, so branches stored with
    //     opposite axis signs (e.g. opposite sides of the trunk) ADD
    //     instead of cancelling — a weighted vector average would give 0.
    //   * FULL CAPACITY: a symmetric 3x3 carries every branch's
    //     constraint at once. axial_penalty(d) = d^T M d
    //     = sum_i w_i (d . a_i)^2 — sliding along ANY nearby branch is
    //     penalized; a single blended vector can encode only one.
    //
    // trace(M) = sum_i w_i (since trace(a a^T) = |a|^2 = 1), so
    // M / trace(M) is a weighted mean of cos^2 terms, bounded in [0, 1].
    //
    // The trunk and other rigid obstacles (exclude_from_aniso) contribute
    // nothing, consistent with every other anisotropy mechanism.
    // Cost: O(N_cyl) with no allocation; a few μs for N ~ 12.
    // ==================================================================
    Eigen::Matrix3f buildLocalAxisTensor(const Eigen::Vector3f& p,
                                         float body_radius,
                                         float envelope,
                                         float* weight_sum_out = nullptr) const {
        Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
        float w_sum = 0.0f;
        for (const auto& cyl : cylinders_) {
            if (cyl.exclude_from_aniso) continue;               // trunk etc.
            if (cyl.axis_unit.squaredNorm() < 0.25f) continue;  // degenerate
            float d = distToCylinder(p, cyl) - body_radius;
            if (d >= envelope) continue;
            float t = 1.0f - std::max(d, 0.0f) / envelope;      // (0, 1]
            float w = t * t * (3.0f - 2.0f * t);                // smoothstep
            M.noalias() += w * (cyl.axis_unit * cyl.axis_unit.transpose());
            w_sum += w;
        }
        if (weight_sum_out) *weight_sum_out = w_sum;
        return M;
    }

    // ------------------------------------------------------------------
    // computeEdgeAxialScoreTensor — edge-refuse score, tensor aggregation
    //
    // Drop-in replacement for computeEdgeAxialScore: same engagement
    // spheres (panda_hand, panda_link6), same displacement dp per sphere,
    // same [0, 1] score contract feeding the same sigmoid. The only
    // change is HOW branches are aggregated:
    //
    //   legacy:  score = MAX over (sphere, branch) of  w_b * |dp_hat . a_b|
    //            → the max SELECTS a winner; near a tie between branches
    //              the selection flips → rejection-axis oscillation.
    //
    //   tensor:  per sphere s at position p, over in-envelope branches i
    //            with smoothstep weights w_i and c_i = |dp_hat . a_i|:
    //              raw   = ( sum_i w_i c_i^8 / sum_i w_i )^(1/8)
    //              gate  = min(1, sum_i w_i)      (envelope proximity ramp)
    //              s_score = gate * raw
    //            score = max over the two engagement spheres of s_score
    //            (max over the FIXED sphere set is continuous; the
    //             discontinuity was in the per-branch selection).
    //
    // DESIGN HISTORY (two failed variants, measured on V6, start1, n=10):
    //
    //  v1: trace-normalized quadratic form sqrt(dp^T M dp / tr M), weights
    //      clamped at contact. axial 0.682 vs legacy MAX 0.428. Two
    //      defects: (a) the MEAN dilutes in envelope overlaps (motion
    //      axial to B but radial to A scores 0.707 < threshold);
    //      (b) see v2.
    //  v2: order-8 normalized soft-max, weights still clamped. axial
    //      0.699 — no better. Telemetry exposed the real defect: the
    //      legacy weight W = 1 − d/envelope is UNCLAMPED — when the
    //      sphere penetrates (d < 0), W grows past 1 (mean legacy score
    //      1.48, reject rate 91%!). Penetration depth AMPLIFIES legacy
    //      rejection, tightening the allowed cone the deeper the contact:
    //      at W = 2 only motions with |cos| < 0.45 (> 63 deg off axis)
    //      survive. Clamping W at contact neutered exactly that engine.
    //
    //  v3 (this code): unnormalized order-8 soft-max over per-
    //      (sphere, branch) scores with the legacy unclamped weight:
    //          W_i   = 1 − d_i/envelope        (> 1 when penetrating)
    //          score = ( sum_i (W_i c_i)^8 )^(1/8)
    //      Single branch: score = W c — EXACTLY the legacy formula,
    //      amplification included. Multi-branch: smooth soft-max ≥ max
    //      (1.09x for two equal contributors) — conservative, and
    //      selection-free: no argmax, nothing flips at branch handoff.
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // Custom scoring-sphere set for tensor edge refuse (experimental)
    //
    // sel = {link name, index of the sphere on that link (0-based, in addRobotSphere order)}.
    // Must be called after all addRobotSphere calls; an empty set restores the default
    // (all spheres of panda_hand + panda_link6). Affects tensor mode only;
    // legacy MAX mode stays fixed at hand+link6.
    // ------------------------------------------------------------------
    void setEdgeRefuseSphereSet(
        const std::vector<std::pair<std::string, int>>& sel) {
        if (sel.empty()) { er_sphere_mask_.clear(); return; }
        er_sphere_mask_.assign(spheres_.size(), 0);
        for (size_t k = 0; k < spheres_.size(); ++k) {
            // Index of this sphere on its link
            int idx = 0;
            for (size_t j = 0; j < k; ++j)
                if (spheres_[j].link_name == spheres_[k].link_name) ++idx;
            for (const auto& s : sel)
                if (s.first == spheres_[k].link_name && s.second == idx)
                    er_sphere_mask_[k] = 1;
        }
    }

    // tensor edge refuse: does sphere k participate in scoring
    bool erSphereActive(size_t k) const {
        if (!er_sphere_mask_.empty()) return er_sphere_mask_[k] != 0;
        const auto& rs = spheres_[k];
        return rs.link_name == "panda_hand" || rs.link_name == "panda_link6";
    }

    float computeEdgeAxialScoreTensor(const Eigen::VectorXf& q_near,
                                      const Eigen::VectorXf& q_new,
                                      float envelope) const override {
        if (!sdf_built_ || cylinders_.empty()) return 0.0f;
        if (q_near.size() != q_new.size()) return 0.0f;
        if (envelope <= 0.0f) return 0.0f;

        // Sphere positions at q_near (engagement spheres only).
        robot_->setJointValues(q_near);
        std::vector<Eigen::Vector3f> p_prev(spheres_.size(),
                                            Eigen::Vector3f::Zero());
        for (size_t k = 0; k < spheres_.size(); ++k) {
            if (!erSphereActive(k)) continue;
            const auto& rs = spheres_[k];
            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            p_prev[k] = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * rs.offset;
        }

        // Evaluate at q_new.
        robot_->setJointValues(q_new);
        float best = 0.0f;
        // Change 2: hand sphere position, used for the goal-proximity fade g at the end
        Eigen::Vector3f p_hand = Eigen::Vector3f::Zero();
        bool hand_seen = false;
        for (size_t k = 0; k < spheres_.size(); ++k) {
            const auto& rs = spheres_[k];
            if (!erSphereActive(k)) continue;

            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            Eigen::Vector3f p_curr = pose.block<3,1>(0,3)
                                   + pose.block<3,3>(0,0) * rs.offset;
            // Note: if the custom sphere set has no hand sphere, the goal fade is inactive (it needs the hand position)
            if (!hand_seen && rs.link_name == "panda_hand"
                && rs.offset.norm() < 1e-4f) {
                p_hand = p_curr;
                hand_seen = true;
            }

            // Fast SDF early-exit (same as legacy): if even the nearest
            // obstacle is beyond the envelope, skip the cylinder loop.
            float sdf_val = sampleSdf(p_curr);
            if (sdf_val - rs.radius > envelope) continue;

            Eigen::Vector3f dp = p_curr - p_prev[k];
            float dp_norm = dp.norm();
            if (dp_norm < 1e-4f) continue;
            Eigen::Vector3f dp_hat = dp / dp_norm;

            // Unnormalized order-8 soft-max with legacy unclamped weight.
            float acc = 0.0f;
            bool  any = false;
            for (const auto& cyl : cylinders_) {
                if (cyl.exclude_from_aniso) continue;
                if (cyl.axis_unit.squaredNorm() < 0.25f) continue;
                float d = distToCylinder(p_curr, cyl) - rs.radius;
                if (d >= envelope) continue;
                float W = 1.0f - d / envelope;   // UNCLAMPED: > 1 inside
                float c = std::abs(dp_hat.dot(cyl.axis_unit));
                float s1 = W * c;
                float s2 = s1 * s1, s4 = s2 * s2;
                acc += s4 * s4;                  // (W c)^8
                any = true;
            }
            if (!any) continue;

            float s = std::sqrt(std::sqrt(std::sqrt(acc)));   // ^(1/8)
            if (s > best) best = s;
        }
        // Change 2: goal-proximity fade — in the last few centimeters the approach direction is
        // dictated by the grasp pose, so rejecting only causes unreachability; fade it out linearly by g.
        if (hand_seen) best *= goalFadeAt(p_hand);
        return best;
    }

    // ------------------------------------------------------------------
    // maxWinnerBranchId — post-hoc "which branch is most dangerous here"
    //
    // Same MAX-of-K over both engagement spheres as computeEdgeAxialScore,
    // but returns the cylinder INDEX achieving the maximum score (weight·cos)
    // instead of the score value. Returns -1 if no non-rigid branch is within
    // envelope of either sphere along this edge.
    //
    // Used only for instrumentation: walking a solution path and counting how
    // often this winner changes edge-to-edge measures oscillation of the
    // rejection axis directly (original edge refuse's de-facto driver). Not used
    // during planning.
    // ------------------------------------------------------------------
    int maxWinnerBranchId(const Eigen::VectorXf& q_near,
                          const Eigen::VectorXf& q_new,
                          float envelope,
                          int   K = 3) const {
        if (!sdf_built_ || cylinders_.empty()) return -1;
        if (q_near.size() != q_new.size()) return -1;
        if (envelope <= 0.0f) return -1;
        if (K < 1) K = 1;

        // Cache q_near sphere positions for the engagement spheres.
        robot_->setJointValues(q_near);
        std::vector<Eigen::Vector3f> p_prev(spheres_.size(), Eigen::Vector3f::Zero());
        for (size_t k = 0; k < spheres_.size(); ++k) {
            const auto& rs = spheres_[k];
            if (rs.link_name != "panda_hand" && rs.link_name != "panda_link6")
                continue;
            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            p_prev[k] = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * rs.offset;
        }

        robot_->setJointValues(q_new);
        float max_score = 0.0f;
        int   best_id   = -1;
        struct BranchHit { int idx; float dist; Eigen::Vector3f axis; };
        std::vector<BranchHit> nearby;
        nearby.reserve(cylinders_.size());

        for (size_t k = 0; k < spheres_.size(); ++k) {
            const auto& rs = spheres_[k];
            if (rs.link_name != "panda_hand" && rs.link_name != "panda_link6")
                continue;
            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            Eigen::Vector3f p_curr = pose.block<3,1>(0,3)
                                   + pose.block<3,3>(0,0) * rs.offset;

            Eigen::Vector3f dp = p_curr - p_prev[k];
            float dp_norm = dp.norm();
            if (dp_norm < 1e-5f) continue;
            Eigen::Vector3f dp_hat = dp / dp_norm;

            nearby.clear();
            for (size_t i = 0; i < cylinders_.size(); ++i) {
                const auto& cyl = cylinders_[i];
                if (cyl.exclude_from_aniso) continue;
                if (cyl.axis_unit.squaredNorm() < 0.25f) continue;
                float d = distToCylinder(p_curr, cyl) - rs.radius;
                if (d > envelope) continue;
                nearby.push_back({static_cast<int>(i), d, cyl.axis_unit});
            }
            if (nearby.empty()) continue;
            if (static_cast<int>(nearby.size()) > K) {
                std::partial_sort(nearby.begin(), nearby.begin() + K, nearby.end(),
                    [](const BranchHit& a, const BranchHit& b){ return a.dist < b.dist; });
                nearby.resize(K);
            }
            for (const auto& br : nearby) {
                float weight = std::max(0.0f, 1.0f - br.dist / envelope);
                float cos_ax = std::abs(dp_hat.dot(br.axis));
                float score  = weight * cos_ax;
                if (score > max_score) { max_score = score; best_id = br.idx; }
            }
        }
        return best_id;
    }

    // ==================================================================
    // Sticky-winner selection (shared by sticky-edge-refuse and sticky-sample-steering variants)
    //
    // Given the parent node's stored winner branch id and the current motion
    // direction, this function:
    //   1. Scores the K nearest non-rigid branches for the current edge.
    //   2. Reads the parent's winner id.
    //   3. Applies hysteresis: switch only if new-MAX exceeds prev-winner
    //      current score by hysteresis_margin (e.g., 1.25).
    //   4. Returns the chosen winner's cylinder index, its current score,
    //      and its axis.
    //
    // If prev_winner_id is out of scope (not in current envelope) or -1,
    // fresh MAX is used.
    //
    // Cost: ~10-15 μs (no eigendecomposition, no tensor).
    // ------------------------------------------------------------------
    struct StickyResult {
        int  chosen_id     = -1;   // cylinder index; -1 = no branch in scope
        int  new_max_id    = -1;   // fresh MAX (for diagnostics)
        float chosen_score = 0.0f; // weight · |Δp̂ · â| for chosen winner
        Eigen::Vector3f chosen_axis = Eigen::Vector3f::Zero();
        bool  switched     = false; // true if we picked new_max over prev
    };

    StickyResult stickyWinnerSelect(const Eigen::VectorXf& q_near,
                                     const Eigen::VectorXf& q_step_dir,
                                     const Eigen::MatrixXf& J_EE,
                                     int   prev_winner_id,
                                     float envelope,
                                     int   K,
                                     float hysteresis_margin) const {
        StickyResult result;
        if (!sdf_built_ || cylinders_.empty()) return result;
        if (envelope <= 0.0f) return result;
        if (K < 1) K = 1;

        // ---- Engagement spheres (panda_hand AND panda_link6) ----
        // The winner-selection gate fires when EITHER sphere is within
        // envelope of a branch (Fix A). Using only panda_hand missed cases
        // where link6 engages a branch the hand has already passed.
        struct EngSphere { Eigen::Vector3f p; float radius; };
        std::vector<EngSphere> eng;
        eng.reserve(2);
        for (const auto& rs : spheres_) {
            if ((rs.link_name == "panda_hand" || rs.link_name == "panda_link6")
                && rs.offset.norm() < 1e-4f) {
                Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
                eng.push_back({pose.block<3,1>(0,3)
                             + pose.block<3,3>(0,0) * rs.offset, rs.radius});
            }
        }
        if (eng.empty()) return result;

        // ---- Predicted workspace motion direction (hand Jacobian) ----
        // Used only for the axial-alignment term of the selection score
        // (identity/hysteresis). The actual rejection strength is measured
        // separately by stickyWinnerEdgeScore over the real edge.
        Eigen::Vector3f dp = J_EE * q_step_dir;
        float dp_norm = dp.norm();
        if (dp_norm < 1e-6f) return result;
        Eigen::Vector3f dp_hat = dp / dp_norm;

        // ---- Collect K nearest non-rigid branches within envelope --------
        // Distance to each branch = MIN over both engagement spheres.
        // Penetration (d < 0) is KEPT (clamped weight) — a branch the sphere
        // has entered is maximally relevant, not to be discarded.
        struct Cand { int idx; float weight; float score; Eigen::Vector3f axis; };
        std::vector<Cand> nearby;
        nearby.reserve(cylinders_.size());
        for (size_t i = 0; i < cylinders_.size(); ++i) {
            const auto& cyl = cylinders_[i];
            if (cyl.exclude_from_aniso) continue;
            if (cyl.axis_unit.squaredNorm() < 0.25f) continue;
            float d = std::numeric_limits<float>::max();
            for (const auto& s : eng) {
                d = std::min(d, distToCylinder(s.p, cyl) - s.radius);
            }
            if (d > envelope) continue;
            float w = std::min(1.0f, std::max(0.0f, 1.0f - d / envelope));
            float cos_val = std::abs(dp_hat.dot(cyl.axis_unit));
            float score = w * cos_val;
            nearby.push_back({static_cast<int>(i), w, score, cyl.axis_unit});
        }
        if (nearby.empty()) return result;

        // Keep K nearest by inverse-weight (highest weight = closest)
        if (static_cast<int>(nearby.size()) > K) {
            std::partial_sort(nearby.begin(), nearby.begin() + K, nearby.end(),
                [](const Cand& a, const Cand& b) { return a.weight > b.weight; });
            nearby.resize(K);
        }

        // ---- Find new MAX and score of previous winner (if in scope) ----
        int max_i = 0;
        for (int i = 1; i < (int)nearby.size(); ++i) {
            if (nearby[i].score > nearby[max_i].score) max_i = i;
        }
        result.new_max_id = nearby[max_i].idx;

        int prev_idx = -1;
        if (prev_winner_id >= 0) {
            for (int i = 0; i < (int)nearby.size(); ++i) {
                if (nearby[i].idx == prev_winner_id) { prev_idx = i; break; }
            }
        }

        // ---- Hysteresis decision ----
        if (prev_idx < 0) {
            // No prev winner OR prev fell out of scope → pick new MAX
            result.chosen_id    = nearby[max_i].idx;
            result.chosen_score = nearby[max_i].score;
            result.chosen_axis  = nearby[max_i].axis;
            result.switched     = (prev_winner_id != result.chosen_id);
        } else {
            float prev_score = nearby[prev_idx].score;
            float new_score  = nearby[max_i].score;
            if (new_score > hysteresis_margin * prev_score) {
                result.chosen_id    = nearby[max_i].idx;
                result.chosen_score = new_score;
                result.chosen_axis  = nearby[max_i].axis;
                result.switched     = true;
            } else {
                result.chosen_id    = nearby[prev_idx].idx;
                result.chosen_score = prev_score;
                result.chosen_axis  = nearby[prev_idx].axis;
                result.switched     = false;
            }
        }
        return result;
    }

    // ------------------------------------------------------------------
    // stickyWinnerEdgeScore — thorough rejection score for sticky-edge-refuse
    //
    // Given the committed winner branch (winner_cyl_id from
    // stickyWinnerSelect), measure how axial the edge q_near→q_new is w.r.t.
    // THAT branch's axis. Unlike the single Jacobian-prediction score returned
    // by stickyWinnerSelect, this:
    //   * uses BOTH engagement spheres (panda_hand AND panda_link6), and
    //   * sub-samples along the edge at DCD granularity, taking the MAX over
    //     sub-segments and spheres.
    //
    // It is still a "winner score" (single branch — per the sticky-winner
    // philosophy, NOT a MAX over different branches), but measured as
    // thoroughly as original edge refuse's isPathValid rejection. This catches edges
    // that ENTER the winner's envelope partway through, or BECOME axial
    // mid-edge — cases a single q_near Jacobian prediction misses.
    //
    // Returns max weight·|Δp̂·â|; 0 if the winner is never within envelope
    // along the edge. dcd_step controls sub-sample count (edge_len/dcd_step).
    // ------------------------------------------------------------------
    float stickyWinnerEdgeScore(const Eigen::VectorXf& q_near,
                                const Eigen::VectorXf& q_new,
                                int   winner_cyl_id,
                                float envelope,
                                float dcd_step) const {
        if (!sdf_built_) return 0.0f;
        if (winner_cyl_id < 0 || winner_cyl_id >= (int)cylinders_.size())
            return 0.0f;
        const auto& wcyl = cylinders_[winner_cyl_id];
        if (wcyl.exclude_from_aniso) return 0.0f;
        if (wcyl.axis_unit.squaredNorm() < 0.25f) return 0.0f;
        if (envelope <= 0.0f) return 0.0f;

        // Sub-sample count matched to DCD granularity, capped for safety.
        float edge_len = (q_new - q_near).norm();
        int num_sub = 1;
        if (dcd_step > 1e-6f) {
            num_sub = std::max(1, (int)std::ceil(edge_len / dcd_step));
        }
        num_sub = std::min(num_sub, 8);

        const Eigen::Vector3f axis = wcyl.axis_unit;
        float max_score = 0.0f;

        for (const auto& rs : spheres_) {
            if (rs.link_name != "panda_hand" &&
                rs.link_name != "panda_link6") {
                continue;
            }

            robot_->setJointValues(q_near);
            Eigen::Matrix4f pose_prev = robot_->getFramePoseById(rs.frame_id);
            Eigen::Vector3f p_prev = pose_prev.block<3,1>(0,3)
                                   + pose_prev.block<3,3>(0,0) * rs.offset;

            for (int s = 1; s <= num_sub; ++s) {
                float t = (float)s / (float)num_sub;
                Eigen::VectorXf q_sub = q_near + t * (q_new - q_near);
                robot_->setJointValues(q_sub);
                Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
                Eigen::Vector3f p_curr = pose.block<3,1>(0,3)
                                       + pose.block<3,3>(0,0) * rs.offset;

                Eigen::Vector3f dp = p_curr - p_prev;
                float dp_norm = dp.norm();
                if (dp_norm >= 1e-5f) {
                    // Envelope gate against the winner branch (incl. penetration)
                    float d = distToCylinder(p_curr, wcyl) - rs.radius;
                    if (d <= envelope) {
                        float w = std::max(0.0f, 1.0f - d / envelope);
                        float cos_ax = std::abs((dp / dp_norm).dot(axis));
                        float score = w * cos_ax;
                        if (score > max_score) max_score = score;
                    }
                }
                p_prev = p_curr;   // next sub-segment starts here
            }
        }
        return max_score;
    }

    // Bend q_step_dir so that the resulting EE workspace motion sits at
    // exactly cos_cap against `axis` (measured by |Δp̂·â|). Deterministic
    // sample-steering-style action. Uses J_EE^T projection, no pseudoinverse.
    // No-op if already off-axis (cos ≤ cos_cap).
    void bendAgainstAxis(Eigen::VectorXf& q_step_dir,
                         const Eigen::MatrixXf& J_EE,
                         const Eigen::Vector3f& axis,
                         float cos_cap) const {
        Eigen::Vector3f dp = J_EE * q_step_dir;
        float dp_norm = dp.norm();
        if (dp_norm < 1e-6f) return;
        float cos_now = std::abs(dp.dot(axis)) / dp_norm;
        if (cos_now <= cos_cap) return;

        Eigen::VectorXf c = J_EE.transpose() * axis;
        float c_norm2 = c.squaredNorm();
        if (c_norm2 < 1e-9f) return;

        float coef  = q_step_dir.dot(c) / c_norm2;
        float alpha = (cos_now - cos_cap) / cos_now;
        q_step_dir -= alpha * coef * c;
    }


    // ------------------------------------------------------------------
    // Change 2: goal-proximity fade g
    //
    // Motivation (completeness): the unclamped depth-amplified weight W = 1 - d/envelope widens the
    // rejection cone with penetration depth; if the apple sits against or inside a branch, every
    // approach direction in the last centimeters is "deep and axial" — edge refuse would reject them
    // all and the apple becomes unreachable. Likewise, steering inside the terminal funnel can only
    // produce bend/re-aim limit cycles. Fix: fade linearly with distance to the apple (smooth, no new threshold surface):
    //     g = clamp(dist(p, apple) / R_relax, 0, 1)
    // g=1 (at >= R) reproduces the historical behavior exactly; g->0 (at the apple) retires both
    // rejection and bending, so the terminal approach is constrained only by plain collision checking
    // (rigid obstacles still block as usual). Both mechanisms share the same g: fading only one would
    // create "rejects but no longer guides" or "pointless bending". Disabled when radius <= 0 (g == 1).
    // ------------------------------------------------------------------
    void setGoalFade(const Eigen::Vector3f& center, float radius) {
        goal_fade_center_ = center;
        goal_fade_radius_ = radius;
    }
    float goalFadeAt(const Eigen::Vector3f& p) const {
        if (goal_fade_radius_ <= 0.0f) return 1.0f;
        float d = (p - goal_fade_center_).norm();
        return std::max(0.0f, std::min(1.0f, d / goal_fade_radius_));
    }

    // Query the raw SDF value at a world point (for visualization/debugging)
    float querySdf(const Eigen::Vector3f& p) const {
        if (!sdf_built_) return std::numeric_limits<float>::max();
        return sampleSdf(p);
    }

    // CostEvaluator interface: distance from p to nearest obstacle surface
    float queryDistance(const Eigen::Vector3f& p) const override {
        return querySdf(p);
    }

    bool isSdfBuilt() const { return sdf_built_; }

    // ==================================================================
    // Nav field (plan B) — geodesic distance field sourced at the spherical cap below the apple
    //
    // A second scalar field on the same grid as the SDF: each voxel stores the "travel cost to the cap
    // along a feasible route". Multi-source Dijkstra, seeds = the cap voxels (the whole cap is a static
    // seed, built once). Traversal cost: free space x1; rigid obstacles and the gate's forbidden core are
    // impassable; inside a soft branch's permitted shell there is a "toll" (growing with depth x stiffness
    // — note this is NOT a repulsion: repulsion pushes step by step during planning, whereas the toll is
    // priced once into the route total at build time and only decides which corridor is cheaper).
    // The downhill direction is the attraction that "goes around the canopy and joins the cap from below".
    // ==================================================================
    void buildNavField(const Eigen::Vector3f& apple,
                       float shell_radius, float cap_height,
                       float robot_r = 0.06f) {
        if (!sdf_built_) {
            std::cerr << "[NAV] ERROR: buildSdf must run first\n";
            return;
        }
        auto t0 = std::chrono::steady_clock::now();
        const long long total = (long long)nx_ * ny_ * nz_;
        const float INF = std::numeric_limits<float>::max();

        // 1. Per-voxel traversal cost multiplier (OpenMP parallel, same pattern as buildSdf)
        std::vector<float> mult(total, 1.0f);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int iz = 0; iz < nz_; ++iz) {
            for (int iy = 0; iy < ny_; ++iy) {
                for (int ix = 0; ix < nx_; ++ix) {
                    Eigen::Vector3f p = voxelCenter(ix, iy, iz);
                    float m = 1.0f;
                    for (const auto& cyl : cylinders_) {
                        Eigen::Vector3f seg = cyl.p1 - cyl.p0;
                        float seg_sq = seg.squaredNorm();
                        float u = (seg_sq > 1e-9f)
                            ? std::max(0.0f, std::min(1.0f,
                                  (p - cyl.p0).dot(seg) / seg_sq))
                            : 0.0f;
                        float dist = (p - (cyl.p0 + u * seg)).norm();
                        float pen = (cyl.radius + robot_r) - dist;
                        if (pen <= 0.0f) continue;
                        if (cyl.exclude_from_aniso) { m = -1.0f; break; }  // rigid: impassable
                        float t_arc = cyl.t0 + u * (cyl.t1 - cyl.t0);
                        float allow = penAllowAt(t_arc);
                        if (pen > allow) { m = -1.0f; break; }             // gate forbidden core
                        float stiff = branchStiffness(t_arc, 3.0f, 0.5f) / 3.0f;
                        m += nav_toll_gain_ * stiff * (pen / std::max(allow, 1e-3f));
                    }
                    mult[linearIdx(ix, iy, iz)] = m;
                }
            }
        }

        // 2. Multi-source seeds: the cap lens region (dist <= shell_r and z below the cap's upper edge)
        nav_grid_.assign(total, INF);
        using QE = std::pair<float, int>;
        std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
        float cap_z_max = apple.z() - (shell_radius - cap_height);
        int n_seed = 0;
        for (int iz = 0; iz < nz_; ++iz)
            for (int iy = 0; iy < ny_; ++iy)
                for (int ix = 0; ix < nx_; ++ix) {
                    Eigen::Vector3f p = voxelCenter(ix, iy, iz);
                    if ((p - apple).norm() > shell_radius) continue;
                    if (p.z() > cap_z_max) continue;
                    int id = linearIdx(ix, iy, iz);
                    if (mult[id] < 0.0f) continue;
                    nav_grid_[id] = 0.0f;
                    pq.push({0.0f, id});
                    ++n_seed;
                }

        // 3. 6-connected Dijkstra
        const int NB[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        long long visited = 0;
        while (!pq.empty()) {
            auto [d, id] = pq.top(); pq.pop();
            if (d > nav_grid_[id]) continue;
            ++visited;
            int ix = id % nx_, iy = (id / nx_) % ny_, iz = id / (nx_ * ny_);
            for (const auto& nb : NB) {
                int jx = ix + nb[0], jy = iy + nb[1], jz = iz + nb[2];
                if (jx < 0 || jx >= nx_ || jy < 0 || jy >= ny_ ||
                    jz < 0 || jz >= nz_) continue;
                int jd = linearIdx(jx, jy, jz);
                if (mult[jd] < 0.0f) continue;
                float w = voxel_size_ * 0.5f * (mult[id] + mult[jd]);
                float nd = d + w;
                if (nd < nav_grid_[jd]) {
                    nav_grid_[jd] = nd;
                    pq.push({nd, jd});
                }
            }
        }
        nav_built_ = true;
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "[NAV] field built: seeds=" << n_seed
                  << ", reachable=" << (100.0 * visited / total) << "% of "
                  << total << " voxels, "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";
    }

    bool isNavBuilt() const { return nav_built_; }

    // Geodesic travel cost to the cap (in meter-equivalents); unreachable / outside the field = more than half of +inf
    float queryNavDistance(const Eigen::Vector3f& p) const {
        if (!nav_built_) return std::numeric_limits<float>::max();
        int ix = (int)((p.x() - bbox_min_.x()) / voxel_size_);
        int iy = (int)((p.y() - bbox_min_.y()) / voxel_size_);
        int iz = (int)((p.z() - bbox_min_.z()) / voxel_size_);
        if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ ||
            iz < 0 || iz >= nz_) return std::numeric_limits<float>::max();
        return nav_grid_[linearIdx(ix, iy, iz)];
    }

    // Downhill direction (26-connected, pick the smallest neighbor; axis-quantized, which is smooth
    // enough after the 50% random blend). No lower neighbor / outside the field / not built -> zero vector, caller falls back to a straight line.
    Eigen::Vector3f queryNavDirection(const Eigen::Vector3f& p) const {
        if (!nav_built_) return Eigen::Vector3f::Zero();
        int ix = (int)((p.x() - bbox_min_.x()) / voxel_size_);
        int iy = (int)((p.y() - bbox_min_.y()) / voxel_size_);
        int iz = (int)((p.z() - bbox_min_.z()) / voxel_size_);
        if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ ||
            iz < 0 || iz >= nz_) return Eigen::Vector3f::Zero();
        float best = nav_grid_[linearIdx(ix, iy, iz)];
        Eigen::Vector3f dir = Eigen::Vector3f::Zero();
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy && !dz) continue;
                    int jx = ix + dx, jy = iy + dy, jz = iz + dz;
                    if (jx < 0 || jx >= nx_ || jy < 0 || jy >= ny_ ||
                        jz < 0 || jz >= nz_) continue;
                    float v = nav_grid_[linearIdx(jx, jy, jz)];
                    if (v < best) {
                        best = v;
                        dir = Eigen::Vector3f((float)dx, (float)dy, (float)dz);
                    }
                }
        float n = dir.norm();
        return (n > 1e-6f) ? Eigen::Vector3f(dir / n) : Eigen::Vector3f::Zero();
    }

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
    // Nav-field state (a second scalar field on the same grid as sdf_grid_)
    std::vector<float> nav_grid_;
    bool               nav_built_ = false;
    float              nav_toll_gain_ = 8.0f;  // soft-branch toll gain
    // Change 2: parameters of the goal-proximity fade g (radius <= 0 = disabled)
    Eigen::Vector3f goal_fade_center_ = Eigen::Vector3f::Zero();
    float           goal_fade_radius_ = 0.0f;
    // Custom scoring-sphere mask for tensor edge refuse (empty = default hand+link6)
    std::vector<char> er_sphere_mask_;

    // ---- Soft-branch penetration gate state ----
    struct PenProbe { int fid; Eigen::Vector3f off; float r; };
    bool                  pen_gate_enabled_    = false;
    float                 pen_gate_allow_root_ = 0.015f;  // allow at the root, m
    float                 pen_gate_allow_tip_  = 0.050f;  // allow(t=1), m
    float                 pen_gate_ramp_t0_    = 0.25f;   // ramp start (width of the root plateau)
    Eigen::Vector3f       pen_gate_exempt_center_ = Eigen::Vector3f::Zero();
    float                 pen_gate_exempt_r2_  = 0.0144f; // 0.12² m²
    float                 pen_gate_axis_clear_ = 0.01f;   // radius of the hard forbidden core around the axis (m)
    std::vector<PenProbe> pen_probes_;
    int                   pen_gate_hand_fid_   = -1;      // descent steering uses hand probes only
    // Depth-steering telemetry (same convention as the er_ counters: races allowed under parallelism, telemetry only)
    mutable unsigned long long pen_steer_calls_ = 0;
    mutable unsigned long long pen_steer_bends_ = 0;

    // ---- Unified refuse state ----
    bool  uni_enabled_   = false;
    float uni_env_       = 0.05f;   // pre-contact envelope (m, outside the branch surface)
    float uni_eta_       = 0.5f;    // upper bound of the pre-contact axial weight (approach-shaping strength)
    float uni_ax_root_   = 0.005f;  // slip budget at the root (m/step; a cantilever root allows almost no dragging)
    float uni_ax_tip_    = 0.03f;   // slip budget at the tip (m/step)
    float uni_in_ramp_   = 0.03f;   // ramp thickness of the inward-pressure approach term (m, before contact)
    float uni_safety_    = 0.8f;    // kappa: max fraction of the remaining depth budget consumed per step
    float uni_clamp_     = 0.7f;    // steering clamp target = midpoint of the edge-level sigmoid (budget usage)
    mutable unsigned long long uni_calls_ = 0;
    mutable unsigned long long uni_bends_ = 0;

    // Geometric core of the unified rule: given a probe position/radius and a soft branch, output the
    // axial weight G, the remaining depth budget slack, the outward radial unit vector r_hat, the slip
    // budget L_ax(t) and the inward-pressure activation h_in. Returns false when outside the envelope.
    // Continuity: G vanishes at the envelope edge (the axial term starts smoothly) and h_in rises 0->1
    // within uni_in_ramp_ before contact (the pressure term starts smoothly) — so severity is continuous
    // everywhere and handoffs between branches cannot jump (a structural inheritance of the tensor's anti-oscillation property).
    bool uniPairGeometry(const Eigen::Vector3f& pp, float probe_r,
                         const CylinderObstacle& cyl,
                         float& G, float& slack, Eigen::Vector3f& r_hat,
                         float& ax_budget, float& h_in) const {
        Eigen::Vector3f seg = cyl.p1 - cyl.p0;
        float seg_sq = seg.squaredNorm();
        float u = (seg_sq > 1e-9f)
            ? std::max(0.0f, std::min(1.0f, (pp - cyl.p0).dot(seg) / seg_sq))
            : 0.0f;
        Eigen::Vector3f radial = pp - (cyl.p0 + u * seg);
        float dist = radial.norm();
        if (dist <= 1e-4f) return false;                  // degenerate on the axis, handled by the gate
        float surf = cyl.radius + probe_r;
        float surf_dist = dist - surf;                    // <0 = already penetrating
        if (surf_dist >= uni_env_) return false;          // outside the envelope
        float t_arc = cyl.t0 + u * (cyl.t1 - cyl.t0);
        float allow = penAllowAt(t_arc);
        // Cap by the same hard forbidden core as the gate (so steering/scoring never drift from the validity criterion)
        float allow_cap = surf - pen_gate_axis_clear_ * (1.0f - t_arc);
        if (allow > allow_cap) allow = allow_cap;
        if (allow < 1e-4f) allow = 1e-4f;
        float pen = -surf_dist;
        slack = allow - std::max(pen, 0.0f);
        if (slack < 0.0f) slack = 0.0f;
        // Two-regime G curve: 0->eta before contact (approach from the side), eta->1 after contact (saw penalty)
        G = (pen <= 0.0f)
            ? uni_eta_ * std::min(1.0f, (uni_env_ - surf_dist) / 0.01f)
            : uni_eta_ + (1.0f - uni_eta_)
                  * std::min(1.0f, pen / std::max(allow, 1e-3f));
        // A4: slip budget varies with arc length (root ~0, tip loose; same cantilever shape as allow(t))
        ax_budget = std::max(1e-4f,
            uni_ax_root_ + (uni_ax_tip_ - uni_ax_root_) * t_arc);
        // A5: inward-pressure activation — 0->1 within the pre-contact ramp, constant 1 after contact;
        // free flight far away does not consume the depth budget
        h_in = (pen > 0.0f)
            ? 1.0f
            : std::max(0.0f, 1.0f - surf_dist / uni_in_ramp_);
        r_hat = radial / dist;
        return true;
    }

    // ------------------------------------------------------------------
    // Geometric helper: find the axis unit vector of the cylinder nearest to p
    //   - used by Option A's axial drag: each sphere needs to know "along which axis the nearest branch runs"
    //   - degenerate case (p0 == p1, e.g. the overhead sphere registered as a cylinder) returns 0
    //   - returns 0 when cylinders_ is empty
    //
    // Rigid cylinders (exclude_from_aniso = true, i.e. the trunk) are skipped: anisotropy is a property
    // of compliant foliage, and "push aside radially" is meaningless for a rigid trunk. Every method
    // (edge refuse / sample steering / sticky) already skipped the trunk; skipping here keeps Option A's
    // axial drag and the two anisotropy metrics consistent with them — otherwise the trunk's vertical
    // axis would be taken as a_hat and mix branch-unrelated motion into the axial/radial statistics.
    // ------------------------------------------------------------------
    Eigen::Vector3f findNearestCylinderAxis(const Eigen::Vector3f& p) const {
        if (cylinders_.empty()) return Eigen::Vector3f::Zero();
        int best_idx = -1;
        float best_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < cylinders_.size(); ++i) {
            if (cylinders_[i].exclude_from_aniso) continue;   // rigid obstacles such as the trunk
            float d = distToCylinder(p, cylinders_[i]);
            if (d < best_dist) {
                best_dist = d;
                best_idx = static_cast<int>(i);
            }
        }
        if (best_idx < 0) return Eigen::Vector3f::Zero();
        const auto& cyl = cylinders_[best_idx];
        Eigen::Vector3f axis = cyl.p1 - cyl.p0;
        float len = axis.norm();
        if (len < 1e-6f) return Eigen::Vector3f::Zero();   // degenerate (e.g., sphere-as-cylinder)
        return axis / len;
    }

    // ------------------------------------------------------------------
    // Geometric helper: signed distance from a point to a cylinder (negative inside)
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
        return (p - closest).norm() - cyl.radius;  // negative = inside the cylinder
    }

    // ------------------------------------------------------------------
    // Grid helpers
    // ------------------------------------------------------------------

    // World coordinates of a voxel center
    inline Eigen::Vector3f voxelCenter(int ix, int iy, int iz) const {
        return bbox_min_ + Eigen::Vector3f(ix * voxel_size_,
                                           iy * voxel_size_,
                                           iz * voxel_size_);
    }

    // Flattened 1D index
    inline int linearIdx(int ix, int iy, int iz) const {
        return ix + nx_ * (iy + ny_ * iz);
    }

    // ------------------------------------------------------------------
    // Trilinearly interpolated SDF query
    // Gives a continuous, smooth distance inside the grid; out-of-range points clamp to the nearest boundary voxel.
    // ------------------------------------------------------------------
    float sampleSdf(const Eigen::Vector3f& p) const {
        // Map to continuous grid coordinates
        const float gx = (p.x() - bbox_min_.x()) / voxel_size_;
        const float gy = (p.y() - bbox_min_.y()) / voxel_size_;
        const float gz = (p.z() - bbox_min_.z()) / voxel_size_;

        // Out of range: clamp straight to the nearest voxel (no interpolation, saves an if-else branch)
        const int x0 = std::max(0, std::min(nx_ - 1, (int)gx));
        const int y0 = std::max(0, std::min(ny_ - 1, (int)gy));
        const int z0 = std::max(0, std::min(nz_ - 1, (int)gz));

        // Check whether the point is really within the interpolatable range
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

        // The 8 corner points
        const float c000 = sdf_grid_[linearIdx(x0, y0, z0)];
        const float c100 = sdf_grid_[linearIdx(x1, y0, z0)];
        const float c010 = sdf_grid_[linearIdx(x0, y1, z0)];
        const float c110 = sdf_grid_[linearIdx(x1, y1, z0)];
        const float c001 = sdf_grid_[linearIdx(x0, y0, z1)];
        const float c101 = sdf_grid_[linearIdx(x1, y0, z1)];
        const float c011 = sdf_grid_[linearIdx(x0, y1, z1)];
        const float c111 = sdf_grid_[linearIdx(x1, y1, z1)];

        // Interpolate along X
        const float c00 = c000 * (1 - fx) + c100 * fx;
        const float c01 = c001 * (1 - fx) + c101 * fx;
        const float c10 = c010 * (1 - fx) + c110 * fx;
        const float c11 = c011 * (1 - fx) + c111 * fx;

        // Interpolate along Y
        const float c0 = c00 * (1 - fy) + c10 * fy;
        const float c1 = c01 * (1 - fy) + c11 * fy;

        // Interpolate along Z
        return c0 * (1 - fz) + c1 * fz;
    }
};

} // namespace grasp_rrt
