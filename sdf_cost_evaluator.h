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
        // 本段在所属枝条上的弧长占比区间 [t0, t1](0 = 枝根, 1 = 枝梢)。
        // 用于 branch-stiffness 刚度模型(根部更"硬")。单段枝默认 0..1。
        float t0 = 0.0f;
        float t1 = 1.0f;
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

    // branch-stiffness 刚度模型:软枝上 t 处的刚度系数。
    //   t = 0(枝根,悬臂最硬)→ k_root;t = 1(枝梢,易弯)→ k_tip。
    //   物理动机:真实枝条是悬臂梁,根部位移代价远大于梢部;把
    //   "从中段/根部硬切过去会折断枝条"编码成场强差异。
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

    // 只读访问全部圆柱障碍 (含 rigid 标志)。用于把场景几何导出给
    // Isaac Sim 对齐 (scene_layout.json), 单一出口, 覆盖所有场景变体。
    const std::vector<CylinderObstacle>& getCylinders() const { return cylinders_; }

    // ---- 软枝穿透闸门 (penetration gate) ----
    //
    // 把 benchmark 事后指标 cut_violations 的同一判据装进有效性层:
    //   pen = (r_枝 + r_探针) − dist_轴 ≤ allow(t) = root + (tip−root)·t
    // 探针集与该指标一致 (link3…hand 共 9 个体积化探针)。豁免球
    // (exempt_center, exempt_radius) 内的接触不计 —— 嵌枝苹果的终端
    // 抓取固有接触。allow 单位: 米。
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

    // 闸门判定。恒真当未启用。FK 走 per-thread 机器人状态, 与
    // computeEdgeAxialScore 相同的并行安全模式。
    bool isPenetrationAllowed(const Eigen::VectorXf& q) const override {
        if (!pen_gate_enabled_) return true;
        robot_->setJointValues(q);
        for (const auto& pr : pen_probes_) {
            Eigen::Matrix4f pose = robot_->getFramePoseById(pr.fid);
            Eigen::Vector3f pp = pose.block<3,1>(0,3)
                               + pose.block<3,3>(0,0) * pr.off;
            if ((pp - pen_gate_exempt_center_).squaredNorm() < pen_gate_exempt_r2_)
                continue;   // 苹果近旁豁免
            for (const auto& cyl : cylinders_) {
                if (cyl.exclude_from_aniso) continue;              // 仅软枝
                if (cyl.axis_unit.squaredNorm() < 0.25f) continue; // 退化跳过
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
                float allow = pen_gate_allow_root_
                    + (pen_gate_allow_tip_ - pen_gate_allow_root_) * t_arc;
                // 轴心硬禁区:探针中心永不进入轴线 axis_clear 内 ——
                // allow 封顶为 (R+r−clear),枝梢也不许"贴轴穿过"。
                // 确定性位置规则(接受概率恒 0),概率滤波给不了这种保证。
                float allow_cap = (cyl.radius + pr.r)
                    - pen_gate_axis_clear_ * (1.0f - t_arc);
                if (allow > allow_cap) allow = allow_cap;
                if (pen > allow) return false;
            }
        }
        return true;
    }

    bool isPenetrationGateEnabled() const { return pen_gate_enabled_; }

    // ---- Depth steering(闸门的提议层配套,见 types.h 注释) ----
    //
    // 把 C-space 提议方向投影到许可穿透边界的切向。对每个接触邻域内
    // (体表距接触面 < margin) 的 (探针, 软枝) 对:
    //   g  = Jᵀ·(−r̂)          — 使该探针最快加深的关节方向 (未归一)
    //   v  = step · dir·g      — 本步预测加深量 (m, 一阶线性化)
    //   v > κ·slack 时 dir 减去沿 ĝ 的超出分量 (slack = allow(t) − pen)
    // 逐约束顺序投影 (多接触时为近似)。调用后 dir 未归一化, 由调用方
    // renormalize (与 applySamplingBias 同约定)。返回是否弯折 (遥测)。
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
                if (dist <= 1e-4f) continue;               // 贴轴退化, 闸门处理
                float surf = cyl.radius + pr.r;            // pen = 0 的接触面
                if (dist - surf > margin) continue;        // 不在接触邻域
                float pen = surf - dist;                   // 可为负 (未接触)
                float t_arc = cyl.t0 + u * (cyl.t1 - cyl.t0);
                float allow = pen_gate_allow_root_
                    + (pen_gate_allow_tip_ - pen_gate_allow_root_) * t_arc;
                float slack = allow - std::max(pen, 0.0f);
                if (slack < 0.0f) slack = 0.0f;

                if (!jac_ready) { robot_->prepareJacobians(); jac_ready = true; }
                Eigen::MatrixXf J = robot_->computeSphereJacobianById(pr.fid, pr.off);
                if (J.rows() != 3 || J.cols() != q_step_dir.size()) continue;
                Eigen::VectorXf g = J.transpose() * (-(radial / dist));
                float gn = g.norm();
                if (gn < 1e-6f) continue;

                float v   = step * q_step_dir.dot(g);      // 预测加深 (m)
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

    // 下探阶段的工作空间版:只夹持手部探针 (下探由手领进枝丛;臂上
    // 探针仍由闸门兜底)。robot 须已 setJointValues 到当前构型。
    // dp 为 EE 位置增量;夹掉超出分量后保持原步幅 (切向滑移)。
    bool applyDepthSteeringDescent(Eigen::Vector3f& dp,
                                   float safety, float margin) const {
        if (!pen_gate_enabled_ || pen_probes_.empty()) return false;
        float dpn = dp.norm();
        if (dpn < 1e-9f) return false;
        bool bent = false;
        for (const auto& pr : pen_probes_) {
            if (pr.fid != pen_gate_hand_fid_) continue;    // 仅手部探针
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
                float allow = pen_gate_allow_root_
                    + (pen_gate_allow_tip_ - pen_gate_allow_root_) * t_arc;
                float slack = allow - std::max(pen, 0.0f);
                if (slack < 0.0f) slack = 0.0f;

                Eigen::Vector3f deep = -(radial / dist);   // 加深方向
                float v   = dp.dot(deep);                  // 本步加深量 (m)
                float lim = safety * slack;
                if (v > lim) {
                    dp -= (v - lim) * deep;
                    bent = true;
                }
            }
        }
        if (bent) {
            float bn = dp.norm();
            if (bn > 1e-9f) dp *= dpn / bn;                // 保幅值: 切向滑移
            pen_steer_bends_++;
        }
        return bent;
    }

    // Depth-steering telemetry (cumulative)
    unsigned long long getDepthSteerCalls() const { return pen_steer_calls_; }
    unsigned long long getDepthSteerBends() const { return pen_steer_bends_; }

    // ---- Unified refuse(统一分支交互规则,见 types.h 注释) ----
    //
    // 单一运动锥替代 {tensor edge refuse + sample steering + depth
    // steering}:对接触邻域内每个 (探针, 软枝) 对
    //   severity = v_in / (κ·slack) + G · v_ax / ax_budget
    //   v_in = 步内径向内压 (m), v_ax = 步内轴向滑移 (m)
    //   G    = 预接触 η·(1 − surf/env)(接近整形: 从侧面接近)
    //          接触后 η + (1−η)·min(1, pen/allow)(拖锯惩罚随深度)
    // severity > 1 时把内压/轴滑分量各乘 1/severity(线性 → 收缩后
    // severity = 1),剩余分量 ≈ 环向。需先 enablePenetrationGate。
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

    // C-space 版(RRT 扩展原语):dir 为归一化提议方向,step 为实际
    // 步长。返回是否弯折;弯折后 dir 未归一化,调用方 renormalize。
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
                float s_in = q_step_dir.dot(g_in);   // 内压速率 (m / 单位步)
                float s_ax = q_step_dir.dot(g_ax);   // 轴滑速率 (带符号)
                float v_in = step * std::max(0.0f, s_in);
                float v_ax = step * std::fabs(s_ax);
                float b_in = std::max(uni_safety_ * slack, 1e-4f);
                float sev  = h_in * v_in / b_in + G * v_ax / ax_b;
                if (sev <= uni_clamp_) continue;

                // 收缩两个越权分量到夹持目标(severity 对二者线性)
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

    // 下探版(工作空间):dp 为 EE 位置增量;仅手部探针(下探由手领进,
    // 臂上仍由闸门兜底)。收缩后保持原步幅(环向滑移)。
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

    // 边级 severity(有效性层配套):对整条边 q1→q2 的探针位移求
    // 最大 pair severity。CSpaceSampled 以 sigmoid(gain·(score−1))
    // 概率拒绝(与旧 edge refuse 同管线,阈值移到 1)。
    float computeEdgeUnifiedSeverity(const Eigen::VectorXf& q1,
                                     const Eigen::VectorXf& q2) const override {
        if (!uni_enabled_ || !pen_gate_enabled_ || pen_probes_.empty())
            return 0.0f;
        // 探针位移:q1 → q2 两次 FK
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

        // 3. 逐体素填充最小有符号距离 (OpenMP 并行化, embarrassingly parallel)
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
            // engagement_only 现跟随规划目标球集(erSphereActive):默认
            // hand+link6;--er-links 自定义时按自定义集合计。注意:这使
            // axial_ee 变成"规划器是否保护了自己的目标"的自评指标,
            // 不同球集的 arm 之间不再直接可比。
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
    // 自定义 tensor edge refuse 的计分球集合(实验用)
    //
    // sel = {链接名, 该链接上第几个球(0 起,按 addRobotSphere 顺序)}。
    // 必须在所有 addRobotSphere 之后调用;空集合 = 恢复默认
    // (panda_hand + panda_link6 的全部球)。只影响 tensor 模式;
    // legacy MAX 模式仍固定 hand+link6。
    // ------------------------------------------------------------------
    void setEdgeRefuseSphereSet(
        const std::vector<std::pair<std::string, int>>& sel) {
        if (sel.empty()) { er_sphere_mask_.clear(); return; }
        er_sphere_mask_.assign(spheres_.size(), 0);
        for (size_t k = 0; k < spheres_.size(); ++k) {
            // 该球在其链接上的序号
            int idx = 0;
            for (size_t j = 0; j < k; ++j)
                if (spheres_[j].link_name == spheres_[k].link_name) ++idx;
            for (const auto& s : sel)
                if (s.first == spheres_[k].link_name && s.second == idx)
                    er_sphere_mask_[k] = 1;
        }
    }

    // tensor edge refuse:第 k 个球是否参与计分
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
        // Change 2:hand 球位置,用于末尾的目标邻近衰减 g
        Eigen::Vector3f p_hand = Eigen::Vector3f::Zero();
        bool hand_seen = false;
        for (size_t k = 0; k < spheres_.size(); ++k) {
            const auto& rs = spheres_[k];
            if (!erSphereActive(k)) continue;

            Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
            Eigen::Vector3f p_curr = pose.block<3,1>(0,3)
                                   + pose.block<3,3>(0,0) * rs.offset;
            // 注:自定义球集不含 hand 时,goal fade 不生效(fade 需 hand 位置)
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
        // Change 2:目标邻近衰减 —— 苹果近旁最后几厘米的接近方向由抓取
        // 位姿决定,拒绝只会造成不可达(完备性),按 g 线性退场。
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
    // applySamplingBias — sample steering with K-nearest structure tensor
    //
    // Multi-branch directional bias:
    //   1. (Envelope gate) If EE body is outside envelope of every branch,
    //      do nothing. The bias only operates inside the envelope.
    //   2. Collect all branches with surface-to-EE distance < envelope.
    //   3. Keep only the K nearest (default K = 5). Bounds compute and
    //      gives cleaner directional signal in dense foliage.
    //   4. Assemble the weighted structure tensor
    //          M = Σ wᵢ · âᵢ · âᵢᵀ,    wᵢ = max(0, 1 − distᵢ/envelope)
    //      This is axis-sign-invariant (â âᵀ = (−â)(−â)ᵀ).
    //   5. Eigendecompose M.  Largest eigenvector = dominant axial direction.
    //      Anisotropy ratio (λmax − λmin)/λmax  ∈ [0, 1].
    //      If < min_anisotropy: branches point too randomly, skip bias.
    //   6. Apply the angle-range cap against the dominant axis (Op 3,
    //      same as single-axis version).  Uses J_EE^T projection — no
    //      pseudoinverse.
    //
    // Modifies q_step_dir IN PLACE; caller renormalizes.
    // Returns true iff the bias was actually applied.
    // Per-call cost ≈ 18-25 μs.
    // ------------------------------------------------------------------
    bool applySamplingBias(const Eigen::VectorXf& q_near,
                            Eigen::VectorXf& q_step_dir,
                            const Eigen::MatrixXf& J_EE,
                            float envelope,
                            float cos_cap,
                            int   K              = 5,
                            float min_anisotropy = 0.3f) const {
        // -------------------------------------------------------------
        // Smooth sample steering: structure-tensor multi-branch aggregation +
        // Option 1 (smoothstep envelope ramp) + Option 3 (penetration skip).
        //
        //   1. Envelope conceptually built around each branch (radius
        //      r_cyl + ee_radius + ramp_envelope around cylinder axis).
        //   2. Collect K=5 nearest non-rigid branches in ramp envelope.
        //   3. Build structure tensor M = Σ wᵢ · âᵢ · âᵢᵀ
        //      with smoothstep weights (Option 1).
        //   4. Eigendecompose M; dominant eigenvector = effective axis.
        //   5. Anisotropy gate: skip if geometry too isotropic.
        //   6. Apply angle-range cap against dominant axis.
        //
        // Smoother than edge refuse-style max aggregation: the eigenvector of M
        // changes continuously as branch weights change, avoiding the
        // discontinuity that comes from picking a single "winning" branch.
        // -------------------------------------------------------------
        if (!sdf_built_) return false;
        if (q_near.size() != q_step_dir.size()) return false;
        if (J_EE.rows() != 3 || J_EE.cols() != q_near.size()) return false;
        if (envelope <= 0.0f) return false;
        if (cylinders_.empty()) return false;
        if (K < 1) K = 1;

        // ---- EE reference: panda_hand sphere at offset (0,0,0) ----
        Eigen::Vector3f p_ee = Eigen::Vector3f::Zero();
        float ee_radius = 0.06f;
        bool found_ee = false;
        for (const auto& rs : spheres_) {
            if (rs.link_name == "panda_hand" && rs.offset.norm() < 1e-4f) {
                Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
                p_ee = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * rs.offset;
                ee_radius = rs.radius;
                found_ee = true;
                break;
            }
        }
        if (!found_ee) return false;

        // ---- Option 3: skip when EE is penetrating any cylinder body ----
        {
            float sdf_at_ee = sampleSdf(p_ee);
            if (sdf_at_ee - ee_radius < 0.0f) return false;
        }

        // ---- Option 1: smoothstep envelope ramp ----
        const float ramp_envelope = 1.5f * envelope;

        // ---- Collect candidate branches (envelope as tube around branch) ----
        struct BranchInfo { float body_dist; Eigen::Vector3f axis; };
        std::vector<BranchInfo> nearby;
        nearby.reserve(cylinders_.size());
        for (const auto& cyl : cylinders_) {
            if (cyl.exclude_from_aniso) continue;
            if (cyl.axis_unit.squaredNorm() < 0.25f) continue;
            float d = distToCylinder(p_ee, cyl) - ee_radius;
            if (d > ramp_envelope) continue;
            if (d < 0.0f) continue;               // EE inside cylinder (Option 3)
            nearby.push_back({d, cyl.axis_unit});
        }
        if (nearby.empty()) return false;

        // ---- Keep K nearest ----
        if (static_cast<int>(nearby.size()) > K) {
            std::partial_sort(
                nearby.begin(), nearby.begin() + K, nearby.end(),
                [](const BranchInfo& a, const BranchInfo& b) {
                    return a.body_dist < b.body_dist;
                });
            nearby.resize(K);
        }

        bool any_bias_applied = false;

        // ============================================================
        // Smooth structure-tensor bias (single-stage)
        //   - Builds M = Σ wᵢ · âᵢ · âᵢᵀ from K nearest with smoothstep
        //     weights, eigendecomposes to find dominant axial direction,
        //     applies cap against it.
        //   - Skips if anisotropy < min_anisotropy (no clear axis).
        // ============================================================
        Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
        float total_w = 0.0f;
        for (const auto& b : nearby) {
            float t = std::max(0.0f, std::min(1.0f, b.body_dist / ramp_envelope));
            float w = 1.0f - (t * t * (3.0f - 2.0f * t));
            if (w <= 1e-6f) continue;
            M += w * (b.axis * b.axis.transpose());
            total_w += w;
        }
        if (total_w <= 1e-6f) return false;

        M /= total_w;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(M);
        if (es.info() != Eigen::Success) return false;

        float lambda_max = es.eigenvalues()(2);
        float lambda_min = es.eigenvalues()(0);
        if (lambda_max <= 1e-6f) return false;

        float anisotropy = (lambda_max - lambda_min) / lambda_max;
        if (anisotropy < min_anisotropy) return false;

        Eigen::Vector3f a_dom = es.eigenvectors().col(2);
        Eigen::Vector3f dp = J_EE * q_step_dir;
        float dp_norm = dp.norm();
        if (dp_norm < 1e-6f) return false;

        float cos_now = std::abs(dp.dot(a_dom)) / dp_norm;
        if (cos_now <= cos_cap) return false;

        Eigen::VectorXf c = J_EE.transpose() * a_dom;
        float c_norm2 = c.squaredNorm();
        if (c_norm2 < 1e-9f) return false;

        float coef  = q_step_dir.dot(c) / c_norm2;
        float alpha = (cos_now - cos_cap) / cos_now;
        q_step_dir -= alpha * coef * c;
        any_bias_applied = true;

        return any_bias_applied;
    }

    // ------------------------------------------------------------------
    // applySamplingBiasTensor — sample steering on the structure tensor
    //
    // Fixes the two defects of applySamplingBias (kept above as the
    // legacy / ablation arm):
    //
    //  (1) WRONG EIGENVECTOR. The legacy version caps against the DOMINANT
    //      eigenvector of M. With one branch that equals the branch axis;
    //      with two non-parallel branches lambda_1 ~ lambda_2 ties and the
    //      dominant eigenvector is numerically arbitrary within their span
    //      — the bias steers against a fictitious direction and can itself
    //      oscillate. Here the FULL quadratic form
    //          s(v) = sqrt(v^T M_hat v)
    //      is capped — every branch constrains the motion simultaneously —
    //      and the escape direction is the MINIMUM eigenvector: the
    //      direction least axial to ALL nearby branches at once.
    //
    //  (2) PENETRATION SKIP. The legacy version returns false the moment
    //      the EE touches a branch — switching off exactly where guidance
    //      matters most (and exactly the regime the axial metrics
    //      measure). Here contact simply saturates the smoothstep weight
    //      at 1, same convention as buildLocalAxisTensor / edge refuse.
    //
    // Cap semantics (v3): s_eff(v) = order-8 soft-max of (W_i * |v . a_i|)
    // with the UNCLAMPED legacy weight — identical scoring to tensor edge
    // refuse. Trigger when s_eff > cos_cap; bend toward the minimum
    // eigenvector until s_eff = cos_cap * overshoot (Improvement C).
    // Improvements A (descent steering, see steerWorkspaceDelta) and
    // B (second sphere) are documented at their definitions below.
    //
    // NOTE: near an eigenvalue tie (lambda_min ~ lambda_mid) the minimum
    // eigenvector can rotate quickly. Unlike the argmax flip this is
    // benign: the two candidate directions score EQUALLY (the penalty is
    // flat between them), so motion quality is unaffected either way.
    // ------------------------------------------------------------------
    // ---- shared steering core -------------------------------------------

    struct WAxis { float w; Eigen::Vector3f a; };

    // Collect in-envelope branch axes at a body point. Legacy UNCLAMPED
    // weight W = 1 − d/envelope (> 1 inside contact: depth tightens the
    // cone; same convention as tensor edge refuse). No penetration skip.
    bool collectSteeringAxes(const Eigen::Vector3f& p_body,
                             float body_radius,
                             float envelope,
                             std::vector<WAxis>& axes,
                             Eigen::Matrix3f& M) const {
        axes.clear();
        M.setZero();
        float w_sum = 0.0f;
        for (const auto& cyl : cylinders_) {
            if (cyl.exclude_from_aniso) continue;
            if (cyl.axis_unit.squaredNorm() < 0.25f) continue;
            float d = distToCylinder(p_body, cyl) - body_radius;
            if (d >= envelope) continue;
            float W = 1.0f - d / envelope;   // UNCLAMPED: > 1 inside
            axes.push_back({W, cyl.axis_unit});
            M.noalias() += W * (cyl.axis_unit * cyl.axis_unit.transpose());
            w_sum += W;
        }
        return w_sum > 1e-6f;
    }

    // Order-8 soft-max axial score (identical to tensor edge refuse).
    static float softmaxAxialScore(const std::vector<WAxis>& axes,
                                   const Eigen::Vector3f& u) {
        float acc = 0.0f;
        for (const auto& wa : axes) {
            float s1 = wa.w * std::abs(u.dot(wa.a));
            float s2 = s1 * s1, s4 = s2 * s2;
            acc += s4 * s4;                  // (W c)^8
        }
        return std::sqrt(std::sqrt(std::sqrt(acc)));
    }

    // Bend unit direction v_hat toward the min-eigenvector of M until the
    // soft-max score drops to `target`. Improvement C: target < trigger,
    // so the bent motion lands INSIDE the allowed cone instead of exactly
    // on its boundary (which the minimal bend used to do — systematically
    // producing the least-radial acceptable motion).
    // Returns false when v_hat already scores <= trigger.
    //
    // Change 1(sticky escape sign,可选参数 sticky):特征向量是"线"
    // 不是"箭头"—— ±e_min 得分严格相等,必须由外部规则定向。旧规则
    // 对齐当前提议方向 v_hat,而 v_hat 一半是随机噪声:当提议近乎垂直
    // 于 e_min(恰是转向触发的常态)时,点积符号逐周期抛硬币,连续扩展
    // 被交替弯向枝条两侧 —— 纯粹由平局裁决制造的之字形。修复:sticky
    // 非空且已有记录时,与上次用过的箭头同侧;本次所用箭头写回 *sticky
    // 供子节点/下一步继承。两个符号得分严格相等,粘滞零质量代价 ——
    // 与已废弃的 sticky winner(得分不等选项间的滞回)本质不同。
    bool bendDirection(const std::vector<WAxis>& axes,
                       const Eigen::Matrix3f& M,
                       const Eigen::Vector3f& v_hat,
                       float trigger, float target,
                       Eigen::Vector3f& u_out,
                       Eigen::Vector3f* sticky = nullptr) const {
        if (softmaxAxialScore(axes, v_hat) <= trigger) return false;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(M);
        if (es.info() != Eigen::Success) return false;
        Eigen::Vector3f e_min = es.eigenvectors().col(0);   // ascending
        if (sticky && sticky->squaredNorm() > 0.5f) {
            if (sticky->dot(e_min) < 0.0f) e_min = -e_min;  // 与上次同侧
        } else if (v_hat.dot(e_min) < 0.0f) {
            e_min = -e_min;                                 // stay on v's side
        }
        if (sticky) *sticky = e_min;   // 记录本次箭头(单位向量)

        float beta = 1.0f;
        if (softmaxAxialScore(axes, e_min) < target) {
            float lo = 0.0f, hi = 1.0f;
            for (int it = 0; it < 20; ++it) {
                float mid = 0.5f * (lo + hi);
                Eigen::Vector3f u = (1.0f - mid) * v_hat + mid * e_min;
                float un = u.norm();
                if (un < 1e-9f) { lo = mid; continue; }
                if (softmaxAxialScore(axes, u / un) > target) lo = mid;
                else hi = mid;
            }
            beta = hi;
        }
        Eigen::Vector3f u = (1.0f - beta) * v_hat + beta * e_min;
        float un = u.norm();
        if (un < 1e-9f) return false;
        u_out = u / un;
        return true;
    }

    // Steer the motion of one body point in joint space via its Jacobian.
    // fade ∈ [0,1](Change 2):按目标邻近度缩放弯折量;sticky(Change 1)
    // 透传给 bendDirection;离开包络(无枝条)时清空 sticky 记忆。
    bool steerPointAgainstAxes(const Eigen::Vector3f& p_body,
                               float body_radius,
                               const Eigen::MatrixXf& J_point,
                               Eigen::VectorXf& q_step_dir,
                               float envelope,
                               float trigger, float target,
                               float fade = 1.0f,
                               Eigen::Vector3f* sticky = nullptr) const {
        std::vector<WAxis> axes;
        Eigen::Matrix3f M;
        if (!collectSteeringAxes(p_body, body_radius, envelope, axes, M)) {
            if (sticky) sticky->setZero();   // 出包络:清除箭头记忆
            return false;
        }
        if (fade <= 0.01f) return false;     // 目标近旁:完全不弯折

        Eigen::Vector3f v = J_point * q_step_dir;
        float v_norm = v.norm();
        if (v_norm < 1e-6f) return false;
        Eigen::Vector3f v_hat = v / v_norm;

        Eigen::Vector3f u;
        if (!bendDirection(axes, M, v_hat, trigger, target, u, sticky))
            return false;

        // Damped least-squares mapping back to joint space, preserving the
        // original workspace speed. Caller renormalizes q_step_dir.
        // fade < 1 时按比例缩小修正量(Change 2:靠近目标弯折渐弱)。
        Eigen::Vector3f dv = (u - v_hat) * v_norm * fade;
        Eigen::Matrix3f JJt = J_point * J_point.transpose()
                            + 1e-6f * Eigen::Matrix3f::Identity();
        q_step_dir += J_point.transpose() * JJt.ldlt().solve(dv);
        return true;
    }

    // applySamplingBiasTensor — tensor sample steering, improvements B + C.
    //   B: steers BOTH engagement spheres (panda_hand via the caller's
    //      cached Jacobian, panda_link6 via its own sphere Jacobian) in a
    //      sequential one-pass bend. The two bends can perturb each other
    //      slightly; a fixed-point iteration is not worth the cost since
    //      both pull toward the same min-eigenvector family.
    //   C: bend target = cos_cap * overshoot (default 0.7) — motions land
    //      inside the allowed cone, mimicking the interior-average that
    //      rejection resampling achieves. Trigger stays at cos_cap.
    bool applySamplingBiasTensor(const Eigen::VectorXf& q_near,
                                 Eigen::VectorXf& q_step_dir,
                                 const Eigen::MatrixXf& J_EE,
                                 float envelope,
                                 float cos_cap,
                                 float overshoot = 1.0f,
                                 Eigen::Vector3f* sticky = nullptr) const {
        if (!sdf_built_) return false;
        if (q_near.size() != q_step_dir.size()) return false;
        if (J_EE.rows() != 3 || J_EE.cols() != q_near.size()) return false;
        if (envelope <= 0.0f || cylinders_.empty()) return false;

        const float target =
            cos_cap * std::max(0.05f, std::min(1.0f, overshoot));
        bool applied = false;

        // Sphere 1: panda_hand (caller's cached Jacobian).
        for (const auto& rs : spheres_) {
            if (rs.link_name == "panda_hand" && rs.offset.norm() < 1e-4f) {
                Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
                Eigen::Vector3f p = pose.block<3,1>(0,3)
                                  + pose.block<3,3>(0,0) * rs.offset;
                // Change 2:目标邻近衰减(按 hand 位置计算,两球共用)
                float fade = goalFadeAt(p);
                applied |= steerPointAgainstAxes(
                    p, rs.radius, J_EE, q_step_dir, envelope, cos_cap, target,
                    fade, sticky);
                break;
            }
        }

        // Sphere 2 (Improvement B): panda_link6, own Jacobian.
        for (const auto& rs : spheres_) {
            if (rs.link_name == "panda_link6" && rs.offset.norm() < 1e-4f) {
                Eigen::Matrix4f pose = robot_->getFramePoseById(rs.frame_id);
                Eigen::Vector3f p = pose.block<3,1>(0,3)
                                  + pose.block<3,3>(0,0) * rs.offset;
                grasp_rrt::MatrixXf J6;
                auto panda = dynamic_cast<PandaPinocchioRobot*>(robot_.get());
                if (panda) {
                    J6 = panda->computeSphereJacobianDirect(rs.frame_id,
                                                            rs.offset);
                } else {
                    robot_->prepareJacobians();
                    J6 = robot_->computeSphereJacobianById(rs.frame_id,
                                                           rs.offset);
                }
                float fade = goalFadeAt(p);
                applied |= steerPointAgainstAxes(
                    p, rs.radius, J6, q_step_dir, envelope, cos_cap, target,
                    fade, sticky);
                break;
            }
        }
        return applied;
    }

    // steerWorkspaceDelta — Improvement A: steer the GRASP DESCENT.
    // moveTowardsGoal's DiffIK steps were previously unguided (steering
    // acted only in tree extension), yet the descent produces the deepest
    // contact — exactly where the penetration-weighted metric accumulates.
    // Bends the workspace position delta dp of one descent step. Pure
    // workspace operation, no Jacobian: DiffIK converts afterwards.
    // Goal-progress guard: if the bent step no longer moves toward the
    // grasp target, the ORIGINAL dp is kept — steering yields to task
    // completion, so the descent cannot be stalled by the bias.
    bool steerWorkspaceDelta(const Eigen::Vector3f& p_body,
                             float body_radius,
                             Eigen::Vector3f& dp,
                             float envelope,
                             float cos_cap,
                             float overshoot,
                             const Eigen::Vector3f& goal_dir,
                             Eigen::Vector3f* sticky = nullptr) const {
        if (!sdf_built_ || cylinders_.empty() || envelope <= 0.0f)
            return false;
        float dp_norm = dp.norm();
        if (dp_norm < 1e-9f) return false;

        std::vector<WAxis> axes;
        Eigen::Matrix3f M;
        if (!collectSteeringAxes(p_body, body_radius, envelope, axes, M)) {
            if (sticky) sticky->setZero();   // 出包络:清除箭头记忆
            return false;
        }

        // Change 2:靠近抓取目标时手的行进方向由抓取位姿决定,弯折
        // 无法再产生径向穿越,只会与目标引力形成 bend/re-aim 极限环
        // (下探之字形的主源)。按目标邻近度衰减弯折。
        float fade = goalFadeAt(p_body);
        if (fade <= 0.01f) return false;

        const float target =
            cos_cap * std::max(0.05f, std::min(1.0f, overshoot));
        Eigen::Vector3f v_hat = dp / dp_norm;
        Eigen::Vector3f u;
        if (!bendDirection(axes, M, v_hat, cos_cap, target, u, sticky))
            return false;

        // fade < 1:在原方向与弯折方向之间插值(线性渐变,无阈值面)
        if (fade < 1.0f) {
            Eigen::Vector3f blend = (1.0f - fade) * v_hat + fade * u;
            float bn = blend.norm();
            if (bn < 1e-9f) return false;
            u = blend / bn;
        }

        if (goal_dir.squaredNorm() > 1e-12f && u.dot(goal_dir) <= 0.0f)
            return false;   // would stall the descent — keep original

        dp = u * dp_norm;
        return true;
    }

    // ------------------------------------------------------------------
    // Change 2:目标邻近衰减 g(goal-proximity fade)
    //
    // 动机(完备性):未截断的深度放大权重 W = 1 − d/envelope 使拒绝锥
    // 随穿入深度变宽;若苹果紧贴/藏于枝条,最后几厘米的一切接近方向都
    // 是"深且偏轴向"—— edge refuse 会全部拒绝,苹果不可达。同理,转向
    // 在终端漏斗内只能制造 bend/re-aim 极限环。修复:以到苹果的距离
    // 线性渐变(平滑,无新阈值面):
    //     g = clamp(dist(p, apple) / R_relax, 0, 1)
    // g=1(≥R 处)行为与历史完全一致;g→0(苹果处)拒绝与弯折全部
    // 退场,终端接近只由普通碰撞检查(刚性障碍照常阻挡)约束。
    // 两个机制共用同一个 g:只衰减其一会造成"拒绝但不再引导"或
    // "无谓弯折"的不一致。radius ≤ 0 时禁用(g 恒为 1)。
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

    // ==================================================================
    // Nav field(B 计划)—— 以苹果下方球冠为源的测地距离场
    //
    // 与 SDF 同网格的第二个标量场:每个体素存"沿可行路线到球冠的
    // 行程代价"。多源 Dijkstra,种子 = 球冠体素(整冠静态种子,
    // 单次构建);通行代价:自由 ×1,刚性障碍 + 闸门禁核不可通行,
    // 软枝许可壳层内为"过路费"(随深度×刚度增长 —— 注意这不是斥力:
    // 斥力是规划时逐步的推挤,过路费是建场时一次性算进路线总价,
    // 只影响"哪条走廊更便宜",不推任何东西)。
    // 下坡方向即"绕开树冠、从下方汇入球冠"的吸引方向。
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

        // 1. 每体素通行代价倍率(OpenMP 并行, 与 buildSdf 同模式)
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
                        if (cyl.exclude_from_aniso) { m = -1.0f; break; }  // 刚性: 不可通行
                        float t_arc = cyl.t0 + u * (cyl.t1 - cyl.t0);
                        float allow = pen_gate_allow_root_
                            + (pen_gate_allow_tip_ - pen_gate_allow_root_) * t_arc;
                        if (pen > allow) { m = -1.0f; break; }             // 闸门禁核
                        float stiff = branchStiffness(t_arc, 3.0f, 0.5f) / 3.0f;
                        m += nav_toll_gain_ * stiff * (pen / std::max(allow, 1e-3f));
                    }
                    mult[linearIdx(ix, iy, iz)] = m;
                }
            }
        }

        // 2. 多源种子:球冠透镜区(dist ≤ shell_r 且 z 低于冠上缘)
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

        // 3. 6-邻接 Dijkstra
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

    // 到球冠的测地行程代价 (m 等效);不可达 / 场外 = +∞ 的一半以上
    float queryNavDistance(const Eigen::Vector3f& p) const {
        if (!nav_built_) return std::numeric_limits<float>::max();
        int ix = (int)((p.x() - bbox_min_.x()) / voxel_size_);
        int iy = (int)((p.y() - bbox_min_.y()) / voxel_size_);
        int iz = (int)((p.z() - bbox_min_.z()) / voxel_size_);
        if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ ||
            iz < 0 || iz >= nz_) return std::numeric_limits<float>::max();
        return nav_grid_[linearIdx(ix, iy, iz)];
    }

    // 下坡方向(26 邻接取最小值邻居;轴量化方向,与 50% 随机混合后
    // 足够平滑)。无更低邻居 / 场外 / 未建 → 零向量,调用方回退直线。
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
    // Nav field 状态(与 sdf_grid_ 同网格的第二标量场)
    std::vector<float> nav_grid_;
    bool               nav_built_ = false;
    float              nav_toll_gain_ = 8.0f;  // 软枝过路费增益
    // Change 2:目标邻近衰减 g 的参数(radius ≤ 0 = 禁用)
    Eigen::Vector3f goal_fade_center_ = Eigen::Vector3f::Zero();
    float           goal_fade_radius_ = 0.0f;
    // tensor edge refuse 自定义计分球掩码(空 = 默认 hand+link6)
    std::vector<char> er_sphere_mask_;

    // ---- 软枝穿透闸门状态 ----
    struct PenProbe { int fid; Eigen::Vector3f off; float r; };
    bool                  pen_gate_enabled_    = false;
    float                 pen_gate_allow_root_ = 0.015f;  // allow(t=0), m
    float                 pen_gate_allow_tip_  = 0.075f;  // allow(t=1), m
    Eigen::Vector3f       pen_gate_exempt_center_ = Eigen::Vector3f::Zero();
    float                 pen_gate_exempt_r2_  = 0.0144f; // 0.12² m²
    float                 pen_gate_axis_clear_ = 0.01f;   // 轴心硬禁区半径 (m)
    std::vector<PenProbe> pen_probes_;
    int                   pen_gate_hand_fid_   = -1;      // 下探转向只用手部探针
    // Depth-steering telemetry (与 er_ 计数同款: 并行下允许竞态, 仅遥测)
    mutable unsigned long long pen_steer_calls_ = 0;
    mutable unsigned long long pen_steer_bends_ = 0;

    // ---- Unified refuse 状态 ----
    bool  uni_enabled_   = false;
    float uni_env_       = 0.05f;   // 预接触包络 (m, 枝面以外)
    float uni_eta_       = 0.5f;    // 预接触轴向权重上限 (接近整形强度)
    float uni_ax_root_   = 0.005f;  // 枝根滑移预算 (m/步, 悬臂根部≈不许拖)
    float uni_ax_tip_    = 0.03f;   // 枝梢滑移预算 (m/步)
    float uni_in_ramp_   = 0.03f;   // 内压项接近性 ramp 厚度 (m, 触前)
    float uni_safety_    = 0.8f;    // κ: 每步最多消耗剩余深度预算的比例
    float uni_clamp_     = 0.7f;    // 转向夹持目标 = 边级 sigmoid 中点 (预算占用)
    mutable unsigned long long uni_calls_ = 0;
    mutable unsigned long long uni_bends_ = 0;

    // 统一规则的几何核心:给定探针位置/半径与软枝,输出轴向权重 G、
    // 剩余深度预算 slack、外向径向单位向量 r̂、滑移预算 L_ax(t)、
    // 内压激活 h_in。不在包络内返回 false。
    // 连续性:G 在包络边缘归零(轴向项平滑启动),h_in 在触前
    // uni_in_ramp_ 内 0→1(内压项平滑启动)—— severity 处处连续,
    // 枝间交接不会跳变(tensor 反振荡性质的结构化继承)。
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
        if (dist <= 1e-4f) return false;                  // 贴轴退化, 闸门处理
        float surf = cyl.radius + probe_r;
        float surf_dist = dist - surf;                    // <0 = 已穿透
        if (surf_dist >= uni_env_) return false;          // 包络外
        float t_arc = cyl.t0 + u * (cyl.t1 - cyl.t0);
        float allow = pen_gate_allow_root_
            + (pen_gate_allow_tip_ - pen_gate_allow_root_) * t_arc;
        // 与闸门一致的轴心硬禁区封顶(转向/评分与有效性判据不脱节)
        float allow_cap = surf - pen_gate_axis_clear_ * (1.0f - t_arc);
        if (allow > allow_cap) allow = allow_cap;
        if (allow < 1e-4f) allow = 1e-4f;
        float pen = -surf_dist;
        slack = allow - std::max(pen, 0.0f);
        if (slack < 0.0f) slack = 0.0f;
        // G 双段曲线:预接触 0→η(从侧面接近),接触后 η→1(拖锯惩罚)
        G = (pen <= 0.0f)
            ? uni_eta_ * std::min(1.0f, (uni_env_ - surf_dist) / 0.01f)
            : uni_eta_ + (1.0f - uni_eta_)
                  * std::min(1.0f, pen / std::max(allow, 1e-3f));
        // A4: 滑移预算随弧长(根≈0 梢宽松,与 allow(t) 同形的悬臂模型)
        ax_budget = std::max(1e-4f,
            uni_ax_root_ + (uni_ax_tip_ - uni_ax_root_) * t_arc);
        // A5: 内压激活 —— 触前 ramp 内 0→1,接触后恒 1;远处自由飞行
        // 不计深度预算
        h_in = (pen > 0.0f)
            ? 1.0f
            : std::max(0.0f, 1.0f - surf_dist / uni_in_ramp_);
        r_hat = radial / dist;
        return true;
    }

    // ------------------------------------------------------------------
    // 几何辅助：找到离 p 最近的圆柱的轴向单位向量
    //   - 用于 Option A 的轴向 drag：需要每个 sphere 知道"最近树枝沿哪条轴"
    //   - 退化情况（p0 == p1，例如 overhead 球被当 cylinder 加进来）返回 0
    //   - 当 cylinders_ 为空时返回 0
    //
    // 刚性圆柱（exclude_from_aniso = true，即树干）被跳过：各向异性是柔性
    // 枝叶的属性，"径向拨开"对刚性树干无意义。所有方法（edge refuse /
    // sample steering / sticky）本就跳过树干；此处的跳过让 Option A 的
    // 轴向 drag 与两个 anisotropy 指标与之保持一致 —— 否则树干的竖直轴会
    // 被当成 â，把与树枝无关的运动混入 axial/radial 统计。
    // ------------------------------------------------------------------
    Eigen::Vector3f findNearestCylinderAxis(const Eigen::Vector3f& p) const {
        if (cylinders_.empty()) return Eigen::Vector3f::Zero();
        int best_idx = -1;
        float best_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < cylinders_.size(); ++i) {
            if (cylinders_[i].exclude_from_aniso) continue;   // 树干等刚性障碍
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
