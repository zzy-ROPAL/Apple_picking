#pragma once

#include "types.h"
#include "cspace_node.h"
#include <vector>
#include <unordered_map>
#include <cmath>
#include <map>
#include <random>
#include <algorithm>

namespace grasp_rrt {

// ============================================================================
// ApproachDiscretization — 对应 Python ApproachDiscretization
// 离散球面: 将接近方向映射到正二十面体面, 保证 RRT 节点的接近方向均匀分布
// ============================================================================

class ApproachDiscretization {
public:
    struct FaceMapping {
        int count = 0;
        std::vector<CSpaceNodePtr> cspace_nodes;
    };

    struct FaceStats {
        int  attempts  = 0;     // 该面被选中尝试 grasp 的次数
        int  failures  = 0;     // 失败次数 (TRAPPED + ERROR)
        bool disabled  = false; // 是否被禁用
    };

    /**
     * @param radius 球面半径
     * @param steps 细分次数 (0 = 基础20面体, 1 = 80面, 2 = 320面...)
     */
    ApproachDiscretization(float radius = 100.0f, int steps = 0)
        : radius_(radius), steps_(steps),
          global_pose_(Matrix4f::Identity()),
          rng_(std::random_device{}()) {
        generateIcosphere(steps);
        // 缩放到指定半径
        for (auto& v : vertices_) v *= radius;
    }

    // ---- 全局位姿 ----

    void setGlobalPose(const Matrix4f& pose) { global_pose_ = pose; }
    const Matrix4f& getGlobalPose() const { return global_pose_; }

    // ---- 球面面 ID 查找 ----

    /**
     * 对应 Python getNearestFaceId
     * 找到球面上离给定位置方向最近的面 ID
     */
    int getNearestFaceId(const Vector3f& pos) const {
        Vector3f center = global_pose_.block<3,1>(0, 3);
        Vector3f direction = pos - center;
        float norm = direction.norm();
        if (norm < 1e-10f) return 0;
        direction /= norm;

        int best_face = 0;
        float best_dot = -std::numeric_limits<float>::max();

        for (size_t i = 0; i < faces_.size(); ++i) {
            Vector3f face_center = (vertices_[faces_[i][0]] +
                                    vertices_[faces_[i][1]] +
                                    vertices_[faces_[i][2]]) / 3.0f;
            float fc_norm = face_center.norm();
            if (fc_norm > 1e-10f) face_center /= fc_norm;

            float dot = direction.dot(face_center);
            if (dot > best_dot) {
                best_dot = dot;
                best_face = static_cast<int>(i);
            }
        }
        return best_face;
    }

    // ---- 节点管理 ----

    /**
     * 对应 Python addCSpaceNode
     * 将 RRT 节点添加到对应球面方向
     */
    void addCSpaceNode(const Vector3f& cart_pos, CSpaceNodePtr node) {
        int face_id = getNearestFaceId(cart_pos);
        auto& mapping = face_to_nodes_[face_id];
        if (mapping.count == 0) {
            active_faces_.push_back(face_id);
        }
        mapping.count++;
        mapping.cspace_nodes.push_back(node);
    }

    /**
     * 从 active 面 (过滤 disabled) 中均匀随机选择一个面, 取出一个节点并移除
     */
    CSpaceNodePtr getGoodRatedNode(int /*loops*/ = 30) {
        if (active_faces_.empty()) return nullptr;

        // 过滤 disabled 面
        std::vector<int> candidates;
        candidates.reserve(active_faces_.size());
        for (int f : active_faces_) {
            auto sit = face_stats_.find(f);
            if (sit == face_stats_.end() || !sit->second.disabled) {
                candidates.push_back(f);
            }
        }
        if (candidates.empty()) return nullptr;

        // 均匀随机选一个
        std::uniform_int_distribution<int> dist(0, static_cast<int>(candidates.size()) - 1);
        int chosen_face = candidates[dist(rng_)];
        last_selected_face_ = chosen_face;

        auto it = face_to_nodes_.find(chosen_face);
        if (it == face_to_nodes_.end() || it->second.cspace_nodes.empty()) {
            return nullptr;
        }

        auto& mapping = it->second;
        CSpaceNodePtr node = mapping.cspace_nodes.back();
        mapping.cspace_nodes.pop_back();
        mapping.count--;

        if (mapping.count <= 0) {
            active_faces_.erase(
                std::remove(active_faces_.begin(), active_faces_.end(), chosen_face),
                active_faces_.end());
        }
        return node;
    }

    // ---- 面禁用机制 ----

    /** 上次 getGoodRatedNode 选中的面 ID (-1 表示无效) */
    int getLastSelectedFace() const { return last_selected_face_; }

    /** 记录某面一次 grasp 尝试结果 */
    void recordGraspAttempt(int face_id, bool success) {
        if (face_id < 0) return;
        auto& s = face_stats_[face_id];
        s.attempts++;
        if (!success) s.failures++;
    }

    /** 禁用某面 (永久) */
    void disableFace(int face_id) {
        if (face_id < 0) return;
        face_stats_[face_id].disabled = true;
    }

    bool isFaceDisabled(int face_id) const {
        auto it = face_stats_.find(face_id);
        return it != face_stats_.end() && it->second.disabled;
    }

    int getFailureCount(int face_id) const {
        auto it = face_stats_.find(face_id);
        return it != face_stats_.end() ? it->second.failures : 0;
    }

    /** 当前未被禁用的 active 面数量 */
    int getActiveFaceCount() const {
        int count = 0;
        for (int f : active_faces_) {
            auto it = face_stats_.find(f);
            if (it == face_stats_.end() || !it->second.disabled) ++count;
        }
        return count;
    }

    int getDisabledFaceCount() const {
        int count = 0;
        for (const auto& kv : face_stats_) {
            if (kv.second.disabled) ++count;
        }
        return count;
    }

    /** 至少被尝试过一次的面数（attempts > 0） */
    int getFacesWithAttempts() const {
        int count = 0;
        for (const auto& kv : face_stats_) {
            if (kv.second.attempts > 0) ++count;
        }
        return count;
    }

    /** 单个面累积的最高失败次数 */
    int getMaxFaceFailures() const {
        int max_f = 0;
        for (const auto& kv : face_stats_) {
            if (kv.second.failures > max_f) max_f = kv.second.failures;
        }
        return max_f;
    }

    /** 总三角面数（icosphere subdivision 后的总数） */
    int getTotalFaceCount() const { return static_cast<int>(faces_.size()); }

    void clearCSpaceNodeMapping() {
        face_to_nodes_.clear();
        active_faces_.clear();
        face_stats_.clear();
        last_selected_face_ = -1;
    }

private:
    float radius_;
    int steps_;
    Matrix4f global_pose_;
    std::mt19937 rng_;

    std::vector<Vector3f> vertices_;
    std::vector<std::array<int, 3>> faces_;

    std::unordered_map<int, FaceMapping> face_to_nodes_;
    std::vector<int> active_faces_;

    // 面禁用机制
    std::unordered_map<int, FaceStats> face_stats_;
    int last_selected_face_ = -1;

    // ---- 正二十面体生成与细分 ----

    void generateIcosphere(int steps) {
        // 初始正二十面体 12 个顶点
        float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
        vertices_ = {
            Vector3f(-1, t, 0).normalized(), Vector3f(1, t, 0).normalized(),
            Vector3f(-1,-t, 0).normalized(), Vector3f(1,-t, 0).normalized(),
            Vector3f(0,-1, t).normalized(),  Vector3f(0, 1, t).normalized(),
            Vector3f(0,-1,-t).normalized(),  Vector3f(0, 1,-t).normalized(),
            Vector3f(t, 0,-1).normalized(),  Vector3f(t, 0, 1).normalized(),
            Vector3f(-t,0,-1).normalized(),  Vector3f(-t,0, 1).normalized()
        };

        // 20 个三角面
        faces_ = {
            {0,11,5}, {0,5,1}, {0,1,7}, {0,7,10}, {0,10,11},
            {1,5,9},  {5,11,4},{11,10,2},{10,7,6}, {7,1,8},
            {3,9,4},  {3,4,2}, {3,2,6}, {3,6,8},  {3,8,9},
            {4,9,5},  {2,4,11},{6,2,10},{8,6,7},  {9,8,1}
        };

        // 细分
        for (int s = 0; s < steps; ++s) {
            std::vector<std::array<int, 3>> new_faces;
            std::map<std::pair<int,int>, int> mid_cache;

            for (const auto& f : faces_) {
                int mids[3];
                int edges[3][2] = {{f[0],f[1]}, {f[1],f[2]}, {f[2],f[0]}};
                for (int e = 0; e < 3; ++e) {
                    auto key = std::make_pair(std::min(edges[e][0], edges[e][1]),
                                              std::max(edges[e][0], edges[e][1]));
                    auto it = mid_cache.find(key);
                    if (it == mid_cache.end()) {
                        Vector3f mid = (vertices_[edges[e][0]] + vertices_[edges[e][1]]).normalized();
                        mids[e] = static_cast<int>(vertices_.size());
                        vertices_.push_back(mid);
                        mid_cache[key] = mids[e];
                    } else {
                        mids[e] = it->second;
                    }
                }
                new_faces.push_back({f[0], mids[0], mids[2]});
                new_faces.push_back({f[1], mids[1], mids[0]});
                new_faces.push_back({f[2], mids[2], mids[1]});
                new_faces.push_back({mids[0], mids[1], mids[2]});
            }
            faces_ = new_faces;
        }
    }
};

} // namespace grasp_rrt
