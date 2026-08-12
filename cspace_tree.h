#pragma once

#include "types.h"
#include "cspace_node.h"
#include "cspace_sampled.h"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <nanoflann.hpp>

namespace grasp_rrt {

// ============================================================================
// CSpaceTree — 对应 Python CSpaceTree
// RRT 树结构: 管理节点的添加、最近邻搜索、路径追溯
//
// NN 搜索使用 nanoflann KD-tree (O(log N))，配置在插入时按 sqrt(metric weights)
// 预缩放，KD-tree 内部用普通 L2 距离即等价于原始加权距离。
// ============================================================================

class CSpaceTree {
public:
    // ---- nanoflann 适配器 ----
    struct CloudAdaptor {
        const std::vector<float>* points = nullptr;   // 扁平存储的预缩放配置 (row-major)
        int dim = 7;

        size_t kdtree_get_point_count() const {
            return points ? points->size() / dim : 0;
        }
        float kdtree_get_pt(size_t idx, size_t d) const {
            return (*points)[idx * dim + d];
        }
        template <class BBOX>
        bool kdtree_get_bbox(BBOX&) const { return false; }
    };

    using KDTreeT = nanoflann::KDTreeSingleIndexDynamicAdaptor<
        nanoflann::L2_Simple_Adaptor<float, CloudAdaptor>,
        CloudAdaptor>;

    // -------------------------------------------------------------------------

    explicit CSpaceTree(CSpaceSampled* cspace)
        : cspace_(cspace), dimension_(cspace->getDimension()), next_id_(0),
          update_children_(false) {
    }

    CSpaceSampled* getCSpace() const { return cspace_; }
    int getDimension() const { return dimension_; }
    int getNrOfNodes() const { return static_cast<int>(nodes_.size()); }

    CSpaceNodePtr getNode(int node_id) const {
        auto it = id_node_map_.find(node_id);
        return (it != id_node_map_.end()) ? it->second : nullptr;
    }

    CSpaceNodePtr getLastAddedNode() const {
        return nodes_.empty() ? nullptr : nodes_.back();
    }

    const std::vector<CSpaceNodePtr>& getNodes() const { return nodes_; }

    void setUpdateChildren(bool b) { update_children_ = b; }

    // ---- 重置 ----

    void reset() {
        nodes_.clear();
        id_node_map_.clear();
        next_id_ = 0;
        scaled_points_.clear();
        kdtree_.reset();
        weights_initialized_ = false;
    }

    // ---- 添加节点 (对应 Python appendNode) ----

    CSpaceNodePtr appendNode(const Config& config, int parent_id) {
        auto node = std::make_shared<CSpaceNode>(next_id_, config, parent_id);
        next_id_++;

        nodes_.push_back(node);
        id_node_map_[node->getId()] = node;

        // 写入 KD-tree
        ensureKDTreeInitialized();
        size_t idx = scaled_points_.size() / dimension_;
        for (int d = 0; d < dimension_; ++d) {
            scaled_points_.push_back(config(d) * sqrt_weights_(d));
        }
        kdtree_->addPoints(idx, idx);

        // 更新父节点的 children 列表
        if (update_children_ && parent_id >= 0) {
            auto parent = getNode(parent_id);
            if (parent) parent->addChild(node);
        }

        return node;
    }

    // ---- 最近邻搜索 (KD-tree O(log N)) ----

    CSpaceNodePtr getNearestNeighbor(const Config& config) const {
        if (nodes_.empty() || !kdtree_) return nullptr;

        std::vector<float> query(dimension_);
        for (int d = 0; d < dimension_; ++d) {
            query[d] = config(d) * sqrt_weights_(d);
        }

        size_t out_idx = 0;
        float out_dist_sq = 0;
        nanoflann::KNNResultSet<float> result(1);
        result.init(&out_idx, &out_dist_sq);
        // nanoflann 新版本把 SearchParams 更名为 SearchParameters (系统装的是新版)
        kdtree_->findNeighbors(result, query.data(), nanoflann::SearchParameters());

        return nodes_[out_idx];
    }

    // 半径搜索目前未被规划主循环调用，保留 Eigen 暴力实现作为回退 (nanoflann 1.4
    // 的 dynamic adaptor 不直接支持 radiusSearch)。
    std::vector<CSpaceNodePtr> getNeighborsWithinRadius(const Config& config, float radius) const {
        std::vector<CSpaceNodePtr> neighbors;
        if (nodes_.empty()) return neighbors;

        float sq_radius = radius * radius;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            Config diff = nodes_[i]->getConfiguration() - config;
            float d2;
            if (cspace_->getUseMetricWeights()) {
                d2 = diff.cwiseProduct(cspace_->getMetricWeights()).squaredNorm();
            } else {
                d2 = diff.squaredNorm();
            }
            if (d2 <= sq_radius) neighbors.push_back(nodes_[i]);
        }
        return neighbors;
    }

    int getNearestNeighborId(const Config& config) const {
        auto nn = getNearestNeighbor(config);
        return nn ? nn->getId() : -1;
    }

    // ---- 路径添加 (对应 Python appendPath) ----

    /**
     * 从 startNode 到 config 添加中间路径节点
     * @return {成功?, 最后添加的节点ID}
     */
    std::pair<bool, int> appendPath(CSpaceNodePtr startNode, const Config& config) {
        ConfigPath path_points = cspace_->createPath(startNode->getConfiguration(), config);
        int last_id = startNode->getId();

        for (size_t i = 1; i < path_points.size(); ++i) {
            auto new_node = appendNode(path_points[i], last_id);

            // RRT*: 更新 cost
            auto parent_node = getNode(last_id);
            if (parent_node) {
                float ec = cspace_->evaluateEdgeCost(parent_node->getConfiguration(), new_node->getConfiguration());
                new_node->setEdgeCostToParent(ec);
                new_node->setCostFromRoot(parent_node->getCostFromRoot() + ec);
            }

            last_id = new_node->getId();
        }
        return {true, last_id};
    }

    /**
     * 对应 Python appendPathUntilCollision
     * @return {是否完整到达?, 最后添加的节点ID}
     */
    std::pair<bool, int> appendPathUntilCollision(CSpaceNodePtr startNode,
                                                   const Config& config) {
        auto [path_points, ratio] = cspace_->createPathUntilInvalid(
            startNode->getConfiguration(), config);

        int last_id = startNode->getId();
        for (size_t i = 1; i < path_points.size(); ++i) {
            auto new_node = appendNode(path_points[i], last_id);

            // RRT*: 更新 cost
            auto parent_node = getNode(last_id);
            if (parent_node) {
                float ec = cspace_->evaluateEdgeCost(parent_node->getConfiguration(), new_node->getConfiguration());
                new_node->setEdgeCostToParent(ec);
                new_node->setCostFromRoot(parent_node->getCostFromRoot() + ec);
            }

            last_id = new_node->getId();
        }
        return {ratio >= 1.0f, last_id};
    }

    // ---- 路径回溯 (对应 Python createPathFromNodes) ----

    ConfigPath createPathFromNodes(CSpaceNodePtr startNode,
                                   CSpaceNodePtr goalNode) const {
        ConfigPath path;
        auto current = goalNode;
        while (current) {
            path.push_back(current->getConfiguration());
            if (current->getId() == startNode->getId()) break;
            current = getNode(current->getParentId());
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

private:
    CSpaceSampled* cspace_;
    int dimension_;
    int next_id_;
    bool update_children_;

    std::vector<CSpaceNodePtr> nodes_;
    std::unordered_map<int, CSpaceNodePtr> id_node_map_;

    // ---- KD-tree 后端 ----
    mutable std::vector<float>           scaled_points_;     // 行主序: idx*dim + d
    mutable Eigen::VectorXf              sqrt_weights_;
    mutable bool                         weights_initialized_ = false;
    mutable CloudAdaptor                 cloud_adaptor_;
    mutable std::unique_ptr<KDTreeT>     kdtree_;

    void ensureKDTreeInitialized() {
        if (kdtree_) return;
        // 初始化 sqrt(weights) — 仅在第一次插入时
        if (cspace_->getUseMetricWeights()) {
            sqrt_weights_ = cspace_->getMetricWeights().cwiseSqrt();
        } else {
            sqrt_weights_ = Eigen::VectorXf::Ones(dimension_);
        }
        weights_initialized_ = true;

        cloud_adaptor_.points = &scaled_points_;
        cloud_adaptor_.dim    = dimension_;
        kdtree_ = std::make_unique<KDTreeT>(
            dimension_,
            cloud_adaptor_,
            nanoflann::KDTreeSingleIndexAdaptorParams(10));
    }
};

} // namespace grasp_rrt
