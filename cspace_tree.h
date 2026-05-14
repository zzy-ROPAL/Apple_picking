#pragma once

#include "types.h"
#include "cspace_node.h"
#include "cspace_sampled.h"
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace grasp_rrt {

// ============================================================================
// CSpaceTree — 对应 Python CSpaceTree
// RRT 树结构: 管理节点的添加、最近邻搜索、路径追溯
// ============================================================================

class CSpaceTree {
public:
    explicit CSpaceTree(CSpaceSampled* cspace)
        : cspace_(cspace), dimension_(cspace->getDimension()), next_id_(0),
          update_children_(false) {
        // 预分配配置矩阵 (用于向量化最近邻搜索)
        configs_matrix_.resize(512, dimension_);
        n_configs_ = 0;
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
        configs_matrix_.resize(512, dimension_);
        n_configs_ = 0;
    }

    // ---- 添加节点 (对应 Python appendNode) ----

    CSpaceNodePtr appendNode(const Config& config, int parent_id) {
        auto node = std::make_shared<CSpaceNode>(next_id_, config, parent_id);
        next_id_++;

        nodes_.push_back(node);
        id_node_map_[node->getId()] = node;

        // 写入预分配矩阵
        if (n_configs_ >= configs_matrix_.rows()) {
            int new_cap = configs_matrix_.rows() * 2;
            configs_matrix_.conservativeResize(new_cap, dimension_);
        }
        configs_matrix_.row(n_configs_) = config.transpose();
        n_configs_++;

        // 更新父节点的 children 列表
        if (update_children_ && parent_id >= 0) {
            auto parent = getNode(parent_id);
            if (parent) parent->addChild(node);
        }

        return node;
    }

    // ---- 最近邻搜索 (对应 Python getNearestNeighbor) ----
    // 使用预分配 Eigen 矩阵做向量化距离计算

    CSpaceNodePtr getNearestNeighbor(const Config& config) const {
        if (nodes_.empty()) return nullptr;

        // 向量化距离计算
        Eigen::MatrixXf data = configs_matrix_.topRows(n_configs_);
        Eigen::MatrixXf diff = data.rowwise() - config.transpose();

        Eigen::VectorXf sq_dists;
        if (cspace_->getUseMetricWeights()) {
            // 加权距离
            Eigen::VectorXf w = cspace_->getMetricWeights();
            sq_dists = (diff.array().colwise() * w.array().transpose().matrix().row(0).transpose().array())
                        .square().rowwise().sum();
        } else {
            sq_dists = diff.rowwise().squaredNorm();
        }

        int best_idx;
        sq_dists.minCoeff(&best_idx);
        return nodes_[best_idx];
    }

    std::vector<CSpaceNodePtr> getNeighborsWithinRadius(const Config& config, float radius) const {
        std::vector<CSpaceNodePtr> neighbors;
        if (nodes_.empty()) return neighbors;

        Eigen::MatrixXf data = configs_matrix_.topRows(n_configs_);
        Eigen::MatrixXf diff = data.rowwise() - config.transpose();

        Eigen::VectorXf sq_dists;
        if (cspace_->getUseMetricWeights()) {
            Eigen::VectorXf w = cspace_->getMetricWeights();
            sq_dists = (diff.array().colwise() * w.array().transpose().matrix().row(0).transpose().array())
                        .square().rowwise().sum();
        } else {
            sq_dists = diff.rowwise().squaredNorm();
        }

        float sq_radius = radius * radius;
        for (int i = 0; i < n_configs_; ++i) {
            if (sq_dists(i) <= sq_radius) {
                neighbors.push_back(nodes_[i]);
            }
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

    // 预分配矩阵用于向量化最近邻
    Eigen::MatrixXf configs_matrix_;
    int n_configs_;
};

} // namespace grasp_rrt
