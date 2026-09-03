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
// CSpaceTree — counterpart of the Python CSpaceTree
// RRT tree structure: node insertion, nearest-neighbor search, path backtracking
//
// NN search uses a nanoflann KD-tree (O(log N)). Configurations are pre-scaled by
// sqrt(metric weights) on insert, so the tree's plain L2 equals the weighted distance.
// ============================================================================

class CSpaceTree {
public:
    // ---- nanoflann adaptor ----
    struct CloudAdaptor {
        const std::vector<float>* points = nullptr;   // flat storage of pre-scaled configs (row-major)
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

    // ---- Reset ----

    void reset() {
        nodes_.clear();
        id_node_map_.clear();
        next_id_ = 0;
        scaled_points_.clear();
        kdtree_.reset();
        weights_initialized_ = false;
    }

    // ---- Node insertion (Python appendNode) ----

    CSpaceNodePtr appendNode(const Config& config, int parent_id) {
        auto node = std::make_shared<CSpaceNode>(next_id_, config, parent_id);
        next_id_++;

        nodes_.push_back(node);
        id_node_map_[node->getId()] = node;

        // Write into the KD-tree
        ensureKDTreeInitialized();
        size_t idx = scaled_points_.size() / dimension_;
        for (int d = 0; d < dimension_; ++d) {
            scaled_points_.push_back(config(d) * sqrt_weights_(d));
        }
        kdtree_->addPoints(idx, idx);

        // Update the parent's children list
        if (update_children_ && parent_id >= 0) {
            auto parent = getNode(parent_id);
            if (parent) parent->addChild(node);
        }

        return node;
    }

    // ---- Nearest-neighbor search (KD-tree, O(log N)) ----

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
        // Newer nanoflann renamed SearchParams to SearchParameters (the system one is new)
        kdtree_->findNeighbors(result, query.data(), nanoflann::SearchParameters());

        return nodes_[out_idx];
    }

    // Radius search is currently never called by the planning main loop; the brute-force
    // Eigen implementation is kept as a fallback (nanoflann 1.4's dynamic adaptor has no radiusSearch).
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

    // ---- Path insertion (Python appendPath) ----

    /**
     * Insert intermediate path nodes from startNode toward config
     * @return {success?, id of the last inserted node}
     */
    std::pair<bool, int> appendPath(CSpaceNodePtr startNode, const Config& config) {
        ConfigPath path_points = cspace_->createPath(startNode->getConfiguration(), config);
        int last_id = startNode->getId();

        for (size_t i = 1; i < path_points.size(); ++i) {
            auto new_node = appendNode(path_points[i], last_id);

            // RRT*: update cost
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
     * Counterpart of the Python appendPathUntilCollision
     * @return {reached the end?, id of the last inserted node}
     */
    std::pair<bool, int> appendPathUntilCollision(CSpaceNodePtr startNode,
                                                   const Config& config) {
        auto [path_points, ratio] = cspace_->createPathUntilInvalid(
            startNode->getConfiguration(), config);

        int last_id = startNode->getId();
        for (size_t i = 1; i < path_points.size(); ++i) {
            auto new_node = appendNode(path_points[i], last_id);

            // RRT*: update cost
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

    // ---- Path backtracking (Python createPathFromNodes) ----

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

    // ---- KD-tree backend ----
    mutable std::vector<float>           scaled_points_;     // row-major: idx*dim + d
    mutable Eigen::VectorXf              sqrt_weights_;
    mutable bool                         weights_initialized_ = false;
    mutable CloudAdaptor                 cloud_adaptor_;
    mutable std::unique_ptr<KDTreeT>     kdtree_;

    void ensureKDTreeInitialized() {
        if (kdtree_) return;
        // Initialize sqrt(weights) — only on the first insertion
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
