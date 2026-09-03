#pragma once

#include "types.h"
#include <vector>
#include <memory>

namespace grasp_rrt {

// ============================================================================
// CSpaceNode — counterpart of the Python CSpaceNode
// A single node of the RRT tree, storing a joint configuration vector
// ============================================================================

class CSpaceNode {
public:
    CSpaceNode()
        : id_(-1), parent_id_(-1), status_(0),
          obstacle_distance_(-1.0f), dyn_dom_radius_(0.0f) {}

    CSpaceNode(int id, const Config& config, int parent_id = -1)
        : id_(id), parent_id_(parent_id), configuration_(config),
          status_(0), obstacle_distance_(-1.0f), dyn_dom_radius_(0.0f),
          cost_from_root_(0.0f), edge_cost_to_parent_(0.0f) {}

    // ---- Accessors ----
    int getId() const { return id_; }
    int getParentId() const { return parent_id_; }
    const Config& getConfiguration() const { return configuration_; }
    int getStatus() const { return status_; }
    float getObstacleDistance() const { return obstacle_distance_; }
    const std::vector<std::shared_ptr<CSpaceNode>>& getChildren() const { return children_; }
    
    // RRT* Cost
    float getCostFromRoot() const { return cost_from_root_; }
    float getEdgeCostToParent() const { return edge_cost_to_parent_; }

    // ---- Mutators ----
    void setId(int id) { id_ = id; }
    void setParentId(int pid) { parent_id_ = pid; }
    void setConfiguration(const Config& c) { configuration_ = c; }
    void setStatus(int s) { status_ = s; }             // 0=normal, 2=created by moveArm
    void setObstacleDistance(float d) { obstacle_distance_ = d; }
    
    void setCostFromRoot(float c) { cost_from_root_ = c; }
    void setEdgeCostToParent(float c) { edge_cost_to_parent_ = c; }

    void addChild(std::shared_ptr<CSpaceNode> child) {
        children_.push_back(child);
    }
    
    void removeChild(int child_id) {
        for (auto it = children_.begin(); it != children_.end(); ++it) {
            if ((*it)->getId() == child_id) {
                children_.erase(it);
                break;
            }
        }
    }

    // ---- Distance computation ----
    float distanceTo(const Config& other) const {
        return (configuration_ - other).norm();
    }

    float distanceTo(const CSpaceNode& other) const {
        return (configuration_ - other.configuration_).norm();
    }

private:
    int    id_;
    int    parent_id_;            // -1 marks the root node
    Config configuration_;        // joint configuration vector
    int    status_;               // 0=normal, 2=created by moveArm
    float  obstacle_distance_;    // workspace distance to the nearest obstacle
    float  dyn_dom_radius_;       // radius for dynamic-domain RRT
    
    float  cost_from_root_;       // RRT*: accumulated cost from the root to here
    float  edge_cost_to_parent_;  // RRT*: edge cost to the parent

    std::vector<std::shared_ptr<CSpaceNode>> children_;
};

using CSpaceNodePtr = std::shared_ptr<CSpaceNode>;

} // namespace grasp_rrt
