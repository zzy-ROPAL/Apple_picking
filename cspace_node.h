#pragma once

#include "types.h"
#include <vector>
#include <memory>

namespace grasp_rrt {

// ============================================================================
// CSpaceNode — 对应 Python CSpaceNode
// RRT 树中的一个节点, 存储关节配置向量
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

    // ---- 访问器 ----
    int getId() const { return id_; }
    int getParentId() const { return parent_id_; }
    const Config& getConfiguration() const { return configuration_; }
    int getStatus() const { return status_; }
    float getObstacleDistance() const { return obstacle_distance_; }
    const std::vector<std::shared_ptr<CSpaceNode>>& getChildren() const { return children_; }
    
    // RRT* Cost
    float getCostFromRoot() const { return cost_from_root_; }
    float getEdgeCostToParent() const { return edge_cost_to_parent_; }

    // Sticky-winner state: identifier of the "most dangerous branch" that was
    // being avoided when this node was created. -1 means no branch was in scope.
    // Propagates from parent to child; hysteresis at the extension primitive
    // decides when to switch the winner. See applySamplingBias in
    // sdf_cost_evaluator.h for use.
    int  getWinnerBranchId() const { return winner_branch_id_; }
    void setWinnerBranchId(int id) { winner_branch_id_ = id; }

    // ---- 修改器 ----
    void setId(int id) { id_ = id; }
    void setParentId(int pid) { parent_id_ = pid; }
    void setConfiguration(const Config& c) { configuration_ = c; }
    void setStatus(int s) { status_ = s; }             // 0=普通, 2=由moveArm创建
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

    // ---- 距离计算 ----
    float distanceTo(const Config& other) const {
        return (configuration_ - other).norm();
    }

    float distanceTo(const CSpaceNode& other) const {
        return (configuration_ - other.configuration_).norm();
    }

private:
    int    id_;
    int    parent_id_;            // -1 表示根节点
    Config configuration_;        // 关节配置向量
    int    status_;               // 0=普通, 2=由moveArm创建
    float  obstacle_distance_;    // 到障碍物的工作空间距离
    float  dyn_dom_radius_;       // 动态域 RRT 的半径
    
    float  cost_from_root_;       // RRT*: 从根节点到这里的总累计代价
    float  edge_cost_to_parent_;  // RRT*: 与父节点的边代价

    int    winner_branch_id_ = -1;  // sticky-winner: id of most-dangerous branch at creation

    std::vector<std::shared_ptr<CSpaceNode>> children_;
};

using CSpaceNodePtr = std::shared_ptr<CSpaceNode>;

} // namespace grasp_rrt
