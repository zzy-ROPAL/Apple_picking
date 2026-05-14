#pragma once

#include "robot_interface.h"

namespace grasp_rrt {

// ============================================================================
// SimpleTargetObject — 球形目标物体的简单实现
// 适用于苹果等近似球形的抓取目标
// ============================================================================

class SimpleTargetObject : public TargetObjectInterface {
public:
    /**
     * @param position  物体中心位置 (全局坐标)
     * @param radius    物体半径
     */
    SimpleTargetObject(const Vector3f& position, float radius = 0.04f)
        : position_(position), radius_(radius) {
        pose_ = Matrix4f::Identity();
        pose_.block<3,1>(0, 3) = position;
    }

    Matrix4f getGlobalPose() const override {
        return pose_;
    }

    Vector3f getPosition() const override {
        return position_;
    }

    /**
     * 球面上距离查询点最近的点
     */
    Vector3f getClosestPoint(const Vector3f& query_pos) const override {
        Vector3f dir = query_pos - position_;
        float dist = dir.norm();
        if (dist < 1e-8f) {
            return position_ + Vector3f(radius_, 0, 0);
        }
        return position_ + dir / dist * radius_;
    }

    /**
     * 到球面的距离 (如果在球内则返回负值)
     */
    float calculateDistance(const Vector3f& pos) const override {
        return (pos - position_).norm() - radius_;
    }

    // ---- 修改 ----
    void setPosition(const Vector3f& p) {
        position_ = p;
        pose_.block<3,1>(0, 3) = p;
    }

    void setRadius(float r) { radius_ = r; }
    float getRadius() const { return radius_; }

private:
    Vector3f position_;
    float radius_;
    Matrix4f pose_;
};

} // namespace grasp_rrt
