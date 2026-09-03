#pragma once

#include "robot_interface.h"

namespace grasp_rrt {

// ============================================================================
// SimpleTargetObject — simple implementation of a spherical target object
// Suitable for approximately spherical grasp targets such as apples
// ============================================================================

class SimpleTargetObject : public TargetObjectInterface {
public:
    /**
     * @param position  object center position (global frame)
     * @param radius    object radius
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
     * Closest point on the sphere surface to the query point
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
     * Distance to the sphere surface (negative if inside the sphere)
     */
    float calculateDistance(const Vector3f& pos) const override {
        return (pos - position_).norm() - radius_;
    }

    // ---- Mutators ----
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
