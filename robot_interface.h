#pragma once

#include "types.h"

namespace grasp_rrt {

// ============================================================================
// RobotInterface — abstract interface replacing Simox VirtualRobot
// Users implement this against their own robot framework
// (e.g. Pinocchio, KDL, the PyBullet C++ API, or custom FK/IK/Jacobian code)
// ============================================================================

class RobotInterface {
public:
    virtual ~RobotInterface() = default;

    // ---- Joint management ----

    /**
     * Set the robot to the given joint configuration
     */
    virtual void setJointValues(const Config& config) = 0;

    /**
     * Get the current joint configuration
     */
    virtual Config getJointValues() const = 0;

    /**
     * Get the number of joints (degrees of freedom)
     */
    virtual int getDimension() const = 0;

    // ---- Forward kinematics ----

    /**
     * Get the global pose of the GCP (Grasp Center Point) as a 4x4 homogeneous matrix
     * The GCP is the grasp center, usually located at the gripper center
     */
    virtual Matrix4f getGCPGlobalPose() const = 0;

    /**
     * Get the global pose of the TCP (Tool Center Point) as a 4x4 homogeneous matrix
     * The TCP is the tool center point of the end effector
     */
    virtual Matrix4f getTCPGlobalPose() const = 0;

    /**
     * Get the global pose of a given link/frame as a 4x4 homogeneous matrix
     */
    virtual Matrix4f getFramePose(const std::string& frame_name) const = 0;

    // ---- Jacobian (differential kinematics) ----

    /**
     * Compute the geometric Jacobian at the current configuration (6 x n_joints)
     * First 3 rows: linear velocity part
     * Last 3 rows: angular velocity part
     */
    virtual MatrixXf computeJacobian() const = 0;

    /**
     * @brief Precompute the Jacobian cache for all joints.
     * Called only when Jacobians are needed (e.g. gradient computation), so the plain FK path stays clean.
     */
    virtual void prepareJacobians(float reuse_threshold_rad = 0.15f) {}

    /**
     * @brief Compute the 3xN translational Jacobian of a locally offset point on a link
     * @param frame_name link name
     * @param local_offset offset of the point in the link's local frame
     * @return the 3xN Jacobian (maps joint velocities to the point's world linear velocity)
     */
    virtual MatrixXf computeSphereJacobian(const std::string& frame_name, const Eigen::Vector3f& local_offset) const {
        return MatrixXf::Zero(3, getDimension());
    }

    /**
     * @brief O(1) access to the cached frame_id and its hash-free variant
     */
    virtual int getFrameId(const std::string& frame_name) const { return -1; }
    
    virtual Matrix4f getFramePoseById(int frame_id) const {
        return Matrix4f::Identity();
    }
    
    virtual MatrixXf computeSphereJacobianById(int frame_id, const Eigen::Vector3f& local_offset) const {
        return MatrixXf::Zero(3, getDimension());
    }

    // ---- Parallelism (Batch RRT) ----
    //
    // Size the internal per-thread state. Defaults to a no-op (single-threaded).
    // Implementations must guarantee that after setNumThreads(N), concurrent FK /
    // Jacobian calls from up to N threads are thread-safe (per-thread mutable state).
    // Read-only data such as the Model / URDF stays shared.
    virtual void setNumThreads(int /*N*/) {}
};

// ============================================================================
// EndEffectorInterface — abstract interface replacing Simox EndEffector
// Responsible for gripper/finger actuation and contact detection
// ============================================================================

class EndEffectorInterface {
public:
    virtual ~EndEffectorInterface() = default;

    /**
     * Close the gripper fingers and return the list of contact points
     * Counterpart of the C++ eef->closeActors(graspCollisionObjects)
     */
    virtual std::vector<ContactInfo> closeActors() = 0;

    /**
     * Open the gripper fingers
     * Counterpart of the C++ eef->openActors()
     */
    virtual void openActors() = 0;

    /**
     * Get the end-effector name (optional)
     */
    virtual std::string getName() const { return "EndEffector"; }
};

// ============================================================================
// GraspQualityInterface — abstract interface replacing Simox GraspQualityMeasure
// Scores grasp quality from the contact points (force-closure scoring)
// ============================================================================

class GraspQualityInterface {
public:
    virtual ~GraspQualityInterface() = default;

    /**
     * Set the contact points
     */
    virtual void setContactPoints(const std::vector<ContactInfo>& contacts) = 0;

    /**
     * Compute the grasp quality
     */
    virtual void calculateGraspQuality() = 0;

    /**
     * Get the grasp-quality score (higher is better)
     */
    virtual float getGraspQuality() const = 0;

    /**
     * Whether the force-closure condition is satisfied
     */
    virtual bool isValid() const = 0;
};

// ============================================================================
// TargetObjectInterface — abstract interface for the target object
// Provides the object pose and closest-surface-point queries
// ============================================================================

class TargetObjectInterface {
public:
    virtual ~TargetObjectInterface() = default;

    /**
     * Get the object's global pose
     */
    virtual Matrix4f getGlobalPose() const = 0;

    /**
     * Get the object's global position (3D)
     */
    virtual Vector3f getPosition() const {
        return getGlobalPose().block<3,1>(0, 3);
    }

    /**
     * Get the point on the object surface closest to a given point
     * Default implementation: returns the object center (users should override for accuracy)
     */
    virtual Vector3f getClosestPoint(const Vector3f& query_pos) const {
        return getPosition();
    }

    /**
     * Compute the distance from a given point to the object
     */
    virtual float calculateDistance(const Vector3f& pos) const {
        return (pos - getPosition()).norm();
    }
};

} // namespace grasp_rrt
