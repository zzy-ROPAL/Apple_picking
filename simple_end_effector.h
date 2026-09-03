#pragma once

#include "robot_interface.h"
#include <cmath>
#include <iostream>

namespace grasp_rrt {

// ============================================================================
// SimpleEndEffector — simplified end-effector (parallel gripper) implementation
//
// Models a parallel-finger gripper:
//   - closeActors(): tests whether the object lies within the gripper opening, generating virtual contacts
//   - openActors():  opens the gripper (resets state)
//
// This is a geometric approximation, sufficient to exercise the GraspRRT algorithm.
// Users may replace it with a real physics simulation (e.g. Bullet3 soft contacts).
// ============================================================================

class SimpleEndEffector : public EndEffectorInterface {
public:
    /**
     * @param robot           robot interface (to query the current end-effector pose)
     * @param target_object   target-object interface (for distance and contact computation)
     * @param gripper_width   maximum gripper opening width (same units as your frame)
     * @param finger_length   finger length
     */
    SimpleEndEffector(std::shared_ptr<RobotInterface> robot,
                      std::shared_ptr<TargetObjectInterface> target,
                      float gripper_width = 0.08f,
                      float finger_length = 0.05f)
        : robot_(robot), target_(target),
          gripper_width_(gripper_width),
          finger_length_(finger_length),
          is_closed_(false) {}

    /**
     * Simulate closing the gripper and detecting contact
     *
     * Logic:
     * 1. Get the current end-effector pose (TCP)
     * 2. Compute the two fingertip positions
     * 3. Check whether the object lies between the two fingers
     * 4. If so, generate two symmetric contact points
     */
    std::vector<ContactInfo> closeActors() override {
        std::vector<ContactInfo> contacts;
        is_closed_ = true;

        Matrix4f tcp_pose = robot_->getGCPGlobalPose();
        Vector3f tcp_pos = tcp_pose.block<3,1>(0, 3);

        // The two fingers are placed symmetrically along the TCP Y axis
        Vector3f y_axis = tcp_pose.block<3,1>(0, 1); // TCP Y axis
        Vector3f z_axis = tcp_pose.block<3,1>(0, 2); // TCP Z axis (approach direction)

        // The two fingertip positions
        Vector3f finger1_tip = tcp_pos + z_axis * finger_length_ + y_axis * (gripper_width_ / 2.0f);
        Vector3f finger2_tip = tcp_pos + z_axis * finger_length_ - y_axis * (gripper_width_ / 2.0f);

        // Get the object position
        Vector3f obj_pos = target_->getPosition();

        // Check whether the object is within gripper range
        // Condition 1: object distance to the TCP forward axis < gripper_width/2
        // Condition 2: object within finger length
        Vector3f to_obj = obj_pos - tcp_pos;
        float along_z = to_obj.dot(z_axis);    // projection along the approach direction
        float along_y = std::abs(to_obj.dot(y_axis)); // projection along the opening direction

        // Object within finger range (tolerance slightly relaxed)
        if (along_z > -0.02f && along_z < finger_length_ * 2.0f &&
            along_y < gripper_width_ * 0.75f) {

            // Object close enough to the TCP
            float dist = target_->calculateDistance(tcp_pos);
            // Force the hand to reach deep into the grasp instead of succeeding from far away
            float max_grasp_dist = finger_length_ * 0.9f;

            if (dist < max_grasp_dist) {
                // Generate two symmetric contact points (finger 1 and finger 2)
                Vector3f contact_normal1 = -y_axis; // finger 1 pushes inward
                Vector3f contact_normal2 = y_axis;  // finger 2 pushes inward

                // Contact points sit on the object surface
                Vector3f contact_pos1 = obj_pos + y_axis * 0.01f;  // point on the object surface
                Vector3f contact_pos2 = obj_pos - y_axis * 0.01f;

                ContactInfo c1;
                c1.position = contact_pos1;
                c1.normal = contact_normal1;
                c1.force = 1.0f;
                c1.finger_id = 0;
                contacts.push_back(c1);

                ContactInfo c2;
                c2.position = contact_pos2;
                c2.normal = contact_normal2;
                c2.force = 1.0f;
                c2.finger_id = 1;
                contacts.push_back(c2);

                // Add extra contact points to meet the minimum contact count (spread along the finger length)
                Vector3f contact_pos3 = obj_pos + y_axis * 0.01f + z_axis * 0.005f;
                ContactInfo c3;
                c3.position = contact_pos3;
                c3.normal = contact_normal1;
                c3.force = 0.8f;
                c3.finger_id = 0;
                contacts.push_back(c3);

                Vector3f contact_pos4 = obj_pos - y_axis * 0.01f + z_axis * 0.005f;
                ContactInfo c4;
                c4.position = contact_pos4;
                c4.normal = contact_normal2;
                c4.force = 0.8f;
                c4.finger_id = 1;
                contacts.push_back(c4);
            }
        }

        return contacts;
    }

    void openActors() override {
        is_closed_ = false;
    }

    std::string getName() const override {
        return "SimpleParallelGripper";
    }

    // ---- Parameter setters ----

    void setGripperWidth(float w) { gripper_width_ = w; }
    void setFingerLength(float l) { finger_length_ = l; }

private:
    std::shared_ptr<RobotInterface> robot_;
    std::shared_ptr<TargetObjectInterface> target_;
    float gripper_width_;
    float finger_length_;
    bool is_closed_;
};

} // namespace grasp_rrt
