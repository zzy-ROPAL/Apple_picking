#pragma once

#include "robot_interface.h"
#include <cmath>
#include <iostream>

namespace grasp_rrt {

// ============================================================================
// SimpleEndEffector — 简化的末端执行器 (平行夹爪) 实现
//
// 模拟一个平行指夹爪:
//   - closeActors(): 检测物体是否在夹爪开口范围内, 生成虚拟接触点
//   - openActors():  打开夹爪 (重置状态)
//
// 这是一个几何近似实现, 足以让 GraspRRT 算法跑通。
// 用户可以替换为真实的物理仿真 (如 Bullet3 软体接触)。
// ============================================================================

class SimpleEndEffector : public EndEffectorInterface {
public:
    /**
     * @param robot           机器人接口 (用于获取当前末端位姿)
     * @param target_object   目标物体接口 (用于计算距离和接触)
     * @param gripper_width   夹爪最大开口宽度 (单位与你的坐标系一致)
     * @param finger_length   手指长度
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
     * 模拟关闭夹爪并检测接触
     *
     * 逻辑:
     * 1. 获取当前末端位姿 (TCP)
     * 2. 计算两个手指尖端位置
     * 3. 检查物体是否在两个手指之间
     * 4. 如果是, 生成两个对称的接触点
     */
    std::vector<ContactInfo> closeActors() override {
        std::vector<ContactInfo> contacts;
        is_closed_ = true;

        Matrix4f tcp_pose = robot_->getGCPGlobalPose();
        Vector3f tcp_pos = tcp_pose.block<3,1>(0, 3);

        // 夹爪的两个手指沿 TCP 的 Y 轴对称分布
        Vector3f y_axis = tcp_pose.block<3,1>(0, 1); // TCP Y 轴
        Vector3f z_axis = tcp_pose.block<3,1>(0, 2); // TCP Z 轴 (接近方向)

        // 两个手指尖端位置
        Vector3f finger1_tip = tcp_pos + z_axis * finger_length_ + y_axis * (gripper_width_ / 2.0f);
        Vector3f finger2_tip = tcp_pos + z_axis * finger_length_ - y_axis * (gripper_width_ / 2.0f);

        // 获取物体位置
        Vector3f obj_pos = target_->getPosition();

        // 检查物体是否在夹爪范围内
        // 条件1: 物体到 TCP 前方轴线的距离 < 夹爪宽度/2
        // 条件2: 物体在手指长度范围内
        Vector3f to_obj = obj_pos - tcp_pos;
        float along_z = to_obj.dot(z_axis);    // 沿接近方向的投影
        float along_y = std::abs(to_obj.dot(y_axis)); // 沿开口方向的投影

        // 物体在手指范围内 (稍微放宽容错)
        if (along_z > -0.02f && along_z < finger_length_ * 2.0f &&
            along_y < gripper_width_ * 0.75f) {

            // 物体到 TCP 的距离足够近
            float dist = target_->calculateDistance(tcp_pos);
            // 强制机械手深入抓取，而不是离老远就算成功
            float max_grasp_dist = finger_length_ * 0.9f;

            if (dist < max_grasp_dist) {
                // 生成两个对称接触点 (手指1和手指2)
                Vector3f contact_normal1 = -y_axis; // 手指1 向内推
                Vector3f contact_normal2 = y_axis;  // 手指2 向内推

                // 接触点在物体表面
                Vector3f contact_pos1 = obj_pos + y_axis * 0.01f;  // 物体表面点
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

                // 添加额外的接触点以满足最小接触数 (沿手指长度方向分布)
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

    // ---- 参数设置 ----

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
