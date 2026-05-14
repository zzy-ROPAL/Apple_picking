#pragma once

#include "collision_checker.h"
#include "robot_interface.h"

#include <btBulletCollisionCommon.h>
#include <BulletCollision/NarrowPhaseCollision/btGjkPairDetector.h>

#include <vector>
#include <memory>
#include <iostream>
#include <functional>

namespace grasp_rrt {

// ============================================================================
// BulletCollisionChecker — 基于 Bullet3 的碰撞检测器
//
// 使用 Bullet 的 btCollisionWorld 进行离散碰撞检测:
// 1. 用户注册机器人连杆的碰撞形状 (box, cylinder, sphere, mesh)
// 2. 用户注册障碍物的碰撞形状
// 3. isCollisionFree() 时:
//    a. 通过机器人接口计算 FK, 更新连杆位姿
//    b. 调用 Bullet 碰撞检测
//    c. 返回是否无碰撞
// ============================================================================

class BulletCollisionChecker : public CollisionChecker {
public:
    BulletCollisionChecker(std::shared_ptr<RobotInterface> robot)
        : robot_(robot), collision_count_(0) {
        // 初始化 Bullet 碰撞世界
        collision_config_   = std::make_unique<btDefaultCollisionConfiguration>();
        dispatcher_         = std::make_unique<btCollisionDispatcher>(collision_config_.get());
        broadphase_         = std::make_unique<btDbvtBroadphase>();
        collision_world_    = std::make_unique<btCollisionWorld>(
            dispatcher_.get(), broadphase_.get(), collision_config_.get());
    }

    ~BulletCollisionChecker() {
        // 从 world 中移除所有碰撞对象
        for (auto* obj : robot_collision_objects_) {
            collision_world_->removeCollisionObject(obj);
            delete obj->getCollisionShape();
            delete obj;
        }
        for (auto* obj : obstacle_collision_objects_) {
            collision_world_->removeCollisionObject(obj);
            delete obj->getCollisionShape();
            delete obj;
        }
    }

    // ---- 注册机器人连杆碰撞体 ----

    /**
     * 添加一个机器人连杆的碰撞形状
     *
     * @param shape     碰撞形状 (调用者分配, 此类负责释放)
     * @param fk_func   FK 回调: 给定关节配置, 返回该连杆的 4x4 全局位姿
     * @param name      连杆名称 (调试用)
     */
    void addRobotLink(btCollisionShape* shape,
                      std::function<Matrix4f(const Config&)> fk_func,
                      const std::string& name = "") {
        auto* obj = new btCollisionObject();
        obj->setCollisionShape(shape);
        collision_world_->addCollisionObject(obj);

        robot_collision_objects_.push_back(obj);
        link_fk_functions_.push_back(fk_func);
        link_names_.push_back(name);
    }

    /**
     * 便捷方法: 添加一个球体连杆
     */
    void addRobotLinkSphere(float radius,
                            std::function<Matrix4f(const Config&)> fk_func,
                            const std::string& name = "") {
        addRobotLink(new btSphereShape(radius), fk_func, name);
    }

    /**
     * 便捷方法: 添加一个胶囊体连杆 (沿局部 Z 轴)
     * 胶囊体 = 圆柱(半高 half_height) + 两端半球(半径 radius)
     * 总长度 = 2 * half_height + 2 * radius
     * 
     * fk_func 应返回胶囊体**中心**的位姿（不是底部）。
     * 例如：连杆从 z=0 延伸到 z=0.3，则中心在 z=0.15，half_height=0.15
     */
    void addRobotLinkCapsule(float radius, float half_height,
                             std::function<Matrix4f(const Config&)> fk_func,
                             const std::string& name = "") {
        addRobotLink(new btCapsuleShapeZ(radius, 2.0f * half_height), fk_func, name);
    }

    /**
     * 便捷方法: 添加一个锥台体（梯形旋转体）连杆
     * 用 btConvexHullShape 的顶点凸包精确模拟梯形截面（如 Panda 法兰）。
     * 锥台沿局部 Z 轴延伸：
     *   z = -half_h 为较大的底面（半径 r_bottom）
     *   z = +half_h 为较小的顶面（半径 r_top）
     * fk_func 应返回锥台「中心」（两端面的中点）的位姿。
     * @param n_sides 多边形近似圆形的边数，默认 12 即可（精度足够且高效）
     */
    void addRobotLinkFrustum(float r_bottom, float r_top, float half_h,
                              std::function<Matrix4f(const Config&)> fk_func,
                              const std::string& name = "",
                              int n_sides = 12) {
        auto* hull = new btConvexHullShape();

        const float two_pi = 2.0f * 3.14159265f;
        for (int i = 0; i < n_sides; ++i) {
            float angle = two_pi * i / n_sides;
            float c = std::cos(angle);
            float s = std::sin(angle);
            // 底面（大端，z = -half_h）
            hull->addPoint(btVector3(r_bottom * c, r_bottom * s, -half_h));
            // 顶面（小端，z = +half_h）
            hull->addPoint(btVector3(r_top * c,    r_top * s,    +half_h));
        }
        hull->recalcLocalAabb();
        addRobotLink(hull, fk_func, name);
    }

    /**
     * 便捷方法: 添加一个圆柱连杆
     */
    void addRobotLinkCylinder(float radius, float half_height,
                              std::function<Matrix4f(const Config&)> fk_func,
                              const std::string& name = "") {
        addRobotLink(new btCylinderShapeZ(btVector3(radius, radius, half_height)),
                     fk_func, name);
    }

    /**
     * 便捷方法: 添加一个长方体连杆
     */
    void addRobotLinkBox(const Vector3f& half_extents,
                         std::function<Matrix4f(const Config&)> fk_func,
                         const std::string& name = "") {
        addRobotLink(new btBoxShape(btVector3(half_extents.x(),
                                              half_extents.y(),
                                              half_extents.z())),
                     fk_func, name);
    }

    // ---- 注册障碍物 ----

    /**
     * 添加一个静态障碍物
     * @param shape    碰撞形状
     * @param pose     全局位姿
     */
    void addObstacle(btCollisionShape* shape, const Matrix4f& pose) {
        auto* obj = new btCollisionObject();
        obj->setCollisionShape(shape);

        btTransform tr = eigenToBullet(pose);
        obj->setWorldTransform(tr);
        collision_world_->addCollisionObject(obj);

        obstacle_collision_objects_.push_back(obj);
    }

    /**
     * 便捷方法: 添加一个球体障碍物
     */
    void addObstacleSphere(float radius, const Vector3f& position) {
        Matrix4f pose = Matrix4f::Identity();
        pose.block<3,1>(0, 3) = position;
        addObstacle(new btSphereShape(radius), pose);
    }

    void addObstacleCylinder(float radius, float half_height, const Matrix4f& pose) {
        addObstacle(new btCylinderShapeZ(btVector3(radius, radius, half_height)), pose);
    }

    // ---- 注册目标物体 (Goal Obstacle, Class 3) ----

    /**
     * 添加一个 "目标物体" 障碍物 (例如苹果).
     * 行为: 对大臂连杆是硬障碍 (碰到即报错), 但允许末端执行器(夹爪)接触.
     * 通过 addGoalObstacleExemptLink() 指定哪些连杆拥有接触豁免权.
     *
     * @param shape    碰撞形状
     * @param pose     全局位姿
     */
    void addGoalObstacle(btCollisionShape* shape, const Matrix4f& pose) {
        auto* obj = new btCollisionObject();
        obj->setCollisionShape(shape);
        btTransform tr = eigenToBullet(pose);
        obj->setWorldTransform(tr);
        collision_world_->addCollisionObject(obj);
        goal_obstacle_objects_.push_back(obj);
    }

    void addGoalObstacleSphere(float radius, const Vector3f& position) {
        Matrix4f pose = Matrix4f::Identity();
        pose.block<3,1>(0, 3) = position;
        addGoalObstacle(new btSphereShape(radius), pose);
    }

    /**
     * 注册一个拥有 "Goal障碍物豁免权" 的机器人连杆名称前缀。
     * 凡是 link_names_ 中包含该前缀的连杆, 与 Goal Obstacle 的碰撞将被忽略。
     * 例如: addGoalObstacleExemptLink("panda_hand") 豁免夹爪所有碰撞体。
     */
    void addGoalObstacleExemptLink(const std::string& link_name_prefix) {
        goal_exempt_link_prefixes_.push_back(link_name_prefix);
    }

    // ---- CollisionChecker 接口实现 ----

    bool isCollisionFree(const Config& config) const override {
        collision_count_++;

        // 1. 更新所有机器人连杆的位姿
        for (size_t i = 0; i < robot_collision_objects_.size(); ++i) {
            Matrix4f link_pose = link_fk_functions_[i](config);
            btTransform tr = eigenToBullet(link_pose);
            robot_collision_objects_[i]->setWorldTransform(tr);
        }

        // 2. 执行碰撞检测
        collision_world_->performDiscreteCollisionDetection();

        // 3. 检查 contact manifolds
        int num_manifolds = dispatcher_->getNumManifolds();
        for (int i = 0; i < num_manifolds; ++i) {
            auto* manifold = dispatcher_->getManifoldByIndexInternal(i);
            int num_contacts = manifold->getNumContacts();

            if (num_contacts > 0) {
                // 检查是否是机器人-障碍物碰撞 (排除机器人-机器人自碰撞, 如果需要的话)
                const btCollisionObject* body0 = manifold->getBody0();
                const btCollisionObject* body1 = manifold->getBody1();

                bool is_robot0 = isRobotObject(body0);
                bool is_robot1 = isRobotObject(body1);
                bool is_obstacle0 = isObstacleObject(body0);
                bool is_obstacle1 = isObstacleObject(body1);

                // 机器人-障碍物碰撞 (Class 1: 树干)
                if ((is_robot0 && is_obstacle1) || (is_robot1 && is_obstacle0)) {
                    for (int j = 0; j < num_contacts; ++j) {
                        if (manifold->getContactPoint(j).getDistance() < 0.01f) {
                            return false; // 碰撞!
                        }
                    }
                }

                // 机器人-目标物体碰撞 (Class 3: 苹果) — 带 ACM 豁免检查
                bool is_goal0 = isGoalObstacleObject(body0);
                bool is_goal1 = isGoalObstacleObject(body1);
                if ((is_robot0 && is_goal1) || (is_robot1 && is_goal0)) {
                    // 找出是哪条机器人连杆
                    const btCollisionObject* robot_obj = is_robot0 ? body0 : body1;
                    int robot_idx = getRobotObjectIndex(robot_obj);
                    // 如果该连杆在豁免名单里, 跳过 (夹爪可以接触苹果)
                    if (!isGoalExemptLink(robot_idx)) {
                        for (int j = 0; j < num_contacts; ++j) {
                            if (manifold->getContactPoint(j).getDistance() < 0.01f) {
                                return false; // 大臂撞到苹果 — 报错!
                            }
                        }
                    }
                }


                // 可选: 自碰撞检测
                if (check_self_collision_ && is_robot0 && is_robot1) {
                    // 跳过相邻连杆 (它们总是接触的)
                    int idx0 = getRobotObjectIndex(body0);
                    int idx1 = getRobotObjectIndex(body1);
                    if (std::abs(idx0 - idx1) > 1) {
                        for (int j = 0; j < num_contacts; ++j) {
                            if (manifold->getContactPoint(j).getDistance() < 0.01f) {
                                return false;
                            }
                        }
                    }
                }
            }
        }

        return true; // 无碰撞
    }

    // ---- 配置 ----

    void enableSelfCollisionCheck(bool enable) { check_self_collision_ = enable; }
    int getCollisionCheckCount() const { return collision_count_; }
    void resetCollisionCheckCount() { collision_count_ = 0; }

private:
    std::shared_ptr<RobotInterface> robot_;
    mutable int collision_count_;
    bool check_self_collision_ = false;

    // Bullet 碰撞系统
    std::unique_ptr<btDefaultCollisionConfiguration> collision_config_;
    std::unique_ptr<btCollisionDispatcher> dispatcher_;
    std::unique_ptr<btDbvtBroadphase> broadphase_;
    std::unique_ptr<btCollisionWorld> collision_world_;

    // 机器人连杆
    std::vector<btCollisionObject*> robot_collision_objects_;
    std::vector<std::function<Matrix4f(const Config&)>> link_fk_functions_;
    std::vector<std::string> link_names_;

    // 障碍物 (Class 1: 树干)
    std::vector<btCollisionObject*> obstacle_collision_objects_;

    // 目标物体 (Class 3: 苹果) — 带 ACM 豁免支持
    std::vector<btCollisionObject*> goal_obstacle_objects_;
    std::vector<std::string> goal_exempt_link_prefixes_; // 豁免连杆名称前缀列表

    // ---- 辅助方法 ----

    static btTransform eigenToBullet(const Matrix4f& m) {
        btTransform t;
        t.setOrigin(btVector3(m(0,3), m(1,3), m(2,3)));
        btMatrix3x3 basis(
            m(0,0), m(0,1), m(0,2),
            m(1,0), m(1,1), m(1,2),
            m(2,0), m(2,1), m(2,2));
        t.setBasis(basis);
        return t;
    }

    bool isRobotObject(const btCollisionObject* obj) const {
        for (auto* r : robot_collision_objects_) {
            if (r == obj) return true;
        }
        return false;
    }

    bool isObstacleObject(const btCollisionObject* obj) const {
        for (auto* o : obstacle_collision_objects_) {
            if (o == obj) return true;
        }
        return false;
    }

    bool isGoalObstacleObject(const btCollisionObject* obj) const {
        for (auto* o : goal_obstacle_objects_) {
            if (o == obj) return true;
        }
        return false;
    }

    // 检查给定索引的机器人连杆是否在 Goal Obstacle 豁免名单里
    bool isGoalExemptLink(int robot_idx) const {
        if (robot_idx < 0 || robot_idx >= (int)link_names_.size()) return false;
        const std::string& name = link_names_[robot_idx];
        for (const auto& prefix : goal_exempt_link_prefixes_) {
            if (name.find(prefix) != std::string::npos) return true;
        }
        return false;
    }


    int getRobotObjectIndex(const btCollisionObject* obj) const {
        for (size_t i = 0; i < robot_collision_objects_.size(); ++i) {
            if (robot_collision_objects_[i] == obj) return static_cast<int>(i);
        }
        return -1;
    }
};

} // namespace grasp_rrt
