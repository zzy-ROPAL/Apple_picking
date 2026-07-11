#pragma once

#include "collision_checker.h"
#include "robot_interface.h"

#include <btBulletCollisionCommon.h>
#include <BulletCollision/NarrowPhaseCollision/btGjkPairDetector.h>

#include <vector>
#include <memory>
#include <iostream>
#include <functional>
#include <string>
#include <atomic>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace grasp_rrt {

// ============================================================================
// BulletCollisionChecker — 基于 Bullet3 的碰撞检测器 (per-thread world)
//
// 线程模型:
//   构造时传入 num_threads，内部为每个线程持有一份独立的:
//     - btCollisionWorld + dispatcher + broadphase + config
//     - 每个 link / obstacle / goal 的 btCollisionObject (存 transform 状态)
//   共享 (只读) 的部分:
//     - btCollisionShape (几何体)
//     - FK 回调、连杆名、Goal 豁免前缀
//
//   isCollisionFree() 用 omp_get_thread_num() 选择自己那份 world。
//   num_threads=1 时行为与单线程版完全一致。
// ============================================================================

class BulletCollisionChecker : public CollisionChecker {
public:
    explicit BulletCollisionChecker(std::shared_ptr<RobotInterface> robot,
                                    int num_threads = 1)
        : robot_(robot), collision_count_(0),
          num_threads_(std::max(1, num_threads)) {
        worlds_.resize(num_threads_);
        for (int t = 0; t < num_threads_; ++t) {
            auto& w = worlds_[t];
            w.config     = std::make_unique<btDefaultCollisionConfiguration>();
            w.dispatcher = std::make_unique<btCollisionDispatcher>(w.config.get());
            w.broadphase = std::make_unique<btDbvtBroadphase>();
            w.world      = std::make_unique<btCollisionWorld>(
                w.dispatcher.get(), w.broadphase.get(), w.config.get());
        }
    }

    ~BulletCollisionChecker() {
        // 先把所有 collision object 从各 world 里取出并 delete
        for (auto& w : worlds_) {
            for (auto* obj : w.robot_objs) {
                w.world->removeCollisionObject(obj);
                delete obj;
            }
            for (auto* obj : w.obstacle_objs) {
                w.world->removeCollisionObject(obj);
                delete obj;
            }
            for (auto* obj : w.goal_objs) {
                w.world->removeCollisionObject(obj);
                delete obj;
            }
        }
        // 然后 delete 共享的 shape (每个只 delete 一次)
        for (auto* s : all_shapes_) delete s;
    }

    // ---- 注册机器人连杆碰撞体 ----

    void addRobotLink(btCollisionShape* shape,
                      std::function<Matrix4f(const Config&)> fk_func,
                      const std::string& name = "") {
        all_shapes_.push_back(shape);
        for (auto& w : worlds_) {
            auto* obj = new btCollisionObject();
            obj->setCollisionShape(shape);
            w.world->addCollisionObject(obj);
            w.robot_objs.push_back(obj);
        }
        link_fk_functions_.push_back(fk_func);
        link_names_.push_back(name);
    }

    void addRobotLinkSphere(float radius,
                            std::function<Matrix4f(const Config&)> fk_func,
                            const std::string& name = "") {
        addRobotLink(new btSphereShape(radius), fk_func, name);
    }

    void addRobotLinkCapsule(float radius, float half_height,
                             std::function<Matrix4f(const Config&)> fk_func,
                             const std::string& name = "") {
        addRobotLink(new btCapsuleShapeZ(radius, 2.0f * half_height), fk_func, name);
    }

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
            hull->addPoint(btVector3(r_bottom * c, r_bottom * s, -half_h));
            hull->addPoint(btVector3(r_top * c,    r_top * s,    +half_h));
        }
        hull->recalcLocalAabb();
        addRobotLink(hull, fk_func, name);
    }

    void addRobotLinkCylinder(float radius, float half_height,
                              std::function<Matrix4f(const Config&)> fk_func,
                              const std::string& name = "") {
        addRobotLink(new btCylinderShapeZ(btVector3(radius, radius, half_height)),
                     fk_func, name);
    }

    void addRobotLinkBox(const Vector3f& half_extents,
                         std::function<Matrix4f(const Config&)> fk_func,
                         const std::string& name = "") {
        addRobotLink(new btBoxShape(btVector3(half_extents.x(),
                                              half_extents.y(),
                                              half_extents.z())),
                     fk_func, name);
    }

    // ---- 注册障碍物 (Class 1) ----

    void addObstacle(btCollisionShape* shape, const Matrix4f& pose) {
        all_shapes_.push_back(shape);
        btTransform tr = eigenToBullet(pose);
        for (auto& w : worlds_) {
            auto* obj = new btCollisionObject();
            obj->setCollisionShape(shape);
            obj->setWorldTransform(tr);
            w.world->addCollisionObject(obj);
            w.obstacle_objs.push_back(obj);
        }
    }

    void addObstacleSphere(float radius, const Vector3f& position) {
        Matrix4f pose = Matrix4f::Identity();
        pose.block<3,1>(0, 3) = position;
        addObstacle(new btSphereShape(radius), pose);
    }

    void addObstacleCylinder(float radius, float half_height, const Matrix4f& pose) {
        addObstacle(new btCylinderShapeZ(btVector3(radius, radius, half_height)), pose);
    }

    // ---- 注册目标物体 (Class 3) ----

    void addGoalObstacle(btCollisionShape* shape, const Matrix4f& pose) {
        all_shapes_.push_back(shape);
        btTransform tr = eigenToBullet(pose);
        for (auto& w : worlds_) {
            auto* obj = new btCollisionObject();
            obj->setCollisionShape(shape);
            obj->setWorldTransform(tr);
            w.world->addCollisionObject(obj);
            w.goal_objs.push_back(obj);
        }
    }

    void addGoalObstacleSphere(float radius, const Vector3f& position) {
        Matrix4f pose = Matrix4f::Identity();
        pose.block<3,1>(0, 3) = position;
        addGoalObstacle(new btSphereShape(radius), pose);
    }

    void addGoalObstacleExemptLink(const std::string& link_name_prefix) {
        goal_exempt_link_prefixes_.push_back(link_name_prefix);
    }

    // ---- CollisionChecker 接口 ----

    bool isCollisionFree(const Config& config) const override {
        collision_count_.fetch_add(1, std::memory_order_relaxed);

        const int tid = getThreadId();
        const auto& w = worlds_[tid];

        // 1. 更新该线程 world 的机器人连杆位姿
        for (size_t i = 0; i < w.robot_objs.size(); ++i) {
            Matrix4f link_pose = link_fk_functions_[i](config);
            btTransform tr = eigenToBullet(link_pose);
            w.robot_objs[i]->setWorldTransform(tr);
        }

        // 2. 执行碰撞检测
        w.world->performDiscreteCollisionDetection();

        // 3. 检查 contact manifolds
        int num_manifolds = w.dispatcher->getNumManifolds();
        for (int i = 0; i < num_manifolds; ++i) {
            auto* manifold = w.dispatcher->getManifoldByIndexInternal(i);
            int num_contacts = manifold->getNumContacts();
            if (num_contacts == 0) continue;

            const btCollisionObject* body0 = manifold->getBody0();
            const btCollisionObject* body1 = manifold->getBody1();

            bool is_robot0    = isRobotObject(w, body0);
            bool is_robot1    = isRobotObject(w, body1);
            bool is_obstacle0 = isObstacleObject(w, body0);
            bool is_obstacle1 = isObstacleObject(w, body1);

            // 机器人 - 障碍物 (Class 1)
            if ((is_robot0 && is_obstacle1) || (is_robot1 && is_obstacle0)) {
                for (int j = 0; j < num_contacts; ++j) {
                    if (manifold->getContactPoint(j).getDistance() < 0.01f) {
                        return false;
                    }
                }
            }

            // 机器人 - 目标物体 (Class 3) — 带豁免
            bool is_goal0 = isGoalObstacleObject(w, body0);
            bool is_goal1 = isGoalObstacleObject(w, body1);
            if ((is_robot0 && is_goal1) || (is_robot1 && is_goal0)) {
                const btCollisionObject* robot_obj = is_robot0 ? body0 : body1;
                int robot_idx = getRobotObjectIndex(w, robot_obj);
                if (!isGoalExemptLink(robot_idx)) {
                    for (int j = 0; j < num_contacts; ++j) {
                        if (manifold->getContactPoint(j).getDistance() < 0.01f) {
                            return false;
                        }
                    }
                }
            }

            // 自碰撞 (可选)
            if (check_self_collision_ && is_robot0 && is_robot1) {
                int idx0 = getRobotObjectIndex(w, body0);
                int idx1 = getRobotObjectIndex(w, body1);
                if (std::abs(idx0 - idx1) > 1) {
                    for (int j = 0; j < num_contacts; ++j) {
                        if (manifold->getContactPoint(j).getDistance() < 0.01f) {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

    // ---- 配置 ----

    void enableSelfCollisionCheck(bool enable) { check_self_collision_ = enable; }
    int getCollisionCheckCount() const {
        return collision_count_.load(std::memory_order_relaxed);
    }
    void resetCollisionCheckCount() {
        collision_count_.store(0, std::memory_order_relaxed);
    }
    int getNumThreads() const { return num_threads_; }

private:
    struct ThreadWorld {
        std::unique_ptr<btDefaultCollisionConfiguration> config;
        std::unique_ptr<btCollisionDispatcher>           dispatcher;
        std::unique_ptr<btDbvtBroadphase>                broadphase;
        std::unique_ptr<btCollisionWorld>                world;
        std::vector<btCollisionObject*>                  robot_objs;
        std::vector<btCollisionObject*>                  obstacle_objs;
        std::vector<btCollisionObject*>                  goal_objs;
    };

    std::shared_ptr<RobotInterface> robot_;
    mutable std::atomic<int>        collision_count_;
    bool                            check_self_collision_ = false;
    int                             num_threads_;

    // Per-thread Bullet 状态
    std::vector<ThreadWorld> worlds_;

    // 共享几何体 (每个 shape 只 delete 一次)
    std::vector<btCollisionShape*> all_shapes_;

    // 共享元数据
    std::vector<std::function<Matrix4f(const Config&)>> link_fk_functions_;
    std::vector<std::string>                            link_names_;
    std::vector<std::string>                            goal_exempt_link_prefixes_;

    // ---- 辅助 ----

    int getThreadId() const {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        if (tid >= num_threads_) tid = 0;  // 兜底
        return tid;
#else
        return 0;
#endif
    }

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

    static bool isRobotObject(const ThreadWorld& w, const btCollisionObject* obj) {
        for (auto* r : w.robot_objs) if (r == obj) return true;
        return false;
    }
    static bool isObstacleObject(const ThreadWorld& w, const btCollisionObject* obj) {
        for (auto* r : w.obstacle_objs) if (r == obj) return true;
        return false;
    }
    static bool isGoalObstacleObject(const ThreadWorld& w, const btCollisionObject* obj) {
        for (auto* r : w.goal_objs) if (r == obj) return true;
        return false;
    }
    static int getRobotObjectIndex(const ThreadWorld& w, const btCollisionObject* obj) {
        for (size_t i = 0; i < w.robot_objs.size(); ++i) {
            if (w.robot_objs[i] == obj) return static_cast<int>(i);
        }
        return -1;
    }

    bool isGoalExemptLink(int robot_idx) const {
        if (robot_idx < 0 || robot_idx >= (int)link_names_.size()) return false;
        const std::string& name = link_names_[robot_idx];
        for (const auto& prefix : goal_exempt_link_prefixes_) {
            if (name.find(prefix) != std::string::npos) return true;
        }
        return false;
    }
};

} // namespace grasp_rrt
