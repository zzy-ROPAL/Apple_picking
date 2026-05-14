#pragma once

#include "types.h"

namespace grasp_rrt {

// ============================================================================
// CollisionChecker — 碰撞检测抽象接口
// 用户需要继承并实现 (例如用 Bullet Physics)
// ============================================================================

class CollisionChecker {
public:
    virtual ~CollisionChecker() = default;

    /**
     * 检测给定关节配置是否无碰撞
     */
    virtual bool isCollisionFree(const Config& config) const = 0;

    /**
     * 检测两个配置之间的路径是否无碰撞 (可选覆盖)
     * 默认实现: 线性插值采样检测
     */
    virtual bool isPathCollisionFree(const Config& c1, const Config& c2,
                                     int num_samples = 10) const {
        if (!isCollisionFree(c1) || !isCollisionFree(c2)) return false;
        for (int i = 1; i < num_samples; ++i) {
            float t = static_cast<float>(i) / num_samples;
            Config mid = c1 + (c2 - c1) * t;
            if (!isCollisionFree(mid)) return false;
        }
        return true;
    }
};

} // namespace grasp_rrt
