#pragma once

#include "types.h"

namespace grasp_rrt {

// ============================================================================
// CollisionChecker — abstract collision-detection interface
// Users subclass and implement it (e.g. with Bullet Physics)
// ============================================================================

class CollisionChecker {
public:
    virtual ~CollisionChecker() = default;

    /**
     * Test whether the given joint configuration is collision-free
     */
    virtual bool isCollisionFree(const Config& config) const = 0;

    /**
     * Test whether the path between two configurations is collision-free (optional override)
     * Default implementation: sampled linear interpolation
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
