#pragma once

#include <random>
#include <thread>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace grasp_rrt {

// ============================================================================
// Thread-local RNG utilities
// One std::mt19937 per thread, seeded from thread::id mixed with omp_thread_num.
// rand()/RAND_MAX is not reliable under multithreading (glibc happens to be, but
// portability does not guarantee it) and gives no control over per-thread streams.
// ============================================================================

inline std::mt19937& threadRng() {
    thread_local std::mt19937 rng = []() {
        std::random_device rd;
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::seed_seq seq{rd(), static_cast<unsigned>(tid),
                          static_cast<unsigned>(
                              std::hash<std::thread::id>{}(std::this_thread::get_id()))};
        return std::mt19937(seq);
    }();
    return rng;
}

inline float randFloat01() {
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    return d(threadRng());
}

inline int randInt(int lo_incl, int hi_incl) {
    std::uniform_int_distribution<int> d(lo_incl, hi_incl);
    return d(threadRng());
}

} // namespace grasp_rrt
