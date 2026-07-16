cmake_minimum_required(VERSION 3.10)
project(GraspRRT_Framework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# 依赖库
# ============================================================================

# Eigen3 (必须)
find_package(Eigen3 REQUIRED NO_MODULE)

# Pinocchio (用于 FK / Jacobian)
find_package(pinocchio REQUIRED)

# Bullet3 (用于碰撞检测)
find_package(Bullet REQUIRED)

# OpenMP (用于并行化 SDF 构建等热点)
find_package(OpenMP REQUIRED)

# ============================================================================
# 包含目录
# ============================================================================

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BULLET_INCLUDE_DIRS}
)

# ============================================================================
# GraspRRT 核心库 (静态库)
# ============================================================================

add_library(grasp_rrt_lib STATIC
    grasp_rrt.cpp
)

target_link_libraries(grasp_rrt_lib
    Eigen3::Eigen
    pinocchio::pinocchio
    ${BULLET_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# ============================================================================
# 示例程序: Panda 抓苹果
# ============================================================================

add_executable(example_grasp_rrt example_grasp_rrt.cpp)
target_link_libraries(example_grasp_rrt
    grasp_rrt_lib
    Eigen3::Eigen
    pinocchio::pinocchio
    ${BULLET_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# ============================================================================
# Benchmark: 多次随机起点跑统计
# ============================================================================

add_executable(benchmark_grasp_rrt benchmark_grasp_rrt.cpp)
target_link_libraries(benchmark_grasp_rrt
    grasp_rrt_lib
    Eigen3::Eigen
    pinocchio::pinocchio
    ${BULLET_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# ============================================================================
# 离线工具:对已保存路径成对测量 TrajOpt 前后指标
# ============================================================================

add_executable(measure_axial measure_axial.cpp)
target_link_libraries(measure_axial
    grasp_rrt_lib
    Eigen3::Eigen
    pinocchio::pinocchio
    ${BULLET_LIBRARIES}
    OpenMP::OpenMP_CXX
)

add_executable(measure_trajopt measure_trajopt.cpp)
target_link_libraries(measure_trajopt
    grasp_rrt_lib
    Eigen3::Eigen
    pinocchio::pinocchio
    ${BULLET_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# ============================================================================
# 打印信息
# ============================================================================

message(STATUS "=== GraspRRT Framework ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen3: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Pinocchio: ${pinocchio_DIR}")
message(STATUS "Bullet: ${BULLET_INCLUDE_DIRS}")
