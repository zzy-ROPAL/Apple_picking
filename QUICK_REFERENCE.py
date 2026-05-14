#!/usr/bin/env python3
"""
快速参考: VF-RRT 轨迹优化集成

这个Python脚本演示如何通过C++接口使用轨迹优化器。
虽然框架是C++，但这个文档说明C++用户应该如何调用。
"""

# ============================================================================
# 快速使用指南 (在C++代码中)
# ============================================================================

"""
1. 基础用法 - 自动优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#include "grasp_rrt.h"

auto planner = std::make_unique<GraspRrt>(cspace, robot, eef, target, quality);
planner->setStart(start_config);

if (planner->plan()) {
    // 自动优化 (使用默认参数)
    planner->optimizeSolution();
    
    auto solution = planner->getSolution();
    // 此时 solution 已是优化后的光滑轨迹
}


2. 自定义参数优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#include "trajectory_optimizer.h"

// 创建自定义配置
TrajectoryOptimizer::OptimizationConfig cfg;
cfg.num_iterations = 100;           // 更多迭代，更精细优化
cfg.learning_rate = 0.02f;          // 更快步长
cfg.smoothness_weight = 2.0f;       // 更重视平滑
cfg.obstacle_weight = 3.0f;         // 更重视避开软障碍
cfg.collision_penalty = 150.0f;     // 更严格硬碰撞
cfg.verbose = true;                 // 打印详细信息

// 创建优化器
auto optimizer = std::make_unique<TrajectoryOptimizer>(cspace, robot, cfg);

// 优化路径
ConfigPath raw_path = planner->getSolution();  // VF-RRT 的原始路径
ConfigPath optimized = optimizer->optimizePath(raw_path);

// 获取优化统计
auto cost_history = optimizer->getCostHistory();
float time_ms = optimizer->getOptimizationTime();

std::cout << "Cost improvement: "
          << (cost_history.front() - cost_history.back()) / cost_history.front() * 100
          << "%\\n";
std::cout << "Optimization time: " << time_ms << " ms\\n";


3. 参数对比表
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

场景1: 快速原型 (速度优先)
  num_iterations = 20
  learning_rate = 0.03f
  smoothness_weight = 0.5f
  → 耗时: 5-20ms, 改进: 20-30%

场景2: 平衡型 (默认)
  num_iterations = 50
  learning_rate = 0.01f
  smoothness_weight = 1.0f
  → 耗时: 20-50ms, 改进: 30-50%

场景3: 精细优化 (质量优先)
  num_iterations = 200
  learning_rate = 0.005f
  smoothness_weight = 3.0f
  obstacle_weight = 5.0f
  → 耗时: 100-200ms, 改进: 50-70%


4. 常见问题 & 解决方案
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: 优化后的路径仍然有碰撞?
A: 增加 collision_penalty 参数，或增加迭代次数

Q: 优化太慢了?
A: 减少 num_iterations，或增加 learning_rate

Q: 优化没有改进成本?
A: 检查初始路径是否已经很优化，或增加其他权重

Q: 为什么最后的路径仍然不够平滑?
A: 增加 smoothness_weight，或增加 num_iterations

Q: 内存溢出?
A: 减少路径插值密度 (num_samples)，或使用较小的路径


5. 集成检查清单
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

编译前检查:
  ☐ trajectory_optimizer.h 存在
  ☐ trajectory_optimizer.cpp 存在
  ☐ CMakeLists.txt 包含新源文件

编译检查:
  ☐ 无编译错误
  ☐ 生成可执行文件
  ☐ 文件大小 > 10MB

运行时检查:
  ☐ planner->optimizeSolution() 不崩溃
  ☐ getSolution() 返回优化后的路径
  ☐ 成本确实下降

"""

# ============================================================================
# 参数详细说明
# ============================================================================

PARAMETER_REFERENCE = {
    "num_iterations": {
        "默认": 50,
        "范围": "10-500",
        "含义": "优化循环次数",
        "增加": "更精细优化，但更慢",
        "减少": "快速完成，但可能不够优化",
        "推荐": "50-100"
    },
    
    "learning_rate": {
        "默认": 0.01,
        "范围": "0.001-0.1",
        "含义": "梯度下降的步长",
        "增加": "快速收敛，但可能发散",
        "减少": "稳定但收敛慢",
        "推荐": "0.005-0.02"
    },
    
    "smoothness_weight": {
        "默认": 1.0,
        "范围": "0.1-5.0",
        "含义": "平滑代价的权重",
        "增加": "路径更光滑",
        "减少": "更接近最优成本",
        "推荐": "0.5-2.0"
    },
    
    "obstacle_weight": {
        "默认": 2.0,
        "范围": "0.5-10.0",
        "含义": "软障碍物代价权重",
        "增加": "更主动避开树叶",
        "减少": "允许更多穿过",
        "推荐": "1.0-5.0"
    },
    
    "collision_penalty": {
        "默认": 100.0,
        "范围": "10-500",
        "含义": "硬碰撞惩罚系数",
        "增加": "更严格避碰",
        "减少": "允许临界接触",
        "推荐": "50-200"
    },
    
    "convergence_threshold": {
        "默认": 0.0001,
        "范围": "1e-6 to 1e-2",
        "含义": "收敛判定阈值",
        "增加": "更早停止",
        "减少": "优化更久",
        "推荐": "1e-5 to 1e-3"
    }
}

# ============================================================================
# 成本函数解释
# ============================================================================

COST_FUNCTION = """
总成本 J = w_smooth * J_smooth + w_obs * J_obs + w_coll * J_coll

J_smooth (平滑代价):
  衡量加速度大小
  J_smooth = ∑ ||q_{i+1} - 2*q_i + q_{i-1}||²
  较小 = 光滑轨迹
  
J_obs (软障碍物代价):
  允许穿过但有代价压力
  J_obs = ∑ cost_evaluator->evaluateCost(q_i)
  由 SdfCostEvaluator 或 TreeCanopyCostEvaluator 提供
  
J_coll (硬碰撞惩罚):
  强制避开不可穿透障碍
  J_coll = ∑ penalty * [isColliding(q_i)]
  任何碰撞 = 立即惩罚
"""

# ============================================================================
# 优化流程图
# ============================================================================

OPTIMIZATION_FLOW = """
输入: VF-RRT 生成的路径 (通常20-50个关键点)
  ↓
[插值] 稀疏→密集 (扩展到200+个采样点)
  ↓
[梯度下降循环] for i=1..num_iterations:
  ├─ 计算平滑代价梯度
  ├─ 计算软障碍物梯度
  ├─ 计算硬碰撞惩罚
  ├─ 组合梯度
  ├─ 线搜索找最佳步长
  ├─ 更新轨迹
  └─ 检查收敛?
  ↓
[采样] 密集→稀疏 (输出与输入相同数量的关键点)
  ↓
输出: 优化后的光滑轨迹
"""

# ============================================================================
# 编译命令
# ============================================================================

COMPILE_STEPS = """
# 1. 进入项目目录
cd /path/to/grasp_rrt_framework_vf

# 2. 创建并进入build目录
rm -rf build && mkdir build && cd build

# 3. CMake配置
cmake ..

# 4. 编译
make -j$(nproc)

# 5. 验证
ls -lh example_grasp_rrt  # 应该显示15MB左右

# 6. 运行集成测试
bash ../test_integration.sh
"""

# ============================================================================
# 输出示例
# ============================================================================

EXPECTED_OUTPUT = """
===== 规划阶段 =====
[GraspRRT] Planning...
[GraspRRT] Cycle 1000/50000, Nodes: 234, Grasps: 5
...
[GraspRRT] Found path in 2345 cycles

===== 优化阶段 =====
[TrajectoryOptimizer] Starting trajectory optimization...
  Original solution waypoints: 45
  Interpolated to: 200 points
  Iter 0: cost = 1234.56
  Iter 10: cost = 987.65
  Iter 20: cost = 756.43
  Iter 30: cost = 645.12
  Iter 40: cost = 598.34
  Converged at iteration 42
[TrajectoryOptimizer] Optimization complete
  Optimized waypoints: 45
  Optimization time: 45.23 ms
  Initial cost: 1234.56
  Final cost: 598.34
  Improvement: 51.50%

===== 最终结果 =====
Path: 45 waypoints (after optimization)
Planning time: 1234.56 ms
Optimization time: 45.23 ms
Total time: 1279.79 ms
"""

print(__doc__)
print("\n" + "="*80)
print("快速参考卡片已生成")
print("="*80)
