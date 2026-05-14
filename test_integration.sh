#!/bin/bash

# TrajectoryOptimization 集成测试脚本

echo "=========================================="
echo "VF-RRT + TrajectoryOptimization 集成测试"
echo "=========================================="
echo ""

WORKSPACE_DIR="/home/zhongyanzhang/Zhongyan_Zhang/April/4-12/TO/grasp_rrt_framework_vf"
BUILD_DIR="$WORKSPACE_DIR/build"
EXECUTABLE="$BUILD_DIR/example_grasp_rrt"

# 检查编译输出
echo "[1] 检查编译结果..."
if [ -f "$EXECUTABLE" ]; then
    SIZE=$(ls -lh "$EXECUTABLE" | awk '{print $5}')
    echo "    ✓ 可执行文件存在: $EXECUTABLE ($SIZE)"
else
    echo "    ✗ 可执行文件不存在！"
    echo "    请先运行: cd $BUILD_DIR && cmake .. && make -j\$(nproc)"
    exit 1
fi

# 检查源文件
echo ""
echo "[2] 检查源文件..."
FILES=(
    "$WORKSPACE_DIR/trajectory_optimizer.h"
    "$WORKSPACE_DIR/trajectory_optimizer.cpp"
    "$WORKSPACE_DIR/grasp_rrt.h"
    "$WORKSPACE_DIR/grasp_rrt.cpp"
    "$WORKSPACE_DIR/cspace_sampled.h"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        LINES=$(wc -l < "$file")
        echo "    ✓ $(basename $file) ($LINES 行)"
    else
        echo "    ✗ $(basename $file) 不存在！"
        exit 1
    fi
done

# 检查修改是否生效
echo ""
echo "[3] 验证关键修改..."

# 检查 optimizeSolution 是否在 grasp_rrt.h 中
if grep -q "optimizeSolution()" "$WORKSPACE_DIR/grasp_rrt.h"; then
    echo "    ✓ grasp_rrt.h: optimizeSolution() 方法已添加"
else
    echo "    ✗ grasp_rrt.h: optimizeSolution() 方法未找到"
    exit 1
fi

# 检查 getCostEvaluator 是否在 cspace_sampled.h 中
if grep -q "getCostEvaluator()" "$WORKSPACE_DIR/cspace_sampled.h"; then
    echo "    ✓ cspace_sampled.h: getCostEvaluator() 方法已添加"
else
    echo "    ✗ cspace_sampled.h: getCostEvaluator() 方法未找到"
    exit 1
fi

# 检查 example 中的优化调用
if grep -q "optimizeSolution()" "$WORKSPACE_DIR/example_grasp_rrt.cpp"; then
    echo "    ✓ example_grasp_rrt.cpp: 轨迹优化调用已添加"
else
    echo "    ✗ example_grasp_rrt.cpp: 轨迹优化调用未找到"
    exit 1
fi

# 检查 CMakeLists.txt
if grep -q "trajectory_optimizer.cpp" "$WORKSPACE_DIR/CMakeLists.txt"; then
    echo "    ✓ CMakeLists.txt: 新源文件已添加"
else
    echo "    ✗ CMakeLists.txt: 新源文件未添加"
    exit 1
fi

# 统计代码行数
echo ""
echo "[4] 代码统计..."
LINES=$(wc -l "$WORKSPACE_DIR/trajectory_optimizer.h" "$WORKSPACE_DIR/trajectory_optimizer.cpp" 2>/dev/null | tail -1 | awk '{print $1}')
echo "    轨迹优化模块: $LINES 行代码"

echo ""
echo "=========================================="
echo "✅ 集成验证成功！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 运行程序: $EXECUTABLE"
echo "  2. 观察 'TRAJECTORY OPTIMIZATION' 输出"
echo "  3. 查看成本改进百分比"
echo ""
