# GraspRRT Framework — 项目交接文档(2026-08-01)

C++17 实现的 **GraspRRT**(边探索边生成抓取假设的运动规划器),Franka Panda 7-DOF,
苹果采摘场景:在树冠(软枝可推开、硬枝/树干不可碰)中规划到苹果的抓取路径。
参考论文:Vahrenkamp et al., *Integrated Grasp and Motion Planning*, ICRA 2010。

**本文件是唯一可信的项目状态文档。** 根目录下 `PROJECT_INFO.txt` / `COMPLETION_REPORT.md` /
`INTEGRATION_SUMMARY.md` / `TRAJECTORY_OPTIMIZATION_README.md` 全部过时,不要当事实来源。
改代码前先 grep 源码。

## 构建与运行

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

依赖:**Eigen3**、**Pinocchio**(FK/Jacobian)、**Bullet3**(碰撞)。URDF 查找:
argv → 环境变量 `PANDA_URDF` → pybullet_data 硬编码路径。

**当前标准实验命令**(tree6 简化树、15mm 枝、苹果在枝根、全机制栈):

```bash
./build/benchmark_grasp_rrt 50 --starts-file <starts.txt> \
  --tree6-simple --branch-rscale 0.5 --apple-root --apple-branch-soft \
  --branch-stiffness --shortcut --uni-refuse --nav-field --or-parallel 8 \
  --label <标签>
```

输出目录名由开关自动拼接(`_pengate_uni_nav_<label>`),内含 `benchmark_results.csv`
(每行一 run)、`grasp_path_XX.csv`(轨迹,7 关节/行)、`scene_layout.json`
(圆柱几何,含每段 `t0/t1` 弧长区间)、`starts_used.txt`(可复现起点)。

## 架构总览(功能 → 机制)

三阶段:Phase1 一次性建 SDF 体素场 + nav 场;Phase2 RRT 建树 + 抓取假设;
Phase3 **只用 shortcut,TrajOpt 已被用户明令禁用**(指标一律 raw/shortcut 路径)。

| 功能 | 机制 | 位置 |
|---|---|---|
| 抓取姿态生成 | GraspRRT:approach sphere(`pose_sphere_`)+ DiffIK 下探(J⁺)+ 接触打分;face disable(≥200 失败禁用,保底 2);下探候选门 = **测地距离 < 0.6m**(nav 场) | `grasp_rrt.cpp`, `approach_discretization.h` |
| 树生长引导 | **VF-RRT**:`dir = normalize(0.5·随机 + 0.5·场)`;采样均匀 → 概率完备保持 | `extendVFRRT` |
| — 吸引(全局路由) | **nav 场**:体素网格上多源 Dijkstra(种子=苹果下方球冠;刚性+禁核不可通行;软枝壳层过路费 ∝ 深度×刚度,gain=8)。远场走洪泛下坡,**测地距离 < 0.15m(+0.10 平滑带)** 切回随机球冠点直线吸引(球冠每 1000 cycle 重采样)。切换一律测地、绝不欧氏(苹果正上方欧氏近测地远,防止从上硬凿) | `buildNavField`, `queryNavDirection/Distance` |
| — 斥力(全臂避障) | SDF 梯度斥力(~20 球覆盖全臂,读预计算距离网格) | `evaluateAnisotropicForce` |
| — 软枝额外推离(轻柔度) | **stiffness 场**(force ∝ stiffness(t)·depth,根硬梢软)。消融结论:去掉它快 32% 但典型接触深度 0.40→0.74,**保留** | `computeVectorField` |
| — 绕树干 | 旋涡场(刚性圆柱切向环流) | 同上 |
| — 采样偏置 | Method A(最优赤道下节点加抖动) | `sampleGoalBiased` |
| 软枝安全交互 | 三层:**unified steering**(提议投影)→ **severity sigmoid**(边级概率拒绝)→ **pen gate**(硬有效性,唯一保证层) | 见下 |
| 碰撞 | Bullet DCD 递归中点,eager | `cspace_sampled.h` |
| 路径平滑 | score-aware **shortcut**(贪心最远直连,每段过全部安全检查;转角 ~40×↓;**不是 TrajOpt**) | `shortcutSolutionPath` |
| 提速 | `--or-parallel 8`(8 树竞速,砍 RNG 长尾)+ OpenMP batch + cheap-check-first | benchmark |

## Unified refuse(合并 edge refuse,核心机制)

每个 (探针, 软枝) 对有**独立损伤预算**,两个消费通道:压深(弯枝)与带触滑移(拖锯)。
每步评分:

```
severity = h_in·内压/(κ·slack) + G·轴滑/L_ax(t)
```

- `slack = allow(t) − pen`;`allow(t)` = 15mm(根)→75mm(梢):**梢部自由 = 径向压入,
  不是轴向滑行**(用户 7-28 澄清,故 `L_ax` 全枝收紧 5→8mm/步)
- `G`:预接触包络(5cm 面-面间隙)内恒 0.5(外缘 1cm 斜坡)= 侧面接近整形;
  接触后 0.5→1 随深度 = 反拖锯
- `h_in`:触前 3cm 斜坡(连续性)
- 超 0.7 → 投影(两分量同乘 0.7/sev,剩环向缠绕);边级仍超 → sigmoid 拒绝
  (中点 0.7, gain 6)
- **9 探针 × 全部软枝逐对计算,取 max —— 无 argmax/无 K 近邻 → 结构性无振荡**
  (tensor 的反振荡性质被继承,tensor 本身退役,留 `--edge-refuse-tensor` 作消融)

**Pen gate(保证层)**:`pen > allow(t)` 或探针进入根部轴心禁区
(`axis_clear = 10mm·(1−t)`,梢部为 0)→ 构型无效。挂在 `isConfigValid` 上,
DCD/下探/shortcut/起点全部自动继承。cuts=0 是构造保证,不是统计结果。

旧机制(depth steering `--pen-steer`、sample steering、tensor edge refuse、sticky)
全部保留在代码里作 ablation,`--uni-refuse` 时自动禁用。

## 关键文件

| 文件 | 职责 |
|---|---|
| `grasp_rrt.cpp/.h` | 规划器主体(~2000 行,重心) |
| `types.h` | `GraspRrtConfig` 全部可调参数(调参先来这里) |
| `sdf_cost_evaluator.h` | SDF 网格 + nav 场 + unified refuse + pen gate(第二重心) |
| `cspace_sampled.h` | DCD + edge-refuse sigmoid 管线 + gate 挂载点 |
| `benchmark_grasp_rrt.cpp` | 实验入口,全部 CLI 开关与场景 |
| `viz_cut_scene.py` | 排名/回放/渲染工具(见下) |
| `probe_dump.cpp` | C++ 探针 FK 导出(与 Python 交叉验证用) |

## 易踩的坑(必读)

1. **机器人基座在 z=0.4m**(`panda_pinocchio.h: base_offset_`)。任何外部可视化/Isaac
   对齐都必须加这个偏移,否则 FK 差 0.4m(已经踩过一次)。
2. **单位**:关节/SDF/米;但 `try_grasps_distance2 = 360000` 是 mm² 遗留,与米制
   `squaredNorm()` 比较**恒真**(实际下探门是 nav 测地 0.6m)。
3. **指标刻度**:`cut_violations` 与 viz 的 ratio 沿用旧 allow 刻度(15→75mm),
   规划器内部可比它严(如 axis-clear)。跨实验对比时刻度一致,别改。
4. **TrajOpt 禁用**:用户明令。路径质量只许 shortcut。
5. 对照实验必须 `--starts-file` 固定起点;**起点要用最严机制栈生成**(gate-valid),
   否则出现"起点本身嵌枝"伪失败(踩过)。
6. 规划器 RNG 是 thread_local random_device 种子:同起点重复 run 天然独立。
7. `scene_layout.json` 的 `t0/t1` 是每段在整枝上的弧长区间,多段枝必须用它算 allow(t)
   (viz 已修)。
8. 老结果目录(几十个 `benchmark_results_*`)是历史对照,别删。

## 可视化 / Isaac

```bash
python3 viz_cut_scene.py --rank <结果目录>                  # 按 pen/allow 排名全部路径
python3 viz_cut_scene.py --show <路径.csv> --scene <场景.json> --title "..."   # pybullet GUI 回放
python3 viz_cut_scene.py --render <路径.csv> --scene <场景.json> --out x.png   # 离屏截图
```

回放染色 = 预算占用:黄 <50%,橙 50–100%,红 >100%(门控 run 不应出现红)。
Isaac 回放:`replay_isaac.py` + `isaac_calibration_tree6.yaml`(记得 z=0.4)。

## 当前状态与基准数字(2026-08-01)

标准场景 5 起点 × 10 次,起点文件已存仓库:**`saved_starts_final_5x10.txt`**
(5 个 gate-valid 起点 × 各重复 10 行 = 50 行,直接 `--starts-file` 用):

| 配置 | 成功 | 均时 | cuts | 接触深度中位 | 轴向分数 |
|---|---|---|---|---|---|
| 无约束 baseline | 49/50 | 3514ms | **23/49 切枝** | — | 0.54 |
| 当前全栈(= 上面标准命令) | 50/50 | **714ms** | **0/50** | **0.40** | **0.45** |
| 全栈去 stiffness(消融) | 50/50 | 482ms | 0/50 | 0.74 | 0.51 |

关键结论链:引导场永远只能减少"可避免的"伤害,保证必须来自有效性层(gate);
约束+路由对齐后,安全栈反而比无约束快 6.4×。

## 待办(按优先级)

1. **各向异性 nav 过路费**:Dijkstra 边权 `w = toll·(1+k·|edge_dir·â|)` —— 让路由层
   在进入树冠前就偏好径向切入而非沿枝走廊(剩余 0.45 轴向分数主要来自走廊几何,
   每步预算管不到)。已设计未实现。
2. **轴向指标拆分**:`axial_contact_fraction` 把"包络内飞行"也计入;拆成 触碰滑移 /
   包络内飞行 两项(最差轴向 run32 max_pen 仅 13mm,证明大头是飞行)。
3. clamp 0.7→0.5 选项(更浅接触,代价时间)——等 Isaac 回放判断是否需要。
4. 预算标定:用 Isaac 回放实测枝条变形,校准 allow(t)/L_ax(t)/η。
5. **git**:仓库是 git,近期全部改动未 commit——新机器上先 `git status` 看情况,
   建议按机制分块提交。

## 工作约定

- 注释/文档中文;调参改 `types.h`,别在 .cpp 硬编码
- 新机制一律加 CLI 开关,旧机制保留作 ablation,输出目录名带后缀
- 每次算法改动:同 starts 文件 5×10 对照 + `--rank` 深度分布 + cuts 必须 0
