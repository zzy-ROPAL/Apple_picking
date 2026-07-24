#!/usr/bin/env python3
"""
calib_compare.py — 对比两台电脑上生成的 FK 标定 CSV

只依赖 Python 标准库(不需要 pybullet / numpy / Isaac),
可以在电脑 A 或电脑 B 上任意一台运行, 把另一台的 CSV 拷过来即可。

用法:
  python3 calib_compare.py calib_bullet_fk.csv calib_isaac_fk.csv
  python3 calib_compare.py a.csv b.csv --pos-tol 1.0 --rot-tol 0.1

判定标准(默认): 位置误差 < 1 mm, 姿态误差 < 0.1 deg。
CSV 格式(两个标定脚本的输出): name,q1..q7,x,y,z,qx,qy,qz,qw
退出码: 0 = 全部 PASS, 1 = 有 FAIL 或数据缺失。
"""

import argparse
import csv
import math
import sys


def load(path):
    rows = {}
    try:
        with open(path) as f:
            for r in csv.reader(f):
                if not r or r[0] == "name":
                    continue
                if len(r) != 15:
                    sys.exit(f"[Error] {path} 中行长度不是 15 列: {r}")
                rows[r[0]] = [float(v) for v in r[1:]]
    except FileNotFoundError:
        sys.exit(f"[Error] 找不到文件: {path}")
    if not rows:
        sys.exit(f"[Error] {path} 是空的")
    return rows


def quat_angle_deg(qa, qb):
    """两个 xyzw 四元数之间的夹角(度), 对 q 和 -q 等价。
    用 ||qa ∓ qb|| = 2 sin(θ/4) 反解, 小角度下比 acos(dot) 数值稳定得多。"""
    s = 1.0 if sum(a * b for a, b in zip(qa, qb)) >= 0 else -1.0
    d = math.sqrt(sum((a - s * b) ** 2 for a, b in zip(qa, qb)))
    return math.degrees(4.0 * math.asin(min(1.0, d / 2.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a", help="第一份 FK CSV (如 calib_bullet_fk.csv)")
    ap.add_argument("csv_b", help="第二份 FK CSV (如 calib_isaac_fk.csv)")
    ap.add_argument("--pos-tol", type=float, default=1.0, help="位置容差 (mm)")
    ap.add_argument("--rot-tol", type=float, default=0.1, help="姿态容差 (deg)")
    args = ap.parse_args()

    a, b = load(args.csv_a), load(args.csv_b)
    ok = True

    print(f"对比: {args.csv_a}  vs  {args.csv_b}")
    print(f"容差: 位置 < {args.pos_tol} mm, 姿态 < {args.rot_tol} deg\n")
    print(f"{'config':<10} {'pos_err(mm)':>12} {'rot_err(deg)':>13} "
          f"{'q_match':>8}  verdict")

    for name in a:
        if name not in b:
            print(f"{name:<10}  —— 在 {args.csv_b} 中缺失")
            ok = False
            continue
        qa_joints, qb_joints = a[name][0:7], b[name][0:7]
        # 先确认两边用的关节角本身一致, 否则位姿对比无意义
        q_match = all(abs(x - y) < 1e-6 for x, y in zip(qa_joints, qb_joints))
        pa, pb = a[name][7:10], b[name][7:10]
        qa, qb = a[name][10:14], b[name][10:14]
        pe = math.dist(pa, pb) * 1000.0
        re = quat_angle_deg(qa, qb)
        passed = q_match and pe < args.pos_tol and re < args.rot_tol
        ok = ok and passed
        print(f"{name:<10} {pe:>12.4f} {re:>13.5f} "
              f"{'yes' if q_match else 'NO!':>8}  {'PASS' if passed else 'FAIL'}")

    extra = [n for n in b if n not in a]
    if extra:
        print(f"\n[Warn] {args.csv_b} 中有 {args.csv_a} 没有的构型: {extra}")

    if ok:
        print("\n==> 标定通过: 两个仿真器 FK 一致, 可以开始回放实验")
    else:
        print("\n==> 标定失败, 常见原因排查顺序:\n"
              "    1. q_match=NO! → 两边测试构型不一致, 检查脚本是否同版本\n"
              "    2. 位置差恒定平移 → 机器人基座位姿不一致\n"
              "    3. 姿态差恒定 45° 或 90° → EE frame 不同 (panda_link8 vs panda_hand)\n"
              "    4. 误差随构型乱变 → 关节顺序/符号不一致, 或 URDF 版本不同\n"
              "    5. 位置差 ×1000 → 单位不一致 (mm vs m, 检查 stage units)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
