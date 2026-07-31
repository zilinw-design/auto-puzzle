"""
测试标定精度 — 移到 (8cm, 15cm) 即 (80mm, 150mm)

用法: python test_calibration.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_controller import ArmController
from calibration import interpolate, safe_pose, X_RANGE, Y_RANGE

PORT = "COM7"
TARGET_X = 80   # mm, X轴: 下→上
TARGET_Y = 150  # mm, Y轴: 左→右


def main():
    # 计算目标脉冲 (IDW 插值)
    target = interpolate(TARGET_X, TARGET_Y)
    safe   = safe_pose()

    print("=" * 55)
    print("  标定精度测试")
    print(f"  目标: X={TARGET_X}mm  Y={TARGET_Y}mm  ({TARGET_X/10:.0f}cm, {TARGET_Y/10:.0f}cm)")
    print(f"  坐标系: 原点=左下 X↑ Y→")
    print("=" * 55)
    print(f"\n  安全高度: {' '.join(f'#{s}={safe[s]}' for s in sorted(safe))}")
    print(f"  目标位姿: {' '.join(f'#{s}={target[s]}' for s in sorted(target))}")

    # 连接
    arm = ArmController(PORT)
    if not arm.connect():
        return

    try:
        # 步骤1: 复位
        print(f"\n[1/2] 复位到安全高度")
        input("  按 Enter 执行: ")
        arm.home()
        arm.wait(2000)

        safe_pos = arm.read_positions()
        if safe_pos:
            print(f"  到位: {' '.join(f'#{s}={safe_pos[s]}' for s in sorted(safe_pos))}")

        # 步骤2: 移到目标
        print(f"\n[2/2] 移到目标位 ({TARGET_X/10:.0f}cm, {TARGET_Y/10:.0f}cm)")
        print(f"  脉冲: {' '.join(f'#{s}={target[s]}' for s in sorted(target))}")
        input("  按 Enter 执行 (p=急停): ")

        arm.move_to_pose(target, time_ms=2000)
        arm.wait(2500)

        actual = arm.read_positions()
        if actual:
            print(f"\n  指令 vs 实际:")
            for s in sorted(target):
                delta = actual[s] - target[s]
                mark = " ⚠" if abs(delta) > 20 else ""
                print(f"    #{s}: {target[s]:4d} → {actual[s]:4d}  (Δ{delta:+d}){mark}")

        print(f"\n测试完成。机械臂停在目标位置。")
        print(f"用上位机验证该位置是否正确，或目视确认机械臂末端是否在 (8, 15)cm 处。")

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
