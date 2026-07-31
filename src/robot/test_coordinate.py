"""
测试 CMD_COORDINATE_SET (4) — 走上位机同款 ikine()

固件复位后初始坐标: X=15cm  Y=0cm  Z=2cm
每次发 Δx,Δy,Δz (单位mm)，固件内部累加后调用 LeArm.lib ikine()

用法: python test_coordinate.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM7"


def main():
    arm = ArmController(PORT)
    if not arm.connect():
        return

    try:
        # 步骤1: 复位 → 初始 (15, 0, 2) cm
        print("=" * 50)
        print("  坐标控制测试 — CMD_COORDINATE_SET (4)")
        print("  复位后初始: X=15cm  Y=0cm  Z=2cm")
        print("=" * 50)

        print("\n[1/3] 复位")
        input("  按 Enter: ")
        arm.home()
        arm.wait(2000)
        pos = arm.read_positions()
        if pos:
            print(f"  脉冲: {' '.join(f'#{s}={pos[s]}' for s in sorted(pos))}")

        # 步骤2: 移动测试 — 向 Y 正方向移 50mm
        print(f"\n[2/3] Y+50mm (向右)")
        print(f"  固件将: y=0→5cm, 内部调用 ikine()")
        input("  按 Enter: ")
        arm.move_by_delta(dx_mm=0, dy_mm=50, dz_mm=0)
        arm.wait(2500)
        pos = arm.read_positions()
        if pos:
            print(f"  脉冲: {' '.join(f'#{s}={pos[s]}' for s in sorted(pos))}")

        # 步骤3: 复位回初始
        print(f"\n[3/3] 复位")
        input("  按 Enter: ")
        arm.home()
        arm.wait(2000)

        print("\n测试完成。")

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
