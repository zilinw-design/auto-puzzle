"""
校准后坐标移动 — 修正 ikine() 的系统偏差

标定数据 (复位点夹爪闭合 → 指令位移 → 实测位移):
  X+30mm → X+44mm Y+12mm
  Y+30mm → X+7mm  Y+32mm
  Z-10mm → Z+2mm  (Z轴暂不修正)

用法: python test_calibrated_move.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM7"

# 修正系数 (实测/指令 的倒数)
# X: 实测=1.33×指令 → 系数=1/1.33≈0.75
# Y: 实测≈指令       → 系数=1.0
# Z: 实测=0.55×指令 → 系数=1/0.55≈1.82
X_SCALE = 0.75
Y_SCALE = 0.67
Z_SCALE = 1.46


def corrected_move(arm, dx_mm, dy_mm, dz_mm):
    """校准后的增量移动"""
    cx = round(dx_mm * X_SCALE)
    cy = round(dy_mm * Y_SCALE)
    cz = round(dz_mm * Z_SCALE)
    print(f"  期望: ({dx_mm:+d}, {dy_mm:+d}, {dz_mm:+d})mm")
    print(f"  指令: ({cx:+d}, {cy:+d}, {cz:+d})mm")
    arm.move_by_delta(cx, cy, cz)
    arm.wait(1000)


def main():
    arm = ArmController(PORT)
    if not arm.connect():
        return

    try:
        # 复位 + 闭合夹爪
        print("复位 + 闭合夹爪...")
        arm.home()
        arm.wait(2000)
        arm.gripper_close()
        arm.wait(2000)

        # 用你上次指令的目标: X+6cm=60mm, Y+1cm=10mm, Z-2cm=-20mm
        print("\n目标: X+60mm  Y+10mm  Z-20mm")
        input("按 Enter 执行: ")
        corrected_move(arm, 60, 10, -20)
        arm.wait(1000)

        print("\n测量末端实际位移后告诉我。")

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
