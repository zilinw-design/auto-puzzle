"""
路径工作流 V11 — CMD_COORDINATE_SET (已验证准) + 固件 3s 慢速

固件已改: 复位闭爪 + ikine 3000ms
用已验证的校准 ikine 指令，位置准。

用法: python run_path_v2.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM9"
SETTLE = 500  # ikine 3s 后额外等 0.5s

# 从复位出发的 ikine 指令 (已验证)
CALIBRATED = {
    "A": (30, -90,   3),   # → (6,-9,+2)
    "B": (45,  82,   3),   # → (6, 9,+2)
}
Z_DOWN = (0, 0, -36)   # 安全→抓取
Z_UP   = (0, 0, +36)


def main():
    arm = ArmController(PORT)
    if not arm.connect():
        return

    try:
        print("[0] 复位")
        input("  按 Enter: ")
        arm.home(); arm.wait(3500)

        # ── A ──
        cx, cy, cz = CALIBRATED["A"]
        print(f"\n[1] → A(6,-9)  {cx:+d},{cy:+d},{cz:+d}mm")
        input("  按 Enter: ")
        arm.move_by_delta(cx, cy, cz); arm.wait(3500 + SETTLE)

        print(f"\n[2] → 抓取")
        input("  按 Enter: ")
        arm.move_by_delta(*Z_DOWN); arm.wait(3500 + SETTLE)

        print(f"\n[3] → 上升")
        input("  按 Enter: ")
        arm.move_by_delta(*Z_UP); arm.wait(3500 + SETTLE)

        # ── 复位 → B ──
        print(f"\n[4] 复位")
        input("  按 Enter: ")
        arm.home(); arm.wait(3500)

        cx, cy, cz = CALIBRATED["B"]
        print(f"\n[5] → B(6,+9)  {cx:+d},{cy:+d},{cz:+d}mm")
        input("  按 Enter: ")
        arm.move_by_delta(cx, cy, cz); arm.wait(3500 + SETTLE)

        print(f"\n[6] → 抓取")
        input("  按 Enter: ")
        arm.move_by_delta(*Z_DOWN); arm.wait(3500 + SETTLE)

        print(f"\n[7] → 上升")
        input("  按 Enter: ")
        arm.move_by_delta(*Z_UP); arm.wait(3500 + SETTLE)

        print(f"\n[8] 复位")
        input("  按 Enter: ")
        arm.home(); arm.wait(3500)
        print("\n完成。")

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
