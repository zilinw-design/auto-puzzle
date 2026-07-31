"""
路径工作流 — 逐点移动，每步 Enter 确认

⚠ 夹爪始终闭合（电磁铁胶带固定），所有复位后自动闭合
路径:
  复位→闭 → (6,-9,-2) → (6,-9,-4) → (6,-9,-2) → 复位→闭 → (6,9,-2) → (6,9,-4) → (6,9,-2) → 复位→闭
  单位: cm, 复位=(0,0,0)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM7"

# 从复位到目标的校准指令 (已验证)
CALIBRATED = {
    (6, -9, -2): (30, -90, -29),   # 复位→点A
    (6,  9, -2): (45,  82, -29),   # 复位→点B
}

# 路径点 (x_cm, y_cm, z_cm) 相对复位原点
WAYPOINTS = [
    {"name": "点A",   "pos": (6, -9, -2)},
    {"name": "点A下",  "pos": (6, -9, -4)},
    {"name": "点A",   "pos": (6, -9, -2)},
    {"name": "点B",   "pos": (6,  9, -2), "far": True},
    {"name": "点B下",  "pos": (6,  9, -4)},
    {"name": "点B",   "pos": (6,  9, -2)},
]


def reset(arm):
    """复位坐标 + 立即闭合夹爪 (CMD_SERVOS_RESET 重置 ikine 内部坐标)"""
    arm.home()           # 固件坐标复位, #1→770 瞬间
    arm.wait(2000)
    arm.move_servo(1, 1500, time_ms=300)  # 立即闭合


def delta(from_pt, to_pt):
    return (to_pt[0] - from_pt[0], to_pt[1] - from_pt[1], to_pt[2] - from_pt[2])


def main():
    arm = ArmController(PORT)
    if not arm.connect():
        return

    try:
        print("[0] 复位 (夹爪保持闭合)")
        input("  按 Enter: ")
        reset(arm)
        current = (0, 0, 0)

        for i, wp in enumerate(WAYPOINTS, 1):
            name, target = wp["name"], wp["pos"]
            dx_cm, dy_cm, dz_cm = delta(current, target)

            if wp.get("far"):
                # 远距离: 先复位 → 再校准移动
                print(f"\n[{i}] {name} → {target}  (远距，先复位)")
                cmd = input("  Enter=执行 s=跳过 p=急停: ").strip().lower()
                if cmd == 'p':
                    arm.panic("用户急停"); return
                elif cmd == 's':
                    current = target; continue

                print("    复位中 (夹爪保持闭合)...")
                reset(arm)
                current = (0, 0, 0)

                if target in CALIBRATED:
                    cx, cy, cz = CALIBRATED[target]
                else:
                    dx_mm, dy_mm, dz_mm = dx_cm*10, dy_cm*10, dz_cm*10
                    cx = round(dx_mm * 0.75)
                    cy = round(dy_mm * (0.91 if dy_mm > 0 else 1.0))  # Y正: 82/90≈0.91
                    cz = round(dz_mm * 1.46)

                print(f"    指令: {cx:+d} {cy:+d} {cz:+d} mm")
                arm.move_by_delta(cx, cy, cz)
                arm.wait(1500)
                current = target

            else:
                # 小步直接用系数
                dx_mm, dy_mm, dz_mm = dx_cm*10, dy_cm*10, dz_cm*10
                cx = round(dx_mm * 0.75)
                cy = round(dy_mm * (0.91 if dy_mm > 0 else 1.0))
                cz = round(dz_mm * 1.46)

                print(f"\n[{i}] {name} → {target}  Δ=({dx_cm:+d},{dy_cm:+d},{dz_cm:+d})cm")
                print(f"    指令: {cx:+d} {cy:+d} {cz:+d} mm")
                cmd = input("  Enter=执行 s=跳过 p=急停: ").strip().lower()
                if cmd == 'p':
                    arm.panic("用户急停"); return
                elif cmd == 's':
                    current = target; continue

                arm.move_by_delta(cx, cy, cz)
                arm.wait(1500)
                current = target

        print(f"\n[最后] 复位 (夹爪保持闭合)")
        input("  按 Enter: ")
        reset(arm)
        print("\n路径完成。")

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
