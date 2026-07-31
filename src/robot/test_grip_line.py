"""
Y=0 线拾取测试: X=0,5,10,15,20

用法: python test_grip_line.py
到达最低点后: Enter=继续上升到下一个, Ctrl+C=急停, s=跳过此点
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController
from pick_and_place import idw_xyz, move_safe, vision_to_arm

PORT = "COM7"
# 边框8点 + 中心
POINTS = [(0,-14.7),(0,0),(0,14.7),(10.45,-14.7),(10.45,0),(10.45,14.7),(20.9,-14.7),(20.9,0),(20.9,14.7)]

def main():
    a = ArmController(PORT)
    if not a.connect(): return

    try:
        for pt in POINTS:
            x, y = pt if isinstance(pt, tuple) else (pt, 0)
            px, py = vision_to_arm(x, y)
            s = idw_xyz(px, py, "safe")
            g = idw_xyz(px, py, "grip")
            dz = g[2] - s[2]

            print(f"\n=== ({x},{y}) safe={s} grip={g} dz={dz} ===")
            print("[0] home")
            a.home(); a.wait(2000)

            print("[1] safe")
            move_safe(a, *s)

            print("[2] descend")
            if abs(dz) > 30:
                a.move_by_delta(0, 0, dz//2); a.wait(1500)
                a.move_by_delta(0, 0, dz - dz//2); a.wait(1500)
            else:
                a.move_by_delta(0, 0, dz); a.wait(2000)

            print(f"[AT GRIP] 空格=急停 Enter=继续 s=跳过")
            try:
                import msvcrt
                print("  (按空格急停, 其他键继续...)")
                key = msvcrt.getch()
                if key == b' ':
                    raise KeyboardInterrupt("空格急停")
                c = key.decode('utf-8', errors='ignore').strip().lower()
            except ImportError:
                c = input("  > ").strip().lower()
            if c == 's':
                a.move_by_delta(0, 0, -dz); a.wait(2000)
                a.home(); a.wait(2000)
                continue

            print("[3] rise")
            a.move_by_delta(0, 0, -dz); a.wait(2000)

            print("[4] home")
            a.home(); a.wait(2000)

        print("\n全部完成。")

    except KeyboardInterrupt:
        print("\n急停!")
    finally:
        a.disconnect()

if __name__ == "__main__":
    main()
