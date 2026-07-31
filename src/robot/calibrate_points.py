"""
标定运行 — 记录每个目标点的脉冲值（仅一次）
之后用 record_points.py 直接 CMD 3 发脉冲，不再走 ikine，爪子全程闭合
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM7"
CALIBRATED = {
    (6, -9, -2): (30, -90, -29),
    (6,  9, -2): (45,  82, -29),
}

# 路径
PATH = [
    (6, -9, -2),
    (6, -9, -4),
    (6, -9, -2),
    (6,  9, -2),
    (6,  9, -4),
    (6,  9, -2),
]

arm = ArmController(PORT)
if not arm.connect():
    exit()

recorded = {}
try:
    # 初始复位
    print("初始复位...")
    arm.home(); arm.wait(2000)
    arm.gripper_close(); arm.wait(500)
    current = (0, 0, 0)

    for i, target in enumerate(PATH, 1):
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        dz = target[2] - current[2]

        if abs(dy) > 6:  # 大跨步 → 先复位
            print(f"\n[{i}] → {target} (远距, 先复位)")
            arm.home(); arm.wait(2000)
            arm.gripper_close(); arm.wait(500)
            current = (0, 0, 0)
            dx, dy, dz = target[0], target[1], target[2]

        key = tuple(target)
        if key in CALIBRATED and current == (0, 0, 0):
            cx, cy, cz = CALIBRATED[key]
        else:
            cx = round(dx * 10 * 0.75)
            cy = round(dy * 10 * (0.91 if dy > 0 else 1.0))
            cz = round(dz * 10 * 1.46)

        print(f"  Δ=({dx},{dy},{dz})cm → 指令({cx},{cy},{cz})mm")
        input("  Enter: ")
        arm.move_by_delta(cx, cy, cz)
        arm.wait(1500)
        current = target

        # 读脉冲
        pos = arm.read_positions()
        name = f"({target[0]},{target[1]},{target[2]})"
        recorded[name] = {str(k): v for k, v in pos.items()}
        print(f"  → 脉冲: {' '.join(f'#{s}={pos[s]}' for s in sorted(pos))}")

    # 保存
    with open("recorded_points.json", "w") as f:
        json.dump(recorded, f, indent=2)
    print(f"\n标定完成，保存到 recorded_points.json")

finally:
    arm.disconnect()
