"""
标定 V2 — 加长等待，确保读到稳定脉冲后记录

用法: python calibrate_v2.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM7"
WAIT = 8000   # 每次移动后等 8 秒, 确保完全稳定

CALIBRATED = {
    (6, -9, -2): (30, -90, -29),
    (6,  9, -2): (45,  82, -29),
}

TARGETS = [
    {"name": "(6,-9,-2)", "key": (6, -9, -2)},
    {"name": "(6,-9,-4)", "key": None, "dx": (0, 0, -29)},
    {"name": "(6,-9,-2)", "key": None, "dx": (0, 0, +29)},
    {"name": "(6, 9,-2)", "key": (6,  9, -2)},
    {"name": "(6, 9,-4)", "key": None, "dx": (0, 0, -29)},
    {"name": "(6, 9,-2)", "key": None, "dx": (0, 0, +29)},
]

arm = ArmController(PORT)
if not arm.connect():
    exit()

recorded = {}
try:
    arm.home(); arm.wait(2000)
    arm.gripper_close(time_ms=3000); arm.wait(4000)
    print("复位+闭合完成\n")

    for t in TARGETS:
        name = t["name"]
        if t["key"]:
            cx, cy, cz = CALIBRATED[t["key"]]
            print(f"→ {name}  ikine({cx:+d},{cy:+d},{cz:+d})")
            arm.move_by_delta(cx, cy, cz)
        else:
            cx, cy, cz = t["dx"]
            print(f"→ {name}  dz={cz:+d}")
            arm.move_by_delta(cx, cy, cz)
        print(f"  等 {WAIT/1000:.0f} 秒...")
        arm.wait(WAIT)

        pos = arm.read_positions()
        recorded[name] = {str(k): v for k, v in pos.items()}
        print(f"  记录: {' '.join(f'#{s}={pos[s]}' for s in sorted(pos))}\n")

    with open("recorded_v2.json", "w") as f:
        json.dump(recorded, f, indent=2)
    print(f"标定完成 → recorded_v2.json")

finally:
    arm.disconnect()
