"""读取当前舵机位置 — 快速标定用"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

port = sys.argv[1] if len(sys.argv) > 1 else "COM9"
arm = ArmController(port)
if arm.connect():
    pos = arm.read_positions()
    if pos:
        print("\n复制下面这行:")
        pulse_str = " ".join(f"#{s}={pos[s]}" for s in sorted(pos))
        print(f"  {pulse_str}")
        print()
    arm.disconnect()
