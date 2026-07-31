"""
测试剩余8个内部点
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController
from pick_and_place import idw_xyz, move_safe, vision_to_arm

PORT = "COM7"
REMAINING = [(3,4),(6,-7),(6,7),(10,-7),(10,7),(16,-7),(16,7),(17,-4)]

def main():
    a = ArmController(PORT)
    if not a.connect(): return
    try:
        for x,y in REMAINING:
            px,py = vision_to_arm(x,y)
            s = idw_xyz(px,py,"safe"); g = idw_xyz(px,py,"grip")
            dz = g[2]-s[2]
            print(f"\n=== ({x},{y}) dz={dz} ===")
            if input("Enter=跑 s=跳过: ").strip()=='s': continue
            a.home(); a.wait(2000)
            move_safe(a,*s)
            a.move_by_delta(0,0,dz); a.wait(2000)
            a.wait(3000)
            a.move_by_delta(0,0,-dz); a.wait(2000)
            a.home(); a.wait(2000)
        print("\nDone.")
    except KeyboardInterrupt: print("Stop")
    finally: a.disconnect()

if __name__=="__main__": main()
