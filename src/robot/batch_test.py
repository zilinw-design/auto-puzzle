"""
批量测试 — 17点全跑: 复位→安全→下降→上升→复位→下一个

用法: python batch_test.py
"""

import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM7"
SAFE_OFFSET = 15
LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ikine_lib.json")

def load_fit():
    lib = json.load(open(LIB_FILE))
    X, Y, DX, DY, DZ = [], [], [], [], []
    for e in lib:
        if e["layer"] == "grip":
            X.append(e["x"]); Y.append(e["y"])
            DX.append(e["dx"]); DY.append(e["dy"]); DZ.append(e["dz"])
    A = np.column_stack([X, Y, np.ones(len(X))])
    return (np.linalg.lstsq(A, DX, rcond=None)[0],
            np.linalg.lstsq(A, DY, rcond=None)[0],
            np.linalg.lstsq(A, DZ, rcond=None)[0])

def ikine(x, y, coef, dz_off=0):
    cx, cy, cz = coef
    return (round(cx[0]*x + cx[1]*y + cx[2]),
            round(cy[0]*x + cy[1]*y + cy[2]),
            round(cz[0]*x + cz[1]*y + cz[2] + dz_off))

def move_safe(arm, dx, dy, dz):
    steps = max(1, max(abs(dx), abs(dy), abs(dz)) // 100 + 1)
    for i in range(steps):
        sx = ((i+1)*dx//steps) - (i*dx//steps)
        sy = ((i+1)*dy//steps) - (i*dy//steps)
        sz = ((i+1)*dz//steps) - (i*dz//steps)
        if sx or sy or sz:
            arm.move_by_delta(sx, sy, sz)
            arm.wait(800)
    arm.wait(2000)

def all_points():
    lib = json.load(open(LIB_FILE))
    return sorted(set((e["x"], e["y"]) for e in lib))

def main():
    pts = all_points()
    coef = load_fit()
    total = len(pts)
    print(f"{total} 点批量测试")
    arm = ArmController(PORT)
    if not arm.connect(): return

    try:
        for i, (x, y) in enumerate(pts, 1):
            safe = ikine(x, y, coef, SAFE_OFFSET)
            grip = ikine(x, y, coef, 0)
            print(f"\n[{i}/{total}] ({x},{y}) safe={safe} grip={grip}")
            if input("  Enter=跑 s=跳过: ").strip().lower() == 's': continue

            arm.home(); arm.wait(2000)
            print("  -> safe")
            move_safe(arm, *safe)
            print("  -> descend")
            arm.move_by_delta(0, 0, -SAFE_OFFSET); arm.wait(2000)
            print("  -> rise")
            arm.move_by_delta(0, 0, +SAFE_OFFSET); arm.wait(2000)
            print("  -> home")
            arm.home(); arm.wait(2000)
        print(f"\n全部 {total} 点完成。")

    except KeyboardInterrupt: print("\nstop")
    finally: arm.disconnect()

if __name__ == "__main__": main()
