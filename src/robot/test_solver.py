"""
测试: XY线性拟合 + Z IDW, 单点或批量

用法: python test_solver.py <x> <y>      单点
      python test_solver.py              批量17点
"""

import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM7"
LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ikine_lib.json")

def fit():
    lib = json.load(open(LIB_FILE))
    X, Y, DX, DY = [], [], [], []
    for e in lib:
        if e["layer"] == "grip":
            X.append(e["x"]); Y.append(e["y"])
            DX.append(e["dx"]); DY.append(e["dy"])
    A = np.column_stack([X, Y, np.ones(len(X))])
    return np.linalg.lstsq(A, DX, rcond=None)[0], np.linalg.lstsq(A, DY, rcond=None)[0]

def predict(x, y, coef):
    return round(coef[0]*x + coef[1]*y + coef[2])

def all_pts():
    lib = json.load(open(LIB_FILE))
    return sorted(set((e["x"], e["y"]) for e in lib))

def move_safe(arm, dx, dy, dz):
    steps = max(1, max(abs(dx), abs(dy), abs(dz))//100+1)
    for i in range(steps):
        sx = ((i+1)*dx//steps)-(i*dx//steps)
        sy = ((i+1)*dy//steps)-(i*dy//steps)
        sz = ((i+1)*dz//steps)-(i*dz//steps)
        if sx or sy or sz: arm.move_by_delta(sx, sy, sz); arm.wait(800)
    arm.wait(2000)

def test_one(arm, x, y, cx, cy):
    dx = predict(x, y, cx)
    dy = predict(x, y, cy)
    from pick_and_place import idw_xyz
    _, _, dz = idw_xyz(x, y, "grip")
    safe_dz = dz + 35

    print(f"\n=== ({x},{y}) dx={dx} dy={dy} grip_dz={dz} ===")
    arm.home(); arm.wait(1000)
    move_safe(arm, dx, dy, safe_dz); arm.wait(2000)

    arm.move_by_delta(0, 0, -(safe_dz - dz)); arm.wait(2000)
    total_down = safe_dz - dz
    print("[at grip] Enter=up s=skip -z N=down p=panic")
    c = input("  > ").strip().lower()
    if c == 'p': return 'panic'
    if c == 's': arm.move_by_delta(0, 0, total_down); arm.wait(2000); arm.home(); arm.wait(1000); return 'next'
    if c.startswith('-z'):
        v = int(c.split()[1]) if len(c.split())>1 else 5
        arm.move_by_delta(0, 0, -v); arm.wait(1500); total_down += v
        print(f"  down={total_down}mm")
    arm.move_by_delta(0, 0, total_down); arm.wait(2000)
    arm.home(); arm.wait(1000)
    return 'next'

def main():
    cx, cy = fit()
    lib = json.load(open(LIB_FILE))
    grips = [e for e in lib if e['layer']=='grip']
    print(f"=== Linear Fit ({len(grips)}pts) ===")
    print(f"  dx = {cx[0]:.1f}*X + {cx[1]:.1f}*Y + {cx[2]:.0f}")
    print(f"  dy = {cy[0]:.1f}*X + {cy[1]:.1f}*Y + {cy[2]:.0f}")

    if len(sys.argv) >= 3:
        pts = [(float(sys.argv[1]), float(sys.argv[2]))]
    else:
        pts = all_pts()
        print(f"Batch {len(pts)} pts. p=panic s=skip")

    arm = ArmController(PORT)
    if not arm.connect(): return
    try:
        for x, y in pts:
            r = test_one(arm, x, y, cx, cy)
            if r == 'panic': print("PANIC!"); break
    except KeyboardInterrupt: print("STOP")
    finally: arm.disconnect()
    print("done.")

if __name__ == "__main__": main()
