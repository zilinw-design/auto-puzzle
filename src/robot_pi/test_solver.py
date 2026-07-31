"""
test_solver.py -- test single point via linear-fit ikine (safe height only)

Usage: python test_solver.py <x_cm> <y_cm>
"""

import sys, os, json, math
import numpy as np

from config import PORT
from arm_controller import ArmController

LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ikine_lib.json")
SAFE_OFFSET = 15

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

def main():
    x = float(sys.argv[1]) if len(sys.argv) > 1 else 6
    y = float(sys.argv[2]) if len(sys.argv) > 2 else -9

    coef = load_fit()
    dx, dy, dz = ikine(x, y, coef, 0)
    sx, sy, sz = ikine(x, y, coef, SAFE_OFFSET)

    print(f"({x},{y}) grip=({dx},{dy},{dz}) safe=({sx},{sy},{sz})")

    arm = ArmController(PORT)
    if not arm.connect(): return
    try:
        print("[0] home"); arm.home(); arm.wait(2000)
        print(f"[1] safe ({sx},{sy},{sz})")
        move_safe(arm, sx, sy, sz); arm.wait(2000)
        print("[2] home"); arm.home(); arm.wait(2000)
        print("done.")
    except KeyboardInterrupt: print("stop")
    finally: arm.disconnect()

if __name__ == "__main__": main()
