"""
全局线性拟合: 16 点 ikine 数据 → 平面拟合 dx/dy/dz
安全 = grip + 10mm (Z)
用法: python test_solver.py <x_cm> <y_cm>
"""

import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM9"
LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ikine_lib.json")

def fit():
    lib = json.load(open(LIB_FILE))
    X, Y, DX, DY, DZ = [], [], [], [], []
    for e in lib:
        if e["layer"] == "grip":
            X.append(e["x"]); Y.append(e["y"])
            DX.append(e["dx"]); DY.append(e["dy"]); DZ.append(e["dz"] - 15)
    A = np.column_stack([X, Y, np.ones(len(X))])
    coef_dx = np.linalg.lstsq(A, DX, rcond=None)[0]
    coef_dy = np.linalg.lstsq(A, DY, rcond=None)[0]
    coef_dz = np.linalg.lstsq(A, DZ, rcond=None)[0]
    return coef_dx, coef_dy, coef_dz

def predict(x, y, coef):
    return round(coef[0]*x + coef[1]*y + coef[2])

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

    cx, cy, cz = fit()
    dx = predict(x, y, cx)
    dy = predict(x, y, cy)
    dz = predict(x, y, cz)
    safe_dz = dz + 10

    # 残差分析
    lib = json.load(open(LIB_FILE))
    print(f"=== 线性拟合 ({len([e for e in lib if e['layer']=='grip'])}点) ===")
    print(f"  dx = {cx[0]:.1f}*X + {cx[1]:.1f}*Y + {cx[2]:.0f}")
    print(f"  dy = {cy[0]:.1f}*X + {cy[1]:.1f}*Y + {cy[2]:.0f}")
    max_err = 0
    for e in lib:
        if e["layer"] == "grip":
            px = predict(e["x"], e["y"], cx)
            py = predict(e["x"], e["y"], cy)
            err = math.hypot(px-e["dx"], py-e["dy"])
            max_err = max(max_err, err)
            if err > 20:
                print(f"  ⚠ ({e['x']:.1f},{e['y']:.1f}) predict=({px},{py}) real=({e['dx']},{e['dy']}) err={err:.0f}")
    print(f"  最大残差: {max_err:.0f}")

    print(f"\n({x},{y}) 预测: dx={dx} dy={dy} dz(grip)={dz} dz(safe)={safe_dz}")

    arm = ArmController(PORT)
    if not arm.connect(): return
    try:
        print("[0] home"); input("  Enter: "); arm.home(); arm.wait(2000)
        print(f"[1] safe ik({dx},{dy},{safe_dz})"); input("  Enter: ")
        move_safe(arm, dx, dy, safe_dz); arm.wait(2000)
        print("[2] home"); input("  Enter: "); arm.home(); arm.wait(2000)
        print("done.")
    except KeyboardInterrupt: print("stop")
    finally: arm.disconnect()

if __name__ == "__main__": main()
