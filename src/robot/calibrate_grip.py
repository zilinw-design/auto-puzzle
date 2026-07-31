"""
抓取高度校准工具 — 到达目标 XY, 手动微调 Z 到刚好吸住, 保存

用法: python calibrate_grip.py <x_cm> <y_cm>

命令:
  +z N  上升 N mm        -z N  下降 N mm
  s     保存当前抓取高度
  r     回到安全高度
  home  复位
  q     退出
"""

import sys, os, json, math
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

def calibrate_one(arm, x, y, coef):
    grip_ik = ikine(x, y, coef, 0)
    safe_ik = ikine(x, y, coef, SAFE_OFFSET)
    cur_z = SAFE_OFFSET

    print(f"\n=== ({x},{y}) safe ik={safe_ik} grip ik={grip_ik} ===")

    arm.home(); arm.wait(2000)
    move_safe(arm, *safe_ik)
    arm.move_by_delta(0, 0, -SAFE_OFFSET); arm.wait(2000)
    cur_dx, cur_dy = 0, 0
    print("[at grip] +/-x/y/z N | s | r | n | q")

    while True:
        c = input("> ").strip().split()
        if not c: continue
        k = c[0].lower()

        if k == 'q': return 'quit'
        elif k == 'n': return 'next'
        elif k == 'home': arm.home(); arm.wait(2000); print("home")
        elif k == 'r':
            arm.move_by_delta(0, 0, SAFE_OFFSET - cur_z); arm.wait(2000)
            cur_z = SAFE_OFFSET; cur_dx = cur_dy = 0
            print("back to safe")
        elif k in ('+x','-x','+y','-y','+z','-z'):
            v = int(c[1]) if len(c) > 1 else 5
            if k == '+x':   arm.move_by_delta(+v,0,0); cur_dx += v
            elif k == '-x': arm.move_by_delta(-v,0,0); cur_dx -= v
            elif k == '+y': arm.move_by_delta(0,+v,0); cur_dy += v
            elif k == '-y': arm.move_by_delta(0,-v,0); cur_dy -= v
            elif k == '+z': arm.move_by_delta(0,0,+v); cur_z -= v
            elif k == '-z': arm.move_by_delta(0,0,-v); cur_z += v
            arm.wait(1500)
            print(f"  d({cur_dx:+d},{cur_dy:+d},z={SAFE_OFFSET-cur_z:+d})mm")
        elif k == 's':
            new_dx = grip_ik[0] + cur_dx
            new_dy = grip_ik[1] + cur_dy
            new_dz = grip_ik[2] + (SAFE_OFFSET - cur_z)
            lib = json.load(open(LIB_FILE))
            updated = False
            for e in lib:
                if abs(e["x"]-x)<0.5 and abs(e["y"]-y)<0.5:
                    if e["layer"] == "grip":
                        e["dx"] = new_dx; e["dy"] = new_dy; e["dz"] = new_dz
                        updated = True
                    elif e["layer"] == "safe":
                        e["dx"] = new_dx; e["dy"] = new_dy; e["dz"] = new_dz + SAFE_OFFSET
            if not updated:
                lib.append({"x":x,"y":y,"dx":new_dx,"dy":new_dy,"dz":new_dz,"layer":"grip"})
                lib.append({"x":x,"y":y,"dx":new_dx,"dy":new_dy,"dz":new_dz+SAFE_OFFSET,"layer":"safe"})
            json.dump(lib, open(LIB_FILE,"w"), indent=2)
            print(f"[saved] ({x},{y}) dx={new_dx} dy={new_dy} dz={new_dz}")

# 所有已标定点 (自动从库中读取)
def all_points():
    lib = json.load(open(LIB_FILE))
    return sorted(set((e["x"], e["y"]) for e in lib))

def main():
    if len(sys.argv) >= 3:
        pts = [(float(sys.argv[i]), float(sys.argv[i+1])) for i in range(1, len(sys.argv), 2)]
    else:
        pts = all_points()
        print(f"默认 {len(pts)} 点: {pts}")
    coef = load_fit()
    arm = ArmController(PORT)
    if not arm.connect(): return

    try:
        for x, y in pts:
            result = calibrate_one(arm, x, y, coef)
            if result == 'quit': break
    except KeyboardInterrupt: print("\nstop")
    finally: arm.disconnect()

if __name__ == "__main__": main()
