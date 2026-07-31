"""
拾放工作流: 复位 → 拾取(抓+等3s) → 释放(放+等3s) → 复位

用法: python pick_and_place.py <pick_x> <pick_y> <place_x> <place_y>
      python pick_and_place.py 6 -9 6 9
"""

import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM7"
WAIT_GRIP = 3000
WAIT_PLACE = 3000
SAFE_OFFSET = 15
VISION_TO_ARM_Y = 14.7 / 14.85

def vision_to_arm(x, y):
    return x, y * VISION_TO_ARM_Y

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

def idw_xyz(x, y, layer="grip"):
    """IDW 查 dx/dy/dz, layer=grip/safe. 库中 dz 已精调"""
    lib = json.load(open(LIB_FILE))
    pts = [e for e in lib if e["layer"] == layer]
    ws = {"dx": 0.0, "dy": 0.0, "dz": 0.0}
    norm = 0.0
    for e in pts:
        d = ((x - e["x"])**2 + (y - e["y"])**2)**0.5
        if d < 0.3:
            return (e["dx"], e["dy"], e["dz"])
        w = 1.0 / (d * d)
        norm += w
        ws["dx"] += w * e["dx"]
        ws["dy"] += w * e["dy"]
        ws["dz"] += w * e["dz"]
    return (round(ws["dx"]/norm), round(ws["dy"]/norm), round(ws["dz"]/norm))

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
    if len(sys.argv) < 5:
        print("用法: python pick_and_place.py <pick_x> <pick_y> <place_x> <place_y>")
        return
    px, py = float(sys.argv[1]), float(sys.argv[2])
    tx, ty = float(sys.argv[3]), float(sys.argv[4])
    px, py = vision_to_arm(px, py)
    tx, ty = vision_to_arm(tx, ty)

    p_safe = idw_xyz(px, py, "safe")
    p_grip = idw_xyz(px, py, "grip")
    t_safe = idw_xyz(tx, ty, "safe")
    t_grip = idw_xyz(tx, ty, "grip")

    print(f"取({px},{py}): safe={p_safe} grip={p_grip}")
    print(f"放({tx},{ty}): safe={t_safe} grip={t_grip}")

    arm = ArmController(PORT)
    if not arm.connect(): return
    try:
        # ── 复位 ──
        print("[0] home"); input("  Enter: "); arm.home(); arm.wait(2000)

        # ── 拾取: 安全 → 纯Z下降10mm → 等3s → 纯Z上升 ──
        print(f"\n[1] pick → safe{p_safe}"); input("  Enter: ")
        move_safe(arm, *p_safe); arm.wait(2000)

        dz_down = p_grip[2] - p_safe[2]
        print(f"[2] descend {dz_down} → grip"); input("  Enter: ")
        if abs(dz_down) > 30:
            arm.move_by_delta(0, 0, dz_down//2); arm.wait(1500)
            arm.move_by_delta(0, 0, dz_down - dz_down//2); arm.wait(1500)
        else:
            arm.move_by_delta(0, 0, dz_down); arm.wait(2000)

        print(f"[3] grip wait {WAIT_GRIP/1000}s"); input("  Enter: ")
        arm.wait(WAIT_GRIP)

        print(f"[4] rise {-dz_down} → safe"); input("  Enter: ")
        arm.move_by_delta(0, 0, -dz_down); arm.wait(2000)

        # ── 复位 → 释放: 安全 → 下降 → 等3s → 上升 ──
        print(f"\n[5] home"); input("  Enter: ")
        arm.home(); arm.wait(2000)

        print(f"[6] place → safe{t_safe}"); input("  Enter: ")
        move_safe(arm, *t_safe); arm.wait(2000)

        dz_down2 = t_grip[2] - t_safe[2]
        print(f"[7] descend {dz_down2} → grip"); input("  Enter: ")
        if abs(dz_down2) > 30:
            arm.move_by_delta(0, 0, dz_down2//2); arm.wait(1500)
            arm.move_by_delta(0, 0, dz_down2 - dz_down2//2); arm.wait(1500)
        else:
            arm.move_by_delta(0, 0, dz_down2); arm.wait(2000)

        print(f"[8] release wait {WAIT_PLACE/1000}s"); input("  Enter: ")
        arm.wait(WAIT_PLACE)

        print(f"[9] rise {-dz_down2} → safe"); input("  Enter: ")
        arm.move_by_delta(0, 0, -dz_down2); arm.wait(2000)

        # ── 最后复位 ──
        print(f"\n[10] home"); input("  Enter: ")
        arm.home(); arm.wait(2000)
        print("\ndone.")

    except KeyboardInterrupt: print("\nstop")
    finally: arm.disconnect()

if __name__ == "__main__": main()
