"""
pick_and_place.py -- LeArm pick-and-place workflow

Usage:
  python pick_and_place.py <pick_x> <pick_y> <place_x> <place_y>
  python pick_and_place.py 6 -9 6 9

Flow: home -> pick(safe->descend->wait3s->rise) -> home -> place(safe->descend->wait3s->rise) -> home
"""

import sys, os, json, math
import numpy as np

from config import PORT
from arm_controller import ArmController

WAIT_GRIP = 3000
WAIT_PLACE = 3000
SAFE_OFFSET = 10   # mm above grip
MOVE_SETTLE = 2000  # wait after big move

LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ikine_lib.json")


def load_fit():
    """Linear fit: dx = a1*X + b1*Y + c1, same for dy, dz"""
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


def ikine(x, y, coef, dz_offset=0):
    cx, cy, cz = coef
    return (round(cx[0]*x + cx[1]*y + cx[2]),
            round(cy[0]*x + cy[1]*y + cy[2]),
            round(cz[0]*x + cz[1]*y + cz[2] + dz_offset))


def move_safe(arm, dx, dy, dz):
    """Send ikine delta, auto-split if exceeds +/-128"""
    steps = max(1, max(abs(dx), abs(dy), abs(dz)) // 100 + 1)
    for i in range(steps):
        sx = ((i+1)*dx//steps) - (i*dx//steps)
        sy = ((i+1)*dy//steps) - (i*dy//steps)
        sz = ((i+1)*dz//steps) - (i*dz//steps)
        if sx or sy or sz:
            arm.move_by_delta(sx, sy, sz)
            arm.wait(800)
    arm.wait(MOVE_SETTLE)


def main():
    if len(sys.argv) < 5:
        print("Usage: python pick_and_place.py <pick_x> <pick_y> <place_x> <place_y>")
        return
    px, py = float(sys.argv[1]), float(sys.argv[2])
    tx, ty = float(sys.argv[3]), float(sys.argv[4])

    coef = load_fit()
    p_safe = ikine(px, py, coef, SAFE_OFFSET)
    t_safe = ikine(tx, ty, coef, SAFE_OFFSET)

    print(f"Pick ({px},{py}): safe={p_safe}")
    print(f"Place({tx},{ty}): safe={t_safe}")

    arm = ArmController(PORT)
    if not arm.connect(): return

    try:
        # -- home --
        print("[0] home"); arm.home(); arm.wait(2000)

        # -- pick --
        print(f"[1] pick safe {p_safe}")
        move_safe(arm, *p_safe)

        print("[2] descend Z-10")
        arm.move_by_delta(0, 0, -SAFE_OFFSET); arm.wait(2000)

        print(f"[3] grip wait {WAIT_GRIP/1000}s")
        arm.wait(WAIT_GRIP)

        print("[4] rise Z+10")
        arm.move_by_delta(0, 0, +SAFE_OFFSET); arm.wait(2000)

        # -- home --
        print("[5] home"); arm.home(); arm.wait(2000)

        # -- place --
        print(f"[6] place safe {t_safe}")
        move_safe(arm, *t_safe)

        print("[7] descend Z-10")
        arm.move_by_delta(0, 0, -SAFE_OFFSET); arm.wait(2000)

        print(f"[8] release wait {WAIT_PLACE/1000}s")
        arm.wait(WAIT_PLACE)

        print("[9] rise Z+10")
        arm.move_by_delta(0, 0, +SAFE_OFFSET); arm.wait(2000)

        # -- home --
        print("[10] home"); arm.home(); arm.wait(2000)
        print("done.")

    except KeyboardInterrupt:
        print("stop")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
