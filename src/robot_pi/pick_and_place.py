"""
pick_and_place.py -- LeArm automated pick-and-place (Raspberry Pi)

Usage:
  python pick_and_place.py <pick_x> <pick_y> <place_x> <place_y>

Flow: home -> safe -> descend -> magnet ON(3s) -> rise -> home
           -> safe -> descend -> magnet OFF(3s) -> rise -> home
"""

import sys, os, json, time
import numpy as np
from config import PORT
from arm_controller import ArmController

WAIT_GRIP = 3000
WAIT_PLACE = 3000
SAFE_ABOVE = 35     # safe = grip + 35mm
PAUSE_SAFE = 0.5    # pause at safe height
VISION_TO_ARM_Y = 14.7 / 14.85

# GPIO
GPIO_PIN = 17
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_PIN, GPIO.OUT)
    GPIO.output(GPIO_PIN, GPIO.LOW)
    GPIO_OK = True
    print(f"[GPIO] BCM pin {GPIO_PIN} ready")
except (ImportError, RuntimeError):
    GPIO_OK = False
    print("[GPIO] unavailable (mock mode)")

def magnet(on):
    if GPIO_OK: GPIO.output(GPIO_PIN, GPIO.HIGH if on else GPIO.LOW)

def vision_to_arm(x, y):
    return x, y * VISION_TO_ARM_Y

LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ikine_lib.json")

def idw_xyz(x, y, layer="grip"):
    lib = json.load(open(LIB_FILE))
    pts = [e for e in lib if e["layer"] == layer]
    ws = {"dx": 0.0, "dy": 0.0, "dz": 0.0}; norm = 0.0
    for e in pts:
        d = ((x - e["x"])**2 + (y - e["y"])**2)**0.5
        if d < 0.3: return (e["dx"], e["dy"], e["dz"])
        w = 1.0 / (d * d); norm += w
        ws["dx"] += w * e["dx"]; ws["dy"] += w * e["dy"]; ws["dz"] += w * e["dz"]
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
    arm.wait(2000)

def do_pick(arm, px, py):
    g = idw_xyz(px, py, "grip")
    s = (g[0], g[1], g[2] + SAFE_ABOVE)
    print(f"  pick({px},{py}) safe={s} grip={g}")
    arm.home(); arm.wait(1000)
    move_safe(arm, *s); time.sleep(PAUSE_SAFE)
    dz = -SAFE_ABOVE
    arm.move_by_delta(0, 0, dz); arm.wait(2000)
    magnet(True); arm.wait(WAIT_GRIP)
    arm.move_by_delta(0, 0, -dz); arm.wait(2000)
    time.sleep(PAUSE_SAFE)

def do_place(arm, tx, ty):
    g = idw_xyz(tx, ty, "grip")
    s = (g[0], g[1], g[2] + SAFE_ABOVE)
    print(f"  place({tx},{ty}) safe={s} grip={g}")
    arm.home(); arm.wait(1000)
    move_safe(arm, *s); time.sleep(PAUSE_SAFE)
    dz = -SAFE_ABOVE
    arm.move_by_delta(0, 0, dz); arm.wait(2000)
    magnet(False); arm.wait(WAIT_PLACE)
    arm.move_by_delta(0, 0, -dz); arm.wait(2000)
    time.sleep(PAUSE_SAFE)

def main():
    if len(sys.argv) < 5:
        print("Usage: python pick_and_place.py <pick_x> <pick_y> <place_x> <place_y>")
        return
    px, py = vision_to_arm(float(sys.argv[1]), float(sys.argv[2]))
    tx, ty = vision_to_arm(float(sys.argv[3]), float(sys.argv[4]))

    print(f"Pick({px},{py}) Place({tx},{ty})  Ctrl+C=ESTOP")
    input("  Enter to start: ")

    arm = ArmController(PORT)
    if not arm.connect(): return
    try:
        do_pick(arm, px, py)
        do_place(arm, tx, ty)
        arm.home(); arm.wait(1000)
        magnet(False)
        print("done.")
    except KeyboardInterrupt:
        print("\nESTOP")
        magnet(False)
    finally:
        arm.disconnect()

if __name__ == "__main__":
    main()
