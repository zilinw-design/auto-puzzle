"""
pick_and_place.py -- LeArm automated pick-and-place (Raspberry Pi)

=== 控制逻辑 ===
下降完成后:
  0.5s 稳臂 (机械振动衰减) → GPIO 开/关电磁铁 → 0.5s 吸合/释放 → 上升
总停留 1.0s。抓取总时间约 5s (安全+下降+停留+上升+复位)。

=== GPIO 接线 ===
Pi GPIO17 → 继电器/MOSFET 模块 → 电磁铁 (外接电源)
Pi 3.3V 不能直驱电磁铁，必须经过驱动模块。

=== 修改指南 ===
- WAIT_PICK / WAIT_RELEASE: 吸合/释放时间 (秒)
- SETTLE_MS: 到位后稳定等待
- SAFE_ABOVE: 安全高度 = grip_dz + N mm
- GPIO_PIN: 电磁铁 BCM pin 号

Usage:
  python pick_and_place.py <pick_x> <pick_y> <place_x> <place_y>
  python pick_and_place.py 6 -9 6 9
"""

import sys, os, json, time
import numpy as np
from config import PORT
from arm_controller import ArmController

# ── 可调参数 ──
SAFE_ABOVE = 35     # mm, 安全高度 = grip dz + 35
SETTLE_MS  = 0.5    # s,  下降后稳臂时间
WAIT_PICK  = 0.5    # s,  电磁铁吸合保持 (开磁铁后等待)
WAIT_RELEASE = 0.3  # s,  释放等待 (关磁铁后等待)
VISION_TO_ARM_Y = 14.7 / 14.85

# ── GPIO 电磁铁 (树莓派 BCM pin 17) ──
GPIO_PIN = 17
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_PIN, GPIO.OUT)
    GPIO.output(GPIO_PIN, GPIO.LOW)
    GPIO_OK = True
    print(f"[GPIO] BCM{pin} ready")
except (ImportError, RuntimeError):
    GPIO_OK = False
    print("[GPIO] unavailable (mock)")

def magnet(on: bool):
    """GPIO.HIGH=吸, GPIO.LOW=放"""
    if GPIO_OK:
        GPIO.output(GPIO_PIN, GPIO.HIGH if on else GPIO.LOW)
        print(f"[GPIO] {'ON' if on else 'OFF'}")

def vision_to_arm(x, y):
    return x, y * VISION_TO_ARM_Y

# ── IDW 插值 ──
LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ikine_lib.json")

def idw_xyz(x, y, layer="grip"):
    lib = json.load(open(LIB_FILE))
    pts = [e for e in lib if e["layer"] == layer]
    ws = {"dx":0,"dy":0,"dz":0}; n = 0.0
    for e in pts:
        d = ((x-e["x"])**2 + (y-e["y"])**2)**0.5
        if d < 0.3: return (e["dx"], e["dy"], e["dz"])
        w = 1.0/(d*d); n += w
        ws["dx"] += w*e["dx"]; ws["dy"] += w*e["dy"]; ws["dz"] += w*e["dz"]
    return (round(ws["dx"]/n), round(ws["dy"]/n), round(ws["dz"]/n))

def move_safe(arm, dx, dy, dz):
    steps = max(1, max(abs(dx),abs(dy),abs(dz)) // 100 + 1)
    for i in range(steps):
        sx = ((i+1)*dx//steps) - (i*dx//steps)
        sy = ((i+1)*dy//steps) - (i*dy//steps)
        sz = ((i+1)*dz//steps) - (i*dz//steps)
        if sx or sy or sz:
            arm.move_by_delta(sx, sy, sz)
            arm.wait(800)
    arm.wait(2000)

# ── 拾取 ──
def do_pick(arm, px, py):
    g = idw_xyz(px, py, "grip")
    s = (g[0], g[1], g[2] + SAFE_ABOVE)
    print(f"  pick({px},{py}) safe={s} grip={g}")
    arm.home(); arm.wait(1000)
    print("[1] safe"); move_safe(arm, *s)
    print("[2] descend"); arm.move_by_delta(0, 0, -SAFE_ABOVE); arm.wait(2000)
    print(f"[3] settle {SETTLE_MS}s"); time.sleep(SETTLE_MS)
    print(f"[4] magnet ON + wait {WAIT_PICK}s"); magnet(True); time.sleep(WAIT_PICK)
    print("[5] rise"); arm.move_by_delta(0, 0, SAFE_ABOVE); arm.wait(2000)

# ── 释放 ──
def do_place(arm, tx, ty):
    g = idw_xyz(tx, ty, "grip")
    s = (g[0], g[1], g[2] + SAFE_ABOVE)
    print(f"  place({tx},{ty}) safe={s} grip={g}")
    arm.home(); arm.wait(1000)
    print("[1] safe"); move_safe(arm, *s)
    print("[2] descend"); arm.move_by_delta(0, 0, -SAFE_ABOVE); arm.wait(2000)
    print(f"[3] settle {SETTLE_MS}s"); time.sleep(SETTLE_MS)
    print(f"[4] magnet OFF + wait {WAIT_RELEASE}s"); magnet(False); time.sleep(WAIT_RELEASE)
    print("[5] rise"); arm.move_by_delta(0, 0, SAFE_ABOVE); arm.wait(2000)

# ── 主流程 ──
def main():
    if len(sys.argv) < 5:
        print("Usage: python pick_and_place.py <pick_x> <pick_y> <place_x> <place_y>")
        return
    px, py = vision_to_arm(float(sys.argv[1]), float(sys.argv[2]))
    tx, ty = vision_to_arm(float(sys.argv[3]), float(sys.argv[4]))

    print(f"Pick({px:.1f},{py:.1f}) Place({tx:.1f},{ty:.1f})  Ctrl+C=ESTOP")
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
