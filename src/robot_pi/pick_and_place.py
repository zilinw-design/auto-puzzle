"""
pick_and_place.py -- LeArm automated pick-and-place

Usage: python pick_and_place.py <pick_x> <pick_y> <place_x> <place_y>

XY: linear fit, Z: IDW from 17 calibrated points
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM7"
WAIT_GRIP = 500
WAIT_PLACE = 300
SAFE_ABOVE = 35
SETTLE_MS = 0.5
PAUSE_SAFE = 0.5
VISION_TO_ARM_Y = 14.7 / 14.85
LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ikine_lib.json")

def vision_to_arm(x, y):
    return x, y * VISION_TO_ARM_Y

def idw_xyz(x, y, layer="grip"):
    """XY=线性拟合(17点残差17/11), Z=IDW(精调过)"""
    dx = round(9.4*x + 0.1*y - 21)
    dy = round(0.3*x + 9.6*y + 2)
    lib = json.load(open(LIB_FILE))
    pts = [e for e in lib if e["layer"] == layer]
    ws, n = 0.0, 0.0
    for e in pts:
        d = ((x-e["x"])**2 + (y-e["y"])**2)**0.5
        if d < 0.3: return (dx, dy, e["dz"])
        w = 1.0/(d*d); n += w; ws += w*e["dz"]
    return (dx, dy, round(ws/n))

def rotate(arm, angle_deg):
    """#2 腕部旋转. PWM = 1500 + angle * 1000/90 (文档标准公式)"""
    pwm = round(1500 + angle_deg * 1000 / 90)
    print(f"  rotate {angle_deg:+.0f}deg -> #{2}={pwm}")
    arm.move_to_pose({2: pwm}, time_ms=2000)
    arm.wait(2500)

def move_safe(arm, dx, dy, dz):
    steps = max(1, max(abs(dx),abs(dy),abs(dz))//100+1)
    for i in range(steps):
        sx = ((i+1)*dx//steps)-(i*dx//steps)
        sy = ((i+1)*dy//steps)-(i*dy//steps)
        sz = ((i+1)*dz//steps)-(i*dz//steps)
        if sx or sy or sz:
            arm.move_by_delta(sx, sy, sz); arm.wait(800)
    arm.wait(2000)

def main():
    if len(sys.argv)<5:
        print("Usage: python pick_and_place.py <px> <py> <tx> <ty>"); return
    px,py = vision_to_arm(float(sys.argv[1]),float(sys.argv[2]))
    tx,ty = vision_to_arm(float(sys.argv[3]),float(sys.argv[4]))

    g1 = idw_xyz(px,py,"grip"); s1 = idw_xyz(px,py,"safe")
    g2 = idw_xyz(tx,ty,"grip"); s2 = idw_xyz(tx,ty,"safe")
    print(f"Pick({px:.1f},{py:.1f}): safe={s1} grip={g1}")
    print(f"Place({tx:.1f},{ty:.1f}): safe={s2} grip={g2}")
    print("Ctrl+C=STOP")
    input("Enter=start: ")

    a = ArmController(PORT)
    if not a.connect(): return
    try:
        dz1 = g1[2]-s1[2]; dz2 = g2[2]-s2[2]
        print("[1] pick"); a.home(); a.wait(1000)
        move_safe(a,*s1)
        a.move_by_delta(0,0,dz1); a.wait(2000)
        time.sleep(SETTLE_MS); a.wait(WAIT_GRIP)
        a.move_by_delta(0,0,-dz1); a.wait(2000)
        time.sleep(PAUSE_SAFE)

        print("[2] place"); a.home(); a.wait(1000)
        move_safe(a,*s2)
        a.move_by_delta(0,0,dz2); a.wait(2000)
        time.sleep(SETTLE_MS); a.wait(WAIT_PLACE)
        a.move_by_delta(0,0,-dz2); a.wait(2000)

        print("[3] home"); a.home(); a.wait(1000)
        print("done.")
    except KeyboardInterrupt: print("\nSTOP")
    finally: a.disconnect()

if __name__=="__main__": main()
