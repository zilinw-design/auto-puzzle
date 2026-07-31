"""
最终标定: 全17点, 方向键微调, 一键确认

按键 (单键, 不需回车):
  方向键 ↑↓←→  : XY微调
  W/S          : Z升降 (安全层)
  w/s          : Z升降 (抓取层)
  1~9          : 设步长 1~9mm
  Space/Enter  : 确认/下一步
  p            : 急停
  q            : 跳过此点

用法: python final_calibrate.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM7"
SAFE_ABOVE = 35
STEP = 5  # 默认步长 mm
LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ikine_lib.json")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_final.json")

CX, CY = (9.6, 0.3, -26), (-0.3, 10.0, -6)

def getch():
    """跨平台单键读入"""
    try:
        import msvcrt
        return msvcrt.getch()
    except ImportError:
        import tty, termios
        fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
        try: tty.setraw(fd); return sys.stdin.read(1).encode()
        finally: termios.tcsetattr(fd, termios.TCSADRAIN, old)

def predict(x,y,c): return round(c[0]*x+c[1]*y+c[2])

def all_pts():
    lib = json.load(open(LIB_FILE))
    return sorted(set((e["x"],e["y"]) for e in lib))

def move_safe(arm, dx, dy, dz):
    steps = max(1, max(abs(dx),abs(dy),abs(dz))//100+1)
    for i in range(steps):
        sx=((i+1)*dx//steps)-(i*dx//steps); sy=((i+1)*dy//steps)-(i*dy//steps); sz=((i+1)*dz//steps)-(i*dz//steps)
        if sx or sy or sz: arm.move_by_delta(sx,sy,sz); arm.wait(800)
    arm.wait(2000)

def idw_dz(x,y):
    lib=json.load(open(LIB_FILE)); pts=[e for e in lib if e["layer"]=="grip"]; ws,n=0.0,0.0
    for e in pts:
        d=((x-e["x"])**2+(y-e["y"])**2)**0.5
        if d<0.3: return e["dz"]
        w=1.0/(d*d); n+=w; ws+=w*e["dz"]
    return round(ws/n)

def adjust(arm, dx, dy, dz, at_grip=False):
    """方向键微调循环, 返回 (dx,dy,dz) 或 None=skip"""
    global STEP
    layer = "grip" if at_grip else "safe"
    print(f"  [{layer}] WASD=XYZ ↑↓←→=XY 1-9=步长 Space=OK q=跳过 p=急停 step={STEP}mm")
    while True:
        ch = getch()
        if ch in (b' ', b'\r', b'\n'):
            return dx, dy, dz
        elif ch == b'p':
            print("PANIC!"); return None
        elif ch == b'q':
            return 'skip'
        elif ch in (b'1',b'2',b'3',b'4',b'5',b'6',b'7',b'8',b'9'):
            STEP = int(ch)
            print(f"  step={STEP}mm")
        elif ch == b'0':
            STEP = 10
            print(f"  step={STEP}mm")
        # 方向键 (Win: \xe0 + code)
        elif ch == b'\xe0':
            ch2 = getch()
            if ch2 == b'H':   # ↑
                arm.move_by_delta(0, +STEP, 0); dy += STEP; arm.wait(1200)
            elif ch2 == b'P': # ↓
                arm.move_by_delta(0, -STEP, 0); dy -= STEP; arm.wait(1200)
            elif ch2 == b'K': # ←
                arm.move_by_delta(-STEP, 0, 0); dx -= STEP; arm.wait(1200)
            elif ch2 == b'M': # →
                arm.move_by_delta(+STEP, 0, 0); dx += STEP; arm.wait(1200)
        # Linux 方向键
        elif ch == b'\x1b':
            ch2 = getch()
            if ch2 == b'[':
                ch3 = getch()
                if ch3 == b'A': arm.move_by_delta(0,+STEP,0); dy+=STEP; arm.wait(1200)
                elif ch3 == b'B': arm.move_by_delta(0,-STEP,0); dy-=STEP; arm.wait(1200)
                elif ch3 == b'C': arm.move_by_delta(+STEP,0,0); dx+=STEP; arm.wait(1200)
                elif ch3 == b'D': arm.move_by_delta(-STEP,0,0); dx-=STEP; arm.wait(1200)
        elif ch == b'w':
            if at_grip: arm.move_by_delta(0,0,+STEP); dz+=STEP; arm.wait(1200)
            else: arm.move_by_delta(0,0,+STEP); dz+=STEP; arm.wait(1200)
        elif ch == b's':
            if at_grip: arm.move_by_delta(0,0,-STEP); dz-=STEP; arm.wait(1200)
            else: arm.move_by_delta(0,0,-STEP); dz-=STEP; arm.wait(1200)
        elif ch == b'W':
            arm.move_by_delta(0,0,+STEP); dz+=STEP; arm.wait(1200)
        elif ch == b'S':
            arm.move_by_delta(0,0,-STEP); dz-=STEP; arm.wait(1200)
        # XY 也映射到键盘
        elif ch == b'a': arm.move_by_delta(-STEP,0,0); dx-=STEP; arm.wait(1200)
        elif ch == b'd': arm.move_by_delta(+STEP,0,0); dx+=STEP; arm.wait(1200)
        if not at_grip:
            print(f"  dx={dx} dy={dy} safe_dz={dz}")
        else:
            print(f"  grip_dz={dz}")

def main():
    pts = all_pts()
    print(f"共 {len(pts)} 点:")
    for i, (x, y) in enumerate(pts):
        print(f"  {i:2d}. ({x:5.1f},{y:5.1f})")
    sel = input("输入序号(逗号分隔, 如 1,3,5) 或 Enter=全部: ").strip()
    if sel:
        idxs = [int(s.strip()) for s in sel.split(",") if s.strip().isdigit()]
        pts = [pts[i] for i in idxs if 0 <= i < len(pts)]
    print(f"将标定 {len(pts)} 点")

    results = []
    arm = ArmController(PORT)
    if not arm.connect(): return

    try:
        for i, (x, y) in enumerate(pts, 1):
            dx = predict(x, y, CX); dy = predict(x, y, CY)
            dz = idw_dz(x, y); safe_dz = dz + SAFE_ABOVE

            print(f"\n{'='*40}")
            print(f"[{i}/{len(pts)}] ({x:.1f},{y:.1f})")
            arm.home(); arm.wait(1000)
            move_safe(arm, dx, dy, safe_dz); arm.wait(1000)

            # 安全层微调
            r = adjust(arm, dx, dy, safe_dz, at_grip=False)
            if r is None: return  # panic
            if r == 'skip': continue
            dx, dy, safe_dz = r

            # 下降
            total_down = safe_dz - dz
            arm.move_by_delta(0, 0, -total_down); arm.wait(2000)

            # 抓取层微调
            r = adjust(arm, dx, dy, dz, at_grip=True)
            if r is None: return
            if r == 'skip': arm.move_by_delta(0,0,total_down); arm.wait(2000); arm.home(); arm.wait(1000); continue
            dx_final, dy_final, grip_dz = r

            # 读PWM
            pos = arm.read_positions()
            pwm_g = {str(k):v for k,v in pos.items()} if pos else {}
            arm.move_by_delta(0, 0, safe_dz - grip_dz); arm.wait(2000)
            pos2 = arm.read_positions()
            pwm_s = {str(k):v for k,v in pos2.items()} if pos2 else {}
            arm.home(); arm.wait(1000)

            entry = {"x":x,"y":y,"dx":dx_final,"dy":dy_final,
                     "safe_dz":safe_dz,"grip_dz":grip_dz,
                     "pwm_safe":pwm_s,"pwm_grip":pwm_g}
            results.append(entry)
            print(f"  [OK] recorded")

    except KeyboardInterrupt: print("\nSTOP")
    finally: arm.disconnect()

    with open(OUT_FILE,"w") as f:
        json.dump({"points":results,"total":len(results),
                   "fit": {"dx":[9.6,0.3,-26],"dy":[-0.3,10.0,-6]}}, f, indent=2)
    print(f"\nSaved {len(results)} pts -> {OUT_FILE}")

if __name__=="__main__": main()
