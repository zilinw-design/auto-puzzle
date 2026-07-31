"""
ikine 校准工具

用法: python calibrate_ikine.py

命令:
  go dx dy dz   → 移动 (自动拆步)
  +x/-x/+y/-y/+z/-z N → 微调
  home → 复位
  show → 锚点库
  save x_cm y_cm 备注 → 保存
  q → 退出
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController

PORT = "COM9"
LIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ikine_lib.json")

def go(arm, dx, dy, dz):
    """发送 ikine, 自动拆步"""
    # 拆成每步 ≤100mm
    need_x = max(1, abs(dx) // 100 + 1)
    need_y = max(1, abs(dy) // 100 + 1)
    need_z = max(1, abs(dz) // 100 + 1)
    steps = max(need_x, need_y, need_z)
    for i in range(steps):
        sx = ((i+1) * dx // steps) - (i * dx // steps)
        sy = ((i+1) * dy // steps) - (i * dy // steps)
        sz = ((i+1) * dz // steps) - (i * dz // steps)
        if sx or sy or sz:
            arm.move_by_delta(sx, sy, sz)
            arm.wait(1200)

def load():
    return json.load(open(LIB_FILE)) if os.path.exists(LIB_FILE) else []
def save(lib): json.dump(lib, open(LIB_FILE,"w"), indent=2)

def main():
    arm = ArmController(PORT)
    if not arm.connect(): return
    lib = load()
    cur = [0,0,0]

    print("\n=== ikine 校准 ===")
    print("go dx dy dz | +x/-x N | home | show | save x y 备注 | q\n")

    try:
        arm.home(); arm.wait(3500)
        while True:
            c = input("> ").strip().split()
            if not c: continue
            k = c[0].lower()

            if k == 'q': break
            elif k == 'home': arm.home(); arm.wait(3500); cur=[0,0,0]; print("复位")
            elif k == 'show':
                print(f"  当前: ({cur[0]},{cur[1]},{cur[2]})mm  锚点 {len(lib)}个")
                for e in lib: print(f"  ({e['x']},{e['y']})cm → ({e['dx']},{e['dy']},{e['dz']}) {e.get('note','')}")
            elif k == 'save':
                if len(c) < 4: print("  用法: save x_cm y_cm 备注"); continue
                layer = "grip" if "抓" in " ".join(c[3:]) else "safe"
                e = {"x":float(c[1]),"y":float(c[2]),"dx":cur[0],"dy":cur[1],"dz":cur[2],"layer":layer}
                lib.append(e); save(lib)
                print(f"  ✓ ({c[1]},{c[2]})cm → ({cur[0]},{cur[1]},{cur[2]})")
            elif k in ('+x','-x','+y','-y','+z','-z'):
                if len(c)<2: print("需要值"); continue
                v = int(c[1]); idx={'x':0,'y':1,'z':2}[k[-1]]
                cur[idx] += v if k[0]=='+' else -v
                print(f"  目标: ({cur[0]},{cur[1]},{cur[2]})")
            elif k == 'go':
                if len(c)<4: print("  用法: go dx dy dz"); continue
                cur[0] += int(c[1]); cur[1] += int(c[2]); cur[2] += int(c[3])
                print(f"  +({c[1]},{c[2]},{c[3]}) → 累计({cur[0]},{cur[1]},{cur[2]})")
                go(arm, int(c[1]), int(c[2]), int(c[3]))
                print(f"  到位。")
            else: print("  未知命令")

    except KeyboardInterrupt: print("\n")
    finally: arm.disconnect()

if __name__=="__main__": main()
