"""
旋转拾放: 拾取→中转(0,0)旋转释放→复位→重拾→目标释放
支持 >45deg 自动拆分, 角度归一化到 [-180,180]

用法: python rotate_place.py <px> <py> <tx> <ty> <angle_deg>
"""

import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController
from pick_and_place import idw_xyz, move_safe, rotate, vision_to_arm, SAFE_ABOVE

PORT = "COM7"
WAIT_G = 1000   # 吸/放 1s
MAX_STEP = 45   # 单次最大旋转角度

def grip(a, pick=True):
    a.move_by_delta(0,0,-SAFE_ABOVE); a.wait(2000)
    time.sleep(0.5)
    a.wait(WAIT_G)
    a.move_by_delta(0,0,SAFE_ABOVE); a.wait(2000)

def compensation(sx, sy, tx, ty, ref=(-13, 0)):
    """ref→源 vs ref→目标的夹角 (deg), +=CCW -=CW"""
    a1 = math.atan2(sy - ref[1], sx - ref[0])
    a2 = math.atan2(ty - ref[1], tx - ref[0])
    return math.degrees(a2 - a1)  # 目标角 - 源角

def split_angle(deg, max_step=45):
    """归一化到 [-180,180], 拆分"""
    deg = ((deg + 180) % 360) - 180
    if deg == 0: return []
    steps = []; r = abs(deg); s = 1 if deg > 0 else -1
    while r > 0.01:
        step = min(max_step, r); steps.append(s * step); r -= step
    return steps

def main():
    if len(sys.argv)<5:
        print("用法: python rotate_place.py <px> <py> <tx> <ty> <angle>"); return
    px,py = vision_to_arm(float(sys.argv[1]),float(sys.argv[2]))
    tx,ty = vision_to_arm(float(sys.argv[3]),float(sys.argv[4]))
    angle = float(sys.argv[5])

    comp = compensation(px, py, tx, ty)
    total = angle - comp
    print(f"  src→tgt自然转: {comp:+.1f}deg  需旋转: {total:.1f}deg")
    # 优化1: |total|<3deg → 自然旋转已满足, 跳过旋转, 直接运输
    if abs(total) < 3:
        print("  total<3deg, 自然旋转已满足, 跳过旋转步骤")

        p_g = idw_xyz(px,py,"grip"); p_s = (p_g[0],p_g[1],p_g[2]+SAFE_ABOVE)
        t_g = idw_xyz(tx,ty,"grip"); t_s = (t_g[0],t_g[1],t_g[2]+SAFE_ABOVE)
        print(f"  直接运输 ({px:.1f},{py:.1f})→({tx:.1f},{ty:.1f})")
        input("Enter=开始: ")
        a = ArmController(PORT)
        if not a.connect(): return
        try:
            print("[1] pick"); a.home(); a.wait(1000)
            move_safe(a,*p_s)
            a.move_by_delta(0,0,-SAFE_ABOVE); a.wait(2000)
            time.sleep(0.5); a.wait(WAIT_G)
            a.move_by_delta(0,0,SAFE_ABOVE); a.wait(2000)
            print("[2] place"); a.home(); a.wait(1000)
            move_safe(a,*t_s)
            a.move_by_delta(0,0,-SAFE_ABOVE); a.wait(2000)
            time.sleep(0.5); a.wait(WAIT_G)
            a.move_by_delta(0,0,SAFE_ABOVE); a.wait(2000)
            print("[3] home"); a.home(); a.wait(1000)
        except KeyboardInterrupt: print("ESTOP")
        finally: a.disconnect()
        return
    # 优化2: ≤90deg一步完成, >90deg分步
    max_step = 90 if abs(total) <= 90 else 45
    steps = split_angle(total, max_step)
    if not steps: steps = [total]

    p_g = idw_xyz(px,py,"grip"); p_s = (p_g[0],p_g[1],p_g[2]+SAFE_ABOVE)
    m_g = idw_xyz(0,0,"grip");   m_s = (m_g[0],m_g[1],m_g[2]+SAFE_ABOVE)
    t_g = idw_xyz(tx,ty,"grip"); t_s = (t_g[0],t_g[1],t_g[2]+SAFE_ABOVE)

    print(f"取({px:.1f},{py:.1f})→(0,0)转{angle:.0f}deg({len(steps)}步:{steps})→放({tx:.1f},{ty:.1f})")
    input("Enter=开始: ")

    a = ArmController(PORT)
    if not a.connect(): return
    try:
        # 1. 拾取源点
        print("[1] pick source"); a.home(); a.wait(1000)
        move_safe(a,*p_s); grip(a,True)

        # 2. 去中转→安全位旋转→下降释放→上升→复位→重拾
        for i, s in enumerate(steps, 1):
            direction = 'CW RIGHT' if s > 0 else 'CCW LEFT'
            print(f"\n[2.{i}] → (0,0) rotate {s:+.0f}deg ({direction})"); a.home(); a.wait(1000)
            move_safe(a,*m_s)                    # 到安全高度
            rotate(a, s)                         # 在安全位旋转
            a.move_by_delta(0,0,-SAFE_ABOVE); a.wait(2000)  # 下降
            time.sleep(0.5); a.wait(WAIT_G)      # 释放
            a.move_by_delta(0,0,SAFE_ABOVE); a.wait(2000)   # 上升
            print(f"[2.{i}] home"); a.home(); a.wait(1000)
            print(f"[2.{i}] re-pick"); move_safe(a,*m_s)
            a.move_by_delta(0,0,-SAFE_ABOVE); a.wait(2000)
            time.sleep(0.5); a.wait(WAIT_G)      # 重拾
            a.move_by_delta(0,0,SAFE_ABOVE); a.wait(2000)

        # 3. 目标释放
        print("\n[3] → target"); a.home(); a.wait(1000)
        move_safe(a,*t_s); grip(a, False)

        print("[4] home"); a.home(); a.wait(1000); print("done.")
    except KeyboardInterrupt: print("\nESTOP")
    finally: a.disconnect()

if __name__=="__main__": main()
