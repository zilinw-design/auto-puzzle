"""
monitor.py — 机械臂舵机实时状态监视器

持续轮询固件回读6路舵机位置，实时刷新显示。
用于诊断舵机抖动、漂移、不到位等问题。

用法:
  python monitor.py COM7
  python monitor.py COM7 --rate 5    # 每秒5次
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController, SERVO_MAP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("port", nargs="?", default="COM7")
    p.add_argument("--rate", type=float, default=3, help="采样率 Hz (默认3)")
    args = p.parse_args()

    arm = ArmController(args.port, timeout=0.3)
    if not arm.connect():
        return

    interval = 1.0 / args.rate
    prev = None
    print(f"\n实时监视 COM7  @ {args.rate}Hz  Ctrl+C 退出\n")
    print(f"{'':>5} {'#1夹爪':>8} {'#2腕翻':>8} {'#3腕俯':>8} "
          f"{'#4肘部':>8} {'#5肩部':>8} {'#6底盘':>8}")
    print(f"{'':>5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    count = 0
    try:
        while True:
            pos = arm.read_positions()
            if pos is None:
                print(f"[{count:4d}] 读取失败")
                time.sleep(interval)
                continue

            # 检测变化
            flags = ""
            if prev:
                for sid in range(1, 7):
                    delta = pos[sid] - prev[sid]
                    if delta != 0:
                        joint = SERVO_MAP[sid]
                        direction = "↑" if delta > 0 else "↓"
                        flags += f"  #{sid}{joint} {prev[sid]}→{pos[sid]} ({direction}{abs(delta)})"
            if flags:
                print(f"\n[{count:4d}] ⚡ 检测到变化:{flags}")

            # 实时行（覆盖刷新）
            line = f"[{count:4d}] "
            for sid in range(1, 7):
                pulse = pos[sid]
                # 标记异常: 跳动超过5算异常
                mark = ""
                if prev and abs(pulse - prev[sid]) > 5:
                    mark = "!"
                line += f" P{pulse:4d}{mark} "
            print(line, end="\r")

            prev = pos
            count += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n监视结束。共采样 {count} 次。")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
