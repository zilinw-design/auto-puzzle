"""
play_frames.py — 把上位机XML动作组逐帧播放

用法:
  python play_frames.py COM13
  python play_frames.py COM13 --step    # 单步模式
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController, xml_to_workflow

# 你的 XML 路径
XML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ai_harness_framework", "references", "raw",
    "LeArm 上位机软件", "LeArm 上位机软件", "temp", "model_2.xml"
)


def main():
    p = argparse.ArgumentParser(description="播放上位机导出的XML动作组")
    p.add_argument("port", nargs="?", default="COM7", help="串口号 (默认 COM7)")
    p.add_argument("--xml", default=XML_PATH, help="XML动作组路径")
    p.add_argument("--step", action="store_true", help="单步模式")
    args = p.parse_args()

    # 1. 读 XML
    print(f"加载: {args.xml}")
    frames = xml_to_workflow(args.xml)  # 返回 [{1:770,2:1500,...}, ...]
    print(f"共 {len(frames)} 帧\n")

    # 2. 打印序列
    for i, f in enumerate(frames):
        t = f.pop("_time_ms", "?")
        pos = " ".join(f"#{s}={f[s]}" for s in sorted(f))
        print(f"  帧{i+1}  T{t}ms: {pos}")
        f["_time_ms"] = t  # 恢复

    # 3. 连接
    print(f"\n连接 {args.port} ...")
    arm = ArmController(args.port)
    if not arm.connect():
        return

    # 4. 逐帧播放
    try:
        for i, frame in enumerate(frames):
            t = frame.pop("_time_ms", 2000)
            pose = {k: v for k, v in frame.items()}  # 去掉 _time_ms

            print(f"\n[{i+1}/{len(frames)}] 帧{i+1} → 移动 {t}ms")
            parts = " ".join(f"#{s}={pose[s]}" for s in sorted(pose))
            print(f"  {parts}")

            if args.step:
                cmd = input("  Enter=执行 s=跳过 p=急停 q=退出: ").strip().lower()
                if cmd == 'q':
                    break
                elif cmd == 'p':
                    arm.panic("用户急停")
                    return
                elif cmd == 's':
                    continue

            # arm_controller 会钳制到 ≥2000ms，用实际值等待
            actual_time = max(2000, t)
            arm.move_to_pose(pose, time_ms=actual_time)
            arm.wait(actual_time + 200)  # 额外等 200ms 确保完全到位

            # 回读固件真实位置，确认到位
            actual = arm.read_positions()
            if actual:
                parts = " ".join(f"#{s}={actual[s]}" for s in sorted(actual))
                print(f"  到位: {parts}")
            else:
                print(f"  ⚠ 回读失败")

        # 5. 自动复位
        print(f"\n全部帧播放完成。自动复位...")
        if args.step:
            input("  按 Enter 执行复位: ")
        arm.home()
        arm.wait(2000)
        print("复位完成。")

    except KeyboardInterrupt:
        print("\n⚠ 用户中断")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
