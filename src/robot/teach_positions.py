"""
teach_positions.py — 机械臂键盘示教工具

用键盘实时控制舵机，记录关键位置，保存为 JSON 供工作流使用。

用法:
  python teach_positions.py --port COM3
  python teach_positions.py --port COM3 --load positions.json   # 加载已有位置继续编辑
  python teach_positions.py --list-ports                         # 列出可用串口

键盘映射:
  1-6    : 选择当前舵机（LED/蜂鸣器会反馈）
  W / S  : 当前舵机 +30 / -30 μs  (粗调)
  A / D  : 当前舵机 +5  / -5  μs   (精调)
  R      : 记录当前位置（输入名称）
  L      : 列出所有已记录位置
  Del    : 删除一个已记录位置
  H      : 所有舵机回中位 (P1500)
  G      : 夹爪开 ↔ 关 切换
  O      : 保存位置到 JSON 文件
  P      : 移动到指定已记录位置
  Q      : 退出

依赖: arm_controller.py (同目录)
"""

import os
import sys
import json
import time
import argparse

# 确保能找到同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import ArmController, PositionRecorder, SERVO_MAP, SERVO_LIMITS, SERVO_CENTER

# =========================================================================
# 跨平台键盘输入
# =========================================================================

def _getch_windows():
    """Windows: 读取单个按键（不回车）"""
    import msvcrt
    ch = msvcrt.getch()
    # 特殊键（方向键等）返回两个字节: \xe0 + code
    if ch == b'\xe0':
        ch2 = msvcrt.getch()
        return None  # 忽略方向键
    try:
        return ch.decode('utf-8').lower()
    except UnicodeDecodeError:
        return None


def _getch_unix():
    """Linux: 读取单个按键（不回车）"""
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower()


def get_key():
    """跨平台读取单个按键"""
    try:
        return _getch_windows()
    except ImportError:
        return _getch_unix()


# =========================================================================
# 示教器
# =========================================================================

class TeachPendant:
    """键盘示教器"""

    COARSE_STEP = 30   # 粗调步长 (μs)
    FINE_STEP   = 5    # 精调步长 (μs)

    def __init__(self, arm: ArmController):
        self.arm = arm
        self.recorder = PositionRecorder(arm)
        self.current_servo = 1       # 当前选中的舵机
        self._running = True

    # ---- 显示 ----

    def _show_help(self):
        print("""
╔══════════════════════════════════════════════════════╗
║          LeArm 机械臂键盘示教工具                     ║
╠══════════════════════════════════════════════════════╣
║  1-6  选择舵机    W/S  粗调 ±30μs    A/D  精调 ±5μs  ║
║  R    记录位置    L    列出位置      Del  删除位置     ║
║  P    跳转位置    G    夹爪开/关     H    回中位       ║
║  O    保存JSON    Q    退出                           ║
╚══════════════════════════════════════════════════════╝
""")

    def _show_state(self):
        """显示当前舵机状态"""
        pos = self.arm.get_positions()
        print(f"\n  ┌─────────────────────────────────────────┐")
        for sid in range(1, 7):
            joint = SERVO_MAP.get(sid, "?")
            pulse = pos.get(sid, SERVO_CENTER)
            lo, hi = SERVO_LIMITS.get(sid, (500, 2500))
            angle = (pulse - 500) / 2000.0 * 180.0
            bar_len = int((pulse - lo) / (hi - lo) * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            marker = " ◀◀" if sid == self.current_servo else ""
            print(f"  │ #{sid} {joint:<13} [{bar}] P{pulse:4d} ({angle:5.1f}°){marker}   │")
        print(f"  └─────────────────────────────────────────┘")

    # ---- 舵机操作 ----

    def _adjust_servo(self, delta: int):
        """调整当前舵机的脉冲值"""
        sid = self.current_servo
        current = self.arm.get_positions().get(sid, SERVO_CENTER)
        new_val = current + delta
        lo, hi = SERVO_LIMITS.get(sid, (500, 2500))
        new_val = max(lo, min(hi, new_val))
        self.arm.move_servo(sid, new_val)  # 使用默认2000ms
        # 立即刷新显示
        joint = SERVO_MAP.get(sid, "?")
        angle = (new_val - 500) / 2000.0 * 180.0
        print(f"\r  → #{sid} {joint}: P{current} → P{new_val}  ({angle:.1f}°)   ", end="")
        sys.stdout.flush()

    def _select_servo(self, sid: int):
        self.current_servo = sid
        self._show_state()

    def _toggle_gripper(self):
        """夹爪开/关切换 (ID=1, 全开P600↔全闭P1500)"""
        pos = self.arm.get_positions()
        current = pos.get(1, 770)
        if current >= 1250:  # 当前偏闭合 → 张开
            self.arm.gripper_open()
            print("\n  夹爪 → 张开 (P600)")
        else:                # 当前偏张开 → 闭合
            self.arm.gripper_close()
            print("\n  夹爪 → 闭合 (P1500)")

    # ---- 位置管理 ----

    def _record_position(self):
        """交互式记录位置"""
        print()
        name = input("  位置名称 (如 above_pick): ").strip()
        if not name:
            print("  ✗ 名称不能为空")
            return
        self.recorder.record(name)

    def _goto_position(self):
        """跳转到已记录位置"""
        if not self.recorder.positions:
            print("\n  ✗ 暂无已记录位置，先按 R 记录")
            return
        self.recorder.list_positions()
        name = input("  跳转到位置名 (回车取消): ").strip()
        if name and name in self.recorder.positions:
            self.arm.move_to_pose(self.recorder.positions[name], time_ms=800)
            print(f"  ✓ 移动到: {name}")

    def _delete_position(self):
        if not self.recorder.positions:
            print("\n  ✗ 暂无已记录位置")
            return
        self.recorder.list_positions()
        name = input("  删除位置名 (回车取消): ").strip()
        if name and name in self.recorder.positions:
            self.recorder.remove(name)

    def _save_positions(self):
        """保存位置到 JSON"""
        print()
        path = input("  保存路径 (如 my_positions.json): ").strip()
        if not path:
            path = "positions.json"
        self.recorder.save(path)

    # ---- 主循环 ----

    def run(self):
        """主示教循环"""
        self._show_help()
        self._show_state()

        while self._running:
            key = get_key()
            if key is None:
                continue

            if key == 'q':
                print("\n\n  退出示教模式...")
                break

            elif key in '123456':
                self._select_servo(int(key))

            elif key == 'w':
                self._adjust_servo(self.COARSE_STEP)

            elif key == 's':
                self._adjust_servo(-self.COARSE_STEP)

            elif key == 'a':
                self._adjust_servo(-self.FINE_STEP)

            elif key == 'd':
                self._adjust_servo(self.FINE_STEP)

            elif key == 'r':
                self._record_position()

            elif key == 'l':
                self.recorder.list_positions()

            elif key == 'p':
                self._goto_position()

            elif key == 'g':
                self._toggle_gripper()

            elif key == 'h':
                self.arm.home(time_ms=800)
                print("\n  回中位 P1500 ×6")
                self._show_state()

            elif key == 'o':
                self._save_positions()

            # Delete 键处理
            elif key == '\x08' or key == '\x7f':  # backspace / delete
                self._delete_position()

        print("  示教结束。")


# =========================================================================
# CLI
# =========================================================================

def main():
    p = argparse.ArgumentParser(description="LeArm 机械臂键盘示教工具")
    p.add_argument("--port", type=str, default=None,
                   help="串口号 (如 COM3, /dev/ttyUSB0)")
    p.add_argument("--list-ports", action="store_true",
                   help="列出可用串口后退出")
    p.add_argument("--load", type=str, default=None,
                   help="加载已有的位置 JSON 文件")
    args = p.parse_args()

    # ---- 列出串口 ----
    if args.list_ports:
        ports = ArmController.list_ports()
        if ports:
            print("可用串口:")
            for p_info in ports:
                print(f"  {p_info}")
        else:
            print("未检测到串口")
        return

    # ---- 连接 ----
    if not args.port:
        ports = ArmController.list_ports()
        if ports:
            print("可用串口:")
            for p_info in ports:
                print(f"  {p_info}")
        args.port = input("\n输入串口号: ").strip()
        if not args.port:
            print("未选择串口，退出。")
            return

    arm = ArmController(args.port)
    if not arm.connect():
        print("无法连接机械臂。请检查:")
        print("  1. USB 线是否接好")
        print("  2. 串口号是否正确")
        print("  3. 舵机控制器是否上电")
        return

    try:
        pendant = TeachPendant(arm)

        # 加载已有位置
        if args.load and os.path.exists(args.load):
            pendant.recorder.load(args.load)

        pendant.run()

    except KeyboardInterrupt:
        print("\n\n⚠ Ctrl+C 中断")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
