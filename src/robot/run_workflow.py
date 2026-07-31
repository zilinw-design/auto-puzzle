"""
run_workflow.py — 机械臂工作流执行器

加载示教好的位置 JSON，按预定义序列执行完整的「抓取→移动→放置」流程。

用法:
  # 干运行（打印步骤，不连接机械臂）
  python run_workflow.py --dry-run

  # 实机运行
  python run_workflow.py --port COM3 --positions positions.json

  # 单步模式（每步确认后执行）
  python run_workflow.py --port COM3 --positions positions.json --step

默认工作流 (pick_and_place):
  1. home()
  2. gripper_open()       → 先张开夹爪
  3. move_to above_pick   → 移动到抓取位上方
  4. move_to pick         → 下降到抓取位
  5. gripper_close()      → 闭合夹爪
  6. move_to above_pick   → 提起物体
  7. move_to above_place  → 移动到放置位上方
  8. move_to place        → 下降到放置位
  9. gripper_open()       → 释放物体
  10. move_to above_place → 提起
  11. home()               → 回中位

你可以通过修改 WORKFLOW 列表或定义自定义工作流来调整序列。

自定义工作流 JSON (workflow.json):
[
  {"action": "home"},
  {"action": "move_to", "position": "above_pick"},
  {"action": "gripper_open"},
  {"action": "move_to", "position": "pick"},
  {"action": "wait", "ms": 500},
  {"action": "gripper_close"},
  {"action": "move_to", "position": "above_pick"},
  {"action": "move_to", "position": "above_place"},
  {"action": "move_to", "position": "place"},
  {"action": "gripper_open"},
  {"action": "move_to", "position": "above_place"},
  {"action": "home"}
]

依赖: arm_controller.py (同目录)
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_controller import (ArmController, SERVO_LIMITS, RESET_DUTY,
                            SafetyFatal, SafetyError, DEFAULT_MOVE_TIME)

# =========================================================================
# 默认工作流：完整的取放循环
# =========================================================================

DEFAULT_WORKFLOW = [
    {"action": "home"},
    {"action": "gripper_open"},
    {"action": "move_to", "position": "above_pick"},
    {"action": "move_to", "position": "pick"},
    {"action": "gripper_close"},
    {"action": "wait", "ms": 500},
    {"action": "move_to", "position": "above_pick"},
    {"action": "move_to", "position": "above_place"},
    {"action": "move_to", "position": "place"},
    {"action": "gripper_open"},
    {"action": "wait", "ms": 300},
    {"action": "move_to", "position": "above_place"},
    {"action": "home"},
]


# =========================================================================
# 内置位置（供 dry-run 使用）
# =========================================================================

# 内置位置（仅供 dry-run 演示用, 值来自固件 global.h）
# ⚠ 实机运行时请使用示教工具记录的 positions.json
EXAMPLE_POSITIONS = {
    "home":        {"1": 770, "2": 1500, "3": 640,  "4": 511,  "5": 1255, "6": 1500},
    "above_pick":  {"1": 770, "2": 1500, "3": 1180, "4": 789,  "5": 1969, "6": 1253},
    "pick":        {"1": 770, "2": 1500, "3": 665,  "4": 1376, "5": 2030, "6": 1253},
    "above_place": {"1": 770, "2": 1500, "3": 738,  "4": 1179, "5": 1917, "6": 1676},
    "place":       {"1": 770, "2": 1500, "3": 1093, "4": 775,  "5": 1868, "6": 1676},
}


# =========================================================================
# 执行器
# =========================================================================

class WorkflowRunner:
    """工作流执行器"""

    def __init__(self, arm: ArmController, positions: dict,
                 step_mode: bool = False, move_time_ms: int = DEFAULT_MOVE_TIME):
        """
        Args:
            arm: 已连接的 ArmController 实例
            positions: {name: {servo_id: pulse}}
            step_mode: True=每步确认后执行
            move_time_ms: 默认移动时间 (≥2000ms 安全)
        """
        self.arm = arm
        self.positions = positions
        self.step_mode = step_mode
        self.move_time_ms = max(DEFAULT_MOVE_TIME, move_time_ms)
        self._stop = False

    def stop(self):
        """紧急停止"""
        self._stop = True
        self.arm.stop()

    def _show_step(self, i: int, total: int, step: dict):
        """打印当前步骤"""
        action = step.get("action", "?")
        pos = step.get("position", "")
        ms = step.get("ms", 0)

        if action == "move_to":
            desc = f"移动到 '{pos}'"
            if pos in self.positions:
                p = self.positions[pos]
                parts = [f"#{s}={p[str(s)]}" for s in sorted(p.keys(), key=int)]
                desc += f"  ({' '.join(parts)})"
            else:
                desc += "  ⚠ 未定义"
        elif action == "gripper_open":
            desc = "夹爪张开"
        elif action == "gripper_close":
            desc = "夹爪闭合"
        elif action == "home":
            desc = "回中位 (P1500 ×6)"
        elif action == "wait":
            desc = f"等待 {ms}ms"
        else:
            desc = str(step)

        print(f"  [{i}/{total}] {desc}")

    def _exec_step(self, step: dict) -> bool:
        """执行一个步骤。SafetyFatal 异常会向上传播以触发急停。"""
        if self._stop:
            return False

        action = step.get("action", "")
        pos_name = step.get("position", "")
        ms = step.get("ms", 0)

        if action == "home":
            self.arm.home()

        elif action == "gripper_open":
            self.arm.gripper_open()

        elif action == "gripper_close":
            self.arm.gripper_close()

        elif action == "move_to":
            if pos_name not in self.positions:
                print(f"    ⚠ 位置 '{pos_name}' 未定义，跳过!")
                return True
            pose = {int(k): v for k, v in self.positions[pos_name].items()}
            self.arm.move_to_pose(pose, time_ms=self.move_time_ms)

        elif action == "wait":
            self.arm.wait(ms)

        else:
            print(f"    ⚠ 未知动作: {action}")
            return True

        return True

    def run(self, workflow: list = None):
        """
        执行工作流。

        Args:
            workflow: 步骤列表，默认 DEFAULT_WORKFLOW
        """
        if workflow is None:
            workflow = DEFAULT_WORKFLOW

        total = len(workflow)
        print(f"\n{'='*55}")
        print(f"  机械臂工作流 — {total} 步")
        if self.step_mode:
            print(f"  单步模式: Enter=执行  s=跳过  p=急停  q=退出")
            print(f"  急停(p)会立即停止舵机并断开串口")
        print(f"{'='*55}")

        i = 0
        while i < total:
            step = workflow[i]
            self._show_step(i + 1, total, step)

            if self.step_mode:
                cmd = input("    → ").strip().lower()

                if cmd == 'q':
                    print("  用户退出。")
                    break
                elif cmd == 'p':
                    self.arm.panic("用户手动急停")
                    print("  ⚡ 已急停，串口断开。")
                    return
                elif cmd == 's':
                    print("  → 跳过\n")
                    i += 1
                    continue
                elif cmd == 'r':
                    # 回退一步（但不 undo 舵机位置）
                    if i > 0:
                        i -= 1
                        print(f"  ← 回退到步骤 {i+1}\n")
                    else:
                        print("  已是第一步")
                    continue
                elif cmd == '':
                    pass  # Enter = 执行
                else:
                    print(f"  未知命令: '{cmd}'  Enter=执行 s=跳过 p=急停 q=退出 r=后退")
                    continue

            try:
                ok = self._exec_step(step)
            except SafetyFatal:
                # arm_controller 已经 panic 了
                print("  ⚡ 安全系统触发急停，串口已断开。")
                return
            except SafetyError as e:
                print(f"  ⚠ 安全拒绝: {e}")
                if self.step_mode:
                    continue
                else:
                    break

            if not ok:
                print("  ⚠ 执行中断!")
                break

            # 执行后回显当前状态
            if self.step_mode:
                pos = self.arm.get_positions()
                parts = [f"#{s}={pos[s]}" for s in sorted(pos)]
                print(f"  ✓ 完成 → {' '.join(parts)}\n")

            i += 1

        if self.step_mode:
            print(f"工作流结束。机械臂停在最后位置。")
        else:
            print(f"工作流完成。")


# =========================================================================
# 位置文件预验证
# =========================================================================

def validate_positions(positions: dict) -> list:
    """
    加载位置文件后预校验所有脉冲值。
    返回警告列表，空列表表示全部通过。
    """
    warnings_list = []
    for name, pose in positions.items():
        for sid_str, pulse in pose.items():
            sid = int(sid_str)
            lo, hi = SERVO_LIMITS.get(sid, (500, 2500))
            if sid not in (1, 2, 3, 4, 5, 6):
                warnings_list.append(
                    f"FATAL: '{name}' 非法舵机ID #{sid}")
            elif pulse < 200 or pulse > 2800:
                warnings_list.append(
                    f"FATAL: '{name}' #{sid} P{pulse} 超出硬件安全范围 [200,2800]")
            elif pulse < lo or pulse > hi:
                warnings_list.append(
                    f"WARNING: '{name}' #{sid} P{pulse} 超出关节限位 [{lo},{hi}]")
    return warnings_list


# =========================================================================
# 干运行
# =========================================================================

def dry_run(workflow: list = None, positions: dict = None):
    """干运行：只打印步骤序列，不连接机械臂。"""
    if workflow is None:
        workflow = DEFAULT_WORKFLOW
    if positions is None:
        positions = EXAMPLE_POSITIONS

    total = len(workflow)
    print(f"\n{'='*55}")
    print(f"  干运行 (DRY RUN) — {total} 步")
    print(f"  不会连接机械臂，仅展示步骤序列")
    print(f"{'='*55}")

    # --- 预校验 ---
    val_warnings = validate_positions(positions)
    if val_warnings:
        print(f"\n  ⚠ 位置文件校验发现问题:")
        for w in val_warnings:
            print(f"    {w}")
        has_fatal = any("FATAL" in w for w in val_warnings)
        if has_fatal:
            print(f"\n  ❌ 存在致命错误，拒绝执行。请修正位置文件后重试。\n")
            return

    # 检查位置完整性
    required = set()
    for step in workflow:
        if step.get("action") == "move_to":
            required.add(step["position"])
    missing = required - set(positions.keys())
    if missing:
        print(f"\n  ⚠ 缺少位置定义: {', '.join(missing)}")
        print(f"  请用示教工具先记录这些位置。")
        print(f"  当前可用: {', '.join(sorted(positions.keys()))}")

    print(f"\n  已加载位置:")
    for name, pose in sorted(positions.items()):
        parts = [f"#{s}={pose[s]}" for s in sorted(pose.keys(), key=int)]
        print(f"    {name:<18} {' '.join(parts)}")

    print(f"\n  执行序列:")
    for i, step in enumerate(workflow, 1):
        action = step.get("action", "?")
        pos = step.get("position", "")
        ms = step.get("ms", 0)

        if action == "move_to":
            mark = " ✓" if pos in positions else " ⚠ 未定义!"
            print(f"  {i:2d}. MOVE → {pos}{mark}")
        elif action == "gripper_open":
            print(f"  {i:2d}. GRIPPER OPEN")
        elif action == "gripper_close":
            print(f"  {i:2d}. GRIPPER CLOSE")
        elif action == "home":
            print(f"  {i:2d}. HOME")
        elif action == "wait":
            print(f"  {i:2d}. WAIT {ms}ms")
        else:
            print(f"  {i:2d}. {step}")

    print(f"\n  干运行完成。确认无误后去掉 --dry-run 实机运行。\n")


# =========================================================================
# CLI
# =========================================================================

def main():
    p = argparse.ArgumentParser(description="LeArm 机械臂工作流执行器")
    p.add_argument("--port", type=str, default=None,
                   help="串口号 (如 COM3)")
    p.add_argument("--positions", type=str, default="positions.json",
                   help="示教位置 JSON 文件路径")
    p.add_argument("--workflow", type=str, default=None,
                   help="自定义工作流 JSON 文件路径")
    p.add_argument("--dry-run", action="store_true",
                   help="干运行模式 — 只打印序列不控制机械臂")
    p.add_argument("--step", action="store_true",
                   help="单步模式 — 每步需按回车确认")
    p.add_argument("--move-time", type=int, default=DEFAULT_MOVE_TIME,
                   help=f"默认移动时间 ms (默认: {DEFAULT_MOVE_TIME}, 最小: {DEFAULT_MOVE_TIME})")
    args = p.parse_args()

    # ---- 加载工作流 ----
    if args.workflow and os.path.exists(args.workflow):
        with open(args.workflow, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        print(f"已加载自定义工作流: {args.workflow}")
    else:
        workflow = DEFAULT_WORKFLOW

    # ---- 加载位置 ----
    positions = {}
    if os.path.exists(args.positions):
        with open(args.positions, "r", encoding="utf-8") as f:
            positions = json.load(f)
        print(f"已加载位置: {args.positions} ({len(positions)} 个)")
    elif args.dry_run:
        print(f"位置文件不存在，使用内置示例: {args.positions}")
        positions = dict(EXAMPLE_POSITIONS)
    else:
        print(f"位置文件不存在: {args.positions}")
        print("请先用 teach_positions.py 示教并保存位置。")
        return

    # ---- 预校验（实机和干运行都执行）----
    val_warnings = validate_positions(positions)
    if val_warnings:
        print(f"\n  ⚠ 位置文件校验:")
        for w in val_warnings:
            print(f"    {w}")
        has_fatal = any("FATAL" in w for w in val_warnings)
        if has_fatal:
            print(f"\n  ❌ 存在致命错误，拒绝执行。\n")
            return

    # ---- 干运行 ----
    if args.dry_run:
        dry_run(workflow, positions)
        return

    # ---- 实机 ----
    if not args.port:
        from arm_controller import ArmController as AC
        ports = AC.list_ports()
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
        print("无法连接机械臂。")
        return

    runner = WorkflowRunner(arm, positions,
                            step_mode=args.step,
                            move_time_ms=args.move_time)

    try:
        runner.run(workflow)
    except KeyboardInterrupt:
        print("\n\n⚠ Ctrl+C — 紧急停止!")
        runner.stop()
    except SafetyFatal as e:
        print(f"\n\n⚡ 致命安全事件: {e}")
        print("  机械臂已急停，串口已断开。")
    except SafetyError as e:
        print(f"\n\n⚠ 安全错误: {e}")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
