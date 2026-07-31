"""
positioning.py — 基于实测 PWM 的 IDW 精准定位

直接使用 calibration_final.json 中 17 个点的 pwm_safe / pwm_grip，
通过反距离加权插值(IDW)计算任意纸面坐标对应的舵机 PWM。

不再使用固件 IK (CMD_COORDINATE_SET)，全部走 CMD_MULT_SERVO_MOVE (CMD 3)。

用法:
    from positioning import PositionSolver

    solver = PositionSolver("calibration_final.json")

    # 纸面坐标 → PWM
    safe = solver.solve(10.0, 5.0, layer="safe")   # 安全高度
    grip = solver.solve(10.0, 5.0, layer="grip")   # 抓取高度

    # 直接控制机械臂
    arm = ArmController("COM7")
    arm.connect()
    arm.move_to_pose(safe, time_ms=2000)   # 到安全高度
    arm.move_to_pose(grip, time_ms=1500)   # 下降到抓取高度
"""

import json
import math
import os
from typing import Dict, List, Tuple


class PositionSolver:
    """纸面坐标 → PWM 的 IDW 定位求解器

    坐标系:
      X: 纸面短边, 0(近臂端) → 20.9cm(远臂端), 单位 cm
      Y: 纸面长边, -14.7(最左) → +14.7(最右), 原点=正中央, 单位 cm

    分层:
      "safe" — 纸面以上 ~35mm 安全高度, 用于平移
      "grip" — 纸面接触高度, 用于电磁铁吸附/夹取
    """

    def __init__(self, calib_path: str = None):
        if calib_path is None:
            calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "calibration_final.json")
        with open(calib_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.points: List[Tuple[float, float, Dict[str, Dict[int, int]]]] = []
        for p in raw["points"]:
            x, y = float(p["x"]), float(p["y"])
            layers = {
                "safe": {int(k): int(v) for k, v in p["pwm_safe"].items()},
                "grip": {int(k): int(v) for k, v in p["pwm_grip"].items()},
            }
            self.points.append((x, y, layers))

        self.x_min = min(p[0] for p in self.points)
        self.x_max = max(p[0] for p in self.points)
        self.y_min = min(p[1] for p in self.points)
        self.y_max = max(p[1] for p in self.points)
        self._n = len(self.points)

    # ------------------------------------------------------------------
    # IDW 插值
    # ------------------------------------------------------------------

    def solve(self, x_cm: float, y_cm: float, layer: str = "safe",
              power: float = 2.5) -> Dict[int, int]:
        """
        纸面坐标 → 6路PWM

        Args:
            x_cm: X坐标 (cm), 0=近臂端, 20.9=远臂端
            y_cm: Y坐标 (cm), 0=正中央, -14.7=最左, +14.7=最右
            layer: "safe" | "grip"
            power: IDW 距离幂次, 越大越偏向最近标定点, 推荐 2.0~3.0

        Returns:
            {1: pwm, 2: pwm, 3: pwm, 4: pwm, 5: pwm, 6: pwm}
        """
        # 钳位到标定范围
        x = max(self.x_min, min(self.x_max, x_cm))
        y = max(self.y_min, min(self.y_max, y_cm))

        wsum = {3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
        norm = 0.0

        for cx, cy, layers in self.points:
            d = math.hypot(x - cx, y - cy)

            # 命中标定点 → 直接返回实测值 (精度 = 标定精度)
            if d < 0.05:  # 0.5mm
                result = {1: 1500, 2: 1500}
                result.update(layers[layer])
                return result

            w = 1.0 / (d ** power)
            norm += w
            pwm = layers[layer]
            for sid in (3, 4, 5, 6):
                wsum[sid] += w * pwm[sid]

        return {
            1: 1500, 2: 1500,
            3: int(round(wsum[3] / norm)),
            4: int(round(wsum[4] / norm)),
            5: int(round(wsum[5] / norm)),
            6: int(round(wsum[6] / norm)),
        }

    # ------------------------------------------------------------------
    # 搬运路径规划
    # ------------------------------------------------------------------

    def plan_pickup(self, x_cm: float, y_cm: float):
        """拾取路径: 返回 [(pose, time_ms, description), ...]"""
        safe = self.solve(x_cm, y_cm, "safe")
        grip = self.solve(x_cm, y_cm, "grip")
        return [
            (safe, 2000, f"安全高度 ({x_cm},{y_cm})"),
            (grip, 1500, f"下降到抓取面"),
            (None, 500,  "等待吸附"),          # 调用者处理电磁铁
            (safe, 1500, f"抬升回安全高度"),
        ]

    def plan_place(self, x_cm: float, y_cm: float):
        """放置路径: 返回 [(pose, time_ms, description), ...]"""
        safe = self.solve(x_cm, y_cm, "safe")
        grip = self.solve(x_cm, y_cm, "grip")
        return [
            (safe, 2000, f"安全高度 ({x_cm},{y_cm})"),
            (grip, 1500, f"下降到放置面"),
            (None, 300,  "等待释放"),          # 调用者处理电磁铁
            (safe, 1500, f"抬升回安全高度"),
        ]

    # ------------------------------------------------------------------
    # 精度评估
    # ------------------------------------------------------------------

    def cross_validate(self, holdout_count: int = 3, power: float = 2.5):
        """
        留出法交叉验证: 报告 IDW 预测值与实测值的偏差。

        依次把每个点作为"未知点"，用其余 16 个点插值预测它的 PWM，
        统计全部 17 个点的误差分布。
        """
        print(f"\n{'='*55}")
        print(f"  IDW 交叉验证 (power={power}, {self._n}点留一法)")
        print(f"{'='*55}")

        all_errors = {"safe": {3:[],4:[],5:[],6:[]},
                       "grip": {3:[],4:[],5:[],6:[]}}

        for holdout_idx in range(self._n):
            hx, hy, hlayers = self.points[holdout_idx]

            # 用其余点插值
            for layer in ("safe", "grip"):
                predicted = self._idw_leave_one_out(hx, hy, layer, holdout_idx, power)
                actual = hlayers[layer]
                for sid in (3, 4, 5, 6):
                    all_errors[layer][sid].append(abs(predicted[sid] - actual[sid]))

        # 报告
        for layer in ("safe", "grip"):
            print(f"\n  [{layer}层]")
            print(f"  {'舵机':<8} {'平均误差':>8} {'最大误差':>8} {'角度误差':>8}")
            print(f"  {'-'*40}")
            for sid in (3, 4, 5, 6):
                errs = all_errors[layer][sid]
                avg = sum(errs) / len(errs)
                mx = max(errs)
                print(f"  #{sid:<7} {avg:>7.1f}PWM {mx:>7.0f}PWM {mx*0.09:>7.1f}°")

    def _idw_leave_one_out(self, x, y, layer, skip_idx, power):
        """留一法 IDW: 排除 skip_idx 号点"""
        wsum = {3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
        norm = 0.0
        for i, (cx, cy, layers) in enumerate(self.points):
            if i == skip_idx:
                continue
            d = math.hypot(x - cx, y - cy)
            if d < 0.05:
                return dict(layers[layer])
            w = 1.0 / (d ** power)
            norm += w
            pwm = layers[layer]
            for sid in (3, 4, 5, 6):
                wsum[sid] += w * pwm[sid]
        return {sid: int(round(wsum[sid] / max(norm, 1e-9))) for sid in (3,4,5,6)}


# =========================================================================
# 自检入口
# =========================================================================

if __name__ == "__main__":
    solver = PositionSolver()

    print("=== 标定数据概况 ===")
    print(f"标定点数: {solver._n}")
    print(f"X 范围: {solver.x_min} ~ {solver.x_max} cm")
    print(f"Y 范围: {solver.y_min} ~ {solver.y_max} cm")
    print()

    # 验证: 标定点自检 (输入 = 标定点坐标, 插值应等于实测)
    print("=== 标定点自检 (应全部通过) ===")
    all_ok = True
    for x, y, layers in solver.points:
        for layer in ("safe", "grip"):
            p = solver.solve(x, y, layer)
            actual = layers[layer]
            for sid in (3, 4, 5, 6):
                if p[sid] != actual[sid]:
                    print(f"  FAIL ({x},{y}) {layer} #{sid}: {p[sid]} != {actual[sid]}")
                    all_ok = False
    if all_ok:
        print("  全部 17 个点 safe + grip 层自检通过 [OK]")
    print()

    # 留一法交叉验证
    solver.cross_validate(holdout_count=1, power=2.5)

    # 演示: 查询非标定点
    print(f"\n=== 演示: 查询非标定点 ===")
    for tx, ty in [(5, 0), (8, -3), (12, 6), (15, -10)]:
        safe = solver.solve(tx, ty, "safe")
        grip = solver.solve(tx, ty, "grip")
        print(f"  ({tx:5.1f}, {ty:5.1f}) "
              f"safe: {{3:{safe[3]}, 4:{safe[4]}, 5:{safe[5]}, 6:{safe[6]}}}  "
              f"grip: {{3:{grip[3]}, 4:{grip[4]}, 5:{grip[5]}, 6:{grip[6]}}}")
