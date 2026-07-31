"""
calibration.py — 工作区域插值坐标系 (Inverse Distance Weighting)

坐标系: 原点=左下, X轴 下→上(0→168mm), Y轴 左→右(0→286mm)

标定点越多越准，当前 5 点:
  左下 (0,0)      #3=845  #4=521  #5=1633 #6=2088
  右下 (0,286)    #3=847  #4=515  #5=1651 #6=865
  左上 (168,0)    #3=1221 #4=899  #5=2065 #6=1815
  右上 (168,286)  #3=1142 #4=1059 #5=2146 #6=1181
  正中 (84,143)   #3=892  #4=523  #5=1659 #6=1486
"""

import json
import math

# 矩形物理尺寸
X_RANGE = 168
Y_RANGE = 286

# IDW 权重指数 (越高越偏向最近点, 2=标准)
IDW_POWER = 2

# 标定点列表: [(x, y, {servo_id: pulse}), ...]
# 注意: 所有标定点夹爪均为1500, 腕翻均为1500, 略去不存
_CAL_POINTS = [
    (  0,   0, {3:  845, 4: 521,  5: 1633, 6: 2088}),  # 左下
    (  0, 286, {3:  847, 4: 515,  5: 1651, 6:  865}),  # 右下
    (168,   0, {3: 1221, 4: 899,  5: 2065, 6: 1815}),  # 左上
    (168, 286, {3: 1142, 4: 1059, 5: 2146, 6: 1181}),  # 右上
    ( 84, 143, {3:  892, 4: 523,  5: 1659, 6: 1486}),  # 正中
]

# 安全高度
SAFE_POSE = {1: 1500, 2: 1500, 3: 640, 4: 511, 5: 1255, 6: 1500}

GRIPPER_OPEN  = 600
GRIPPER_CLOSE = 1500


def add_cal_point(x_mm: float, y_mm: float, pulses: dict):
    """添加新的标定点以提高精度"""
    _CAL_POINTS.append((x_mm, y_mm, {k: v for k, v in pulses.items() if k in (3, 4, 5, 6)}))


def interpolate(x_mm: float, y_mm: float) -> dict:
    """
    逆距离加权插值 (IDW): 用所有标定点加权平均。
    标定点越近权重越大，角点处精确=标定值。
    """
    x = max(0, min(X_RANGE, x_mm))
    y = max(0, min(Y_RANGE, y_mm))

    # 距离最近点 < 1mm → 直接返回
    for cx, cy, cp in _CAL_POINTS:
        if abs(x - cx) < 1 and abs(y - cy) < 1:
            pose = {1: 1500, 2: 1500}
            pose.update(cp)
            return pose

    # IDW
    weights = {}
    for cx, cy, cp in _CAL_POINTS:
        d = math.hypot(x - cx, y - cy)
        if d < 0.01:
            d = 0.01  # 防除零
        w = 1.0 / (d ** IDW_POWER)
        for sid, pulse in cp.items():
            weights[sid] = weights.get(sid, 0) + w * pulse
        weights["_sum"] = weights.get("_sum", 0) + w

    pose = {1: 1500, 2: 1500}
    wsum = weights.pop("_sum", 1)
    for sid in (3, 4, 5, 6):
        pose[sid] = int(weights.get(sid, 0) / wsum)

    return pose


def safe_pose(grip_open: bool = True) -> dict:
    p = dict(SAFE_POSE)
    p[1] = GRIPPER_OPEN if grip_open else GRIPPER_CLOSE
    return p


# =========================================================================
# 测试 & 验证
# =========================================================================

if __name__ == "__main__":
    print("=== 标定数据 ===")
    print(f"坐标系: 原点=左下, X↑ 0~{X_RANGE}mm, Y→ 0~{Y_RANGE}mm")
    print(f"插值方式: IDW (power={IDW_POWER}), {len(_CAL_POINTS)} 个标定点")
    print(f"安全高度: {' '.join(f'#{s}={SAFE_POSE[s]}' for s in sorted(SAFE_POSE))}")

    print(f"\n标定点自检 (应精确复现):")
    for cx, cy, cp in _CAL_POINTS:
        p = interpolate(cx, cy)
        errors = []
        for sid in (3, 4, 5, 6):
            e = p[sid] - cp[sid]
            if e != 0:
                errors.append(f"#{sid}:{e:+d}")
        status = " ✓" if not errors else f" ⚠ {' '.join(errors)}"
        print(f"  ({cx:3d},{cy:3d}): {' '.join(f'#{s}={p[s]}' for s in sorted(p))}{status}")

    print(f"\n实测验证点:")
    # 中心实测: #3=892 #4=523 #5=1659 #6=1486
    p = interpolate(84, 143)
    print(f"  中心(84,143) 插值: {' '.join(f'#{s}={p[s]}' for s in sorted(p))}")
    print(f"  中心(84,143) 实测: #1=1500 #2=1500 #3=892 #4=523 #5=1659 #6=1486")

    # (80,150) - 对比之前双线性
    p = interpolate(80, 150)
    print(f"\n  (80,150) IDW: {' '.join(f'#{s}={p[s]}' for s in sorted(p))}")
    p_old_bilinear = {1:1500, 2:1500, 3:1004, 4:739, 5:1863, 6:1463}
    print(f"  (80,150) 旧双线性: {' '.join(f'#{s}={p_old_bilinear[s]}' for s in sorted(p_old_bilinear))}")
