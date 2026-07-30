"""
pipeline.py — 检测 → 求解 → 机械臂指令 桥接层

输入: 检测器输出的多边形（像素坐标）+ 目标矩形尺寸
输出: arm_instructions 兼容的指令列表

步骤:
  1. 多边形清洗（去共线、合并近点、RDP重逼近）
  2. 数量守卫 + 总面积守卫
  3. 调用 puzzle_assembler.solve()
  4. 矩阵 → (x,y,θ) + 像素→mm
  5. 外边排序
"""

import math, cv2, numpy as np


def clean_polygon(poly, min_edge_px=5, collinear_tol_deg=3, merge_dist_px=3):
    """
    清洗检测到的多边形。

    步骤:
      1. 合并近邻顶点 (dist < merge_dist_px → 取中点)
      2. 删除共线顶点 (夹角 > 180-collinear_tol_deg → 删中间点)
      3. RDP 重逼近 (epsilon 0.008→0.012→0.018→0.025→0.035)
         直到 3 ≤ 顶点数 ≤ 5
      4. 最小边长标记 (len < min_edge_px 的短边不参与配对)
      5. CCW 排序

    Returns: 清洗后的 (N,2) ndarray，或 None（失败）
    """
    if len(poly) < 3:
        return None

    pts = poly.astype(np.float32).copy()

    # 1. 合并近邻顶点
    if merge_dist_px > 0 and len(pts) > 3:
        merged = [pts[0]]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - merged[-1]) > merge_dist_px:
                merged.append(pts[i])
            else:
                merged[-1] = (merged[-1] + pts[i]) / 2.0
        # 检查首尾
        if len(merged) >= 3 and np.linalg.norm(merged[-1] - merged[0]) <= merge_dist_px:
            merged[0] = (merged[0] + merged[-1]) / 2.0
            merged.pop()
        pts = np.array(merged, dtype=np.float32)

    if len(pts) < 3:
        return None

    # 2. 删除共线顶点
    if collinear_tol_deg > 0 and len(pts) > 3:
        keep = []
        n = len(pts)
        for i in range(n):
            prev = pts[(i - 1) % n]
            curr = pts[i]
            nxt = pts[(i + 1) % n]
            v1 = prev - curr; v2 = nxt - curr
            len1 = np.linalg.norm(v1); len2 = np.linalg.norm(v2)
            if len1 < 1e-6 or len2 < 1e-6:
                continue
            cos_a = np.dot(v1, v2) / (len1 * len2)
            cos_a = max(-1.0, min(1.0, cos_a))
            angle = math.degrees(math.acos(cos_a))
            if angle < (180 - collinear_tol_deg):
                keep.append(curr)
        if len(keep) >= 3:
            pts = np.array(keep, dtype=np.float32)

    # 3. RDP 重逼近
    if len(pts) < 3:
        return None
    contour = pts.reshape(-1, 1, 2).astype(np.int32)
    peri = cv2.arcLength(contour, True)
    for eps_factor in [0.008, 0.012, 0.018, 0.025, 0.035]:
        approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
        if 3 <= len(approx) <= 5:
            pts = approx.reshape(-1, 2).astype(np.float32)
            break
    else:
        # 兜底：取离质心最远5个顶点 → convexHull
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) > 5:
            centroid = pts.mean(axis=0)
            dists = np.linalg.norm(pts - centroid, axis=1)
            top5 = pts[np.argsort(dists)[-5:]]
            hull = cv2.convexHull(top5.reshape(-1, 1, 2).astype(np.int32))
            pts = hull.reshape(-1, 2).astype(np.float32)

    if len(pts) < 3:
        return None

    # 4 & 5: 短边标记 + CCW 排序（短边仅标记，不删除；CCW确保方向一致）
    # 确保 CCW
    n = len(pts)
    signed_area = sum(pts[i][0] * pts[(i+1)%n][1] - pts[(i+1)%n][0] * pts[i][1]
                      for i in range(n))
    if signed_area < 0:
        pts = pts[::-1]

    return pts


# =========================================================================
# 求解 + 坐标转换
# =========================================================================

def solve_and_convert(solver_polys, target_w_mm, target_h_mm, px_per_mm):
    """
    清洗 → 求解 → 提取坐标。

    Args:
        solver_polys: list of (N,2) ndarray，像素坐标
        target_w_mm, target_h_mm: 目标矩形物理尺寸 (mm)
        px_per_mm: 像素/mm 比例

    Returns:
        dict: {"ok": True, "instructions": [...]} 或 {"ok": False, "error": "..."}
    """
    import puzzle_assembler

    # ---- 守卫 ----
    if len(solver_polys) != 4:
        return {"ok": False, "error": f"碎片数量异常: {len(solver_polys)}，期望 4"}

    # ---- 清洗 ----
    cleaned = []
    for i, poly in enumerate(solver_polys):
        cp = clean_polygon(poly)
        if cp is None:
            return {"ok": False, "error": f"碎片 F{i} 多边形清洗失败"}
        cleaned.append(cp)

    # ---- 总面积守卫 ----
    target_w_px = target_w_mm * px_per_mm
    target_h_px = target_h_mm * px_per_mm
    expected = target_w_px * target_h_px
    actual = sum(abs(cv2.contourArea(c.astype(np.int32).reshape(-1, 1, 2)))
                 for c in cleaned)
    ratio = actual / expected if expected > 0 else 0
    if ratio < 0.75 or ratio > 1.20:
        return {"ok": False,
                "error": f"碎片总面积异常: {ratio*100:.0f}% (允许 75-120%)"}

    # ---- 求解 ----
    transforms, score, match_count = puzzle_assembler.solve(
        cleaned, target_w_px, target_h_px, timeout_s=5.0)

    if transforms is None:
        return {"ok": False,
                "error": f"求解失败: 最佳评分 {score:.1f}，未找到有效拼法"}

    # ---- 提取指令 ----
    instructions = []
    for i, (poly, h) in enumerate(zip(cleaned, transforms)):
        # 质心
        cx = poly[:, 0].mean()
        cy = poly[:, 1].mean()

        # 目标位置
        target_pt = puzzle_assembler.apply_h(np.array([[cx, cy]]), h)[0]

        # 旋转角度
        angle_rad = math.atan2(h[1, 0], h[0, 0])
        angle_deg = math.degrees(angle_rad)
        angle_deg = round(angle_deg % 360, 1)
        if angle_deg > 180:
            angle_deg -= 360

        # 像素 → mm
        pickup_x = round(cx / px_per_mm, 1)
        pickup_y = round((297.0 - cy / px_per_mm), 1)  # y轴翻转
        place_x = round(target_pt[0] / px_per_mm, 1)
        place_y = round((297.0 - target_pt[1] / px_per_mm), 1)

        dist = round(math.hypot(place_x - pickup_x, place_y - pickup_y), 1)

        instructions.append({
            "piece_id": i,
            "num_edges": len(poly),
            "pickup_mm": (pickup_x, pickup_y),
            "place_mm": (place_x, place_y),
            "rotation_deg": angle_deg,
            "distance_mm": dist,
        })

    # ---- 外边排序（离纸边近的先放） ----
    def boundary_dist(inst):
        x, y = inst["place_mm"]
        return -min(x, target_w_mm - x, y, target_h_mm - y)
    instructions.sort(key=boundary_dist)

    return {
        "ok": True,
        "instructions": instructions,
        "target_mm": (target_w_mm, target_h_mm),
        "score": round(score, 2),
        "matches": match_count,
    }


# =========================================================================
# 离线测试入口
# =========================================================================

def test_on_gt(gt_path, px_per_mm=3.0):
    """用 ground_truth JSON 中的碎片顶点测试求解器。"""
    import json
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    rw, rh = gt["target_rectangle_size_mm"]
    frags = gt["fragments"]

    # GT中顶点是mm坐标 → 缩放为像素坐标（求解器工作在像素空间）
    polys = []
    for fg in frags:
        p = np.array([(v[0] * px_per_mm, v[1] * px_per_mm) for v in fg["vertices_rect_mm"]],
                     dtype=np.float32)
        polys.append(p)

    # 加点噪声模拟真实检测
    rng = np.random.RandomState(42)
    noisy = []
    for p in polys:
        noise = rng.normal(0, 0.5, p.shape).astype(np.float32)
        noisy.append(p + noise)

    result = solve_and_convert(noisy, rw, rh, px_per_mm)
    return result


if __name__ == "__main__":
    import os, json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_dir = os.path.join(script_dir, "..", "vision", "test_data", "ground_truth")
    files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".json")])
    if files:
        print(f"测试 {files[0]} ...")
        r = test_on_gt(os.path.join(gt_dir, files[0]))
        if r["ok"]:
            print(f"  求解成功! 评分: {r['score']}  匹配:{r['matches']}")
            for inst in r["instructions"]:
                print(f"  F{inst['piece_id']}: pickup({inst['pickup_mm'][0]:.1f},{inst['pickup_mm'][1]:.1f}) "
                      f"→ place({inst['place_mm'][0]:.1f},{inst['place_mm'][1]:.1f}) "
                      f"rot={inst['rotation_deg']:.1f}° dist={inst['distance_mm']:.1f}mm")
        else:
            print(f"  求解失败: {r['error']}")
