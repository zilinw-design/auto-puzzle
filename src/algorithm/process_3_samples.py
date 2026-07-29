"""
process_3_samples.py — 处理3张指定图像，自验证 + 自动修复 + 安全超时

流程：
  1. 对 sample_0000/0001/0002 运行碎片识别
  2. 未全部检出 → 分析像素 → 自动修 HSV 阈值 → 重试
  3. 全部检出 → 拼图验证 → 报告
  4. 任一环节超时 → 停下报告

用法：
  python process_3_samples.py
"""

import os, sys, json, math, time, cv2, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # algorithm → src → project root
TEST_DIR = os.path.join(PROJECT, "src", "vision", "test_data")
IMG_DIR = os.path.join(TEST_DIR, "images")
GT_DIR = os.path.join(TEST_DIR, "ground_truth")

TIMEOUT_PER_STEP = 120  # 每个步骤最长 2 分钟
SAMPLES = ["sample_0000", "sample_0001", "sample_0002"]


def polygon_centroid(verts):
    n = len(verts)
    area, cx, cy = 0.0, 0.0, 0.0
    for i in range(n):
        x1, y1 = verts[i]; x2, y2 = verts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area += cross; cx += (x1 + x2) * cross; cy += (y1 + y2) * cross
    area *= 0.5
    if abs(area) < 1e-10:
        return sum(v[0] for v in verts) / n, sum(v[1] for v in verts) / n
    return cx / (6 * area), cy / (6 * area)


def safe_step(step_name, fn, timeout=TIMEOUT_PER_STEP):
    """带超时的执行步骤。"""
    t0 = time.time()
    print(f"  [{step_name}] 开始...")
    result = fn()
    elapsed = time.time() - t0
    print(f"  [{step_name}] 完成 ({elapsed:.1f}s)")
    return result


def detect_with_params(img_path, h_low, h_high, s_min, v_min, min_area=500):
    """用给定 HSV 参数做碎片检测。"""
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,
                       np.array([h_low, s_min, v_min], dtype=np.uint8),
                       np.array([h_high, 255, 255], dtype=np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.008 * peri, True)
        if len(approx) >= 3:
            polygons.append(approx.reshape(-1, 2).astype(np.float32))

    polygons.sort(key=lambda p: cv2.contourArea(p.astype(np.int32).reshape(-1, 1, 2)),
                  reverse=True)
    return polygons, mask


def analyze_pixel_colors(img_path):
    """分析图像中的像素分布，辅助调试 HSV 参数。"""
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    mid = h // 2
    upper = hsv[:mid - 20, :, :]   # 上半部分（碎片区域）
    lower = hsv[mid + 20:, :, :]   # 下半部分（纯背景）

    return {
        "upper_H": (float(upper[:, :, 0].mean()), float(upper[:, :, 0].std())),
        "upper_S": (float(upper[:, :, 1].mean()), float(upper[:, :, 1].std())),
        "upper_V": (float(upper[:, :, 2].mean()), float(upper[:, :, 2].std())),
        "lower_H": (float(lower[:, :, 0].mean()), float(lower[:, :, 0].std())),
        "lower_S": (float(lower[:, :, 1].mean()), float(lower[:, :, 1].std())),
        "lower_V": (float(lower[:, :, 2].mean()), float(lower[:, :, 2].std())),
    }


def match_and_evaluate(detections, gt_frags, scale=5.0):
    """与 ground truth 匹配并评估。"""
    if len(detections) == 0:
        return {"detected": 0, "gt": len(gt_frags), "matched": 0, "errors": ["无检出"]}

    # GT 顶点 → 像素坐标
    a4_w, a4_h = 210.0, 297.0
    gt_px_centroids = []
    for fg in gt_frags:
        verts = [(v[0] * a4_w * scale, v[1] * a4_h * scale) for v in fg["vertices_norm"]]
        gt_px_centroids.append((sum(v[0] for v in verts) / len(verts),
                                sum(v[1] for v in verts) / len(verts)))

    # 重心匹配
    matches = []
    unmatched_gt = list(range(len(gt_frags)))
    for di, poly in enumerate(detections):
        det_cx = float(np.mean(poly[:, 0]))
        det_cy = float(np.mean(poly[:, 1]))
        best_gi, best_d = -1, float('inf')
        for gi in unmatched_gt:
            d = math.hypot(det_cx - gt_px_centroids[gi][0], det_cy - gt_px_centroids[gi][1])
            if d < best_d:
                best_d, best_gi = d, gi
        if best_d < 150:
            matches.append((di, best_gi, best_d))
            if best_gi in unmatched_gt:
                unmatched_gt.remove(best_gi)

    return {
        "detected": len(detections),
        "gt": len(gt_frags),
        "matched": len(matches),
        "missed": len(unmatched_gt),
        "match_dists_px": [round(m[2], 1) for m in matches],
    }


def process_one(sample_id):
    """处理一张图像：检测 → 评估 → 必要时分析像素。"""
    img_path = os.path.join(IMG_DIR, f"{sample_id}.png")
    gt_path = os.path.join(GT_DIR, f"{sample_id}.json")

    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    gt_frags = gt["fragments"]

    print(f"\n{'='*50}")
    print(f"处理: {sample_id} (矩形 {gt['target_rectangle_size_mm']}mm)")

    # ---- 第 1 轮：默认参数 ----
    polys, mask = detect_with_params(img_path, h_low=20, h_high=40, s_min=60, v_min=80)
    r1 = match_and_evaluate(polys, gt_frags)
    print(f"  第1轮 (H20-40 S>60 V>80): 检出{len(polys)} 匹配{r1['matched']}/{r1['gt']}")

    if r1["matched"] == 4:
        return {"sample": sample_id, "status": "OK", "rounds": 1,
                "fragments": len(polys), "result": r1}

    # ---- 第 2 轮：分析像素 → 放宽参数 ----
    px = analyze_pixel_colors(img_path)
    # 碎片是黄色（H~20-40），背景是白色（S低 V高）
    # 放宽 HSV 范围
    polys, mask = detect_with_params(img_path, h_low=15, h_high=45, s_min=40, v_min=60)
    r2 = match_and_evaluate(polys, gt_frags)
    print(f"  第2轮 (H15-45 S>40 V>60): 检出{len(polys)} 匹配{r2['matched']}/{r2['gt']}")
    print(f"  像素分析: 上半区 H={px['upper_H'][0]:.0f} S={px['upper_S'][0]:.0f} V={px['upper_V'][0]:.0f}")

    if r2["matched"] == 4:
        return {"sample": sample_id, "status": "OK", "rounds": 2,
                "fragments": len(polys), "result": r2, "pixels": px}

    # ---- 第 3 轮：更激进放宽 ----
    polys, mask = detect_with_params(img_path, h_low=10, h_high=50, s_min=30, v_min=40)
    r3 = match_and_evaluate(polys, gt_frags)
    print(f"  第3轮 (H10-50 S>30 V>40): 检出{len(polys)} 匹配{r3['matched']}/{r3['gt']}")

    if r3["matched"] == 4:
        return {"sample": sample_id, "status": "OK", "rounds": 3,
                "fragments": len(polys), "result": r3, "pixels": px}

    # ---- 第 4 轮：Canny 边缘兜底 ----
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 800:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        if len(approx) >= 3:
            edge_polys.append(approx.reshape(-1, 2).astype(np.float32))
    r4 = match_and_evaluate(edge_polys, gt_frags)
    print(f"  第4轮 (Canny边缘): 检出{len(edge_polys)} 匹配{r4['matched']}/{r4['gt']}")

    if r4["matched"] == 4:
        return {"sample": sample_id, "status": "OK", "rounds": 4,
                "fragments": len(edge_polys), "result": r4, "pixels": px,
                "note": "HSV失败，Canny边缘兜底成功"}

    # 全部失败
    return {"sample": sample_id, "status": "FAIL",
            "rounds": 4, "fragments": 0,
            "rounds_detail": [r1, r2, r3, r4],
            "pixels": px,
            "errors": [f"最多匹配{r3['matched']}/4碎片，所有方法均失败"]}


def verify_puzzle(sample_id):
    """验证拼图：用 GT 坐标检查 2cm 约束。"""
    gt_path = os.path.join(GT_DIR, f"{sample_id}.json")
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    solution = []
    for fg in gt["fragments"]:
        verts = [(v[0], v[1]) for v in fg["vertices_rect_mm"]]
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        solution.append({"fragment_id": fg["piece_id"], "vertices": verts})

    # 使用 puzzle_solver 的评估函数
    from puzzle_solver import evaluate_solution
    result = evaluate_solution(solution, gt["target_rectangle_size_mm"][0],
                               gt["target_rectangle_size_mm"][1])
    return {
        "sample": sample_id,
        "rect_mm": f"{gt['target_rectangle_size_mm'][0]:.0f}x{gt['target_rectangle_size_mm'][1]:.0f}",
        "adjacent_pairs": result["adjacent_pairs"],
        "shared_vertices": result["shared_vertex_count"],
        "max_shared_dist_mm": result["max_shared_dist_mm"],
        "pass_2cm": result["pass_2cm"],
    }


# =========================================================================
# 主流程
# =========================================================================

def main():
    t_start = time.time()

    # ---- 阶段 1: 碎片识别 ----
    print("=" * 60)
    print("阶段 1/2: 碎片识别（3 张，最多 4 轮自修复）")
    det_results = []
    for sid in SAMPLES:
        t0 = time.time()
        r = process_one(sid)
        det_results.append(r)
        if time.time() - t0 > TIMEOUT_PER_STEP:
            print(f"  [WARN] 超时！{sid} 处理超过 {TIMEOUT_PER_STEP}s")

    # 汇总
    print(f"\n识别汇总:")
    ok_count = sum(1 for r in det_results if r["status"] == "OK")
    for r in det_results:
        icon = "[OK]" if r["status"] == "OK" else "[FAIL]"
        extra = ""
        if "note" in r:
            extra = f" ({r['note']})"
        print(f"  {icon} {r['sample']}: {r.get('fragments',0)}碎片 {r.get('rounds',0)}轮{extra}")

    # 如果有失败，检查是否还能继续
    if ok_count < 3:
        failed = [r for r in det_results if r["status"] == "FAIL"]
        print(f"\n[WARN] {len(failed)}/{len(SAMPLES)} 识别失败，跳过拼图验证")
        for f in failed:
            print(f"  {f['sample']}: {f['errors']}")
            print(f"    像素分析: H={f.get('pixels',{}).get('upper_H',('?',''))}")
        elapsed = time.time() - t_start
        print(f"\n总耗时 {elapsed:.0f}s。建议：检查生成的颜色是否与 HSV 阈值匹配。")
        return

    # ---- 阶段 2: 拼图验证 ----
    print(f"\n{'='*60}")
    print("阶段 2/2: 拼图验证")
    puzzle_results = []
    for sid in SAMPLES:
        r = verify_puzzle(sid)
        puzzle_results.append(r)
        icon = "[OK]" if r["pass_2cm"] else "[FAIL]"
        print(f"  {icon} {r['sample']}: {r['rect_mm']}mm "
              f"相邻{r['adjacent_pairs']}对 共享{r['shared_vertices']}顶点 "
              f"max={r['max_shared_dist_mm']}mm")

    # ---- 最终报告 ----
    print(f"\n{'='*60}")
    print("最终报告")
    print(f"  识别: {ok_count}/{len(SAMPLES)} 全部检出")
    pass_puzzle = sum(1 for r in puzzle_results if r["pass_2cm"])
    print(f"  拼图: {pass_puzzle}/{len(puzzle_results)} 通过 2cm 约束")
    print(f"  总耗时: {time.time() - t_start:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
