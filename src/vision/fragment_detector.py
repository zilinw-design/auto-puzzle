"""
fragment_detector.py — 碎片识别与评估

从生成的测试图像中识别碎片，输出每个碎片的多边形顶点。
配合 dataset_generator.py 的 ground_truth JSON 做精度评估。

识别策略（非深度学习，纯 OpenCV）：
  1. 读图
  2. HSV 色彩空间分割（深色背景 vs 亮色碎片）
  3. 形态学去噪
  4. findContours → approxPolyDP 多边形逼近
  5. 几何特征提取（顶点、重心、方向）

评估：
  - 与 ground_truth 对比，重心匹配 → 顶点误差、检出率

用法：
  python fragment_detector.py                             # 跑 test_data 下全部
  python fragment_detector.py --image path/to/img.png     # 单张
  python fragment_detector.py --image img.png --gt gt.json # 单张+评估
"""

import os
import sys
import json
import math
import time
import argparse
import cv2
import numpy as np


# =========================================================================
# 几何工具
# =========================================================================

def polygon_centroid(verts):
    n = len(verts)
    area, cx, cy = 0.0, 0.0, 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area *= 0.5
    if abs(area) < 1e-10:
        return sum(v[0] for v in verts) / n, sum(v[1] for v in verts) / n
    return cx / (6 * area), cy / (6 * area)


# =========================================================================
# HSV 分割
# =========================================================================

def segment_fragments(img_bgr):
    """
    HSV 分割：黄色碎片 vs 白色背景。

    白色: S 低, V 高
    黄色: H∈[20,40], S 高, V 高
    黑色分界线: V 低

    策略：在 V 通道排除暗色（分界线）→ 在 H+S 通道锁定黄色。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    # 排除暗色区域（黑色分界线 V < 80）
    _, mask_bright = cv2.threshold(hsv[:, :, 2], 80, 255, cv2.THRESH_BINARY)

    # 黄色范围：H 在 20-40，S 足够高（排除白色背景）
    mask_yellow = cv2.inRange(hsv,
                              np.array([20, 60, 80], dtype=np.uint8),
                              np.array([40, 255, 255], dtype=np.uint8))

    # 合并：既亮且黄
    mask = cv2.bitwise_and(mask_bright, mask_yellow)

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


# =========================================================================
# 轮廓提取
# =========================================================================

def extract_polygons(mask, min_area_px=500, epsilon_ratio=0.008):
    """
    从二值掩码提取碎片多边形。

    :param mask: 二值图（碎片=255）
    :param min_area_px: 最小面积阈值
    :param epsilon_ratio: approxPolyDP epsilon = epsilon_ratio × 周长
    :return: list of (N, 2) ndarray，每个是多边形顶点（像素坐标）
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
        if len(approx) < 3:
            continue
        polygons.append(approx.reshape(-1, 2).astype(np.float32))

    # 按面积降序
    polygons.sort(key=lambda p: cv2.contourArea(p.astype(np.int32).reshape(-1, 1, 2)),
                  reverse=True)
    return polygons


# =========================================================================
# 单张检测
# =========================================================================

def detect(image_path, min_area=500, epsilon=0.008):
    """读取图像 → 分割 → 提取多边形 → 返回碎片列表。"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取: {image_path}")

    mask = segment_fragments(img)
    polys = extract_polygons(mask, min_area_px=min_area, epsilon_ratio=epsilon)

    results = []
    for i, poly in enumerate(polys):
        verts = [(float(v[0]), float(v[1])) for v in poly]
        cx, cy = polygon_centroid(verts)
        # minAreaRect 方向
        rect = cv2.minAreaRect(poly.reshape(-1, 1, 2).astype(np.float32))
        _, (rw, rh), angle = rect
        if rw < rh:
            angle += 90

        results.append({
            "id": i,
            "vertices_px": [[round(v[0], 1), round(v[1], 1)] for v in verts],
            "num_edges": len(verts),
            "centroid_px": [round(cx, 1), round(cy, 1)],
            "orientation_deg": round(angle, 1),
            "area_px": round(cv2.contourArea(poly.astype(np.int32).reshape(-1, 1, 2)), 1),
        })

    return results, mask


# =========================================================================
# 评估
# =========================================================================

def evaluate(detections, ground_truth, match_max_dist_px=150):
    """
    将检测结果与 ground_truth 对比。

    :return: dict 含各项指标
    """
    gt_frags = ground_truth["fragments"]
    scale = ground_truth.get("scale_px_per_mm", 5.0)

    # 重心匹配
    matches = []
    unmatched_gt = list(range(len(gt_frags)))

    for di, det in enumerate(detections):
        best_gi, best_dist = -1, float('inf')
        for gi in unmatched_gt:
            gt_cx = gt_frags[gi]["vertices_norm"][0][0] * ground_truth["a4_size_mm"][0] * scale
            gt_cy = gt_frags[gi]["vertices_norm"][0][1] * ground_truth["a4_size_mm"][1] * scale
            # 用 ground truth 中碎片所有顶点的重心
            gt_verts_norm = gt_frags[gi]["vertices_norm"]
            gt_verts_px = [(v[0] * ground_truth["a4_size_mm"][0] * scale,
                            v[1] * ground_truth["a4_size_mm"][1] * scale)
                           for v in gt_verts_norm]
            gt_cx = sum(v[0] for v in gt_verts_px) / len(gt_verts_px)
            gt_cy = sum(v[1] for v in gt_verts_px) / len(gt_verts_px)

            dist = math.hypot(det["centroid_px"][0] - gt_cx, det["centroid_px"][1] - gt_cy)
            if dist < best_dist:
                best_dist = dist
                best_gi = gi

        if best_dist < match_max_dist_px:
            matches.append((di, best_gi, best_dist))
            if best_gi in unmatched_gt:
                unmatched_gt.remove(best_gi)

    # 指标
    vertex_errors, centroid_errors = [], []
    for di, gi, _ in matches:
        det = detections[di]
        gt = gt_frags[gi]
        gt_verts_norm = gt["vertices_norm"]
        scale_w = ground_truth["a4_size_mm"][0] * scale
        scale_h = ground_truth["a4_size_mm"][1] * scale
        gt_verts_px = [(v[0] * scale_w, v[1] * scale_h) for v in gt_verts_norm]

        # 重心误差（mm）
        gt_cx_px = sum(v[0] for v in gt_verts_px) / len(gt_verts_px)
        gt_cy_px = sum(v[1] for v in gt_verts_px) / len(gt_verts_px)
        c_err_mm = math.hypot(det["centroid_px"][0] - gt_cx_px,
                              det["centroid_px"][1] - gt_cy_px) / scale
        centroid_errors.append(c_err_mm)

        # 顶点误差（mm）：最近邻匹配
        det_verts = det["vertices_px"]
        for dv in det_verts:
            min_d = min(math.hypot(dv[0] - gv[0], dv[1] - gv[1]) for gv in gt_verts_px)
            vertex_errors.append(min_d / scale)

    return {
        "num_gt": len(gt_frags),
        "num_detected": len(detections),
        "num_matched": len(matches),
        "num_missed": len(unmatched_gt),
        "detection_rate": len(matches) / len(gt_frags) if gt_frags else 0,
        "vertex_error_mm": {
            "mean": round(np.mean(vertex_errors), 3) if vertex_errors else None,
            "max": round(np.max(vertex_errors), 3) if vertex_errors else None,
        },
        "centroid_error_mm": {
            "mean": round(np.mean(centroid_errors), 3) if centroid_errors else None,
            "max": round(np.max(centroid_errors), 3) if centroid_errors else None,
        },
    }


# =========================================================================
# 可视化
# =========================================================================

def draw_detections(img, detections, out_path):
    """在图像上绘制检测到的碎片多边形。"""
    vis = img.copy()
    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for det in detections:
        c = colors[det["id"] % len(colors)]
        pts = np.array([(int(v[0]), int(v[1])) for v in det["vertices_px"]], np.int32)
        cv2.polylines(vis, [pts.reshape((-1, 1, 2))], True, c, 2)
        cx, cy = int(det["centroid_px"][0]), int(det["centroid_px"][1])
        cv2.circle(vis, (cx, cy), 4, c, -1)
        cv2.putText(vis, f"F{det['id']}", (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    cv2.imwrite(out_path, vis)
    return vis


# =========================================================================
# 批量测试
# =========================================================================

def run_batch(test_dir, output_dir=None):
    """对 test_data 下所有图像跑检测并评估。"""
    img_dir = os.path.join(test_dir, "images")
    gt_dir = os.path.join(test_dir, "ground_truth")

    if not os.path.isdir(img_dir):
        print(f"图像目录不存在: {img_dir}")
        return

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))])

    if output_dir is None:
        output_dir = os.path.join(test_dir, "detection_results")
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    total_time = 0.0

    print(f"检测 {len(img_files)} 张图像...")
    print("-" * 60)

    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        prefix = os.path.splitext(fname)[0]
        gt_path = os.path.join(gt_dir, f"{prefix}.json")

        t0 = time.time()
        detections, mask = detect(img_path)
        elapsed = time.time() - t0
        total_time += elapsed

        # 评估
        eval_result = None
        if os.path.exists(gt_path):
            with open(gt_path, encoding="utf-8") as f:
                gt = json.load(f)
            eval_result = evaluate(detections, gt)

        # 保存可视化
        img = cv2.imread(img_path)
        vis_path = os.path.join(output_dir, f"{prefix}_detected.png")
        draw_detections(img, detections, vis_path)

        # 输出
        if eval_result:
            all_metrics.append(eval_result)
            status = "OK" if eval_result["num_missed"] == 0 else f"MISS:{eval_result['num_missed']}"
            vmax = eval_result["vertex_error_mm"]["max"]
            cmax = eval_result["centroid_error_mm"]["max"]
            print(f"  {fname}: {eval_result['num_detected']}/{eval_result['num_gt']} "
                  f"vmax={vmax}mm cmax={cmax}mm {elapsed:.3f}s [{status}]")
        else:
            print(f"  {fname}: {len(detections)} fragments {elapsed:.3f}s (no GT)")

    print("-" * 60)
    print(f"总耗时: {total_time:.3f}s (平均 {total_time/len(img_files):.3f}s/张)")

    if all_metrics:
        rates = [m["detection_rate"] for m in all_metrics]
        vmaxs = [m["vertex_error_mm"]["max"] for m in all_metrics if m["vertex_error_mm"]["max"]]
        print(f"检出率: {np.mean(rates)*100:.1f}%  顶点误差 max avg: {np.mean(vmaxs):.3f}mm  通过(100%检出): {sum(1 for r in rates if r >= 1.0)}/{len(rates)}")

    # 保存汇总
    summary = {"detections": all_metrics}
    spath = os.path.join(output_dir, "detection_summary.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {spath}")


# =========================================================================
# CLI
# =========================================================================

def main():
    p = argparse.ArgumentParser(description="拼图碎片识别")
    p.add_argument("--image", type=str, help="单张图像路径")
    p.add_argument("--gt", type=str, help="ground_truth JSON 路径（与 --image 配合）")
    p.add_argument("--test-dir", type=str, help="测试集目录（含 images/ 和 ground_truth/）")
    p.add_argument("--output", type=str, help="输出目录")
    args = p.parse_args()

    if args.image:
        detections, mask = detect(args.image)
        print(f"检测到 {len(detections)} 个碎片")
        for d in detections:
            print(f"  F{d['id']}: {d['num_edges']}边 重心=({d['centroid_px'][0]:.0f},{d['centroid_px'][1]:.0f}) θ={d['orientation_deg']}°")

        if args.gt and os.path.exists(args.gt):
            with open(args.gt, encoding="utf-8") as f:
                gt = json.load(f)
            ev = evaluate(detections, gt)
            print(f"\n评估: 检出={ev['num_detected']}/{ev['num_gt']} "
                  f"顶点误差 max={ev['vertex_error_mm']['max']}mm "
                  f"重心误差 max={ev['centroid_error_mm']['max']}mm")

        # 可视化
        img = cv2.imread(args.image)
        out = args.output or os.path.splitext(args.image)[0] + "_detected.png"
        draw_detections(img, detections, out)
        print(f"可视化: {out}")

    elif args.test_dir:
        run_batch(args.test_dir, args.output)
    else:
        # 默认跑 test_data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default = os.path.join(script_dir, "test_data")
        if os.path.isdir(os.path.join(default, "images")):
            run_batch(default, args.output)
        else:
            print("未找到测试数据。请先运行 dataset_generator.py 生成数据。")


if __name__ == "__main__":
    main()
