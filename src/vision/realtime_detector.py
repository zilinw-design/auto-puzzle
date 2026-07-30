"""
realtime_detector.py — 实时碎片识别（红色A4纸 + 白色碎片）

流水线：
  1. 摄像头自动控制（AE + AWB）
  2. 红色纸面检测（H通道红色 + S通道高饱和）→ 纸面Mask
  3. 红色纸面四角 → 透视矫正
  4. 限定在纸面范围内：V通道OTSU + S低饱和辅助 + Canny融合
  5. 形态学 + 重叠合并 + top-4

用法：
  python realtime_detector.py                        # HTTP 流
  python realtime_detector.py --display               # 本地窗口
"""

import cv2, numpy as np, argparse, time, subprocess, os

# 鼠标 mm 坐标（全局变量，供 draw_overlay 和 display 函数共享）
_mouse_px = np.array([0.0, 0.0])
_mouse_H = None

def _mouse_cb(event, x, y, flags, param):
    global _mouse_px
    _mouse_px = np.array([float(x), float(y)])

# =========================================================================
# 参数
# =========================================================================
MIN_AREA = 200
MAX_AREA = 99999
BORDER_CROP = 5
EPSILON = 0.008
TARGET_BRIGHTNESS = 120
BRIGHTNESS_SAFE_MIN = 80
BRIGHTNESS_SAFE_MAX = 160
OTSU_FACTOR = 0.75
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# =========================================================================
# 摄像头自动控制
# =========================================================================

def camera_auto_setup(device="/dev/video0"):
    try:
        subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl=auto_exposure=3"],
                       capture_output=True, timeout=3)
        subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl=white_balance_automatic=1"],
                       capture_output=True, timeout=3)
        print("[Camera] AE + AWB = ON")
    except Exception:
        pass


# =========================================================================
# 红色纸面定位（替换原白色角标方案）
# =========================================================================

_smoothed_corners = None
EMA_ALPHA = 0.4


def _get_red_paper_mask(frame_bgr):
    """
    检测红色纸面区域。返回 Mask（纸内=255, 纸外=0）。
    红色判定：H ∈ [0,10] ∪ [170,180] AND S > 80。
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = cv2.inRange(h, 0, 10) | cv2.inRange(h, 170, 180)
    _, s_thresh = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, s_thresh)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    paper = max(contours, key=cv2.contourArea)
    paper_mask = np.zeros_like(mask)
    cv2.drawContours(paper_mask, [paper], -1, 255, -1)

    # 向内轻微腐蚀，排除纸边缘反光过渡带
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    paper_mask_tight = cv2.erode(paper_mask, k3, iterations=1)

    return paper_mask, paper_mask_tight


def _extract_red_paper_corners(frame_bgr):
    """
    从红色纸面提取四角（用于透视矫正）。
    返回: (4×2) np.float32 或 None
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = cv2.inRange(h, 0, 10) | cv2.inRange(h, 170, 180)
    _, s_thresh = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, s_thresh)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    paper = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(paper)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    # 四边形逼近失败时的兜底
    rect = cv2.minAreaRect(paper)
    return cv2.boxPoints(rect).astype(np.float32)


def _quality_check(pts, frame_shape):
    """验证四点是否构成合理的 A4 纸四边形。"""
    if pts is None or len(pts) != 4:
        return False
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                        pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)
    w_top = np.linalg.norm(ordered[1] - ordered[0])
    w_bot = np.linalg.norm(ordered[2] - ordered[3])
    h_left = np.linalg.norm(ordered[3] - ordered[0])
    h_right = np.linalg.norm(ordered[2] - ordered[1])
    ratio = ((w_top + w_bot) / 2) / ((h_left + h_right) / 2) if (h_left + h_right) > 0 else 0
    if ratio < 0.5 or ratio > 0.9:
        return False
    area = cv2.contourArea(ordered.astype(np.int32).reshape(-1, 1, 2))
    if area < frame_shape[0] * frame_shape[1] * 0.03:
        return False
    return True


def _do_warp(frame_bgr, pts):
    """四点透视变换：梯形→矩形。"""
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                        pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)
    A4_W, A4_H = 210, 297
    scale = 3.0
    dst = np.array([[0, 0], [A4_W * scale, 0],
                    [A4_W * scale, A4_H * scale], [0, A4_H * scale]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (int(A4_W * scale), int(A4_H * scale)))
    h_warped = warped.shape[0]
    roi = warped[:h_warped // 2, :, :]
    return warped, roi, True


def find_red_paper_roi(frame_bgr):
    """
    红色纸面检测 + 帧间平滑 + 透视矫正。
    返回: (warped, roi, has_warp)
    """
    global _smoothed_corners

    raw_pts = _extract_red_paper_corners(frame_bgr)

    if raw_pts is None:
        if _smoothed_corners is None:
            return frame_bgr, None, False
        raw_pts = _smoothed_corners

    if _smoothed_corners is None:
        _smoothed_corners = raw_pts.astype(np.float32)
    else:
        _smoothed_corners = (EMA_ALPHA * raw_pts +
                             (1 - EMA_ALPHA) * _smoothed_corners).astype(np.float32)

    if not _quality_check(_smoothed_corners, frame_bgr.shape):
        return frame_bgr, None, False

    return _do_warp(frame_bgr, _smoothed_corners)


# =========================================================================
# Gamma 亮度自适应（安全区：80-160 不触发）
# =========================================================================

def gamma_correct(img_bgr, roi=None):
    if roi is not None and roi.size > 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

    if BRIGHTNESS_SAFE_MIN <= brightness <= BRIGHTNESS_SAFE_MAX:
        return img_bgr, brightness, 1.0

    gamma = np.log(TARGET_BRIGHTNESS / 255.0) / np.log(brightness / 255.0)
    gamma = max(0.5, min(2.0, gamma))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, lut), brightness, gamma


# =========================================================================
# 检测核心：先限定红色纸面，再在纸内找白色碎片
# =========================================================================

def _polys_from_mask(mask, px_per_mm=3.0, min_area=MIN_AREA, max_area=MAX_AREA):
    """
    自适应核大小：根据纸面像素比例缩放形态学核。
    px_per_mm=3 (近距离): 核 7/11/5, merge_gap=6
    px_per_mm=1 (远距离): 核 3/5/3, merge_gap=2
    保证 Close 最多桥接 3mm 物理间隙（碎片内部裂缝），
    而碎片间 1cm 间距永远不被桥接。
    """
    k_close = max(3, int(px_per_mm * 2.5))   # 主 Close 核
    k_close2 = max(3, int(px_per_mm * 4))     # 大 Close 核
    k_open = max(3, int(px_per_mm * 1.8))     # Open 核
    merge_gap = max(2, int(px_per_mm * 2))    # 合并距离 (~2mm 物理)

    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    kernel_c2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close2, k_close2))
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_c, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_c2, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_o, iterations=1)

    mask[:BORDER_CROP, :] = 0
    mask[-BORDER_CROP:, :] = 0
    mask[:, :BORDER_CROP] = 0
    mask[:, -BORDER_CROP:] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    merged = _merge_overlapping(filtered, overlap_thresh=0.7)
    merged = _merge_close_neighbors(merged, max_gap_px=merge_gap)
    merged.sort(key=cv2.contourArea, reverse=True)
    merged = merged[:4]

    polys = []
    for cnt in merged:
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, EPSILON * peri, True)
        if len(approx) >= 3:
            polys.append(approx.reshape(-1, 2).astype(np.int32))
    return polys


def _merge_overlapping(contours, overlap_thresh=0.5):
    if len(contours) <= 1:
        return contours
    rects = [cv2.boundingRect(c) for c in contours]
    parent = list(range(len(contours)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            x1, y1, w1, h1 = rects[i]
            x2, y2, w2, h2 = rects[j]
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ix * iy
            a1, a2 = w1 * h1, w2 * h2
            if inter > 0 and (inter / a1 > overlap_thresh or inter / a2 > overlap_thresh):
                union(i, j)
    groups = {}
    for i in range(len(contours)):
        root = find(i)
        groups.setdefault(root, []).append(contours[i])
    result = []
    for g in groups.values():
        result.append(g[0] if len(g) == 1 else cv2.convexHull(np.vstack(g)))
    return result


def _merge_close_neighbors(contours, max_gap_px=6):
    if len(contours) <= 1:
        return contours
    n = len(contours)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    for i in range(n):
        for j in range(i + 1, n):
            min_d = cv2.pointPolygonTest(contours[i],
                    (float(cv2.boundingRect(contours[j])[0] + cv2.boundingRect(contours[j])[2] / 2),
                     float(cv2.boundingRect(contours[j])[1] + cv2.boundingRect(contours[j])[3] / 2)),
                    True)
            min_d = min(abs(min_d), _contour_distance(contours[i], contours[j], max_gap_px))
            if min_d < max_gap_px:
                union(i, j)
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(contours[i])
    result = []
    for g in groups.values():
        result.append(g[0] if len(g) == 1 else cv2.convexHull(np.vstack(g)))
    return result


def _contour_distance(c1, c2, max_gap=6):
    r1 = cv2.boundingRect(c1)
    r2 = cv2.boundingRect(c2)
    dx = max(0, max(r1[0], r2[0]) - min(r1[0] + r1[2], r2[0] + r2[2]))
    dy = max(0, max(r1[1], r2[1]) - min(r1[1] + r1[3], r2[1] + r2[3]))
    if dx > max_gap or dy > max_gap:
        return float('inf')
    min_d = float('inf')
    for p1 in c1.reshape(-1, 2).astype(np.float32):
        min_d = min(min_d, abs(cv2.pointPolygonTest(c2, tuple(p1), True)))
    return min_d


def detect_fused(frame_bgr):
    """
    红色纸面 + 白色碎片：
    1. 先检测红色纸面 Mask（排除木板）
    2. 只在纸面范围内做 V 通道 OTSU + S 低饱和检测
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # ---- 第一步：红色纸面 Mask ----
    mask_red = cv2.inRange(h, 0, 10) | cv2.inRange(h, 170, 180)
    _, s_high = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY)
    paper_mask = cv2.bitwise_and(mask_red, s_high)

    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, k9, iterations=3)
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, k9, iterations=1)

    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], 0

    paper_mask = np.zeros_like(paper_mask)
    cv2.drawContours(paper_mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    # 向内腐蚀，排除纸边缘反光过渡带
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    paper_mask = cv2.erode(paper_mask, k3, iterations=1)

    # 计算纸面像素比例（A4 宽 210mm）
    paper_width_px = float(np.max(np.sum(paper_mask > 0, axis=0)))
    px_per_mm = max(0.8, paper_width_px / 210.0)

    # ---- 第二步：只在纸面范围内做 V 通道 OTSU ----
    v_eq = CLAHE.apply(v)
    v_inside = v_eq[paper_mask > 0]
    if len(v_inside) < 1000:
        return [], 0

    v_thresh, _ = cv2.threshold(v_inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_thresh = v_thresh * OTSU_FACTOR
    _, mask_v = cv2.threshold(v_eq, v_thresh, 255, cv2.THRESH_BINARY)

    # ---- 第三步：S 通道低饱和 = 白色碎片 ----
    s_eq = CLAHE.apply(s)
    _, mask_s = cv2.threshold(s_eq, 40, 255, cv2.THRESH_BINARY_INV)

    # ---- 第四步：融合 + 纸面限定（木板彻底排除）----
    mask = cv2.bitwise_and(mask_v, mask_s)
    mask = cv2.bitwise_and(mask, paper_mask)

    # ---- 第五步：Canny 边缘补强（同样限定纸面内）----
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges[paper_mask == 0] = 0
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    core_dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k7, iterations=2)
    edges_filtered = cv2.bitwise_and(edges, core_dilated)
    edges_closed = cv2.morphologyEx(edges_filtered, cv2.MORPH_CLOSE, k7, iterations=2)
    mask = cv2.bitwise_or(mask, edges_closed)

    polys = _polys_from_mask(mask, px_per_mm)
    return polys, v_thresh, paper_mask, px_per_mm


def detect_robust(frame_bgr):
    """完整检测管线：红纸定位 → 碎片检测 → 像素转mm坐标。
    返回: (polys_px, paper_H, paper_corners_px, mode, brightness, gamma, warped, has_warp)
      polys_px: 碎片多边形(像素坐标)
      paper_H: 3×3单应性矩阵(像素→A4纸面mm)
      paper_corners_px: 纸面四角(像素)
    """
    # 红纸四角定位
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_red = cv2.inRange(h, 0, 10) | cv2.inRange(h, 170, 180)
    _, s_high = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY)
    paper_raw = cv2.bitwise_and(mask_red, s_high)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    paper_raw = cv2.morphologyEx(paper_raw, cv2.MORPH_CLOSE, k9, iterations=3)
    paper_raw = cv2.morphologyEx(paper_raw, cv2.MORPH_OPEN, k9, iterations=1)

    cnts, _ = cv2.findContours(paper_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_H = None
    paper_corners_px = None

    if cnts:
        paper_cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(paper_cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) == 4:
            # 排序四角: TL, TR, BR, BL
            pts = approx.reshape(4, 2).astype(np.float32)
            s_pts = pts.sum(axis=1); d_pts = np.diff(pts, axis=1)
            ordered = np.array([
                pts[np.argmin(s_pts)],      # TL
                pts[np.argmin(d_pts)],      # TR
                pts[np.argmax(s_pts)],      # BR
                pts[np.argmax(d_pts)],      # BL
            ], dtype=np.float32)
            paper_corners_px = ordered
            # 自动判断纸面方向：长边=297mm，短边=210mm
            w_px = np.linalg.norm(ordered[1] - ordered[0])
            h_px = np.linalg.norm(ordered[3] - ordered[0])
            if w_px > h_px:
                # 横向: X=297mm(长), Y=210mm(短)
                dst = np.array([[0, 0], [297, 0], [297, 210], [0, 210]], dtype=np.float32)
            else:
                dst = np.array([[0, 0], [210, 0], [210, 297], [0, 297]], dtype=np.float32)
            paper_H = cv2.getPerspectiveTransform(ordered, dst)

    # Gamma + 检测
    corrected, brightness, gamma = gamma_correct(frame_bgr, None)
    polys_px, v_thresh, _, _ = detect_fused(corrected)

    has_warp = paper_H is not None
    mode = f"Red+V+S({v_thresh:.0f})" if has_warp else "Raw"

    # 转换碎片 → mm 坐标
    polys_mm = []
    if paper_H is not None and len(polys_px) > 0:
        for poly_px in polys_px:
            pts_h = np.c_[poly_px.astype(np.float32), np.ones(len(poly_px))]
            mm_h = (paper_H @ pts_h.T).T
            pts_mm = mm_h[:, :2] / mm_h[:, 2:3]
            polys_mm.append(pts_mm)

    return polys_px, polys_mm, paper_H, paper_corners_px, mode, brightness, gamma, corrected, has_warp


# =========================================================================
# 绘制
# =========================================================================

COLORS = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

def draw_overlay(warped, polygons_px, polys_mm, fps, mode, brightness, gamma, has_warp, paper_corners_px=None, paper_H=None):
    global _mouse_px, _mouse_H
    _mouse_H = paper_H
    vis = warped.copy()
    overlay = warped.copy()
    for i, poly in enumerate(polygons_px):
        cv2.fillPoly(overlay, [poly.reshape((-1, 1, 2))], COLORS[i % 4])
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    # 纸面四角标注
    if paper_corners_px is not None and paper_H is not None:
        w_px = np.linalg.norm(paper_corners_px[1] - paper_corners_px[0])
        h_px = np.linalg.norm(paper_corners_px[3] - paper_corners_px[0])
        pw, ph = (297, 210) if w_px > h_px else (210, 297)
        names = [f"TL(0,0)", f"TR({pw},0)", f"BR({pw},{ph})", f"BL(0,{ph})"]
        for i, pt in enumerate(paper_corners_px.astype(np.int32)):
            cv2.circle(vis, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(vis, names[i], (pt[0] + 10, pt[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.polylines(vis, [paper_corners_px.reshape((-1, 1, 2)).astype(np.int32)], True, (255, 100, 0), 2)
        cv2.polylines(vis, [paper_corners_px.reshape((-1, 1, 2)).astype(np.int32)], True, (255, 100, 0), 2)


    for i, poly in enumerate(polygons_px):
        c = COLORS[i % 4]
        pts = poly.reshape(-1, 2)
        cv2.polylines(vis, [pts.reshape((-1, 1, 2))], True, c, 1)
        hull = cv2.convexHull(pts)
        cv2.polylines(vis, [hull], True, c, 2)
        rect = cv2.minAreaRect(pts.astype(np.float32))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(vis, [box.reshape((-1, 1, 2))], True, c, 1)
        M = cv2.moments(poly)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.circle(vis, (cx, cy), 4, c, -1)
            if i < len(polys_mm):
                mm_c = polys_mm[i].mean(axis=0)
                label = f"F{i}({mm_c[0]:.0f},{mm_c[1]:.0f})mm"
            else:
                label = f"F{i}"
            cv2.putText(vis, label, (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)

    h, w = vis.shape[:2]
    bar_h = 42
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8) + 40
    warp_str = 'H' if has_warp else 'Raw'
    bri_str = f"Bri:{brightness:.0f}"
    if abs(gamma - 1) > 0.01:
        bri_str += f" G:{gamma:.2f}"

    # 鼠标 mm 坐标
    mm_str = ""
    if _mouse_H is not None and paper_corners_px is not None:
        pt_h = np.array([_mouse_px[0], _mouse_px[1], 1.0])
        mm = _mouse_H @ pt_h
        mx, my = mm[0]/mm[2], mm[1]/mm[2]
        w_px2 = np.linalg.norm(paper_corners_px[1] - paper_corners_px[0])
        h_px2 = np.linalg.norm(paper_corners_px[3] - paper_corners_px[0])
        pw2, ph2 = (297, 210) if w_px2 > h_px2 else (210, 297)
        if 0 <= mx <= pw2 and 0 <= my <= ph2:
            mm_str = f" | 鼠标:({mx:.0f},{my:.0f})mm"
        else:
            mm_str = " | 鼠标:纸面外"

    cv2.putText(bar, f"Frags:{len(polygons_px)} | {warp_str} | {mode} | {bri_str} | {fps:.1f}fps{mm_str}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.vstack([bar, vis])


# =========================================================================
# 本地窗口
# =========================================================================

def run_display(cap, width, height):
    cv2.namedWindow("Detector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Detector", _mouse_cb)
    cv2.resizeWindow("Detector", min(1000, width), min(700, height + 40))
    print("C=捕获校准点 | P=打印校准表 | Q=退出")
    # 校准点收集: [(algo_x, algo_y, real_x, real_y), ...]
    calib_pts = []
    fps_t0, fps_cnt, fps_val = time.time(), 0, 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame.shape[1] != width: frame = cv2.resize(frame, (width, height))
        polys_px, polys_mm, H, corners, mode, bri, gam, warped, has_warp = detect_robust(frame)
        vis = draw_overlay(warped, polys_px, polys_mm, fps_val, mode, bri, gam, has_warp, corners, H)

        # 在画面上标记已采集的校准点
        if H is not None and len(calib_pts) > 0:
            H_inv = np.linalg.inv(H)
            for i, (ax, ay, _, _) in enumerate(calib_pts):
                pt_h = np.array([ax, ay, 1.0])
                px = (H_inv @ pt_h)
                px = tuple((px[:2] / px[2]).astype(int))
                cv2.circle(vis, px, 5, (255, 0, 255), -1)
                cv2.putText(vis, f"P{i+1}", (px[0] + 8, px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)

        cv2.imshow("Detector", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            # 捕获当前鼠标位置为校准点
            if H is not None:
                pt_h = np.array([_mouse_px[0], _mouse_px[1], 1.0])
                mm = H @ pt_h
                mx, my = mm[0]/mm[2], mm[1]/mm[2]
                if 0 <= mx <= 210 and 0 <= my <= 297:
                    calib_pts.append((mx, my, 0, 0))
                    print(f"  捕获 P{len(calib_pts)}: 算法=({mx:.1f}, {my:.1f})mm  ← 请用尺子量此处实际坐标")
                else:
                    print("  鼠标不在纸面范围内")

        elif key == ord('p'):
            # 打印完整校准表
            n = len(calib_pts)
            if n == 0 and len(polys_mm) >= 1:
                # 无校准点时，打印碎片检测结果
                print(f"\n--- 碎片检测 ({len(polys_mm)}块) ---")
                for i, mm in enumerate(polys_mm):
                    c = mm.mean(axis=0)
                    print(f"  F{i}: 算法=({c[0]:.1f}, {c[1]:.1f})mm")
                for i in range(len(polys_mm)):
                    for j in range(i+1, len(polys_mm)):
                        d = np.linalg.norm(polys_mm[i].mean(0) - polys_mm[j].mean(0))
                        print(f"  F{i}↔F{j}: {d:.1f}mm")
                print()
            elif n > 0:
                print(f"\n{'='*65}")
                print(f"  校准数据 ({n}个点)")
                print(f"  {'点':<6}{'算法X':>8}{'算法Y':>8}{'实测X':>8}{'实测Y':>8}{'比值X':>8}{'比值Y':>8}")
                print(f"  {'-'*55}")
                ratios_x, ratios_y = [], []
                for i, (ax, ay, rx, ry) in enumerate(calib_pts):
                    if rx > 0 and ry > 0:
                        rx_v, ry_v = rx / ax, ry / ay
                        ratios_x.append(rx_v); ratios_y.append(ry_v)
                    else:
                        rx_v, ry_v = 0, 0
                    print(f"  P{i+1:<5}{ax:>8.1f}{ay:>8.1f}{rx:>8.1f}{ry:>8.1f}{rx_v:>8.3f}{ry_v:>8.3f}")
                if ratios_x:
                    print(f"  {'-'*55}")
                    print(f"  均值:{'':>14}{np.mean(ratios_x):>8.3f}{np.mean(ratios_y):>8.3f}")
                print(f"\n  复制下面填入实测值后重新校准:")
                print(f"  calib_data = [")
                for i, (ax, ay, rx, ry) in enumerate(calib_pts):
                    print(f"    ({ax:.1f}, {ay:.1f}, 0, 0),  # P{i+1}: 实测=(?, ?)mm")
                print(f"  ]")
                print(f"{'='*65}\n")

        elif key == ord('r') and len(calib_pts) > 0:
            calib_pts.pop()
            print(f"  删除最后一个点, 剩余{len(calib_pts)}个")

        fps_cnt += 1
        if fps_cnt % 30 == 0:
            fps_val = 30 / (time.time() - fps_t0); fps_t0 = time.time()
    cv2.destroyAllWindows()

    # 退出时打印最终数据
    if len(calib_pts) > 0:
        print(f"\n退出。共捕获{len(calib_pts)}个校准点。")
        print("填入实测值后更新 PAPER_SCALE 或做非均匀修正。")


# =========================================================================
# HTTP 流
# =========================================================================

def run_http_stream(cap, width, height, port=8080):
    from flask import Flask, Response
    app = Flask(__name__)
    fps_val, fps_t0, fps_cnt = [0.0], [time.time()], [0]

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame.shape[1] != width: frame = cv2.resize(frame, (width, height))
            polys_px, polys_mm, H, corners, mode, bri, gam, warped, has_warp = detect_robust(frame)
            vis = draw_overlay(warped, polys_px, polys_mm, fps_val[0], mode, bri, gam, has_warp, corners, H)
            _, jpg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   jpg.tobytes() + b'\r\n')
            fps_cnt[0] += 1
            if fps_cnt[0] % 30 == 0:
                fps_val[0] = 30 / (time.time() - fps_t0[0]); fps_t0[0] = time.time()

    @app.route('/')
    def index():
        return f"""<html><head><title>Puzzle Detector</title></head>
        <body style="background:#111;text-align:center;font-family:Arial">
        <h2 style="color:#fff">Fragment Detector — Red Paper + White Fragments</h2>
        <img src="/stream" style="max-width:100%">
        <p style="color:#666">{width}x{height} | :{port}</p></body></html>"""

    @app.route('/stream')
    def stream():
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    print(f"\n{'='*50}")
    print(f"  HTTP: http://<IP>:{port}    {width}x{height}")
    print(f"  红色纸面+白碎片 | H通道红色定位 | V通道OTSU | Gamma安全区80-160")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, threaded=True)


# =========================================================================
# 主入口
# =========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--display", action="store_true")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    camera_auto_setup(f"/dev/video{args.device}")

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"[ERROR] /dev/video{args.device}"); return

    print(f"摄像头: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

    if args.display:
        run_display(cap, args.width, args.height)
    else:
        run_http_stream(cap, args.width, args.height, args.port)
    cap.release()


if __name__ == "__main__":
    main()
