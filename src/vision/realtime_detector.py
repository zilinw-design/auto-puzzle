"""
realtime_detector.py — 实时碎片识别（深色底板 + 白色碎片，V通道OTSU）

流水线：
  1. 摄像头自动控制（AE + AWB）
  2. A4纸四角白标定位 → 透视矫正
  3. ROI亮度 → Gamma安全区 (80-160不触发)
  4. CLAHE V通道增强
  5. OTSU自动阈值 → V通道Mask
  6. Canny边缘融合
  7. Close→Open→Dilate + convexHull + minAreaRect

用法：
  python realtime_detector.py                        # HTTP 流
  python realtime_detector.py --display               # 本地窗口
"""

import cv2, numpy as np, argparse, time, subprocess, os


# =========================================================================
# 参数
# =========================================================================
MIN_AREA = 800     # 物理面积下限 (~2cm²碎片 @3px/mm)
MAX_AREA = 5000    # 物理面积上限 (~14cm²碎片)，排除整张纸误检
BORDER_CROP = 10   # 透视矫正后向内裁剪 px，消除边界泄露
EPSILON = 0.008
TARGET_BRIGHTNESS = 120
BRIGHTNESS_SAFE_MIN = 80
BRIGHTNESS_SAFE_MAX = 160
OTSU_FACTOR = 0.85
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
# A4 纸定位（深色纸+白色角标） + 透视矫正
# =========================================================================

def find_a4_roi(frame_bgr):
    """
    深色 A4 纸：四个角贴白色标记点。
    Canny 梯度 + V 通道亮斑双条件检测。
    找不到则退回原图。
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 方法1：Canny 边缘 → 最大四边形
    edges = cv2.Canny(blur, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        best = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > frame_bgr.shape[0] * frame_bgr.shape[1] * 0.1:
            return _do_warp(frame_bgr, approx.reshape(4, 2).astype(np.float32))

    # 方法2：V通道亮斑 → 白色角标 → 取四角的外接四边形
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    _, bright = cv2.threshold(v, 180, 255, cv2.THRESH_BINARY)
    contours_b, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_b:
        all_pts = np.vstack([c.reshape(-1, 2) for c in contours_b])
        if len(all_pts) >= 4:
            rect = cv2.minAreaRect(all_pts.astype(np.float32))
            box = cv2.boxPoints(rect).astype(np.float32)
            if cv2.contourArea(box.astype(np.int32)) > frame_bgr.shape[0] * frame_bgr.shape[1] * 0.1:
                return _do_warp(frame_bgr, box)

    return frame_bgr, None, False


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

    # 安全区：不触发 Gamma
    if BRIGHTNESS_SAFE_MIN <= brightness <= BRIGHTNESS_SAFE_MAX:
        return img_bgr, brightness, 1.0

    gamma = np.log(TARGET_BRIGHTNESS / 255.0) / np.log(brightness / 255.0)
    gamma = max(0.5, min(2.0, gamma))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, lut), brightness, gamma


# =========================================================================
# 检测核心：V通道 + OTSU + Canny融合
# =========================================================================

def _polys_from_mask(mask, min_area=MIN_AREA, max_area=MAX_AREA):
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 1. Close(7x7,3) + Close(15x15,1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k15, iterations=1)
    # 2. Open(5x5,1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5, iterations=1)

    # 3. border crop
    mask[:BORDER_CROP, :] = 0
    mask[-BORDER_CROP:, :] = 0
    mask[:, :BORDER_CROP] = 0
    mask[:, -BORDER_CROP:] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    merged = _merge_overlapping(filtered)
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


def detect_fused(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    v_eq = CLAHE.apply(gray)
    otsu_thresh, _ = cv2.threshold(v_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = otsu_thresh * OTSU_FACTOR
    _, mask_v = cv2.threshold(v_eq, thresh, 255, cv2.THRESH_BINARY)
    mid_y = mask_v.shape[0] // 2
    mask_v[mid_y - 8 : mid_y + 8, :] = 0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    core_dilated = cv2.morphologyEx(mask_v, cv2.MORPH_DILATE, k7, iterations=2)
    edges_filtered = cv2.bitwise_and(edges, core_dilated)
    edges_closed = cv2.morphologyEx(edges_filtered, cv2.MORPH_CLOSE, k7, iterations=2)
    mask = cv2.bitwise_or(mask_v, edges_closed)
    return _polys_from_mask(mask), otsu_thresh

def detect_robust(frame_bgr):
    warped, roi, has_warp = find_a4_roi(frame_bgr)
    corrected, brightness, gamma = gamma_correct(warped, roi)

    polys, otsu_thresh = detect_fused(corrected)
    mode = f"OTSU({otsu_thresh:.0f})" if has_warp else f"OTSU({otsu_thresh:.0f}) Raw"

    return polys, mode, brightness, gamma, warped, has_warp


# =========================================================================
# 绘制
# =========================================================================

COLORS = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

def draw_overlay(warped, polygons, fps, mode, brightness, gamma, has_warp):
    vis = warped.copy()
    overlay = warped.copy()
    for i, poly in enumerate(polygons):
        cv2.fillPoly(overlay, [poly.reshape((-1, 1, 2))], COLORS[i % 4])
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
    for i, poly in enumerate(polygons):
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
            cv2.putText(vis, f"F{i}", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)

    h, w = vis.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8) + 40
    warp_str = 'Warp' if has_warp else 'Raw'
    bri_str = f"Bri:{brightness:.0f}"
    if abs(gamma - 1) > 0.01:
        bri_str += f" G:{gamma:.2f}"
    cv2.putText(bar, f"Frags:{len(polygons)} | {warp_str} | {mode} | {bri_str} | {fps:.1f}fps",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.vstack([bar, vis])


# =========================================================================
# 本地窗口
# =========================================================================

def run_display(cap, width, height):
    cv2.namedWindow("Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detector", width, height + 40)
    print(f"[Display] {width}x{height}  Q=退出")
    fps_t0, fps_cnt, fps_val = time.time(), 0, 0.0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame.shape[1] != width: frame = cv2.resize(frame, (width, height))
        polys, mode, bri, gam, warped, has_warp = detect_robust(frame)
        vis = draw_overlay(warped, polys, fps_val, mode, bri, gam, has_warp)
        cv2.imshow("Detector", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        fps_cnt += 1
        if fps_cnt % 30 == 0:
            fps_val = 30 / (time.time() - fps_t0); fps_t0 = time.time()
    cv2.destroyAllWindows()


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
            polys, mode, bri, gam, warped, has_warp = detect_robust(frame)
            vis = draw_overlay(warped, polys, fps_val[0], mode, bri, gam, has_warp)
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
        <h2 style="color:#fff">Fragment Detector — V-Channel OTSU</h2>
        <img src="/stream" style="max-width:100%">
        <p style="color:#666">{width}x{height} | :{port}</p></body></html>"""

    @app.route('/stream')
    def stream():
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    print(f"\n{'='*50}")
    print(f"  HTTP: http://<IP>:{port}    {width}x{height}")
    print(f"  深色底板+白碎片 | V通道OTSU | Gamma安全区80-160")
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
