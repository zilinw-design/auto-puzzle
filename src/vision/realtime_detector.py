"""
realtime_detector.py — 实时碎片识别（ROI光照鲁棒 + 透视矫正）

流水线：
  1. 摄像头自动控制（AE + AWB）
  2. 定位 A4 纸 ROI → 透视矫正
  3. ROI 亮度 → Gamma 自适应校正
  4. CLAHE 局部增强
  5. 双模式检测：正常亮度→HSV / 低亮度→边缘融合

用法：
  python realtime_detector.py                        # HTTP 流
  python realtime_detector.py --display               # 本地窗口
"""

import cv2, numpy as np, argparse, time, subprocess, os


# =========================================================================
# 参数
# =========================================================================
YELLOW_LOW  = np.array([18, 50, 70], dtype=np.uint8)
YELLOW_HIGH = np.array([43, 255, 255], dtype=np.uint8)
MIN_AREA = 500
EPSILON = 0.008
TARGET_BRIGHTNESS = 120   # 目标 ROI 平均亮度
BRIGHTNESS_TOL = 30        # 正常亮度范围 ±30
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
# ROI 定位 + 透视矫正
# =========================================================================

def find_a4_roi(frame_bgr):
    """
    在图像中找到 A4 纸四角，返回透视校正后的图像和 ROI 区域。
    找不到则返回原图和全图 ROI。
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame_bgr, None, False

    # 找最大四边形
    best = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)
    if len(approx) != 4:
        return frame_bgr, None, False

    # 面积太小 → 不是纸
    if cv2.contourArea(approx) < frame_bgr.shape[0] * frame_bgr.shape[1] * 0.15:
        return frame_bgr, None, False

    # 排序：左上 右上 右下 左下
    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                        pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)

    A4_W, A4_H = 210, 297
    scale = 3.0
    dst = np.array([[0, 0], [A4_W * scale, 0],
                    [A4_W * scale, A4_H * scale], [0, A4_H * scale]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (int(A4_W * scale), int(A4_H * scale)))

    # ROI：纸面上半部分（碎片区域）
    h_warped = warped.shape[0]
    roi = warped[:h_warped // 2, :, :]
    return warped, roi, True


# =========================================================================
# Gamma 亮度自适应
# =========================================================================

def gamma_correct(img_bgr, roi=None):
    """
    根据 ROI 亮度自动 Gamma 校正。
    - 太暗 → Gamma<1 提亮
    - 太亮 → Gamma>1 压暗
    - 正常 → 不处理
    """
    if roi is not None and roi.size > 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

    if abs(brightness - TARGET_BRIGHTNESS) < BRIGHTNESS_TOL:
        return img_bgr, brightness, 1.0  # 正常，不调

    gamma = np.log(TARGET_BRIGHTNESS / 255.0) / np.log(brightness / 255.0)
    gamma = max(0.5, min(2.0, gamma))  # 限制范围

    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    corrected = cv2.LUT(img_bgr, lut)
    return corrected, brightness, gamma


# =========================================================================
# 检测层
# =========================================================================

def _mask_to_polygons(mask, min_area=MIN_AREA):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPSILON * peri, True)
        if len(approx) >= 3:
            polys.append(approx.reshape(-1, 2).astype(np.int32))
    polys.sort(key=lambda p: cv2.contourArea(p.reshape(-1, 1, 2)), reverse=True)
    return polys


def detect_hsv(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # CLAHE 增强 V 通道
    h, s, v = cv2.split(hsv)
    v = CLAHE.apply(v)
    hsv = cv2.merge([h, s, v])
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    return _mask_to_polygons(mask)


def detect_edges(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    return _mask_to_polygons(closed, min_area=800)


# =========================================================================
# 双模式主检测
# =========================================================================

def detect_robust(frame_bgr):
    """
    1. ROI 定位 + 透视矫正
    2. Gamma 亮度自适应
    3. 双模式：正常→HSV，偏暗→边缘融合
    """
    warped, roi, has_warp = find_a4_roi(frame_bgr)

    # Gamma 校正
    corrected, brightness, gamma = gamma_correct(warped, roi)

    # 正常亮度 → HSV
    if abs(brightness - TARGET_BRIGHTNESS) < BRIGHTNESS_TOL * 1.5:
        polys = detect_hsv(corrected)
        mode = "HSV"
    else:
        # 偏暗或偏亮 → HSV + 边缘融合
        hsv_polys = detect_hsv(corrected)
        edge_polys = detect_edges(corrected)
        polys = hsv_polys if len(hsv_polys) >= len(edge_polys) else edge_polys
        mode = "HSV+Edge"

    if len(polys) < 4 and mode == "HSV":
        edge_polys = detect_edges(corrected)
        polys = edge_polys if len(edge_polys) > len(polys) else polys
        mode = "HSV→Edge"

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
        cv2.polylines(vis, [poly.reshape((-1, 1, 2))], True, c, 2)
        M = cv2.moments(poly)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.circle(vis, (cx, cy), 4, c, -1)
            cv2.putText(vis, f"F{i}", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)

    h, w = vis.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8) + 40
    warp_str = 'Warp' if has_warp else 'Raw'
    bri_str = f"Bri:{brightness:.0f}" + (f" G:{gamma:.2f}" if abs(gamma - 1) > 0.01 else "")
    cv2.putText(bar, f"Frags:{len(polygons)} | {warp_str} | {mode} | {bri_str} | {fps:.1f}fps",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.vstack([bar, vis])


# =========================================================================
# 本地窗口
# =========================================================================

def run_display(cap, width, height):
    cv2.namedWindow("Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detector", width, height + 40)
    print(f"[Display] {width}x{height}  按 Q 退出")
    fps_t0, fps_cnt, fps_val = time.time(), 0, 0.0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
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
            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
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
        <h2 style="color:#fff">Fragment Detector</h2>
        <img src="/stream" style="max-width:100%">
        <p style="color:#666">{width}x{height} | :{port}</p></body></html>"""

    @app.route('/stream')
    def stream():
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    print(f"\n{'='*50}")
    print(f"  HTTP: http://<IP>:{port}    {width}x{height}")
    print(f"  透视矫正 + Gamma自适应 + CLAHE + 双模式检测")
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
