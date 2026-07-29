"""
realtime_detector.py — 摄像头实时碎片识别（鲁棒增强版）

多层检测策略（光照变化自适应）：
  第 1 层：HSV 颜色检测（光照正常时，~15ms）
  第 2 层：Canny 边缘兜底（光照差时，~30ms）
  第 3 层：CLAHE 增强 + HSV + 边缘融合（极端光照）

用法：
  python realtime_detector.py                        # HTTP 流模式
  python realtime_detector.py --display               # 本地窗口模式
  python realtime_detector.py --width 1280 --height 720

依赖：
  pip install opencv-python flask numpy
"""

import cv2, numpy as np, argparse, time, subprocess, os


# =========================================================================
# HSV 阈值（保持严格，用多层兜底代替盲目放宽）
# =========================================================================
YELLOW_LOW  = np.array([18, 50, 70], dtype=np.uint8)
YELLOW_HIGH = np.array([43, 255, 255], dtype=np.uint8)
MIN_AREA = 500
EPSILON_RATIO = 0.008

# CLAHE 增强器（全局单例）
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# =========================================================================
# 摄像头自动控制
# =========================================================================

def camera_auto_setup(device="/dev/video0"):
    """设置摄像头为自动曝光 + 自动白平衡（光照变化自适应）。"""
    try:
        subprocess.run(["v4l2-ctl", "-d", device,
                        "--set-ctrl=auto_exposure=3"],        # Aperture Priority
                       capture_output=True, timeout=3)
        subprocess.run(["v4l2-ctl", "-d", device,
                        "--set-ctrl=white_balance_automatic=1"],
                       capture_output=True, timeout=3)
        print("[Camera] AE=AperturePriority  AWB=ON")
        return True
    except Exception as e:
        print(f"[Camera] v4l2-ctl failed: {e}  (ignored, continuing)")
        return False


# =========================================================================
# 第 1 层：HSV 颜色检测
# =========================================================================

def detect_hsv(frame_bgr):
    """HSV 黄色碎片检测。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    return _mask_to_polygons(mask)


# =========================================================================
# 第 2 层：Canny 边缘兜底
# =========================================================================

def detect_edges(frame_bgr):
    """Canny 边缘检测 → 形态学闭合 → 提取多边形。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    return _mask_to_polygons(closed, min_area=800)


# =========================================================================
# 第 3 层：CLAHE 增强 + HSV + 边缘融合
# =========================================================================

def detect_enhanced(frame_bgr):
    """CLAHE 局部增强 → HSV + 边缘融合。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # CLAHE 增强 V 通道（暗处提亮、亮处压暗）
    v_eq = CLAHE.apply(v)
    hsv_eq = cv2.merge([h, s, v_eq])

    # HSV 检测
    mask_hsv = cv2.inRange(hsv_eq, YELLOW_LOW, YELLOW_HIGH)

    # 边缘检测
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_edge = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 融合：HSV 或 边缘命中即算
    mask = cv2.bitwise_or(mask_hsv, mask_edge)
    return _mask_to_polygons(mask, min_area=600)


# =========================================================================
# 多边形提取（共用）
# =========================================================================

def _mask_to_polygons(mask, min_area=MIN_AREA):
    """二值掩码 → findContours → approxPolyDP → 多边形列表。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPSILON_RATIO * peri, True)
        if len(approx) >= 3:
            polygons.append(approx.reshape(-1, 2).astype(np.int32))

    polygons.sort(key=lambda p: cv2.contourArea(p.reshape(-1, 1, 2)), reverse=True)
    return polygons


# =========================================================================
# 多层融合主入口
# =========================================================================

def detect_robust(frame_bgr):
    """
    多层鲁棒检测：
      第 1 层 HSV → 第 2 层边缘兜底 → 第 3 层 CLAHE 融合
    返回 (polygons, method_name)
    """
    # 第 1 层：HSV
    polygons = detect_hsv(frame_bgr)
    if len(polygons) >= 4:
        return polygons, "HSV"

    # 第 2 层：边缘兜底
    edge_polys = detect_edges(frame_bgr)
    if len(edge_polys) > len(polygons):
        polygons = edge_polys
    if len(polygons) >= 4:
        return polygons, "Canny"

    # 第 3 层：CLAHE 融合
    enhanced = detect_enhanced(frame_bgr)
    if len(enhanced) >= len(polygons):
        polygons = enhanced
    return polygons, "CLAHE" if len(polygons) >= 4 else "Multi"


# =========================================================================
# 绘制
# =========================================================================

COLORS = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


def draw_overlay(frame, polygons, fps, method):
    vis = frame.copy()
    overlay = frame.copy()
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
            cv2.putText(vis, f"F{i} {len(poly)}e", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)

    h, w = vis.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8) + 40
    cv2.putText(bar, f"Frags: {len(polygons)} | {method} | {fps:.1f} fps",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.vstack([bar, vis])


# =========================================================================
# 本地窗口
# =========================================================================

def run_display(cap, width, height):
    cv2.namedWindow("Fragment Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fragment Detector", width, height + 40)
    print(f"[Display] {width}x{height}  按 Q 退出")
    fps_t0, fps_count, fps_val = time.time(), 0, 0.0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        polys, method = detect_robust(frame)
        vis = draw_overlay(frame, polys, fps_val, method)
        cv2.imshow("Fragment Detector", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        fps_count += 1
        if fps_count % 30 == 0:
            fps_val = 30 / (time.time() - fps_t0)
            fps_t0 = time.time()
    cv2.destroyAllWindows()


# =========================================================================
# HTTP 流
# =========================================================================

def run_http_stream(cap, width, height, port=8080):
    from flask import Flask, Response
    app = Flask(__name__)
    fps_val, fps_t0, fps_count = [0.0], [time.time()], [0]

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            polys, method = detect_robust(frame)
            vis = draw_overlay(frame, polys, fps_val[0], method)
            _, jpg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   jpg.tobytes() + b'\r\n')
            fps_count[0] += 1
            if fps_count[0] % 30 == 0:
                fps_val[0] = 30 / (time.time() - fps_t0[0])
                fps_t0[0] = time.time()

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
    print(f"  HTTP: http://<IP>:{port}")
    print(f"  {width}x{height}  |  多层检测: HSV → Canny → CLAHE")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, threaded=True)


# =========================================================================
# 主入口
# =========================================================================

def main():
    p = argparse.ArgumentParser(description="碎片实时识别（鲁棒版）")
    p.add_argument("--display", action="store_true")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    # 摄像头自动控制
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
        print(f"[ERROR] 摄像头 /dev/video{args.device} 无法打开")
        return

    print(f"摄像头: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

    if args.display:
        run_display(cap, args.width, args.height)
    else:
        run_http_stream(cap, args.width, args.height, args.port)
    cap.release()


if __name__ == "__main__":
    main()
