"""
realtime_detector.py — 摄像头实时碎片识别

两种显示模式：
  1. HTTP 流（默认）→ 浏览器打开 http://<树莓派IP>:8080 看实时画面
  2. 本地窗口 → --display 参数（需要连接显示器）

用法：
  python realtime_detector.py                        # HTTP 流模式
  python realtime_detector.py --display               # 本地窗口模式
  python realtime_detector.py --width 1280 --height 720  # 低分辨率

依赖（树莓派）：
  pip install opencv-python flask
"""

import cv2
import numpy as np
import argparse
import time
import threading


# =========================================================================
# HSV 碎片检测（与 fragment_detector.py 相同逻辑）
# =========================================================================

YELLOW_LOW = np.array([18, 50, 70], dtype=np.uint8)
YELLOW_HIGH = np.array([43, 255, 255], dtype=np.uint8)
MIN_AREA = 500
EPSILON_RATIO = 0.008


def detect_fragments(frame_bgr):
    """检测黄色碎片，返回多边形列表 (N,2) 像素坐标。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPSILON_RATIO * peri, True)
        if len(approx) >= 3:
            polygons.append(approx.reshape(-1, 2).astype(np.int32))

    polygons.sort(key=lambda p: cv2.contourArea(p.reshape(-1, 1, 2)), reverse=True)
    return polygons, mask


# =========================================================================
# 绘制检测框
# =========================================================================

COLORS = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


def draw_overlay(frame, polygons, fps):
    """在帧上绘制碎片多边形和信息。"""
    vis = frame.copy()

    # 半透明覆盖
    overlay = frame.copy()
    for i, poly in enumerate(polygons):
        color = COLORS[i % 4]
        cv2.fillPoly(overlay, [poly.reshape((-1, 1, 2))], color)
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    # 多边形边框
    for i, poly in enumerate(polygons):
        color = COLORS[i % 4]
        cv2.polylines(vis, [poly.reshape((-1, 1, 2))], True, color, 2)

        # 重心 + 标签
        M = cv2.moments(poly)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(vis, (cx, cy), 4, color, -1)
            cv2.putText(vis, f"F{i}", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 顶点数
            cv2.putText(vis, f"{len(poly)}edges", (cx + 8, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # 顶部状态栏
    h, w = vis.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    cv2.putText(bar, f"Fragments: {len(polygons)}  |  FPS: {fps:.1f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(bar, "Press Q to quit",
                (w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return np.vstack([bar, vis])


# =========================================================================
# 本地窗口模式
# =========================================================================

def run_display(cap, width, height):
    """本地 OpenCV 窗口显示。"""
    cv2.namedWindow("Fragment Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fragment Detector", width, height + 40)

    print(f"[Display] 分辨率 {width}x{height}  按 Q 退出")

    fps_t0 = time.time()
    fps_count = 0
    fps_val = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 可选：缩放到处理分辨率
        if frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        polygons, mask = detect_fragments(frame)
        vis = draw_overlay(frame, polygons, fps_val)

        cv2.imshow("Fragment Detector", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        fps_count += 1
        if fps_count % 30 == 0:
            elapsed = time.time() - fps_t0
            fps_val = 30 / elapsed
            fps_t0 = time.time()

    cv2.destroyAllWindows()


# =========================================================================
# HTTP MJPEG 流模式（浏览器查看）
# =========================================================================

def run_http_stream(cap, width, height, port=8080):
    """HTTP MJPEG 流，浏览器打开 http://<IP>:<port> 查看。"""
    try:
        from flask import Flask, Response
    except ImportError:
        print("[ERROR] 需要 flask: pip install flask")
        return

    app = Flask(__name__)
    fps_val = [0.0]  # mutable reference
    fps_t0 = [time.time()]
    fps_count = [0]

    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))

            polygons, _ = detect_fragments(frame)
            vis = draw_overlay(frame, polygons, fps_val[0])

            _, jpeg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')

            fps_count[0] += 1
            if fps_count[0] % 30 == 0:
                elapsed = time.time() - fps_t0[0]
                fps_val[0] = 30 / elapsed
                fps_t0[0] = time.time()

    @app.route('/')
    def index():
        return f"""<html><head><title>Fragment Detector</title></head>
        <body style="background:#111;text-align:center;font-family:Arial">
        <h2 style="color:#fff">Real-time Fragment Detection</h2>
        <img src="/stream" style="max-width:100%">
        <p style="color:#666">Resolution: {width}x{height} | Port: {port}</p>
        </body></html>"""

    @app.route('/stream')
    def stream():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    print(f"\n{'='*50}")
    print(f"  HTTP 流已启动")
    print(f"  浏览器打开: http://<树莓派IP>:{port}")
    print(f"  分辨率: {width}x{height}")
    print(f"  按 Ctrl+C 停止")
    print(f"{'='*50}\n")

    app.run(host='0.0.0.0', port=port, threaded=True)


# =========================================================================
# 主入口
# =========================================================================

def main():
    p = argparse.ArgumentParser(description="摄像头实时碎片识别")
    p.add_argument("--display", action="store_true", help="本地窗口模式（需显示器）")
    p.add_argument("--port", type=int, default=8080, help="HTTP 流端口（默认 8080）")
    p.add_argument("--width", type=int, default=1280, help="处理宽度")
    p.add_argument("--height", type=int, default=720, help="处理高度")
    p.add_argument("--device", type=int, default=0, help="摄像头设备号（默认 0）")
    args = p.parse_args()

    # 打开摄像头（V4L2 后端，避免 GStreamer 兼容问题）
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 如果 V4L2 + MJPG 失败，尝试默认后端
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 /dev/video{args.device}")
        print("  检查: ls -l /dev/video*")
        print("  尝试: python realtime_detector.py --device 1")
        return

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头已打开: {actual_w:.0f}x{actual_h:.0f} @ {fps:.0f}fps")

    if args.display:
        run_display(cap, args.width, args.height)
    else:
        run_http_stream(cap, args.width, args.height, args.port)

    cap.release()


if __name__ == "__main__":
    main()
