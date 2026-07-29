"""
realtime_detector.py -- 实时碎片识别（融合检测版）

检测管线：
  1. 摄像头 AE + AWB
  2. A4 纸 ROI + 透视矫正
  3. ROI 亮度 Gamma 自适应
  4. HSV 粗定位 -> 闭运算 -> 膨胀 -> Canny 融合 -> convexHull
  5. HTTP 流 / 本地窗口

用法：
  python realtime_detector.py              # HTTP 流
  python realtime_detector.py --display     # 本地窗口
"""

import cv2, numpy as np, argparse, time, subprocess

# ====== 参数 ======
YELLOW_LOW  = np.array([18, 50, 70], dtype=np.uint8)
YELLOW_HIGH = np.array([43, 255, 255], dtype=np.uint8)
MIN_AREA = 500
EPSILON = 0.008
TARGET_BRIGHTNESS = 120
BRIGHTNESS_TOL = 30
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
COLORS = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


# ====== 摄像头 ======
def camera_setup(device="/dev/video0"):
    """
    按 HD USB Camera Manual 最佳配置：
      - 手动曝光 + 固定低快门时间  -> 解锁暗光降帧，稳定 60 FPS
      - 自动白平衡保持色彩稳定
    """
    try:
        subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl=auto_exposure=1"],
                       capture_output=True, timeout=3)     # Manual Mode
        subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl=exposure_time_absolute=78"],
                       capture_output=True, timeout=3)      # 1/128s ~7.8ms，解锁满帧
        subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl=white_balance_automatic=1"],
                       capture_output=True, timeout=3)
        print("[Camera] Manual exposure 1/128s | AWB=ON")
    except Exception:
        pass


# ====== A4 透视矫正 ======
def find_a4_roi(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame_bgr, None, False
    best = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)
    if len(approx) != 4:
        return frame_bgr, None, False
    if cv2.contourArea(approx) < frame_bgr.shape[0] * frame_bgr.shape[1] * 0.15:
        return frame_bgr, None, False
    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
    ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                        pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)
    A4_W, A4_H, scale = 210, 297, 3.0
    dst = np.array([[0, 0], [A4_W * scale, 0],
                    [A4_W * scale, A4_H * scale], [0, A4_H * scale]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (int(A4_W * scale), int(A4_H * scale)))
    h_warped = warped.shape[0]
    roi = warped[:h_warped // 2, :, :]
    return warped, roi, True


# ====== Gamma ======
def gamma_correct(img_bgr, roi=None):
    if roi is not None and roi.size > 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
    if abs(brightness - TARGET_BRIGHTNESS) < BRIGHTNESS_TOL:
        return img_bgr, brightness, 1.0
    gamma = np.log(TARGET_BRIGHTNESS / 255.0) / np.log(brightness / 255.0)
    gamma = max(0.5, min(2.0, gamma))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, lut), brightness, gamma


# ====== 融合检测 ======
def detect_fused(frame_bgr):
    """
    HSV 粗定位 -> 闭运算 -> 膨胀 -> Canny 融合 -> convexHull
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = CLAHE.apply(v)
    hsv = cv2.merge([h, s, v])

    # 1. HSV 严格阈值 -> Mask_core
    mask_core = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)

    # 2. 闭运算 5x5 填孔
    mask_core = cv2.morphologyEx(mask_core, cv2.MORPH_CLOSE, K5, iterations=2)

    # 3. 膨胀 7x7 扩展到边缘
    mask_dilated = cv2.dilate(mask_core, K7, iterations=2)

    # 4. Canny 边缘
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    mask_edge = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, K7, iterations=2)

    # 5. 融合
    mask = cv2.bitwise_or(mask_dilated, mask_edge)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K5, iterations=1)

    # 6. findContours -> convexHull -> approxPolyDP
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, EPSILON * peri, True)
        if len(approx) >= 3:
            polys.append(approx.reshape(-1, 2).astype(np.int32))
    polys.sort(key=lambda p: cv2.contourArea(p.reshape(-1, 1, 2)), reverse=True)
    return polys, mask


# ====== 绘制 ======
def draw_overlay(warped, polygons, fps, brightness, gamma, has_warp):
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
            cv2.putText(vis, f"F{i} {len(poly)}e", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)
    h, w = vis.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8) + 40
    warp_str = "Warp" if has_warp else "Raw"
    bri_str = f"Bri:{brightness:.0f}" + (f" G:{gamma:.2f}" if abs(gamma - 1) > 0.01 else "")
    cv2.putText(bar, f"Frags:{len(polygons)} | {warp_str} | {bri_str} | {fps:.1f}fps",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.vstack([bar, vis])


# ====== 显示 ======
def run_display(cap, width, height):
    cv2.namedWindow("Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detector", width, height + 40)
    print(f"[Display] {width}x{height}  Q=quit")
    fps_t0, fps_cnt, fps_val = time.time(), 0, 0.0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        warped, _, has_warp = find_a4_roi(frame)
        corrected, bri, gam = gamma_correct(warped)
        polys, _ = detect_fused(corrected)
        vis = draw_overlay(warped, polys, fps_val, bri, gam, has_warp)
        cv2.imshow("Detector", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        fps_cnt += 1
        if fps_cnt % 30 == 0:
            fps_val = 30 / (time.time() - fps_t0); fps_t0 = time.time()
    cv2.destroyAllWindows()


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
            warped, _, has_warp = find_a4_roi(frame)
            corrected, bri, gam = gamma_correct(warped)
            polys, _ = detect_fused(corrected)
            vis = draw_overlay(warped, polys, fps_val[0], bri, gam, has_warp)
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
    print(f"  HSV -> Close -> Dilate -> Canny -> convexHull")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, threaded=True)


# ====== 入口 ======
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--display", action="store_true")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    camera_setup(f"/dev/video{args.device}")

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲，消除帧积压延迟
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 60)
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
