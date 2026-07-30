"""
calibrate_camera.py — 摄像头标定（棋盘格）

用法：
  python calibrate_camera.py                    # 采集+标定
  python calibrate_camera.py --recalibrate      # 仅用已有图片重新标定
  python calibrate_camera.py --help

输出：camera_matrix.npz（内参+畸变，供 aruco_detector 使用）

依赖：pip install opencv-python numpy
"""

import cv2, numpy as np, argparse, os, time, glob


# =========================================================================
# 修改这里：你的棋盘格规格
# =========================================================================
CHESSBOARD_SIZE = (8, 11)      # 内角点 (列, 行)
SQUARE_SIZE_MM = 22.5           # 每格边长 (mm)


def calibrate_from_images(image_dir="calib_images", output="camera_matrix.npz"):
    """
    从已采集的棋盘格图像计算内参。
    """
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points = []  # 3D
    img_points = []  # 2D

    files = sorted(glob.glob(f"{image_dir}/*.jpg")) + sorted(glob.glob(f"{image_dir}/*.png"))
    if not files:
        print(f"[ERROR] {image_dir}/ 下无图像，请先采集")
        return

    gray_shape = None
    for f in files:
        img = cv2.imread(f)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if not ret:
            print(f"  ✗ {os.path.basename(f)} — 未检测到棋盘格")
            continue

        # 亚像素精炼
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_points.append(objp)
        img_points.append(corners)
        print(f"  ✓ {os.path.basename(f)}")

    if len(obj_points) < 10:
        print(f"[ERROR] 只有 {len(obj_points)} 张有效图像，至少需要 10 张")
        return

    print(f"\n计算内参（{len(obj_points)} 张）...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray_shape, None, None)  # 标准5参数模型

    # 重投影误差
    mean_error = 0
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        mean_error += np.mean(np.linalg.norm(img_points[i].reshape(-1, 2) - proj.reshape(-1, 2), axis=1))
    mean_error /= len(obj_points)

    print(f"重投影误差: {mean_error:.4f} px  {'OK' if mean_error < 0.5 else '偏高，建议补拍'}")
    print(f"内参矩阵:\n{mtx}")
    print(f"畸变系数: {dist.ravel()}")

    np.savez(output, mtx=mtx, dist=dist, error=mean_error, size=CHESSBOARD_SIZE, square_mm=SQUARE_SIZE_MM)
    print(f"\n已保存: {output}")


def _open_camera(device=0):
    """跨平台打开摄像头（Windows=DSHOW+MJPG, Linux=V4L2+MJPG）。"""
    import platform
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def collect_images(output_dir="calib_images", device=0):
    """
    实时采集棋盘格图像。按 SPACE 拍照，按 Q 完成。
    引导将棋盘格移至画面 9 个区域，确保畸变标定覆盖全画面。
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = _open_camera(device)
    if not cap.isOpened():
        print(f"[ERROR] 摄像头无法打开 (device={device})")
        return

    # 4 个目标区域（2×2 网格）：按左上→右下顺序
    regions = ["左上", "右上", "左下", "右下"]
    region_shots = {r: 0 for r in regions}
    total = 0
    shots_per_region = 4  # 每区域4张: 正面+左倾+右倾+近/远

    checker_detected = False

    print(f"\n{'='*60}")
    print(f"  棋盘格标定采集: {CHESSBOARD_SIZE[0]}×{CHESSBOARD_SIZE[1]}  格边{SQUARE_SIZE_MM}mm")
    print(f"  操作: SPACE=拍照 | Q=完成标定")
    print(f"  要求: 棋盘格覆盖画面 9 个区域，每个区域 2-3 张(正面+倾斜)")
    print(f"  目标: 共 20-30 张，覆盖全画面各区域")
    print(f"{'='*60}\n")

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]

        # --- 绘制 2×2 引导网格 ---
        vis = frame.copy()
        dx, dy = w // 2, h // 2
        cv2.line(vis, (dx, 0), (dx, h), (80, 80, 80), 1)
        cv2.line(vis, (0, dy), (w, dy), (80, 80, 80), 1)

        # 棋盘格检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret_cb:
            cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, ret_cb)
            # 判断棋盘格在哪个区域
            pts = corners.reshape(-1, 2)
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            col = 0 if cx < dx else 1
            row = 0 if cy < dy else 1
            current_region = regions[row * 2 + col]
            checker_detected = True
        else:
            current_region = None
            checker_detected = False

        # --- 状态栏 ---
        bar_h = 80
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8) + 35
        cv2.putText(bar, f"Total: {total}  [SPACE=capture Q=done]",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(bar, f"Region: {current_region or 'N/A'}  {'DETECTED' if checker_detected else 'NO CHECKER'}",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if checker_detected else (0, 0, 255), 2)

        # 各区域采集进度条
        x0 = 10
        for ri, rname in enumerate(regions):
            filled = region_shots[rname] >= shots_per_region
            color = (0, 200, 0) if filled else (150, 150, 150)
            cv2.putText(bar, f"{rname}:{region_shots[rname]}/{shots_per_region}",
                        (x0, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            x0 += 110

        vis = np.vstack([bar, vis])

        cv2.imshow("Calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and ret_cb:
            path = os.path.join(output_dir, f"calib_{total:03d}.jpg")
            cv2.imwrite(path, frame)
            if current_region:
                region_shots[current_region] += 1
            total += 1
            filled_count = sum(1 for v in region_shots.values() if v >= shots_per_region)
            print(f"  [{total}] {path}  region={current_region}  ({filled_count}/4 regions done)")
            # 提示语
            if filled_count >= 4:
                print(f"  → 全部区域已完成，按 Q 结束并标定")

        if key == ord('q') or key == 27:
            if total < 15:
                print(f"\n[WARN] 仅采集 {total} 张，建议至少 20 张。确认完成? (Q=退出, 其他键=继续)")
                key2 = cv2.waitKey(0) & 0xFF
                if key2 != ord('q'):
                    continue
            break

    cap.release()
    cv2.destroyAllWindows()

    filled_count = sum(1 for v in region_shots.values() if v >= shots_per_region)
    print(f"\n采集完成: {total} 张, 覆盖 {filled_count}/4 区域")
    for rname in regions:
        print(f"  {rname}: {region_shots[rname]}/{shots_per_region}")
    return total


def collect_web(output_dir="calib_images", device=0, port=8081):
    """HTTP 流模式：浏览器看棋盘格 → 手动保存截图。"""
    from flask import Flask, Response
    os.makedirs(output_dir, exist_ok=True)

    cap = _open_camera(device)

    app = Flask(__name__)
    count = [0]

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_cb, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            vis = frame.copy()
            if ret_cb:
                cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, ret_cb)
            cv2.putText(vis, f"Captured: {count[0]}  SPACE=save",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if ret_cb else (0, 0, 255), 2)
            _, jpg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

    @app.route('/')
    def index():
        return f"""<html><head><title>Calibration</title></head>
        <body style="background:#111;text-align:center;font-family:Arial">
        <h2 style="color:#fff">Camera Calibration ({CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} {SQUARE_SIZE_MM}mm)</h2>
        <img src="/stream" style="max-width:100%">
        <p><a href="/capture" style="font-size:20px;color:#0f0">[ SAVE FRAME ]</a> — {count[0]} saved</p>
        <p style="color:#666">棋盘格检测到（绿色标记）→ 点 SAVE → 换角度 → 重复 20+ 次</p></body></html>"""

    @app.route('/stream')
    def stream():
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/capture')
    def capture():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_cb, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            if ret_cb:
                path = os.path.join(output_dir, f"calib_{count[0]:03d}.jpg")
                cv2.imwrite(path, frame)
                count[0] += 1
                return f"<h2 style='color:#0f0'>Saved #{count[0]}</h2><a href='/'>Back</a>"
            return f"<h2 style='color:red'>Not detected!</h2><a href='/'>Back</a>"
        return "<h2 style='color:red'>Read error</h2>"

    print(f"\n{'='*55}")
    print(f"  标定采集（HTTP 模式）")
    print(f"  浏览器: http://<IP>:{port}")
    print(f"  点 SAVE FRAME 拍照 | 换角度 | 重复 20+ 次")
    print(f"{'='*55}\n")
    app.run(host='0.0.0.0', port=port, threaded=True)
    return count[0]


def main():
    p = argparse.ArgumentParser(description="摄像头标定")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--recalibrate", action="store_true", help="跳过采集，直接用已有图片标定")
    p.add_argument("--web", action="store_true", help="HTTP流模式（树莓派无显示器时使用）")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--output", type=str, default="camera_matrix.npz")
    p.add_argument("--image-dir", type=str, default="calib_images")
    args = p.parse_args()

    if args.recalibrate:
        calibrate_from_images(args.image_dir, args.output)
    elif args.web:
        n = collect_web(args.image_dir, args.device, args.port)
        if n >= 10:
            print("开始标定...")
            calibrate_from_images(args.image_dir, args.output)
    else:
        n = collect_images(args.image_dir, args.device)
        if n >= 10:
            print("开始标定...")
            calibrate_from_images(args.image_dir, args.output)


if __name__ == "__main__":
    main()
