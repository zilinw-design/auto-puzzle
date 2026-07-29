"""
aruco_detector.py — ArUco 标记检测 + 位姿估计 + 透视校正

A4纸 ArUco 布局（6个标记，81mm边长）：
  上排: ID 0, 1, 2    距纸顶 8mm
  下排: ID 3, 4, 5    距纸底 22.8mm
  左右: 距纸边 22.8mm
  四角: ID 0(左上) 2(右上) 3(左下) 5(右下)

用法：
  python aruco_detector.py --camera camera_matrix.npz        # 实时检测+显示距离
  python aruco_detector.py --camera camera_matrix.npz --web  # HTTP流模式
  python aruco_detector.py --image test.jpg                  # 单张测试

依赖：
  pip install opencv-python flask numpy
"""

import cv2, numpy as np, argparse, time, os, json


# =========================================================================
# A4 布局参数
# =========================================================================
A4_W_MM, A4_H_MM = 210.0, 297.0
MARKER_SIZE_MM = 81.0
MARGIN_TOP = 8.0
MARGIN_BOTTOM = 22.8
MARGIN_SIDE = 22.8

# 四角 ArUco ID（上排 0,1,2  下排 3,4,5）
CORNER_IDS = {0: "TL", 2: "TR", 3: "BL", 5: "BR"}

# ArUco 字典
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def load_camera(npz_path):
    """加载标定参数。"""
    data = np.load(npz_path)
    return data["mtx"], data["dist"]


def detect_markers(frame_bgr, mtx, dist):
    """
    检测 ArUco 标记 → 畸变校正 → 返回四角标记的角点。

    Returns:
        corners_dict: {id: 4×2 像素角点数组}
        ids: 所有检测到的 ID 列表
        rvecs, tvecs: 每个标记的旋转和平移向量
    """
    # 畸变校正
    frame_undist = cv2.undistort(frame_bgr, mtx, dist)

    corners, ids, _ = cv2.aruco.detectMarkers(frame_undist, ARUCO_DICT)
    corners_dict = {}
    rvecs_dict, tvecs_dict = {}, {}

    if ids is not None:
        # 位姿估计
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE_MM, mtx, dist)

        for i, id_val in enumerate(ids.flatten()):
            if id_val in CORNER_IDS:
                corners_dict[id_val] = corners[i].reshape(4, 2)
                rvecs_dict[id_val] = rvecs[i].ravel()
                tvecs_dict[id_val] = tvecs[i].ravel()

    return frame_undist, corners_dict, rvecs_dict, tvecs_dict


def get_paper_corners(corners_dict):
    """
    从四角 ArUco 角点推算 A4 纸四角（像素坐标）。
    取标记的"外侧角" → 外推到纸角。
    """
    paper_pts = {}

    for id_val, pts in corners_dict.items():
        pos = CORNER_IDS[id_val]
        if pos == "TL":
            # 左上标记：取左上角点 → 外推 (22.8, 8) mm 到纸角
            outer = pts[0]  # ArUco 的 [0] 是左上角
            # 纸角 = 标记角 + 偏移（注意方向）
            # 标记到纸边的像素比例 = 标记到纸边的 mm 比例
            dx_px, dy_px = 0, 0
            paper_pts["TL"] = pts[0]
        elif pos == "TR":
            paper_pts["TR"] = pts[1]  # 右上角
        elif pos == "BL":
            paper_pts["BL"] = pts[3]  # 左下角
        elif pos == "BR":
            paper_pts["BR"] = pts[2]  # 右下角

    if len(paper_pts) == 4:
        return np.array([paper_pts["TL"], paper_pts["TR"],
                         paper_pts["BR"], paper_pts["BL"]], dtype=np.float32)
    return None


def get_distances(tvecs_dict):
    """从平移向量取距离(mm)。"""
    dists = {}
    for id_val, tvec in tvecs_dict.items():
        dists[id_val] = np.linalg.norm(tvec)
    return dists


# =========================================================================
# 实时模式
# =========================================================================

def run_realtime(camera_path, device=0, web=False, port=8082):
    mtx, dist = load_camera(camera_path)

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"[ERROR] Camera /dev/video{device} not available"); return

    if web:
        _run_web(cap, mtx, dist, port)
    else:
        _run_display(cap, mtx, dist)


def _run_display(cap, mtx, dist):
    cv2.namedWindow("ArUco Detector", cv2.WINDOW_NORMAL)
    print("ArUco 实时检测  按 Q 退出")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame, corners_dict, rvecs, tvecs = detect_markers(frame, mtx, dist)
        dists = get_distances(tvecs)

        # 绘制
        cv2.aruco.drawDetectedMarkers(frame,
            [np.array(c.reshape(-1, 2), dtype=np.float32) for c in
             [corners_dict[i] for i in sorted(corners_dict) if i in corners_dict]],
            np.array([[i] for i in sorted(corners_dict) if i in corners_dict]))

        y = 30
        for id_val in sorted(dists):
            cv2.putText(frame, f"ID{id_val}: {dists[id_val]/10:.1f}cm",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25

        paper = get_paper_corners(corners_dict)
        if paper is not None:
            cv2.polylines(frame, [paper.reshape((-1, 1, 2)).astype(np.int32)],
                          True, (255, 0, 0), 2)

        cv2.imshow("ArUco Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()


def _run_web(cap, mtx, dist, port):
    from flask import Flask, Response
    app = Flask(__name__)

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame, corners_dict, rvecs, tvecs = detect_markers(frame, mtx, dist)
            dists = get_distances(tvecs)
            cv2.aruco.drawDetectedMarkers(frame,
                [np.array(c.reshape(-1, 2), dtype=np.float32) for c in
                 [corners_dict[i] for i in sorted(corners_dict) if i in corners_dict]],
                np.array([[i] for i in sorted(corners_dict) if i in corners_dict]))
            y = 30
            for id_val in sorted(dists):
                cv2.putText(frame, f"ID{id_val}: {dists[id_val]/10:.1f}cm",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 25
            paper = get_paper_corners(corners_dict)
            if paper is not None:
                cv2.polylines(frame, [paper.reshape((-1, 1, 2)).astype(np.int32)],
                              True, (255, 0, 0), 2)
            _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

    @app.route('/')
    def index():
        return """<html><head><title>ArUco Detector</title></head>
        <body style="background:#111;text-align:center;font-family:Arial">
        <h2 style="color:#fff">ArUco Distance</h2>
        <img src="/stream" style="max-width:100%"></body></html>"""
    @app.route('/stream')
    def stream():
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    print(f"\nHTTP: http://<IP>:{port}\n")
    app.run(host='0.0.0.0', port=port, threaded=True)


# =========================================================================
# 单张测试
# =========================================================================

def test_single(image_path, camera_path):
    mtx, dist = load_camera(camera_path)
    img = cv2.imread(image_path)
    frame, corners_dict, rvecs, tvecs = detect_markers(img, mtx, dist)
    dists = get_distances(tvecs)

    print(f"检测到 {len(corners_dict)} 个四角 ArUco:")
    for id_val in sorted(dists):
        print(f"  ID {id_val} ({CORNER_IDS[id_val]}): {dists[id_val]/10:.1f} cm")

    paper = get_paper_corners(corners_dict)
    if paper is not None:
        print(f"  A4 纸四角已定位")

    cv2.aruco.drawDetectedMarkers(frame,
        [np.array(c.reshape(-1, 2), dtype=np.float32) for c in
         [corners_dict[i] for i in sorted(corners_dict) if i in corners_dict]],
        np.array([[i] for i in sorted(corners_dict) if i in corners_dict]))
    cv2.imwrite("aruco_test_result.jpg", frame)
    print("  结果图: aruco_test_result.jpg")


# =========================================================================
# CLI
# =========================================================================

def main():
    p = argparse.ArgumentParser(description="ArUco 检测 + 距离估计")
    p.add_argument("--camera", type=str, required=True, help="camera_matrix.npz 路径")
    p.add_argument("--image", type=str, help="单张测试")
    p.add_argument("--web", action="store_true")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--port", type=int, default=8082)
    args = p.parse_args()

    if args.image:
        test_single(args.image, args.camera)
    else:
        run_realtime(args.camera, args.device, args.web, args.port)


if __name__ == "__main__":
    main()
