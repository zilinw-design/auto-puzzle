"""
aruco_detector.py — ArUco 标记检测 + 纸面位姿联合求解

A4纸 ArUco 布局（6个标记，81mm边长）：
  上排: ID 0, 1, 2    距纸顶 8mm
  下排: ID 3, 4, 5    距纸底 22.8mm
  左右: 距纸边 22.8mm

所有标记联合 solvePnP → 统一纸面位姿 → 单标记误差互相抵消。

用法：
  python aruco_detector.py --camera camera_matrix.npz
"""

import cv2, numpy as np, argparse, time, platform

A4_W_MM, A4_H_MM = 210.0, 297.0
MARKER_SIZE_MM = 81.0
MARGIN_TOP, MARGIN_BOTTOM, MARGIN_SIDE = 8.0, 22.8, 22.8
CORNER_IDS = {0: "TL", 2: "TR", 3: "BL", 5: "BR"}
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
FX_SCALE = 1.0  # 联合求解用原始标定fx，修正后按需要调整


def load_camera(npz_path):
    d = np.load(npz_path)
    return d["mtx"], d["dist"]


def detect_markers(frame_bgr, mtx, dist):
    """
    检测所有标记 → 联合求解整张纸的统一 3D 位姿。
    返回: (frame, corners_dict, dists_dict, paper_pose)
      corners_dict: {id: 4x2像素角点}
      dists_dict: {id: (D_cm, Z_cm)}
      paper_pose: (rvec, tvec) 纸面在相机坐标系下的位姿
    """
    frame_undist = cv2.undistort(frame_bgr, mtx, dist)

    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)
        corners, ids, _ = detector.detectMarkers(frame_undist)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(frame_undist, ARUCO_DICT)

    # 亚像素精炼
    gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if ids is not None and len(corners) > 0:
        for i in range(len(corners)):
            cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), crit)

    mtx_s = mtx.copy()
    mtx_s[0, 0] *= FX_SCALE
    mtx_s[1, 1] *= FX_SCALE

    corners_dict = {}
    dists_dict = {}
    rvec_paper = tvec_paper = None

    if ids is not None and len(corners) >= 4:
        # 构建: 每个标记4角在A4纸面上的3D坐标 + 对应的2D像素
        obj_pts, img_pts = [], []
        marker_info = {}  # {id: (cx_mm, cy_mm)}

        for i, id_val in enumerate(ids.flatten()):
            id_int = int(id_val)
            c = corners[i].reshape(4, 2)
            corners_dict[id_int] = c

            # 标记中心在A4纸上的坐标 (原点=纸面左上角, X→右, Y→下)
            col = id_int % 3
            row = 0 if id_int <= 2 else 1
            cx_mm = MARGIN_SIDE + MARKER_SIZE_MM / 2 + col * (A4_W_MM - 2 * MARGIN_SIDE - MARKER_SIZE_MM) / 2
            cy_mm = (MARGIN_TOP if row == 0 else A4_H_MM - MARGIN_BOTTOM) + MARKER_SIZE_MM / 2

            half = MARKER_SIZE_MM / 2
            # 标记4角 (左上、右上、右下、左下)
            marker_obj = np.array([
                [cx_mm - half, cy_mm - half, 0],
                [cx_mm + half, cy_mm - half, 0],
                [cx_mm + half, cy_mm + half, 0],
                [cx_mm - half, cy_mm + half, 0],
            ], dtype=np.float32)
            obj_pts.append(marker_obj)
            img_pts.append(c)
            marker_info[id_int] = (cx_mm, cy_mm)

        # 一次 solvePnP 解出整张纸的统一姿态
        ok, rvec, tvec = cv2.solvePnP(
            np.vstack(obj_pts), np.vstack(img_pts), mtx_s, None)

        if ok:
            rvec_paper = rvec.ravel()
            tvec_paper = tvec.ravel()
            R, _ = cv2.Rodrigues(rvec_paper)

            # 各标记中心的距离（用于验证）
            for id_int, (cx, cy) in marker_info.items():
                p3d = np.array([cx, cy, 0], dtype=np.float32)
                pw = R @ p3d + tvec_paper
                dists_dict[id_int] = (
                    float(np.linalg.norm(pw)) / 10,  # D cm
                    float(abs(pw[2])) / 10)           # Z cm

    return frame_undist, corners_dict, dists_dict, (rvec_paper, tvec_paper)


def get_paper_corners(corners_dict):
    """从四角标记的角点推算A4纸四角像素坐标。"""
    needed = [0, 2, 3, 5]
    if not all(i in corners_dict for i in needed):
        return None
    tl_outer = corners_dict[0][0]  # ID0 左上角
    tr_outer = corners_dict[2][1]  # ID2 右上角
    br_outer = corners_dict[5][2]  # ID5 右下角
    bl_outer = corners_dict[3][3]  # ID3 左下角
    return np.array([tl_outer, tr_outer, br_outer, bl_outer], dtype=np.float32)


# =========================================================================
# 实时模式
# =========================================================================

def run_realtime(camera_path, device=0, web=False, port=8082):
    mtx, dist = load_camera(camera_path)

    if platform.system() == "Windows":
        cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print("[ERROR] Camera not available"); return

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
        fu, cd, dists, pose = detect_markers(frame, mtx, dist)

        # 绘制
        for idv, pts in cd.items():
            pts_i = pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(fu, [pts_i], True, (0, 255, 0), 2)
            ct = tuple(pts.mean(axis=0).astype(int))
            cv2.putText(fu, f"ID{idv}", ct, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if idv in dists:
                d3d, z = dists[idv]
                cv2.putText(fu, f"Z={z:.1f} D={d3d:.1f}", (ct[0], ct[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 纸面框
        paper = get_paper_corners(cd)
        if paper is not None:
            cv2.polylines(fu, [paper.reshape((-1, 1, 2)).astype(np.int32)], True, (255, 0, 0), 2)

        # 底部信息
        h, w = fu.shape[:2]
        bar = np.zeros((35, w, 3), dtype=np.uint8) + 30
        info = " | ".join(f"ID{i}:Z={dists[i][1]:.1f} D={dists[i][0]:.1f}" for i in sorted(dists))
        cv2.putText(bar, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow("ArUco Detector", np.vstack([bar, fu]))

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()


def _run_web(cap, mtx, dist, port):
    from flask import Flask, Response
    app = Flask(__name__)
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret: break
            fu, cd, dists, pose = detect_markers(frame, mtx, dist)
            for idv, pts in cd.items():
                cv2.polylines(fu, [pts.reshape((-1, 1, 2)).astype(np.int32)], True, (0, 255, 0), 2)
                ctr = tuple(pts.mean(axis=0).astype(int))
                cv2.putText(fu, f"ID{idv}", ctr, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if idv in dists:
                    cv2.putText(fu, f"Z={dists[idv][1]:.1f}", (ctr[0], ctr[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            paper = get_paper_corners(cd)
            if paper is not None:
                cv2.polylines(fu, [paper.reshape((-1, 1, 2)).astype(np.int32)], True, (255, 0, 0), 2)
            _, jpg = cv2.imencode('.jpg', fu, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

    @app.route('/')
    def index():
        return """<html><body style="background:#111;text-align:center">
        <h2 style="color:#fff">ArUco Paper Pose</h2>
        <img src="/stream" style="max-width:100%"></body></html>"""
    @app.route('/stream')
    def stream():
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    print(f"HTTP: http://<IP>:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)


# =========================================================================
# 单张测试
# =========================================================================

def test_single(image_path, camera_path):
    mtx, dist = load_camera(camera_path)
    img = cv2.imread(image_path)
    fu, cd, dists, pose = detect_markers(img, mtx, dist)
    print(f"检测到 {len(cd)} 个标记, 纸面位姿: {'OK' if pose[0] is not None else 'FAIL'}")
    for i in sorted(dists):
        print(f"  ID{i}: Z={dists[i][1]:.1f}cm D={dists[i][0]:.1f}cm")
    for idv, pts in cd.items():
        cv2.polylines(fu, [pts.reshape((-1, 1, 2)).astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite("aruco_test_result.jpg", fu)
    print("  结果: aruco_test_result.jpg")


# =========================================================================

def main():
    p = argparse.ArgumentParser(description="ArUco 检测 + 纸面位姿联合求解")
    p.add_argument("--camera", type=str, required=True)
    p.add_argument("--image", type=str)
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
