"""
homography_verify.py — 单应性映射验证工具

验证 ArUco 四角 → A4 纸面 mm 坐标的精度的可视化脚本。

操作：
  目测画面中的蓝色框是否与 A4 纸四边精确重合。
  绿色小圈 = 标记外角，应紧贴纸角。
  偏差按 +/- 键调整，Q 退出时打印最终参数。

用法：
  python src/vision/homography_verify.py --camera camera_matrix.npz
"""

import cv2
import numpy as np
import argparse
import platform


A4_W, A4_H = 210.0, 297.0
MARKER_SIZE = 81.0
MARGIN_TOP, MARGIN_BOTTOM, MARGIN_SIDE = 8.0, 22.8, 22.8
CORNER_IDS = {0: "TL", 2: "TR", 3: "BL", 5: "BR"}
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 鼠标坐标
mouse_px = np.array([0.0, 0.0])
current_H = None

def mouse_callback(event, x, y, flags, param):
    global mouse_px
    mouse_px = np.array([float(x), float(y)])
offset_tl = np.array([19.0, 19.0])    # 左上  (已校准)
offset_tr = np.array([-21.0, 18.0])   # 右上  (已校准)
offset_bl = np.array([19.0, -37.0])   # 左下  (已校准)
offset_br = np.array([-21.0, -41.0])  # 右下  (已校准)
selected_corner = 0  # 0=TL, 1=TR, 2=BL, 3=BR
offsets = [offset_tl, offset_tr, offset_bl, offset_br]
corner_names = ["左上(TL)", "右上(TR)", "左下(BL)", "右下(BR)"]
step_mm = 1.0  # 每次按键调整步长


def load_camera(npz_path):
    d = np.load(npz_path)
    return d["mtx"], d["dist"]


def detect_corners(frame_bgr, mtx, dist):
    """检测四角 ArUco 标记，返回纸面四角的像素坐标"""
    frame_undist = cv2.undistort(frame_bgr, mtx, dist)

    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)
        corners, ids, _ = detector.detectMarkers(frame_undist)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(frame_undist, ARUCO_DICT)

    gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if ids is not None:
        for c in corners:
            cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), crit)

    paper_corners_px = {}  # {"TL": (x,y), "TR": ..., "BL": ..., "BR": ...}
    if ids is not None:
        for i, id_val in enumerate(ids.flatten()):
            if id_val in CORNER_IDS:
                pos = CORNER_IDS[id_val]
                c = corners[i].reshape(4, 2)
                if pos == "TL":
                    paper_corners_px["TL"] = c[0]  # 标记左上角=纸角(约)
                elif pos == "TR":
                    paper_corners_px["TR"] = c[1]  # 标记右上角
                elif pos == "BL":
                    paper_corners_px["BL"] = c[3]  # 标记左下角
                elif pos == "BR":
                    paper_corners_px["BR"] = c[2]  # 标记右下角

    if len(paper_corners_px) == 4:
        return frame_undist, paper_corners_px
    return frame_undist, None


def compute_homography(paper_corners_px):
    """从纸面四角像素坐标 + A4 物理尺寸 + 偏移，计算单应性矩阵"""
    # A4 物理角点 (mm)，含偏移
    dst = np.array([
        [0 + offsets[0][0], 0 + offsets[0][1]],           # TL
        [A4_W + offsets[1][0], 0 + offsets[1][1]],        # TR
        [A4_W + offsets[3][0], A4_H + offsets[3][1]],     # BR
        [0 + offsets[2][0], A4_H + offsets[2][1]],        # BL
    ], dtype=np.float32)

    src = np.array([
        paper_corners_px["TL"],
        paper_corners_px["TR"],
        paper_corners_px["BR"],
        paper_corners_px["BL"],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    return H, src, dst


def draw_grid(vis, H, w, h):
    """在画面上绘制 mm 网格，验证映射精度"""
    # 用逆变换：mm → px，在画面上画参考线
    H_inv = np.linalg.inv(H)

    # 画 50mm 间距的网格线
    for x_mm in range(0, int(A4_W) + 1, 50):
        pts_mm = np.array([[x_mm, 0], [x_mm, A4_H]], dtype=np.float32)
        pts_mm = np.c_[pts_mm, np.ones(2)]  # 齐次坐标
        pts_px = (H_inv @ pts_mm.T).T
        pts_px = (pts_px[:, :2] / pts_px[:, 2:3]).astype(np.int32)
        cv2.line(vis, tuple(pts_px[0]), tuple(pts_px[1]), (200, 200, 200), 1)

    for y_mm in range(0, int(A4_H) + 1, 50):
        pts_mm = np.array([[0, y_mm], [A4_W, y_mm]], dtype=np.float32)
        pts_mm = np.c_[pts_mm, np.ones(2)]
        pts_px = (H_inv @ pts_mm.T).T
        pts_px = (pts_px[:, :2] / pts_px[:, 2:3]).astype(np.int32)
        cv2.line(vis, tuple(pts_px[0]), tuple(pts_px[1]), (200, 200, 200), 1)

    # 纸面中心标注
    center_mm = np.array([[A4_W / 2, A4_H / 2, 1]]).T
    center_px = (H_inv @ center_mm).flatten()
    cx, cy = int(center_px[0] / center_px[2]), int(center_px[1] / center_px[2])
    cv2.circle(vis, (cx, cy), 8, (255, 0, 255), -1)
    cv2.putText(vis, f"A4 {A4_W:.0f}x{A4_H:.0f}mm", (cx - 50, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


def main():
    global selected_corner, step_mm, current_H

    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=str, required=True)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    mtx, dist = load_camera(args.camera)

    if platform.system() == "Windows":
        cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print("摄像头无法打开！"); return

    print("\n" + "=" * 60)
    print("  单应性映射验证")
    print("  蓝色框 = 脚本算出的 A4 纸边界 → 目测是否与实物重合")
    print("  绿色圈 = 标记外角 → 应紧贴纸角")
    print("  灰色网格 = 50mm 间距参考线")
    print("  操作: 1-4 选角 | ↑↓←→ 微调偏移 | +/- 改步长 | Q 退出")
    print("=" * 60)

    cv2.namedWindow("Homography Verify", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Homography Verify", mouse_callback)

    last_action = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        fu, paper_px = detect_corners(frame, mtx, dist)

        if paper_px is not None:
            H, src, dst = compute_homography(paper_px)
            current_H = H

            # 绘制
            vis = fu.copy()
            draw_grid(vis, H, fu.shape[1], fu.shape[0])

            # 用 H 逆变换算出纸面四角的像素位置
            H_inv = np.linalg.inv(H)
            paper_corners_mm = np.array([[0, 0], [A4_W, 0], [A4_W, A4_H], [0, A4_H]], dtype=np.float32)
            paper_corners_mm_h = np.c_[paper_corners_mm, np.ones(4)]
            paper_corners_px_all = (H_inv @ paper_corners_mm_h.T).T
            paper_corners_px_all = (paper_corners_px_all[:, :2] / paper_corners_px_all[:, 2:3]).astype(np.int32)
            paper_px_TL, paper_px_TR, paper_px_BR, paper_px_BL = paper_corners_px_all
            paper_pts_all = [(paper_px_TL, "TL"), (paper_px_TR, "TR"), (paper_px_BR, "BR"), (paper_px_BL, "BL")]

            # 蓝色粗框 = A4 纸边界
            cv2.polylines(vis, [paper_corners_px_all.reshape((-1, 1, 2))], True, (255, 80, 0), 4)

            # 四角：大红色十字 = 纸角
            for i, (pt, name) in enumerate(paper_pts_all):
                x, y = int(pt[0]), int(pt[1])
                size = 10
                cv2.line(vis, (x - size, y - size), (x + size, y + size), (0, 0, 255), 3)
                cv2.line(vis, (x - size, y + size), (x + size, y - size), (0, 0, 255), 3)
                cv2.putText(vis, f"A4_{name}", (x + 15, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 小绿色实心圆 = ArUco 标记黑框外角（检测锚点）
            anchor_names = [("TL", paper_px["TL"]), ("TR", paper_px["TR"]),
                           ("BL", paper_px["BL"]), ("BR", paper_px["BR"])]
            mark_colors = [(0, 255, 0)] * 4
            mark_colors[selected_corner] = (0, 255, 255)  # 选中=黄色
            for i, (name, px) in enumerate(anchor_names):
                pt = tuple(np.int32(px))
                cv2.circle(vis, pt, 7, mark_colors[i], -1)
                cv2.circle(vis, pt, 10, mark_colors[i], 2)
                # 连线：锚点→纸角（显示偏移量）
                pp = paper_pts_all[i][0]
                cv2.line(vis, pt, (int(pp[0]), int(pp[1])), mark_colors[i], 1, cv2.LINE_AA)
                cv2.putText(vis, f"锚点{name}", (pt[0] + 12, pt[1] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, mark_colors[i], 1)
                cv2.putText(vis, f"{offsets[i][0]:+.0f},{offsets[i][1]:+.0f}mm",
                            (pt[0] + 12, pt[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        else:
            vis = fu.copy()
            cv2.putText(vis, "未检测到四角标记 (需要 ID0,2,3,5)",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 底部状态
        h, w = vis.shape[:2]
        bar = np.zeros((55, w, 3), dtype=np.uint8) + 30

        # 鼠标 mm 坐标
        mm_str = ""
        if current_H is not None and paper_px is not None:
            px_h = np.array([mouse_px[0], mouse_px[1], 1.0])
            mm_h = current_H @ px_h
            mx, my = mm_h[0] / mm_h[2], mm_h[1] / mm_h[2]
            if 0 <= mx <= A4_W and 0 <= my <= A4_H:
                mm_str = f"  |  鼠标: ({mx:.1f}, {my:.1f})mm"
            else:
                mm_str = f"  |  鼠标: 纸面外"

        cv2.putText(bar, f"选中: {corner_names[selected_corner]}  偏移: {offsets[selected_corner]} mm  步长: {step_mm:.0f}mm{mm_str}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(bar, "1-4选角 | WASD 调整 | +/- 步长 | R 重置 | 鼠标悬停看mm | Q 退出并打印参数",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        if last_action:
            cv2.putText(bar, last_action, (w - 300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        cv2.imshow("Homography Verify", np.vstack([bar, vis]))
        key = cv2.waitKey(1)

        if key == -1:
            pass  # 无按键
        elif key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('1'): selected_corner = 0; last_action = "选中 左上"
        elif key == ord('2'): selected_corner = 1; last_action = "选中 右上"
        elif key == ord('3'): selected_corner = 2; last_action = "选中 左下"
        elif key == ord('4'): selected_corner = 3; last_action = "选中 右下"
        # 方向键
        elif key == 2490368 or key == ord('w') or key == ord('W'):  # Up / W
            offsets[selected_corner][1] -= step_mm
            last_action = f"{corner_names[selected_corner]} Y-={step_mm:.0f}mm → {offsets[selected_corner]}"
        elif key == 2621440 or key == ord('s') or key == ord('S'):  # Down / S
            offsets[selected_corner][1] += step_mm
            last_action = f"{corner_names[selected_corner]} Y+={step_mm:.0f}mm → {offsets[selected_corner]}"
        elif key == 2424832 or key == ord('a') or key == ord('A'):  # Left / A
            offsets[selected_corner][0] -= step_mm
            last_action = f"{corner_names[selected_corner]} X-={step_mm:.0f}mm → {offsets[selected_corner]}"
        elif key == 2555904 or key == ord('d') or key == ord('D'):  # Right / D
            offsets[selected_corner][0] += step_mm
            last_action = f"{corner_names[selected_corner]} X+={step_mm:.0f}mm → {offsets[selected_corner]}"
        elif key == ord('+') or key == ord('='):
            step_mm = min(10, step_mm + 1)
            last_action = f"步长→{step_mm:.0f}mm"
        elif key == ord('-') or key == ord('_'):
            step_mm = max(1, step_mm - 1)
            last_action = f"步长→{step_mm:.0f}mm"
        elif key == ord('r') or key == ord('R'):
            for i in range(4):
                offsets[i][:] = 0.0
            step_mm = 1.0
            last_action = "已全部重置"
        else:
            last_action = f"key={key}"  # 调试：显示按键码

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n最终校准偏移 (mm):")
    for i in range(4):
        print(f"  {corner_names[i]}: {offsets[i]}")


if __name__ == "__main__":
    main()
