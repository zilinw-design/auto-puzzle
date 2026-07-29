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
CHESSBOARD_SIZE = (9, 12)      # 内角点 (列, 行)
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
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_shape, None, None)

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


def collect_images(output_dir="calib_images", device=0):
    """
    实时采集棋盘格图像。按 SPACE 拍照，按 Q 完成。
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"[ERROR] 摄像头 /dev/video{device} 无法打开")
        return

    print(f"\n{'='*55}")
    print(f"  棋盘格采集: {CHESSBOARD_SIZE[0]}×{CHESSBOARD_SIZE[1]}  格边{SQUARE_SIZE_MM}mm")
    print(f"  操作: SPACE=拍照 | Q=完成标定")
    print(f"  要求: 20-30张，覆盖不同角度、距离、倾斜")
    print(f"{'='*55}\n")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        vis = frame.copy()
        if ret_cb:
            cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, ret_cb)

        cv2.putText(vis, f"Captured: {count}  [SPACE=capture Q=done]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ret_cb else (0, 0, 255), 2)

        cv2.imshow("Calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and ret_cb:
            path = os.path.join(output_dir, f"calib_{count:03d}.jpg")
            cv2.imwrite(path, frame)
            print(f"  [{count+1}] {path}")
            count += 1

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n采集完成，共 {count} 张 → {output_dir}/\n")
    return count


def main():
    p = argparse.ArgumentParser(description="摄像头标定")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--recalibrate", action="store_true", help="跳过采集，直接用已有图片标定")
    p.add_argument("--output", type=str, default="camera_matrix.npz")
    p.add_argument("--image-dir", type=str, default="calib_images")
    args = p.parse_args()

    if args.recalibrate:
        calibrate_from_images(args.image_dir, args.output)
    else:
        n = collect_images(args.image_dir, args.device)
        if n >= 10:
            print("开始标定...")
            calibrate_from_images(args.image_dir, args.output)


if __name__ == "__main__":
    main()
