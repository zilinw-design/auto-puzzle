"""
camera_test.py — USB 摄像头实时画面显示（Windows）
640×480 @ 30fps，MJPEG 格式，无裁切

用法：
  python camera_test.py

依赖：pip install opencv-python numpy
"""

import cv2
import time
import numpy as np


def main():
    W, H, FPS = 1280, 720, 30

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    # 裁切比例：上下 10%，左右 15%
    CROP_TB = 0.10
    CROP_LR = 0.15
    X1 = int(W * CROP_LR)
    Y1 = int(H * CROP_TB)
    X2 = int(W * (1 - CROP_LR))
    Y2 = int(H * (1 - CROP_TB))

    print(f"{W}x{H} @ {FPS}fps | 裁切 TB{int(CROP_TB*100)}% LR{int(CROP_LR*100)}% | Q 退出 | S 截图")

    cv2.namedWindow("Camera 720p", cv2.WINDOW_NORMAL)

    fps_t0 = time.time()
    fps_cnt = 0
    fps_val = 0.0
    snap = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 四周裁切 10%
        frame = frame[Y1:Y2, X1:X2]

        fps_cnt += 1
        if fps_cnt % 30 == 0:
            now = time.time()
            fps_val = 30 / (now - fps_t0)
            fps_t0 = now

        ch, cw = frame.shape[:2]
        cv2.putText(frame, f"{cw}x{ch}  {fps_val:.1f}fps", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Camera 720p", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"snap_{snap}.jpg", frame)
            print(f"saved: snap_{snap}.jpg")
            snap += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
