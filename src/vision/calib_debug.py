"""
calib_debug.py — 棋盘格检测调试工具
显示实时画面，自动尝试多种棋盘格尺寸，帮助找到正确参数

用法：
  python src/vision/calib_debug.py
"""

import cv2
import numpy as np
import platform


# 常见棋盘格尺寸（内角点 列×行）
COMMON_SIZES = [
    (9, 12),   # 你的代码当前用的
    (8, 11),
    (9, 6),
    (8, 6),
    (7, 7),
    (7, 10),
    (6, 9),
    (5, 8),
    (10, 7),
    (11, 8),
]


def main():
    # 打开摄像头（Windows DS-E12）
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    print("棋盘格检测调试 — 自动尝试多种尺寸")
    print("绿色=检测成功(显示匹配尺寸) | 红色=未检测到")
    print("Q 退出 | 右上角显示检测到哪个尺寸")
    print()

    cv2.namedWindow("Calib Debug", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found = False
        for cols, rows in COMMON_SIZES:
            ret_cb, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
            )

            if ret_cb:
                cv2.drawChessboardCorners(frame, (cols, rows), corners, ret_cb)
                cv2.putText(frame, f"DETECTED: {cols}x{rows} corners",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                found = True
                break

        if not found:
            cv2.putText(frame, "NO CHESSBOARD — 调整角度/距离/光线",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        cv2.putText(frame, "Q=quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Calib Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
