"""
live_solve.py — 摄像头拍一张 → 检测 → 求解 → 输出机械臂指令

用法:
  python live_solve.py                      # 拍一张，求解
  python live_solve.py --display             # 带画面预览
"""

import os, sys, argparse, cv2, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "algorithm"))
import pipeline

# 复用 realtime_detector 的检测函数
from realtime_detector import detect_fused, camera_auto_setup, CLAHE

PX_PER_MM = 3.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-w", type=float, default=100, help="目标矩形宽 mm (默认100)")
    p.add_argument("--target-h", type=float, default=60, help="目标矩形高 mm (默认60)")
    p.add_argument("--display", action="store_true")
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    camera_auto_setup(f"/dev/video{args.device}")

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] 摄像头无法打开"); return

    print("摄像头已就绪。拍摄中...")

    # 拍一张（多读几帧让 AE 稳定）
    for _ in range(10):
        ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] 读取帧失败"); return

    print(f"图像: {frame.shape[1]}×{frame.shape[0]}")

    # 检测
    polys, v_thresh = detect_fused(frame)
    print(f"检测到 {len(polys)} 个碎片")

    if args.display:
        vis = frame.copy()
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        for i, poly in enumerate(polys):
            cv2.polylines(vis, [poly.reshape((-1, 1, 2))], True, colors[i % 4], 2)
            M = cv2.moments(poly)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.circle(vis, (cx, cy), 4, colors[i % 4], -1)
        cv2.imwrite("live_solve_debug.jpg", vis)
        print("调试图已保存: live_solve_debug.jpg")

    # 求解
    solver_polys = [p.astype(np.float32) for p in polys]
    result = pipeline.solve_and_convert(
        solver_polys, args.target_w, args.target_h, PX_PER_MM)

    # 输出
    from arm_instructions import generate_from_pipeline
    generate_from_pipeline(result)


if __name__ == "__main__":
    main()
