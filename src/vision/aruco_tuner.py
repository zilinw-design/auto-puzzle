"""
aruco_tuner.py — ArUco 距离交互校准工具（滑条+按键微调版）

用法：
  python src/vision/aruco_tuner.py --camera camera_matrix.npz
"""

import cv2
import numpy as np
import argparse
import platform
import tkinter as tk
from tkinter import ttk
import threading
import time
from PIL import Image, ImageTk


CORNER_IDS = {0: "TL", 2: "TR", 3: "BL", 5: "BR"}
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


class ArucoTuner:
    def __init__(self, camera_path, device=0):
        self.mtx_orig, self.dist_orig = self._load_camera(camera_path)
        self.root = tk.Tk()

        # 参数
        self.fx_scale = tk.IntVar(value=100)
        self.marker_mm = tk.IntVar(value=81)
        self.dist_scale = tk.IntVar(value=100)  # 畸变强度缩放%
        self.id_scales = {i: tk.IntVar(value=100) for i in range(6)}

        self.results = {}
        self.device = device
        self.cap = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        self._build_ui()

    def _load_camera(self, path):
        d = np.load(path)
        return d["mtx"], d["dist"]

    def _open_camera(self):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.device)
        return cap

    def _step(self, var, delta, lo, hi):
        v = var.get() + delta
        var.set(max(lo, min(hi, v)))

    def _build_ui(self):
        self.root.title("ArUco 距离校准")
        self.root.geometry("1380x820")
        self.root.minsize(1100, 650)

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6, pady=6)

        # 左侧视频
        left = ttk.Frame(body, width=900, height=650)
        body.add(left, weight=3)
        self.video_label = ttk.Label(left, background="#111")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # 右侧控制面板
        right = ttk.Frame(body, width=440)
        body.add(right, weight=1)
        right.pack_propagate(False)

        # ─── 全局参数 ───
        gf = ttk.LabelFrame(right, text="全局参数", padding=8)
        gf.pack(fill="x", pady=(0, 6))

        # fx 缩放
        row = ttk.Frame(gf)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="fx 缩放", width=10).pack(side="left")
        ttk.Button(row, text="-2", width=3,
                   command=lambda: self._step(self.fx_scale, -2, 50, 400)).pack(side="left", padx=1)
        ttk.Scale(row, from_=50, to=400, variable=self.fx_scale,
                  orient="horizontal", length=180).pack(side="left", padx=4)
        ttk.Button(row, text="+2", width=3,
                   command=lambda: self._step(self.fx_scale, 2, 50, 400)).pack(side="left", padx=1)
        self.fx_label = ttk.Label(row, text="100% → fx=718", width=16, foreground="#00a")
        self.fx_label.pack(side="left", padx=8)
        self.fx_scale.trace_add("write", lambda *_: self.fx_label.configure(
            text=f"{self.fx_scale.get()}% → fx={718 * self.fx_scale.get() / 100:.0f}"))

        # marker mm
        row2 = ttk.Frame(gf)
        row2.pack(fill="x", pady=3)
        ttk.Label(row2, text="标记 mm", width=10).pack(side="left")
        ttk.Button(row2, text="-2", width=3,
                   command=lambda: self._step(self.marker_mm, -2, 50, 120)).pack(side="left", padx=1)
        ttk.Scale(row2, from_=50, to=120, variable=self.marker_mm,
                  orient="horizontal", length=180).pack(side="left", padx=4)
        ttk.Button(row2, text="+2", width=3,
                   command=lambda: self._step(self.marker_mm, 2, 50, 120)).pack(side="left", padx=1)
        self.mm_label = ttk.Label(row2, text="81mm", width=16)
        self.mm_label.pack(side="left", padx=8)
        self.marker_mm.trace_add("write", lambda *_: self.mm_label.configure(
            text=f"{self.marker_mm.get()}mm"))

        # 畸变强度
        row3 = ttk.Frame(gf)
        row3.pack(fill="x", pady=3)
        ttk.Label(row3, text="畸变强度", width=10).pack(side="left")
        ttk.Button(row3, text="-5", width=3,
                   command=lambda: self._step(self.dist_scale, -5, 0, 200)).pack(side="left", padx=1)
        ttk.Scale(row3, from_=0, to=200, variable=self.dist_scale,
                  orient="horizontal", length=180).pack(side="left", padx=4)
        ttk.Button(row3, text="+5", width=3,
                   command=lambda: self._step(self.dist_scale, 5, 0, 200)).pack(side="left", padx=1)
        self.dist_label = ttk.Label(row3, text="100% (原始)", width=16)
        self.dist_label.pack(side="left", padx=8)
        self.dist_scale.trace_add("write", lambda *_: self.dist_label.configure(
            text=f"{self.dist_scale.get()}%" + (" (原始)" if self.dist_scale.get() == 100 else "")))

        # ─── 逐ID校准 ───
        idf = ttk.LabelFrame(right, text="逐ID校准（步进±2）", padding=8)
        idf.pack(fill="x", pady=(0, 6))

        self.id_scale_labels = {}
        self.id_z_labels = {}
        self.id_d_labels = {}
        self.id_xyz_labels = {}

        for i in range(6):
            rf = ttk.Frame(idf)
            rf.pack(fill="x", pady=2)

            ttk.Label(rf, text=f"ID{i}", width=3, font=("", 9, "bold")).pack(side="left", padx=(0, 3))
            ttk.Button(rf, text="-", width=2,
                       command=lambda idx=i: self._step(self.id_scales[idx], -2, 50, 200)).pack(side="left")
            ttk.Scale(rf, from_=50, to=200, variable=self.id_scales[i],
                      orient="horizontal", length=100).pack(side="left", padx=2)
            ttk.Button(rf, text="+", width=2,
                       command=lambda idx=i: self._step(self.id_scales[idx], 2, 50, 200)).pack(side="left")
            self.id_scale_labels[i] = ttk.Label(rf, text="100%", width=5, foreground="#666")
            self.id_scale_labels[i].pack(side="left", padx=3)
            self.id_z_labels[i] = ttk.Label(rf, text="Z=--", width=9, foreground="#080")
            self.id_z_labels[i].pack(side="left", padx=2)
            self.id_d_labels[i] = ttk.Label(rf, text="D=--", width=9, foreground="#048")
            self.id_d_labels[i].pack(side="left", padx=2)
            self.id_xyz_labels[i] = ttk.Label(rf, text="", width=28, foreground="#666")
            self.id_xyz_labels[i].pack(side="left", padx=2)

            # 滑条变化时更新
            self.id_scales[i].trace_add("write",
                lambda *_, idx=i: self.id_scale_labels[idx].configure(
                    text=f"{self.id_scales[idx].get()}%"))

        # ─── 实测值校准 ───
        cal = ttk.LabelFrame(right, text="实测值校准（输入尺子测量的D值cm，回车生效）", padding=6)
        cal.pack(fill="x", pady=(0, 6))

        self.measured_vars = {}
        self.correction_labels = {}
        cal_row = ttk.Frame(cal)
        cal_row.pack(fill="x")
        for i in range(6):
            col = ttk.Frame(cal_row)
            col.pack(side="left", padx=4)
            ttk.Label(col, text=f"ID{i}", font=("", 8, "bold")).pack()
            self.measured_vars[i] = tk.StringVar(value="")
            e = ttk.Entry(col, textvariable=self.measured_vars[i], width=6)
            e.pack()
            e.bind("<Return>", lambda evt, idx=i: self._apply_measured(idx))
            self.correction_labels[i] = ttk.Label(col, text="", font=("", 7), foreground="#c00")
            self.correction_labels[i].pack()
        ttk.Button(cal, text="全部应用", command=self._apply_all_measured).pack(pady=4)

        # ─── 快捷按钮 ───
        bf = ttk.Frame(right)
        bf.pack(fill="x", pady=(0, 6))
        ttk.Button(bf, text="全部重置", command=self._reset).pack(side="left", padx=2)
        ttk.Button(bf, text="打印参数", command=self._print).pack(side="left", padx=2)

        # ─── 图例 ───
        leg = ttk.LabelFrame(right, text="显示说明", padding=6)
        leg.pack(fill="x")
        ttk.Label(leg, text=(
            "Z=垂直距离(光轴方向) 用量尺比对\n"
            "D=3D斜距(含X/Y偏移)\n"
            "x/y/z=标记在相机坐标系下的3D坐标(cm)\n"
            "滑条拖动粗调 / +-按钮步进±2精调\n"
            "目标: 中心ID1的Z = 尺子实测值"),
            justify="left", font=("", 8)).pack()

        self.status = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status, relief="sunken",
                  anchor="w", padding=(10, 3)).pack(side="bottom", fill="x")

        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._start_camera)

    def _reset(self):
        self.fx_scale.set(100)
        self.marker_mm.set(81)
        self.dist_scale.set(100)
        for i in range(6):
            self.id_scales[i].set(100)
        self.status.set("已全部重置")

    def _apply_measured(self, idx):
        """输入单个ID的实测D值，自动计算该ID缩放"""
        try:
            measured = float(self.measured_vars[idx].get())
        except (ValueError, tk.TclError):
            return
        if idx not in self.results:
            self.status.set(f"ID{idx} 未检测到，无法校准")
            return
        current_d = self.results[idx]["D"]
        if current_d < 1:
            return
        ratio = measured / current_d
        new_scale = self.id_scales[idx].get() * ratio
        new_scale = max(50, min(200, int(new_scale)))
        self.id_scales[idx].set(new_scale)
        self.correction_labels[idx].configure(
            text=f"×{ratio:.3f}", foreground="#080" if 0.95 < ratio < 1.05 else "#c00")
        self.status.set(f"ID{idx}: 实测{measured:.1f}/当前{current_d:.1f}={ratio:.3f} → 缩放{new_scale}%")

    def _apply_all_measured(self):
        """用实测Z值校准（所有标记应在同一平面，Z应相等）"""
        ratios = {}
        for i in range(6):
            try:
                measured = float(self.measured_vars[i].get())
            except (ValueError, tk.TclError):
                continue
            if i in self.results and self.results[i]["Z"] > 1:
                ratios[i] = measured / self.results[i]["Z"]

        if not ratios:
            self.status.set("请先输入至少一个ID的实测Z值")
            return

        avg_ratio = sum(ratios.values()) / len(ratios)
        new_fx = self.fx_scale.get() * avg_ratio
        self.fx_scale.set(max(50, min(400, int(new_fx))))

        for i, ratio in ratios.items():
            self.correction_labels[i].configure(
                text=f"Z×{ratio:.3f}", foreground="#080" if 0.95 < ratio < 1.05 else "#c00")

        self.status.set(f"全局fx→{self.fx_scale.get()}% (Z×{avg_ratio:.3f})")

    def _print(self):
        print(f"\n=== 参数 ===")
        print(f"fx={self.fx_scale.get()}% ({718 * self.fx_scale.get() / 100:.0f}) "
              f"marker={self.marker_mm.get()}mm dist={self.dist_scale.get()}%")
        for i in range(6):
            print(f"  ID{i}: {self.id_scales[i].get()}% "
                  f"eff_marker={self.marker_mm.get() * self.id_scales[i].get() / 100:.1f}mm")
        for i in sorted(self.results.keys()):
            r = self.results[i]
            print(f"  ID{i}: Z={r['Z']:.1f}cm D={r['D']:.1f}cm x={r['X']:.1f} y={r['Y']:.1f}")
        print()

    def _start_camera(self):
        self.cap = self._open_camera()
        if not self.cap.isOpened():
            self.status.set("摄像头无法打开！")
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        self.root.after(60, self._update)

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            time.sleep(0.005)

    def _detect(self, frame_bgr):
        # 畸变系数缩放（100%=原始标定值）
        ds = self.dist_scale.get() / 100.0
        dist_scaled = self.dist_orig * ds
        frame_undist = cv2.undistort(frame_bgr, self.mtx_orig, dist_scaled)
        mtx_s = self.mtx_orig.copy()
        fs = self.fx_scale.get() / 100.0
        mtx_s[0, 0] *= fs
        mtx_s[1, 1] *= fs
        mm = self.marker_mm.get()

        try:
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)
            corners, ids, _ = detector.detectMarkers(frame_undist)
        except AttributeError:
            corners, ids, _ = cv2.aruco.detectMarkers(frame_undist, ARUCO_DICT)

        gray = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2GRAY)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        if ids is not None and len(corners) > 0:
            for i in range(len(corners)):
                cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), crit)

        cd, td = {}, {}
        if ids is not None and len(corners) > 0:
            for i, idv in enumerate(ids.flatten()):
                c = corners[i].reshape(4, 2)
                cd[idv] = c
                em = mm * (self.id_scales.get(int(idv), tk.IntVar(value=100)).get() / 100.0)
                obj = np.array([[0, 0, 0], [em, 0, 0], [em, em, 0], [0, em, 0]], dtype=np.float32)
                ok, rv, tv = cv2.solvePnP(obj, c, mtx_s, None)
                if ok:
                    td[idv] = tv.ravel()
        return frame_undist, cd, td, mtx_s

    def _update(self):
        if not self.running:
            return
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
        if frame is not None:
            fu, cd, td, mtx_s = self._detect(frame)
            h, w = fu.shape[:2]

            # 收集每个ID的数据，统一画在底部信息条
            id_info_lines = []
            for idv, pts in cd.items():
                pi = pts.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(fu, [pi], True, (0, 255, 0), 2)
                ct = tuple(pts.mean(axis=0).astype(int))
                cv2.circle(fu, ct, 5, (0, 255, 255), -1)
                # 只显示ID号，不显示距离数字
                cv2.putText(fu, f"ID{idv}", (ct[0] - 15, ct[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 3)
                cv2.putText(fu, f"ID{idv}", (ct[0] - 15, ct[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                if idv in td:
                    tv = td[idv]
                    zc, dc = abs(tv[2]) / 10, np.linalg.norm(tv) / 10
                    self.results[int(idv)] = {"Z": zc, "D": dc, "X": tv[0] / 10, "Y": tv[1] / 10}
                    id_info_lines.append(f"ID{idv}: Z={zc:.1f} D={dc:.1f} |")

            # 底部信息条：两行
            bar_h = 50
            bar = np.zeros((bar_h, w, 3), dtype=np.uint8) + 35
            cv2.putText(bar, f"fx={mtx_s[0,0]:.0f}  marker={self.marker_mm.get()}mm  dist={self.dist_scale.get()}%",
                        (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            # 距离数据一行显示
            info_text = "  ".join(id_info_lines)
            cv2.putText(bar, info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)
            fu = np.vstack([bar, fu])

            scale = min(920 / w, 620 / (h + bar_h))
            dw, dh = max(1, int(w * scale)), max(1, int((h + bar_h) * scale))
            disp = cv2.resize(fu, (dw, dh), interpolation=cv2.INTER_AREA)
            disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(disp_rgb))
            self.video_label.configure(image=self.photo)

            # 更新右侧数据
            for i in range(6):
                if i in self.results:
                    r = self.results[i]
                    self.id_z_labels[i].configure(text=f"Z={r['Z']:.1f}cm")
                    self.id_d_labels[i].configure(text=f"D={r['D']:.1f}cm")
                    self.id_xyz_labels[i].configure(
                        text=f"x={r['X']:.1f} y={r['Y']:.1f} z={r['Z']:.1f}")
                else:
                    self.id_z_labels[i].configure(text="Z=--")
                    self.id_d_labels[i].configure(text="D=--")
                    self.id_xyz_labels[i].configure(text="")

        self.root.after(60, self._update)

    def _close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=str, required=True)
    p.add_argument("--device", type=int, default=0)
    ArucoTuner(p.parse_args().camera, p.parse_args().device).run()


if __name__ == "__main__":
    main()
