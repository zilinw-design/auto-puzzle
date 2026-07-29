"""
arm_instructions.py — 将拼图解转换为机械臂指令

输入: ground truth JSON（含碎片当前位置 + 目标矩形位置）
输出: 每个碎片的机械臂操作序列

用法:
  python arm_instructions.py                              # 单张示例
  python arm_instructions.py --all                        # 全部输出
  python arm_instructions.py --gt ground_truth/sample_0000.json
"""

import os, sys, json, math, argparse
import cv2, numpy as np


def polygon_centroid(verts):
    n = len(verts)
    area, cx, cy = 0.0, 0.0, 0.0
    for i in range(n):
        x1, y1 = verts[i]; x2, y2 = verts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area += cross; cx += (x1 + x2) * cross; cy += (y1 + y2) * cross
    area *= 0.5
    if abs(area) < 1e-10:
        return sum(v[0] for v in verts) / n, sum(v[1] for v in verts) / n
    return cx / (6 * area), cy / (6 * area)


def compute_orientation(verts):
    """用 PCA 或 minAreaRect 计算碎片主方向（°）。"""
    pts = np.array(verts, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    _, (rw, rh), angle = rect
    if rw < rh:
        angle += 90
    # 规范化到 [0, 180)
    angle = angle % 180
    return round(angle, 1)


def generate_instructions(gt_path):
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    rect_w, rect_h = gt["target_rectangle_size_mm"]
    frags = gt["fragments"]

    instructions = []
    for fg in frags:
        pid = fg["piece_id"]

        # 当前位置（在 A4 纸上的 mm 坐标）
        placed_verts = [(v[0], v[1]) for v in fg["vertices_placed_mm"]]
        cur_cx, cur_cy = polygon_centroid(placed_verts)
        cur_angle = compute_orientation(placed_verts)

        # 目标位置（在矩形中的 mm 坐标）
        rect_verts = [(v[0], v[1]) for v in fg["vertices_rect_mm"]]
        tgt_cx, tgt_cy = polygon_centroid(rect_verts)
        tgt_angle = compute_orientation(rect_verts)

        # Δθ = 目标角度 - 当前角度
        delta_theta = round(tgt_angle - cur_angle, 1)
        # 取最短旋转路径
        if delta_theta > 90:
            delta_theta -= 180
        elif delta_theta < -90:
            delta_theta += 180

        instructions.append({
            "piece_id": pid,
            "num_edges": fg["num_edges"],
            "from_mm": [round(cur_cx, 1), round(cur_cy, 1)],
            "from_angle_deg": cur_angle,
            "to_mm": [round(tgt_cx, 1), round(tgt_cy, 1)],
            "to_angle_deg": tgt_angle,
            "delta_angle_deg": delta_theta,
            "distance_mm": round(math.hypot(tgt_cx - cur_cx, tgt_cy - cur_cy), 1),
        })

    # 放置顺序：按目标位置从外到内（先放靠边的）
    def boundary_score(inst):
        x, y = inst["to_mm"]
        return -min(x, rect_w - x, y, rect_h - y)  # 离边越近越先放
    instructions.sort(key=boundary_score)

    return instructions, rect_w, rect_h


def print_instructions(instructions, rect_w, rect_h, sample_id=""):
    print(f"\n{'='*65}")
    print(f"  机械臂指令序列 — {sample_id}  (目标矩形: {rect_w:.0f}×{rect_h:.0f}mm)")
    print(f"{'='*65}")
    print(f"  {'#':<4} {'F':<5} {'当前位置(mm)':<18} {'角度':<8} {'→':<4} "
          f"{'目标位置(mm)':<18} {'角度':<8} {'Δθ':<8} {'移动':<8}")
    print(f"  {'-'*60}")

    for i, inst in enumerate(instructions):
        print(f"  {i+1:<4} "
              f"F{inst['piece_id']:<4} "
              f"({inst['from_mm'][0]:6.1f}, {inst['from_mm'][1]:6.1f})  "
              f"{inst['from_angle_deg']:5.1f}°"
              f"  →  "
              f"({inst['to_mm'][0]:6.1f}, {inst['to_mm'][1]:6.1f})  "
              f"{inst['to_angle_deg']:5.1f}°"
              f"  {inst['delta_angle_deg']:+6.1f}°"
              f"  {inst['distance_mm']:5.1f}mm")

    print(f"  {'='*60}")
    print(f"  操作序列:")
    for i, inst in enumerate(instructions):
        print(f"  {i+1}. MOVE to ({inst['from_mm'][0]:.1f}, {inst['from_mm'][1]:.1f}) "
              f"→ GRIP → ROTATE {inst['delta_angle_deg']:+.1f}° "
              f"→ MOVE to ({inst['to_mm'][0]:.1f}, {inst['to_mm'][1]:.1f}) → RELEASE")
    print()


def main():
    p = argparse.ArgumentParser(description="拼图→机械臂指令")
    p.add_argument("--gt", type=str)
    p.add_argument("--all", action="store_true")
    p.add_argument("--json", type=str, help="输出 JSON 指令文件")
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_dir = os.path.join(script_dir, "..", "vision", "test_data", "ground_truth")

    if args.gt:
        inst, rw, rh = generate_instructions(args.gt)
        print_instructions(inst, rw, rh, os.path.basename(args.gt))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump({"rect_mm": [rw, rh], "instructions": inst}, f,
                          ensure_ascii=False, indent=2)

    elif args.all:
        files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".json")])
        all_inst = []
        for fn in files:
            inst, rw, rh = generate_instructions(os.path.join(gt_dir, fn))
            print_instructions(inst, rw, rh, fn)
            all_inst.append({"sample": fn, "rect_mm": [rw, rh], "instructions": inst})
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(all_inst, f, ensure_ascii=False, indent=2)

    else:
        # 默认：第一个样本
        files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".json")])
        if files:
            inst, rw, rh = generate_instructions(os.path.join(gt_dir, files[0]))
            print_instructions(inst, rw, rh, files[0])


if __name__ == "__main__":
    main()
