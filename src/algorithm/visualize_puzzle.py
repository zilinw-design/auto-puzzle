"""
visualize_puzzle.py — 拼图结果可视化（期望 vs 求解）

左侧：Ground Truth 正确拼合   右侧：求解器输出

用法：
  python visualize_puzzle.py                             # 第一个样本
  python visualize_puzzle.py --gt sample_0000.json       # 指定样本
  python visualize_puzzle.py --all                        # 全部
"""

import os, sys, json, math, argparse
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from puzzle_solver import solve, evaluate_solution, load_from_gt

SCALE = 5.0
COLORS = [(0, 180, 0), (0, 180, 255), (255, 0, 180), (180, 255, 0)]


def draw_solution(verts_list, rect_w, rect_h, labels, title, pass_2cm=None):
    """绘制拼图结果。"""
    w_px = int(rect_w * SCALE) + 40
    h_px = int(rect_h * SCALE) + 70
    img = np.full((h_px, w_px, 3), (245, 245, 245), dtype=np.uint8)

    # 矩形外框
    m = 20
    pts_r = np.array([[m, m], [m + int(rect_w * SCALE), m],
                      [m + int(rect_w * SCALE), m + int(rect_h * SCALE)],
                      [m, m + int(rect_h * SCALE)]], np.int32)
    cv2.polylines(img, [pts_r.reshape((-1, 1, 2))], True, (0, 0, 0), 2)

    for i, verts in enumerate(verts_list):
        c = COLORS[i % 4]
        pts = np.array([(m + int(v[0] * SCALE), m + int((rect_h - v[1]) * SCALE))
                        for v in verts], np.int32)
        cv2.fillPoly(img, [pts.reshape((-1, 1, 2))], (*c, 80))
        cv2.polylines(img, [pts.reshape((-1, 1, 2))], True, c, 2)

        # 碎片 ID
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        cv2.putText(img, labels[i] if labels else f"F{i}",
                    (m + int(cx * SCALE) - 10, m + int((rect_h - cy) * SCALE) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # 标题 + 尺寸
    status = ""
    if pass_2cm is not None:
        status = "  [PASS]" if pass_2cm else "  [FAIL]"
    cv2.putText(img, f"{title}  {rect_w:.0f}x{rect_h:.0f}mm{status}",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return img


def visualize_one(gt_path, out_path=None):
    """生成一张对比图：左=期望，右=求解结果。"""
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    sample_id = gt["sample_id"]
    rect_w, rect_h = gt["target_rectangle_size_mm"]

    # 左侧：Ground Truth
    gt_verts = [[(v[0], v[1]) for v in fg["vertices_rect_mm"]] for fg in gt["fragments"]]
    gt_labels = [f"F{fg['piece_id']}" for fg in gt["fragments"]]
    img_gt = draw_solution(gt_verts, rect_w, rect_h, gt_labels, f"{sample_id}  GT (Expected)")

    # 右侧：求解器输出
    frags = [[(v[0], v[1]) for v in fg["vertices_rect_mm"]] for fg in gt["fragments"]]
    sol = solve(frags, rect_w, rect_h)
    solver_verts = [pr["vertices"] for pr in sol] if sol else gt_verts
    solver_labels = [f"F{pr['fragment_id']}" for pr in sol] if sol else gt_labels

    pass_2cm = None
    if sol:
        ev = evaluate_solution(sol, rect_w, rect_h)
        pass_2cm = ev["pass_2cm"]

    img_solver = draw_solution(solver_verts, rect_w, rect_h, solver_labels,
                               f"{sample_id}  Solver Output", pass_2cm)

    # 左右拼接
    h = max(img_gt.shape[0], img_solver.shape[0])
    w = img_gt.shape[1] + img_solver.shape[1] + 10
    combined = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
    combined[:img_gt.shape[0], :img_gt.shape[1]] = img_gt
    combined[:img_solver.shape[0], img_gt.shape[1] + 10:] = img_solver

    if out_path is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "vision", "test_data", "puzzle_viz")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{sample_id}_compare.png")

    cv2.imwrite(out_path, combined)
    return out_path, pass_2cm


def main():
    p = argparse.ArgumentParser(description="拼图可视化 — 期望 vs 求解")
    p.add_argument("--gt", type=str, help="ground_truth JSON")
    p.add_argument("--all", action="store_true")
    p.add_argument("--output", type=str)
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_dir = os.path.join(script_dir, "..", "vision", "test_data", "ground_truth")
    out_dir = os.path.join(script_dir, "..", "vision", "test_data", "puzzle_viz")
    os.makedirs(out_dir, exist_ok=True)

    if args.gt:
        visualize_one(args.gt, args.output)
        return

    if args.all or True:
        files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".json")])
        passes = 0
        for fn in files:
            path, ok = visualize_one(os.path.join(gt_dir, fn))
            if ok:
                passes += 1
        print(f"共 {len(files)} 张 → {out_dir}  通过: {passes}/{len(files)}")
        return

    files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".json")])
    if files:
        visualize_one(os.path.join(gt_dir, files[0]), args.output)


if __name__ == "__main__":
    main()
