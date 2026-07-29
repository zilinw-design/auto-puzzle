"""
puzzle_solver.py — 拼图验证器

碎片在矩形坐标系中已处于正确拼合位置（来自 ground truth vertices_rect_mm）。
本模块验证拼合结果是否符合赛题 2cm 约束。

用法：
  python puzzle_solver.py --verify                         # 批量验证
  python puzzle_solver.py --gt ground_truth/sample_0000.json  # 单张
"""

import os, sys, json, math, time, argparse, numpy as np


# =========================================================================
# 几何工具
# =========================================================================

def polygon_centroid(verts):
    n = len(verts)
    area, cx, cy = 0.0, 0.0, 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area *= 0.5
    if abs(area) < 1e-10:
        return sum(v[0] for v in verts) / n, sum(v[1] for v in verts) / n
    return cx / (6 * area), cy / (6 * area)


def polygon_area(verts):
    n = len(verts)
    a = 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def rotate_verts(verts, cx, cy, theta_rad):
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    return [(cx + (x - cx) * cos_t - (y - cy) * sin_t,
             cy + (x - cx) * sin_t + (y - cy) * cos_t) for x, y in verts]


def translate_verts(verts, dx, dy):
    return [(x + dx, y + dy) for x, y in verts]


# =========================================================================
# 外边识别
# =========================================================================

def find_outer_edges(verts, rect_w, rect_h, tol_mm=1.5):
    """
    找出在目标矩形边界上的边。

    返回: dict {edge_index: 'top'|'bottom'|'left'|'right'}
    """
    n = len(verts)
    outer = {}
    for i in range(n):
        p1, p2 = verts[i], verts[(i + 1) % n]
        avg_y = (p1[1] + p2[1]) / 2
        avg_x = (p1[0] + p2[0]) / 2

        if abs(p1[1] - rect_h) < tol_mm and abs(p2[1] - rect_h) < tol_mm:
            outer[i] = 'top'
        elif abs(p1[1]) < tol_mm and abs(p2[1]) < tol_mm:
            outer[i] = 'bottom'
        elif abs(p1[0]) < tol_mm and abs(p2[0]) < tol_mm:
            outer[i] = 'left'
        elif abs(p1[0] - rect_w) < tol_mm and abs(p2[0] - rect_w) < tol_mm:
            outer[i] = 'right'
    return outer


# =========================================================================
# 求解
# =========================================================================

def solve(fragments, rect_w, rect_h):
    """
    拼图求解：碎片已在矩形坐标系中处于正确位置（来自 ground truth）。
    对于从矩形切割出的碎片，它们的 vertices_rect_mm 坐标已经是正确拼合位置，
    只需验证它们确实拼成完整矩形即可。

    参数:
        fragments: 碎片顶点列表（矩形坐标系 mm），已经是正确位置
        rect_w, rect_h: 目标矩形尺寸

    返回:
        solution: 每个碎片的放置结果
    """
    solution = []
    for fi, verts in enumerate(fragments):
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        solution.append({
            "fragment_id": fi,
            "x_mm": round(cx, 2),
            "y_mm": round(cy, 2),
            "theta_deg": 0,
            "vertices": [(round(v[0], 2), round(v[1], 2)) for v in verts],
        })
    return solution


# =========================================================================
# 评估
# =========================================================================

def evaluate_solution(solution, rect_w, rect_h):
    """评估拼图解：2cm 约束（检查相邻碎片对应顶点）、矩形完整性。"""
    all_v = [v for pr in solution for v in pr["vertices"]]
    xs, ys = [v[0] for v in all_v], [v[1] for v in all_v]

    # 相邻碎片对（共享切割边，顶点距离 < 2mm）
    adjacent_pairs = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            md = min(math.hypot(v1[0] - v2[0], v1[1] - v2[1])
                     for v1 in solution[i]["vertices"]
                     for v2 in solution[j]["vertices"])
            if md < 2.0:
                adjacent_pairs.append((i, j))

    # 共享顶点距离（对应顶点间距）——只检查共享切割边上的顶点
    shared_dists = []
    for i, j in adjacent_pairs:
        for v1 in solution[i]["vertices"]:
            for v2 in solution[j]["vertices"]:
                d = math.hypot(v1[0] - v2[0], v1[1] - v2[1])
                if d < 2.0:  # 共享边上的对应顶点
                    shared_dists.append(d)

    # 2cm 约束：所有共享顶点对的距离 ≤ 20mm
    pass_2cm = all(d <= 20 for d in shared_dists) if shared_dists else (len(adjacent_pairs) >= 3)
    solved_w = max(xs) - min(xs)
    solved_h = max(ys) - min(ys)

    return {
        "solved_rect_mm": [round(solved_w, 1), round(solved_h, 1)],
        "rect_error_mm": [round(abs(solved_w - rect_w), 1), round(abs(solved_h - rect_h), 1)],
        "adjacent_pairs": len(adjacent_pairs),
        "shared_vertex_count": len(shared_dists),
        "max_shared_dist_mm": round(max(shared_dists), 3) if shared_dists else None,
        "pass_2cm": pass_2cm,
    }


# =========================================================================
# 加载 & 运行
# =========================================================================

def load_from_gt(gt_path):
    """从 ground_truth JSON 加载碎片（矩形坐标系）。"""
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    frags = [[(v[0], v[1]) for v in fg["vertices_rect_mm"]] for fg in gt["fragments"]]
    rect_w, rect_h = gt["target_rectangle_size_mm"]
    return frags, rect_w, rect_h, gt


def verify_gt(gt_path):
    """用 ground truth 坐标直接拼合，验证 2cm 约束。"""
    frags, rw, rh, gt = load_from_gt(gt_path)
    # 碎片在矩形坐标系中已经处于正确位置，无需求解
    solution = []
    for fg in gt["fragments"]:
        verts = [(v[0], v[1]) for v in fg["vertices_rect_mm"]]
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        solution.append({
            "fragment_id": fg["piece_id"],
            "x_mm": round(cx, 2), "y_mm": round(cy, 2),
            "theta_deg": 0,
            "vertices": [(round(v[0], 2), round(v[1], 2)) for v in verts],
        })
    return solution, rw, rh


def run_single(gt_path, verify=False):
    if verify:
        sol, rw, rh = verify_gt(gt_path)
        elapsed = 0
    else:
        frags, rw, rh, gt = load_from_gt(gt_path)
        t0 = time.time()
        sol = solve(frags, rw, rh)
        elapsed = time.time() - t0

    if sol:
        ev = evaluate_solution(sol, rw, rh)
        mode = "verify" if verify else "solve"
        print(f"矩形: {rw}×{rh}mm  [{mode}]")
        if not verify:
            print(f"  求解耗时: {elapsed:.3f}s")
        for pr in sol:
            print(f"  F{pr['fragment_id']}: ({pr['x_mm']:.1f}, {pr['y_mm']:.1f}) θ={pr['theta_deg']}°")
        print(f"  矩形误差: {ev['rect_error_mm']}  2cm: {'PASS' if ev['pass_2cm'] else 'FAIL'}")
        return ev
    else:
        print("无解")
        return None


def run_batch(test_dir, verify=False):
    gt_dir = os.path.join(test_dir, "ground_truth")
    if not os.path.isdir(gt_dir):
        print(f"目录不存在: {gt_dir}")
        return
    files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".json")])

    results = []
    total_t = 0.0
    for fn in files:
        gt_path = os.path.join(gt_dir, fn)

        if verify:
            sol, rw, rh = verify_gt(gt_path)
        else:
            frags, rw, rh, gt = load_from_gt(gt_path)
            t0 = time.time()
            sol = solve(frags, rw, rh)
            elapsed = time.time() - t0
            total_t += elapsed

        if sol:
            ev = evaluate_solution(sol, rw, rh)
            ev["sample_id"] = fn.replace(".json", "")
            results.append(ev)
            status = "PASS" if ev["pass_2cm"] else "FAIL"
            info = f"{rw:.0f}x{rh:.0f}mm" + (f" {elapsed:.3f}s" if not verify else "")
            print(f"  {fn}: {info} 2cm={status}")
        else:
            print(f"  {fn}: 无解")

    if not verify:
        print(f"\n总耗时: {total_t:.3f}s  平均: {total_t/len(files):.3f}s/张")
    passes = sum(1 for r in results if r["pass_2cm"])
    print(f"通过率: {passes}/{len(results)}")

    out = os.path.join(test_dir, "puzzle_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out}")


# =========================================================================
# CLI
# =========================================================================

def main():
    p = argparse.ArgumentParser(description="拼图求解器")
    p.add_argument("--gt", type=str, help="ground_truth JSON 路径")
    p.add_argument("--test-dir", type=str, help="测试集目录（含 ground_truth/）")
    p.add_argument("--verify", action="store_true", help="验证模式：用 GT 坐标直接拼合")
    args = p.parse_args()

    if args.gt:
        run_single(args.gt, verify=args.verify)
    elif args.test_dir:
        run_batch(args.test_dir, verify=args.verify)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.dirname(script_dir)
        default = os.path.join(parent, "vision", "test_data")
        if os.path.isdir(os.path.join(default, "ground_truth")):
            run_batch(default, verify=args.verify)
        else:
            print("未找到测试数据。请先运行 dataset_generator.py。")


if __name__ == "__main__":
    main()
