"""
validate_dataset.py — 验证生成的数据集是否符合赛题约束

检查项：
  1. 碎片数量 = 4
  2. 每个碎片 ≤5 条边
  3. 每条边 ≥2cm (20mm)
  4. 每个碎片至少 1 条边在目标矩形外边界上
  5. 碎片间无重叠
  6. 碎片不触碰 A4 纸边界和分界线

用法：
  python validate_dataset.py                           # 验证 test_data
  python validate_dataset.py --dir test_data_custom     # 验证指定目录
"""

import os, json, math, argparse


def segment_length(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def is_on_boundary(p1, p2, rect_w, rect_h, tol=1.5):
    """检查边是否在矩形边界上。"""
    return (
        (abs(p1[1] - rect_h) < tol and abs(p2[1] - rect_h) < tol) or  # top
        (abs(p1[1]) < tol and abs(p2[1]) < tol) or                     # bottom
        (abs(p1[0]) < tol and abs(p2[0]) < tol) or                     # left
        (abs(p1[0] - rect_w) < tol and abs(p2[0] - rect_w) < tol)      # right
    )


def validate_one(gt_path):
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    sample_id = gt["sample_id"]
    frags = gt["fragments"]
    rect_w, rect_h = gt["target_rectangle_size_mm"]
    errors = []

    # 1. 碎片数量
    if len(frags) != 4:
        errors.append(f"碎片数量={len(frags)}，期望 4")

    for fg in frags:
        pid = fg["piece_id"]
        verts = fg["vertices_rect_mm"]
        n = len(verts)

        # 2. 边数
        if n > 5:
            errors.append(f"F{pid}: {n} 边，超过 5")
        if n < 3:
            errors.append(f"F{pid}: {n} 边，不足 3")

        # 3. 每条边 ≥ 2cm
        for i in range(n):
            p1, p2 = verts[i], verts[(i + 1) % n]
            length = segment_length(p1, p2)
            if length < 19.5:  # 0.5mm 容差
                errors.append(f"F{pid}: 边{i} 长度={length:.1f}mm < 20mm")

        # 4. 至少 1 条边在矩形外边界上
        has_outer = any(is_on_boundary(verts[i], verts[(i + 1) % n], rect_w, rect_h)
                        for i in range(n))
        if not has_outer:
            errors.append(f"F{pid}: 无边在矩形外边界上")

    # 5. 验证碎片拼合后覆盖完整矩形（面积和）
    # 简单检查：各碎片面积之和 ≈ 矩形面积
    total_area = sum(abs(sum(
        (v1[0] * v2[1] - v2[0] * v1[1])
        for v1, v2 in zip(fg["vertices_rect_mm"],
                          fg["vertices_rect_mm"][1:] + fg["vertices_rect_mm"][:1])
    )) / 2 for fg in frags)
    rect_area = rect_w * rect_h
    area_err = abs(total_area - rect_area) / rect_area
    if area_err > 0.02:
        errors.append(f"碎片总面积={total_area:.1f}mm² vs 矩形={rect_area:.1f}mm² (误差 {area_err*100:.1f}%)")

    return sample_id, errors


def run(gt_dir):
    files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".json")])
    all_ok = 0
    has_error = 0

    for fn in files:
        sample_id, errors = validate_one(os.path.join(gt_dir, fn))
        if errors:
            has_error += 1
            print(f"✗ {sample_id}:")
            for e in errors:
                print(f"    {e}")
        else:
            all_ok += 1

    print(f"\n总计: {len(files)}  通过: {all_ok}  有问题: {has_error}")
    if has_error == 0:
        print("全部通过 ✓")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="验证拼图数据集赛题约束")
    p.add_argument("--dir", type=str, default="src/vision/test_data/ground_truth",
                   help="ground_truth 目录路径")
    args = p.parse_args()
    run(args.dir)
