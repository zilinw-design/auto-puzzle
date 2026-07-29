"""
dataset_generator.py — 电赛 E 题拼图数据集生成器

按赛题要求生成训练和验证数据：
  - 竖直 A4 纸 (210×297mm)，上半部分随机放置碎片，下半部分留空
  - 碎片按赛题图 2 精确尺寸 或 随机自定义裁切 (9-12cm × 5-9cm 目标矩形)
  - 每个碎片 ≤5 条边、每条边 ≥2cm、至少 1 条边在矩形外边界上
  - 碎片不重叠、不触碰 A4 边界和分界线
  - 纯色碎片 + 深色/浅色背景 + 分界线

输出：
  - YOLOv8-seg 标注 (.txt，归一化坐标)
  - ground_truth JSON（mm 坐标，用于拼图算法验证）
  - 预览图像 (.png)

用法：
  python dataset_generator.py                        # 默认 50 张，Fig2 精确模式
  python dataset_generator.py --num 200              # 生成 200 张
  python dataset_generator.py --mode custom          # 随机矩形裁切模式
  python dataset_generator.py --seed 42 --num 10     # 固定种子，可复现
  python dataset_generator.py --preview              # 只生成一张预览图并显示

依赖：
  pip install opencv-python numpy shapely
"""

import os
import math
import random
import json
import argparse
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


class PuzzleDatasetGenerator:
    """电赛拼图数据集生成器。"""

    def __init__(self, scale=5.0, seed=None):
        """
        :param scale: 像素/毫米比例 (默认 5 px/mm，A4 = 1050×1485 px)
        :param seed: 随机种子（None = 不固定）
        """
        self.scale = scale
        self.a4_w_mm = 210.0
        self.a4_h_mm = 297.0
        self.w_px = int(self.a4_w_mm * scale)
        self.h_px = int(self.a4_h_mm * scale)
        self.divider_y_mm = self.a4_h_mm / 2.0
        self.rng = random.Random(seed)

        # 输出目录（相对于项目根目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(script_dir, "test_data")

    # =========================================================================
    # 碎片生成：赛题图 2 精确尺寸
    # =========================================================================

    def get_fig2_exact_pieces(self):
        """
        精确按赛题【图 2】尺寸生成 4 个碎片 (mm 坐标)。
        目标矩形: 100mm × 60mm。
        裁切方式：
          - 主斜线: (20, 0) → (100, 60), 长度 100mm
          - 分支线 1: (0, 20) → 主斜线上距起点 20mm 处 P1(36, 12)
          - 分支线 2: (0, 30) → 主斜线上距终点 30mm 处 P2(76, 42)
        每个碎片顶点为 CCW 顺序。
        """
        p1 = [(0, 0), (20, 0), (36, 12), (0, 20)]            # 左上 4 边
        p2 = [(0, 20), (36, 12), (76, 42), (0, 30)]          # 左中 4 边
        p3 = [(0, 30), (76, 42), (100, 60), (0, 60)]          # 左下 4 边
        p4 = [(20, 0), (100, 0), (100, 60)]                   # 右上 3 边
        return [p1, p2, p3, p4]

    # =========================================================================
    # 碎片生成：随机自定义裁切（现场测评 9×5cm ~ 12×9cm）
    # =========================================================================

    def get_custom_cut_pieces(self, rect_w_mm=None, rect_h_mm=None):
        """
        按赛题图 2 裁切风格生成随机尺寸矩形的拼图碎片。
        赛题约束：每个碎片 ≤5 条边、每条边 ≥2cm、至少 1 条边在矩形外边界上。

        :param rect_w_mm: 目标矩形宽度 (mm)，None 则随机 90-120
        :param rect_h_mm: 目标矩形高度 (mm)，None 则随机 50-90
        """
        MIN_EDGE = 20.0  # 赛题要求每条边 ≥ 2cm

        if rect_w_mm is None:
            rect_w_mm = self.rng.uniform(90, 120)
        if rect_h_mm is None:
            rect_h_mm = self.rng.uniform(50, 90)

        # 裁切参数：边界切割点位置，全部 clamp 保证每条边 ≥ MIN_EDGE
        # top_cut: 顶边切割点 x 坐标，需 ≥ 20mm 且留够余量
        top_cut = max(MIN_EDGE, rect_w_mm * self.rng.uniform(0.22, 0.38))
        top_cut = min(top_cut, rect_w_mm - MIN_EDGE)

        # left_cut1: 左边下切割点 y 坐标（靠下），需 ≥ 20mm
        left_cut1 = max(MIN_EDGE, rect_h_mm * self.rng.uniform(0.30, 0.45))

        # left_cut2: 左边上切割点 y 坐标（靠上），需 ≥ left_cut1+20 且 ≤ h-20
        left_cut2 = max(left_cut1 + MIN_EDGE,
                        rect_h_mm * self.rng.uniform(0.55, 0.75))
        left_cut2 = min(left_cut2, rect_h_mm - MIN_EDGE)

        # 主斜线方向：(top_cut, 0) → (rect_w_mm, rect_h_mm)
        dx = rect_w_mm - top_cut
        dy = rect_h_mm
        main_len = math.hypot(dx, dy)
        ux, uy = dx / main_len, dy / main_len

        # 主斜线上两个分支交点，与两端各留 ≥ 20mm
        d1 = max(MIN_EDGE, main_len * self.rng.uniform(0.18, 0.30))
        d2 = min(main_len - MIN_EDGE, main_len * self.rng.uniform(0.68, 0.82))

        pmid1 = (top_cut + ux * d1, uy * d1)
        pmid2 = (top_cut + ux * d2, uy * d2)

        p1 = [(0, 0), (top_cut, 0), pmid1, (0, left_cut1)]
        p2 = [(0, left_cut1), pmid1, pmid2, (0, left_cut2)]
        p3 = [(0, left_cut2), pmid2, (rect_w_mm, rect_h_mm), (0, rect_h_mm)]
        p4 = [(top_cut, 0), (rect_w_mm, 0), (rect_w_mm, rect_h_mm)]

        pieces = [p1, p2, p3, p4]

        # 后验证：每条边 ≥ MIN_EDGE，不满足则返回 None（调用方重试）
        if not self._validate_all_edges(pieces, MIN_EDGE):
            return None

        return pieces

    # =========================================================================
    # 验证
    # =========================================================================

    @staticmethod
    def _validate_all_edges(pieces, min_edge_mm=20.0):
        """验证所有碎片的所有边 ≥ min_edge_mm。"""
        for poly in pieces:
            n = len(poly)
            for i in range(n):
                p1, p2 = poly[i], poly[(i + 1) % n]
                if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < min_edge_mm - 0.5:
                    return False
        return True

    # =========================================================================
    # 画布与渲染
    # =========================================================================

    def create_a4_canvas(self, bg_color=(235, 206, 180), line_color=(0, 0, 0), line_width_mm=3):
        """
        生成 A4 画布并绘制中间分界线 (BGR 格式)。

        :param bg_color: 画布颜色 (BGR)，默认淡蓝/浅灰
        :param line_width_mm: 分界线宽度 (实线，≤5mm)
        """
        img = np.full((self.h_px, self.w_px, 3), bg_color, dtype=np.uint8)
        div_y_px = int(self.divider_y_mm * self.scale)
        lw_px = max(1, int(line_width_mm * self.scale))
        cv2.line(img, (0, div_y_px), (self.w_px, div_y_px), line_color, lw_px)
        return img

    # =========================================================================
    # 碎片放置
    # =========================================================================

    def place_pieces_randomly_upper(self, raw_pieces_mm, margin_mm=8, min_gap_mm=3,
                                     max_attempts=1000):
        """
        将碎片随机旋转并无重叠地放置在 A4 纸上半区域。
        碎片之间保持至少 min_gap_mm 间距，防止边界粘连。

        :param min_gap_mm: 碎片间最小间距 (mm)，太近会导致轮廓粘连
        :return: list of dict，每个 dict 包含 piece_id, polygon_shapely, pts_px, pts_mm
        :return: None 表示放置失败（调用方应重试）
        """
        placed_info = []
        placed_polygons = []

        min_x, max_x = margin_mm, self.a4_w_mm - margin_mm
        min_y, max_y = margin_mm, self.divider_y_mm - margin_mm

        for idx, pts in enumerate(raw_pieces_mm):
            poly_orig = Polygon(pts)
            centroid = poly_orig.centroid
            poly_centered = translate(poly_orig, -centroid.x, -centroid.y)

            placed = False
            for _ in range(max_attempts):
                angle = self.rng.uniform(0, 360)
                rotated_poly = rotate(poly_centered, angle, origin=(0, 0))

                minx, miny, maxx, maxy = rotated_poly.bounds
                poly_w = maxx - minx
                poly_h = maxy - miny

                if poly_w >= (max_x - min_x) or poly_h >= (max_y - min_y):
                    continue

                cx = self.rng.uniform(min_x + poly_w / 2.0, max_x - poly_w / 2.0)
                cy = self.rng.uniform(min_y + poly_h / 2.0, max_y - poly_h / 2.0)

                candidate_poly = translate(rotated_poly, cx, cy)

                has_overlap = any(candidate_poly.intersects(p) for p in placed_polygons)
                too_close = any(candidate_poly.distance(p) < min_gap_mm for p in placed_polygons)
                if has_overlap or too_close:
                    continue

                placed_polygons.append(candidate_poly)
                coords_mm = np.array(candidate_poly.exterior.coords)[:-1]
                coords_px = (coords_mm * self.scale).astype(np.int32)

                placed_info.append({
                    'piece_id': idx,
                    'polygon_shapely': candidate_poly,
                    'pts_px': coords_px,
                    'pts_mm': coords_mm,              # A4 纸坐标系中的 mm 坐标
                })
                placed = True
                break

            if not placed:
                return None

        return placed_info

    # =========================================================================
    # Ground Truth 构建
    # =========================================================================

    def build_ground_truth(self, sample_id, raw_pieces_mm, placed_info, rect_w_mm, rect_h_mm):
        """
        构建 ground_truth JSON，包含每个碎片的原始形状和在图像中的位置。

        注意：raw_pieces_mm 中的坐标是矩形坐标系（左下角原点），
        placed_info 中的 pts_mm 是 A4 纸坐标系（放置后）。
        """
        fragments = []
        for item in placed_info:
            idx = item['piece_id']
            raw_verts = raw_pieces_mm[idx]
            # 原始矩形中的碎片顶点
            raw_poly = Polygon(raw_verts)

            fragments.append({
                "piece_id": idx,
                "vertices_rect_mm": [[round(v[0], 2), round(v[1], 2)] for v in raw_verts],
                "num_edges": len(raw_verts),
                "area_rect_mm2": round(raw_poly.area, 2),
                # 在 A4 纸上的放置多边形
                "vertices_placed_mm": [[round(v[0], 2), round(v[1], 2)] for v in item['pts_mm']],
                # YOLO 归一化坐标 (方便查)
                "vertices_norm": [[round(v[0] / self.a4_w_mm, 6), round(v[1] / self.a4_h_mm, 6)]
                                  for v in item['pts_mm']],
            })

        return {
            "sample_id": sample_id,
            "image_size_px": [self.w_px, self.h_px],
            "a4_size_mm": [self.a4_w_mm, self.a4_h_mm],
            "scale_px_per_mm": self.scale,
            "divider_y_mm": self.divider_y_mm,
            "target_rectangle_size_mm": [round(rect_w_mm, 2), round(rect_h_mm, 2)],
            "num_pieces": len(raw_pieces_mm),
            "fragments": fragments,
        }

    # =========================================================================
    # 批量生成
    # =========================================================================

    def render_and_generate_dataset(self, num_samples=100, use_fig2_exact=True):
        """
        批量生成图像训练集和标注文件。

        输出（在 self.output_dir 下）：
          images/         — PNG 图像
          labels/         — YOLOv8-seg .txt 标注
          ground_truth/   — 拼图算法验证用 JSON
        """
        img_dir = os.path.join(self.output_dir, "images")
        lbl_dir = os.path.join(self.output_dir, "labels")
        gt_dir = os.path.join(self.output_dir, "ground_truth")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        print(f"开始生成数据集，共计 {num_samples} 张...")
        print(f"模式: {'Fig2 精确尺寸' if use_fig2_exact else '随机自定义裁切'}")
        print(f"输出目录: {self.output_dir}")

        success_count = 0
        while success_count < num_samples:
            # 1. 生成碎片
            if use_fig2_exact:
                raw_pieces = self.get_fig2_exact_pieces()
                rect_w, rect_h = 100.0, 60.0   # Fig2 固定尺寸
            else:
                # 随机自定义裁切：边长不达标时自动重试
                raw_pieces = None
                for _ in range(50):
                    rect_w = self.rng.uniform(90, 120)
                    rect_h = self.rng.uniform(50, 90)
                    raw_pieces = self.get_custom_cut_pieces(rect_w, rect_h)
                    if raw_pieces is not None:
                        break
                if raw_pieces is None:
                    continue  # 50 次重试都失败，跳过这轮

            # 2. 赛题固定配色
            #    碎片：纯黄色（BGR = 青+品红=0, 黄=255）→ (0, 255, 255)
            #    背景：白色 (255, 255, 255)
            #    分界线：黑色 (0, 0, 0)，宽度 3mm（≤5mm 符合赛题）
            bg_color = (255, 255, 255)      # 白色背景
            piece_color = (0, 255, 255)     # 纯黄色碎片 (BGR: 蓝=0, 绿=255, 红=255)
            line_color = (0, 0, 0)          # 黑色分界线
            line_width = 3.0                # 3mm 宽

            canvas = self.create_a4_canvas(
                bg_color=bg_color, line_color=line_color, line_width_mm=line_width
            )

            # 3. 随机无重叠放置
            placed_info = self.place_pieces_randomly_upper(raw_pieces)
            if placed_info is None:
                continue

            # 4. 绘制碎片
            label_lines = []
            for item in placed_info:
                pts_px = item['pts_px']
                cv2.fillPoly(canvas, [pts_px], piece_color)
                cv2.polylines(canvas, [pts_px], isClosed=True, color=(0, 0, 0), thickness=1)

                # YOLO 归一化标注
                norm_vals = []
                for pt in item['pts_mm']:
                    norm_vals.extend([
                        f"{pt[0] / self.a4_w_mm:.6f}",
                        f"{pt[1] / self.a4_h_mm:.6f}",
                    ])
                label_lines.append("0 " + " ".join(norm_vals))

            # 5. 保存
            prefix = f"sample_{success_count:04d}"
            cv2.imwrite(os.path.join(img_dir, f"{prefix}.png"), canvas)
            with open(os.path.join(lbl_dir, f"{prefix}.txt"), "w") as f:
                f.write("\n".join(label_lines))

            gt = self.build_ground_truth(prefix, raw_pieces, placed_info, rect_w, rect_h)
            with open(os.path.join(gt_dir, f"{prefix}.json"), "w", encoding="utf-8") as f:
                json.dump(gt, f, ensure_ascii=False, indent=2)

            success_count += 1
            if success_count % 20 == 0:
                print(f"  已完成: {success_count}/{num_samples}")

        # 汇总文件
        summary = {
            "mode": "fig2_exact" if use_fig2_exact else "custom_cut",
            "num_samples": num_samples,
            "image_size_px": [self.w_px, self.h_px],
            "a4_size_mm": [self.a4_w_mm, self.a4_h_mm],
            "scale_px_per_mm": self.scale,
        }
        with open(os.path.join(self.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"数据集生成完毕！共 {num_samples} 张 → {self.output_dir}")


# =========================================================================
# 命令行入口
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="电赛 E 题拼图数据集生成器 — Fig2 精确尺寸 / 随机裁切"
    )
    parser.add_argument("--num", type=int, default=50,
                        help="生成数量 (默认 50)")
    parser.add_argument("--mode", type=str, default="fig2",
                        choices=["fig2", "custom"],
                        help="fig2=赛题图2精确尺寸, custom=随机自定义裁切")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (固定后结果可复现)")
    parser.add_argument("--preview", action="store_true",
                        help="生成单张预览图并保存（不显示窗口）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认 src/vision/test_data/)")

    args = parser.parse_args()

    generator = PuzzleDatasetGenerator(scale=5.0, seed=args.seed)

    if args.output:
        generator.output_dir = args.output

    if args.preview:
        # 单张预览模式
        os.makedirs(generator.output_dir, exist_ok=True)
        raw = generator.get_fig2_exact_pieces() if args.mode == "fig2" else generator.get_custom_cut_pieces()
        rect_w, rect_h = (100.0, 60.0) if args.mode == "fig2" else (100.0, 70.0)

        canvas = generator.create_a4_canvas(bg_color=(255, 255, 255), line_color=(0, 0, 0))
        placed = generator.place_pieces_randomly_upper(raw)
        if placed:
            piece_color = (0, 255, 255)  # 纯黄色
            for item in placed:
                cv2.fillPoly(canvas, [item['pts_px']], piece_color)
                cv2.polylines(canvas, [item['pts_px']], True, (0, 0, 0), 2)

            preview_path = os.path.join(generator.output_dir, "preview.png")
            cv2.imwrite(preview_path, canvas)
            print(f"预览图已保存: {preview_path}")

            gt = generator.build_ground_truth("preview", raw, placed, rect_w, rect_h)
            gt_path = os.path.join(generator.output_dir, "preview_ground_truth.json")
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt, f, ensure_ascii=False, indent=2)
            print(f"Ground truth 已保存: {gt_path}")
        else:
            print("预览生成失败：碎片无法放置，请重试")
        return

    # 批量生成
    use_fig2 = (args.mode == "fig2")
    generator.render_and_generate_dataset(num_samples=args.num, use_fig2_exact=use_fig2)


if __name__ == "__main__":
    main()
