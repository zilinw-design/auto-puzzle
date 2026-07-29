import os, sys, json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def animate_to_web(gt_path, output_html="puzzle_animation.html"):
    if not os.path.exists(gt_path):
        print(f"❌ 找不到文件: {gt_path}")
        return

    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    rect_w, rect_h = gt["target_rectangle_size_mm"]
    fragments_gt = [[(v[0], v[1]) for v in fg["vertices_rect_mm"]] for fg in gt["fragments"]]
    num_frags = len(fragments_gt)

    targets = []
    for verts in fragments_gt:
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        targets.append({"verts": verts, "cx": cx, "cy": cy, "angle": 0.0})

    np.random.seed(42)
    starts = []
    for target in targets:
        offset_x = np.random.uniform(-rect_w * 0.8, rect_w * 1.8)
        offset_y = np.random.uniform(-rect_h * 0.8, rect_h * 1.8)
        if 0 <= offset_x <= rect_w and 0 <= offset_y <= rect_h:
            offset_x += rect_w * (1 if np.random.rand() > 0.5 else -1)
        starts.append({"cx": offset_x, "cy": offset_y, "angle": np.random.uniform(-math.pi, math.pi)})

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_aspect('equal')
    margin = max(rect_w, rect_h) * 0.6
    ax.set_xlim(-margin, rect_w + margin)
    ax.set_ylim(-margin, rect_h + margin)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f"拼图移动与验证 (目标: {rect_w}×{rect_h}mm)", fontsize=13)

    # 绘制目标框
    ax.add_patch(patches.Rectangle((0, 0), rect_w, rect_h, linewidth=2, edgecolor='red', facecolor='none', linestyle='--'))

    cmap = plt.get_cmap('tab10')
    patches_list, labels_list = [], []

    for idx in range(num_frags):
        color = cmap(idx / max(num_frags, 1))
        poly = patches.Polygon([(0,0)], closed=True, facecolor=color, edgecolor='black', alpha=0.8, linewidth=1.5)
        ax.add_patch(poly)
        patches_list.append(poly)
        txt = ax.text(0, 0, f"F{idx}", fontsize=10, fontweight='bold', ha='center', va='center', color='white',
                      bbox=dict(boxstyle='circle,pad=0.2', facecolor='black', alpha=0.6))
        labels_list.append(txt)

    total_frames = 50

    def update(frame):
        progress = 1 / (1 + math.exp(-10 * (frame / total_frames - 0.5)))
        for i in range(num_frags):
            st, tg = starts[i], targets[i]
            cur_cx = st["cx"] + (tg["cx"] - st["cx"]) * progress
            cur_cy = st["cy"] + (tg["cy"] - st["cy"]) * progress
            cur_angle = st["angle"] + (tg["angle"] - st["angle"]) * progress

            current_verts = []
            for vx, vy in tg["verts"]:
                lx, ly = vx - tg["cx"], vy - tg["cy"]
                rx = lx * math.cos(cur_angle) - ly * math.sin(cur_angle)
                ry = lx * math.sin(cur_angle) + ly * math.cos(cur_angle)
                current_verts.append((rx + cur_cx, ry + cur_cy))

            patches_list[i].set_xy(current_verts)
            labels_list[i].set_position((cur_cx, cur_cy))

        return patches_list + labels_list

    anim = FuncAnimation(fig, update, frames=total_frames + 10, interval=40, blit=True)

    # 导出为交互式 HTML 网页
    print(f"⌛ 正在生成网页 HTML...")
    html_content = anim.to_jshtml()
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    plt.close(fig) # 释放内存
    print(f"🎉 网页文件生成成功！路径：{os.path.abspath(output_html)}")

if __name__ == "__main__":
    # 自动找 ground_truth 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_dir = os.path.join(script_dir, "..", "vision", "test_data", "ground_truth")
    gt_path = os.path.join(gt_dir, "sample_0000.json")
    animate_to_web(gt_path)