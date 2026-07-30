"""
puzzle_assembler.py — 拼图盲解求解器

从参考实现 puzzle_sim.py 提取的核心算法：
  边长配对 → 匹配集合构建 → 刚体变换传播 → 全局优化 → 评分选最优

输入: 4个碎片的多边形顶点列表 + 目标矩形尺寸
输出: 4个3×3齐次变换矩阵，描述每片从当前位置到矩形中正确位置的变换

第一版：保守策略。12%单档容差，无宽松匹配，无凹多边形支持。
"""

import itertools, math, time, numpy as np


def edges(poly):
    """多边形 → 边列表 [(p1,p2), (p2,p3), ...]"""
    return [(poly[i], poly[(i+1) % len(poly)]) for i in range(len(poly))]


def rigid(angle, tx, ty):
    """生成3×3齐次刚体变换矩阵。"""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])


def apply_h(points, h):
    """对点集应用齐次变换。"""
    q = np.c_[points, np.ones(len(points))] @ h.T
    return q[:, :2] / q[:, 2, None]


def align_edge(src_a, src_b, dst_a, dst_b):
    """计算让边(src_a→src_b)对齐到边(dst_a→dst_b)的刚体变换。"""
    u, v = src_b - src_a, dst_b - dst_a
    angle = math.atan2(v[1], v[0]) - math.atan2(u[1], u[0])
    r = rigid(angle, 0, 0)
    mapped = apply_h(np.array([src_a]), r)[0]
    r[:2, 2] = dst_a - mapped
    return r


# =========================================================================
# 边长配对
# =========================================================================

def candidate_matchings(pieces, tolerance=0.12):
    """遍历所有碎片的所有边对，按边长相似度排序。"""
    all_edges = {(i, e): edge for i, p in enumerate(pieces)
                 for e, edge in enumerate(edges(p))}
    candidates = []
    for (i, ei), (j, ej) in itertools.combinations(all_edges, 2):
        if i == j:
            continue
        a, b = all_edges[(i, ei)]
        c, d = all_edges[(j, ej)]
        la, lb = np.linalg.norm(b - a), np.linalg.norm(d - c)
        rel = abs(la - lb) / max(la, lb)
        if rel < tolerance:
            candidates.append((rel, i, ei, j, ej, 0.0, 1.0, 0.0, 1.0))
    candidates.sort()
    return candidates[:80]


# =========================================================================
# 匹配集合
# =========================================================================

def match_segments(pieces, match):
    """从候选匹配中提取两条边的实际端点。"""
    _, i, ei, j, ej, ia0, ia1, ja0, ja1 = match
    a, b = edges(pieces[i])[ei]
    c, d = edges(pieces[j])[ej]
    return (a + (b - a) * ia0, a + (b - a) * ia1,
            c + (d - c) * ja0, c + (d - c) * ja1)


def matching_sets(pieces):
    """从候选配对构建连通匹配集合。每个集合有N-1对匹配。"""
    count = len(pieces)
    if count == 1:
        yield ()
        return
    cand = candidate_matchings(pieces)
    full = [m for m in cand if tuple(m[5:]) == (0.0, 1.0, 0.0, 1.0)]
    pair_count = count - 1
    for combo in itertools.combinations(full, pair_count):
        used, degree = set(), [0] * count
        ok = True
        graph = [set() for _ in range(count)]
        for match in combo:
            _, i, ei, j, ej = match[:5]
            if (i, ei) in used or (j, ej) in used:
                ok = False; break
            used |= {(i, ei), (j, ej)}
            degree[i] += 1; degree[j] += 1
            graph[i].add(j); graph[j].add(i)
        if not ok or any(d == 0 for d in degree):
            continue
        seen, stack = {0}, [0]
        while stack:
            for j in graph[stack.pop()]:
                if j not in seen:
                    seen.add(j); stack.append(j)
        if len(seen) == count:
            yield combo


# =========================================================================
# 刚体变换传播
# =========================================================================

def assemble_from_matches(pieces, matches):
    """匹配集合 → 刚体变换传播 → 每片的初步位置。"""
    adjacency = [[] for _ in pieces]
    for match in matches:
        _, i, _ei, j, _ej = match[:5]
        adjacency[i].append((j, match, False))
        adjacency[j].append((i, match, True))
    transforms = [None] * len(pieces)
    transforms[0] = np.eye(3)
    stack = [0]
    while stack:
        i = stack.pop()
        for j, match, rev in adjacency[i]:
            ia, ib, ja, jb = match_segments(pieces, match)
            if rev:
                ia, ib, ja, jb = ja, jb, ia, ib
            wa, wb = apply_h(np.array([ia, ib]), transforms[i])
            proposed = align_edge(ja, jb, wb, wa)
            if transforms[j] is None:
                transforms[j] = proposed; stack.append(j)
    return transforms


# =========================================================================
# 全局优化
# =========================================================================

def optimize_pose_graph(pieces, matches, initial):
    """Gauss-Newton全局优化：分摊闭环误差。"""
    if len(pieces) < 3:
        return initial

    def pack(poses):
        vals = []
        for h in poses[1:]:
            vals.extend([math.atan2(h[1, 0], h[0, 0]), h[0, 2], h[1, 2]])
        return np.asarray(vals, dtype=float)

    def unpack(x):
        poses = [initial[0]]
        for k in range(len(pieces) - 1):
            theta, tx, ty = x[3*k:3*k+3]
            poses.append(rigid(theta, tx, ty))
        return poses

    def residual(x):
        poses = unpack(x)
        vals = []
        for match in matches:
            ia, ib, ja, jb = match_segments(pieces, match)
            wi = apply_h(np.array([ia, ib]), poses[match[1]])
            wj = apply_h(np.array([jb, ja]), poses[match[3]])
            vals.extend((wi - wj).ravel())
        return np.asarray(vals)

    x = pack(initial)
    for _ in range(20):
        r0 = residual(x)
        jac = np.empty((len(r0), len(x)))
        for k in range(len(x)):
            step = 1e-5 if k % 3 == 0 else 1e-3
            shifted = x.copy(); shifted[k] += step
            jac[:, k] = (residual(shifted) - r0) / step
        delta, *_ = np.linalg.lstsq(jac, -r0, rcond=None)
        x += delta
        if np.linalg.norm(delta) < 1e-7:
            break
    return unpack(x)


# =========================================================================
# 评分
# =========================================================================

def score_solution(transforms, pieces, target_w, target_h):
    """
    评估拼图解质量：矩形填充率 + 外边覆盖率。
    分数越低越好。
    """
    all_placed = []
    for p, h in zip(pieces, transforms):
        placed = apply_h(p, h)
        all_placed.append(placed)

    # 矩形填充率：碎片覆盖了多少目标矩形面积
    canvas = np.zeros((int(target_h) + 4, int(target_w) + 4), dtype=np.uint8)
    for placed in all_placed:
        pts = np.round(placed).astype(np.int32)
        if len(pts) >= 3:
            cv2 = __import__('cv2')
            cv2.fillPoly(canvas, [pts.reshape((-1, 1, 2))], 255)
    fill = np.sum(canvas > 0) / (target_w * target_h)

    # 外边覆盖率：矩形的4条边界是否被覆盖
    outer_score = 0.0
    for placed in all_placed:
        for x, y in placed:
            d = min(x, target_w - x, y, target_h - y)
            if d < 3:
                outer_score += 1.0
    outer_coverage = outer_score / (len(pieces) * 4 + 1e-6)

    score = (1.0 - fill) * 50.0 + (1.0 - min(outer_coverage, 1.0)) * 30.0
    return score


# =========================================================================
# 顶层入口
# =========================================================================

def solve(pieces, target_w, target_h, timeout_s=5.0):
    """
    求解拼图。

    Args:
        pieces: list of (N,2) ndarray，碎片多边形顶点（像素坐标）
        target_w, target_h: 目标矩形尺寸（像素）
        timeout_s: 超时（秒）

    Returns:
        transforms: list of 4×3×3 ndarray，每片的齐次变换矩阵
        best_score: float
        match_count: 使用的匹配数量
        或 (None, inf, 0) 表示失败
    """
    t0 = time.time()
    best = None
    best_score = float('inf')
    best_match_count = 0

    for matches in matching_sets(pieces):
        if time.time() - t0 > timeout_s:
            break

        transforms = assemble_from_matches(pieces, matches)
        if any(t is None for t in transforms):
            continue

        transforms = optimize_pose_graph(pieces, matches, transforms)
        score = score_solution(transforms, pieces, target_w, target_h)

        if score < best_score:
            best_score = score
            best = transforms
            best_match_count = len(matches)

    return best, best_score, best_match_count
