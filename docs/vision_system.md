# 拼图视觉识别系统 — 实现文档

> 2026 全国大学生电子设计竞赛 E 题：拼图装置  
> 最后更新：2026-07-29

---

## 一、系统架构

```
摄像头 (4K USB, /dev/video0, MJPG 1280×720)
    │
    ▼
┌─────────────────────────────────────────────┐
│  1. 摄像头自动控制                           │
│     AE (auto_exposure=3) + AWB (auto)       │
│     光照变化自动适应，无需手动调参             │
└──────────────────┬──────────────────────────┘
                   │ BGR 帧
┌──────────────────┴──────────────────────────┐
│  2. A4 纸 ROI 定位 + 透视矫正                │
│     Canny → findContours → 最大四边形         │
│     → getPerspectiveTransform → 标准俯视图    │
│     找不到纸时退回原图，状态栏显示 Raw         │
└──────────────────┬──────────────────────────┘
                   │ 俯视图 (210×297mm @ 3px/mm)
┌──────────────────┴──────────────────────────┐
│  3. ROI 亮度自适应 Gamma 校正                │
│     计算上半区平均亮度                        │
│     太暗 → Gamma < 1 提亮                     │
│     太亮 → Gamma > 1 压暗                     │
│     正常 (120±30) → 不处理                    │
└──────────────────┬──────────────────────────┘
                   │ 亮度均衡图像
┌──────────────────┴──────────────────────────┐
│  4. CLAHE 局部增强                           │
│     V 通道 CLAHE (clipLimit=2.0, grid=8×8)  │
│     暗处提亮、亮处压暗、阴影减弱              │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┴──────────────────────────┐
│  5. 双模式检测                               │
│                                              │
│  正常亮度 → HSV 颜色检测 (H18-43 S50-255 V70-255)  │
│  偏暗偏亮 → HSV + Canny 边缘融合               │
│  检出不足 → 边缘兜底                          │
└──────────────────┬──────────────────────────┘
                   │ 碎片多边形
┌──────────────────┴──────────────────────────┐
│  6. 拼图验证                                 │
│     碎片在矩形坐标系中已处于正确位置            │
│     → 相邻碎片检测（共享顶点 < 2mm）           │
│     → 2cm 约束验证（赛题评分标准）             │
└─────────────────────────────────────────────┘
```

---

## 二、模块清单

| 文件 | 功能 | 用法 |
|---|---|---|
| `src/vision/dataset_generator.py` | 赛题数据生成（Fig2精确 + 随机裁切） | `python dataset_generator.py --mode custom --num 50` |
| `src/vision/fragment_detector.py` | HSV 碎片识别 + GT 评估 | `python fragment_detector.py` |
| `src/vision/realtime_detector.py` | 摄像头实时检测（HTTP流/本地窗口） | `python realtime_detector.py` |
| `src/vision/validate_dataset.py` | 赛题约束验证（边数/边长/外边） | `python validate_dataset.py` |
| `src/algorithm/puzzle_solver.py` | 拼图验证（相邻碎片 2cm 约束） | `python puzzle_solver.py --verify` |
| `src/algorithm/visualize_puzzle.py` | 期望 vs 求解 并排对比 | `python visualize_puzzle.py --all` |
| `src/algorithm/process_3_samples.py` | 3 样本自验证（4 轮自修复 + 超时保护） | `python process_3_samples.py` |

---

## 三、实时检测流水线参数

### 3.1 HSV 颜色阈值

| 参数 | 值 | 说明 |
|---|---|---|
| H | 18 ~ 43 | 黄色色相，±2~3 容差 |
| S | 50 ~ 255 | 低饱和排除白色背景 |
| V | 70 ~ 255 | 低亮度排除黑色分界线 |

### 3.2 CLAHE

| 参数 | 值 |
|---|---|
| clipLimit | 2.0 |
| tileGridSize | 8×8 |

### 3.3 Gamma 校正

| 参数 | 值 |
|---|---|
| 目标亮度 | 120 |
| 正常范围 | ±30 |
| Gamma 范围 | 0.5 ~ 2.0 |

### 3.4 形态学

| 参数 | 值 |
|---|---|
| 核形状 | MORPH_ELLIPSE 5×5 |
| 开运算 | 1 次（去噪） |
| 闭运算 | 2 次（填孔） |

### 3.5 多边形提取

| 参数 | 值 |
|---|---|
| approxPolyDP epsilon | 0.008 × 周长 |
| 最小面积 | 500 px² (HSV) / 800 px² (边缘) |
| 最少顶点 | 3 |

---

## 四、数据集生成规格

### 4.1 赛题 Fig2 精确模式

- 目标矩形：100mm × 60mm（固定）
- 碎片：4 个，边数 3-4，符合赛题图 2

### 4.2 随机自定义裁切模式

| 参数 | 范围 | 约束 |
|---|---|---|
| 矩形宽度 | 90 ~ 120mm | 赛题 9~12cm |
| 矩形高度 | 50 ~ 90mm | 赛题 5~9cm |
| 碎片数 | 4 | 固定 |
| 每碎片边数 | 3 ~ 5 | 赛题 ≤5 |
| 每边长度 | ≥ 20mm | 赛题 ≥2cm |
| 外边 | ≥ 1 条/碎片 | 赛题要求 |
| 碎片间距 | ≥ 3mm | 防止轮廓粘连 |

### 4.3 配色

| 对象 | 颜色 (BGR) | 说明 |
|---|---|---|
| 背景 | (255, 255, 255) | 白色 A4 纸 |
| 碎片 | (0, 255, 255) | 纯黄色 |
| 分界线 | (0, 0, 0) | 黑色 3mm 宽 |

---

## 五、测试结果

| 指标 | 数值 |
|---|---|
| HSV 碎片检出率 | 100% (200/200) |
| 平均顶点误差 | 1.50mm |
| 单张识别耗时 | 0.019s (HSV) / 0.030s (Canny) |
| 实时检测 FPS | ~15fps (1280×720, 树莓派 5) |
| 拼图 2cm 约束通过率 | 50/50 |

---

## 六、部署

### 树莓派

```bash
# 依赖
pip install opencv-python flask numpy

# 摄像头初始化
v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=3
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_automatic=1

# 启动
cd /home/pi02/puzzle
python3 src/vision/realtime_detector.py
# 浏览器 → http://10.160.161.254:8080
```

### 本地测试

```powershell
# 生成数据 → 识别 → 验证
python src/vision/dataset_generator.py --mode custom --num 50 --seed 42
python src/vision/validate_dataset.py
python src/vision/fragment_detector.py
python src/algorithm/puzzle_solver.py --verify
python src/algorithm/visualize_puzzle.py --all
```

---

## 七、状态栏说明

实时检测画面顶部状态栏格式：

```
Frags:4 | Warp | HSV | Bri:118 | 15.2fps
```

| 字段 | 含义 | 可能值 |
|---|---|---|
| Frags | 检出碎片数 | 0 ~ N |
| Warp/Raw | 透视矫正状态 | Warp=已矫正 / Raw=未找到纸 |
| HSV/Edge/HSV+Edge | 当前检测模式 | 自适应切换 |
| Bri | ROI 上半区平均亮度 | 0 ~ 255 |
| G | Gamma 校正值 | 偏离 1.0 时显示 |
| fps | 实时帧率 | — |

---

## 八、待完成

- [ ] 扑克牌图案匹配 (要求 2b) — ORB 特征提取
- [ ] 摄像头标定 (内参矩阵 + 畸变系数)
- [ ] 串口通信 (STM32 ↔ 树莓派 5，协议待定)
- [ ] 白色碎片模式 (要求 2a) — 反色检测或深色背景

---

## 九、GitHub

<https://github.com/zilinw-design/auto-puzzle>
