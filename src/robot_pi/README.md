# LeArm 机械臂控制 SDK

STM32F103RBT6 + PWM 舵机，跨平台（树莓派/Windows）。

---

## 硬件连接

```
树莓派 USB → STM32 控制板 → 6路 PWM 舵机
```

STM32 固件已修改：复位不张开夹爪、CMD_COORDINATE_SET 移动时间 1500ms。

---

## 树莓派部署

### 1. 安装依赖

```bash
sudo apt update
sudo apt install python3-pip
pip3 install pyserial numpy
```

### 2. 配置串口

```bash
# 查看串口号
ls /dev/ttyUSB* /dev/ttyAMA*

# 方式A: 环境变量 (推荐)
export LEARM_PORT=/dev/ttyUSB0

# 方式B: 直接改 config.py 第9行
# PORT = "/dev/ttyUSB0"
```

### 3. 权限

```bash
sudo usermod -a -G dialout $USER
# 重新登录生效
```

### 4. 测试连接

```bash
python3 -c "from arm_controller import ArmController; a=ArmController(); a.connect(); a.home(); a.disconnect()"
```

---

## 使用方法

### 单点测试

```bash
python3 test_solver.py 6 -9        # 机械臂移动到 (6,-9)cm 上方
```

### 拾放工作流

```bash
python3 pick_and_place.py <pick_x> <pick_y> <place_x> <place_y>
python3 pick_and_place.py 6 -9 6 9  # 从A拾取, 放到B
```

### Python 调用

```python
from arm_controller import ArmController
from config import PORT

arm = ArmController(PORT)
arm.connect()
arm.home()                            # 复位
arm.move_by_delta(30, -90, 3)         # CMD 4: ikine 移动到 (6,-9) 安全高度
arm.move_to_pose({1: 1500}, 2000)     # CMD 3: 闭合夹爪
arm.disconnect()
```

---

## 控制原理

### 架构

```
PC (Python) --serial 9600bps--> STM32F103 --PWM--> 6x 舵机
                                   |
                               LeArm.lib (ikine)
```

### 串口协议

```
帧格式: 0x55 0x55 [len] [cmd] [data...]

CMD 3  (MULT_SERVO_MOVE):  直接指定6路PWM脉冲, 时间可控
CMD 4  (COORDINATE_SET):   发 (dx,dy,dz)mm, 固件 ikine() 算脉冲
CMD 12 (SERVOS_RESET):     复位 (夹爪保持闭合, 固件修改)
CMD 13 (ANGLE_BACK_READ):  回读6路实际脉冲
```

### 坐标定位

采用**线性拟合模型**，17 个标定点 → 全局平面拟合 → 任意 (x,y) → ikine delta：

```
dx = 9.7*X + 0.4*Y - 17
dy = -0.1*X + 10.0*Y - 6
dz = f(X,Y)           (安全高度 = dz + 10mm)
```

### 工作流

```
home → ikine 到拾取点安全高度 → Z-10 下降 → 等3s抓取
     → Z+10 上升 → home → ikine 到释放点安全高度
     → Z-10 下降 → 等3s释放 → Z+10 上升 → home
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `arm_controller.py` | 串口驱动，实现 CMD 3/4/12/13 |
| `pick_and_place.py` | 拾放工作流 |
| `test_solver.py` | 单点定位测试 |
| `solver.py` | IDW 脉冲插值 |
| `calibrate_ikine.py` | 交互式 ikine 标定工具 (Windows) |
| `ikine_lib.json` | 34 个 ikine 标定锚点 |
| `config.py` | 端口配置 |

---

## 坐标系

```
原点 = 工作区左下角
X轴 下→上  0~20.9 cm
Y轴 左→右 -14.7~+14.7 cm
Z轴: 安全高度 = 抓取高度 + 10mm
```

---

## 安全

- `move_by_delta` 单步限 ±128mm, 自动拆步
- PULSE 范围 500~2500 (夹爪 500~1500)
- 致命越界自动断开串口急停
- Ctrl+C 安全中断

---

## 兼容性

| 平台 | Python | 串口 | 依赖 |
|------|--------|------|------|
| 树莓派 (Raspberry Pi OS) | 3.9+ | /dev/ttyUSB0 | pyserial, numpy |
| Windows 10/11 | 3.9+ | COM7 | pyserial, numpy |
| Linux | 3.9+ | /dev/ttyUSB0 | pyserial, numpy |
