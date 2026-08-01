"""
pick_place_simple.py — LeArm 机械臂拾放工作流 (树莓派版, 不含旋转)

=== 硬件 ===
  - LeArm STM32F103RBT6 + 6路PWM舵机
  - 串口 9600bps 8N1
  - 夹爪(#1) 常闭 P1500 (电磁铁胶带绑死)

=== 坐标系 ===
  - 原点: 工作区左下角
  - X轴: 下→上 0~20.9cm
  - Y轴: 左→右 -14.7~+14.7cm
  - Z: 安全高度=grip+35mm, 下降35mm到抓取面

=== 定位算法 ===
  XY: 线性拟合 (19点标定)
    dx = 9.4*X + 0.1*Y - 21  (CMD4 X增量 mm)
    dy = 0.3*X + 9.6*Y + 2   (CMD4 Y增量 mm)
  Z: IDW插值 (19点实测grip_dz)
  安全高度 = grip_dz + 35mm

=== 用法 ===
  python3 pick_place_simple.py <pick_x> <pick_y> <place_x> <place_y>
  python3 pick_place_simple.py 6 -9 6 9

=== GPIO 电磁铁 ===
  BCM pin 17, HIGH=ON, LOW=OFF
  需外接继电器/MOSFET驱动模块
  若无RPi.GPIO则跳过(Windows调试模式)
"""

import sys, os, json, time, math

# ====== 配置 ======
PORT = os.environ.get("LEARM_PORT", "/dev/ttyUSB0")
BAUDRATE = 9600
SAFE_ABOVE = 35          # 安全高度=grip+35mm
WAIT_GRIP = 1.0          # 吸合停留 1s
WAIT_RELEASE = 1.0       # 释放停留 1s
SETTLE_TIME = 0.5        # 到位稳定时间
VISION_TO_ARM_Y = 14.7 / 14.85  # 视觉Y→机械臂Y缩放

# GPIO 电磁铁
GPIO_PIN = 17
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_PIN, GPIO.OUT)
    GPIO.output(GPIO_PIN, GPIO.LOW)
    GPIO_OK = True
except (ImportError, RuntimeError):
    GPIO_OK = False
    print("[GPIO] unavailable (mock mode)")

def magnet(on: bool):
    if GPIO_OK: GPIO.output(GPIO_PIN, GPIO.HIGH if on else GPIO.LOW)

# ====== 串口驱动 (CMD 12 + CMD 4 + CMD 3) ======

class LeArm:
    """精简串口: 复位(CMD12), ikine移动(CMD4), 脉冲(CMD3)"""

    def __init__(self, port=PORT):
        import serial
        self.ser = serial.Serial(port, BAUDRATE, timeout=0.2)
        time.sleep(1.5)

    def close(self):
        if self.ser and self.ser.is_open: self.ser.close()

    def _send(self, frame: bytes):
        self.ser.write(frame); self.ser.flush()

    def home(self):
        """CMD12 复位 (夹爪保持闭合 P1500, 固件已修改)"""
        self._send(bytes([0x55, 0x55, 0x02, 0x0C]))
        time.sleep(1.0)

    def move_by_delta(self, dx_mm, dy_mm, dz_mm):
        """CMD4: ikine增量移动 (自动拆步)"""
        for v, name in [(dx_mm,"dx"),(dy_mm,"dy"),(dz_mm,"dz")]:
            if v < -128 or v > 127:
                raise ValueError(f"{name}={v}mm out of [-128,+127]")
        frame = bytes([0x55,0x55,0x05,0x04, dx_mm&0xFF, dy_mm&0xFF, dz_mm&0xFF])
        self._send(frame)
        time.sleep(1.5)

    def move_safe(self, dx, dy, dz):
        """CMD4大位移, 自动拆步到每步<100mm"""
        steps = max(1, max(abs(dx),abs(dy),abs(dz)) // 100 + 1)
        for i in range(steps):
            sx = ((i+1)*dx//steps) - (i*dx//steps)
            sy = ((i+1)*dy//steps) - (i*dy//steps)
            sz = ((i+1)*dz//steps) - (i*dz//steps)
            if sx or sy or sz:
                self.move_by_delta(sx, sy, sz)
                time.sleep(0.8)
        time.sleep(1.5)

# ====== 定位求解 (XY线性拟合 + Z的IDW) ======

# 19点标定库 (calibration_final.json 内嵌)
_CALIB = {
  "points": [
    {"x":0,"y":-14.7,"dx":-25,"dy":-138,"grip_dz":-48,"safe_dz":-3},
    {"x":0,"y":0,"dx":-11,"dy":4,"grip_dz":-40,"safe_dz":-5},
    {"x":0,"y":14.7,"dx":-17,"dy":141,"grip_dz":-43,"safe_dz":2},
    {"x":3,"y":4,"dx":9,"dy":43,"grip_dz":-38,"safe_dz":2},
    {"x":6,"y":-7,"dx":29,"dy":-68,"grip_dz":-44,"safe_dz":-4},
    {"x":6,"y":7,"dx":34,"dy":72,"grip_dz":-45,"safe_dz":10},
    {"x":7.9,"y":2.5,"dx":56,"dy":27,"grip_dz":-39,"safe_dz":1},
    {"x":10,"y":-14.7,"dx":71,"dy":-136,"grip_dz":-25,"safe_dz":20},
    {"x":10,"y":-7,"dx":68,"dy":-64,"grip_dz":-35,"safe_dz":15},
    {"x":10,"y":0,"dx":70,"dy":6,"grip_dz":-36,"safe_dz":4},
    {"x":10,"y":7,"dx":67,"dy":71,"grip_dz":-26,"safe_dz":9},
    {"x":10,"y":14.7,"dx":69,"dy":143,"grip_dz":-28,"safe_dz":7},
    {"x":10,"y":5,"dx":67,"dy":51,"grip_dz":-51,"safe_dz":-11},
    {"x":16,"y":-7,"dx":136,"dy":-61,"grip_dz":-11,"safe_dz":24},
    {"x":16,"y":7,"dx":145,"dy":79,"grip_dz":-13,"safe_dz":12},
    {"x":17,"y":-4,"dx":146,"dy":-36,"grip_dz":-16,"safe_dz":19},
    {"x":20.9,"y":-14.7,"dx":170,"dy":-129,"grip_dz":-22,"safe_dz":43},
    {"x":20.9,"y":0,"dx":180,"dy":3,"grip_dz":-18,"safe_dz":12},
    {"x":20.9,"y":14.7,"dx":169,"dy":150,"grip_dz":-14,"safe_dz":61}
  ]
}

def ikine_xy(x_cm, y_cm):
    """线性拟合: 纸面坐标(cm) -> (dx_mm, dy_mm)"""
    dx = round(9.4 * x_cm + 0.1 * y_cm - 21)
    dy = round(0.3 * x_cm + 9.6 * y_cm + 2)
    return dx, dy

def idw_dz(x_cm, y_cm, layer="grip"):
    """IDW: 从标定库查dz"""
    pts = _CALIB["points"]
    key = "grip_dz" if layer == "grip" else "safe_dz"
    ws, n = 0.0, 0.0
    for p in pts:
        d = math.hypot(x_cm - p["x"], y_cm - p["y"])
        if d < 0.3: return p[key]
        w = 1.0 / (d * d); n += w; ws += w * p[key]
    return round(ws / n)

def solve_ikine(x_cm, y_cm, layer="grip"):
    dx, dy = ikine_xy(x_cm, y_cm)
    dz = idw_dz(x_cm, y_cm, layer)
    return dx, dy, dz

# ====== 工作流 ======

def vision_to_arm(x, y):
    return x, y * VISION_TO_ARM_Y

def main():
    if len(sys.argv) < 5:
        print("Usage: python3 pick_place_simple.py <pick_x> <pick_y> <place_x> <place_y>")
        sys.exit(1)

    px, py = vision_to_arm(float(sys.argv[1]), float(sys.argv[2]))
    tx, ty = vision_to_arm(float(sys.argv[3]), float(sys.argv[4]))

    p_dx, p_dy, p_dz_grip = solve_ikine(px, py, "grip")
    p_dz_safe = idw_dz(px, py, "safe")
    t_dx, t_dy, t_dz_grip = solve_ikine(tx, ty, "grip")
    t_dz_safe = idw_dz(tx, ty, "safe")

    print(f"Pick ({px:.1f},{py:.1f}): safe({p_dx},{p_dy},{p_dz_safe}) grip_dz={p_dz_grip}")
    print(f"Place({tx:.1f},{ty:.1f}): safe({t_dx},{t_dy},{t_dz_safe}) grip_dz={t_dz_grip}")
    print("Ctrl+C = emergency stop")
    input("Enter to start: ")

    arm = LeArm(PORT)
    try:
        # -- 拾取 --
        print("[1] home"); arm.home()
        print("[2] safe"); arm.move_safe(p_dx, p_dy, p_dz_safe)
        time.sleep(SETTLE_TIME)
        print("[3] descend"); arm.move_by_delta(0, 0, p_dz_grip - p_dz_safe)
        time.sleep(SETTLE_TIME)
        magnet(True); print("[4] grip"); time.sleep(WAIT_GRIP)
        print("[5] rise"); arm.move_by_delta(0, 0, p_dz_safe - p_dz_grip)

        # -- 放置 --
        print("[6] home"); arm.home()
        print("[7] safe"); arm.move_safe(t_dx, t_dy, t_dz_safe)
        time.sleep(SETTLE_TIME)
        print("[8] descend"); arm.move_by_delta(0, 0, t_dz_grip - t_dz_safe)
        time.sleep(SETTLE_TIME)
        magnet(False); print("[9] release"); time.sleep(WAIT_RELEASE)
        print("[10] rise"); arm.move_by_delta(0, 0, t_dz_safe - t_dz_grip)

        # -- 复位 --
        print("[11] home"); arm.home()
        magnet(False)
        print("done.")

    except KeyboardInterrupt:
        print("\nESTOP")
        magnet(False)
    finally:
        arm.close()

if __name__ == "__main__":
    main()
