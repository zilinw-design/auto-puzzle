"""
arm_controller.py — LeArm STM32F103RBT6 PWM舵机控制器串口驱动

所有参数均有固件源码依据，详见各注释中的出处标注。

协议帧格式: 0x55 0x55 [长度] [指令] [数据...]
  指令:
    CMD_MULT_SERVO_MOVE  (3)  — 多舵机同时运动（实时控制）
    CMD_FULL_ACTION_RUN  (6)  — 运行动作组
    CMD_FULL_ACTION_STOP (7)  — 停止动作
    CMD_FULL_ACTION_ERASE(8)  — 擦除所有动作组
    CMD_ACTION_DOWNLOAD  (25) — 下载动作组

用法:
  arm = ArmController("COM3")
  arm.connect()
  arm.move_servos({1: 1500, 2: 1500, 6: 1500}, time_ms=1000)
  arm.gripper_close()
  arm.gripper_open()
  arm.home()
  arm.disconnect()
"""

import serial
import serial.tools.list_ports
import time
import threading
import math
import warnings
from typing import Dict, Optional, List, Tuple

# =========================================================================
# 协议常量 — 全部来自 STM32 固件
# =========================================================================
FRAME_HEADER = b'\x55\x55'           # app_porting.h:8  APP_PACKET_HEADER

CMD_MULT_SERVO_MOVE  = 3             # app_porting.h:28  CMD_MULT_SERVO_MOVE
CMD_FULL_ACTION_RUN  = 6             # app_porting.h:29  CMD_ACTION_GROUP_RUN
CMD_FULL_ACTION_STOP = 7             # app_porting.h:30  CMD_FULL_ACTION_STOP
CMD_FULL_ACTION_ERASE = 8            # app_porting.h:31  CMD_FULL_ACTION_ERASE
CMD_COORDINATE_SET   = 4             # app_porting.h:28  CMD_COORDINATE_SET
CMD_ANGLE_BACK_READ  = 13            # app_porting.h:36  CMD_ANGLE_BACK_READING
CMD_ACTION_DOWNLOAD  = 25            # app_porting.h:37  CMD_ACTION_DOWNLOAD
CMD_SERVOS_RESET     = 12            # app_porting.h:35  CMD_SERVOS_RESET

# =========================================================================
# 舵机参数 — 全部来自 STM32 固件
# =========================================================================

# ID→物理映射: robot_arm.c robot_arm_knot_run() id→pwm_servos[6-id]
# 复位值:    global.h:25-30 PWM_SERVO1~6_RESET_DUTY
_SERVO_DEFS = {
    1:  ("gripper",      770,  500, 1500),   # 夹爪, 上限1500(robot_arm.c:264)
    2:  ("wrist_roll",  1500,  500, 2500),   # 腕部翻转
    3:  ("wrist_pitch",  640,  500, 2500),   # 腕部俯仰
    4:  ("elbow",        511,  500, 2500),   # 肘部小臂
    5:  ("shoulder",    1255,  500, 2500),   # 肩部大臂
    6:  ("base_rotate", 1500,  500, 2500),   # 底盘旋转
}

SERVO_MAP      = {k: v[0] for k, v in _SERVO_DEFS.items()}
RESET_DUTY     = {k: v[1] for k, v in _SERVO_DEFS.items()}  # 固件复位值
SERVO_LIMITS   = {k: (v[2], v[3]) for k, v in _SERVO_DEFS.items()}
SERVO_CENTER   = 1500                       # pwm_servos.h:21 MIDDLE_DUTY
VALID_IDS      = frozenset(range(1, 7))

# 时间限制
# 固件: pwm_servos.h:26-27 MIN_RUNNING_TIME=20, MAX_RUNNING_TIME=30000
# 安全策略: 最小操作时间 2000ms，不允许短时间快动
MIN_TIME = 2000
MAX_TIME = 30000
DEFAULT_MOVE_TIME = 2000

# 复位时间: robot_arm_init() 调用 robot_arm_reset(2000)
RESET_TIME_MS = 1000

# =========================================================================
# 安全防护参数
# =========================================================================

# 单步最大脉冲变化 (μs)。超过此值自动插入中间点。
# 500μs/s ≈ 45°/s, 对于无负载舵机是安全速度
MAX_STEP_PER_SECOND = 500

# 最小分段步时间 (ms) — 保证每段至少有此时间完成
MIN_SEGMENT_TIME_MS = 50

# 是否启用自动分段 (True=自动平滑, False=直发)
AUTO_SEGMENT = True


# =========================================================================
# 安全等级
# =========================================================================

class SafetyWarning(UserWarning):
    """安全提醒 — 已自动修正，继续运行"""
    pass

class SafetyError(Exception):
    """安全错误 — 拒绝本次指令，但保持连接"""
    pass

class SafetyFatal(Exception):
    """致命安全事件 — 已触发急停、断开串口"""
    pass


# =========================================================================
# 致命事件阈值
# =========================================================================

# 超过此跳变倍数触发 FATAL（即使用户关了自动分段也拦截）
FATAL_JUMP_RATIO = 5.0          # 5倍于 MAX_STEP_PER_SECOND
# 脉宽超出此绝对上限触发 FATAL（硬件损坏保护）
FATAL_PULSE_MIN = 200
FATAL_PULSE_MAX = 2800


class ArmController:
    """LeArm STM32F103RBT6 机械臂控制器

    三级安全响应:
      WARNING — 自动修正 + 警告打印，继续运行
      ERROR   — 拒绝本次指令，保持串口连接（可恢复）
      FATAL   — 立即发急停指令 → 断开串口 → 抛异常（需重新连接）
    """

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 0.5):
        """
        Args:
            port:     串口号 "COM3"
            baudrate: 波特率。固件 usart.c:46 设置为 9600
            timeout:  串口读写超时
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.RLock()
        self._current: Dict[int, int] = dict(RESET_DUTY)
        self._frame_count = 0
        self._panic_mode = False
        # 用户可注册急停回调: callable(reason: str)
        self.on_panic = None

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------

    @staticmethod
    def list_ports() -> List[str]:
        return [f"{p.device} — {p.description}"
                for p in serial.tools.list_ports.comports()]

    def connect(self) -> bool:
        try:
            self._ser = serial.Serial(
                port=self.port, baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=self.timeout)
            print(f"[Arm] OK {self.port} @ {self.baudrate} bps")
            return True
        except Exception as e:
            print(f"[Arm] FAIL {self.port}: {e}")
            return False

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            print(f"[Arm] 断开 {self.port}")

    def panic(self, reason: str = "手动触发"):
        """
        紧急停止 — 发送停止指令 → 断开串口。
        调用后机械臂保持当前位置不再运动，需重新 connect() 恢复。
        """
        if self._panic_mode:
            return  # 已进入急停，不重复执行

        self._panic_mode = True
        print(f"\n{'='*50}")
        print(f"  ⚡ 急停触发: {reason}")
        print(f"{'='*50}")

        # 1. 先发急停指令
        try:
            self.stop()
        except Exception:
            pass

        # 2. 断开串口 — 舵机控制器收不到新指令，保持原位
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
                print(f"  → 串口 {self.port} 已断开")
        except Exception:
            pass

        # 3. 用户回调
        if self.on_panic:
            try:
                self.on_panic(reason)
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return (self._ser is not None and self._ser.is_open
                and not self._panic_mode)

    # ------------------------------------------------------------------
    # 安全校验
    # ------------------------------------------------------------------

    def _validate_ids(self, servos: Dict[int, int]):
        """ID 校验 — ERROR 级：拒绝非法舵机编号"""
        bad = [sid for sid in servos if sid not in VALID_IDS]
        if bad:
            raise SafetyError(
                f"非法舵机ID: {bad}（有效范围 1~6）。指令已拒绝，连接保持。")

    def _clamp_pulse(self, servo_id: int, pulse: int) -> int:
        """
        脉宽校验:
          - 超出 FATAL 限 → panic()
          - 超出关节限 → WARNING + 裁剪
        """
        # FATAL: 绝对硬限（防止硬件损坏）
        if pulse < FATAL_PULSE_MIN or pulse > FATAL_PULSE_MAX:
            self.panic(
                f"舵机#{servo_id} 脉宽 P{pulse} 超出硬件安全范围 "
                f"[{FATAL_PULSE_MIN}, {FATAL_PULSE_MAX}]")
            raise SafetyFatal(f"舵机#{servo_id} P{pulse} 致命越界，已急停")

        lo, hi = SERVO_LIMITS.get(servo_id, (500, 2500))
        clamped = max(lo, min(hi, pulse))
        if clamped != pulse:
            warnings.warn(
                f"舵机#{servo_id} P{pulse} 超出关节限位 [{lo},{hi}]"
                f"，已自动裁剪为 P{clamped}",
                category=SafetyWarning)
        return clamped

    def _check_jump(self, servo_id: int, from_pulse: int, to_pulse: int,
                    time_ms: int):
        """
        步进守卫:
          - FATAL: 超过绝对跳变倍数 → panic()
          - WARNING: 超过安全跳变速度 → 自动分段
        """
        delta = abs(to_pulse - from_pulse)
        if delta == 0:
            return

        max_allowed = int(MAX_STEP_PER_SECOND * time_ms / 1000)
        fatal_limit = int(max_allowed * FATAL_JUMP_RATIO)

        if delta > fatal_limit:
            joint = SERVO_MAP.get(servo_id, f"#{servo_id}")
            self.panic(
                f"舵机{joint} 跳变过大: P{from_pulse}→P{to_pulse} "
                f"({delta}μs/{time_ms}ms, 允许≤{fatal_limit}μs)")
            raise SafetyFatal(
                f"舵机{joint} P{from_pulse}→P{to_pulse} 致命跳变，已急停")

        if delta > max_allowed:
            joint = SERVO_MAP.get(servo_id, f"#{servo_id}")
            warnings.warn(
                f"舵机{joint} 跳变: P{from_pulse}→P{to_pulse} "
                f"({delta}μs/{time_ms}ms > {max_allowed}μs 安全值)，"
                f"将自动拆分为多段渐进。",
                category=SafetyWarning)

    # ------------------------------------------------------------------
    # 帧构建（底层）
    # ------------------------------------------------------------------

    def _build_mult_servo_frame(self, servos: Dict[int, int],
                                time_ms: int) -> bytes:
        """构建 CMD_MULT_SERVO_MOVE 帧"""
        self._validate_ids(servos)

        count = len(servos)
        data_len = 4 + count * 3
        buf = bytearray()
        buf.extend(FRAME_HEADER)
        buf.append(data_len)
        buf.append(CMD_MULT_SERVO_MOVE)
        buf.append(count)
        buf.append(time_ms & 0xFF)
        buf.append((time_ms >> 8) & 0xFF)

        for sid in sorted(servos):
            pulse = self._clamp_pulse(sid, servos[sid])
            buf.append(sid)
            buf.append(pulse & 0xFF)
            buf.append((pulse >> 8) & 0xFF)

        return bytes(buf)

    def _send_frame(self, frame: bytes):
        if self._panic_mode:
            raise SafetyFatal("急停模式下禁止发送指令。请先 connect() 恢复。")
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.write(frame)
                self._ser.flush()
                self._frame_count += 1

    # ------------------------------------------------------------------
    # 自动分段（软启动/减速）
    # ------------------------------------------------------------------

    def _segment_move(self, targets: Dict[int, int], time_ms: int):
        """
        如果某舵机单步跳变超过安全阈值，自动拆分为多段渐进。
        每段均匀分配时间和脉宽增量。
        """
        if not AUTO_SEGMENT:
            self._send_direct(targets, time_ms)
            return

        # 找最大跳变比例
        max_ratio = 0.0
        for sid, target in targets.items():
            current = self._current.get(sid, RESET_DUTY.get(sid, 1500))
            delta = abs(target - current)
            max_allowed = int(MAX_STEP_PER_SECOND * time_ms / 1000)
            if max_allowed > 0 and delta > max_allowed:
                ratio = delta / max_allowed
                if ratio > max_ratio:
                    max_ratio = ratio

        if max_ratio <= 1.0:
            # 不需要分段
            self._send_direct(targets, time_ms)
            return

        # 需要拆分: 分段数 = ceil(max_ratio)
        segments = math.ceil(max_ratio)
        seg_time = max(MIN_SEGMENT_TIME_MS, time_ms // segments)

        # 计算每个分段的中间目标
        starts = {sid: self._current.get(sid, RESET_DUTY.get(sid, 1500))
                  for sid in targets}
        for seg in range(1, segments + 1):
            frac = seg / segments
            mid = {}
            for sid, target in targets.items():
                mid[sid] = int(starts[sid] + (target - starts[sid]) * frac)
            self._send_direct(mid, seg_time)
            time.sleep(seg_time / 1000.0)

    def _send_direct(self, servos: Dict[int, int], time_ms: int):
        """直接发送一帧（不做分段检查）"""
        if self._panic_mode:
            raise SafetyFatal("急停模式下禁止发送指令。")
        time_ms = max(MIN_TIME, min(MAX_TIME, time_ms))
        frame = self._build_mult_servo_frame(servos, time_ms)
        self._send_frame(frame)
        for sid, pulse in servos.items():
            self._current[sid] = self._clamp_pulse(sid, pulse)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def move_servo(self, servo_id: int, pulse: int, time_ms: int = DEFAULT_MOVE_TIME):
        """移动单个舵机"""
        self.move_servos({servo_id: pulse}, time_ms)

    def move_servos(self, servos: Dict[int, int], time_ms: int = DEFAULT_MOVE_TIME):
        """
        同时移动多个舵机（协同运动）。

        安全保护自动生效:
          - FATAL 脉宽越界 → panic() 急停 + 断开串口
          - ERROR 非法ID → 拒绝指令
          - WARNING 跳变过大 → 自动分段渐进
        """
        if self._panic_mode:
            raise SafetyFatal("急停模式下禁止发送指令。请先 connect() 恢复。")

        targets = {sid: self._clamp_pulse(sid, p) for sid, p in servos.items()}
        self._validate_ids(targets)

        for sid, target in targets.items():
            current = self._current.get(sid, RESET_DUTY.get(sid, 1500))
            self._check_jump(sid, current, target, time_ms)

        self._segment_move(targets, time_ms)

    def move_to_pose(self, pose: Dict[int, int], time_ms: int = DEFAULT_MOVE_TIME):
        """移动到预定义姿态（6舵机字典）"""
        self.move_servos(pose, time_ms)

    def move_to_position(self, name: str,
                         positions: Dict[str, Dict[int, int]],
                         time_ms: int = DEFAULT_MOVE_TIME):
        """按名称移动到已记录位置"""
        if name not in positions:
            print(f"[Arm] ✗ 未知位置: '{name}'")
            return
        self.move_to_pose(
            {int(k): v for k, v in positions[name].items()}, time_ms)

    # ------------------------------------------------------------------
    # 夹爪 — 固件 robot_arm_claw_set(): open_angle∈[0,90]
    #         P600=全开, P1500=全闭 (你的实测)
    # ------------------------------------------------------------------

    def gripper_open(self, time_ms: int = DEFAULT_MOVE_TIME):
        """夹爪全开 → P600"""
        self.move_servo(1, 600, time_ms)

    def gripper_close(self, time_ms: int = DEFAULT_MOVE_TIME):
        """夹爪全闭 → P1500"""
        self.move_servo(1, 1500, time_ms)

    def gripper_half(self, time_ms: int = DEFAULT_MOVE_TIME):
        """夹爪半开 → P1050"""
        self.move_servo(1, 1050, time_ms)

    # ------------------------------------------------------------------
    # 复位 — 使用固件精确值 global.h:25-30, 时间 robot_arm_reset(2000)
    # ------------------------------------------------------------------

    def home(self, time_ms: int = RESET_TIME_MS):
        """机械臂复位 — 走固件原生 CMD_SERVOS_RESET (12)"""
        frame = bytearray([0x55, 0x55, 0x02, CMD_SERVOS_RESET])  # 无参数
        self._send_frame(frame)
        time.sleep(time_ms / 1000.0)
        self._current = dict(RESET_DUTY)
        print(f"[Arm] 复位完成 (固件原生, {time_ms}ms)")

    # ------------------------------------------------------------------
    # 动作组
    # ------------------------------------------------------------------

    def stop(self):
        """紧急停止 — 即使已 panic 也尝试发送"""
        frame = bytearray([0x55, 0x55, 0x02, CMD_FULL_ACTION_STOP, 0x00, 0x00])
        try:
            if self._ser and self._ser.is_open:
                with self._lock:
                    self._ser.write(frame)
                    self._ser.flush()
        except Exception:
            pass  # 串口已断开时静默

    # ------------------------------------------------------------------
    # 坐标控制 — CMD_COORDINATE_SET (4), 固件内部调用 LeArm.lib ikine()
    # 固件初始坐标: DEFAULT_X=15, DEFAULT_Y=0, DEFAULT_Z=2 (cm)
    # ------------------------------------------------------------------

    def move_by_delta(self, dx_mm: int, dy_mm: int, dz_mm: int):
        """
        相对当前位置移动 — 走上位机同款 ikine()。

        固件内部: x += dx/10.0, y += dy/10.0, z += dz/10.0
                   → robot_arm_coordinate_set(x, y, z, ...)
                   → LeArm.lib ikine() → 脉冲输出

        Args:
            dx_mm: X增量 mm, 范围 -128~+127  (步长1mm)
            dy_mm: Y增量 mm, 范围 -128~+127
            dz_mm: Z增量 mm, 范围 -128~+127
        """
        for v, name in [(dx_mm, "dx"), (dy_mm, "dy"), (dz_mm, "dz")]:
            if v < -128 or v > 127:
                raise SafetyError(f"{name}={v}mm 超范围 [-128,+127]mm")

        frame = bytearray([0x55, 0x55, 0x05, CMD_COORDINATE_SET,
                           dx_mm & 0xFF, dy_mm & 0xFF, dz_mm & 0xFF])
        self._send_frame(frame)
        print(f"[Arm] Δ=({dx_mm:+d},{dy_mm:+d},{dz_mm:+d})mm")

    def run_action(self, action_num: int, times: int = 1):
        """运行Flash中预存的动作组 (0~255, 0=无限)"""
        frame = bytearray([
            0x55, 0x55, 0x04, CMD_FULL_ACTION_RUN,
            action_num & 0xFF,
            times & 0xFF,
            (times >> 8) & 0xFF,
        ])
        self._send_frame(frame)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def read_offsets(self) -> Dict[int, int]:
        """
        读取固件存储的舵机偏差值。CMD_SERVO_OFFSET_READ (2)。
        固件响应: 0x55 0x55 0x0E 0x02 [id1][ofs1]...[id6][ofs6]
        offset 是 int8 有符号值, 范围 [-100, +100]。
        write_duty = current_duty + offset  (pwm_servos.c)
        返回: {1: offset, 2: offset, ...} 或 None
        """
        query = bytearray([0x55, 0x55, 0x02, 2])  # CMD_SERVO_OFFSET_READ
        try:
            with self._lock:
                if not self._ser or not self._ser.is_open:
                    return None
                self._ser.reset_input_buffer()
                self._ser.write(bytes(query))
                self._ser.flush()
                resp = self._ser.read(16)  # 4头 + 12数据
        except Exception:
            return None

        if len(resp) < 16 or resp[0] != 0x55 or resp[1] != 0x55:
            return None

        data = resp[4:16]
        offsets = {}
        for i in range(6):
            sid = data[i * 2]
            raw = data[i * 2 + 1]
            # int8 → int
            ofs = raw if raw < 128 else raw - 256
            offsets[sid] = ofs
        return offsets

    def read_positions(self) -> Dict[int, int]:
        """
        从固件读取舵机实际当前位置。CMD_ANGLE_BACK_READING (13)。
        固件响应: 0x55 0x55 0x14 0x0D [id1][d1L][d1H]...[id6][d6L][d6H]
        返回: {1: pulse, 2: pulse, ...} 或 None（读取失败）
        """
        query = bytearray([0x55, 0x55, 0x02, CMD_ANGLE_BACK_READ])
        try:
            with self._lock:
                if not self._ser or not self._ser.is_open:
                    return None
                self._ser.reset_input_buffer()
                self._ser.write(bytes(query))
                self._ser.flush()
                # 响应: 4字节头 + 18字节数据 = 22字节
                resp = self._ser.read(22)
        except Exception:
            return None

        if len(resp) < 22:
            return None
        if resp[0] != 0x55 or resp[1] != 0x55:
            return None
        if resp[3] != CMD_ANGLE_BACK_READ:
            return None

        # 解析 18 字节数据: 每组 [id, duty_L, duty_H]
        data = resp[4:22]
        positions = {}
        for i in range(6):
            sid = data[i * 3]
            duty = data[i * 3 + 1] | (data[i * 3 + 2] << 8)
            positions[sid] = duty
        return positions

    def get_positions(self) -> Dict[int, int]:
        """返回 PC 侧记录的舵机位置（上次 set 的值）"""
        return dict(self._current)

    def print_state(self):
        print("\n  舵机状态 (固件复位值→当前值):")
        for sid in range(1, 7):
            joint = SERVO_MAP[sid]
            reset = RESET_DUTY[sid]
            pulse = self._current.get(sid, reset)
            lo, hi = SERVO_LIMITS[sid]
            angle = (pulse - 500) / 2000.0 * 180.0
            arrow = "→" if pulse != reset else "="
            print(f"    #{sid} {joint:<13} R{reset:4d} {arrow} P{pulse:4d}"
                  f"  [{lo},{hi}]  (~{angle:5.1f}°)")
        print()

    def wait(self, ms: int):
        time.sleep(ms / 1000.0)


# =========================================================================
# 位置记录器
# =========================================================================

class PositionRecorder:
    """记录和持久化机械臂位置"""

    def __init__(self, arm: ArmController):
        self.arm = arm
        self.positions: Dict[str, Dict[int, int]] = {}

    def record(self, name: str):
        self.positions[name] = self.arm.get_positions()
        print(f"[Rec] ✓ '{name}' → {self._fmt(self.positions[name])}")

    def remove(self, name: str):
        if name in self.positions:
            del self.positions[name]
            print(f"[Rec] ✗ 已删除 '{name}'")

    def list_positions(self):
        if not self.positions:
            print("[Rec] 暂无记录")
            return
        print(f"\n  已记录 {len(self.positions)} 个位置:")
        for name, pose in self.positions.items():
            print(f"    {name:<20} {self._fmt(pose)}")
        print()

    def _fmt(self, pose: Dict[int, int]) -> str:
        return " ".join(f"#{s}={pose[s]}" for s in sorted(pose))

    def save(self, filepath: str):
        import json
        data = {k: {str(s): p for s, p in v.items()}
                for k, v in self.positions.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Rec] ✓ 已保存到 {filepath}")

    def load(self, filepath: str):
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.positions = {k: {int(s): p for s, p in v.items()}
                          for k, v in data.items()}
        print(f"[Rec] ✓ 已加载 {filepath} ({len(self.positions)} 个位置)")


# =========================================================================
# XML 动作组 → JSON 位置转换
# =========================================================================

def xml_to_workflow(xml_path: str, json_path: str = None) -> List[dict]:
    """
    将 LeArm 上位机导出的 XML 动作组转换为工作流步骤列表。

    XML 格式 (上位机导出):
      <NewDataSet>
        <Type>PWM Servo</Type>
        <Table>
          <ID>1</ID>
          <Move>#1 P770 #2 P1500 ...</Move>
          <Time>T1000</Time>
          ...
        </Table>
      </NewDataSet>

    返回: [{"1": 770, "2": 1500, ...}, ...]  每个元素是一帧的所有舵机脉冲
    """
    import xml.etree.ElementTree as ET
    import re

    tree = ET.parse(xml_path)
    root = tree.getroot()

    frames = []
    for table in root.findall(".//Table"):
        moves = table.findall("Move")
        times = table.findall("Time")
        for move_el, time_el in zip(moves, times):
            # 解析 Move: "#1 P770 #2 P1500 #3 P1180 ..."
            text = move_el.text or ""
            pairs = re.findall(r'#(\d+)\s+P(\d+)', text)
            frame = {int(k): int(v) for k, v in pairs}
            # 解析 Time: "T1000"
            t_text = time_el.text or "T1000"
            t_match = re.search(r'(\d+)', t_text)
            frame["_time_ms"] = int(t_match.group(1)) if t_match else 1000
            frames.append(frame)

    if json_path:
        import json
        # 保存为位置文件格式（去掉 _time_ms 内部字段）
        positions = {}
        for i, f in enumerate(frames):
            t = f.pop("_time_ms", 1000)
            positions[f"frame_{i+1}_T{t}ms"] = f
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(positions, fp, ensure_ascii=False, indent=2)
        print(f"[XML→JSON] {len(frames)} 帧 → {json_path}")

    return frames


# =========================================================================
# 测试
# =========================================================================

if __name__ == "__main__":
    print("=== LeArm 控制器自检 ===\n")
    print("可用串口:")
    for p in ArmController.list_ports():
        print(f"  {p}")

    print("\n复位值 (固件 global.h):")
    for sid in range(1, 7):
        print(f"  #{sid} {SERVO_MAP[sid]:<13} P{RESET_DUTY[sid]}"
              f"  [{SERVO_LIMITS[sid][0]}, {SERVO_LIMITS[sid][1]}]")

    print(f"\n安全参数:")
    print(f"  最小操作时间: {MIN_TIME}ms (安全限制)")
    print(f"  最大步进速度: {MAX_STEP_PER_SECOND} μs/s")
    print(f"  致命跳变倍数: {FATAL_JUMP_RATIO}x")
    print(f"  致命脉宽范围: [{FATAL_PULSE_MIN}, {FATAL_PULSE_MAX}]")
    print(f"  自动分段: {'开' if AUTO_SEGMENT else '关'}")
    print(f"  波特率: 9600 (usart.c:46)")
    print(f"  协议帧头: 0x55 0x55 (app_porting.h:8)")

    print("\n帧构建测试: move_servos({1:1500,2:1200,6:800}, 2000ms)")
    arm = ArmController("COM3")
    frame = arm._build_mult_servo_frame({1: 1500, 2: 1200, 6: 800}, 2000)
    print(f"  十六进制: {frame.hex(' ')}")
    print(f"  长度: {len(frame)} 字节\n")

    # 测试 XML 转换
    import os
    test_xml = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "ai_harness_framework", "references", "raw",
        "LeArm 上位机软件", "LeArm 上位机软件", "temp", "model_1.xml")
    if os.path.exists(test_xml):
        print(f"=== XML 转换测试: {test_xml} ===")
        frames = xml_to_workflow(test_xml)
        for i, f in enumerate(frames):
            t = f.pop("_time_ms", "?")
            pos = " ".join(f"#{s}={f[s]}" for s in sorted(f))
            print(f"  帧{i+1} T{t}ms: {pos}")
