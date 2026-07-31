"""
arm_controller.py -- LeArm STM32F103RBT6 serial driver (cross-platform)

Usage:
  from arm_controller import ArmController
  arm = ArmController('/dev/ttyUSB0')   # Raspberry Pi
  arm = ArmController('COM7')           # Windows
  arm.connect()
  arm.home()
  arm.move_by_delta(30, -90, 3)
  arm.disconnect()
"""

import serial
import serial.tools.list_ports
import time
import threading
from typing import Dict, Optional, List

FRAME_HEADER = b'\x55\x55'
CMD_MULT_SERVO_MOVE  = 3
CMD_COORDINATE_SET   = 4
CMD_FULL_ACTION_RUN  = 6
CMD_FULL_ACTION_STOP = 7
CMD_FULL_ACTION_ERASE = 8
CMD_ANGLE_BACK_READ  = 13
CMD_SERVOS_RESET     = 12

_SERVO_DEFS = {
    1:  ("gripper",      770,  500, 1500),
    2:  ("wrist_roll",  1500,  500, 2500),
    3:  ("wrist_pitch",  640,  500, 2500),
    4:  ("elbow",        511,  500, 2500),
    5:  ("shoulder",    1255,  500, 2500),
    6:  ("base_rotate", 1500,  500, 2500),
}

SERVO_MAP      = {k: v[0] for k, v in _SERVO_DEFS.items()}
RESET_DUTY     = {k: v[1] for k, v in _SERVO_DEFS.items()}
SERVO_LIMITS   = {k: (v[2], v[3]) for k, v in _SERVO_DEFS.items()}
VALID_IDS      = frozenset(range(1, 7))
MIN_TIME = 20
MAX_TIME = 30000
MAX_STEP_PER_SECOND = 500
FATAL_JUMP_RATIO = 5.0
FATAL_PULSE_MIN = 200
FATAL_PULSE_MAX = 2800


class SafetyError(Exception):
    pass

class SafetyFatal(Exception):
    pass


class ArmController:

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 0.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.RLock()
        self._current: Dict[int, int] = dict(RESET_DUTY)
        self._frame_count = 0
        self._panic_mode = False
        self.on_panic = None

    @staticmethod
    def list_ports() -> List[str]:
        return [f"{p.device} - {p.description}"
                for p in serial.tools.list_ports.comports()]

    def connect(self) -> bool:
        try:
            self._ser = serial.Serial(
                port=self.port, baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=self.timeout)
            print(f"[Arm] OK {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"[Arm] FAIL {self.port}: {e}")
            return False

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def panic(self, reason: str = "manual"):
        if self._panic_mode: return
        self._panic_mode = True
        print(f"\n[PANIC] {reason}")
        try: self.stop()
        except: pass
        try:
            if self._ser and self._ser.is_open: self._ser.close()
        except: pass

    @property
    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open and not self._panic_mode

    def _clamp(self, sid, pulse):
        if pulse < FATAL_PULSE_MIN or pulse > FATAL_PULSE_MAX:
            self.panic(f"servo#{sid} pulse={pulse} out of [{FATAL_PULSE_MIN},{FATAL_PULSE_MAX}]")
            raise SafetyFatal(f"servo#{sid} pulse={pulse} fatal")
        lo, hi = SERVO_LIMITS.get(sid, (500, 2500))
        return max(lo, min(hi, pulse))

    def _send_frame(self, frame: bytes):
        if self._panic_mode:
            raise SafetyFatal("panic mode: cannot send")
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.write(frame)
                self._ser.flush()
                self._frame_count += 1

    # -- public API --

    def home(self):
        frame = bytearray([0x55, 0x55, 0x02, CMD_SERVOS_RESET])
        self._send_frame(frame)
        time.sleep(1.5)
        self._current = dict(RESET_DUTY)

    def move_by_delta(self, dx_mm: int, dy_mm: int, dz_mm: int):
        for v, name in [(dx_mm,"dx"),(dy_mm,"dy"),(dz_mm,"dz")]:
            if v < -128 or v > 127:
                raise SafetyError(f"{name}={v}mm out of [-128,+127]")
        frame = bytearray([0x55,0x55,0x05,CMD_COORDINATE_SET,
                           dx_mm & 0xFF, dy_mm & 0xFF, dz_mm & 0xFF])
        self._send_frame(frame)

    def move_to_pose(self, pose: dict, time_ms: int = 3000):
        if self._panic_mode: raise SafetyFatal("panic mode")
        time_ms = max(MIN_TIME, min(MAX_TIME, time_ms))
        servo_count = len(pose)
        buf = bytearray()
        buf.extend(FRAME_HEADER)
        buf.append(4 + servo_count * 3)
        buf.append(CMD_MULT_SERVO_MOVE)
        buf.append(servo_count)
        buf.append(time_ms & 0xFF)
        buf.append((time_ms >> 8) & 0xFF)
        for sid in sorted(pose):
            p = self._clamp(sid, pose[sid])
            buf.append(sid)
            buf.append(p & 0xFF)
            buf.append((p >> 8) & 0xFF)
        self._send_frame(bytes(buf))
        for sid, pulse in pose.items():
            self._current[sid] = self._clamp(sid, pulse)

    def gripper_close(self, time_ms: int = 2000):
        self.move_to_pose({1: 1500}, time_ms)

    def gripper_open(self, time_ms: int = 2000):
        self.move_to_pose({1: 600}, time_ms)

    def stop(self):
        frame = bytearray([0x55, 0x55, 0x02, CMD_FULL_ACTION_STOP, 0x00, 0x00])
        try:
            if self._ser and self._ser.is_open:
                with self._lock:
                    self._ser.write(frame)
                    self._ser.flush()
        except: pass

    def read_positions(self) -> Optional[Dict[int, int]]:
        query = bytearray([0x55, 0x55, 0x02, CMD_ANGLE_BACK_READ])
        try:
            with self._lock:
                if not self._ser or not self._ser.is_open: return None
                self._ser.reset_input_buffer()
                self._ser.write(bytes(query))
                self._ser.flush()
                resp = self._ser.read(22)
        except: return None
        if len(resp) < 22 or resp[0] != 0x55 or resp[1] != 0x55: return None
        if resp[3] != CMD_ANGLE_BACK_READ: return None
        data = resp[4:22]
        return {data[i*3]: data[i*3+1] | (data[i*3+2] << 8) for i in range(6)}

    def wait(self, ms: int):
        time.sleep(ms / 1000.0)
