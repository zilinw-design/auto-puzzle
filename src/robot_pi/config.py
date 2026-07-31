"""
config.py -- LeArm 配置 (树莓派/Windows 通用)
树莓派: PORT = '/dev/ttyUSB0'
Windows: PORT = 'COM7'
"""

import platform
import os

# 串口: 环境变量优先, 否则按平台默认
PORT = os.environ.get("LEARM_PORT", "/dev/ttyUSB0" if platform.system() == "Linux" else "COM7")

# 固件参数
BAUDRATE = 9600
