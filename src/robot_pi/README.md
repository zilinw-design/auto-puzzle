# LeArm Robot Control — Raspberry Pi SDK

## Setup

```bash
pip3 install pyserial numpy
sudo usermod -a -G dialout $USER  # re-login
export LEARM_PORT=/dev/ttyUSB0
```

## Files

| File | Purpose |
|------|---------|
| `arm_controller.py` | Serial driver (CMD 3/4/12/13, cross-platform) |
| `config.py` | Port config (env var LEARM_PORT) |
| `pick_and_place.py` | Automated pick-and-place workflow |
| `test_solver.py` | Single/batch point testing |
| `solver.py` | IDW pulse interpolation (backup) |
| `ikine_lib.json` | 17-point calibration library |

## Usage

```bash
# Single point test (safe -> descend -> grip -> rise -> home)
python3 test_solver.py 6 9

# Batch all 17 calibrated points
python3 test_solver.py

# Full pick-and-place
python3 pick_and_place.py <pick_x> <pick_y> <place_x> <place_y>
```

## How Positioning Works

### XY: Linear Fit

17 calibrated points -> least-squares plane fit:

```
dx = 9.6*X + 0.3*Y - 26
dy = -0.3*X + 10.0*Y - 6
```

Max residual: dx<17, dy<11. Adjust constants if all points bias consistently.

### Z (Grip Height): IDW from library

17 grip dz values in `ikine_lib.json`, inverse-distance-weighted interpolation.
Safe height = grip_dz + 35mm.

### Coordinate Mapping

Vision Y range (+-14.85cm) -> Arm Y range (+-14.7cm):
`y_arm = y_vis * 14.7/14.85`

## Modifying Calibration Data

### Adjust grip depth at specific points

Edit `ikine_lib.json` -> find entries with `"layer":"grip"` -> modify `"dz"` value.
More negative = deeper. Then update `"layer":"safe"` entry for same XY: safe_dz = grip_dz + 35.

### Adjust XY bias

Edit `pick_and_place.py` line 27-28, modify constants in dx/dy formulas:
- Larger dx constant = shift X positive
- Larger dy constant = shift Y positive

### Add new calibration points

Add two entries to `ikine_lib.json` per point:
```json
{"x": X, "y": Y, "dx": DX, "dy": DY, "dz": DZ, "layer": "grip"},
{"x": X, "y": Y, "dx": DX, "dy": DY, "dz": DZ+35, "layer": "safe"}
```
DX/DY values from linear fit formula above.

## Rotation (Wrist #2) — NOT YET IMPLEMENTED

The `rotate()` function exists but CMD 3 control of servo #2 is unreliable
on this specific arm. Physical mapping: P500=-45deg, P1500=0deg, P2500=+45deg.
Formula: `pwm = 1500 + angle * 1000/45`.

Rotation compensation logic is in `rotate_place.py` (Windows only, not in Pi folder).
Before enabling rotation on Pi, verify CMD 3 #2 control works reliably.

## GPIO Electromagnet

BCM pin 17, HIGH=on (grip), LOW=off (release).
Requires relay/MOSFET driver module (Pi 3.3V cannot drive directly).
Auto-detected: if RPi.GPIO unavailable, runs in mock mode.

## Safety

- CMD 4 delta limited to +-128mm per call, auto-split
- Ctrl+C = emergency stop
- `p` key = panic during batch test
