# eyeSwitch

Cross-platform take on [Glance Switch](https://glanceswitch.com/): the webcam tells which
monitor your head is facing and keyboard focus (and the pointer) follow you there.
Runs as a desktop app with a tray icon and a small settings window (Qt).

| OS | Status |
|----|--------|
| Linux (X11) | tested (Ubuntu GNOME) |
| Windows 10/11 | written, **not tested yet** |
| macOS 12+ | written, **not tested yet** |
| Linux (Wayland) | not supported: Wayland blocks focus and pointer control |

## Install

Python 3.10+ is required. Use a virtual environment (recent Ubuntu/Debian refuse a plain
`pip install` with "externally-managed-environment").

```bash
git clone git@github.com:pedroslev/eyeSwitch.git
cd eyeSwitch

python3 -m venv .venv               # Windows: py -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt     # the face model downloads itself on first run
python install.py                   # adds "eyeSwitch" to your application menu
```

The menu entry remembers this venv, so you never need to activate it again.
Do the per-OS step below **before** the first launch.

Per OS:

- **Linux**: Qt needs `libxcb-cursor`. Debian/Ubuntu `sudo apt install libxcb-cursor0`,
  Fedora `sudo dnf install xcb-util-cursor`, Arch `sudo pacman -S xcb-util-cursor`,
  openSUSE `sudo zypper install libxcb-cursor0`. Log in with an **X11 session**.
  The tray icon works on KDE, XFCE and friends; GNOME needs the AppIndicator extension
  (Ubuntu ships it enabled).
- **Windows**: `install.py` creates a Start Menu shortcut. No extra permissions.
- **macOS**: `install.py` creates `~/Applications/eyeSwitch.app`. Allow Camera on first
  launch, and add the app under System Settings > Privacy & Security > Accessibility
  (needed to focus windows). The pause hotkey may also ask for Input Monitoring.

`python install.py --uninstall` removes the menu entry. Re-run the installer if you move
this folder or change Python environments. To run without installing: `python eyeswitch.py`.

## Use

1. Open **eyeSwitch**.
2. Pick a mode and calibrate (Esc cancels a calibration):

| Mode | What it does | Calibration |
|------|--------------|-------------|
| **Monitor only** (default) | Focus follows the monitor your head is turned to. | *Quick*: look at one dot per monitor, ~5 s each. |
| **Monitor and window** | Focus follows the window you are looking at, on any monitor. | *Full*: follow a dot over a 3x3 grid on each monitor, ~35 s total. |

3. Keep it running. Closing the window hides it to the tray; Quit is in the tray menu.

Window mode estimates where on the desktop you are looking from your head pose and the position
of your irises. The full calibration fits that model and reports its estimated error (about 5% of a
monitor width is typical for a webcam). Tick **Show where it thinks I am looking** to see the
estimate as a red dot while you tune it. It works best with big windows and a steady seat;
redo the full calibration when you move the camera or change your posture a lot.
The quick calibration discards a previous gaze model, since both describe the same setup.

Settings: camera, mode, delay before switching monitor (default 300 ms), delay before focusing a
window (500 ms), wait after typing (2 s: window mode never changes focus inside a monitor while you
are typing), start at login. Ctrl+Shift+G pauses and resumes; while paused the camera is released.
Nothing happens while you are using the mouse (and for 1.5 s after), and glances shorter than the
delay are ignored. Recalibrate when you move the webcam or the monitors.

Settings and calibration live in `~/.config/eyeswitch/` (Linux),
`%APPDATA%\eyeSwitch` (Windows) or `~/Library/Application Support/eyeSwitch` (macOS).

## How it works

- `tracker.py`: MediaPipe Face Landmarker (~15 fps, 640x480): head yaw/pitch/position from the face
  transform matrix, iris position from the landmarks.
- `gaze.py`: ridge regression from those features to a desktop point, degree and regularization
  chosen by leave-one-point-out cross-validation (this is also the error estimate).
- `backends/`: one module per OS behind the same interface (`base.py`): monitors, pointer,
  windows (focus, which one is under a point), typing detection, pause hotkey. `x11.py` uses RandR/EWMH, `windows.py` Win32 via ctypes,
  `macos.py` Quartz plus the Accessibility API.
- `engine.py`: background loop, nearest-monitor choice with hysteresis, dwell timers,
  restores the last focused window and pointer position on the target monitor.
- `eyeswitch.py`: Qt app (settings window, calibration, tray). `autostart.py`: start at login.

Everything runs locally; frames are never stored.

## Tests

```bash
python -m unittest discover tests
```

They cover the gaze fit, the face feature geometry, and the engine decisions (fake camera and fake
OS backend). Nothing in them touches your camera or your settings.

## Limits

A webcam gives a rough gaze estimate: small or tightly tiled windows can be confused, which is why
monitor-only mode stays the default. Monitors are expected side by side (left to right).
