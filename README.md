# eyeSwitch

Linux (X11) take on [Glance Switch](https://glanceswitch.com/): the webcam tells which
monitor your head is facing and keyboard focus (and the pointer) follow you there.

## Setup

```bash
pip install -r requirements.txt   # the face model downloads itself on first run
```

If OpenCV complains about GUI support, make sure only `opencv-python` is installed
(not `opencv-python-headless` next to it).

## Use

```bash
python3 eyeswitch.py --calibrate   # look at the dot on each monitor, ~5 s each
python3 eyeswitch.py               # run (calibrates first if needed)
python3 eyeswitch.py --debug       # live yaw / target, handy for tuning
```

Options: `--camera N` (`/dev/videoN`), `--delay MS` (default 300).

Ctrl+Shift+G pauses and resumes. Nothing happens while you are using the mouse
(and for 1.5 s after), and brief glances shorter than `--delay` are ignored.
Recalibrate when you move the webcam or the monitors; changing the monitor layout
invalidates the saved calibration.

## How it works

- `tracker.py`: MediaPipe Face Landmarker, head yaw from the face transform matrix (~15 fps, 640x480).
- `xwin.py`: monitors via RandR, window focus via EWMH `_NET_ACTIVE_WINDOW`, pointer warp, global hotkey.
- `eyeswitch.py`: calibration (median yaw per monitor), nearest-monitor choice with hysteresis,
  dwell timer, and restoring the last focused window and pointer position on the target monitor.

Everything runs locally; frames are never stored.

## Limits

X11 only (Wayland blocks focus and pointer control). Head yaw only, no eye gaze yet,
so it distinguishes monitors but not windows within one monitor.
