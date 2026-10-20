# eyeSwitch

Linux (X11) take on [Glance Switch](https://glanceswitch.com/): the webcam tells which
monitor your head is facing and keyboard focus (and the pointer) follow you there.
Runs as a desktop app with a tray icon and a small settings window.

## Install

```bash
pip install -r requirements.txt     # the face model downloads itself on first run
sudo apt install gir1.2-ayatanaappindicator3-0.1   # tray icon (optional but recommended)
./install.sh                        # adds "eyeSwitch" to your application menu
```

`python3-gi` (GTK3 bindings) comes with Ubuntu GNOME. Re-run `./install.sh` if you move
this folder; `./install.sh --uninstall` removes the menu entry.

## Use

1. Open **eyeSwitch** from the application menu.
2. Click **Calibrate** and look at the dot on each monitor (~5 s each, Esc cancels).
3. Keep it running. Closing the window hides it to the tray; Quit is in the tray menu.

Settings: camera, delay before switching (default 300 ms), start at login.
Ctrl+Shift+G pauses and resumes; while paused the camera is released.
Nothing happens while you are using the mouse (and for 1.5 s after), and glances
shorter than the delay are ignored. Recalibrate when you move the webcam or the monitors.

Settings and calibration live in `~/.config/eyeswitch/`.

## How it works

- `tracker.py`: MediaPipe Face Landmarker, head yaw from the face transform matrix (~15 fps, 640x480).
- `xwin.py`: monitors via RandR, window focus via EWMH `_NET_ACTIVE_WINDOW`, pointer warp, global hotkey.
- `engine.py`: background loop, nearest-monitor choice with hysteresis, dwell timer,
  restores the last focused window and pointer position on the target monitor.
- `eyeswitch.py`: GTK3 app (settings window, calibration, tray, autostart).

Everything runs locally; frames are never stored.

## Limits

X11 only (Wayland blocks focus and pointer control). Head yaw only, no eye gaze yet,
so it distinguishes monitors but not windows within one monitor.
