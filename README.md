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
| **Monitor and window** | Focus follows the window you are looking at, on any monitor. | *Full*: follow a dot over a 5x3 grid on each monitor, ~1 min total. |

3. Keep it running. Closing the window hides it to the tray; Quit is in the tray menu.

Window mode estimates where on the desktop you are looking from your head pose and the position
of your irises. The full calibration fits that model and reports its estimated error (about 5% of a
monitor width is typical for a webcam), for the centre of the screens and for their edges. It also
saves `calibration_samples.npz` and `calibration_report.json` (grid and check errors per point)
next to the settings, so a bad calibration can be analyzed. Tick **Show where it thinks I am looking** to see it while you tune it: a **red dot** is your gaze, smoothed
and following you live (a One Euro filter, drawn at 60 fps with a speed cap so a long jump is seen
travelling), and a **blue ring** is the steady point that actually decides which window gets focus.
The ring holds still while you fixate and moves when you look somewhere else; the dot flows. **Camera view...** opens a live view of what the tracker sees (face mesh, eye corners in green,
irises in red, yaw/pitch, iris offsets, head position, the gaze estimate), and by default the same view
appears in the corner of the calibration screens. Use it first when the estimate looks wrong: if the
irises are not on your irises, or the face keeps getting lost, calibration cannot work.

The full calibration ends with **3 check dots per monitor** at positions it has never seen. The model
is fitted on the grid only, and the error measured on those dots is the real accuracy (reported as
good, fair or poor); the saved model then also uses them. Two guards keep a bad frame from throwing the
estimate across the screen: features that carry no information (the vertical iris offset is mostly
noise at webcam resolution) are dropped automatically, and live inputs are clamped to the range seen
while calibrating; a median over the last 5 frames removes one-frame spikes. While you calibrate the
test dot is hidden, since it would still show the previous calibration.

### How the gaze estimate is made (and why)

Window mode fits, from your calibration, a mapping from **head pose (yaw, pitch), where the nose sits
relative to the eyes, and the iris offset inside each eye** to a point on the desktop. Head pointing
carries most of the signal (at webcam resolution an eye is only a few dozen pixels wide), the irises
refine it. Things learned the hard way with real calibration data, all covered by tests:

- **Head position in the camera frame (where you sit) is not used.** A model fitted with it moved the
  estimate about 1000 px per centimetre of seat shift, and was less accurate on new dots too.
- **Blinks** are detected from the face tracker's eyelid scores against your own open-eye level and are
  dropped from calibration with a margin (the signals stay off ~0.4 s after the lids reopen); a dot that
  ends up with too few clean frames keeps being sampled for up to 3 s. In use, the estimate is held
  through a blink.
- **Frames where the head is still travelling** between dots are dropped, and the dots come in
  **random order**, because in a fixed sweep where the head points for a given dot depends on where
  it came from (about 5 degrees of yaw in real data).
- **Two feature sets compete** (head pose and iris, with and without the nose); cross-validation picks
  one, preferring the smaller on a tie. Features without information are dropped, inputs are clamped to
  the calibrated range, and if your pose leaves that range (the camera or your chair moved) eyeSwitch
  says so and does nothing instead of acting on a meaningless estimate.
- **The output is steadied with a fixation filter** (the I-DT idea from eye tracking): while you fixate,
  the point stays put, and it moves once several consecutive frames agree on a new place. Its radius
  comes from the noise measured during your calibration. It beat a median+EMA filter on both jitter and
  lag on real data.
- **Calibration is tied to where the camera sits.** Move the camera and recalibrate.

In this mode the camera runs at 720p, because at 640x480 an eye is only ~20 px wide. It works best with big windows and a steady seat;
redo the full calibration when you move the camera or change your posture a lot.
The quick calibration discards a previous gaze model, since both describe the same setup.

Settings: camera, mode, delay before switching monitor (default 300 ms), delay before focusing a
window (500 ms), steadiness 1-5 (how firmly the gaze point is held still; higher is steadier and a little slower),
window edge tolerance (keeps the focus on the active window while your gaze is within
this distance of it, `auto` follows the gaze error; raise it if focus flickers between two windows), wait after typing (2 s: window mode never changes focus inside a monitor while you
are typing), start at login. Ctrl+Shift+G pauses and resumes; while paused the camera is released.
Nothing happens while you are using the mouse (and for 1.5 s after), and glances shorter than the
delay are ignored. Recalibrate when you move the webcam or the monitors.

Settings and calibration live in `~/.config/eyeswitch/` (Linux),
`%APPDATA%\eyeSwitch` (Windows) or `~/Library/Application Support/eyeSwitch` (macOS).

## Gestures

**Gestures...** lets a facial expression press a key or click, using the face tracker's 52 expression
signals (eyebrows, mouth, jaw...). Faces rest differently (and a camera angle skews everything), so
each gesture is **trained on your face**: relax for 2 s, then do the gesture and hold it for 2.5 s.
eyeSwitch keeps the few signals that tell the two apart and rates the result: *reliable* or *weak,
retrain*. A live bar shows when it would trigger.

Per gesture you choose what it does:

- **Press keys**: `super`, `alt+tab`, `ctrl+shift+g`... (`win`, `cmd`, `meta` are all `super`).
- **Click where I look**: moves the pointer to the point you are looking at and clicks (needs the
  full calibration; otherwise it clicks where the pointer is).
- **Pause / resume eyeSwitch.**

A gesture fires once when held for its hold time (default 250 ms), and again only after the face
relaxes. After a *key* gesture the automatic focus stays quiet for a few seconds (default 6), so it
does not fight a menu or an overview you just opened. For example: raise your eyebrows (`super` opens
the overview), look at an app, bite your lip (click where I look).

Eyebrows up works well. Subtler ones, like biting a lip, depend on your face and may rate as weak.
Expressions in conversation can trigger gestures by accident: give them a longer hold, or pause
eyeSwitch with Ctrl+Shift+G during a call. Key and click injection uses XTEST (X11), SendInput
(Windows) and Quartz events (macOS; needs the Accessibility permission).

## Video calls (sharing the camera)

A webcam can be streamed by one program at a time, so while eyeSwitch is running other apps
cannot use it. What eyeSwitch does about it:

- **macOS** shares cameras natively, nothing to do. **Windows** normally shares them through its
  camera frame server (not tested yet).
- **Linux, automatic:** when another app opens the camera, eyeSwitch lets go of it and takes it
  back a few seconds after the other app is done. The call app may fail on its very first attempt
  (it already opened the device before we could react): just retry. While it waits, eyeSwitch
  shows "Camera in use by another app". Pausing (Ctrl+Shift+G) also releases the camera at once.
- **Linux, virtual camera (recommended if you take calls with eyeSwitch on):** eyeSwitch keeps the
  real camera and republishes it as **"eyeSwitch Camera"**, so a call and eyeSwitch work at the
  same time and nothing is ever contended. One-time setup (needs sudo, loads the
  [v4l2loopback](https://github.com/umlaeute/v4l2loopback) kernel module):

  ```bash
  bash scripts/setup-virtual-camera.sh
  ```

  Then tick **Share the camera with video calls** and choose "eyeSwitch Camera" in Zoom, Meet or
  Teams. The virtual camera needs eyeSwitch to be running and active (when paused, the real camera
  is free again: pick it directly). If Secure Boot is on, the module has to be signed.

## How it works

- `tracker.py`: MediaPipe Face Landmarker (~15 fps, 640x480): head yaw/pitch/position from the face
  transform matrix, iris position from the landmarks.
- `camwatch.py` / `virtualcam.py`: step aside when another app opens the camera (inotify); republish
  the camera as a virtual one (pyvirtualcam, v4l2loopback).
- `gaze.py`: ridge regression from those features to a desktop point, degree and regularization
  chosen by leave-one-point-out cross-validation (this is also the error estimate); the simplest
  model within 5% of the best wins, since complex ones overshoot at the screen edges. Also the
  fixation filter that steadies the output.
- `samples.py`: blink detection and cleaning of calibration captures (blinks, head in motion).
- `preview.py`: draws the camera debug view.
- `gestures.py`: trains a gesture from blendshapes (linear discriminant) and detects it live.
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

They cover the gaze fit (including noisy features and spikes), gesture training and detection, the
full calibration flow offscreen with a simulated camera, the face feature geometry, and the engine
decisions, including sticky window edges, gestures and camera sharing (fake camera, fake OS backend). Nothing in them touches your camera or your settings.

## Limits

A webcam gives a rough gaze estimate: small or tightly tiled windows can be confused, which is why
monitor-only mode stays the default. Monitors are expected side by side (left to right).
