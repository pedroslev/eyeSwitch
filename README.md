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
| **Monitor + overview** | Like monitor only, and when you press Super (the all-apps view opens) your eyes choose the app: the pointer follows your gaze there and clicks after a moment. Nothing happens outside the overview. | *Full*, same as window mode. |
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
- **The model is linear on four signals**: yaw, pitch and the iris offset (h, v). A nose cue was tried in
  two calibrations and never helped (120 vs 125 px, 166 vs 169 px), and quadratic gave the same accuracy
  (166 px both) while letting cross-validation choose between the two was *worse* (177 px): with about 30
  dots that choice is noise. Features without information are dropped, inputs are clamped to the
  calibrated range, and if your pose leaves that range (the camera or your chair moved) eyeSwitch says
  so and does nothing instead of acting on a meaningless estimate.
- **The output is steadied with a fixation filter** (the I-DT idea from eye tracking): while you fixate,
  the point stays put, and it moves once several consecutive frames agree on a new place. Its radius
  comes from the noise measured during your calibration. It beat a median+EMA filter on both jitter and
  lag on real data.
- **Calibration is tied to where the camera sits.** Move the camera and recalibrate.
- **Your posture drifts while you calibrate**: in one measurement the vertical estimate slid by about 85 px
  over the ~1 minute the 36 dots take. Sit as you normally do and stay put; that is why each dot is sampled
  for as little as 30 fps allows (0.6 s).

What was tried on real calibration data (36 dots, leave-one-dot-out error; the current model scores
120 px, 84 px on the held-out check dots) and did **not** help, so it is not in the code:

| Idea | Result |
|------|--------|
| Iris offset in the head's own frame, built from the landmarks' depth | much worse (214 px): the landmark z is too noisy |
| Each eye separately, roll, nose cue, a distance proxy, eyelid opening, iris between the lids | within noise (117-125 px) |
| tan() of the angles (a pinhole-camera model), one model per monitor, kernel ridge | same or worse |
| Quadratic instead of linear | same accuracy (166 px both); picking between them by cross-validation was worse (177 px), so it is always linear |
| Head pose only, no irises | 282 px: the irises cut the error by more than half |
| 1080p instead of 720p | 7-32% less noise in pitch and iris offset (little in yaw and vertical iris); costs about 20% of one CPU core more (78% against 58% of a core at 15 fps) |

That is why the camera runs at its best resolution, up to 1080p, in every mode (4K is not tried: the iris
landmarks stop improving and the pixels only cost CPU).

The remaining error is mostly vertical (8% of the screen height against 4% of the width): the eyelid
hides part of the iris when you look up or down, and at 720p an eye is only ~36 px wide.

Each full calibration also writes `calibration_raw.npz` (the 478 landmarks, the face pose and the
face's own expression scores for every kept frame: numbers only, no images), in the same row order as
`calibration_samples.npz`, so a different model can be tried on exactly the same calibration.

In this mode the camera runs at 720p, because at 640x480 an eye is only ~20 px wide. It works best with big windows and a steady seat;
redo the full calibration when you move the camera or change your posture a lot.
The quick calibration discards a previous gaze model, since both describe the same setup.

Settings: camera, mode, delay before switching monitor (default 300 ms), delay before focusing a
window (500 ms), steadiness 1-5 (how firmly the gaze point is held still; higher is steadier and a little slower),
camera resolution (Auto by default: the best the camera really delivers, up to 1080p; or 1080p / 720p.
A gaze calibration only fits the resolution it was made at, and eyeSwitch says so if they differ),
tracking rate (30 per second by default, or 15) and whether to slow down when nothing is happening,
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
relaxes.

**The overview session.** It lives inside the system's all-apps view (GNOME's overview), and **only there**:
looking at things never clicks anywhere else. eyeSwitch asks GNOME Shell whether the overview is open (it is
told when that changes, so it costs nothing): opening it, by the Super key, by a gesture that presses `super`,
by a hot corner, starts the session, and closing it ends it. The pointer does not move and nothing is clicked
while it is closed. Where the overview cannot be seen (another desktop, Windows, macOS, which does not
have this yet), a key gesture or Super still starts a short session of a few seconds (default 6) where the pointer
follows the gaze, but **nothing is clicked by looking**: use a click gesture. The overview is drawn by the system above every window, so the test dot cannot be seen
there, and its windows are rearranged into a grid, so "the window under your gaze" is no longer where it
was on the desktop. What the system does know is where each thumbnail is, and it shows that through the
**mouse pointer**: so during the session **the pointer follows your gaze** (the system draws it above the
overview and highlights what is under it), and you pick an app by **looking at one spot for a moment**
(1.2 s by default, then it clicks) or with a "Click where I look" gesture. The session ends when you pick
something, when you move the mouse yourself, or when its time runs out; automatic focus changes stay quiet
meanwhile and resume right after a click. It needs the full calibration; both the pointer and the dwell
can be turned off in the Gestures screen. For example: open your mouth (`super`), look at an app, hold
your gaze, or raise your eyebrows (click where I look).

The pointer is moved with XTEST on X11, which looks like a real device moving (XWarpPointer does not), and
with an extra mouse-moved event on macOS, so hover effects follow it.

**Suspend and resume with a gesture.** The *Pause / resume eyeSwitch* action, and the hand option **open palm,
then close it into a fist**, toggle a **suspended** state, which is not the pause of the hotkey and the
tray. Pausing lets go of the camera (so nothing could see a gesture to come back); suspended keeps
watching but acts on nothing: no focus changes, no pointer, no other gestures, no test dot. Only the
way back is evaluated, and the face model is switched off if that way is the hand. The hotkey and the
tray always win and clear a suspension. The hand is recognized by MediaPipe's gesture recognizer (an 8 MB
model, downloaded once); the sequence (a held palm, then a fist within 2 s, then 1.5 s of cool-down) is
meant to be deliberate, so a hand resting on the keyboard does not trigger it. Show the palm facing the camera
for half a second, then close the hand; the Gestures screen shows what the camera sees as "Hand now".

**Super on the keyboard works too.** Tapping Super on its own (not Super+Tab or other shortcuts) starts the same
session: the overview opens as usual and the pointer follows your gaze, so you can pick the app with your eyes
and look at it for a moment (or use a click gesture). No gesture is needed. It is seen through X11's RECORD
extension, which only observes the keyboard (nothing is grabbed or delayed), so it is Linux (X11) only for now;
turn it off with *Also when I press Super on the keyboard* in Gestures.

**Sitting somewhere else than where you calibrated.** A calibration is for one seat position: roll the chair a few
centimetres and the same glance lands somewhere else. On real data the head moved only ~0.5 cm during a
calibration, and that movement was tangled up with where you looked (correlation -0.64 with the target), so the
position cannot be learned from it (it was tried: the coefficients exploded). A full geometric model (a ray from
the eye to the screen plane, 13 parameters) was also tried on the same data and was worse (371 px against 162 px),
because one seat position cannot pin down so many numbers. What is used instead is a **first-order geometric
correction on top of the model**, with the head position the tracker already measures (in cm): if the eye moves,
the point it aims at moves with it, and if it moves away from the screen the point moves out from the eye's own
projection on the screen, in proportion to the distance. Only the pixel density is a guess (read from the
monitors' reported size). It is on by default (*Correct for where I sit*), it is saved with each calibration (older
ones get it from their own saved rows), and the status says so when you are more than 5 cm from where you
calibrated. **How much it helps has not been measured yet**: *Check accuracy now* shows eight dots the
calibration never saw, reports the error with and without the correction and how far you are from the calibration
position, and logs it (numbers only) in `~/.config/eyeswitch/accuracy_checks.jsonl`. Each calibration also keeps a dated
copy of its files in `~/.config/eyeswitch/history/` (the last 10). Run the check at your usual seat and again after
getting up and sitting back down, and compare the two numbers.

**Train the hand.** The ready-made recognizer reads other people's hands and may misjudge yours, a palm seen
from the side, or a camera angle that is not frontal. *Train hand...* in Gestures records about 15 seconds of
your own open palm, fist and resting hand and learns to tell them apart (the way face gestures are trained,
from distances between the 21 hand points, so distance to the camera and direction do not matter). *Use the
generic one* goes back.

**A hand in view blocks the face gestures.** A palm over the face, or beside it, makes the face model invent
expressions (an open mouth, for one). While the hand option is on, any hand seen in the last second cancels
the face gestures, and a face gesture that starts to rise makes the hand watcher look at once. Without the
hand option nothing watches for hands, so this protection needs it on.

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

## CPU use

Measured with a 1080p webcam on a 12-core Linux desktop, the whole engine (camera, tracking, filters)
with a stand-in desktop so nothing was touched:

| Configuration | CPU |
|---------------|-----|
| 30 fps, OpenCV on all 12 threads (the naive setup) | 131% of one core |
| 30 fps always, OpenCV on one thread | 70% |
| **30 fps adaptive (default)** | **28%**, in the idle rate 90% of the time |
| 15 fps always | 36% |

Where it went, and what was done:

- **OpenCV used every core for trivial work.** A BGR to RGB conversion takes 0.2 ms either way, but cost
  13 times more CPU on 12 threads. `cv2.setNumThreads(1)` took the whole pipeline from 43.5 to 23 ms of
  CPU per 1080p frame, with identical results.
- **Frames nobody needs are not even decoded.** At rest the tracker drops to 10 fps (`grab` without
  `retrieve`, so the JPEG decode is skipped) and returns to full speed at the first sign of movement:
  the head moves, the gaze leaves the active window (while it wanders inside it nothing can be decided), a decision or a gesture is pending, a calibration or the
  camera view is open, or the test dot is shown. Blinks do not count: they say nothing about where you
  look. Turn **Slow down when nothing is happening** off for the fastest possible first reaction (the
  worst case when starting from rest is one idle frame, 100 ms).
- **The rest is already cheap**: the system calls around each frame cost 0.23 ms (0.7% of a core at
  30 fps), and the window is up 0.3 s after launch (MediaPipe is imported in the engine thread, not
  before the window), idling at 0.2% of a core.
- **Rewriting in another language would not help**: what is left is decoding the camera's JPEG and
  MediaPipe's models, both native code already. Python's own share is a few percent.
- **Watching for a hand costs about 8% of a core** (a recognition takes 25 to 31 ms of CPU even with no hand in
  view, so it runs 3 times a second, and 10 per second only while a palm or a fist is around, which is when a
  palm-then-fist needs timing). Suspended with no hand around is about 10% of a core. These are component
  measurements: the engine as a whole could not be re-measured with the hand option because the camera
  was in use by another program.
- Tried and discarded: MediaPipe's GPU delegate (not available in this Linux build), and cropping the image
  to the face (saves the same preparation cost, but MediaPipe estimates the head pose assuming the whole
  image, so a moving crop would bias yaw and pitch).

**The camera's resolution changes what it measures.** With the head at rest, the same face gave different
numbers at 720p and 1080p (yaw 4.4 degrees, pitch 9.2 degrees, vertical iris 0.059; webcams often use a
different field of view per mode and MediaPipe assumes one). That is why a gaze calibration is stored
with the resolution it was made at, and why a calibration made at another one is reported instead of
used.

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
