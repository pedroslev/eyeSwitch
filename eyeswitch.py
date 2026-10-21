#!/usr/bin/env python3
"""eyeSwitch: move keyboard focus to the monitor (or window) you are looking at (webcam)."""
import ctypes.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
SINGLE_INSTANCE = "eyeswitch-single-instance"


def check_linux_qt_deps():
    """Qt >= 6.5 aborts without libxcb-cursor; say so instead of dying silently."""
    if (not sys.platform.startswith("linux") or os.environ.get("QT_QPA_PLATFORM")
            or ctypes.util.find_library("xcb-cursor")):
        return
    msg = ("eyeSwitch needs the libxcb-cursor library.\n"
           "Debian/Ubuntu: sudo apt install libxcb-cursor0\n"
           "Fedora: sudo dnf install xcb-util-cursor\n"
           "Arch: sudo pacman -S xcb-util-cursor\n"
           "openSUSE: sudo zypper install libxcb-cursor0")
    print(msg, file=sys.stderr)
    try:
        subprocess.run(["notify-send", "-i", "dialog-error", "eyeSwitch cannot start", msg], timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass
    sys.exit(1)


check_linux_qt_deps()

import numpy as np  # noqa: E402
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtGui import (QAction, QColor, QFont, QGuiApplication, QIcon, QImage, QPainter,  # noqa: E402
                           QPen, QPixmap)
from PySide6.QtNetwork import QLocalServer, QLocalSocket  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,  # noqa: E402
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMenu, QMessageBox,
                               QProgressBar, QPushButton, QSpinBox, QSystemTrayIcon, QVBoxLayout,
                               QWidget)

import autostart  # noqa: E402
import backends  # noqa: E402
import gaze  # noqa: E402
import gestures  # noqa: E402
import virtualcam  # noqa: E402
from engine import (CONFIG_DIR, GAZE_COLUMN_SETS, Engine, load_config, nearest_monitor,  # noqa: E402
                    save_config)
from tracker import Face, list_cameras  # noqa: E402

MODES = [("Monitor only (head direction)", "monitor"), ("Monitor and window (gaze)", "window")]
GRID_X = (0.06, 0.28, 0.5, 0.72, 0.94)  # dot positions of the full calibration, as screen fractions
GRID_Y = (0.08, 0.5, 0.92)


def icon(name):
    return QIcon(os.path.join(DATA_DIR, name))


def screen_for_monitor(i, count):
    """Qt screen matching the i-th monitor from left to right."""
    screens = sorted(QGuiApplication.screens(), key=lambda s: (s.geometry().x(), s.geometry().y()))
    return screens[i] if len(screens) == count else QGuiApplication.primaryScreen()


def to_qimage(bgr):
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, 3 * w, QImage.Format_BGR888).copy()      # copy: detach from the array


class CalibrationWindow(QWidget):
    """Fullscreen black window with one dot, shown on a given screen."""
    PREVIEW_W = 320

    def __init__(self, screen, on_cancel):
        super().__init__()
        self.text, self.on_cancel, self.preview = "", on_cancel, None
        self.fx, self.fy, self.sampling = 0.5, 0.5, False
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())

    def set_text(self, text):
        self.text = text
        self.update()

    def set_dot(self, fx, fy, sampling=False):
        self.fx, self.fy, self.sampling = fx, fy, sampling
        self.update()

    def set_preview(self, image):
        self.preview = image
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.on_cancel()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        c = QPointF(self.width() * self.fx, self.height() * self.fy)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(60, 220, 120) if self.sampling else QColor(255, 200, 0))
        p.drawEllipse(c, 18, 18)
        p.setBrush(Qt.black)
        p.drawEllipse(c, 4, 4)
        p.setPen(Qt.white)
        p.setFont(QFont(self.font().family(), 24))
        p.drawText(self.rect().adjusted(0, int(self.height() * 0.30), 0, 0), Qt.AlignHCenter | Qt.AlignTop,
                   self.text)
        if self.preview is not None:              # what the tracker sees, in the bottom-right corner
            w = self.PREVIEW_W
            h = w * self.preview.height() / self.preview.width()
            p.drawImage(QRectF(self.width() - w - 24, self.height() - h - 24, w, h), self.preview)


class CalibrationBase:
    def __init__(self, engine, on_done):
        self.engine, self.on_done = engine, on_done
        self.monitors = list(engine.monitors)
        self.win, self.cancelled, self.preview_timer = None, False, None
        engine.calibrating = True
        if engine.cfg["calibration_preview"]:
            engine.preview_wanted += 1
            self.preview_timer = QTimer()
            self.preview_timer.timeout.connect(self._pump_preview)
            self.preview_timer.start(70)

    def _pump_preview(self):
        if self.win is not None and self.engine.preview is not None:
            self.win.set_preview(to_qimage(self.engine.preview))

    def cancel(self):
        self.cancelled = True

    def _open_window(self, mon_idx):
        self._close_window()
        self.win = CalibrationWindow(screen_for_monitor(mon_idx, len(self.monitors)), self.cancel)
        self.win.showFullScreen()
        self.win.activateWindow()

    def _close_window(self):
        if self.win:
            self.win.close()
            self.win = None

    def _finish(self, ok, message):
        self.engine.calibrating = False
        if self.preview_timer:
            self.preview_timer.stop()
            self.engine.preview_wanted -= 1
            self.preview_timer = None
        self._close_window()
        self.on_done(ok, message)

    def _check_spacing(self, yaws):
        s = sorted(yaws)
        if min(b - a for a, b in zip(s, s[1:])) < 5:
            self._finish(False, "The head angle between monitors is under 5 degrees. "
                                "Turn your head more clearly towards each dot.")
            return False
        return True


class QuickCalibration(CalibrationBase):
    """Per monitor: 2.5 s to turn the head, 2.5 s of samples; stores the median head yaw."""

    def __init__(self, engine, on_done):
        super().__init__(engine, on_done)
        self.yaws, self.i = [], 0
        self._step()

    def _step(self):
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        n = len(self.monitors)
        if self.i == n:
            if self._check_spacing(self.yaws):
                self.engine.set_calibration(self.yaws, self.monitors)
                self._finish(True, "Calibration saved.")
            return
        self._open_window(self.i)
        self.win.set_text(f"Monitor {self.i + 1} of {n}: look at the dot")
        QTimer.singleShot(2500, self._begin_capture)

    def _begin_capture(self):
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        self.engine.begin_capture()
        self.win.set_text("Hold still...")
        QTimer.singleShot(2500, self._end_capture)

    def _end_capture(self):
        faces = self.engine.end_capture()
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        if len(faces) < 10:
            return self._finish(False, f"No face detected while looking at monitor {self.i + 1}.")
        self.yaws.append(statistics.median(f.yaw for f in faces))
        self.i += 1
        QTimer.singleShot(0, self._step)


class FullCalibration(CalibrationBase):
    """Follow a dot over a 5x3 grid on each monitor with your eyes, then a few check dots.

    The model is fitted on the grid; the check dots are new positions it has never seen, so the
    error measured on them is the real accuracy, not a promise. The final model uses all of them.

    Blinks and frames where the head is still travelling are dropped from every dot, and a dot that
    ends up with too few clean frames is shown again once. The dots come in random order: in a fixed
    sweep, where the head points for a given dot depends on where it came from (about 5 degrees of
    yaw in real data), which the fit then mistakes for a property of the position.
    """

    SETTLE_MS, SAMPLE_MS, INTRO_MS, CHECK_MS = 700, 600, 1800, 1200
    TARGET_CLEAN, MAX_SAMPLE_MS = 10, 3000           # keep sampling until this many clean frames, up to this long
    CHECKS = ((0.17, 0.30), (0.62, 0.70), (0.83, 0.30))
    MIN_CLEAN = 6                                    # clean frames a dot needs to count

    def __init__(self, engine, on_done):
        super().__init__(engine, on_done)
        rng = random.Random()
        self.steps = []                              # (monitor, fx, fy, kind)
        for mon in range(len(self.monitors)):
            grid = [(fx, fy) for fy in GRID_Y for fx in GRID_X]
            rng.shuffle(grid)
            self.steps += [(mon, fx, fy, "grid") for fx, fy in grid]
            self.steps += [(mon, fx, fy, "check") for fx, fy in self.CHECKS]
        self.samples, self.raw, self.i, self.win_mon = [], [], 0, None
        self.retries, self.dropped_blink, self.dropped_move = {}, 0, 0
        self.t0 = time.monotonic()
        engine.want_shapes = True                    # blink scores need the face blendshapes
        self._next()

    def _finish(self, ok, message):
        self.engine.want_shapes = False
        super()._finish(ok, message)

    def _next(self):
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        if self.i == 0 and time.monotonic() - self.engine.shapes_seen > 0.5:      # camera not ready yet
            if time.monotonic() - self.t0 > 12:
                return self._finish(False, "No face data came from the camera. Is another app using it?")
            QTimer.singleShot(100, self._next)
            return
        if self.i == len(self.steps):
            return self._solve()
        mon, fx, fy, kind = self.steps[self.i]
        delay = self.SETTLE_MS
        if mon != self.win_mon:
            self._open_window(mon)
            self.win_mon = mon
            self.win.set_text(f"Monitor {mon + 1} of {len(self.monitors)}: follow the dot with your eyes")
            delay += self.INTRO_MS
        elif kind == "check" and self.steps[self.i - 1][3] == "grid":
            self.win.set_text("Now a quick accuracy check: keep following the dot")
            delay += self.CHECK_MS
        self.win.set_dot(fx, fy)
        QTimer.singleShot(delay, self._begin)

    def _begin(self):
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        self.win.set_text("")
        self.win.set_dot(*self.steps[self.i][1:3], sampling=True)
        self.engine.begin_capture()
        self.sample_t0 = time.monotonic()
        QTimer.singleShot(self.SAMPLE_MS, self._end)

    def _end(self):
        waited = (time.monotonic() - self.sample_t0) * 1000
        if (not self.cancelled and waited < self.MAX_SAMPLE_MS
                and self.engine.capture_clean_count() < self.TARGET_CLEAN):
            return QTimer.singleShot(250, self._end)         # a blink got in the way: keep looking
        faces = self.engine.end_capture()
        stats = self.engine.last_capture
        self.dropped_blink += stats.get("blink", 0)
        self.dropped_move += stats.get("moving", 0)
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        if len(faces) < self.MIN_CLEAN:
            if self.retries.get(self.i, 0) < 1:      # a blink or a glance away: once more
                self.retries[self.i] = self.retries.get(self.i, 0) + 1
                self._next()
                self.win.set_text("One more time: keep looking at the dot")
                return
            return self._finish(False, f"Not enough clean frames at point {self.i + 1} (blinks, movement or "
                                       "no face). Look at each dot steadily and try again.")
        self.samples.append(faces)
        self.raw.append(list(getattr(self.engine, "last_capture_raw", [])))
        self.i += 1
        QTimer.singleShot(0, self._next)

    def _rows(self, picks):
        feats, targets, groups = [], [], []
        for k in picks:
            mon, fx, fy, _ = self.steps[k]
            m = self.monitors[mon]
            for face in self.samples[k]:
                feats.append(np.asarray(face, float))
                targets.append((m.x + fx * m.w, m.y + fy * m.h))
                groups.append(k)
        return np.array(feats), np.array(targets), np.array(groups)

    def _save_raw(self):
        """Raw landmarks and face pose of every kept frame, in the same order as the saved feature rows,
        so a different model can be tried on exactly the same calibration later. Numbers only."""
        flat = [r for per_dot in self.raw for r in per_dot]
        if not flat or any(r is None for r in flat) or len(flat) != sum(len(s) for s in self.samples):
            return
        extra = {}
        if all(len(r) > 3 and r[3] for r in flat):                      # the face's own expression scores
            names = sorted(flat[0][3])
            extra = {"blendshape_names": np.array(names),
                     "blendshapes": np.array([[r[3].get(n, 0.0) for n in names] for r in flat], np.float32)}
        np.savez_compressed(os.path.join(CONFIG_DIR, "calibration_raw.npz"),
                            landmarks=np.stack([r[0] for r in flat]), matrices=np.stack([r[1] for r in flat]),
                            frame_wh=np.array(flat[0][2]), **extra)

    def _solve(self):
        grid = [k for k, st in enumerate(self.steps) if st[3] == "grid"]
        check = [k for k, st in enumerate(self.steps) if st[3] == "check"]
        yaws = [statistics.median(f.yaw for f in self.samples[k])        # the centre dot doubles as the
                for k in grid if self.steps[k][1:3] == (0.5, 0.5)]       # head-yaw calibration
        if not self._check_spacing(yaws):
            return

        held_out = gaze.fit(*self._rows(grid), column_sets=GAZE_COLUMN_SETS)
        check_errors = []
        for k in check:
            feats, targets, _ = self._rows([k])
            guess = np.median([held_out.predict(f) for f in feats], axis=0)
            check_errors.append(float(np.hypot(*(guess - targets[0]))))
        model = gaze.fit(*self._rows(grid + check), column_sets=GAZE_COLUMN_SETS)
        self.engine.set_calibration(yaws, self.monitors, model, getattr(self.engine, "active_height", None))

        width = statistics.mean(m.w for m in self.monitors)
        edge, inner, report = [], [], []
        for k, (mon, fx, fy, kind) in enumerate(self.steps):
            err = model.point_errors.get(k)
            on_edge = fx in (GRID_X[0], GRID_X[-1]) or fy in (GRID_Y[0], GRID_Y[-1])
            if err is not None and kind == "grid":
                (edge if on_edge else inner).append(err)
            report.append({"monitor": mon + 1, "x": fx, "y": fy, "kind": kind, "error_px": err,
                           "samples": len(self.samples[k]), "edge": on_edge})
        mean_check, worst_check = statistics.mean(check_errors), max(check_errors)
        used = [Face._fields[i] for i in model.active]
        try:                      # kept so a bad calibration can be analyzed offline
            feats, targets, groups = self._rows(range(len(self.steps)))
            os.makedirs(CONFIG_DIR, exist_ok=True)
            np.savez_compressed(os.path.join(CONFIG_DIR, "calibration_samples.npz"), features=feats,
                                targets=targets, groups=groups, fields=np.array(Face._fields),
                                steps=np.array([(m, x, y, 0 if kd == "grid" else 1) for m, x, y, kd in self.steps]))
            self._save_raw()
            with open(os.path.join(CONFIG_DIR, "calibration_report.json"), "w") as f:
                json.dump({"degree": model.degree, "features": used, "error_px": model.error_px,
                           "noise_px": model.noise_px, "check_mean_px": mean_check,
                           "check_worst_px": worst_check, "dropped_blink_frames": self.dropped_blink,
                           "dropped_moving_frames": self.dropped_move, "points": report}, f, indent=2)
        except OSError:
            pass

        share = mean_check / width
        verdict = "good" if share < 0.08 else "fair" if share < 0.14 else "poor"
        message = (f"Calibration saved.\nCheck on {len(check)} new dots: average error {mean_check:.0f} px "
                   f"({100 * share:.0f}% of a monitor width), worst {worst_check:.0f} px: {verdict}.")
        if edge and inner:
            message += (f"\nGrid points, centre ~{statistics.mean(inner):.0f} px, "
                        f"edges ~{statistics.mean(edge):.0f} px.")
        message += (f"\nUses: {', '.join(used)}. Dropped {self.dropped_blink} blink and "
                    f"{self.dropped_move} head-motion frames.")
        if verdict == "poor":
            message += ("\n\nThat is not good enough for window mode. Sit as you normally do, keep the camera "
                        "still and follow each dot with your head and eyes, then calibrate again. The camera "
                        "preview shows whether your face and irises are being detected.")
        self._finish(True, message)


MAX_DOT_SPEED = 4000.0       # px/s: a long jump is drawn as a run across the screen, not a tug


def glide(pos, target, dt, tau=0.07, vmax=MAX_DOT_SPEED):
    """Move `pos` towards `target`: an exponential approach (smooth whatever the frame rate) whose
    step is capped at `vmax` px/s, so even a screen-wide jump of the estimate is seen travelling."""
    a = 1.0 - math.exp(-dt / tau)
    dx, dy = (target[0] - pos[0]) * a, (target[1] - pos[1]) * a
    step = math.hypot(dx, dy)
    if step > vmax * dt:
        dx, dy = dx * vmax * dt / step, dy * vmax * dt / step
    return pos[0] + dx, pos[1] + dy


class GazeDot(QWidget):
    """Click-through marker that glides to where the gaze model puts you.

    kind "live": a filled dot, the continuously smoothed estimate. kind "held": a ring, the steady
    point that decisions use. The ring jumps when you fixate somewhere new, the dot flows.
    """
    SIZES = {"live": 26, "held": 48}

    def __init__(self, kind):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.WindowTransparentForInput)
        self.kind, self.size_px = kind, self.SIZES[kind]
        self.target = self.pos_f = None
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.resize(self.size_px, self.size_px)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        box = self.rect().adjusted(3, 3, -3, -3)
        if self.kind == "live":
            p.setBrush(QColor(255, 60, 60, 170))
            p.setPen(QPen(Qt.white, 2))
        else:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(70, 170, 255, 230), 4))
        p.drawEllipse(box)

    @staticmethod
    def qt_point(point, monitors):
        """Gaze point in desktop (backend) pixels -> position in Qt's coordinates."""
        idx = nearest_monitor(point[0], point[1], monitors)
        m, geo = monitors[idx], screen_for_monitor(idx, len(monitors)).geometry()
        fx = min(max((point[0] - m.x) / m.w, 0), 1)
        fy = min(max((point[1] - m.y) / m.h, 0), 1)
        return geo.x() + fx * geo.width(), geo.y() + fy * geo.height()

    def set_target(self, point, monitors):
        if point is None:
            self.target = self.pos_f = None
            self.hide()
            return
        self.target = self.qt_point(point, monitors)
        if self.pos_f is None:
            self.pos_f = self.target

    def tick(self, dt):
        if self.target is None:
            return
        self.pos_f = glide(self.pos_f, self.target, dt)
        self.move(int(self.pos_f[0] - self.size_px / 2), int(self.pos_f[1] - self.size_px / 2))
        if not self.isVisible():
            self.show()


ACTIONS = [("Press keys", "keys"), ("Click where I look", "click"), ("Pause / resume eyeSwitch", "toggle")]


class TrainDialog(QDialog):
    """Records a relaxed face, then the gesture, and trains the model from the two."""
    REACT_MS, RELAXED_MS, GESTURE_MS = 1500, 2000, 2500       # time to get ready, then to record each

    def __init__(self, parent, engine, on_done):
        super().__init__(parent)
        self.setWindowTitle("Train gesture")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.engine, self.on_done, self.finished_ = engine, on_done, False
        self.relaxed = None
        self.text = QLabel("Getting the camera ready...")
        f = self.text.font()
        f.setPointSize(f.pointSize() + 3)
        f.setBold(True)
        self.text.setFont(f)
        self.text.setAlignment(Qt.AlignCenter)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(self.text)
        layout.addWidget(cancel)
        engine.want_shapes = engine.calibrating = True       # camera on, blendshapes on, no auto focus
        self.t0 = time.monotonic()
        self.poll = QTimer(self)
        self.poll.timeout.connect(self._ready)
        self.poll.start(100)

    def _ready(self):
        if time.monotonic() - self.engine.shapes_seen < 0.5:
            self.poll.stop()
            self._say("Relax your face and look at the screen", self.REACT_MS, self._start_relaxed)
        elif time.monotonic() - self.t0 > 12:
            self._finish(None, "No face data came from the camera. Is another app using it?")

    def _say(self, text, delay_ms, then):
        self.text.setText(text)
        QTimer.singleShot(delay_ms, lambda: not self.finished_ and then())

    def _start_relaxed(self):
        self.engine.begin_shape_capture()
        self.text.setText("Hold still... (relaxed)")
        QTimer.singleShot(self.RELAXED_MS, lambda: not self.finished_ and self._end_relaxed())

    def _end_relaxed(self):
        self.relaxed = self.engine.end_shape_capture()
        if len(self.relaxed) < 15:
            return self._finish(None, "Your face was not detected. Face the camera and try again.")
        self._say("Now make the gesture and HOLD it", self.REACT_MS, self._start_gesture)

    def _start_gesture(self):
        self.engine.begin_shape_capture()
        self.text.setText("Keep holding it...")
        QTimer.singleShot(self.GESTURE_MS, lambda: not self.finished_ and self._end_gesture())

    def _end_gesture(self):
        shown = self.engine.end_shape_capture()
        if len(shown) < 15:
            return self._finish(None, "Your face was not detected. Face the camera and try again.")
        self._finish(gestures.train(self.relaxed, shown), "")

    def _finish(self, model, message):
        if self.finished_:
            return
        self.finished_ = True
        self.engine.want_shapes = self.engine.calibrating = False
        self.accept()
        self.on_done(model, message)

    def reject(self):
        if not self.finished_:
            self.finished_ = True
            self.engine.want_shapes = self.engine.calibrating = False
        super().reject()


class GestureRow(QWidget):
    def __init__(self, dialog, binding):
        super().__init__()
        self.dialog, self.binding = dialog, binding
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit(binding["label"])
        self.name.setFixedWidth(120)
        self.action = QComboBox()
        for label, value in ACTIONS:
            self.action.addItem(label, value)
        self.action.setCurrentIndex(max(self.action.findData(binding.get("action", "keys")), 0))
        self.action.setMinimumWidth(200)
        self.keys = QLineEdit(binding.get("keys", ""))
        self.keys.setPlaceholderText("super, alt+tab, ctrl+shift+g")
        self.keys.setFixedWidth(170)
        self.hold = QSpinBox()
        self.hold.setRange(100, 1500)
        self.hold.setSingleStep(50)
        self.hold.setSuffix(" ms")
        self.hold.setValue(binding.get("hold_ms", 250))
        self.hold.setToolTip("How long the gesture must be held")
        self.hold.setFixedWidth(95)
        self.train_btn = QPushButton("Train")
        self.status = QLabel()
        self.status.setFixedWidth(150)
        self.bar = QProgressBar()
        self.bar.setRange(0, 150)
        self.bar.setTextVisible(False)
        self.bar.setFixedWidth(70)
        self.bar.setToolTip("Live: fills past two thirds when the gesture is detected")
        remove = QPushButton("\u2715")
        remove.setFixedWidth(30)
        for w in (self.name, self.action, self.keys, self.hold, self.train_btn, self.status, self.bar, remove):
            row.addWidget(w)
        self.name.editingFinished.connect(self._changed)
        self.action.currentIndexChanged.connect(self._changed)
        self.keys.editingFinished.connect(self._changed)
        self.hold.valueChanged.connect(self._changed)
        self.train_btn.clicked.connect(lambda: dialog.train(self))
        remove.clicked.connect(lambda: dialog.remove(self))
        self._changed()
        self.show_status()

    def _changed(self, *_):
        b = self.binding
        b["label"], b["action"] = self.name.text().strip() or "Gesture", self.action.currentData()
        b["keys"], b["hold_ms"] = self.keys.text().strip(), self.hold.value()
        self.keys.setEnabled(b["action"] == "keys")
        self.dialog.engine.gestures_changed()

    def show_status(self):
        model = self.binding.get("model")
        if not model:
            self.status.setText("not trained")
        else:
            m = gestures.GestureModel.from_dict(model)
            self.status.setText(("reliable" if m.reliable else "weak: retrain") + f" ({m.quality:.1f})")
            self.status.setToolTip("Uses: " + ", ".join(m.names))


class HandTrainDialog(QDialog):
    """Records the user's open palm, fist and resting hand, and trains the hand poses from them."""
    REACT_MS, RECORD_MS, MIN_SAMPLES = 1800, 2500, 12
    STEPS = (("open", "Show your OPEN PALM to the camera, the way you will do it, and hold it"),
             ("fist", "Now close it into a FIST and hold it"),
             ("rest", "Now your hand as it usually is: resting on the keyboard or the mouse, relaxed"))

    def __init__(self, parent, engine, on_done):
        super().__init__(parent)
        self.setWindowTitle("Train hand")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.engine, self.on_done, self.finished_, self.samples, self.step = engine, on_done, False, {}, 0
        self.text = QLabel("Getting the camera ready...")
        f = self.text.font()
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        self.text.setFont(f)
        self.text.setWordWrap(True)
        self.text.setAlignment(Qt.AlignCenter)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(self.text)
        layout.addWidget(cancel)
        engine.hand_wanted += 1
        engine.calibrating = True                          # camera on, nothing acts meanwhile
        self.t0 = time.monotonic()
        self.poll = QTimer(self)
        self.poll.timeout.connect(self._ready)
        self.poll.start(100)

    def _ready(self):
        if getattr(self.engine, "hand_text", ""):          # the recognizer has answered: it is running
            self.poll.stop()
            self._next()
        elif time.monotonic() - self.t0 > 20:
            self._finish(None, "The hand recognizer did not start. Is another app using the camera?")

    def _later(self, ms, then):
        QTimer.singleShot(ms, lambda: not self.finished_ and then())

    def _next(self):
        if self.step >= len(self.STEPS):
            return self._train()
        self.text.setText(f"{self.step + 1}/3  {self.STEPS[self.step][1]}")
        self._later(self.REACT_MS, self._record)

    def _record(self):
        self.engine.begin_hand_capture()
        self.text.setText(self.text.text() + "\n...recording")
        self._later(self.RECORD_MS, self._stop)

    def _stop(self):
        name = self.STEPS[self.step][0]
        self.samples[name] = self.engine.end_hand_capture()
        if len(self.samples[name]) < self.MIN_SAMPLES:
            return self._finish(None, "Your hand was not detected clearly enough. Keep it in view of the camera, "
                                      "not too close, and try again.")
        self.step += 1
        self._next()

    def _train(self):
        model = gestures.HandPoses.train(self.samples["open"], self.samples["fist"], self.samples["rest"])
        self._finish(model, "")

    def _finish(self, model, message):
        if self.finished_:
            return
        self.finished_ = True
        self.engine.hand_wanted = max(0, self.engine.hand_wanted - 1)
        self.engine.calibrating = False
        self.accept()
        self.on_done(model, message)

    def reject(self):
        if not self.finished_:
            self.finished_ = True
            self.engine.hand_wanted = max(0, self.engine.hand_wanted - 1)
            self.engine.calibrating = False
        super().reject()


class GesturesDialog(QDialog):
    def __init__(self, parent, engine, cfg):
        super().__init__(parent)
        self.engine, self.g = engine, cfg["gestures"]
        self.setWindowTitle("Gestures")
        self.setMinimumWidth(920)
        self.enabled = QCheckBox("Enable gestures")
        self.enabled.setChecked(self.g["enabled"])
        self.enabled.toggled.connect(self._enabled)
        self.pause = QSpinBox()
        self.pause.setRange(0, 30)
        self.pause.setSuffix(" s")
        self.pause.setValue(int(self.g.get("pause_s", 6)))
        self.pause.valueChanged.connect(self._pause)
        self.cursor = QCheckBox("After a key gesture (like Super), the pointer follows my gaze")
        self.cursor.setChecked(self.g.get("gaze_cursor", True))
        self.cursor.setToolTip("The overview is drawn by the system above every window, so the test dot cannot be "
                               "seen there. The mouse pointer can: it follows your gaze and the system highlights "
                               "what is under it. Needs the full calibration.")
        self.cursor.toggled.connect(self._cursor)
        self.kb_super = QCheckBox("Also when I press Super on the keyboard")
        self.kb_super.setChecked(self.g.get("keyboard_super", True))
        self.kb_super.setToolTip("Tapping Super on its own (not Super+Tab...) opens the overview, and the pointer "
                                 "follows your gaze there just as after a gesture. Linux (X11) only for now.")
        self.kb_super.toggled.connect(self._kb_super)
        self.hand = QCheckBox("Pause / resume eyeSwitch with a hand: open palm, then close it into a fist")
        self.hand.setChecked(self.g.get("hand_toggle", False))
        self.hand.setToolTip("Suspended means the flag is off but the camera keeps watching, so the same gesture "
                             "brings it back; nothing else is acted on meanwhile. It downloads an 8 MB model "
                             "once and costs about 10% of a CPU core (a recognition is slow, so it looks 3 times a second, "
                             "10 while a hand is around). Show your palm to the camera for a moment, then close "
                             "your hand.")
        self.hand.toggled.connect(self._hand)
        self.hand_live = QLabel()
        self.hand_live.setEnabled(False)
        self.hand_train = QPushButton("Train hand...")
        self.hand_train.setToolTip("Teach eyeSwitch your own open palm, fist and resting hand (about 15 seconds). "
                                   "Without it the ready-made recognizer is used, which reads other people's hands.")
        self.hand_train.clicked.connect(self._train_hand)
        self.hand_reset = QPushButton("Use the generic one")
        self.hand_reset.clicked.connect(self._reset_hand)
        self.hand_status = QLabel()
        self.hand_status.setWordWrap(True)
        self._hand_status()
        self.dwell = QSpinBox()
        self.dwell.setRange(0, 5000)
        self.dwell.setSingleStep(100)
        self.dwell.setSuffix(" ms")
        self.dwell.setSpecialValueText("off")
        self.dwell.setValue(int(self.g.get("dwell_ms", 1200)))
        self.dwell.setToolTip("During that time, looking at one spot for this long clicks it, so you can pick "
                              "an app with your eyes alone. It only clicks while the all-apps view (GNOME's overview) is open, "
                              "never anywhere else. Off: use a click gesture instead.")
        self.dwell.valueChanged.connect(self._dwell)
        form = QFormLayout()
        form.addRow("How long that lasts (no automatic focus changes either)", self.pause)
        form.addRow("Click after looking at one spot for", self.dwell)
        self.rows_box = QVBoxLayout()
        add = QPushButton("Add gesture")
        add.clicked.connect(self.add)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        self.log = QLabel()
        self.log.setEnabled(False)
        hint = QLabel("Train each gesture with your own face: relax, then do it and hold it. Eyebrows up "
                      "works well; subtler ones (biting a lip) may come out as \u201cweak\u201d. "
                      "Tip: a gesture that presses Super opens the overview; then the pointer follows your gaze, and "
                      "you pick an app by looking at it for a moment, or with a \u201cClick where I look\u201d gesture.")
        hint.setWordWrap(True)
        hint.setEnabled(False)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(self.enabled)
        layout.addWidget(self.cursor)
        layout.addWidget(self.kb_super)
        layout.addWidget(self.hand)
        layout.addWidget(self.hand_live)
        hand_row = QHBoxLayout()
        hand_row.addWidget(self.hand_train)
        hand_row.addWidget(self.hand_reset)
        layout.addLayout(hand_row)
        layout.addWidget(self.hand_status)
        layout.addLayout(form)
        layout.addLayout(self.rows_box)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addStretch()
        buttons.addWidget(close)
        layout.addLayout(buttons)
        layout.addWidget(self.log)
        layout.addWidget(hint)
        self.rows = []
        for b in self.g["bindings"]:
            self._add_row(b)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._live)
        self.timer.start(100)

    def _add_row(self, binding):
        row = GestureRow(self, binding)
        self.rows.append(row)
        self.rows_box.addWidget(row)

    def add(self):
        names = {b["label"] for b in self.g["bindings"]}
        n = 1
        while f"Gesture {n}" in names:
            n += 1
        b = {"label": f"Gesture {n}", "action": "keys", "keys": "super", "hold_ms": 250, "model": None}
        self.g["bindings"].append(b)
        self._add_row(b)
        self.engine.gestures_changed()

    def remove(self, row):
        self.g["bindings"].remove(row.binding)
        self.rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self.engine.gestures_changed()
        self.adjustSize()

    def train(self, row):
        TrainDialog(self, self.engine, lambda model, msg: self._trained(row, model, msg)).exec()

    def _trained(self, row, model, message):
        if model is None:
            return QMessageBox.warning(self, "eyeSwitch", message)
        row.binding["model"] = model.to_dict()
        row.show_status()
        self.engine.gestures_changed()
        if not model.reliable:
            QMessageBox.warning(self, "eyeSwitch", "This gesture looks too much like your relaxed face "
                                "(separation %.1f). Try a bigger or more distinct movement and train it "
                                "again; otherwise it will trigger by accident or not at all." % model.quality)

    def _enabled(self, on):
        self.g["enabled"] = on
        self.engine.gestures_changed()

    def _hand(self, on):
        self.g["hand_toggle"] = on
        self.engine.gestures_changed()

    def _hand_status(self):
        poses = self.g.get("hand_poses")
        quality = min(poses["open"]["quality"], poses["fist"]["quality"]) if poses else None
        self.hand_status.setText(
            "Using the generic hand recognizer." if not poses else
            f"Trained on your hand (separation {quality:.1f}" + ("): good." if quality >= gestures.MIN_QUALITY else
                                                                  "): weak, train again with clearer poses."))
        self.hand_reset.setEnabled(bool(poses))

    def _train_hand(self):
        def done(model, message):
            if model is None:
                QMessageBox.warning(self, "Train hand", message)
            else:
                self.g["hand_poses"] = model.to_dict()
                self.engine.gestures_changed()
                self._hand_status()
                if not model.reliable:
                    QMessageBox.information(self, "Train hand", "The poses look too alike to tell apart reliably. "
                                            "Make the palm fully open and the fist tight, and train again.")
        HandTrainDialog(self, self.engine, done).exec()

    def _reset_hand(self):
        self.g.pop("hand_poses", None)
        self.engine.gestures_changed()
        self._hand_status()

    def _kb_super(self, on):
        self.g["keyboard_super"] = on
        self.engine.gestures_changed()

    def _cursor(self, on):
        self.g["gaze_cursor"] = on
        self.engine.gestures_changed()

    def _dwell(self, value):
        self.g["dwell_ms"] = value
        self.engine.gestures_changed()

    def _pause(self, value):
        self.g["pause_s"] = value
        self.engine.gestures_changed()

    def _live(self):
        for row in self.rows:
            score = self.engine.gesture_scores.get(row.binding["label"])
            live = self.g["enabled"] and row.binding.get("model") and score is not None
            row.bar.setValue(int(max(0, min(150, score * 100))) if live else 0)
        self.log.setText(self.engine.gesture_log)
        hand = getattr(self.engine, "hand_text", "")
        self.hand_live.setText(f"Hand now: {hand}" if self.g.get("hand_toggle") and hand else "")


class PreviewWindow(QWidget):
    """Live camera view with the face mesh, irises and the numbers the model gets."""

    def __init__(self, engine):
        super().__init__()
        self.engine, self.counted = engine, False
        self.setWindowTitle("eyeSwitch camera")
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.label = QLabel("Opening the camera...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(480, 270)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.label)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    def _tick(self):
        if self.engine.preview is not None:
            self.label.setPixmap(QPixmap.fromImage(to_qimage(self.engine.preview)))

    def showEvent(self, event):
        if not self.counted:
            self.engine.preview_wanted += 1
            self.counted = True
        self.timer.start(66)
        super().showEvent(event)

    def hideEvent(self, event):
        if self.counted:
            self.engine.preview_wanted -= 1
            self.counted = False
        self.timer.stop()
        super().hideEvent(event)


class Bridge(QObject):
    """Lets the engine thread poke the GUI thread."""
    enabled_changed = Signal()


class MainWindow(QWidget):
    def __init__(self, app, engine, cfg, tray):
        super().__init__()
        self.app, self.engine, self.cfg, self.tray = app, engine, cfg, tray
        self.calibration, self.dots, self.dot_t = None, {}, time.monotonic()
        self.setWindowTitle("eyeSwitch")
        self.setWindowIcon(icon("eyeswitch.svg"))

        self.status = QLabel()
        f = self.status.font()
        f.setPointSize(f.pointSize() + 4)
        f.setBold(True)
        self.status.setFont(f)
        self.detail = QLabel()
        self.detail.setEnabled(False)           # dimmed text

        self.enabled = QCheckBox("Active  (Ctrl+Shift+G)")
        self.enabled.setChecked(cfg["enabled"])
        self.enabled.toggled.connect(engine.set_enabled)

        self.camera = QComboBox()
        for idx, name in list_cameras():
            self.camera.addItem(f"{name} ({idx})" if sys.platform != "linux" else
                                f"{name} (/dev/video{idx})", idx)
        self.camera.setCurrentIndex(max(self.camera.findData(cfg["camera"]), 0))
        self.camera.currentIndexChanged.connect(self._on_camera)

        self.mode = QComboBox()
        for label, value in MODES:
            self.mode.addItem(label, value)
        self.mode.setCurrentIndex(max(self.mode.findData(cfg["mode"]), 0))
        self.mode.currentIndexChanged.connect(self._on_mode)

        self.delay = self._spin(QSpinBox(), 100, 2000, 50, " ms", cfg["delay_ms"], "delay_ms")
        self.window_delay = self._spin(QSpinBox(), 100, 3000, 50, " ms", cfg["window_delay_ms"],
                                       "window_delay_ms")
        self.typing_wait = self._spin(QDoubleSpinBox(), 0, 10, 0.5, " s", cfg["typing_wait_s"],
                                      "typing_wait_s")
        self.edge_px = self._spin(QSpinBox(), 0, 400, 10, " px", cfg["edge_px"], "edge_px")
        self.edge_px.setSpecialValueText("auto")
        self.edge_px.setToolTip("Keep the focus on the active window while your gaze is within this "
                                "distance of it. Raise it if focus flickers between two windows.")
        self.resolution = QComboBox()
        self.resolution.addItem("Auto: the camera's best (up to 1080p)", "auto")
        self.resolution.addItem("1080p", 1080)
        self.resolution.addItem("720p", 720)
        self.resolution.setCurrentIndex(max(self.resolution.findData(cfg["camera_height"]), 0))
        self.resolution.setToolTip("Camera resolution. Auto picks the best one the camera really delivers, up to "
                                   "1080p (more pixels give steadier pitch and iris readings; 4K is not worth "
                                   "its CPU). A gaze calibration only fits the resolution it was made at: "
                                   "calibrate again after changing it.")
        self.resolution.currentIndexChanged.connect(self._on_resolution)
        self.rate = QComboBox()
        self.rate.addItem("30 per second: faster and steadier", 30)
        self.rate.addItem("15 per second: lighter on the CPU", 15)
        self.rate.setCurrentIndex(max(self.rate.findData(cfg["tracking_fps"]), 0))
        self.rate.setToolTip("How often the face is tracked. 30 reacts sooner and averages twice the samples "
                             "for the same wait, for about 60% more CPU (around 11% of a 12-core machine). "
                             "If the camera cannot deliver 30, it simply runs at what it gives.")
        self.rate.currentIndexChanged.connect(lambda _: self._set("tracking_fps", self.rate.currentData()))
        self.adaptive = QCheckBox("Slow down when nothing is happening (saves CPU)")
        self.adaptive.setChecked(cfg["adaptive_rate"])
        self.adaptive.setToolTip("Tracks at full speed while you move your head, your eyes, blink or make a "
                                 "gesture, and at 10 per second after 1.5 s of calm, which takes about two "
                                 "thirds off the CPU. Untick it for the fastest possible first reaction.")
        self.adaptive.toggled.connect(lambda on: self._set("adaptive_rate", on))
        self.stability = self._spin(QSpinBox(), 1, 5, 1, "", cfg["stability"], "stability")
        self.stability.setToolTip("How steady the gaze point is: higher holds it still more firmly and "
                                  "follows you a little later. Raise it if focus jumps around.")
        self.show_gaze = QCheckBox("Show where it thinks I am looking (test dot)")
        self.show_gaze.setToolTip("Red dot: your gaze, smoothed and following you live. Blue ring: the steady "
                                  "point that decides which window gets focus; it holds still while you "
                                  "fixate and jumps when you look somewhere else.")
        self.show_gaze.setChecked(cfg["show_gaze"])
        self.show_gaze.toggled.connect(self._on_show_gaze)

        self.vcam = QCheckBox("Share the camera with video calls (virtual camera)")
        self.vcam.setChecked(cfg["virtual_camera"])
        self.vcam.toggled.connect(lambda on: self._set("virtual_camera", on))
        self.vcam.setVisible(virtualcam.AVAILABLE)
        self.vcam_note = QLabel()
        self.vcam_note.setWordWrap(True)
        self.vcam_note.setEnabled(False)
        self.vcam_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.vcam_note.setVisible(virtualcam.AVAILABLE)

        self.cal_preview = QCheckBox("Show the camera in a corner while calibrating")
        self.cal_preview.setChecked(cfg["calibration_preview"])
        self.cal_preview.toggled.connect(lambda on: self._set("calibration_preview", on))
        self.preview_win = PreviewWindow(engine)
        self.preview_btn = QPushButton("Camera view...")
        self.preview_btn.setToolTip("See what the tracker detects: face mesh, irises and live values")
        self.preview_btn.clicked.connect(lambda: (self.preview_win.show(), self.preview_win.raise_()))

        self.autostart = QCheckBox("Start at login")
        self.autostart.setChecked(autostart.is_enabled())
        self.autostart.toggled.connect(autostart.set_enabled)

        self.quick_btn = QPushButton("Quick calibration (monitors)")
        self.quick_btn.clicked.connect(lambda: self._calibrate(QuickCalibration))
        self.full_btn = QPushButton("Full calibration (monitors and windows)")
        self.full_btn.clicked.connect(lambda: self._calibrate(FullCalibration))
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setEnabled(False)

        self.form = QFormLayout()
        self.form.addRow("Camera", self.camera)
        self.form.addRow("Camera resolution", self.resolution)
        self.form.addRow("Tracking rate", self.rate)
        self.form.addRow("Mode", self.mode)
        self.form.addRow("Delay before switching monitor", self.delay)
        self.form.addRow("Delay before focusing a window", self.window_delay)
        self.form.addRow("Wait after typing", self.typing_wait)
        self.form.addRow("Window edge tolerance", self.edge_px)
        self.form.addRow("Steadiness (1-5)", self.stability)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)
        for w in (self.status, self.detail, self.enabled):
            layout.addWidget(w)
        layout.addLayout(self.form)
        layout.addWidget(self.adaptive)
        layout.addWidget(self.show_gaze)
        layout.addWidget(self.vcam)
        layout.addWidget(self.vcam_note)
        layout.addWidget(self.cal_preview)
        layout.addWidget(self.autostart)
        self.gestures_btn = QPushButton("Gestures...")
        self.gestures_btn.clicked.connect(lambda: GesturesDialog(self, self.engine, self.cfg).exec())
        tools = QHBoxLayout()
        tools.addWidget(self.gestures_btn)
        tools.addWidget(self.preview_btn)
        layout.addLayout(tools)
        buttons = QHBoxLayout()
        buttons.addWidget(self.quick_btn)
        buttons.addWidget(self.full_btn)
        layout.addLayout(buttons)
        layout.addWidget(self.note)
        warning = backends.unsupported_reason() or ("" if tray else
                                                    "No system tray available: closing this window quits the app.")
        if warning:
            w = QLabel(warning)
            w.setWordWrap(True)
            w.setEnabled(False)
            layout.addWidget(w)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(300)
        self.dot_timer = QTimer(self)
        self.dot_timer.timeout.connect(self._animate_dots)
        self.dot_timer.start(16)
        self.setMinimumWidth(460)
        self._apply_mode()
        self.refresh()

    def _spin(self, spin, lo, hi, step, suffix, value, key):
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setValue(value)
        spin.valueChanged.connect(lambda v: self._set(key, v))
        return spin

    def _set(self, key, value):
        self.cfg[key] = value
        save_config(self.cfg)

    def sync_enabled(self):
        """Pull `enabled` from config (hotkey or tray changed it)."""
        self.enabled.blockSignals(True)
        self.enabled.setChecked(self.cfg["enabled"])
        self.enabled.blockSignals(False)

    def _apply_mode(self):
        window = self.cfg["mode"] == "window"
        for w in (self.window_delay, self.typing_wait, self.edge_px, self.stability):
            self.form.setRowVisible(w, window)
        self.show_gaze.setVisible(window)
        self.dot_timer.setInterval(16 if window and self.cfg["show_gaze"] else 400)
        self._update_note()
        self.adjustSize()

    def _update_note(self):
        if self.engine.stale_model:
            text = ("The saved gaze calibration comes from an older version and cannot be used: "
                    "run the full calibration again.")
        elif self.engine.resolution_mismatch:
            text = (f"The gaze calibration was made at {self.engine.calib_height}p and the camera now runs at "
                    f"{self.engine.active_height}p: run the full calibration again (or change the resolution).")
        elif self.cfg["mode"] == "window" and self.engine.gaze_model is None:
            text = "Window mode needs the full calibration. Until then it works like monitor-only mode."
        elif self.cfg["mode"] == "window":
            text = ("Window mode: look at a window and it gets focus. Redo the full calibration when "
                    "you move the camera or change your seat.")
        else:
            text = "Monitor-only mode needs just the quick calibration. The full one is for window mode."
        self.note.setText(text)

    def _update_vcam_note(self):
        if not virtualcam.AVAILABLE:
            return
        state = self.engine.vcam_state
        if not self.cfg["virtual_camera"]:
            text = ("Without it, eyeSwitch lets go of the camera when another app opens it and takes it "
                    "back afterwards. With it, calls and eyeSwitch work at the same time.")
        elif state == "active":
            text = "In your video call app, choose the camera named \u201c" + virtualcam.LABEL + "\u201d."
        elif state == "not set up":
            text = ("One-time setup needed. Run this in a terminal:\n"
                    f"bash {os.path.join(HERE, 'scripts', 'setup-virtual-camera.sh')}")
        else:
            text = f"Virtual camera: {state}"
        self.vcam_note.setText(text)

    def refresh(self):
        e = self.engine
        self.status.setText(e.status)
        if e.calibrated:
            cal = "Calibrated (" + ", ".join(f"M{i + 1} {y:+.0f}°" for i, y in enumerate(e.centers)) + ")"
            if e.gaze_model is not None:
                cal += f", gaze error ~{e.gaze_model.error_px:.0f} px"
        elif e.centers is not None:
            cal = "Monitor layout changed: recalibrate"
        else:
            cal = "Not calibrated yet"
        yaw = "no face" if e.yaw is None else f"{e.yaw:+.1f}°"
        self.detail.setText(f"Head yaw: {yaw}  |  {cal}")
        self._update_note()
        self._update_vcam_note()

    def _animate_dots(self):
        now = time.monotonic()
        dt, self.dot_t = min(now - self.dot_t, 0.1), now
        e = self.engine
        show = self.cfg["mode"] == "window" and self.cfg["show_gaze"] and not e.calibrating
        for kind, point in (("live", e.gaze_live), ("held", e.gaze)):
            dot = self.dots.get(kind)
            if not show:                          # while calibrating the dots would show the old model
                if dot:
                    dot.set_target(None, e.monitors)
                continue
            if dot is None:
                dot = self.dots[kind] = GazeDot(kind)
            dot.set_target(point, e.monitors)
            dot.tick(dt)

    def _on_camera(self, _i):
        self._set("camera", self.camera.currentData())

    def _on_resolution(self, _i):
        self._set("camera_height", self.resolution.currentData())
        self._update_note()

    def _on_mode(self, _i):
        self._set("mode", self.mode.currentData())
        self._apply_mode()

    def _on_show_gaze(self, on):
        self._set("show_gaze", on)
        self._apply_mode()

    def _calibrate(self, kind):
        if len(self.engine.monitors) < 2:
            return self._message("Only one monitor detected.", QMessageBox.Warning)
        self.quick_btn.setEnabled(False)
        self.full_btn.setEnabled(False)
        self.calibration = kind(self.engine, self._calibration_done)

    def _calibration_done(self, ok, message):
        self.calibration = None
        self.quick_btn.setEnabled(True)
        self.full_btn.setEnabled(True)
        self.show_window()
        self._message(message, QMessageBox.Information if ok else QMessageBox.Warning)

    def _message(self, text, kind):
        QMessageBox(kind, "eyeSwitch", text, QMessageBox.Ok, self).exec()

    def show_window(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event):
        if self.tray:
            event.ignore()
            self.hide()
        else:
            self.app.quit()


def build_tray(window, engine, cfg, app):
    if not QSystemTrayIcon.isSystemTrayAvailable():
        return None
    tray = QSystemTrayIcon(icon("eyeswitch-tray.svg"), app)
    menu = QMenu()
    open_action = QAction("Open eyeSwitch", menu)
    open_action.triggered.connect(lambda: window.show_window())
    active = QAction("Active", menu, checkable=True)
    active.setChecked(cfg["enabled"])
    active.toggled.connect(engine.set_enabled)
    quit_action = QAction("Quit", menu)
    quit_action.triggered.connect(app.quit)
    for a in (open_action, active):
        menu.addAction(a)
    menu.addSeparator()
    menu.addAction(quit_action)
    tray.setContextMenu(menu)
    tray.setToolTip("eyeSwitch")
    tray.activated.connect(lambda reason: reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick)
                           and window.show_window())

    def sync():
        on = cfg["enabled"] and not engine.suspended
        state = ("on" if on else "suspended") if cfg["enabled"] else "off"
        tray.setIcon(icon({"on": "eyeswitch-tray.svg", "suspended": "eyeswitch-tray-suspended.svg",
                           "off": "eyeswitch-tray-paused.svg"}[state]))
        tray.setToolTip({"on": "eyeSwitch: ON", "off": "eyeSwitch: OFF (camera released)",
                         "suspended": "eyeSwitch: SUSPENDED (watching, acting on nothing; repeat the gesture to resume)"}[state])
        active.blockSignals(True)
        active.setChecked(on)
        active.blockSignals(False)

    tray.sync = sync
    tray.show()
    return tray


def already_running():
    """Ask a running instance to show its window. True if there was one."""
    sock = QLocalSocket()
    sock.connectToServer(SINGLE_INSTANCE)
    if sock.waitForConnected(300):
        sock.write(b"show")
        sock.waitForBytesWritten(300)
        return True
    return False


def main():
    background = "--background" in sys.argv
    app = QApplication([sys.argv[0]])       # first: Qt sets DPI awareness on Windows
    app.setApplicationName("eyeSwitch")
    app.setDesktopFileName("eyeswitch")
    app.setWindowIcon(icon("eyeswitch.svg"))
    app.setQuitOnLastWindowClosed(False)

    if already_running():
        return 0
    QLocalServer.removeServer(SINGLE_INSTANCE)
    server = QLocalServer()
    server.listen(SINGLE_INSTANCE)

    cfg = load_config()
    engine = Engine(cfg)
    bridge = Bridge()
    engine.on_change = bridge.enabled_changed.emit
    window = MainWindow(app, engine, cfg, QSystemTrayIcon.isSystemTrayAvailable())
    tray = build_tray(window, engine, cfg, app)

    def on_enabled_changed():
        window.sync_enabled()
        if tray:
            tray.sync()

    bridge.enabled_changed.connect(on_enabled_changed, Qt.QueuedConnection)
    server.newConnection.connect(lambda: (server.nextPendingConnection(), window.show_window()))
    app.aboutToQuit.connect(engine.stop)
    on_enabled_changed()
    if not (background and tray):
        window.show_window()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
