#!/usr/bin/env python3
"""eyeSwitch: move keyboard focus to the monitor (or window) you are looking at (webcam)."""
import ctypes.util
import os
import statistics
import subprocess
import sys

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
from PySide6.QtCore import QObject, QPointF, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QAction, QColor, QFont, QGuiApplication, QIcon, QPainter, QPen  # noqa: E402
from PySide6.QtNetwork import QLocalServer, QLocalSocket  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,  # noqa: E402
                               QFormLayout, QHBoxLayout, QLabel, QMenu, QMessageBox,
                               QPushButton, QSpinBox, QSystemTrayIcon, QVBoxLayout, QWidget)

import autostart  # noqa: E402
import backends  # noqa: E402
import gaze  # noqa: E402
import virtualcam  # noqa: E402
from engine import Engine, load_config, nearest_monitor, save_config  # noqa: E402
from tracker import list_cameras  # noqa: E402

MODES = [("Monitor only (head direction)", "monitor"), ("Monitor and window (gaze)", "window")]
GRID = (0.08, 0.5, 0.92)               # dot positions of the full calibration, as screen fractions


def icon(name):
    return QIcon(os.path.join(DATA_DIR, name))


def screen_for_monitor(i, count):
    """Qt screen matching the i-th monitor from left to right."""
    screens = sorted(QGuiApplication.screens(), key=lambda s: (s.geometry().x(), s.geometry().y()))
    return screens[i] if len(screens) == count else QGuiApplication.primaryScreen()


class CalibrationWindow(QWidget):
    """Fullscreen black window with one dot, shown on a given screen."""

    def __init__(self, screen, on_cancel):
        super().__init__()
        self.text, self.on_cancel = "", on_cancel
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


class CalibrationBase:
    def __init__(self, engine, on_done):
        self.engine, self.on_done = engine, on_done
        self.monitors = list(engine.monitors)
        self.win, self.cancelled = None, False
        engine.calibrating = True

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
    """Follow a dot over a 3x3 grid on each monitor with your eyes; fits the gaze model."""

    SETTLE_MS, SAMPLE_MS, INTRO_MS = 800, 1000, 1800

    def __init__(self, engine, on_done):
        super().__init__(engine, on_done)
        self.steps = []                              # (monitor, fx, fy), snake order per monitor
        for mon in range(len(self.monitors)):
            for row, fy in enumerate(GRID):
                for fx in (GRID if row % 2 == 0 else GRID[::-1]):
                    self.steps.append((mon, fx, fy))
        self.samples, self.i, self.win_mon = [], 0, None
        self._next()

    def _next(self):
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        if self.i == len(self.steps):
            return self._solve()
        mon, fx, fy = self.steps[self.i]
        delay = self.SETTLE_MS
        if mon != self.win_mon:
            self._open_window(mon)
            self.win_mon = mon
            self.win.set_text(f"Monitor {mon + 1} of {len(self.monitors)}: follow the dot with your eyes")
            delay += self.INTRO_MS
        self.win.set_dot(fx, fy)
        QTimer.singleShot(delay, self._begin)

    def _begin(self):
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        self.win.set_text("")
        self.win.set_dot(*self.steps[self.i][1:], sampling=True)
        self.engine.begin_capture()
        QTimer.singleShot(self.SAMPLE_MS, self._end)

    def _end(self):
        faces = self.engine.end_capture()
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        if len(faces) < 8:
            return self._finish(False, f"No face detected while looking at point {self.i + 1}.")
        self.samples.append(faces)
        self.i += 1
        QTimer.singleShot(0, self._next)

    def _solve(self):
        feats, targets, groups, yaws = [], [], [], []
        for k, ((mon, fx, fy), faces) in enumerate(zip(self.steps, self.samples)):
            m = self.monitors[mon]
            for f in faces:
                feats.append(np.asarray(f, float))
                targets.append((m.x + fx * m.w, m.y + fy * m.h))
                groups.append(k)
            if (fx, fy) == (0.5, 0.5):               # the center dot doubles as the head-yaw calibration
                yaws.append(statistics.median(f.yaw for f in faces))
        if not self._check_spacing(yaws):
            return
        model = gaze.fit(np.array(feats), np.array(targets), np.array(groups))
        self.engine.set_calibration(yaws, self.monitors, model)
        width = statistics.mean(m.w for m in self.monitors)
        self._finish(True, f"Calibration saved.\nEstimated gaze error: about {model.error_px:.0f} px "
                           f"({100 * model.error_px / width:.0f}% of a monitor width).")


class GazeDot(QWidget):
    """Click-through dot that shows where the gaze model thinks you are looking."""
    SIZE = 28

    def __init__(self):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.resize(self.SIZE, self.SIZE)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(255, 60, 60, 170))
        p.setPen(QPen(Qt.white, 2))
        p.drawEllipse(self.rect().adjusted(2, 2, -2, -2))

    def place(self, point, monitors):
        """Move to a gaze point given in desktop (backend) pixels."""
        idx = nearest_monitor(point[0], point[1], monitors)
        m, geo = monitors[idx], screen_for_monitor(idx, len(monitors)).geometry()
        fx = min(max((point[0] - m.x) / m.w, 0), 1)
        fy = min(max((point[1] - m.y) / m.h, 0), 1)
        self.move(int(geo.x() + fx * geo.width() - self.SIZE / 2),
                  int(geo.y() + fy * geo.height() - self.SIZE / 2))
        self.show()


class Bridge(QObject):
    """Lets the engine thread poke the GUI thread."""
    enabled_changed = Signal()


class MainWindow(QWidget):
    def __init__(self, app, engine, cfg, tray):
        super().__init__()
        self.app, self.engine, self.cfg, self.tray = app, engine, cfg, tray
        self.calibration, self.dot = None, None
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
        self.show_gaze = QCheckBox("Show where it thinks I am looking (test dot)")
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
        self.form.addRow("Mode", self.mode)
        self.form.addRow("Delay before switching monitor", self.delay)
        self.form.addRow("Delay before focusing a window", self.window_delay)
        self.form.addRow("Wait after typing", self.typing_wait)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)
        for w in (self.status, self.detail, self.enabled):
            layout.addWidget(w)
        layout.addLayout(self.form)
        layout.addWidget(self.show_gaze)
        layout.addWidget(self.vcam)
        layout.addWidget(self.vcam_note)
        layout.addWidget(self.autostart)
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
        self.dot_timer.timeout.connect(self._move_dot)
        self.dot_timer.start(50)
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
        for w in (self.window_delay, self.typing_wait):
            self.form.setRowVisible(w, window)
        self.show_gaze.setVisible(window)
        self.dot_timer.setInterval(50 if window and self.cfg["show_gaze"] else 400)
        self._update_note()
        self.adjustSize()

    def _update_note(self):
        if self.cfg["mode"] == "window" and self.engine.gaze_model is None:
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

    def _move_dot(self):
        want = self.cfg["mode"] == "window" and self.cfg["show_gaze"] and self.engine.gaze is not None
        if not want:
            if self.dot:
                self.dot.hide()
            return
        if self.dot is None:
            self.dot = GazeDot()
        self.dot.place(self.engine.gaze, self.engine.monitors)

    def _on_camera(self, _i):
        self._set("camera", self.camera.currentData())

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
        on = cfg["enabled"]
        tray.setIcon(icon("eyeswitch-tray.svg" if on else "eyeswitch-tray-paused.svg"))
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
