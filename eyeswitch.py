#!/usr/bin/env python3
"""eyeSwitch: move keyboard focus to the monitor you are facing (webcam)."""
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

from PySide6.QtCore import QObject, QPointF, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QAction, QColor, QFont, QGuiApplication, QIcon, QPainter  # noqa: E402
from PySide6.QtNetwork import QLocalServer, QLocalSocket  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QLabel,  # noqa: E402
                               QMenu, QMessageBox, QPushButton, QSpinBox, QSystemTrayIcon,
                               QVBoxLayout, QWidget)

import autostart  # noqa: E402
import backends  # noqa: E402
from engine import Engine, load_config, save_config  # noqa: E402
from tracker import list_cameras  # noqa: E402


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
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())

    def set_text(self, text):
        self.text = text
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.on_cancel()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        c = QPointF(self.width() / 2, self.height() / 2)
        p.setBrush(QColor(255, 200, 0))
        p.setPen(Qt.NoPen)
        p.drawEllipse(c, 18, 18)
        p.setPen(Qt.white)
        p.setFont(QFont(self.font().family(), 24))
        rect = self.rect().adjusted(0, 0, 0, -int(self.height() / 2 + 60))
        p.drawText(rect, Qt.AlignHCenter | Qt.AlignBottom, self.text)


class Calibration:
    """Per monitor: 2.5 s to turn the head, 2.5 s of samples; stores the median yaw."""

    def __init__(self, engine, on_done):
        self.engine, self.on_done = engine, on_done
        self.monitors = list(engine.monitors)
        self.yaws, self.i, self.win, self.cancelled = [], 0, None, False
        engine.calibrating = True
        self._step()

    def cancel(self):
        self.cancelled = True

    def _finish(self, ok, message):
        self.engine.calibrating = False
        if self.win:
            self.win.close()
            self.win = None
        self.on_done(ok, message)

    def _step(self):
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        n = len(self.monitors)
        if self.i == n:
            s = sorted(self.yaws)
            if min(b - a for a, b in zip(s, s[1:])) < 5:
                return self._finish(False, "The head angle between monitors is under 5 degrees. "
                                           "Turn your head more clearly towards each dot.")
            self.engine.set_calibration(self.yaws, self.monitors)
            return self._finish(True, "Calibration saved.")
        self.win = CalibrationWindow(screen_for_monitor(self.i, n), self.cancel)
        self.win.set_text(f"Monitor {self.i + 1} of {n}: look at the dot")
        self.win.showFullScreen()
        self.win.activateWindow()
        QTimer.singleShot(2500, self._begin_capture)

    def _begin_capture(self):
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        self.engine.begin_capture()
        self.win.set_text("Hold still...")
        QTimer.singleShot(2500, self._end_capture)

    def _end_capture(self):
        samples = self.engine.end_capture()
        self.win.close()
        self.win = None
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        if len(samples) < 10:
            return self._finish(False, f"No face detected while looking at monitor {self.i + 1}.")
        self.yaws.append(statistics.median(samples))
        self.i += 1
        QTimer.singleShot(0, self._step)


class Bridge(QObject):
    """Lets the engine thread poke the GUI thread."""
    enabled_changed = Signal()


class MainWindow(QWidget):
    def __init__(self, app, engine, cfg, tray):
        super().__init__()
        self.app, self.engine, self.cfg, self.tray = app, engine, cfg, tray
        self.calibration = None
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
        pos = self.camera.findData(cfg["camera"])
        self.camera.setCurrentIndex(max(pos, 0))
        self.camera.currentIndexChanged.connect(self._on_camera)

        self.delay = QSpinBox()
        self.delay.setRange(100, 2000)
        self.delay.setSingleStep(50)
        self.delay.setSuffix(" ms")
        self.delay.setValue(cfg["delay_ms"])
        self.delay.valueChanged.connect(self._on_delay)

        self.autostart = QCheckBox("Start at login")
        self.autostart.setChecked(autostart.is_enabled())
        self.autostart.toggled.connect(autostart.set_enabled)

        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self._on_calibrate)

        form = QFormLayout()
        form.addRow("Camera", self.camera)
        form.addRow("Delay before switching", self.delay)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)
        for w in (self.status, self.detail, self.enabled):
            layout.addWidget(w)
        layout.addLayout(form)
        layout.addWidget(self.autostart)
        layout.addWidget(self.calibrate_btn)
        warning = backends.unsupported_reason() or ("" if tray else
                                                    "No system tray available: closing this window quits the app.")
        if warning:
            note = QLabel(warning)
            note.setWordWrap(True)
            note.setEnabled(False)
            layout.addWidget(note)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(300)
        self.refresh()
        self.setMinimumWidth(400)

    def sync_enabled(self):
        """Pull `enabled` from config (hotkey or tray changed it)."""
        self.enabled.blockSignals(True)
        self.enabled.setChecked(self.cfg["enabled"])
        self.enabled.blockSignals(False)

    def refresh(self):
        e = self.engine
        self.status.setText(e.status)
        if e.calibrated:
            cal = "Calibrated (" + ", ".join(f"M{i + 1} {y:+.0f}°" for i, y in enumerate(e.centers)) + ")"
        elif e.centers is not None:
            cal = "Monitor layout changed: recalibrate"
        else:
            cal = "Not calibrated yet"
        yaw = "no face" if e.yaw is None else f"{e.yaw:+.1f}°"
        self.detail.setText(f"Head yaw: {yaw}  |  {cal}")

    def _on_camera(self, _i):
        self.cfg["camera"] = self.camera.currentData()
        save_config(self.cfg)

    def _on_delay(self, value):
        self.cfg["delay_ms"] = value
        save_config(self.cfg)

    def _on_calibrate(self):
        if len(self.engine.monitors) < 2:
            return self._message("Only one monitor detected.", QMessageBox.Warning)
        self.calibrate_btn.setEnabled(False)
        self.calibration = Calibration(self.engine, self._calibration_done)

    def _calibration_done(self, ok, message):
        self.calibration = None
        self.calibrate_btn.setEnabled(True)
        self.show_window()
        self._message(message, QMessageBox.Information if ok else QMessageBox.Warning)

    def _message(self, text, kind):
        box = QMessageBox(kind, "eyeSwitch", text, QMessageBox.Ok, self)
        box.exec()

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
