#!/usr/bin/env python3
"""eyeSwitch: move keyboard focus to the monitor you are facing (webcam, X11)."""
import importlib
import os
import statistics
import sys

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

from engine import Engine, list_cameras, load_config, save_config  # noqa: E402

APP_ID = "io.github.pedroslev.eyeSwitch"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
AUTOSTART = os.path.join(GLib.get_user_config_dir(), "autostart", "eyeswitch.desktop")
TRAY_HINT = "sudo apt install gir1.2-ayatanaappindicator3-0.1"

AppIndicator = None
for _ns in ("AyatanaAppIndicator3", "AppIndicator3"):
    try:
        gi.require_version(_ns, "0.1")
        AppIndicator = importlib.import_module("gi.repository." + _ns)
        break
    except (ValueError, ImportError):
        continue


class CalibrationWindow(Gtk.Window):
    """Fullscreen black window with one dot, shown on a given monitor."""

    def __init__(self, mon, on_cancel):
        super().__init__(title="eyeSwitch calibration")
        self.text = ""
        self.set_decorated(False)
        self.set_keep_above(True)
        area = Gtk.DrawingArea()
        area.connect("draw", self._draw)
        self.add(area)
        self.connect("key-press-event", lambda w, e: e.keyval == Gdk.KEY_Escape and on_cancel())
        display = Gdk.Display.get_default()
        for n in range(display.get_n_monitors()):
            g = display.get_monitor(n).get_geometry()
            if (g.x, g.y) == (mon.x, mon.y):
                self.fullscreen_on_monitor(display.get_default_screen(), n)
                break
        else:
            self.move(mon.x, mon.y)
            self.fullscreen()

    def set_text(self, text):
        self.text = text
        self.queue_draw()

    def _draw(self, widget, cr):
        w, h = widget.get_allocated_width(), widget.get_allocated_height()
        cr.set_source_rgb(0, 0, 0)
        cr.paint()
        cr.set_source_rgb(1, 0.78, 0)
        cr.arc(w / 2, h / 2, 18, 0, 6.2832)
        cr.fill()
        cr.set_source_rgb(1, 1, 1)
        cr.set_font_size(32)
        ext = cr.text_extents(self.text)
        cr.move_to(w / 2 - ext.width / 2, h / 2 - 70)
        cr.show_text(self.text)


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
            self.win.destroy()
        self.on_done(ok, message)

    def _step(self):
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.")
        if self.i == len(self.monitors):
            gaps = [abs(a - b) for a, b in zip(sorted(self.yaws), sorted(self.yaws)[1:])]
            if min(gaps) < 5:
                return self._finish(False, "The head angle between monitors is under 5 degrees. "
                                           "Turn your head more clearly towards each dot.")
            self.engine.set_calibration(self.yaws, self.monitors)
            return self._finish(True, "Calibration saved.")
        self.win = CalibrationWindow(self.monitors[self.i], self.cancel)
        self.win.set_text(f"Monitor {self.i + 1} of {len(self.monitors)}: look at the dot")
        self.win.show_all()
        GLib.timeout_add(2500, self._begin_capture)

    def _begin_capture(self):
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.") or False
        self.engine.begin_capture()
        self.win.set_text("Hold still...")
        GLib.timeout_add(2500, self._end_capture)
        return False

    def _end_capture(self):
        samples = self.engine.end_capture()
        self.win.destroy()
        self.win = None
        if self.cancelled:
            return self._finish(False, "Calibration cancelled.") or False
        if len(samples) < 10:
            return self._finish(False, f"No face detected while looking at monitor {self.i + 1}.") or False
        self.yaws.append(statistics.median(samples))
        self.i += 1
        GLib.idle_add(self._step)
        return False


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, app, engine, cfg):
        super().__init__(application=app, title="eyeSwitch")
        self.engine, self.cfg, self.app = engine, cfg, app
        self.calibration = None
        self.set_default_size(380, -1)
        self.set_resizable(False)
        self._syncing = False

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=18)
        self.add(box)

        self.status = Gtk.Label(xalign=0)
        self.status.get_style_context().add_class("title-3")
        self.detail = Gtk.Label(xalign=0)
        self.detail.get_style_context().add_class("dim-label")
        box.pack_start(self.status, False, False, 0)
        box.pack_start(self.detail, False, False, 0)

        grid = Gtk.Grid(column_spacing=12, row_spacing=10)
        box.pack_start(grid, False, False, 0)

        self.enabled = Gtk.Switch(halign=Gtk.Align.END)
        self.enabled.connect("notify::active", self._on_enabled)
        self._row(grid, 0, "Active  (Ctrl+Shift+G)", self.enabled)

        self.camera = Gtk.ComboBoxText()
        self.cameras = list_cameras()
        for idx, name in self.cameras:
            self.camera.append(str(idx), f"{name} (/dev/video{idx})")
        self.camera.set_active_id(str(cfg["camera"]))
        self.camera.connect("changed", self._on_camera)
        self._row(grid, 1, "Camera", self.camera)

        self.delay = Gtk.SpinButton.new_with_range(100, 2000, 50)
        self.delay.set_value(cfg["delay_ms"])
        self.delay.connect("value-changed", self._on_delay)
        self._row(grid, 2, "Delay before switching (ms)", self.delay)

        self.autostart = Gtk.Switch(halign=Gtk.Align.END, active=os.path.exists(AUTOSTART))
        self.autostart.connect("notify::active", self._on_autostart)
        self._row(grid, 3, "Start at login", self.autostart)

        self.calibrate_btn = Gtk.Button(label="Calibrate")
        self.calibrate_btn.get_style_context().add_class("suggested-action")
        self.calibrate_btn.connect("clicked", self._on_calibrate)
        box.pack_start(self.calibrate_btn, False, False, 0)

        if AppIndicator is None:
            hint = Gtk.Label(xalign=0, wrap=True, selectable=True,
                             label=f"No tray icon support. Closing this window quits the app.\n"
                                   f"To get the tray icon: {TRAY_HINT}")
            hint.get_style_context().add_class("dim-label")
            box.pack_start(hint, False, False, 0)

        self.connect("delete-event", self._on_close)
        box.show_all()                  # present() alone shows an empty window
        self.sync()
        GLib.timeout_add(300, self._refresh)

    @staticmethod
    def _row(grid, row, label, widget):
        lbl = Gtk.Label(label=label, xalign=0, hexpand=True)
        grid.attach(lbl, 0, row, 1, 1)
        grid.attach(widget, 1, row, 1, 1)

    def sync(self):
        """Pull `enabled` from the engine config into the switch (hotkey, tray)."""
        self._syncing = True
        self.enabled.set_active(self.cfg["enabled"])
        self._syncing = False

    def _refresh(self):
        e = self.engine
        self.status.set_text(e.status)
        if e.calibrated:
            yaws = ", ".join(f"M{i + 1} {y:+.0f}°" for i, y in enumerate(e.centers))
            cal = f"Calibrated ({yaws})"
        elif e.centers is not None:
            cal = "Monitor layout changed: recalibrate"
        else:
            cal = "Not calibrated yet"
        yaw = "no face" if e.yaw is None else f"{e.yaw:+.1f}°"
        self.detail.set_text(f"Head yaw: {yaw}  |  {cal}")
        self.app.update_tray()
        return True

    def _on_enabled(self, sw, _):
        if not self._syncing:
            self.engine.set_enabled(sw.get_active())

    def _on_camera(self, combo):
        self.cfg["camera"] = int(combo.get_active_id())
        save_config(self.cfg)

    def _on_delay(self, spin):
        self.cfg["delay_ms"] = int(spin.get_value())
        save_config(self.cfg)

    def _on_autostart(self, sw, _):
        if sw.get_active():
            os.makedirs(os.path.dirname(AUTOSTART), exist_ok=True)
            with open(AUTOSTART, "w") as f:
                f.write("[Desktop Entry]\nType=Application\nName=eyeSwitch\nIcon=eyeswitch\n"
                        f"Exec={sys.executable} {os.path.join(HERE, 'eyeswitch.py')} --background\n"
                        "X-GNOME-Autostart-enabled=true\n")
        elif os.path.exists(AUTOSTART):
            os.remove(AUTOSTART)

    def _on_calibrate(self, _btn):
        if len(self.engine.monitors) < 2:
            return self._message("Only one monitor detected.")
        self.calibrate_btn.set_sensitive(False)
        self.calibration = Calibration(self.engine, self._calibration_done)

    def _calibration_done(self, ok, message):
        self.calibration = None
        self.calibrate_btn.set_sensitive(True)
        self.present()
        self._message(message, Gtk.MessageType.INFO if ok else Gtk.MessageType.WARNING)

    def _message(self, text, kind=Gtk.MessageType.WARNING):
        dlg = Gtk.MessageDialog(transient_for=self, modal=True, message_type=kind,
                                buttons=Gtk.ButtonsType.OK, text=text)
        dlg.run()
        dlg.destroy()

    def _on_close(self, *_):
        if AppIndicator is None:
            self.app.quit_app()
        else:
            self.hide()
        return True


class EyeSwitchApp(Gtk.Application):
    def __init__(self, background):
        super().__init__(application_id=APP_ID)
        self.background = background
        self.window = self.engine = self.indicator = None

    def do_activate(self):
        if self.window is None:
            self.cfg = load_config()
            if os.environ.get("XDG_SESSION_TYPE") == "wayland":
                print("warning: Wayland session; focus control only works on X11", file=sys.stderr)
            self.engine = Engine(self.cfg)
            self.hold()
            self.window = MainWindow(self, self.engine, self.cfg)
            self.engine.on_change = lambda: GLib.idle_add(self._on_engine_change)
            self._build_tray()
            if self.background and AppIndicator is not None:
                return
        self.window.present()

    def _build_tray(self):
        if AppIndicator is None:
            return
        self.indicator = AppIndicator.Indicator.new(
            "eyeswitch", "eyeswitch-tray", AppIndicator.IndicatorCategory.APPLICATION_STATUS)
        self.indicator.set_icon_theme_path(DATA_DIR)
        self.indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
        menu = Gtk.Menu()
        open_item = Gtk.MenuItem(label="Open eyeSwitch")
        open_item.connect("activate", lambda *_: self.window.present())
        self.pause_item = Gtk.CheckMenuItem(label="Active")
        self.pause_item.connect("toggled", self._on_tray_toggle)
        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect("activate", lambda *_: self.quit_app())
        for item in (open_item, self.pause_item, Gtk.SeparatorMenuItem(), quit_item):
            menu.append(item)
        menu.show_all()
        self.indicator.set_menu(menu)
        self.update_tray()

    def update_tray(self):
        if self.indicator is None:
            return
        enabled = self.cfg["enabled"]
        self.indicator.set_icon_full("eyeswitch-tray" if enabled else "eyeswitch-tray-paused",
                                     "eyeSwitch")
        if self.pause_item.get_active() != enabled:
            self.pause_item.handler_block_by_func(self._on_tray_toggle)
            self.pause_item.set_active(enabled)
            self.pause_item.handler_unblock_by_func(self._on_tray_toggle)

    def _on_tray_toggle(self, item):
        self.engine.set_enabled(item.get_active())

    def _on_engine_change(self):
        self.window.sync()
        self.update_tray()
        return False

    def quit_app(self):
        if self.engine:
            self.engine.stop()
        self.quit()


def main():
    GLib.set_prgname("eyeswitch")
    GLib.set_application_name("eyeSwitch")
    background = "--background" in sys.argv
    Gtk.IconTheme.get_default().append_search_path(DATA_DIR)
    Gtk.Window.set_default_icon_name("eyeswitch")
    app = EyeSwitchApp(background)
    sys.exit(app.run([sys.argv[0]]))


if __name__ == "__main__":
    main()
