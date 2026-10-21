"""X11 backend: RandR monitors, EWMH window focus, pointer warp, XGrabKey hotkey."""
import atexit
import re
import subprocess
import threading
import time

from Xlib import X, XK, display, protocol
from Xlib.ext import record, xtest
from Xlib.protocol import rq
from Xlib.error import CatchError

from .base import BaseSession, Monitor


class Session(BaseSession):
    def __init__(self):
        self.d = display.Display()
        self.root = self.d.screen().root
        self._atoms = {}

    def atom(self, name):
        if name not in self._atoms:
            self._atoms[name] = self.d.get_atom(name)
        return self._atoms[name]

    def _prop(self, win, name):
        p = win.get_full_property(self.atom(name), X.AnyPropertyType)
        return list(p.value) if p else []

    def monitors(self):
        ms = self.root.xrandr_get_monitors().monitors
        out = [Monitor(m.x, m.y, m.width_in_pixels, m.height_in_pixels) for m in ms]
        return sorted(out, key=lambda m: m.x)

    # --- pointer -----------------------------------------------------------
    def pointer(self):
        p = self.root.query_pointer()
        return p.root_x, p.root_y

    def warp(self, x, y):
        try:
            # XTEST motion looks like a real device moving (XI2 RawMotion and Motion); XWarpPointer only yields
            # Motion, and a shell's overview may not highlight what is under a pointer moved that way.
            xtest.fake_input(self.d, X.MotionNotify, False, x=int(x), y=int(y))
        except Exception:
            self.root.warp_pointer(int(x), int(y))
        self.d.sync()

    # --- windows -----------------------------------------------------------
    def active_window(self):
        v = self._prop(self.root, "_NET_ACTIVE_WINDOW")
        return v[0] if v and v[0] else None

    def _bounds(self, wid):
        """(x, y, w, h) in root coordinates, or None if the window is gone/unmapped."""
        try:
            win = self.d.create_resource_object("window", wid)
            g = win.get_geometry()
            c = self.root.translate_coords(win, 0, 0)
            return c.x, c.y, g.width, g.height
        except Exception:
            return None

    def _center(self, wid):
        b = self._bounds(wid)
        return None if b is None else (b[0] + b[2] // 2, b[1] + b[3] // 2)

    def window_monitor(self, wid, monitors):
        c = self._center(wid)
        return None if c is None else self.monitor_at(monitors, *c)

    def _is_normal(self, wid):
        try:
            win = self.d.create_resource_object("window", wid)
            types = self._prop(win, "_NET_WM_WINDOW_TYPE")
            ok = {self.atom("_NET_WM_WINDOW_TYPE_NORMAL"), self.atom("_NET_WM_WINDOW_TYPE_DIALOG")}
            if types and not ok.intersection(types):
                return False
            if self.atom("_NET_WM_STATE_HIDDEN") in self._prop(win, "_NET_WM_STATE"):
                return False
            return True
        except Exception:
            return False

    def top_window_on(self, monitors, idx):
        """Topmost normal, visible window whose center is on monitor idx."""
        for wid in reversed(self._prop(self.root, "_NET_CLIENT_LIST_STACKING")):
            if self._is_normal(wid) and self.window_monitor(wid, monitors) == idx:
                return wid
        return None

    def window_at(self, x, y):
        for wid in reversed(self._prop(self.root, "_NET_CLIENT_LIST_STACKING")):
            b = self._bounds(wid) if self._is_normal(wid) else None
            if b and b[0] <= x < b[0] + b[2] and b[1] <= y < b[1] + b[3]:
                return wid
        return None

    _MODIFIERS = ("Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Super_L",
                  "Super_R", "Meta_L", "Meta_R", "Caps_Lock", "Num_Lock", "ISO_Level3_Shift")

    def seconds_since_key(self):
        """Polls the keyboard state; sustained typing is caught, a lone tap may be missed."""
        if not hasattr(self, "_mod_codes"):
            codes = {self.d.keysym_to_keycode(XK.string_to_keysym(n)) for n in self._MODIFIERS}
            self._mod_codes, self._last_key = codes, float("-inf")
        keymap = self.d.query_keymap()
        if any(keymap[c // 8] >> (c % 8) & 1 for c in range(8, 256) if c not in self._mod_codes):
            self._last_key = time.monotonic()
        return time.monotonic() - self._last_key

    def window_bounds(self, wid):
        return self._bounds(wid)

    MAX_TAP_S = 2.0
    _SHELL = ["--session", "--dest", "org.gnome.Shell", "--object-path", "/org/gnome/Shell"]

    def overview_open(self):
        """GNOME Shell's overview, from its OverviewActive property (it announces changes, so this costs
        nothing per call). None where there is no GNOME Shell to ask."""
        if not hasattr(self, "_overview"):
            self._overview = None
            threading.Thread(target=self._watch_overview, daemon=True).start()
        return self._overview

    def _watch_overview(self):
        proc = None
        try:
            proc = subprocess.Popen(["gdbus", "monitor", *self._SHELL], stdout=subprocess.PIPE, text=True,
                                    stderr=subprocess.DEVNULL)
            atexit.register(proc.kill)
            got = subprocess.run(["gdbus", "call", *self._SHELL, "--method", "org.freedesktop.DBus.Properties.Get",
                                  "org.gnome.Shell", "OverviewActive"], capture_output=True, text=True, timeout=3)
            if got.returncode != 0:
                return                                              # no GNOME Shell: stay unknown
            self._overview = "true" in got.stdout
            for line in proc.stdout:
                found = re.search(r"'OverviewActive': <(true|false)>", line)
                if found:
                    self._overview = found.group(1) == "true"
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            self._overview = None                                    # the shell went away or never answered
            if proc is not None:
                proc.kill()

    def super_tapped(self):
        """Taps of the Super key on the real keyboard, seen through the RECORD extension (it only observes:
        nothing is grabbed or delayed). A tap is Super pressed and released with no other key in between."""
        if not hasattr(self, "_taps"):
            self._taps, self._lock = 0, threading.Lock()
            threading.Thread(target=self._record_keys, daemon=True).start()
        with self._lock:
            taps, self._taps = self._taps, 0
        return taps > 0

    def _record_keys(self):
        try:
            d = display.Display()
            if not d.has_extension("RECORD"):
                return
            supers = {d.keysym_to_keycode(XK.string_to_keysym(n)) for n in ("Super_L", "Super_R")} - {0}
            state = {"since": None, "alone": False}

            def handle(reply):
                if reply.category != record.FromServer or reply.client_swapped or not reply.data or reply.data[0] < 2:
                    return
                data = reply.data
                while data:
                    event, data = rq.EventField(None).parse_binary_value(data, d.display, None, None)
                    now = time.monotonic()
                    if event.type == X.KeyPress:
                        if event.detail in supers:
                            if state["since"] is None:
                                state["since"], state["alone"] = now, True
                        elif state["since"] is not None:
                            state["alone"] = False           # Super+something: a shortcut, not a tap
                    elif event.type == X.KeyRelease and event.detail in supers and state["since"] is not None:
                        if state["alone"] and now - state["since"] <= self.MAX_TAP_S:
                            with self._lock:
                                self._taps += 1
                        state["since"] = None

            ctx = d.record_create_context(0, [record.AllClients], [{
                "core_requests": (0, 0), "core_replies": (0, 0), "ext_requests": (0, 0, 0, 0),
                "ext_replies": (0, 0, 0, 0), "delivered_events": (0, 0),
                "device_events": (X.KeyPress, X.KeyRelease), "errors": (0, 0),
                "client_started": False, "client_died": False}])
            d.record_enable_context(ctx, handle)             # blocks, calling handle for every key event
        except Exception as e:                                # no RECORD, or the display went away
            print(f"cannot watch the Super key: {e}", file=__import__("sys").stderr)

    _KEYSYMS = {"super": "Super_L", "ctrl": "Control_L", "shift": "Shift_L", "alt": "Alt_L",
                "enter": "Return", "esc": "Escape", "tab": "Tab", "space": "space", "left": "Left",
                "right": "Right", "up": "Up", "down": "Down", "backspace": "BackSpace",
                "delete": "Delete", "home": "Home", "end": "End", "pageup": "Prior", "pagedown": "Next"}

    def _keycode(self, name):
        sym = self._KEYSYMS.get(name) or (name.upper() if len(name) > 1 and name[0] == "f" and name[1:].isdigit()
                                          else name)
        code = self.d.keysym_to_keycode(XK.string_to_keysym(sym))
        if not code:
            raise ValueError(f"unknown key: {name}")
        return code

    def press_keys(self, keys):
        codes = [self._keycode(k) for k in keys]
        for c in codes:
            xtest.fake_input(self.d, X.KeyPress, c)
        for c in reversed(codes):
            xtest.fake_input(self.d, X.KeyRelease, c)
        self.d.sync()

    def click(self):
        xtest.fake_input(self.d, X.ButtonPress, 1)
        xtest.fake_input(self.d, X.ButtonRelease, 1)
        self.d.sync()

    def exists(self, wid):
        return wid in self._prop(self.root, "_NET_CLIENT_LIST")

    def activate(self, wid):
        win = self.d.create_resource_object("window", wid)
        ev = protocol.event.ClientMessage(
            window=win, client_type=self.atom("_NET_ACTIVE_WINDOW"),
            data=(32, [2, X.CurrentTime, 0, 0, 0]))  # source=2: pager, bypasses focus-steal guard
        self.root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        self.d.flush()

    # --- global hotkey -----------------------------------------------------
    def grab_hotkey(self, key="g", mods=X.ControlMask | X.ShiftMask):
        """Grab Ctrl+Shift+<key> on the root window. Returns False if already taken."""
        self._hotkey = (self.d.keysym_to_keycode(XK.string_to_keysym(key)), mods)
        ok = True
        for extra in (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask):  # caps/num lock
            err = CatchError()
            self.root.grab_key(self._hotkey[0], mods | extra, True,
                               X.GrabModeAsync, X.GrabModeAsync, onerror=err)
            self.d.sync()
            ok = ok and err.get_error() is None
        return ok

    def hotkey_pressed(self):
        pressed = False
        while self.d.pending_events():
            ev = self.d.next_event()
            if ev.type == X.KeyPress and ev.detail == self._hotkey[0]:
                pressed = True
        return pressed
