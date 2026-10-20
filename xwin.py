"""X11 helpers: monitors, window focus, pointer, global pause hotkey."""
from collections import namedtuple

from Xlib import X, XK, display, protocol
from Xlib.error import CatchError

Monitor = namedtuple("Monitor", "x y w h")


class XSession:
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

    @staticmethod
    def monitor_at(monitors, x, y):
        for i, m in enumerate(monitors):
            if m.x <= x < m.x + m.w and m.y <= y < m.y + m.h:
                return i
        return None

    # --- pointer -----------------------------------------------------------
    def pointer(self):
        p = self.root.query_pointer()
        return p.root_x, p.root_y

    def warp(self, x, y):
        self.root.warp_pointer(x, y)
        self.d.sync()

    # --- windows -----------------------------------------------------------
    def active_window(self):
        v = self._prop(self.root, "_NET_ACTIVE_WINDOW")
        return v[0] if v and v[0] else None

    def _center(self, wid):
        """Window center in root coordinates, or None if the window is gone/unmapped."""
        try:
            win = self.d.create_resource_object("window", wid)
            g = win.get_geometry()
            c = self.root.translate_coords(win, 0, 0)
            return c.x + g.width // 2, c.y + g.height // 2
        except Exception:
            return None

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
