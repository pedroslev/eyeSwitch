"""Interface every OS backend implements.

All coordinates are in the same global pixel space as the monitors (physical pixels on
X11 and Windows, points on macOS). Window ids are opaque ints, only valid for one backend.
A backend object is used from a single thread (the engine thread); create it there.
"""
from collections import namedtuple

Monitor = namedtuple("Monitor", "x y w h")


MODIFIERS = ("super", "ctrl", "shift", "alt")
ALIASES = {"win": "super", "windows": "super", "cmd": "super", "command": "super", "meta": "super",
           "control": "ctrl", "return": "enter", "escape": "esc", "option": "alt"}


def parse_keys(text):
    """'Win+Tab' -> ['super', 'tab'] (lowercase, aliases resolved)."""
    keys = [k.strip().lower() for k in text.replace(",", "+").split("+") if k.strip()]
    return [ALIASES.get(k, k) for k in keys]


class BaseSession:
    def monitors(self):
        """[Monitor] sorted left to right."""
        raise NotImplementedError

    def pointer(self):
        """(x, y) of the mouse pointer."""
        raise NotImplementedError

    def warp(self, x, y):
        """Move the mouse pointer."""
        raise NotImplementedError

    def active_window(self):
        """Id of the focused window, or None."""
        raise NotImplementedError

    def window_monitor(self, wid, monitors):
        """Index of the monitor holding the window's center, or None if unknown."""
        raise NotImplementedError

    def top_window_on(self, monitors, idx):
        """Topmost normal visible window centered on monitor idx, or None."""
        raise NotImplementedError

    def window_at(self, x, y):
        """Topmost normal visible window containing the point, or None."""
        raise NotImplementedError

    def seconds_since_key(self):
        """Seconds since a typing key (not a modifier) was last seen pressed."""
        raise NotImplementedError

    def window_bounds(self, wid):
        """(x, y, w, h) of the window, or None."""
        raise NotImplementedError

    def press_keys(self, keys):
        """Press the keys together, then release them (names as returned by parse_keys).
        Raises ValueError for a key it does not know."""
        raise NotImplementedError

    def click(self):
        """Left click at the current pointer position."""
        raise NotImplementedError

    def exists(self, wid):
        raise NotImplementedError

    def activate(self, wid):
        """Give keyboard focus to the window and raise it."""
        raise NotImplementedError

    def super_tapped(self):
        """True once per tap of the Super (Windows / Command) key on its own, as seen on the real keyboard since
        the last call. Super with another key (Super+Tab...) is not a tap. False where it cannot be watched."""
        return False

    def overview_open(self):
        """Is the system's all-apps view (GNOME's overview, macOS Mission Control) open right now?
        True / False, or None when this backend cannot tell."""
        return None

    def grab_hotkey(self):
        """Register Ctrl+Shift+G for hotkey_pressed(). Returns False if unavailable."""
        raise NotImplementedError

    def hotkey_pressed(self):
        """True once per press of the hotkey since the last call."""
        raise NotImplementedError

    @staticmethod
    def monitor_at(monitors, x, y):
        for i, m in enumerate(monitors):
            if m.x <= x < m.x + m.w and m.y <= y < m.y + m.h:
                return i
        return None

    @staticmethod
    def center_monitor(monitors, left, top, right, bottom):
        return BaseSession.monitor_at(monitors, (left + right) // 2, (top + bottom) // 2)
