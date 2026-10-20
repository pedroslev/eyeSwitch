"""macOS backend: Quartz for displays/windows/pointer, Accessibility API for focus.

Needs the Accessibility permission (System Settings > Privacy & Security) for the app
that launches it, and Camera for the tracker. Coordinates are points, origin at the top
left of the main display (the space CGDisplayBounds and CGWindowList share).
"""
import os
import time

import Quartz
from AppKit import NSRunningApplication, NSWorkspace
from ApplicationServices import (
    AXIsProcessTrustedWithOptions, AXUIElementCopyAttributeValue, AXUIElementCreateApplication,
    AXUIElementPerformAction, AXUIElementSetAttributeValue, AXValueGetValue,
    kAXTrustedCheckOptionPrompt, kAXValueCGPointType, kAXValueCGSizeType)

from .base import BaseSession, Monitor

KEY_CONTROL, KEY_SHIFT, KEY_G = 59, 56, 5     # virtual key codes
CACHE_TTL = 0.1


class Session(BaseSession):
    def __init__(self):
        self._hotkey_down = False
        self._cache = (0.0, [])
        self._pid = os.getpid()
        AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})   # prompts once

    def monitors(self):
        _err, ids, _n = Quartz.CGGetActiveDisplayList(16, None, None)
        out = []
        for d in ids:
            r = Quartz.CGDisplayBounds(d)
            out.append(Monitor(int(r.origin.x), int(r.origin.y), int(r.size.width), int(r.size.height)))
        return sorted(out, key=lambda m: m.x)

    # --- pointer -----------------------------------------------------------
    def pointer(self):
        p = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        return int(p.x), int(p.y)

    def warp(self, x, y):
        Quartz.CGWarpMouseCursorPosition(Quartz.CGPointMake(x, y))
        Quartz.CGAssociateMouseAndMouseCursorPosition(True)
        # A warp alone raises no mouse-moved event, so hover effects (Mission Control) would not follow it.
        ev = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, (x, y), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    # --- windows -----------------------------------------------------------
    def _windows(self):
        """Normal on-screen windows of other apps, front to back: [(id, pid, (x, y, w, h))]."""
        now = time.monotonic()
        if now - self._cache[0] < CACHE_TTL:
            return self._cache[1]
        opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        out = []
        for w in Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []:
            b = w[Quartz.kCGWindowBounds]
            if (w.get(Quartz.kCGWindowLayer) != 0 or w.get(Quartz.kCGWindowAlpha, 1) == 0
                    or w[Quartz.kCGWindowOwnerPID] == self._pid or b["Width"] < 80 or b["Height"] < 60):
                continue
            out.append((int(w[Quartz.kCGWindowNumber]), int(w[Quartz.kCGWindowOwnerPID]),
                        (b["X"], b["Y"], b["Width"], b["Height"])))
        self._cache = (now, out)
        return out

    def _find(self, wid):
        return next((w for w in self._windows() if w[0] == wid), None)

    def active_window(self):
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return None
        pid = int(app.processIdentifier())
        return next((w[0] for w in self._windows() if w[1] == pid), None)

    def window_monitor(self, wid, monitors):
        w = self._find(wid)
        if w is None:
            return None
        x, y, width, height = w[2]
        return self.center_monitor(monitors, x, y, x + width, y + height)

    def exists(self, wid):
        return self._find(wid) is not None

    def top_window_on(self, monitors, idx):
        for wid, _pid, (x, y, w, h) in self._windows():
            if self.center_monitor(monitors, x, y, x + w, y + h) == idx:
                return wid
        return None

    def window_at(self, x, y):
        for wid, _pid, (wx, wy, w, h) in self._windows():
            if wx <= x < wx + w and wy <= y < wy + h:
                return wid
        return None

    def seconds_since_key(self):
        return Quartz.CGEventSourceSecondsSinceLastEventType(
            Quartz.kCGEventSourceStateCombinedSessionState, Quartz.kCGEventKeyDown)

    def window_bounds(self, wid):
        w = self._find(wid)
        return None if w is None else tuple(w[2])

    # virtual key codes; "super" is the Command key
    _KEYS = {"super": 55, "ctrl": 59, "shift": 56, "alt": 58, "enter": 36, "esc": 53, "tab": 48,
             "space": 49, "left": 123, "right": 124, "down": 125, "up": 126, "backspace": 51,
             "delete": 117, "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
             **dict(zip("asdfhgzxcvbqweryt", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17))),
             **dict(zip("123465=9780", (18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29))),   # row of digits
             **dict(zip("-]o[uilpjk;", (27, 30, 31, 33, 32, 34, 37, 35, 38, 40, 41))),
             **dict(zip("\\,/nm.", (42, 43, 44, 45, 46, 47))),
             **{f"f{i}": k for i, k in enumerate((122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111), 1)}}
    _FLAGS = {"super": Quartz.kCGEventFlagMaskCommand, "ctrl": Quartz.kCGEventFlagMaskControl,
              "shift": Quartz.kCGEventFlagMaskShift, "alt": Quartz.kCGEventFlagMaskAlternate}

    def press_keys(self, keys):
        codes = []
        for k in keys:
            if k not in self._KEYS:
                raise ValueError(f"unknown key: {k}")
            codes.append(self._KEYS[k])
        flags = 0

        def post(code, down, flags):
            ev = Quartz.CGEventCreateKeyboardEvent(None, code, down)
            Quartz.CGEventSetFlags(ev, flags)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

        for k, code in zip(keys, codes):
            flags |= self._FLAGS.get(k, 0)
            post(code, True, flags)
        for k, code in reversed(list(zip(keys, codes))):
            post(code, False, flags)
            flags &= ~self._FLAGS.get(k, 0)

    def click(self):
        x, y = self.pointer()
        for kind in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
            ev = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    def activate(self, wid):
        w = self._find(wid)
        if w is None:
            return
        _wid, pid, (x, y, width, height) = w
        app_el = AXUIElementCreateApplication(pid)
        err, windows = AXUIElementCopyAttributeValue(app_el, "AXWindows", None)
        target = None
        for el in (windows or []) if err == 0 else []:
            if self._ax_bounds(el) is not None and all(
                    abs(a - b) <= 2 for a, b in zip(self._ax_bounds(el), (x, y, width, height))):
                target = el
                break
        AXUIElementSetAttributeValue(app_el, "AXFrontmost", True)
        if target is not None:
            AXUIElementPerformAction(target, "AXRaise")
            AXUIElementSetAttributeValue(target, "AXMain", True)
            AXUIElementSetAttributeValue(target, "AXFocused", True)
        else:                                  # no Accessibility access or window not matched
            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if app is not None:
                app.activateWithOptions_(2)    # NSApplicationActivateIgnoringOtherApps

    @staticmethod
    def _ax_bounds(el):
        e1, pos = AXUIElementCopyAttributeValue(el, "AXPosition", None)
        e2, size = AXUIElementCopyAttributeValue(el, "AXSize", None)
        if e1 or e2:
            return None
        ok1, p = AXValueGetValue(pos, kAXValueCGPointType, None)
        ok2, s = AXValueGetValue(size, kAXValueCGSizeType, None)
        return (p.x, p.y, s.width, s.height) if ok1 and ok2 else None

    # --- hotkey (polled) ---------------------------------------------------
    def grab_hotkey(self):
        return True

    def hotkey_pressed(self):
        state = Quartz.kCGEventSourceStateCombinedSessionState
        down = all(Quartz.CGEventSourceKeyState(state, k) for k in (KEY_CONTROL, KEY_SHIFT, KEY_G))
        fired = down and not self._hotkey_down
        self._hotkey_down = down
        return fired
