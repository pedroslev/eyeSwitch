"""Windows backend: Win32 through ctypes (no extra dependencies).

Coordinates are physical pixels; Qt makes the process per-monitor DPI aware when the
QApplication is created, so create it before the first Session.
"""
import ctypes
from ctypes import wintypes

from .base import BaseSession, Monitor

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")

HWND = wintypes.HWND
user32.GetForegroundWindow.restype = HWND
for _fn in ("IsWindow", "IsWindowVisible", "IsIconic", "SetForegroundWindow", "BringWindowToTop"):
    getattr(user32, _fn).argtypes = [HWND]
user32.GetWindowRect.argtypes = [HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowTextLengthW.argtypes = [HWND]
user32.GetWindowLongW.argtypes = [HWND, ctypes.c_int]
user32.GetWindow.argtypes = [HWND, wintypes.UINT]
user32.GetWindow.restype = HWND
user32.GetClassNameW.argtypes = [HWND, wintypes.LPWSTR, ctypes.c_int]
user32.ShowWindow.argtypes = [HWND, ctypes.c_int]
user32.GetWindowThreadProcessId.argtypes = [HWND, ctypes.POINTER(wintypes.DWORD)]
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
dwmapi.DwmGetWindowAttribute.argtypes = [HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW, WS_EX_APPWINDOW = 0x80, 0x40000
GW_OWNER = 4
SW_RESTORE = 9
DWMWA_CLOAKED = 14
SHELL_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}
VK_CONTROL, VK_SHIFT, VK_G = 0x11, 0x10, 0x47

_MONITORENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.HDC,
                                      ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, HWND, wintypes.LPARAM)


class Session(BaseSession):
    def __init__(self):
        self._hotkey_down = False

    def monitors(self):
        found = []

        def cb(_hmon, _hdc, rect, _lparam):
            r = rect.contents
            found.append(Monitor(r.left, r.top, r.right - r.left, r.bottom - r.top))
            return True

        user32.EnumDisplayMonitors(None, None, _MONITORENUMPROC(cb), 0)
        return sorted(found, key=lambda m: m.x)

    # --- pointer -----------------------------------------------------------
    def pointer(self):
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def warp(self, x, y):
        user32.SetCursorPos(int(x), int(y))

    # --- windows -----------------------------------------------------------
    def active_window(self):
        return user32.GetForegroundWindow() or None

    def _rect(self, hwnd):
        r = wintypes.RECT()
        return r if user32.GetWindowRect(HWND(hwnd), ctypes.byref(r)) else None

    def window_monitor(self, hwnd, monitors):
        r = self._rect(hwnd)
        return None if r is None else self.center_monitor(monitors, r.left, r.top, r.right, r.bottom)

    def exists(self, hwnd):
        return bool(user32.IsWindow(HWND(hwnd)))

    def _is_normal(self, hwnd):
        h = HWND(hwnd)
        if not user32.IsWindowVisible(h) or user32.IsIconic(h):
            return False
        if user32.GetWindowTextLengthW(h) == 0 or user32.GetWindow(h, GW_OWNER):
            return False
        ex = user32.GetWindowLongW(h, GWL_EXSTYLE)
        if ex & WS_EX_TOOLWINDOW and not ex & WS_EX_APPWINDOW:
            return False
        cloaked = wintypes.DWORD(0)       # hidden UWP / other-desktop windows
        dwmapi.DwmGetWindowAttribute(h, DWMWA_CLOAKED, ctypes.byref(cloaked), 4)
        if cloaked.value:
            return False
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(h, cls, 64)
        return cls.value not in SHELL_CLASSES

    def top_window_on(self, monitors, idx):
        result = []

        def cb(hwnd, _lparam):               # EnumWindows walks the z-order top to bottom
            if self._is_normal(hwnd) and self.window_monitor(hwnd, monitors) == idx:
                result.append(hwnd)
                return False
            return True

        user32.EnumWindows(_WNDENUMPROC(cb), 0)
        return result[0] if result else None

    def activate(self, hwnd):
        h = HWND(hwnd)
        if user32.IsIconic(h):
            user32.ShowWindow(h, SW_RESTORE)
        # Windows only lets the foreground thread hand over focus: attach to its input queue.
        fg = user32.GetForegroundWindow()
        fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        my_tid = kernel32.GetCurrentThreadId()
        attached = bool(fg_tid) and fg_tid != my_tid and user32.AttachThreadInput(my_tid, fg_tid, True)
        user32.BringWindowToTop(h)
        user32.SetForegroundWindow(h)
        if attached:
            user32.AttachThreadInput(my_tid, fg_tid, False)

    # --- hotkey (polled; no message loop needed) ---------------------------
    def grab_hotkey(self):
        return True

    def hotkey_pressed(self):
        down = all(user32.GetAsyncKeyState(k) & 0x8000 for k in (VK_CONTROL, VK_SHIFT, VK_G))
        fired = down and not self._hotkey_down
        self._hotkey_down = down
        return fired
