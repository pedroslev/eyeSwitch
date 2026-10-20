"""OS backends. `Session` is the backend class for the running platform."""
import importlib
import os
import sys

if sys.platform.startswith("win"):
    _module = "windows"
elif sys.platform == "darwin":
    _module = "macos"
else:
    _module = "x11"


def Session():
    """Create a backend session. Call it from the thread that will use it."""
    return importlib.import_module(f".{_module}", __name__).Session()


def config_dir():
    if _module == "windows":
        return os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "eyeSwitch")
    if _module == "macos":
        return os.path.expanduser("~/Library/Application Support/eyeSwitch")
    return os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "eyeswitch")


def platform_name():
    return _module


def unsupported_reason():
    """Why focus control cannot work in this session, or None."""
    if _module == "x11" and os.environ.get("XDG_SESSION_TYPE") == "wayland":
        return "Wayland session: focus control only works on X11 (log in with an X11 session)."
    return None
