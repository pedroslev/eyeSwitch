"""Start-at-login for the current user, per OS."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "eyeswitch.py")


def _python():
    exe = sys.executable
    if sys.platform.startswith("win"):          # pythonw has no console window
        w = os.path.join(os.path.dirname(exe), "pythonw.exe")
        return w if os.path.exists(w) else exe
    return exe


if sys.platform.startswith("win"):
    import winreg
    _KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"

    def is_enabled():
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _KEY) as k:
                winreg.QueryValueEx(k, "eyeSwitch")
            return True
        except FileNotFoundError:
            return False

    def set_enabled(on):
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _KEY, 0, winreg.KEY_SET_VALUE) as k:
            if on:
                winreg.SetValueEx(k, "eyeSwitch", 0, winreg.REG_SZ,
                                  f'"{_python()}" "{SCRIPT}" --background')
            else:
                try:
                    winreg.DeleteValue(k, "eyeSwitch")
                except FileNotFoundError:
                    pass

elif sys.platform == "darwin":
    _PLIST = os.path.expanduser("~/Library/LaunchAgents/io.github.pedroslev.eyeswitch.plist")

    def is_enabled():
        return os.path.exists(_PLIST)

    def set_enabled(on):
        if not on:
            if os.path.exists(_PLIST):
                os.remove(_PLIST)
            return
        os.makedirs(os.path.dirname(_PLIST), exist_ok=True)
        with open(_PLIST, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                    '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0"><dict>\n'
                    '<key>Label</key><string>io.github.pedroslev.eyeswitch</string>\n'
                    f'<key>ProgramArguments</key><array><string>{_python()}</string>'
                    f'<string>{SCRIPT}</string><string>--background</string></array>\n'
                    '<key>RunAtLoad</key><true/>\n</dict></plist>\n')

else:
    _DESKTOP = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
                            "autostart", "eyeswitch.desktop")

    def is_enabled():
        return os.path.exists(_DESKTOP)

    def set_enabled(on):
        if not on:
            if os.path.exists(_DESKTOP):
                os.remove(_DESKTOP)
            return
        os.makedirs(os.path.dirname(_DESKTOP), exist_ok=True)
        with open(_DESKTOP, "w") as f:
            f.write("[Desktop Entry]\nType=Application\nName=eyeSwitch\nIcon=eyeswitch\n"
                    f'Exec="{_python()}" "{SCRIPT}" --background\n'
                    "X-GNOME-Autostart-enabled=true\n")
