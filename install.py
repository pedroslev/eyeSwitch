#!/usr/bin/env python3
"""Add eyeSwitch to the application menu of the current user (Linux, Windows, macOS).

    python install.py              install
    python install.py --uninstall  remove

Re-run it if you move this folder or switch Python environments: the launcher points at
this script and at the Python that ran the installer.
"""
import ctypes.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SCRIPT = HERE / "eyeswitch.py"
PY = sys.executable


def render_png(size, out):
    """Render data/eyeswitch.svg to a PNG with Qt (works headless)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication, QIcon
    app = QGuiApplication.instance() or QGuiApplication([])
    QIcon(str(DATA / "eyeswitch.svg")).pixmap(size, size).save(str(out), "PNG")
    return app


# --- Linux -------------------------------------------------------------------
def linux(uninstall):
    apps = Path.home() / ".local/share/applications"
    icons = Path.home() / ".local/share/icons/hicolor/scalable/apps"
    if uninstall:
        for p in [apps / "eyeswitch.desktop", Path.home() / ".config/autostart/eyeswitch.desktop",
                  *icons.glob("eyeswitch*.svg")]:
            p.unlink(missing_ok=True)
        return
    apps.mkdir(parents=True, exist_ok=True)
    icons.mkdir(parents=True, exist_ok=True)
    for svg in DATA.glob("*.svg"):
        shutil.copy(svg, icons / svg.name)
    (apps / "eyeswitch.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=eyeSwitch\n"
        "Comment=Keyboard focus follows your head across monitors\n"
        f'Exec="{PY}" "{SCRIPT}"\nIcon=eyeswitch\nTerminal=false\nCategories=Utility;\n'
        "StartupWMClass=eyeswitch\n")
    for cmd in (["update-desktop-database", str(apps)],
                ["gtk-update-icon-cache", "-q", "-t", str(Path.home() / ".local/share/icons/hicolor")]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=False)
    if not ctypes.util.find_library("xcb-cursor"):
        print("\nMissing libxcb-cursor, which Qt needs:\n"
              "  Debian/Ubuntu: sudo apt install libxcb-cursor0\n"
              "  Fedora:        sudo dnf install xcb-util-cursor\n"
              "  Arch:          sudo pacman -S xcb-util-cursor\n"
              "  openSUSE:      sudo zypper install libxcb-cursor0")
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        print("\nYou are on Wayland: eyeSwitch can only control focus on an X11 session "
              "(pick 'Xorg' or 'X11' on the login screen).")


# --- Windows -----------------------------------------------------------------
def windows(uninstall):
    start = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs"
    link = start / "eyeSwitch.lnk"
    ico_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "eyeSwitch"
    if uninstall:
        link.unlink(missing_ok=True)
        shutil.rmtree(ico_dir, ignore_errors=True)
        return
    ico_dir.mkdir(parents=True, exist_ok=True)
    ico = ico_dir / "eyeswitch.ico"
    render_png(256, ico)                     # Qt writes real .ico when the extension says so
    pyw = Path(PY).with_name("pythonw.exe")
    target = pyw if pyw.exists() else Path(PY)
    ps = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
          "$s.TargetPath='{t}';$s.Arguments='\"{script}\"';$s.WorkingDirectory='{wd}';"
          "$s.IconLocation='{ico}';$s.Save()").format(
              lnk=link, t=target, script=SCRIPT, wd=HERE, ico=ico)
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


# --- macOS -------------------------------------------------------------------
INFO_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>eyeSwitch</string>
<key>CFBundleDisplayName</key><string>eyeSwitch</string>
<key>CFBundleIdentifier</key><string>io.github.pedroslev.eyeswitch</string>
<key>CFBundleExecutable</key><string>eyeSwitch</string>
<key>CFBundleIconFile</key><string>eyeswitch</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSUIElement</key><true/>
<key>NSCameraUsageDescription</key><string>eyeSwitch uses the camera to see which monitor you are facing.</string>
</dict></plist>
"""


def macos(uninstall):
    app = Path.home() / "Applications/eyeSwitch.app"
    if uninstall:
        shutil.rmtree(app, ignore_errors=True)
        Path.home().joinpath("Library/LaunchAgents/io.github.pedroslev.eyeswitch.plist").unlink(missing_ok=True)
        return
    shutil.rmtree(app, ignore_errors=True)
    (app / "Contents/MacOS").mkdir(parents=True)
    (app / "Contents/Resources").mkdir(parents=True)
    (app / "Contents/Info.plist").write_text(INFO_PLIST)
    exe = app / "Contents/MacOS/eyeSwitch"
    exe.write_text(f'#!/bin/bash\nexec "{PY}" "{SCRIPT}" "$@"\n')
    exe.chmod(0o755)
    iconset = app / "Contents/Resources/eyeswitch.iconset"
    iconset.mkdir()
    for size in (16, 32, 128, 256, 512):
        render_png(size, iconset / f"icon_{size}x{size}.png")
        render_png(size * 2, iconset / f"icon_{size}x{size}@2x.png")
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o",
                    str(app / "Contents/Resources/eyeswitch.icns")], check=False)
    shutil.rmtree(iconset, ignore_errors=True)
    print("\nOn first launch macOS asks for Camera access. Also allow eyeSwitch under\n"
          "System Settings > Privacy & Security > Accessibility (needed to focus windows).")


def main():
    uninstall = "--uninstall" in sys.argv
    if sys.platform.startswith("win"):
        windows(uninstall)
    elif sys.platform == "darwin":
        macos(uninstall)
    else:
        linux(uninstall)
    print("eyeSwitch removed from the application menu." if uninstall else
          "Installed. Look for 'eyeSwitch' in your application menu.")


if __name__ == "__main__":
    main()
