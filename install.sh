#!/usr/bin/env bash
# Adds eyeSwitch to the desktop application menu (current user only). Re-run if you move this folder.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/scalable/apps"

if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$APPS/eyeswitch.desktop" "$HOME/.config/autostart/eyeswitch.desktop" "$ICONS"/eyeswitch*.svg
    echo "eyeSwitch removed from the menu."
    exit 0
fi

mkdir -p "$APPS" "$ICONS"
cp "$DIR"/data/*.svg "$ICONS/"
cat > "$APPS/eyeswitch.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=eyeSwitch
Comment=Keyboard focus follows your head across monitors
Exec=$(command -v python3) "$DIR/eyeswitch.py"
Icon=eyeswitch
Terminal=false
Categories=Utility;
StartupWMClass=eyeswitch
DESKTOP
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t "$HOME/.local/share/icons/hicolor" || true
echo "Installed. Search for 'eyeSwitch' in the application menu."
