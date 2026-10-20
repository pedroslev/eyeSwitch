#!/usr/bin/env bash
# One-time setup of the "eyeSwitch Camera" virtual camera (v4l2loopback, Linux). Needs sudo.
# Afterwards, enable "Share the camera with video calls" in eyeSwitch and pick
# "eyeSwitch Camera" in Zoom / Meet / Teams.
set -euo pipefail

LABEL="eyeSwitch Camera"
NR=10
SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"

if ! modinfo v4l2loopback >/dev/null 2>&1; then
    echo "Installing the v4l2loopback kernel module..."
    if command -v apt-get >/dev/null; then
        $SUDO apt-get install -y v4l2loopback-dkms "linux-headers-$(uname -r)"
    elif command -v pacman >/dev/null; then
        $SUDO pacman -S --needed v4l2loopback-dkms linux-headers
    else
        echo "Install v4l2loopback for your distro first (Fedora: akmod-v4l2loopback from RPM Fusion), then rerun."
        exit 1
    fi
fi

echo "Configuring it to load at boot..."
echo "v4l2loopback" | $SUDO tee /etc/modules-load.d/eyeswitch-v4l2loopback.conf >/dev/null
echo "options v4l2loopback devices=1 video_nr=$NR card_label=\"$LABEL\" exclusive_caps=1" |
    $SUDO tee /etc/modprobe.d/eyeswitch-v4l2loopback.conf >/dev/null

if lsmod | grep -q '^v4l2loopback'; then
    $SUDO modprobe -r v4l2loopback || {
        echo "v4l2loopback is already loaded and in use (OBS?). Close what uses it and rerun, or reboot."
        exit 1
    }
fi
$SUDO modprobe v4l2loopback devices=1 video_nr=$NR card_label="$LABEL" exclusive_caps=1

if grep -qx "$LABEL" /sys/class/video4linux/*/name 2>/dev/null; then
    echo "Done: \"$LABEL\" is ready (/dev/video$NR)."
else
    echo "The module loaded but the device was not found; check 'dmesg | tail'."
    exit 1
fi
