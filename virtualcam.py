"""Republish the camera on a v4l2loopback device, so video calls can use it while eyeSwitch tracks.

Linux only (macOS and Windows already let several apps share a camera). One-time setup:
scripts/setup-virtual-camera.sh.
"""
import glob
import os
import sys

import cv2

LABEL = "eyeSwitch Camera"
WIDTH, HEIGHT, FPS = 1280, 720, 30
AVAILABLE = sys.platform.startswith("linux")


def find_device():
    """Path of the v4l2loopback device labeled for eyeSwitch, or None."""
    for d in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            with open(f"{d}/name") as f:
                if f.read().strip() == LABEL:
                    return "/dev/" + os.path.basename(d)
        except OSError:
            continue
    return None


class VirtualCamera:
    def __init__(self, device):
        import pyvirtualcam
        self.cam = pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS, device=device,
                                       fmt=pyvirtualcam.PixelFormat.BGR)
        self.failed = None

    def send(self, frame):
        """Never raises: a broken virtual camera must not stop the tracking."""
        if self.failed:
            return
        try:
            if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
            self.cam.send(frame)
        except Exception as e:
            self.failed = str(e)

    def close(self):
        try:
            self.cam.close()
        except Exception:
            pass
