#!/usr/bin/env python3
"""eyeSwitch: move keyboard focus to the monitor you are facing (webcam, X11)."""
import argparse
import json
import os
import statistics
import time

import cv2
import numpy as np

from tracker import HeadTracker
from xwin import XSession

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
SMOOTH = 0.4            # EMA factor for the yaw signal
HYSTERESIS = 0.25       # must be this much closer to the target than to the current monitor
MOUSE_QUIET = 1.5       # seconds without touching the mouse before we act
FAR_FACTOR = 1.5        # ignore poses farther than this many spacings from every monitor


def calibrate(xs, tracker, monitors):
    """Look at the dot on each monitor; store the median yaw of each."""
    yaws = []
    for i, m in enumerate(monitors):
        win = f"eyeSwitch calibration {i}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.moveWindow(win, m.x, m.y)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        samples, t0 = [], time.monotonic()
        while time.monotonic() - t0 < 5.0:
            elapsed = time.monotonic() - t0
            canvas = np.zeros((m.h, m.w, 3), np.uint8)
            text = f"Monitor {i + 1}: look at the dot" if elapsed < 2.5 else "Hold still..."
            cv2.putText(canvas, text, (m.w // 2 - 250, m.h // 2 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.circle(canvas, (m.w // 2, m.h // 2), 18, (0, 200, 255), -1)
            cv2.imshow(win, canvas)
            cv2.waitKey(1)
            yaw, _ = tracker.read()
            if yaw is not None and elapsed >= 2.5:   # first 2.5 s: time to move the head
                samples.append(yaw)
        cv2.destroyWindow(win)
        cv2.waitKey(1)
        if len(samples) < 10:
            raise SystemExit(f"Monitor {i + 1}: face not detected, calibration aborted")
        yaws.append(statistics.median(samples))
        print(f"monitor {i + 1}: yaw {yaws[-1]:+.1f}")
    if len(set(round(y) for y in yaws)) < len(yaws) or min(abs(a - b) for a, b in zip(yaws, yaws[1:])) < 5:
        raise SystemExit("Yaw difference between monitors is under 5 degrees: turn your head more")
    data = {"monitors": [list(m) for m in monitors], "yaw": yaws}
    with open(CALIB_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print("saved", CALIB_PATH)
    return yaws


def load_calibration(monitors):
    try:
        with open(CALIB_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    if [list(m) for m in monitors] != data["monitors"]:
        print("Monitor layout changed since calibration; recalibrate with --calibrate")
        return None
    return data["yaw"]


def pick_target(yaw, centers, current, hysteresis=HYSTERESIS):
    """Index of the monitor the head is facing, or None if ambiguous / out of range."""
    dist = [abs(yaw - c) for c in centers]
    best = int(np.argmin(dist))
    spacing = min(abs(a - b) for a, b in zip(sorted(centers), sorted(centers)[1:]))
    if dist[best] > FAR_FACTOR * spacing:
        return None
    if best != current and dist[best] > dist[current] * (1 - hysteresis):
        return current
    return best


def run(xs, tracker, monitors, centers, delay, debug):
    last_window = {}                    # monitor index -> last focused window id
    last_pointer = {}                   # monitor index -> last pointer position
    px, py = xs.pointer()
    current = xs.monitor_at(monitors, px, py) or 0
    mouse_time, smooth = 0.0, None
    cand, cand_since, paused = None, 0.0, False
    if not xs.grab_hotkey():
        print("warning: Ctrl+Shift+G is already taken, pause hotkey disabled")
    print(f"running on monitor {current + 1}. Ctrl+Shift+G pauses, Ctrl+C quits.")

    while True:
        yaw, frame = tracker.read()
        now = time.monotonic()
        if xs.hotkey_pressed():
            paused = not paused
            cand = None
            print("paused" if paused else "resumed")

        # Track what the user is doing, so we can restore focus later.
        nx, ny = xs.pointer()
        if abs(nx - px) + abs(ny - py) > 2:
            mouse_time = now
            m = xs.monitor_at(monitors, nx, ny)
            if m is not None:
                current = m
        px, py = nx, ny
        m = xs.monitor_at(monitors, px, py)
        if m is not None:
            last_pointer[m] = (px, py)
        aw = xs.active_window()
        if aw:
            wm = xs.window_monitor(aw, monitors)
            if wm is not None:
                last_window[wm] = aw

        if yaw is None:
            cand = None
            continue
        smooth = yaw if smooth is None else smooth + SMOOTH * (yaw - smooth)
        target = pick_target(smooth, centers, current)
        if debug:
            print(f"\ryaw {smooth:+6.1f}  facing {None if target is None else target + 1}  "
                  f"current {current + 1}  {'PAUSED' if paused else ''}   ", end="", flush=True)

        if paused or target is None or target == current or now - mouse_time < MOUSE_QUIET:
            cand = None
            continue
        if cand != target:
            cand, cand_since = target, now
        elif now - cand_since >= delay:
            switch(xs, monitors, target, last_window, last_pointer)
            current, cand = target, None
            px, py = xs.pointer()       # our own warp must not count as mouse activity


def switch(xs, monitors, idx, last_window, last_pointer):
    wid = last_window.get(idx)
    if wid is None or not xs.exists(wid) or xs.window_monitor(wid, monitors) != idx:
        wid = xs.top_window_on(monitors, idx)
    m = monitors[idx]
    x, y = last_pointer.get(idx, (m.x + m.w // 2, m.y + m.h // 2))
    xs.warp(x, y)
    if wid:
        xs.activate(wid)
    print(f"\n-> monitor {idx + 1}" + ("" if wid else " (no window to focus)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true", help="run calibration, then exit")
    ap.add_argument("--camera", type=int, default=0, help="camera index (/dev/videoN)")
    ap.add_argument("--delay", type=int, default=300, help="ms to face a monitor before switching")
    ap.add_argument("--debug", action="store_true", help="print live yaw and target")
    args = ap.parse_args()

    xs = XSession()
    monitors = xs.monitors()
    if len(monitors) < 2:
        raise SystemExit("Need at least 2 monitors")
    tracker = HeadTracker(args.camera)
    try:
        centers = load_calibration(monitors)
        if args.calibrate or centers is None:
            centers = calibrate(xs, tracker, monitors)
            if args.calibrate:
                return
        run(xs, tracker, monitors, centers, args.delay / 1000, args.debug)
    except KeyboardInterrupt:
        print()
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
