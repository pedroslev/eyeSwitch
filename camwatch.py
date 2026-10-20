"""Linux: notice when another process opens the camera device (inotify), so we can step aside."""
import ctypes
import os
import select
import threading
import time

IN_OPEN = 0x20
EVENT_HEADER = 16                    # sizeof(struct inotify_event) without the name


class CameraWatcher:
    def __init__(self, path):
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self._fd < 0 or libc.inotify_add_watch(self._fd, path.encode(), IN_OPEN) < 0:
            raise OSError(ctypes.get_errno(), f"cannot watch {path}")
        self.path = path
        self.foreign_open = threading.Event()       # set when someone else opened the device
        self._ignore_until = 0.0
        self._wake_r, self._wake_w = os.pipe()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def ignore_for(self, seconds):
        """Our own opens (and their retries) must not count as someone else's."""
        self._ignore_until = time.monotonic() + seconds

    def _run(self):
        while True:
            ready, _, _ = select.select([self._fd, self._wake_r], [], [])
            if self._wake_r in ready:
                return
            try:
                data = os.read(self._fd, 4096)
            except BlockingIOError:
                continue
            if data and time.monotonic() >= self._ignore_until:
                self.foreign_open.set()

    def stop(self):
        os.write(self._wake_w, b"x")
        self._thread.join(timeout=1)
        for fd in (self._fd, self._wake_r, self._wake_w):
            os.close(fd)
