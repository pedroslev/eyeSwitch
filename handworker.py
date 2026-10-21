"""The hand recognizer, run in its own process (`python handworker.py MODEL`).

Two MediaPipe graphs (face and hand) in one process crashed the whole app with "Fatal Python error:
none_dealloc" on a real session. Here a crash only costs the hand signal, and the 25-30 ms per recognition
leaves the tracking thread. Protocol on stdin/stdout, little-endian: request = width, height, timestamp ms
(3 x uint32) + RGB bytes; reply = one text line "name score" or "-".
"""
import struct
import sys

HEADER = struct.Struct("<III")


def serve(model_path, stdin=None, stdout=None):
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    stdin, stdout = stdin or sys.stdin.buffer, stdout or sys.stdout.buffer
    recognizer = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_hands=1))
    stdout.write(b"ready\n")
    stdout.flush()
    while True:
        head = stdin.read(HEADER.size)
        if len(head) < HEADER.size:
            return
        w, h, ts = HEADER.unpack(head)
        data = stdin.read(w * h * 3)
        if len(data) < w * h * 3:
            return
        rgb = np.frombuffer(data, np.uint8).reshape(h, w, 3)
        res = recognizer.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        top = res.gestures[0][0] if res.gestures else None
        stdout.write((f"{top.category_name} {top.score:.3f}\n" if top else "-\n").encode())
        stdout.flush()


if __name__ == "__main__":
    serve(sys.argv[1])
