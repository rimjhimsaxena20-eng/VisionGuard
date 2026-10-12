"""Optional, low-rate JPEG preview stream for the desktop control center."""
from __future__ import annotations

import socket
import struct
import time


class PreviewStream:
    """Send small preview frames over a local TCP connection.

    Connection setup happens after model loading. Frames are resized, JPEG
    encoded, and sent at a capped rate; transport failures never stop inference.
    """

    def __init__(self, address: str, max_fps: float = 12.0, max_width: int = 800):
        host, port = address.rsplit(":", 1)
        self.socket = socket.create_connection((host, int(port)), timeout=2.0)
        self.socket.settimeout(0.1)
        self.max_fps = max_fps
        self.max_width = max_width
        self.last_sent = 0.0
        self.closed = False

    def send(self, frame, cv2):
        if self.closed:
            return
        now = time.monotonic()
        if now - self.last_sent < 1.0 / self.max_fps:
            return
        self.last_sent = now
        height, width = frame.shape[:2]
        if width > self.max_width:
            new_height = max(1, round(height * self.max_width / width))
            frame = cv2.resize(frame, (self.max_width, new_height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            return
        payload = encoded.tobytes()
        try:
            self.socket.sendall(struct.pack("!I", len(payload)) + payload)
        except OSError:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
