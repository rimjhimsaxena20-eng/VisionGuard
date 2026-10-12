"""Video PTS/fps timing; latest-frame live capture with monotonic timestamps."""
from dataclasses import dataclass
from pathlib import Path
import math
import threading
import time


def parse_source(value):
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Camera index must be nonnegative")
        return value
    value = str(value).strip()
    if value.isdecimal():
        return int(value)
    if not value or (value.startswith('-') and value[1:].isdecimal()):
        raise ValueError("Use a nonnegative camera index or local video path")
    return Path(value).expanduser()


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class FramePacket:
    index: int
    frame: object
    timestamp: float
    captured_at: float


class FileClock:
    def __init__(self, fps):
        self.fps = fps
        self.last = None
        self.offset = 0.0
        self.use_pts = True

    def timestamp(self, index, pts_seconds):
        valid_pts = math.isfinite(pts_seconds) and pts_seconds >= 0
        if self.use_pts and valid_pts and (self.last is None or pts_seconds > self.last):
            value = pts_seconds
        else:
            if self.use_pts:
                self.offset = (self.last + 1 / self.fps - index / self.fps) if self.last is not None else 0
            self.use_pts = False  # Never bounce between inconsistent clock origins.
            value = self.offset + index / self.fps
        self.last = value
        return value


class VideoReader:
    def __init__(self, source, fps_override=None, camera_width=640, camera_height=480):
        import cv2
        self.cv2 = cv2
        self.source = parse_source(source)
        self.live = isinstance(self.source, int)
        if not self.live and not self.source.is_file():
            raise FileNotFoundError(f"Video file not found: {self.source}")
        self._cap = cv2.VideoCapture(self.source if self.live else str(self.source))
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest = None
        self._error = None
        self._thread = None
        try:
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open {'camera' if self.live else 'video'} {source}. Check permissions/source.")
            if self.live:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fps = fps_override or self._cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                if self.live:
                    fps = 30.0  # Recording metadata only; never used for live motion.
                else:
                    raise ValueError("Video FPS missing/invalid; specify --source-fps with its known FPS")
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError("Source reported invalid dimensions")
            count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.meta = VideoMeta(fps, width, height, int(count) if math.isfinite(count) and count > 0 else 0)
            self.clock = FileClock(fps)
        except Exception:
            self._cap.release()
            raise

    @property
    def failed(self):
        return self._error is not None

    def _capture(self):
        index = 0
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                captured = time.monotonic()
                if not ok:
                    raise RuntimeError("Camera capture failed; guidance stopped")
                with self._condition:
                    self._latest = FramePacket(index, frame, captured, captured)
                    self._condition.notify_all()
                index += 1
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def frames(self):
        if self.live:
            self._thread = threading.Thread(target=self._capture, name="camera", daemon=True)
            self._thread.start()
            last_index = -1
            while not self._stop.is_set():
                with self._condition:
                    available = self._condition.wait_for(
                        lambda: self._error or (self._latest and self._latest.index != last_index), timeout=2.0)
                    if self._error:
                        raise self._error
                    if not available:
                        raise RuntimeError("Camera timed out; guidance stopped")
                    packet = self._latest
                last_index = packet.index
                yield packet
        else:
            index = 0
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    if index == 0 or (self.meta.frame_count and index < self.meta.frame_count - 1):
                        raise RuntimeError(f"Video decoding failed at frame {index}")
                    return
                pts = self._cap.get(self.cv2.CAP_PROP_POS_MSEC) / 1000
                timestamp = self.clock.timestamp(index, pts)
                yield FramePacket(index, frame, timestamp, time.monotonic())
                index += 1

    def release(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=.5)
        self._cap.release()
        if self._thread:
            self._thread.join(timeout=.5)
