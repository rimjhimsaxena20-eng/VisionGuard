"""Image-space motion and swept footprint, sharing history with drawn trails."""
from collections import deque
from dataclasses import dataclass
from ego_vision.reasoning.ego_zone import normalized_footprint
from ego_vision.reasoning.ttc import robust_slope


@dataclass(frozen=True)
class Motion:
    footprint: tuple
    vx: float = 0.0
    vy: float = 0.0
    ready: bool = False
    stable: bool = True
    observations: int = 1
    direction: str | None = None
    trail: tuple = ()

    def sweep(self, horizon=1.2):
        x1, x2, y = self.footprint
        if not self.ready or not self.stable:
            return [self.footprint]
        return [(x1 + self.vx * horizon * i / 12,
                 x2 + self.vx * horizon * i / 12,
                 min(1., max(0., y + self.vy * horizon * i / 12))) for i in range(13)]


class MotionEstimator:
    def __init__(self):
        self._tracks = {}

    def update(self, detections, timestamp, frame_shape):
        for tid, buf in list(self._tracks.items()):
            if timestamp - buf[-1][0] > 2.0 or timestamp < buf[-1][0]:
                del self._tracks[tid]
        out = {}
        for d in detections:
            foot = normalized_footprint(d.box_xyxy, frame_shape)
            if d.track_id is None:
                continue
            buf = self._tracks.setdefault(d.track_id, deque(maxlen=32))
            center = (foot[0] + foot[1]) / 2
            if buf and timestamp - buf[-1][0] > 1.0:
                buf.clear()
            jump = bool(buf and abs(center - buf[-1][1]) > .25)
            if jump:
                buf.clear()
            if not buf or timestamp > buf[-1][0]:
                buf.append((timestamp, center, foot[2]))
            while len(buf) > 1 and timestamp - buf[0][0] > 1.2:
                buf.popleft()
            ready = len(buf) >= 3 and buf[-1][0] - buf[0][0] >= .25
            vx = vy = 0.0
            stable = not jump
            direction = None
            if ready:
                vx, residual = robust_slope([(t, x) for t, x, y in buf])
                vy, yr = robust_slope([(t, y) for t, x, y in buf])
                stable = stable and residual < .025 and yr < .025 and abs(vx) < .8 and abs(vy) < .5
                recent = (buf[-1][1] - buf[-2][1]) / (buf[-1][0] - buf[-2][0])
                if abs(vx) > .035 and recent * vx < -.001:
                    stable = False
                if stable and abs(vx) > .035:
                    direction = "left" if vx < 0 else "right"
            out[d.track_id] = Motion(foot, vx, vy, ready, stable, len(buf), direction,
                                    tuple((x, y) for t, x, y in buf))
        return out
