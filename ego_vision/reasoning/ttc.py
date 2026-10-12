"""Longitudinal TTC from bounded, timestamped, fresh metric-depth samples."""
from collections import deque
from dataclasses import dataclass, field
import math
from statistics import median


def robust_slope(samples):
    """Theil-Sen slope and median absolute residual; no scipy dependency."""
    slopes = [(b[1] - a[1]) / (b[0] - a[0])
              for i, a in enumerate(samples) for b in list(samples)[i + 1:]
              if b[0] - a[0] >= .04]
    if not slopes:
        return 0.0, 0.0
    slope = median(slopes)
    intercept = median(v - slope * (t - samples[0][0]) for t, v in samples)
    residual = median(abs(v - (intercept + slope * (t - samples[0][0])))
                      for t, v in samples)
    return slope, residual


@dataclass
class TrackDepth:
    samples: deque = field(default_factory=lambda: deque(maxlen=24))
    uncertainty: float | None = None
    closing: float | None = None
    ttc: float | None = None
    status: str = "insufficient history"


class TtcEstimator:
    def __init__(self, max_age=1.0, max_gap=1.0, expire_after=2.0,
                 min_span=.25, max_speed=20.0, max_distance=200.0):
        self.max_age, self.max_gap = max_age, max_gap
        self.expire_after, self.min_span = expire_after, min_span
        self.max_speed, self.max_distance = max_speed, max_distance
        self._tracks = {}

    def update(self, detections, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError("Measurement timestamp must be finite")
        for tid, state in list(self._tracks.items()):
            if timestamp - state.samples[-1][0] > self.expire_after or timestamp < state.samples[-1][0]:
                del self._tracks[tid]
        for d in detections:
            d.ttc_sec = d.closing_mps = d.estimate_age = None
            d.ttc_valid = False
            d.ttc_status = "insufficient history"
            state = self._tracks.get(d.track_id)
            if d.depth_fresh:
                t, value = d.distance_timestamp, d.distance_m
                valid = (t is not None and math.isfinite(t) and t <= timestamp
                         and value is not None and math.isfinite(value)
                         and .05 < value <= self.max_distance)
                if not valid:
                    self._tracks.pop(d.track_id, None)
                    d.distance_m = d.distance_age = None
                    d.ttc_status = "invalid depth"
                    continue
                if d.track_id is not None:
                    state = self._ingest(d.track_id, t, value, d.depth_uncertainty)
                    d.distance_timestamp, d.distance_m = state.samples[-1]
                d.distance_age = timestamp - d.distance_timestamp
            elif state:
                d.distance_timestamp, d.distance_m = state.samples[-1]
                d.depth_uncertainty = state.uncertainty
                d.distance_age = timestamp - d.distance_timestamp
            else:
                d.distance_m = d.distance_timestamp = d.distance_age = None
            if d.distance_age is not None and d.distance_age > self.max_age:
                d.distance_m = None
                d.ttc_status = "stale depth"
                continue
            if state:
                d.estimate_age = timestamp - state.samples[-1][0]
                d.closing_mps, d.ttc_sec = state.closing, state.ttc
                d.ttc_status = state.status
                d.ttc_valid = state.ttc is not None
        return detections

    def invalidate_motion(self, detection):
        self._tracks.pop(detection.track_id, None)
        detection.ttc_sec = detection.closing_mps = detection.estimate_age = None
        detection.ttc_valid = False
        detection.ttc_status = "unstable image motion"

    def _ingest(self, tid, t, distance, uncertainty):
        state = self._tracks.setdefault(tid, TrackDepth())
        buf = state.samples
        if buf and t <= buf[-1][0]:
            return state
        if buf:
            dt, delta = t - buf[-1][0], distance - buf[-1][1]
            if dt > self.max_gap or abs(delta) / dt > self.max_speed:
                buf.clear()
            elif delta > max(.08, .2 * dt):
                # Meaningful retreat immediately invalidates approaching history.
                buf.clear()
        buf.append((t, distance))
        while len(buf) > 1 and t - buf[0][0] > 1.2:
            buf.popleft()
        state.uncertainty = uncertainty
        state.closing = state.ttc = None
        state.status = "insufficient history"
        if len(buf) < 3 or buf[-1][0] - buf[0][0] < self.min_span:
            return state
        slope, noise = robust_slope(buf)
        span = buf[-1][0] - buf[0][0]
        state.closing = -slope
        state.status = "not approaching"
        if -slope > .2 and -slope * span > max(.1, 3 * noise):
            state.ttc = distance / -slope
            state.status = "approaching"
        return state
