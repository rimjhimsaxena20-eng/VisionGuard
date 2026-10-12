"""Robust inset sampling; distances are approximate camera-relative metric depth."""
from dataclasses import dataclass
import math
import numpy as np
from ego_vision.config.settings import DEPTH_SCALE, DEPTH_MAX_M


@dataclass(frozen=True)
class DepthSample:
    distance: float | None
    relative_spread: float | None
    valid_fraction: float


def sample_depth(depth_map, box_xyxy, scale=DEPTH_SCALE, max_m=DEPTH_MAX_M):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Depth scale must be finite and positive")
    if depth_map is None or depth_map.ndim != 2:
        return DepthSample(None, None, 0.0)
    h, w = depth_map.shape
    if not all(math.isfinite(v) for v in box_xyxy):
        return DepthSample(None, None, 0.0)
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return DepthSample(None, None, 0.0)
    # Exclude box edges/ground; keep the middle/lower object interior.
    xa, xb = max(0, int(x1 + .2 * bw)), min(w, math.ceil(x2 - .2 * bw))
    ya, yb = max(0, int(y1 + .35 * bh)), min(h, math.ceil(y2 - .15 * bh))
    if xb <= xa or yb <= ya:
        return DepthSample(None, None, 0.0)
    patch = depth_map[ya:yb, xa:xb]
    valid = patch[np.isfinite(patch) & (patch > .05) & (patch <= max_m)]
    fraction = valid.size / patch.size
    if valid.size < 9 or fraction < .6:
        return DepthSample(None, None, fraction)
    median = float(np.median(valid))
    spread = float(np.percentile(valid, 75) - np.percentile(valid, 25)) / median
    if spread > .65:
        return DepthSample(None, spread, fraction)
    return DepthSample(median * scale, spread, fraction)


def sample_distance(depth_map, box_xyxy, scale=DEPTH_SCALE):
    return sample_depth(depth_map, box_xyxy, scale).distance


def attach_depth(detections, depth_map, timestamp, scale=1.0, max_m=80.0):
    """Only call with boxes and a depth map from the SAME captured frame."""
    for d in detections:
        sample = sample_depth(depth_map, d.box_xyxy, scale, max_m)
        d.distance_m = sample.distance
        d.depth_uncertainty = sample.relative_spread
        d.distance_timestamp = timestamp
        d.depth_fresh = True
