"""Explainable proximity/path rules and conservative alternative-side evidence."""
from dataclasses import dataclass, field
import numpy as np
from ego_vision.reasoning.ego_zone import normalized_footprint
from ego_vision.reasoning.motion import Motion

SEVERITY = {"LOW": 0, "CAUTION": 1, "UNKNOWN": 2, "HIGH": 3, "CRITICAL": 4}
SIDE_REGIONS = {"LEFT": (.06, .58, .35, .92), "RIGHT": (.65, .58, .94, .92)}


@dataclass
class Threat:
    detection: object
    motion: Motion
    current_overlap: bool
    predicted_overlap: bool
    relevant: bool
    risk: str
    reason: str


@dataclass(frozen=True)
class SideEvidence:
    usable: bool = False
    clearance: float | None = None
    reason: str = "depth unavailable"


@dataclass
class Assessment:
    risk: str
    reason: str
    threat: Threat | None = None
    threats: list = field(default_factory=list)
    sides: dict = field(default_factory=dict)
    healthy: bool = True


def depth_usable(depth):
    if depth is None or depth.size == 0:
        return False
    h, w = depth.shape
    region = depth[int(.5*h):int(.95*h), int(.05*w):int(.95*w)]
    return region.size > 0 and np.mean(np.isfinite(region) & (region > .05) & (region <= 200)) >= .7


def side_evidence(side, depth, threats, horizon):
    x1, y1, x2, y2 = SIDE_REGIONS[side]
    # All detected lower-image obstacles matter, even those with unknown depth.
    for threat in threats:
        d = threat.detection
        if d.distance_m is not None and d.distance_m > 5:
            continue
        for a, b, y in threat.motion.sweep(horizon):
            if y >= y1 and b + .025 >= x1 and a - .025 <= x2:
                return SideEvidence(reason="detected obstacle")
    if not depth_usable(depth):
        return SideEvidence()
    h, w = depth.shape
    patch = depth[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
    clearances = []
    # Every tile must have useful depth; a distant background alone is insufficient.
    for row in np.array_split(patch, 3, axis=0):
        for tile in np.array_split(row, 3, axis=1):
            good = tile[np.isfinite(tile) & (tile > .05) & (tile <= 200)]
            if tile.size < 9 or good.size / tile.size < .8:
                return SideEvidence(reason="incomplete depth coverage")
            clearance = float(np.percentile(good, 10))
            if clearance < 2.0:
                return SideEvidence(False, clearance, "near depth surface")
            # Very dispersed depth is weak evidence for the whole candidate region.
            if (np.percentile(good, 75) - np.percentile(good, 25)) / np.median(good) > .65:
                return SideEvidence(reason="uncertain depth")
            clearances.append(clearance)
    return SideEvidence(True, min(clearances), "depth-supported candidate")


def assess(detections, motions, frame_shape, corridor, depth=None, healthy=True, horizon=1.2):
    threats = []
    for d in detections:
        m = motions.get(d.track_id, Motion(normalized_footprint(d.box_xyxy, frame_shape)))
        current = corridor.intersects(m.footprint)
        predicted = any(corridor.intersects(p) for p in m.sweep(horizon))
        dist = d.distance_m
        # Distance invalidation is performed by the TTC/freshness layer.
        immediate = dist is not None and dist <= .8
        relevant = current or predicted or immediate
        risk, reason = "LOW", "outside predicted path"
        if relevant:
            if dist is None:
                risk, reason = "UNKNOWN", "relevant obstacle; depth unknown"
            elif immediate or (d.ttc_valid and d.ttc_sec <= 1.2):
                risk, reason = "CRITICAL", "immediate proximity or short TTC"
            elif dist <= 2.5 or (d.ttc_valid and d.ttc_sec <= 4):
                risk, reason = "HIGH", "obstacle in current or predicted path"
            elif dist <= 5 or (predicted and not current and dist <= 8):
                risk, reason = "CAUTION", "path conflict ahead"
            else:
                risk, reason = "LOW", "distant path obstacle"
        threats.append(Threat(d, m, current, predicted, relevant, risk, reason))
    selected = max((t for t in threats if t.relevant and t.risk != "LOW"),
                   key=lambda t: (SEVERITY[t.risk], -(t.detection.distance_m or 999)), default=None)
    sides = {s: side_evidence(s, depth, threats, horizon) for s in SIDE_REGIONS}
    if not healthy or not depth_usable(depth):
        # CRITICAL remains visible, but movement is forbidden by healthy=False.
        if selected and selected.risk == "CRITICAL":
            return Assessment("CRITICAL", selected.reason, selected, threats, sides, False)
        return Assessment("UNKNOWN", "observations unavailable or stale", selected, threats, sides, False)
    if selected:
        return Assessment(selected.risk, selected.reason, selected, threats, sides)
    return Assessment("LOW", "no relevant detected threat", None, threats, sides)
