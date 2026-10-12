"""Normalized assumed walking corridor; no semantic ground segmentation."""
from dataclasses import dataclass
import math
import numpy as np
from ego_vision.config.settings import (EGO_ZONE_CENTER, EGO_ZONE_BOTTOM_WIDTH,
    EGO_ZONE_TOP_WIDTH, EGO_ZONE_TOP_Y, FOOTPRINT_MARGIN)


@dataclass(frozen=True)
class Corridor:
    center: float = EGO_ZONE_CENTER
    bottom_width: float = EGO_ZONE_BOTTOM_WIDTH
    top_width: float = EGO_ZONE_TOP_WIDTH
    top_y: float = EGO_ZONE_TOP_Y
    margin: float = FOOTPRINT_MARGIN

    def __post_init__(self):
        values = (self.center, self.bottom_width, self.top_width, self.top_y, self.margin)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Corridor geometry must be finite")
        if not (0 < self.top_y < 1 and 0 < self.top_width <= self.bottom_width <= 1
                and self.bottom_width / 2 <= self.center <= 1 - self.bottom_width / 2
                and 0 <= self.margin <= .1):
            raise ValueError("Invalid normalized corridor geometry")

    def bounds(self, y):
        ratio = max(0., min(1., (y - self.top_y) / (1 - self.top_y)))
        half = (self.top_width + ratio * (self.bottom_width - self.top_width)) / 2
        return self.center - half, self.center + half

    def intersects(self, footprint):
        x1, x2, y = footprint
        if y < self.top_y or y > 1.1:
            return False
        left, right = self.bounds(y)
        return x2 + self.margin >= left and x1 - self.margin <= right

    def pixels(self, width, height):
        left, right = self.bounds(1)
        top_left, top_right = self.bounds(self.top_y)
        return np.array([(left * (width-1), height-1), (right * (width-1), height-1),
                         (top_right * (width-1), self.top_y * (height-1)),
                         (top_left * (width-1), self.top_y * (height-1))], dtype=np.int32)


def normalized_footprint(box, frame_shape):
    h, w = frame_shape[:2]
    return max(0., box[0] / w), min(1., box[2] / w), min(1., box[3] / h)


def contact_point(box_xyxy):
    return (box_xyxy[0] + box_xyxy[2]) / 2., box_xyxy[3]


def make_ego_zone(width, height, **kwargs):
    return Corridor(**kwargs).pixels(width, height)
