"""
Bird's-eye view mini-map.
Top-left transparent panel. 
Each detection with a metric distance is plotted as a dot in a top-down view.
"""

import cv2
import numpy as np

from ego_vision.config.settings import (
    BEV_LATERAL_SCALE,
    BEV_MAX_RANGE_M,
    GROUP_COLORS,
    GROUP_VEHICLE,
    GROUP_VRU,
    GROUP_OBSTACLE,
)
from ego_vision.viz.common import _translucent_panel, scale

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_bev(frame, detections, panel_size=None, max_range_m=BEV_MAX_RANGE_M):
    """Render BEV in the top-left corner."""
    s = scale(frame)
    if panel_size is None:
        panel_size = int(250 * s)
    margin = int(16 * s)
    frame_h, frame_w = frame.shape[:2]
    x0, y0 = margin, int(90 * max(1., frame_w / 960))
    x1, y1 = x0 + panel_size, y0 + panel_size

    _translucent_panel(frame, (x0, y0), (x1, y1), alpha=0.7, border=True)

    cv2.putText(frame, "Approx. view", (x0 + int(12 * s), y0 + int(24 * s)),
                FONT, 0.6 * s, (200, 200, 200), 1, cv2.LINE_AA)

    # leave room at top for title and at bottom for ego marker.
    plot_top = y0 + int(36 * s)
    plot_bottom = y1 - int(20 * s)
    ego_cx = (x0 + x1) // 2
    pixels_per_m_y = (plot_bottom - plot_top) / max_range_m
    pixels_per_m_x = (panel_size - int(20 * s)) / max_range_m

    # Distance rings (horizontal lines at 10m intervals).
    for r in range(10, int(max_range_m) + 1, 10):
        y = plot_bottom - int(r * pixels_per_m_y)
        cv2.line(frame, (x0 + int(6 * s), y), (x1 - int(6 * s), y), (45, 45, 45), 1)
        cv2.putText(frame, f"{r}m", (x0 + int(8 * s), y - int(2 * s)),
                    FONT, 0.35 * s, (110, 110, 110), 1, cv2.LINE_AA)

    # Vertical center line (ego heading).
    cv2.line(frame, (ego_cx, plot_top), (ego_cx, plot_bottom), (45, 45, 45), 1)

    # Detection dots.
    edge = int(8 * s)
    for d in detections:
        if d.distance_m is None or d.distance_m > max_range_m:
            continue
        if d.class_group not in (GROUP_VEHICLE, GROUP_VRU, GROUP_OBSTACLE):
            continue

        bbox_cx = (d.box_xyxy[0] + d.box_xyxy[2]) / 2.0
        norm_x = bbox_cx / frame_w
        lateral_m = (norm_x - 0.5) * BEV_LATERAL_SCALE * d.distance_m

        px = ego_cx + int(lateral_m * pixels_per_m_x)
        py = plot_bottom - int(d.distance_m * pixels_per_m_y)

        if px < x0 + edge or px > x1 - edge or py < plot_top or py > plot_bottom:
            continue

        color = GROUP_COLORS.get(d.class_group, (200, 200, 200))
        radius = max(2, int((5 if d.class_group == GROUP_VEHICLE else 3) * s))
        cv2.circle(frame, (px, py), radius, color, -1)
        cv2.circle(frame, (px, py), radius + 1, (240, 240, 240), 1)

    # Green triangle pointing up at the bottom-center, the car itself.
    tri_h = int(12 * s)
    tri_w = int(8 * s)
    pts = np.array([
        (ego_cx, plot_bottom - tri_h),
        (ego_cx - tri_w, plot_bottom + int(4 * s)),
        (ego_cx + tri_w, plot_bottom + int(4 * s)),
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (0, 200, 0))

    return frame
