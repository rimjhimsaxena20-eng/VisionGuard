"""Shared overlay helpers."""

import cv2

from ego_vision.config.settings import OVERLAY_REFERENCE_WIDTH


def scale(frame):
    """Per-frame scale factor for overlay sizing.

    Every viz function multiplies its pixel/font constants by this so the
    layout reads the same on any frame size.
    """
    return max(0.8, min(1.25, frame.shape[1] / OVERLAY_REFERENCE_WIDTH))


def _translucent_panel(frame, p1, p2, alpha=0.7, border=True):
    """Draw a translucent dark fill on `frame` between p1 and p2 (in-place)."""
    overlay = frame.copy()
    cv2.rectangle(overlay, p1, p2, (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    if border:
        cv2.rectangle(frame, p1, p2, (90, 90, 90), 1)
