"""Readable VisionGuard status at 640x480 and 1280x720."""
import cv2
import numpy as np
from ego_vision.config.settings import RISK_COLORS


def text(frame, message, point, font_scale=.5, color=(240, 240, 240), thickness=1):
    # Clamp long strings to available pixel width rather than shrinking below readability.
    x, y = point
    max_width = frame.shape[1] - x - 8
    while message and cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] > max_width:
        message = message[:-4] + "..." if len(message) > 4 else ""
    cv2.putText(frame, message, point, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_ego_zone(frame, zone):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], (80, 145, 50))
    cv2.addWeighted(overlay, .12, frame, .88, 0, frame)
    cv2.polylines(frame, [zone], True, (90, 200, 120), 2)


def draw_hud(frame, decision, scene, fps, device, depth_age, audio_status, tracked, count, latency, detection_fresh=True, depth_fresh=True):
    h, w = frame.shape[:2]
    s = max(1.0, min(1.35, w / 960))
    row = int(23*s)
    top = int(76*s)
    bottom = int(100*s)
    cv2.rectangle(frame, (0, 0), (w, top), (24, 28, 31), -1)
    cv2.rectangle(frame, (0, h-bottom), (w, h), (24, 28, 31), -1)
    color = RISK_COLORS[decision.risk]
    text(frame, f"VisionGuard | {decision.risk}", (12, row), .65*s, color, 2)
    age = "unknown" if depth_age is None else f"{depth_age:.2f}s"
    text(frame, f"{device} | {fps:.1f} proc FPS | latency {latency:.2f}s | depth age {age}", (12, 2*row), .48*s)
    state = "fresh" if detection_fresh else "STALE"
    depth_state = ("fresh" if depth_fresh else "cached") if scene.healthy else "UNKNOWN"
    text(frame, f"Det: {state} | IDs: {tracked}/{count} | Depth: {depth_state} | Audio: {audio_status}", (12, 3*row), .48*s)
    text(frame, f"{decision.action} | {decision.reason}", (12, h-bottom+row), .60*s, color, 2)
    sides = " | ".join(f"{name}: {e.reason}" for name, e in scene.sides.items())
    text(frame, sides, (12, h-bottom+2*row), .46*s)
    text(frame, "Assumed walking corridor | Approximate depth | q / Esc: quit", (12, h-bottom+3*row), .46*s)
    return frame
