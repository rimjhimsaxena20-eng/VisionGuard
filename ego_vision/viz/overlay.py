"""Track boxes, risk, selected threat and trails from the motion estimator."""
import cv2
import numpy as np
from ego_vision.config.settings import RISK_COLORS
from ego_vision.viz.hud import text


def draw_detections(frame, threats, selected=None):
    h, w = frame.shape[:2]
    s = max(1., min(1.35, w / 960))
    for t in threats:
        d = t.detection
        x1, y1, x2, y2 = (int(v) for v in d.box_xyxy)
        x1, x2 = max(0, x1), min(w-1, x2)
        y1, y2 = max(0, y1), min(h-1, y2)
        color = RISK_COLORS[t.risk]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if t is selected else 1)
        ident = str(d.track_id) if d.track_id is not None else "?"
        distance = f"~{d.distance_m:.1f}m" if d.distance_m is not None else "depth ?"
        ttc = f"TTC {d.ttc_sec:.1f}s" if d.ttc_valid else "TTC ?"
        label = f"{'* ' if t is selected else ''}{d.class_name} #{ident} {distance} {ttc}"
        label_y = max(int(100*s), min(h-int(110*s), y1-8))
        text(frame, label, (x1, label_y), .48*s, color, 1)
        points = np.array([(int(x*w), int(y*h)) for x, y in t.motion.trail], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(frame, [points], False, color, 2)
            if t.motion.ready and t.motion.stable:
                a, b, y = t.motion.sweep(.4)[-1]
                target = (int(np.clip((a+b)/2, 0, 1)*w), int(np.clip(y, 0, 1)*h))
                cv2.arrowedLine(frame, tuple(points[-1]), target, color, 2, tipLength=.25)
    return frame
