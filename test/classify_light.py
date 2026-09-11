"""
author myat

Manually run light classification on a real image.
Usage:
    python -m test.classify_light path/to/image.jpg
"""

import argparse
import cv2
import sys
import numpy as np
from pathlib import Path

from ego_vision.perception.light_classifier import (
    classify_light,
)


# _RED_RANGES = [
#     ((0, 120, 100), (10, 255, 255)),
#     ((170, 120, 100), (180, 255, 255)),
# ]
# _YELLOW_RANGE = ((15, 120, 120), (35, 255, 255))
# _GREEN_RANGE = ((40, 80, 80), (90, 255, 255))

# # Below this fraction of the crop matching any color, return "unknown".
# _MIN_COLOR_FRACTION = 0.02


# def classify_light(crop_bgr) -> str:
#     if crop_bgr is None or crop_bgr.size == 0:
#         return "unknown"

#     hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

#     red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
#     for lo, hi in _RED_RANGES:
#         red_mask = cv2.bitwise_or(
#             red_mask, cv2.inRange(hsv, np.array(lo), np.array(hi))
#         )
#     yellow_mask = cv2.inRange(hsv, np.array(_YELLOW_RANGE[0]), np.array(_YELLOW_RANGE[1]))
#     green_mask = cv2.inRange(hsv, np.array(_GREEN_RANGE[0]), np.array(_GREEN_RANGE[1]))

#     counts = {
#         "red": int(red_mask.sum() // 255),
#         "yellow": int(yellow_mask.sum() // 255),
#         "green": int(green_mask.sum() // 255),
#     }

#     total = crop_bgr.shape[0] * crop_bgr.shape[1]
#     best = max(counts, key=counts.get)
#     if total == 0 or counts[best] / total < _MIN_COLOR_FRACTION:
#         return "unknown"
#     return best


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path, help="path to the image file")
    return p.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = parse_args(argv)

    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 2

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"Could not read image: {args.image}")
        return 2

    label = classify_light(frame)

    print(f"\nimage:     {args.image}")
    print(f"predicted: {label}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
