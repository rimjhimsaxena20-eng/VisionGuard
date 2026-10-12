"""Downscale frames for fast YOLO + depth inference; scale results back.

Keeps the pipeline's per-frame loop free of bookkeeping. Construct with the
native frame size and the desired inference width; call `to_inference()` to
get the small frame and `scale_box_to_native` / `scale_depth_to_native` to
bring detector boxes and depth maps back into the native coordinate space.

When the target width is 0 or larger than the native frame, the resizer
becomes a no-op (every method returns its input unchanged).
"""

import cv2


class InferenceResizer:
    def __init__(self, native_w, native_h, target_width):
        if native_w <= 0 or native_h <= 0 or target_width < 0:
            raise ValueError("Dimensions must be positive and resize width nonnegative")
        self.native_w = native_w
        self.native_h = native_h
        if target_width and target_width < native_w:
            self.infer_w = target_width
            self.infer_h = max(1, int(round(native_h * (target_width / native_w))))
        else:
            self.infer_w, self.infer_h = native_w, native_h
        self.scale_x = native_w / self.infer_w
        self.scale_y = native_h / self.infer_h
        self.active = (self.infer_w, self.infer_h) != (native_w, native_h)

    def to_inference(self, frame):
        if not self.active:
            return frame
        return cv2.resize(frame, (self.infer_w, self.infer_h), interpolation=cv2.INTER_AREA)

    def scale_box_to_native(self, box):
        if not self.active:
            return box
        x1, y1, x2, y2 = box
        return (x1 * self.scale_x, y1 * self.scale_y, x2 * self.scale_x, y2 * self.scale_y)

    def scale_depth_to_native(self, depth_map):
        if not self.active:
            return depth_map
        return cv2.resize(depth_map, (self.native_w, self.native_h), interpolation=cv2.INTER_NEAREST)
