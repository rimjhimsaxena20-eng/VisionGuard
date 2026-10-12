"""YOLO detector wrapper."""

from dataclasses import dataclass

from ego_vision.config.settings import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    ROAD_CLASSES,
    class_group,
)


@dataclass
class Detection:
    class_name: str
    class_group: str
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    track_id: int | None = None
    light_state: str | None = None
    distance_m: float | None = None
    ttc_sec: float | None = None
    distance_timestamp: float | None = None
    depth_fresh: bool = False
    distance_age: float | None = None
    depth_uncertainty: float | None = None
    closing_mps: float | None = None
    ttc_valid: bool = False
    estimate_age: float | None = None
    ttc_status: str = "insufficient history"


class YoloDetector:
    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        device=None,
        track=True,
    ):
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._conf = conf
        self._iou = iou
        self._device = device
        self._track = track
        self._names = self._model.names

    def detect(self, frame) -> list[Detection]:
        if self._track:
            results = self._model.track(
                frame,
                conf=self._conf,
                iou=self._iou,
                device=self._device,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
        else:
            results = self._model.predict(
                frame,
                conf=self._conf,
                iou=self._iou,
                device=self._device,
                verbose=False,
            )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            ids = [None] * len(boxes)

        out = []
        for (x1, y1, x2, y2), conf, cls_id, tid in zip(xyxy, confs, cls_ids, ids):
            name = self._names.get(int(cls_id), "")
            if name not in ROAD_CLASSES:
                continue
            group = class_group(name)
            if group is None:
                continue
            out.append(
                Detection(
                    class_name=name,
                    class_group=group,
                    box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    track_id=int(tid) if tid is not None else None,
                )
            )
        return out
