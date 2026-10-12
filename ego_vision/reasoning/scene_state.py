"""One scene assembly path for visualization, decisions and CSV evidence."""
from ego_vision.reasoning.risk import assess


def assemble(detections, frame_shape, corridor, motions, depth=None, healthy=True, horizon=1.2):
    return assess(detections, motions, frame_shape, corridor, depth, healthy, horizon)
