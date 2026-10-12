"""VisionGuard defaults. Thresholds are demonstration heuristics, not probabilities."""
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.50
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle", "train"})
VRU_CLASSES = frozenset({"person", "dog", "cat", "horse"})
OBSTACLE_CLASSES = frozenset({"chair", "bench", "fire hydrant", "suitcase"})
ROAD_CLASSES = VEHICLE_CLASSES | VRU_CLASSES | OBSTACLE_CLASSES
GROUP_VEHICLE, GROUP_VRU, GROUP_OBSTACLE = "vehicle", "vru", "obstacle"
GROUP_CONTROL = "control"  # Legacy experimental speed module only; never enabled.
GROUP_COLORS = {GROUP_VEHICLE: (0, 200, 255), GROUP_VRU: (0, 160, 255),
                GROUP_OBSTACLE: (220, 170, 80)}


def class_group(name):
    if name in VEHICLE_CLASSES:
        return GROUP_VEHICLE
    if name in VRU_CLASSES:
        return GROUP_VRU
    if name in OBSTACLE_CLASSES:
        return GROUP_OBSTACLE
    return None


DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
INDOOR_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_SCALE = 1.0  # Multiply predictions by known_distance / predicted_distance.
DEPTH_MAX_M = 80.0
MAX_DEPTH_AGE = 1.0
MAX_LIVE_AGE = 1.5
HYSTERESIS_N = 3
EGO_ZONE_CENTER = 0.5
EGO_ZONE_BOTTOM_WIDTH = 0.32
EGO_ZONE_TOP_WIDTH = 0.10
EGO_ZONE_TOP_Y = 0.50
PREDICTION_HORIZON = 1.2
FOOTPRINT_MARGIN = 0.025
BEV_LATERAL_SCALE = 1.15  # Illustrative FOV assumption only.
BEV_MAX_RANGE_M = 10.0
OVERLAY_REFERENCE_WIDTH = 960
RISK_COLORS = {"LOW": (80, 190, 90), "UNKNOWN": (180, 180, 180),
               "CAUTION": (0, 220, 240), "HIGH": (0, 140, 255),
               "CRITICAL": (40, 40, 255)}
