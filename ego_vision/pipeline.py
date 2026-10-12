"""VisionGuard: python -m ego_vision.pipeline --source 0 --show."""
import argparse
import csv
import math
import os
from pathlib import Path
import signal
import sys
import time
from ego_vision.config import settings as cfg
from ego_vision.ingest.video_reader import parse_source


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def fraction(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VisionGuard pedestrian collision-warning prototype")
    p.add_argument("--source", "--input", "-i", default="0", help="camera index or local video")
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--preview-stream", help=argparse.SUPPRESS)
    p.add_argument("--output", "-o", type=Path, help="annotated MP4 recording")
    p.add_argument("--events", type=Path, help="CSV measurement and speech-event log")
    p.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mute", dest="audio", action="store_false")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda:0 or GPU index")
    p.add_argument("--model", default=cfg.DEFAULT_MODEL)
    p.add_argument("--depth-model", default=cfg.DEPTH_MODEL, help="HF metric model id or downloaded local folder")
    p.add_argument("--indoor", action="store_true", help="use Indoor Small instead of Outdoor Small")
    p.add_argument("--depth-scale", type=positive_float, default=1.0)
    p.add_argument("--depth-every", type=positive_int, default=1)
    p.add_argument("--every", type=int, choices=[1], default=1, help="tracking must run on every processed frame")
    p.add_argument("--resize-width", type=nonnegative_int, default=640, help="inference downscale width; 0=native")
    p.add_argument("--camera-width", type=positive_int, default=640)
    p.add_argument("--camera-height", type=positive_int, default=480)
    p.add_argument("--source-fps", type=positive_float, help="override missing/incorrect source FPS")
    p.add_argument("--record-fps", type=positive_float, help="constant output-video FPS")
    p.add_argument("--conf", type=fraction, default=cfg.DEFAULT_CONF)
    p.add_argument("--iou", type=fraction, default=cfg.DEFAULT_IOU)
    p.add_argument("--hysteresis", type=positive_int, default=cfg.HYSTERESIS_N)
    p.add_argument("--repeat-cooldown", type=positive_float, default=5.0)
    p.add_argument("--max-depth-age", type=positive_float, default=cfg.MAX_DEPTH_AGE)
    p.add_argument("--max-live-age", type=positive_float, default=cfg.MAX_LIVE_AGE)
    p.add_argument("--horizon", type=positive_float, default=cfg.PREDICTION_HORIZON)
    p.add_argument("--corridor-center", type=fraction, default=cfg.EGO_ZONE_CENTER)
    p.add_argument("--corridor-top", type=fraction, default=cfg.EGO_ZONE_TOP_Y)
    p.add_argument("--corridor-top-width", type=fraction, default=cfg.EGO_ZONE_TOP_WIDTH)
    p.add_argument("--corridor-bottom-width", type=fraction, default=cfg.EGO_ZONE_BOTTOM_WIDTH)
    p.add_argument("--bev", action="store_true", help="illustrative, uncalibrated bird's-eye view")
    p.add_argument("--offline", action="store_true", help="require downloaded local models; no model downloads")
    p.add_argument("--download-models", action="store_true", help="download/load configured models then exit")
    p.add_argument("--max-frames", type=positive_int, help="stop after this many processed frames")
    args = p.parse_args(argv)
    try:
        args.source = parse_source(args.source)
        from ego_vision.reasoning.ego_zone import Corridor
        Corridor(args.corridor_center, args.corridor_bottom_width, args.corridor_top_width, args.corridor_top)
    except ValueError as exc:
        p.error(str(exc))
    paths = [v.resolve() for v in (args.source, args.output, args.events) if isinstance(v, Path)]
    if len(paths) != len(set(paths)):
        p.error("Source, output recording and event CSV must have different paths")
    if args.output and args.output.suffix.lower() != ".mp4":
        p.error("Use an .mp4 output filename")
    return args


class EventLog:
    FIELDS = ("kind", "source_time_s", "monotonic_s", "frame", "track_id", "class", "distance_m",
              "depth_age_s", "depth_fresh", "depth_spread", "closing_mps", "ttc_s", "ttc_valid",
              "estimate_age_s", "ttc_status", "overlap_now", "overlap_predicted", "relevant", "risk",
              "selected", "action", "reason", "warning", "left_evidence", "right_evidence")

    def __init__(self, path):
        self.file = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
            self.writer.writeheader()

    def write(self, **row):
        if self.file:
            self.writer.writerow(row)

    def observations(self, packet, scene, decision):
        common = dict(kind="observation", source_time_s=packet.timestamp, monotonic_s=time.monotonic(),
                      frame=packet.index, action=decision.action, reason=decision.reason,
                      left_evidence=scene.sides["LEFT"].reason, right_evidence=scene.sides["RIGHT"].reason)
        for t in scene.threats:
            d = t.detection
            self.write(**common, track_id=d.track_id, **{"class": d.class_name}, distance_m=d.distance_m,
                       depth_age_s=d.distance_age, depth_fresh=d.depth_fresh, depth_spread=d.depth_uncertainty,
                       closing_mps=d.closing_mps, ttc_s=d.ttc_sec, ttc_valid=d.ttc_valid,
                       estimate_age_s=d.estimate_age, ttc_status=d.ttc_status, overlap_now=t.current_overlap,
                       overlap_predicted=t.predicted_overlap, relevant=t.relevant, risk=t.risk, selected=t is scene.threat)
        if not scene.threats:
            self.write(**common, risk=scene.risk)
        if self.file:
            self.file.flush()

    def speech(self, speaker):
        for kind, now, event in speaker.drain_events():
            self.write(kind=kind, monotonic_s=now, track_id=event.key[0], **{"class": event.key[1]},
                       action=event.key[2], risk=event.key[3], warning=event.text)

    def close(self):
        if self.file:
            self.file.close()


def run(args):
    reader = writer = speaker = log = None
    preview_stream = None
    cv2 = None
    exit_code = 0
    processed = 0
    previous_sigterm = None
    try:
        # Let desktop launchers stop the process through the normal cleanup path.
        if hasattr(signal, "SIGTERM"):
            previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, signal.default_int_handler)
        # Heavy optional imports are delayed: --help and core tests need no models.
        import cv2
        from ego_vision.audio.warnings import WarningSpeaker
        from ego_vision.decision.rule_engine import RuleEngine
        from ego_vision.ingest.video_reader import VideoReader
        from ego_vision.perception.detector import YoloDetector
        from ego_vision.perception.depth_estimator import DepthEstimator, resolve_device
        from ego_vision.perception.inference_resizer import InferenceResizer
        from ego_vision.reasoning.distance import attach_depth
        from ego_vision.reasoning.ego_zone import Corridor
        from ego_vision.reasoning.motion import MotionEstimator
        from ego_vision.reasoning.scene_state import assemble
        from ego_vision.reasoning.ttc import TtcEstimator
        from ego_vision.viz.hud import draw_hud, draw_ego_zone
        from ego_vision.viz.overlay import draw_detections
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["YOLO_OFFLINE"] = "true"
            if not Path(args.model).is_file():
                raise FileNotFoundError(f"Offline YOLO weights missing: {args.model}; run --download-models online first")
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        device = resolve_device(args.device)
        if not args.download_models:
            reader = VideoReader(args.source, args.source_fps, args.camera_width, args.camera_height)
        print(f"Loading VisionGuard models on {device}...", flush=True)
        try:
            detector = YoloDetector(args.model, args.conf, args.iou, device, track=True)
            depth_estimator = DepthEstimator(cfg.INDOOR_DEPTH_MODEL if args.indoor else args.depth_model,
                                             device, args.offline)
        except Exception as exc:
            raise RuntimeError(f"Model loading failed: {exc}. Check installed versions/model downloads and --device cpu.") from exc
        if args.download_models:
            print("Models loaded. YOLO weights are in the working directory; depth files are in the Hugging Face cache.")
            return 0
        if args.preview_stream:
            try:
                from ego_vision.viz.preview_stream import PreviewStream
                preview_stream = PreviewStream(args.preview_stream)
                print("Embedded preview connected.", flush=True)
            except Exception as exc:
                print(f"Embedded preview unavailable; continuing inference: {exc}", file=sys.stderr, flush=True)
        meta = reader.meta
        resizer = InferenceResizer(meta.width, meta.height, args.resize_width)
        corridor = Corridor(args.corridor_center, args.corridor_bottom_width, args.corridor_top_width, args.corridor_top)
        engine = RuleEngine(args.hysteresis)
        ttc = TtcEstimator(max_age=args.max_depth_age)
        motion = MotionEstimator()
        log = EventLog(args.events)
        speaker = WarningSpeaker(args.audio, args.repeat_cooldown, lambda: reader.failed)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.record_fps or meta.fps, (meta.width, meta.height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open recording {args.output}; check codec, folder permissions and disk space")
        if args.show:
            cv2.namedWindow("VisionGuard", cv2.WINDOW_NORMAL)
        depth_map = depth_timestamp = None
        start = time.monotonic()
        for packet in reader.frames():
            frame = packet.frame
            if frame.shape[:2] != (meta.height, meta.width):
                raise RuntimeError("Source dimensions changed; restart to recalibrate geometry")
            inference_frame = resizer.to_inference(frame)
            detections = detector.detect(inference_frame)
            for d in detections:
                d.box_xyxy = resizer.scale_box_to_native(d.box_xyxy)
            fresh_depth = processed % args.depth_every == 0
            if fresh_depth:
                raw = resizer.scale_depth_to_native(depth_estimator.estimate_bgr(inference_frame))
                model_max = 20.0 if args.indoor else cfg.DEPTH_MAX_M
                attach_depth(detections, raw, packet.timestamp, args.depth_scale, model_max)
                import numpy as np
                depth_map = np.where(np.isfinite(raw) & (raw > .05) & (raw <= model_max),
                                     raw * args.depth_scale, np.nan)
                depth_timestamp = packet.timestamp
            # Cached depth is restored by track ID; current boxes never sample old maps.
            ttc.update(detections, packet.timestamp)
            motions = motion.update(detections, packet.timestamp, frame.shape)
            for d in detections:
                if d.track_id in motions and not motions[d.track_id].stable:
                    ttc.invalidate_motion(d)
            now = time.monotonic()
            latency = now - packet.captured_at
            depth_age = packet.timestamp - depth_timestamp + (latency if reader.live else 0)
            healthy = depth_age <= args.max_depth_age and (not reader.live or latency <= args.max_live_age) and not reader.failed
            # Invalidate per-object age at decision time for a slow live inference.
            if reader.live:
                for d in detections:
                    if d.distance_age is not None:
                        d.distance_age += latency
                        if d.estimate_age is not None:
                            d.estimate_age += latency
                        if d.distance_age > args.max_depth_age:
                            d.distance_m = d.ttc_sec = d.closing_mps = None
                            d.ttc_valid = False
                            d.ttc_status = "stale at decision time"
            usable_map = depth_map if healthy else None
            scene = assemble(detections, frame.shape, corridor, motions, usable_map, healthy, args.horizon)
            if not fresh_depth:
                from ego_vision.reasoning.risk import SideEvidence
                scene.sides = {side: SideEvidence(reason="waiting for matched depth") for side in scene.sides}
            decision = engine.decide(scene)
            if healthy:
                ttl = min(2.0, args.max_live_age-latency, args.max_depth_age-depth_age) if reader.live else 2.0
                speaker.publish(decision, ttl=max(0., ttl))
            else:
                speaker.clear()
            log.observations(packet, scene, decision)
            log.speech(speaker)
            draw_ego_zone(frame, corridor.pixels(meta.width, meta.height))
            draw_detections(frame, scene.threats, scene.threat)
            if args.bev:
                from ego_vision.viz.bev_map import draw_bev
                draw_bev(frame, detections)
            processed += 1
            fps = processed / max(.001, time.monotonic() - start)
            draw_hud(frame, decision, scene, fps, device, depth_age, speaker.status,
                     sum(d.track_id is not None for d in detections), len(detections), latency,
                     detection_fresh=(not reader.live or latency <= args.max_live_age) and not reader.failed,
                     depth_fresh=fresh_depth)
            if writer:
                writer.write(frame)
                if not writer.isOpened():
                    raise RuntimeError("Recording writer failed during processing")
            if preview_stream:
                preview_stream.send(frame, cv2)
            if processed == 1 or processed % 30 == 0:
                print(f"frame={packet.index} source={packet.timestamp:.3f}s {decision.risk} {decision.action}: {decision.reason}")
            if args.show:
                cv2.imshow("VisionGuard", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                if cv2.getWindowProperty("VisionGuard", cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_frames and processed >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("Quit requested.")
    except Exception as exc:
        print(f"VisionGuard stopped: {exc}", file=sys.stderr)
        exit_code = 2
        if log:
            log.write(kind="runtime_error", monotonic_s=time.monotonic(), risk="UNKNOWN", action="STOP", reason=str(exc))
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if preview_stream:
            preview_stream.close()
        if speaker:
            speaker.close()
            if log:
                log.speech(speaker)
        if reader:
            reader.release()
        if writer:
            writer.release()
            if processed and (not args.output.exists() or args.output.stat().st_size < 100):
                print("Recording is empty or incomplete; check codec/disk space.", file=sys.stderr)
                exit_code = 3
        if log:
            log.close()
        if cv2 is not None and args.show:
            cv2.destroyAllWindows()
    return exit_code


def main():
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
