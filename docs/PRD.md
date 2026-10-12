# VisionGuard MVP product requirements

## Purpose

Assist a controlled college demonstration of selective collision-relevance warnings from a normal webcam or local video. Minimum implementation effort is prioritized by retaining pretrained YOLO, ByteTrack, metric depth and OpenCV. The novelty claim is the application-specific temporal/path relevance and selective local warning layer.

## In scope

- Windows 11 / Python 3.11 target; CUDA auto selection and explicit CPU fallback.
- Camera index and local file input, explicit capture/presentation timestamps, latest-frame webcam capture.
- YOLO11 nano, persistent explicit ByteTrack IDs, existing COCO obstacle classes only.
- One Depth Anything V2 Metric Small model, outdoor default or optional indoor, neutral calibration scale.
- Matched-frame depth sampling, bounded histories, robust closing-speed/TTC estimates, uncertainty and freshness.
- Assumed normalized trapezoidal corridor; full footprint width and short-horizon image motion.
- LOW/CAUTION/HIGH/CRITICAL/UNKNOWN; highest-priority relevant threat.
- NONE/MOVE LEFT/MOVE RIGHT/STOP, side obstacle and depth evidence, conservative fallback.
- Deterministic short local speech; bounded queue, expiry, deduplication, escalation and clean shutdown.
- OpenCV visual evidence, optional annotated MP4 and event CSV, no repeated clear narration.

## Excluded

Training, semantic sidewalk/ground segmentation, world-coordinate trajectories, calibrated BEV, guaranteed metric distances or collision probabilities, GPS, route planning, special sensors, robotics hardware, cloud inference, LLM narration, frontend/dashboard/mobile deployment and traffic-light/stop-sign driving control.

## Acceptance criteria

| Requirement | Acceptance evidence |
|---|---|
| Correct timing | File PTS/FPS independent of inference speed; live monotonic timestamps; invalid intervals rejected |
| Fresh-only depth motion | Cached-map frames do not append TTC history or sample current boxes against old pixels |
| TTC validity | Approach finite; meaningful recession clears; noise/invalid readings/gaps/duplicate timestamps handled |
| Path relevance | Wide overlapping objects and impending crossings detected; beside-path objects suppressed except immediate proximity |
| Stationary obstacles | Nearby in-path chair warns without requiring object-world motion |
| Alternative directions | Detected/near-depth obstruction vetoes that side; both blocked/unknown means STOP |
| Persistence | Movement confirmed; emergency and unknown observations bypass delay; opposite movement never retained during switch |
| Local speech | One worker/pending event, cooldown, urgent replacement, stale/obsolete cancellation, unavailable-audio fallback |
| Display/evidence | Risk distinct from UNKNOWN, approximate distances, unknown values, actual device/status, selected threat, CSV |
| Resource handling | Release capture/writer/speech/window on quit, failures and model load errors |
| Offline | Initial packages/models downloaded once; subsequent configured models loaded locally |
| Honest verification | Separate synthetic checks, source review, actual inference and untested hardware |

## Release gate

Model-free tests and syntax/core imports must pass. Laptop owner must then complete README's short actual inference test, two-resolution HUD check, camera-disconnect check, native SAPI speech/cancellation and GPU/CPU runs. Measure latency/FPS/memory, depth error and false-warning behavior before presenting performance claims. A conservative STOP caused by insufficient side evidence is valid; an unsupported movement recommendation is not.
