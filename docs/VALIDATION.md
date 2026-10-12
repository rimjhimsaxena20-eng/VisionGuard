# Validation record

Implementation date: 2026-10-09. Supplied upstream commit: `bcd2a3854a5f1a629e95a8fde3c0dd27e07dde0b`.

## Executed here

Environment: Linux, Python 3.12.14, NumPy 2.3.5. This is different from the documented target Windows 11 / Python 3.11 / NumPy 1.26.4 setup.

- `python -m unittest discover -s tests -v`: **45 tests passed**. See `test-results.txt` for the actual output. Tests cover all 16 requested categories, motion discontinuity, upper-image suppression, resolution invariance, source clock fallback, side-switch persistence, CSV unknown values and integration of the real depth-sampling/TTC/motion/risk/decision code.
- Worker lifecycle tests used an explicitly controlled fake speech driver to verify worker ownership, start/cancel events, camera-failure cancellation, expiration during a stalled inference interval and shutdown. They do **not** verify SAPI, audible sound or Windows COM integration.
- `python -m compileall -q ego_vision tests`: syntax compilation succeeded.
- Core module import smoke test succeeded without downloading or importing model libraries.
- `python -m ego_vision.pipeline --help`: command parser loads and lists options without model dependencies.
- `python -m ego_vision.pipeline --source 0 --no-show --mute`: correctly exited with code 2 and `VisionGuard stopped: No module named 'cv2'`. This was an error-path check, **not a camera or inference test**.
- Attempted `python -m pip install opencv-python==4.11.0.86 --disable-pip-version-check --retries 0 --timeout 15`: unavailable package from this environment's accessible package source (`No matching distribution found`). No dependency/model stack was installed.

## Source inspection and published compatibility checks

Inspected supplied authored Python files, configuration, README, dependency list and MIT notice; inventoried generated caches and preserved original Git objects/refs. Before implementation, `git diff --ignore-space-at-eol` was empty. Camera parsing, stale TTC, cached/misaligned depth, driving rules, hardcoded UI status, missing trail source and lane import side effect were confirmed in the supplied code.

Verified the Torch 2.8.0 / Torchvision 0.23.0 pairing and CPU/CUDA 12.6 wheel commands against https://pytorch.org/get-started/previous-versions/#v280 . Depth model cards require Transformers >=4.45.0; the chosen pin 4.57.1 satisfies that published minimum. Verified published releases for Ultralytics 8.3.203, pyttsx3 2.99, LAP 0.5.12 (including a CPython 3.11 Windows wheel) and comtypes 1.4.12. This establishes a documented candidate combination, not a successful full dependency-resolution or Windows runtime test.

Reviewed matching-resolution conversions, current-frame depth attachment, cache restoration by ID, GUI/recording cleanup paths, worker expiry independent of inference progress, device forwarding, offline-loading controls and exclusion of legacy speed/lane/light control from the active loop.

Additional source/package checks: static local import-name audit passed; `git fsck --no-reflogs` succeeded; whitespace checking with inherited CRLF differences ignored succeeded. The original MIT notice and every supplied Git object were compared byte for byte against the input archive.

## Not executed here

- Full pinned dependency installation or `pip check` in a clean Python 3.11 environment.
- Actual YOLO/ByteTrack or metric-depth inference, model downloads or a real-video inference smoke test.
- Webcam access, latest-frame camera-driver behavior, Windows device permissions or disconnect handling against real hardware.
- Native Windows SAPI/COM audio, audible escalation, interrupted utterances or voice availability.
- OpenCV preview/HUD visual QA at 640×480 or 1280×720; real MP4 encoding and playback.
- CUDA, GPU memory measurements, CPU/GPU throughput or 4 GB VRAM compatibility.
- Detection accuracy, depth error, collision-warning precision/recall or controlled pedestrian trial outcomes.

## Required laptop checks

Follow README setup and run `pip check`, core tests, one muted 60-frame actual video smoke test, then webcam/audio. Repeat at both display resolutions. Confirm the device label, log latency/FPS and use Task Manager or `nvidia-smi` for memory. Verify metric depth at several known distances. Exercise all ten README demo checks, particularly both-sides-blocked and camera disconnect. Repeat with `--device cpu`; slow runs may intentionally become UNKNOWN because measurements are too old. Repeat offline after successful initial model downloads.

MP4 is constant-FPS processed-frame evidence, not a time-faithful live capture. Treat CSV source timestamps as authoritative for temporal measurements. Review the resulting recording/CSV and capture the actual laptop package versions before making performance claims.
