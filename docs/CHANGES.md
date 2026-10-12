# Changes from supplied Ego Vision

Baseline `bcd2a3854a5f1a629e95a8fde3c0dd27e07dde0b`; pre-existing text differences were CRLF line endings only. See BASELINE.md for the supplied inventory. Original MIT notice, legacy source/examples and Git objects/refs are preserved. No commit, push or publication was performed.

## Reuse versus contribution

| Files/component | Reused upstream work | VisionGuard changes |
|---|---|---|
| `perception/detector.py` | Ultralytics YOLO wrapper, explicit ByteTrack, box extraction and IDs | Nano default, expanded supported COCO classes, timestamp/freshness/uncertainty/TTC fields |
| `perception/depth_estimator.py` | Metric Small model, Transformers processor, output interpolation | One resolved device, inference mode, lazy imports, optional indoor/local/offline loading |
| `perception/inference_resizer.py` | Downscale/inverse box/depth coordinate conversion | Dimension validation and minimum derived height |
| `reasoning/distance.py` | Per-detection metric depth interface | Inset robust sample, valid fraction/spread checks, matched-frame timestamps, neutral scale |
| `reasoning/ttc.py` | Per-ID temporal distance history and distance/closing-speed TTC | Explicit time, bounded robust trend, freshness, recession/invalid clearing, gap/jump/expiry handling |
| `reasoning/ego_zone.py` | Configurable trapezoidal region | Normalized width-aware intersection and uncertainty margin |
| `reasoning/scene_state.py` | Single scene assembly stage | Unified pedestrian assessment shared by decisions, HUD and logging |
| `decision/rule_engine.py` | Deterministic priority rules and hysteresis concept | Pedestrian risk/actions, evidence-gated side recommendations, emergency bypass and safe switching |
| `pipeline.py`, `ingest/video_reader.py` | OpenCV input, per-frame pipeline, display and recording | Camera indices, explicit clocks, latest-frame live capture, integrated speech/CSV, errors/cleanup |
| `viz/*` | OpenCV boxes/corridor/BEV concept; BEV/shared helpers retained | VisionGuard HUD, readable text, real statuses, selected threat, motion trails, optional labeled BEV |
| `config/settings.py` | Shared class/configuration structure | Pedestrian defaults/classes and thresholds; removes inherited 0.35 calibration |
| `perception/lane_detector.py` | Entire legacy demo algorithm retained | Import-time invocation moved behind main guard |
| `reasoning/speed.py`, light classifier, original `test/` | Retained legacy examples | Remain disconnected from active pipeline |

## New functional modules

1. `reasoning/motion.py`: one bounded normalized position history for prediction and trails.
2. `reasoning/risk.py`: path/proximity relevance, explainable risk and obstacle/depth side evidence.
3. `audio/warnings.py`: bounded warning policy and cancellable local worker (`audio/__init__.py` added).

Other additions: `tests/test_core.py`, `tests/test_integration.py`, `requirements-test.txt` and repository documentation. Runtime requirements are pinned; unused MoviePy/scikit-image/scikit-learn direct requirements were removed after reviewing retained imports. NumPy, Torch/Torchvision, Pillow, tracker LAP and local speech dependencies are explicitly declared. `.gitignore` covers virtual environments, weights, credentials, recordings/logs and caches. Generated `.pyc` files and `__pycache__` directories were removed from the deliverable working tree; original history still contains its earlier versions.

## Audit issue mapping

| Supplied issue | Resolution |
|---|---|
| 1 webcam Path/string parsing | Numeric indices remain int; explicit path handling |
| 2 stale TTC | Derived fields always reset; retreat invalidates approaching history |
| 3 cached depth as fresh | Only matched depth attaches new measurement timestamp |
| 4 new boxes on old maps | Reuse distance by track ID; never resample old map with a new box |
| 5 FPS/frame-count TTC | Measurement seconds; video PTS/FPS and live monotonic clocks |
| 6 emergency delayed | CRITICAL/HIGH STOP and UNKNOWN bypass ordinary delay |
| 7 false approaching narration | Direction/closing history gates motion wording |
| 8 discarded chairs/benches | Existing COCO obstacle labels retained throughout |
| 9 bottom-edge 9×9 sample | Clipped inset region and median |
| 10 invalid depth | Positive finite/range, coverage and spread checks |
| 11 inherited scale/comment | Neutral 1.0; known/predicted calibration formula |
| 12 center-only path | Footprint width, swept projection and normalized margin |
| 13 inconsistent device | One validated resolved device for both models |
| 14 invalid intervals/dimensions | CLI/config/source checks |
| 15 hardcoded OK | Actual detection freshness, track count, depth freshness, audio state |
| 16 tiny UI | Text size floor and width-aware clipping; laptop visual QA outstanding |
| 17 lane import side effect | Main guard |
| 18 experimental speed | Kept disabled |
| 19 missing motion-trails source | New estimator's position history rendered directly |
| 20 deps/tests | Exact direct pins and focused model-free tests |
| 21 generated caches | Expanded ignore rules; caches removed without rewriting history |
