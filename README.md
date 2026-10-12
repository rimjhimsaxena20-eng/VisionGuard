# VisionGuard

A local college Computer Vision prototype for selective pedestrian obstacle warnings, adapted from [myatthukyaw/ego-vision](https://github.com/myatthukyaw/ego-vision), supplied at `bcd2a3854a5f1a629e95a8fde3c0dd27e07dde0b`.

Camera/video → YOLO11 nano → ByteTrack → Depth Anything V2 Metric Small → timestamped distance/closing speed/TTC → current and predicted walking-corridor overlap → pedestrian rules → selective local speech.

**Our contribution:** pedestrian-specific collision relevance, approximate path-intersection prediction, obstacle-aware directional recommendations and selective local audio. YOLO, ByteTrack, metric depth models and the original repository are pretrained/upstream work. No new model is trained.

This is a controlled demonstration prototype. It does not certify safe travel, detect every obstacle, identify a sidewalk, or replace a cane/guide. A LOW label means no relevant *detected* threat; UNKNOWN means essential evidence is unavailable. A GitHub link alone does not run a local webcam application.

## 1. Windows 11 setup (PowerShell)

Install 64-bit Python 3.11 and an NVIDIA driver suitable for the CUDA 12.6 PyTorch wheel if using the GPU. Extract this archive, open PowerShell in the directory containing `VisionGuard`, then:

```powershell
cd .\VisionGuard
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These commands call the virtual environment's Python directly, so PowerShell activation-policy changes are unnecessary. The Torch/Torchvision pair and CUDA/CPU install routes are listed by [PyTorch](https://pytorch.org/get-started/previous-versions/#v280). This proposed Python 3.11 dependency set was checked against published requirements; a full Windows install was **not** executed in the implementation environment. Direct dependencies are pinned; this is not a complete transitive lockfile.

**CPU route:** replace the CUDA-wheel command above with the following. It can also replace an existing CUDA Torch installation in this environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`--device auto` chooses CUDA when available, otherwise CPU. `--device cpu` forces CPU; an unavailable explicitly requested CUDA device gives an error with a CPU fallback instruction. GPU memory use, achievable FPS and 4 GB compatibility have not been measured. Nano detection, Small depth, one image per inference and 640-wide input are starting defaults, not a performance guarantee.

## 2. Download models once

```powershell
.\.venv\Scripts\python.exe -m ego_vision.pipeline --download-models --device auto
```

This loads `yolo11n.pt` into the current working directory and `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` through the Hugging Face cache. Internet access is needed for initial Python packages, YOLO weights, and depth model/config/processor files. Weights and environments are not bundled in this ZIP. ByteTrack needs no separate pretrained checkpoint; `lap` is explicitly installed so it does not need an automatic runtime install.

An indoor demonstration can use **one alternative** Small depth model:

```powershell
.\.venv\Scripts\python.exe -m ego_vision.pipeline --download-models --indoor --device auto
```

After downloading the appropriate variant, use `--offline`; keep `--indoor` for indoor runs. Depth loading then uses local cache files only, and YOLO must already exist locally. No cloud inference or external speech API is used. Windows speech uses installed SAPI voices. A different computer needs its own dependency install and downloads or correctly copied local model/cache files.

## 3. Run

### Desktop control center

Launch the integrated desktop UI from the project directory:

```powershell
.\.venv\Scripts\python.exe -m ego_vision.ui
```

Choose a camera index or local video, select compute/audio/cache options, and optionally save an annotated MP4 and event CSV under `outputs`. The processed camera preview appears inside the control center after model initialization. The preview is resized and streamed at up to 12 fps; inference and event logging continue through the existing pipeline. Offline mode is optional and expects local model files/cache; on first setup, leave it off so the depth model can download.

The UI is a launcher and session monitor, not a separate inference implementation. All hazard decisions, preview overlays, and local speech remain in the existing pipeline. The safety and hardware limitations described below still apply.

Webcam, automatic GPU/CPU, local audio:

```powershell
.\.venv\Scripts\python.exe -m ego_vision.pipeline --source 0 --show --offline
```

Prerecorded video:

```powershell
.\.venv\Scripts\python.exe -m ego_vision.pipeline --source ".\clips\demo.mp4" --show --offline
```

Mute / CPU fallback:

```powershell
.\.venv\Scripts\python.exe -m ego_vision.pipeline --source 0 --mute --device cpu --offline
```

Annotated recording plus event CSV:

```powershell
.\.venv\Scripts\python.exe -m ego_vision.pipeline --source 0 --output ".\outputs\demo.mp4" --events ".\outputs\events.csv" --offline
```

Short video smoke test on the laptop:

```powershell
.\.venv\Scripts\python.exe -m ego_vision.pipeline --source ".\clips\demo.mp4" --no-show --mute --max-frames 60 --output ".\outputs\smoke.mp4" --events ".\outputs\smoke.csv" --offline
```

Use an existing clip in place of the example path. Quit using `q`, Escape, window close or Ctrl+C. `--input` / `-i` alias `--source`. `--no-audio` aliases mute; `--no-show` supports headless processing. Camera indices are integers; for a literal numeric filename use a relative/absolute path such as `./0`. `--bev` enables an approximate illustrative view, not a calibrated map.

Default inference downscales to 640 pixels wide and preserves native recording coordinates. Use `--camera-width 1280 --camera-height 720` to request larger webcam capture, subject to camera support. UI layout targets 640×480 and 1280×720; OpenCV rendering still needs visual verification on the laptop. `--resize-width 0` retains native inference input size. Depth's processor has its own model input resizing, so lowering the frame width does not guarantee proportional memory savings.

Tracking runs on every processed frame. `--every` accepts only `1`; detector skipping was deliberately removed. `--depth-every 3` is optional: on intervening frames, only track-associated distance estimates are cached, current boxes never sample an old map, and directional recommendations wait for a matched depth frame. Start with the default `1` to validate temporal behavior. Skipping inference does not reduce resident model weight memory.

## 4. Calibration and geometry

The neutral depth multiplier is `1.0`. With the camera fixed at the intended height/angle:

1. Place a large, reliably detected object at measured distances such as 1, 2, 3 and 5 m. Measure approximately along the camera's forward direction, not as ground distance to an arbitrary pixel.
2. At `--depth-scale 1`, collect several steady displayed/CSV estimates at each distance. Reject frames with occlusion, wrong boxes or high spread.
3. For each reference distance, compute `scale = known_distance / predicted_distance`. Choose a representative median of those ratios.
4. Run `--depth-scale <scale>` and repeat the checks at all distances. Report remaining error rather than assuming one scale makes all scenes accurate.

Example: 2 m measured / 4 m predicted = scale 0.5, so `--depth-scale 0.5`. This is a multiplicative camera/model correction, not intrinsic calibration or guaranteed metric accuracy. Use the indoor variant when appropriate and calibrate it independently.

The assumed corridor is a normalized trapezoid: center 0.5, bottom width 0.32, top width 0.10, top y 0.50. Adjust `--corridor-center`, `--corridor-bottom-width`, `--corridor-top-width`, `--corridor-top` for a fixed camera placement. Width overlap uses both box edges plus a 0.025-image-width uncertainty margin. Upper-image box footprints do not qualify solely by horizontal alignment; valid immediate proximity is a separate exception.

## 5. How the decisions work

### Measurement and motion

- Video uses increasing presentation timestamps; if invalid/non-increasing, it switches to an FPS-based clock while preserving continuity. Missing file FPS requires `--source-fps` with known source FPS. Inference wall-clock time is never used for video motion.
- Live capture runs in a latest-frame thread, stamps successful reads with `time.monotonic()`, and drops superseded frames. OpenCV cannot guarantee the sensor's actual exposure time or eliminate all driver buffering.
- Depth is sampled from an inset box region on the same frame as the detection; median depth rejects invalid/nonpositive values and excessive spatial spread. No mask/segmentation is claimed.
- TTC retains at most 24 fresh samples over 1.2 s per ID, requires ≥3 samples and ≥0.25 s, and uses a median pairwise slope. It rejects >20 m/s jumps, resets after >1 s gaps, and expires tracks after >2 s.
- Closing speed must exceed 0.2 m/s and total depth change must exceed 0.1 m and three times the regression residual. `TTC = current distance / closing speed`. Meaningful recession clears prior approaching history immediately. Non-approaching and insufficient-history values are explicit, not zero.
- Image motion retains ≤32 positions over 1.2 s, requires ≥3 observations and ≥0.25 s, and predicts a swept footprint up to 1.2 s ahead. Arrows/trails use the same history. Unstable image motion invalidates TTC and forbids directional advice.
- Cached distance expires after `--max-depth-age` (default 1 s). Live inference latency also counts toward depth age. `--max-live-age` defaults to 1.5 s; evidence that arrives too late becomes UNKNOWN. Slow CPU runs can legitimately produce mostly UNKNOWN. Do not increase these limits simply to hide latency.

### Risk rules

Rules apply to current/predicted path overlap, except immediate proximity which applies anywhere in view. Thresholds are tunable demonstration heuristics, not collision probabilities.

| Risk | Condition | Result |
|---|---|---|
| CRITICAL | Valid distance ≤0.8 m, or relevant TTC ≤1.2 s | STOP immediately |
| HIGH | Relevant distance ≤2.5 m or TTC ≤4 s | Warn; STOP until a supported alternative persists |
| CAUTION | Relevant distance ≤5 m, or predicted crossing within 8 m | Persistent warning/direction if supported |
| UNKNOWN | Missing/stale essential observations or relevant object depth unavailable | STOP; cancel obsolete movement speech |
| LOW | No qualifying relevant threat | NONE; silence |

The highest-priority relevant threat is selected, with closer distance breaking ties. Missing depth is not silently converted into a safe distance. A stationary chair remains relevant by proximity/path rules even with no finite TTC. Traffic lights and stop signs have no active driving-control role.

Left/right candidates cover fixed lower-image regions. Any overlapping detected obstacle within 5 m, or of unknown distance, blocks its side; projected footprints also count. Every tile in a 3×3 grid needs ≥80% valid depth, a 10th-percentile depth of ≥2 m, and acceptable spread. This is modest evidence of clearance, **not proof of walkability**: it does not establish floor continuity, curb/drop-off safety or a collision-free full-body route. Unstable motion, untracked threats, both sides blocked, or inadequate depth leads to STOP. The engine preserves a still-supported side; changes require persistence, with STOP during confirmation.

### Speech and evidence

One local worker initializes and pumps its speech engine in its own context, including COM initialization on Windows. There is at most one pending warning, replaced by the current higher-priority/changed decision. Warnings expire, obsolete active speech is cancelled, and repeat deduplication uses threat/class/action/severity with a default five-second cooldown. Emergency escalation bypasses ordinary delays. Clearing never generates repeated “path clear” narration.

Only supported temporal evidence yields “approaching” or “crossing from the right/left”; otherwise wording is neutral. If the engine fails, visual processing continues with Audio: unavailable. On source failure or stale live inference, movement speech is cancelled; on fully unavailable observations the pipeline stays visually UNKNOWN and does not keep issuing new movement speech.

CSV rows include frame/source time, IDs/classes, depth age/spread/freshness, closing speed, TTC validity/status, current/predicted overlap, risk, selected threat, action, side evidence and worker speech events. Unknown numerics are empty. `speech_submitted` means handed to the driver; `speech_started` is the driver's start callback; neither verifies physical audibility. Speech timestamps are local monotonic time, not video presentation time. `runtime_error` records termination failures when the CSV is open.

Recordings contain annotated processed frames at a constant FPS (source FPS or `--record-fps`). Live frame dropping or slow processing can make playback speed differ from real capture time; the CSV preserves actual measurement timestamps. MP4 audio is not embedded. OpenCV reports writer-open failures, but cannot reliably report every disk/codec write error; verify the output after recording.

## 6. Controlled demonstration checklist

Keep the camera stable, use measured distances and a sighted operator; do not demonstrate in traffic.

- [ ] Person approaches the corridor: positive closing speed, finite TTC, relevant warning.
- [ ] Person passes beside it beyond immediate-proximity distance: LOW/NONE and silence.
- [ ] Bicycle moves laterally toward the corridor: predicted overlap before current overlap, supported crossing wording.
- [ ] Stationary chair in the corridor: warning even if TTC is unknown/non-approaching.
- [ ] Block one side: that side must never be suggested. A supported other side may be offered; otherwise STOP is correct.
- [ ] Block both sides or cover candidate depth regions: STOP.
- [ ] Hold an unchanged threat: no rapid repeated narration; cooldown applies.
- [ ] Reverse approach: previous finite TTC clears.
- [ ] Disconnect camera / lose depth: UNKNOWN or clean termination; no stale movement speech.
- [ ] Repeat a short clip offline; inspect output video and CSV, actual FPS/device, and memory usage.

## 7. Verification and limitations

See [docs/VALIDATION.md](docs/VALIDATION.md) for exactly what ran. The model-free suite covers the requested 16 failure-mode categories plus integration cases. Execute it without models using `python -m unittest discover -s tests -v` (only NumPy required). The implementation environment had Python 3.12.14 / NumPy 2.3.5, not the target Windows 3.11 stack.

No actual model inference, webcam capture, Windows SAPI output, CUDA memory/FPS benchmark, OpenCV GUI rendering or real codec recording was executed here: those dependencies/hardware were unavailable and package installation failed. Their exact checks are provided above. This repository is an integrated implementation awaiting those hardware validations, not a measured real-time system.

Monocular depth can be biased; box sampling mixes surfaces; ByteTrack IDs can switch; body/camera motion can resemble object motion; rolling histories introduce delay. TTC is longitudinal under simplifying assumptions, not full collision prediction. Linear image-space projections and side depth checks are approximate. Glass, thin poles, stairs, drop-offs, untrained classes and occluded obstacles may be missed. Stock pretrained COCO names are filtered to supported people/animals, selected vehicles, chairs/benches/hydrants/suitcases; adding arbitrary names does not teach a new class.

Experimental lane and speed code remains disabled. The lane demo now has a main guard. Original traffic-light helpers and sample images remain for provenance, outside the active pipeline.

## 8. Project files and attribution

- `ego_vision/pipeline.py`: one entry point; capture/inference/reasoning/speech/CSV/GUI lifecycle.
- `config/settings.py`, `ingest/video_reader.py`, `perception/`: configuration, timing, YOLO/ByteTrack and metric depth.
- `reasoning/`: robust distance, repaired TTC, corridor, motion and risk.
- `decision/rule_engine.py`: deterministic pedestrian rules and persistence.
- `audio/warnings.py`: suppression policy and local worker.
- `viz/`: OpenCV HUD, boxes/trails and optional illustrative BEV.
- `tests/`: model-free regression and integration tests.
- [docs/PRD.md](docs/PRD.md), [docs/CHANGES.md](docs/CHANGES.md), [docs/BASELINE.md](docs/BASELINE.md), [docs/LICENSES.md](docs/LICENSES.md): scope, contribution map, source baseline and external licensing.

The upstream MIT LICENSE and Git history are preserved. Changes are left uncommitted for review; nothing was pushed or published. Generated Python caches have been removed from the delivered working tree (historical commits are untouched). Dependency/model licenses apply separately; the upstream MIT notice does not relicense Ultralytics or downloaded weights.
