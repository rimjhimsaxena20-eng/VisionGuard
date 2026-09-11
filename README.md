# Ego Vision

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

This project performs real-time driving action prediction from car dashcam video by combining YOLOv11 for object detection, ByteTrack for multi-object tracking, and Depth Anything V2 for depth estimation. Based on the detected results, the system classifies each scene into one of four driving actions: GO, SLOW DOWN, STOP, or EMERGENCY BRAKE. The predicted action is rendered as a live overlay on the video.

---

<table>
  <tr>
    <td width="50%" align="center">
      <video src="https://github.com/user-attachments/assets/1ec120a0-c606-4ff4-9ed5-112f515052ce" controls width="100%"></video>
    </td>
    <td width="50%" align="center">
      <video src="https://github.com/user-attachments/assets/89c5f5fa-2e61-4271-b154-36a667b92bf8" controls width="100%"></video>
    </td>
  </tr>
</table>

**[Watch the full demo on YouTube](https://www.youtube.com/watch?v=mKYMzkYYfiM)**

---


## How it works

| Step | What it does | How |
|---|---|---|
| **Find objects** | Detect cars, people, traffic lights, and signs | YOLO11 (Ultralytics) |
| **Read light color** | Classify red, yellow, or green color of each traffic light | Color analysis of the light box |
| **Track objects** | Track objects | ByteTrack (Ultralytics) |
| **Measure distance** | Works out how many meters away each object is | Depth Anything V2, a depth-from-one-camera model |
| **Understand the scene** | Distance, closing speed, time to impact, and which light is ahead | Math on the tracked objects |
| **Decide** | Turns all of that into one action | A list of rules |
| **Draw** | Shows the boxes, IDs, distances, and the chosen action | OpenCV |

The objects the camera can see are sorted into three groups, and each group is handled differently:

- **Vehicles** (car, truck, bus, motorcycle, bicycle, train): used to decide whether to
  follow, slow for, or stop behind the vehicle ahead.
- **People and animals** (person, dog, cat, horse): if one is in the car's path, the car
  stops to protect it.
- **Traffic control** (traffic light, stop sign): used for the stop and go rules.

## How it decides

The rules are checked in order, and the **first one that matches wins**:

| Order | Action | When |
|---|---|---|
| 1 | **EMERGENCY BRAKE** | Something in the path ahead is very close to impact. Beats every other rule. |
| 2 | **STOP** | A person or animal is in the path, or a stop sign is ahead, or the light is red, or the car ahead is very close. |
| 3 | **SLOW DOWN (YIELD)** | The light is yellow, or a person or animal is near the path (not yet in it), or the car ahead is getting close. |
| 4 | **GO** | The way is clear, with nothing blocking and no stop signal. |

To stop the on-screen action from flickering with different actions, a new action has to repeat for n number of frames in a row before it actually changes on screen.


You can adjust the numbers in [`ego_vision/config/settings.py`](ego_vision/config/settings.py):

| Setting | Default | Meaning |
|---|---|---|
| `EMERGENCY_TTC_SEC` | `2.0` | Fewer than this many seconds to impact triggers an emergency brake |
| `LEAD_STOP_DISTANCE_M` | `10.0` | A car this close in your path makes you stop |
| `LEAD_YIELD_DISTANCE_M` | `40.0` | A car this close in your path makes you slow down |
| `HYSTERESIS_N` | `5` | How many frames an action must repeat before it changes on screen |
| `EGO_NEAR_MARGIN_PX` | `80` | How far outside the path still counts as "near" |
| `DEPTH_MAX_M` | `80.0` | Farthest distance the depth model can measure |
| `DEPTH_SCALE` | `0.35` | Per-camera calibration applied to every depth reading. Tune by eyeballing a known-distance object |
| `BEV_LATERAL_SCALE` | `1.15` | Bird's-eye-view lateral conversion factor (assumes ~60 deg dashcam FOV) |
| `BEV_MAX_RANGE_M` | `50.0` | How far ahead the bird's-eye mini-map looks |

The `EGO_ZONE_*` settings in the same file control the size and shape of the path-ahead
area shown on screen.

## Installation

```bash
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

```bash
# Watch it live in a window
python -m ego_vision.pipeline -i data/video.mp4 --show

# Save the result as a video file
python -m ego_vision.pipeline -i data/video.mp4 -o out.mp4 --show

# Don't run depth estimation on every frame to get faster FPS
python -m ego_vision.pipeline -i data/video.mp4 -o out.mp4 --show --depth-every 5
```


### Options

| Option | Default | What it does |
|---|---|---|
| `--input`, `-i` | required | The input video file |
| `--output`, `-o` | none | Save the result to this video file |
| `--show` | off | Show the result live in a window |
| `--model` | `yolo11m.pt` | Which detection model to use |
| `--conf` | `0.30` | How sure the model must be to report an object |
| `--iou` | `0.50` | How much overlapping boxes are merged |
| `--device` | `cuda:0` | Where to run: `cpu`, `0`, `cuda:0`, and so on |
| `--every` | `1` | Run detection on every Nth frame (higher is faster, rougher) |
| `--hysteresis` | `5` | Frames an action must repeat before it changes on screen |
| `--depth-every` | `1` | Measure depth every Nth frame and reuse it in between |
| `--no-ego-zone` | off | Hide the path-ahead shape |
| `--no-debug` | off | Hide the strip of details at the bottom |
| `--no-conf` | off | Hide the confidence numbers on labels |


## Project layout

```
ego_vision/
├── config/
│   └── settings.py          config
├── ingest/
│   └── video_reader.py      reads the video frame by frame
├── perception/
│   ├── detector.py          YOLO object detector
│   ├── light_classifier.py  red / yellow / green from each light box
│   ├── depth_estimator.py   distance in meters from the Depth Anything V2 model
│   └── lane_detector.py     lane detection (still being worked on)
├── reasoning/
│   ├── ego_zone.py          builds the path-ahead shape
│   ├── distance.py          reads distance from the depth result
│   ├── ttc.py               time to impact from how distance changes over time
│   └── scene_state.py       gathers everything known about the current frame
├── decision/
│   └── rule_engine.py       the rules, plus reason text
├── viz/
│   ├── overlay.py           draws boxes, IDs, and distances
│   ├── hud.py               action card, system status panel, and debug strip
│   ├── motion_trails.py     fading per-vehicle and per-person motion trails
│   └── bev_map.py           top-left bird's-eye-view mini-map
└── pipeline.py              runs all the steps in order
```

## To Do

- [x] Detect and track objects on road
- [x] Detect and classify traffic-light color
- [x] Scene reasoning and the rules with the action display
- [x] Estimate distance in meters with the depth model
- [x] Time to impact (TTC)
- [x] System status panel (top-right): live status of each subsystem and the current track count
- [x] Per-vehicle and per-person motion trails (fading polyline behind every tracked object)
- [x] Bird's-eye-view mini-map (top-left): tracked objects plotted on a top-down view with distance rings
- [ ] Integrate a lane-detection model so the path-ahead shape follows the real lane instead of a fixed guess
- [ ] Improve speed estimation in the HUD. The code is in [`ego_vision/reasoning/speed.py`](ego_vision/reasoning/speed.py) and is currently disabled. It needs more accurate distance readings first.
- [ ] Speed-limit sign detection and OCR (e.g. a small YOLO traffic-sign detector + PaddleOCR for the digits)



## Built with

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for detection and tracking
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for distance from one camera
