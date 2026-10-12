# Upstream and dependency/model notices

The original `LICENSE` is preserved byte for byte from the upload: MIT, Copyright (c) 2026 Myat Thu Kyaw. Original repository: https://github.com/myatthukyaw/ego-vision . This notice describes that source; it does not relicense third-party packages or downloaded weights.

The following are source references, not a bundled license replacement. Models/packages are downloaded separately; retain their own notices when redistributing them.

| Component | License/source information |
|---|---|
| Ultralytics YOLO11 / ByteTrack integration | Ultralytics declares AGPL-3.0 and a separate Enterprise option for its software/models. https://pypi.org/project/ultralytics/8.3.203/ and https://github.com/ultralytics/ultralytics/blob/main/LICENSE |
| ByteTrack research | Original tracker attribution: https://github.com/ifzhang/ByteTrack ; active implementation is supplied by Ultralytics |
| Depth Anything V2 Small | Upstream README declares Small under Apache-2.0; Base/Large/Giant have different terms. https://github.com/DepthAnything/Depth-Anything-V2#license |
| Metric Outdoor Small conversion | Exact model/config origin: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf |
| Metric Indoor Small conversion | Optional alternative origin: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf |
| pyttsx3 2.99 | MPL-2.0 as declared at https://pypi.org/project/pyttsx3/2.99/ |
| PyTorch / Torchvision | Preserve their distributed license notices: https://github.com/pytorch/pytorch/blob/main/LICENSE and https://github.com/pytorch/vision/blob/main/LICENSE |
| Transformers | https://github.com/huggingface/transformers/blob/main/LICENSE |
| OpenCV Python | https://github.com/opencv/opencv-python/blob/master/LICENSE.txt ; wheels may include separately licensed components |
| NumPy / Pillow / LAP | https://numpy.org/doc/stable/license.html ; https://pillow.readthedocs.io/en/stable/about.html#license ; https://github.com/gatagat/lap/blob/master/LICENSE |
| Windows COM / SAPI | comtypes: https://pypi.org/project/comtypes/1.4.12/ ; pywin32: https://pypi.org/project/pywin32/311/ ; installed Microsoft voices remain subject to their distribution terms |

The dormant lane module retains its original source-reference comment. It is not used by VisionGuard's active pipeline.

No claim is made that adding VisionGuard branding or preserving an MIT file makes the entire dependency/model combination MIT-licensed.
