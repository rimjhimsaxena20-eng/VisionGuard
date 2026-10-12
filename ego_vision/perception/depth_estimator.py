"""Metric monocular depth via one Depth Anything V2 Small model."""
from ego_vision.config.settings import DEPTH_MODEL


def resolve_device(requested="auto"):
    import torch
    requested = str(requested).lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.isdecimal():
        requested = "cuda:" + requested
    if requested == "cuda":
        requested = "cuda:0"
    if requested == "cpu":
        return requested
    if requested.startswith("cuda:") and requested[5:].isdecimal():
        if not torch.cuda.is_available() or int(requested[5:]) >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {requested} is unavailable; use --device cpu or install CUDA Torch")
        return requested
    raise ValueError("Device must be auto, cpu, cuda:0 or a CUDA index")


class DepthEstimator:
    def __init__(self, model=DEPTH_MODEL, device="cpu", offline=False):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model, local_files_only=offline)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model, local_files_only=offline, use_safetensors=True).to(self.device).eval()

    def estimate_bgr(self, frame_bgr):
        from PIL import Image
        torch = self.torch
        h, w = frame_bgr.shape[:2]
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            depth = self.model(**inputs).predicted_depth
            if depth.ndim == 3:
                depth = depth.unsqueeze(1)
            depth = torch.nn.functional.interpolate(depth, size=(h, w), mode="bicubic", align_corners=False)
        return depth.squeeze().float().cpu().numpy()
