import base64
import os
import sys
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


custom_ultralytics_path = os.getenv("CUSTOM_ULTRALYTICS_PATH")
if custom_ultralytics_path and Path(custom_ultralytics_path).is_dir():
    sys.path.insert(0, custom_ultralytics_path)

from ultralytics import YOLO  # noqa: E402


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cudnn.benchmark = True
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()


class MaturityGradingInterface:
    def __init__(self) -> None:
        self.status = "initializing"
        self.status_msg = "initializing..."
        self.model = None
        self.labels = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device == "cuda"
        self.imgsz = int(os.getenv("INFERENCE_IMGSZ", "1280"))
        self.conf = float(os.getenv("INFERENCE_CONF", "0.001"))
        self.iou = float(os.getenv("INFERENCE_IOU", "0.5"))
        self.max_det = int(os.getenv("INFERENCE_MAX_DET", "300"))

    def init(self, model_path: str) -> None:
        try:
            print(f"[Init] Loading model from {model_path} on {self.device} (half={self.half})")
            self.model = YOLO(model_path, task="detect")
            self.model.to(self.device)

            if self.half:
                self.model.model.half()

            self.labels = self.model.names
            self._warmup()
            self.status = "ready"
            self.status_msg = "normal"
        except Exception as exc:
            self.status = "abnormal"
            self.status_msg = str(exc)
            print(f"[Init][ERROR] {exc}")

    def _warmup(self) -> None:
        if self.model is None:
            return

        dummy = (np.random.rand(self.imgsz, self.imgsz, 3) * 255).astype(np.uint8)
        with torch.inference_mode():
            for _ in range(3):
                self.model.predict(
                    source=dummy,
                    save=False,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    max_det=self.max_det,
                    half=self.half,
                    device=self.device,
                    verbose=False,
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_status(self) -> str:
        return self.status

    def load_image(self, image_path: str) -> np.ndarray:
        try:
            img_pil = Image.open(image_path)

            if img_pil.width > 4000 or img_pil.height > 4000:
                scale_divisor = 4
            elif img_pil.width > 1000 or img_pil.height > 1000:
                scale_divisor = 2
            else:
                scale_divisor = 1

            if scale_divisor > 1:
                target_size = (img_pil.width // scale_divisor, img_pil.height // scale_divisor)
                img_pil.draft("RGB", target_size)

            img_np = np.asarray(img_pil)
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ValueError(f"Failed to load image: {exc}") from exc

    def process_image(self, image_path: str, args=None, **kwargs) -> dict:
        response = {"objects": [], "classifications": [], "segmentations": []}

        try:
            if self.status != "ready":
                raise RuntimeError(f"Model is not ready, current status: {self.status}")

            with Image.open(image_path) as temp_img:
                orig_w, orig_h = temp_img.size

            image = self.load_image(image_path)
            autocast_context = torch.cuda.amp.autocast(enabled=True) if self.half else nullcontext()

            with torch.inference_mode(), autocast_context:
                results = self.model.predict(
                    source=image,
                    save=False,
                    imgsz=kwargs.get("imgsz", self.imgsz),
                    conf=kwargs.get("conf", self.conf),
                    iou=kwargs.get("iou", self.iou),
                    max_det=kwargs.get("max_det", self.max_det),
                    half=self.half,
                    device=self.device,
                    verbose=False,
                )

            result = results[0] if results else None
            if result is None:
                return response

            self._append_objects(response, result, orig_w, orig_h)
            self._append_segmentations(response, result, orig_w, orig_h)
            return response
        except Exception as exc:
            print(f"[Process][ERROR] {exc}")
            return {"objects": [], "classifications": [], "segmentations": [], "error": str(exc)}

    def _append_objects(self, response: dict, result, width: int, height: int) -> None:
        if not hasattr(result, "boxes") or result.boxes is None:
            return

        boxes = result.boxes.xyxyn.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = box
            x1_abs, y1_abs = int(x1 * width), int(y1 * height)
            x2_abs, y2_abs = int(x2 * width), int(y2 * height)
            label = self.labels[int(class_id)] if self.labels else str(int(class_id))

            response["objects"].append(
                {
                    "label": label,
                    "prob": float(conf),
                    "type": 1,
                    "points": [
                        {"x": x1_abs, "y": y1_abs},
                        {"x": x2_abs, "y": y1_abs},
                        {"x": x2_abs, "y": y2_abs},
                        {"x": x1_abs, "y": y2_abs},
                    ],
                }
            )

    def _append_segmentations(self, response: dict, result, width: int, height: int) -> None:
        if not hasattr(result, "masks") or result.masks is None:
            return

        masks = result.masks.data.cpu().numpy()
        has_boxes = hasattr(result, "boxes") and result.boxes is not None
        box_confs = result.boxes.conf.cpu().numpy() if has_boxes else []
        box_xyxyn = result.boxes.xyxyn.cpu().numpy() if has_boxes else []

        for idx, mask in enumerate(masks):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            buffer = BytesIO()
            mask_img.save(buffer, format="JPEG")
            mask_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            label = self.labels[idx] if self.labels and idx < len(self.labels) else "segment"

            item = {
                "mask": mask_b64,
                "ratio": float(np.sum(mask) / (mask.shape[0] * mask.shape[1])),
                "label": label,
            }

            if idx < len(box_confs):
                item["prob"] = float(box_confs[idx])

            if idx < len(box_xyxyn):
                x1, y1, x2, y2 = box_xyxyn[idx]
                item["points"] = [
                    {"x": float(x1 * width), "y": float(y1 * height)},
                    {"x": float(x2 * width), "y": float(y1 * height)},
                    {"x": float(x2 * width), "y": float(y2 * height)},
                    {"x": float(x1 * width), "y": float(y2 * height)},
                ]

            response["segmentations"].append(item)


interface_instance = MaturityGradingInterface()


def init(model_path: str) -> None:
    interface_instance.init(model_path)


def get_status() -> str:
    return interface_instance.get_status()


def process_image(image_path: str, args=None, **kwargs) -> dict:
    return interface_instance.process_image(image_path, args, **kwargs)
