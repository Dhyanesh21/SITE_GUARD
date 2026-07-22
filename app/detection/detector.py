"""
app/detection/detector.py — thin wrapper around a YOLOv8 model that turns one
raw frame into a validated List[Detection] (the shared schema).

Everything downstream (tracking, violation rules, persistence, alerts, CAM)
consumes ONLY this output shape. This file's whole job is to be the single
place where "raw model output" gets converted into "trustworthy structured
data" — confidence and NMS thresholds applied, classes named, boxes typed.

--------------------------------------------------------------------------
DECISION (surfaced explicitly, not silent): class names come from the
LOADED MODEL's own `names` dict (result.names), not from
config.classes.names.

Why: config.classes.names describes the PPE-DATASET label set (Hardhat,
NO-Hardhat, Safety Vest, ...) — it is only correct once we load PPE-trained
weights (Step 8's output). Right now, per the plan's flagged bootstrap
decision, model_path points at public yolov8n.pt trained on COCO, whose
`names` are COCO categories (id 0 = "person", etc — NOT id 5 = "Person").

If the detector hardcoded config.classes.names, every detection from the
COCO bootstrap model would be mislabeled (e.g. class_id 0 stamped as
"Hardhat" instead of "person"). Making the detector read the model's own
native names keeps it CORRECT for whichever weights are loaded, and makes
the swap to PPE weights in Step 8 a config change only — zero code change.

The consequence: config.classes.* (person_class, ppe_absence_classes) is
NOT used by the detector. It's consumed later, by the violation layer
(Step 3), which only runs meaningfully once PPE-trained weights are in
place. That's an intentional, explicit boundary, not an oversight.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import settings  # noqa: F401  (side effect: sets YOLO_AUTOINSTALL before ultralytics import below)
from ultralytics import YOLO

from app.schemas import BBox, Detection


class Detector:
    """Loads one YOLOv8 model and runs inference frame-by-frame.

    All thresholds default from config/config.yaml (config-driven, per the
    non-negotiable) but can be overridden per-instance for tests.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
    ) -> None:
        cfg = settings.config.detection
        self.model_path = model_path or cfg.model_path
        self.conf_threshold = conf_threshold if conf_threshold is not None else cfg.conf_threshold
        self.iou_threshold = iou_threshold if iou_threshold is not None else cfg.iou_threshold
        self.device = device or cfg.device
        self.imgsz = imgsz or cfg.imgsz

        # Ultralytics auto-downloads recognized official weights (e.g.
        # "yolov8n.pt") to this path if the file isn't present yet.
        self.model = YOLO(self.model_path)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on ONE frame (BGR np.ndarray). Returns validated
        Detection objects — confidence/NMS already applied by Ultralytics
        using OUR config thresholds, not its own defaults.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        result = results[0]
        names = result.names  # native to the loaded weights (see module docstring)

        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=names[class_id],
                    confidence=confidence,
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return detections
