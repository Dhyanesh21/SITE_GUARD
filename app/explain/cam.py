"""
app/explain/cam.py — Eigen-CAM heatmap: for one frame, visualize WHERE the
model's convolutional features were most active, as a sanity check on WHAT
it's actually looking at.

WHY EIGEN-CAM, NOT GRAD-CAM (the default choice for CNN classifiers)
  Grad-CAM needs the gradient of one specific output SCALAR (e.g. one
  class's logit) w.r.t. an intermediate activation map. That's natural for
  a classifier with one softmax output, but YOLO's final output has already
  gone through box decoding and NMS by the time you have "this specific
  detection" — there's no single clean differentiable scalar tied to "why
  did this exact box get drawn here" without re-deriving gradients through
  the whole detection head, which is finicky and detector-version-specific.

  Eigen-CAM sidesteps this entirely: it uses NO gradients and NO class
  targets. It takes one intermediate conv layer's activations for the whole
  image (a stack of feature maps), treats them as a matrix, and takes the
  first principal component (via SVD) as "the direction of greatest
  activation variance" — a model-agnostic, forward-pass-only proxy for what
  spatially excited that layer the most. Robust to detection post-processing,
  simple, and fast (no backward pass at all).

  HONEST LIMITATION (this matters for a safety system): Eigen-CAM is
  CLASS-AGNOSTIC. It answers "what did this layer attend to overall in this
  image," not "why was THIS SPECIFIC NO-Hardhat box flagged." It's visual
  reassurance the model is looking at people/relevant regions in general —
  not a rigorous per-detection explanation. A per-box method (Grad-CAM++,
  or EigenCAM restricted to one box's RoI) would be the next step if a real
  audit needed to defend one specific flagged violation.

WHY A SEPARATE Detector INSTANCE, NOT THE ONE Pipeline/Tracker ALREADY LOADS
  CamExplainer loads its own YOLO weights independently of the Detector
  buried inside Tracker (Step 2). This is a genuine, stated inefficiency —
  the same weights end up loaded twice in one process if both are used
  concurrently — accepted for this pass because CAM is a separate, opt-in
  diagnostic endpoint (not on the hot detect/track/violate path), and
  sharing a loaded model across layers would require threading a model
  instance through Tracker/Detector's constructors, a bigger refactor not
  justified by what this step needs. Flagged explicitly, not hidden.

WHY model.model.model[-2] (the LAST C2f BLOCK) IS THE TARGET LAYER
  Ultralytics' DetectionModel is layer index 0 (Conv) ... 22 (Detect) for
  YOLOv8n. Index -1 is the Detect head itself — not a plain conv layer, so
  CAM on its output doesn't correspond to consistent spatial activations.
  Index -2 is the last C2f block feeding the head: the deepest layer that's
  still a normal spatial conv feature map, giving the best resolution/
  semantic-depth tradeoff available without reaching into the head.

WHY THE FORWARD-OUTPUT WRAPPER
  In inference mode, Ultralytics' Detect head returns a (decoded_preds,
  raw_feature_maps) TUPLE, not a plain tensor. pytorch_grad_cam's internals
  assume the model's forward() returns one tensor (to build a default
  target for backward-gradient CAMs, even though EigenCAM itself doesn't
  need it — the check runs unconditionally). _ForwardOnlyWrapper strips the
  tuple down to just the first element so the library's assumption holds.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from app.detection.detector import Detector


class _ForwardOnlyWrapper(torch.nn.Module):
    """Collapses Ultralytics' (decoded_preds, raw_feature_maps) inference
    output down to just decoded_preds, so pytorch_grad_cam sees a plain
    tensor instead of a tuple."""

    def __init__(self, torch_model: torch.nn.Module) -> None:
        super().__init__()
        self.model = torch_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


class CamExplainer:
    def __init__(self, detector: Optional[Detector] = None, target_layer_index: int = -2) -> None:
        self.detector = detector or Detector()
        torch_model = self.detector.model.model
        torch_model.eval()

        wrapper = _ForwardOnlyWrapper(torch_model)
        target_layers = [torch_model.model[target_layer_index]]
        self._cam = EigenCAM(model=wrapper, target_layers=target_layers)

    def heatmap(self, frame: np.ndarray) -> np.ndarray:
        """frame: BGR uint8 np.ndarray (OpenCV's native layout, same as every
        other frame in this system). Returns a BGR uint8 np.ndarray at the
        SAME size as the input, with the CAM heatmap overlaid.
        """
        img_size = self.detector.imgsz
        resized = cv2.resize(frame, (img_size, img_size))
        rgb_float = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0)
        device = next(self._cam.model.parameters()).device
        tensor = tensor.to(device)

        grayscale_cam = self._cam(tensor)[0]  # HxW in [0, 1]
        overlay_rgb = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        return cv2.resize(overlay_bgr, (frame.shape[1], frame.shape[0]))
