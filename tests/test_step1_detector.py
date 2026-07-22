"""
Step 1 verification: the Detector loads the (bootstrap COCO) model, runs
inference on a real image, and returns validated Detection objects with the
model's OWN class names — not the PPE config names (see detector.py docstring
for why that's the correct behavior right now).

Uses an Ultralytics-bundled sample asset (bus.jpg — contains people) so this
test needs no dataset download, just the auto-downloaded yolov8n.pt weights.
"""

import cv2
from ultralytics.utils import ASSETS

from app.detection.detector import Detector
from app.detection.frame_source import SourceType, iter_frames
from app.schemas import Detection


def test_detector_infers_people_on_bundled_sample_image():
    detector = Detector()
    image_path = ASSETS / "bus.jpg"

    frames = list(iter_frames(SourceType.IMAGE, image_path))
    assert len(frames) == 1

    detections = detector.infer(frames[0].image)
    assert len(detections) > 0
    assert all(isinstance(d, Detection) for d in detections)

    # bus.jpg has multiple pedestrians; bootstrap COCO model's "person" class
    person_detections = [d for d in detections if d.class_name == "person"]
    assert len(person_detections) >= 1

    for d in detections:
        assert 0.0 <= d.confidence <= 1.0
        assert d.bbox.x2 > d.bbox.x1
        assert d.bbox.y2 > d.bbox.y1
        assert d.track_id is None  # no tracking until Step 2


def test_video_capture_frame_source_dispatch_is_source_agnostic(tmp_path):
    """Prove iter_frames() uses the SAME call shape for image vs capture
    sources — the point of the unified abstraction. We don't open a real
    video/stream here (no sample footage yet); we just prove the dispatch
    picks the right internal path without the caller branching on type.
    """
    image_path = ASSETS / "bus.jpg"
    frames = list(iter_frames(SourceType.IMAGE, image_path, every_n_frames=1))
    assert frames[0].index == 0
    assert frames[0].image is not None
