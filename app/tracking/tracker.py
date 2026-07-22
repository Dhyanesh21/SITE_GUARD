"""
app/tracking/tracker.py — persistent-identity detection via Ultralytics'
built-in ByteTrack integration (model.track()).

WHY THIS EXISTS (this is the hinge of the whole system, per the plan):
  Detection (Step 1) tells you WHAT is in a frame. Tracking tells you WHICH
  physical worker that is, frame after frame. Without a stable identity,
  frame-sampled violation logic would count "no hardhat" as a NEW event on
  every sampled frame the same person appears in — hugely inflating counts.
  With a persistent track_id, Step 3 can instead ask "is THIS track_id still
  in violation?" and accumulate ONE incident with a real duration
  (last_seen - first_seen), instead of a frame tally.

WHY BYTETRACK, NOT A SIMPLE IoU-GREEDY TRACKER:
  A naive tracker matches this frame's boxes to last frame's boxes purely by
  IoU overlap, using only HIGH-confidence detections. The instant a worker is
  partially occluded (walks behind machinery, turns side-on) their detection
  confidence dips below the matching threshold and a naive tracker drops the
  track — the NEXT detection gets a brand-new ID — so Step 3 sees the SAME
  ongoing violation as two separate incidents (an "ID switch"). ByteTrack's
  two-stage association ALSO matches low-confidence boxes against
  still-unmatched, motion-predicted (Kalman-filtered) tracks, which is
  meaningfully more robust than the naive approach.

  EMPIRICAL CAVEAT (measured, not assumed — see scripts/demo_step2_tracking_
  video.py): this robustness is NOT a guarantee. In a synthetic test where a
  tracked person was occluded across the HEAD/TORSO (the informative region,
  not just the legs) for ~8 consecutive frames, the detected box shrank
  enough that its IoU with the Kalman-predicted (pre-occlusion, full-height)
  box fell too low to match — ByteTrack dropped the original track and
  spawned a NEW track_id for the occluded shape, i.e. a real ID switch. It
  happened to re-acquire the original track_id once the occlusion ended and
  the box reverted to full height (better IoU match against the still-alive,
  not-yet-expired original track) — but that reacquisition is incidental to
  IoU matching, not a designed guarantee, and would not necessarily happen in
  a busier scene. A milder occlusion (legs only, ~45% of width, same test
  harness) produced zero switches and barely dented confidence — because IoU
  match quality depends on box-SHAPE stability, not confidence alone.
  Takeaway: ByteTrack meaningfully raises the bar over naive IoU tracking,
  but severe occlusion of the detector's most informative region can still
  fragment one real incident into two track_ids — Step 3's incident logic
  must tolerate short track gaps/switches rather than assume identity is
  perfectly continuous.

DECISION (surfaced, not silent): we track ALL classes, not just "Person".
  PPE-absence classes (NO-Hardhat, NO-Safety Vest) still need to be detected
  every frame — Step 3 needs their boxes to judge a person's compliance.
  ByteTrack will assign them track IDs too, as a side effect of one combined
  call, but those IDs are UNUSED by design: only a Person detection's
  track_id is ever treated as a worker identity downstream. Filtering to
  classes=[person_class] here would silently discard the PPE signals Step 3
  depends on.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import settings  # noqa: F401  (side effect: sets YOLO_AUTOINSTALL before ultralytics import below)
from ultralytics import YOLO

from app.schemas import BBox, Detection


class Tracker:
    """Stateful wrapper: one instance == one continuous tracking SESSION
    (one video file, or one live stream connection). See reset() for why
    session boundaries matter and must be handled explicitly.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        tracker_cfg: Optional[str] = None,
        persist: Optional[bool] = None,
    ) -> None:
        det_cfg = settings.config.detection
        trk_cfg = settings.config.tracking
        self.model_path = model_path or det_cfg.model_path
        self.conf_threshold = conf_threshold if conf_threshold is not None else det_cfg.conf_threshold
        self.iou_threshold = iou_threshold if iou_threshold is not None else det_cfg.iou_threshold
        self.device = device or det_cfg.device
        self.imgsz = imgsz or det_cfg.imgsz
        self.tracker_cfg = tracker_cfg or trk_cfg.tracker
        self.persist = trk_cfg.persist if persist is None else persist

        self.model = YOLO(self.model_path)

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Run detection+tracking on ONE frame of an ONGOING session.

        persist=True tells Ultralytics to carry its internal ByteTrack state
        (Kalman filters, next-ID counter, unmatched-track buffer) over from
        the PREVIOUS call on THIS SAME model instance. This is what makes IDs
        persistent ACROSS frames — call this once per frame, in order, for
        the same video/stream. Calling it out of order, or on frames from
        two different sources without reset() between them, corrupts identity.
        """
        results = self.model.track(
            source=frame,
            persist=self.persist,
            tracker=self.tracker_cfg,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        result = results[0]
        names = result.names

        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            # box.id is None when ByteTrack hasn't confirmed this detection
            # into an established track yet (e.g. a brand-new, unmatched
            # object on its first frame or two).
            track_id = int(box.id[0]) if box.id is not None else None

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=names[class_id],
                    confidence=confidence,
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    track_id=track_id,
                )
            )
        return detections

    def reset(self) -> None:
        """Clear ByteTrack's internal state so a NEW session (a different
        video file, a new stream reconnect) doesn't inherit track IDs or
        motion history from whatever was tracked before on this instance.

        Why this matters, concretely: persist=True is a per-CALL flag, not a
        per-SESSION boundary. If you keep calling track() with persist=True
        across two unrelated videos on the same Tracker object without
        resetting first, ByteTrack will try to match the new video's first
        frame against stale tracks left over from the previous video's last
        frame — a real identity-leak bug that shows up as impossible track
        IDs (e.g. a video's first-ever person appearing as "track 47").
        """
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor else None
        if trackers:
            for t in trackers:
                t.reset()
