"""
app/schemas.py — THE single, canonical data model set for the whole system.

Every layer imports from here:
    detection  -> produces Detection
    tracking   -> fills Detection.track_id
    violations -> produces ViolationEvent
    db         -> persists ViolationEvent
    api        -> serializes both as JSON responses
    alerting   -> reads ViolationEvent to build a Slack message

Non-negotiable enforced here: "one consistent violation event schema (incl.
track id + duration) used across detection, DB, API, and alerts." There is
exactly ONE class to change if the shape must change.

NOTE: These are Pydantic v2 models. Some ViolationEvent fields (duration,
status) get their precise SEMANTICS nailed down in Step 3; the shape is a
first draft here so downstream layers can already import a stable name.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
class BBox(BaseModel):
    """Axis-aligned bounding box in ABSOLUTE PIXEL coordinates, xyxy order.

    Decision: xyxy absolute-pixel floats are the canonical internal format.
      * Ultralytics returns xyxy natively -> no conversion guesswork.
      * Absolute pixels survive any later resize (we convert once, at the
        detector boundary, and nothing downstream re-interprets the numbers).
    (x1, y1) = top-left, (x2, y2) = bottom-right, origin at top-left of frame.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @computed_field  # exposed in JSON, but derived — not stored redundantly
    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @computed_field
    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """Geometric centre — used by some zone-membership strategies."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Midpoint of the bottom edge ~ where the person meets the ground.

        This is the point we'll test for zone membership in Step 3: a worker's
        FEET decide which zone they're standing in, not the middle of their
        torso (which can hang over a zone boundary while they stand outside).
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)


# ---------------------------------------------------------------------------
# Detection (Step 1 output; Step 2 fills track_id)
# ---------------------------------------------------------------------------
class Detection(BaseModel):
    """One object detected in one frame.

    track_id is Optional because plain detection (Step 1) has no notion of
    identity; ByteTrack (Step 2) populates it. Keeping it on the SAME model
    (rather than a separate TrackedDetection) means the downstream pipeline
    doesn't branch on "tracked vs not" — it just checks whether track_id is set.
    """

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    track_id: Optional[int] = None


class FrameDetections(BaseModel):
    """All detections for a single frame, plus the context needed downstream.

    frame_index + timestamp let the violation layer compute DURATION and the
    persistence layer record WHEN. camera_id/zone context is attached later.
    """

    frame_index: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detections: list[Detection] = Field(default_factory=list)
    camera_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Violations (Step 3 output; Step 4 persists; Step 6 alerts on)
# ---------------------------------------------------------------------------
class ViolationType(str, Enum):
    """The specific PPE-absence that constitutes the violation.

    Values match the dataset's class NAMES so there's no translation layer
    between what the model outputs and what we store/alert on.
    """

    NO_HARDHAT = "NO-Hardhat"
    NO_SAFETY_VEST = "NO-Safety Vest"


class ViolationStatus(str, Enum):
    OPEN = "open"       # incident ongoing (worker still non-compliant in zone)
    CLOSED = "closed"   # incident ended (cleared, or track lost past timeout)


class ViolationEvent(BaseModel):
    """THE canonical violation record. One row == one continuous incident for
    one person (track_id), NOT one row per frame. This per-incident shape is
    the direct consequence of track-ID dedup (Step 3) and is what makes the
    'evidenced, non-inflated, duration-aware' claim true.

    Fields whose exact semantics are finalized in Step 3 are marked (S3).
    """

    # Identity of the incident -------------------------------------------------
    id: Optional[int] = None                 # DB-assigned; None until persisted
    track_id: int                            # which tracked person (from Step 2)
    violation_type: ViolationType            # what PPE was missing

    # Where -------------------------------------------------------------------
    camera_id: str
    zone_id: str

    # When + how long ---------------------------------------------------------
    first_seen: datetime                     # incident start
    last_seen: datetime                      # most recent frame still violating
    status: ViolationStatus = ViolationStatus.OPEN   # (S3)

    # Evidence ----------------------------------------------------------------
    confidence: float = Field(ge=0.0, le=1.0)  # confidence at trigger frame
    bbox: BBox                                  # person bbox at trigger frame
    snapshot_path: Optional[str] = None         # saved evidence image (S3/S4)

    @computed_field
    @property
    def duration_seconds(self) -> float:
        """Live-derived incident length. Computed from timestamps rather than
        stored as a mutable counter so it can never drift out of sync with
        first_seen/last_seen. (S3 may also persist a frozen copy at close.)"""
        return max(0.0, (self.last_seen - self.first_seen).total_seconds())


# ---------------------------------------------------------------------------
# API response envelopes (fleshed out in Step 5; declared here for one home)
# ---------------------------------------------------------------------------
class DetectResponse(BaseModel):
    frame: FrameDetections


class AnalyticsResponse(BaseModel):
    total_violations: int
    compliance_rate: float                        # 0..1
    most_common: dict[str, int] = Field(default_factory=dict)
