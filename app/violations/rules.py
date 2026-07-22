"""
app/violations/rules.py — turns tracked-person + PPE-absence detections into
ViolationEvent incidents: zone-scoped, deduplicated by track_id, with a real
open->closed lifecycle and duration.

WHAT THIS LAYER OWNS
  Given one frame's detections (already carrying track_id from Step 2), for
  each (camera, track_id, violation_type) combination that is BOTH (a) inside
  a config-defined zone and (b) currently missing required PPE, this engine
  maintains exactly ONE open ViolationEvent — extending its `last_seen` (and
  therefore its derived duration_seconds) every frame the condition persists,
  and CLOSING it once the condition hasn't been observed for
  `incident_timeout_seconds`. This is the mechanism that turns "N sampled
  frames of the same non-compliant worker" into "one incident with a
  duration", not N separate violations.

INHERITED LIMITATION (stated plainly, not hidden): this engine trusts
track_id as ground truth identity. If the upstream tracker suffers an ID
SWITCH (see the empirical caveat in app/tracking/tracker.py — proven with
real numbers via scripts/demo_step2_tracking_video.py: severe occlusion of
the detector's informative region can flip a person onto a new track_id),
this layer has no way to know the new track_id is "the same person" as the
old one. The result: one continuous real-world violation gets recorded as
TWO incidents. This is a genuine, known boundary of identity-based dedup —
not something Step 3 can independently fix; a full fix would require
appearance-based re-identification (e.g. BoT-SORT with a ReID embedding),
which is a real but heavier alternative to ByteTrack.

WHY ZONE MEMBERSHIP USES bbox.bottom_center, NOT THE BOX CENTER
  A standing person's bottom-center approximates where their FEET meet the
  ground — i.e. which zone they're actually standing in. The box's geometric
  center is roughly torso height, which can be well inside a zone while the
  person's feet (and therefore the person) are actually just outside its
  boundary, or vice versa on a boundary the camera views at an angle.

WHY PPE-ABSENCE BOXES ARE ASSOCIATED TO A PERSON BY OVERLAP RATIO, NOT
NEAREST-CENTER DISTANCE
  A NO-Hardhat box is small and sits near the top of a much larger person
  box. "Which person is this closest to?" can be fooled by two people
  standing near each other; "what fraction of the small PPE box's area
  falls inside this person's box?" directly tests containment, which is
  what a head/torso detection being PART OF a specific person actually
  means. A detection is attributed to the person with the highest overlap
  ratio, and only if that ratio clears `ppe_person_min_overlap` — otherwise
  it's dropped as unattributable rather than guessed at.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import NamedTuple, Optional

import cv2
import numpy as np

from app.config import AppConfig, settings
from app.schemas import BBox, Detection, ViolationEvent, ViolationStatus, ViolationType


class IncidentKey(NamedTuple):
    """Identity of ONE ongoing incident: a specific person, in a specific
    camera's frame, missing a specific piece of PPE. Two different missing
    items (no hardhat AND no vest) on the same person are TWO incidents,
    tracked independently — they can start/end at different times.
    """

    camera_id: str
    track_id: int
    violation_type: ViolationType


class ViolationEngine:
    """Stateful, per-run engine — one instance covers one continuous
    monitoring session (mirrors Tracker's session model from Step 2). Call
    process_frame() once per processed frame, in order; call finalize() once
    the session ends so no incident is left silently open forever.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._cfg = config or settings.config
        self._person_class = self._cfg.classes.person_class
        self._ppe_absence = self._cfg.classes.ppe_absence_classes  # class_id -> name
        self._incident_timeout = timedelta(seconds=self._cfg.violation.incident_timeout_seconds)
        self._min_incident = timedelta(seconds=self._cfg.violation.min_incident_seconds)
        self._min_overlap = self._cfg.violation.ppe_person_min_overlap

        self._open: dict[IncidentKey, ViolationEvent] = {}
        self._zone_polygons: dict[str, np.ndarray] = {
            z.id: np.array(z.polygon, dtype=np.float32) for z in self._cfg.zones
        }

    # -- public API -----------------------------------------------------------

    def process_frame(
        self, camera_id: str, timestamp: datetime, detections: list[Detection]
    ) -> list[ViolationEvent]:
        """Update incident state for ONE frame. Returns incidents that just
        CLOSED as a result of this call (empty most frames).

        IMPORTANT CALLER INVARIANT: call this every sampled frame, even when
        `detections` is empty. The timeout sweep below only runs when this
        method is called — an engine that's never called again (e.g. because
        the caller skips empty frames) will leave incidents open forever.
        """
        persons = [
            d for d in detections
            if d.class_id == self._person_class and d.track_id is not None
        ]
        ppe_absence = [d for d in detections if d.class_id in self._ppe_absence]
        assignment = self._associate_ppe_to_persons(persons, ppe_absence)

        for person in persons:
            zone_id = self._zone_for_point(camera_id, person.bbox.bottom_center)
            if zone_id is None:
                continue  # outside any defined zone -> not enforced here

            for ppe in assignment.get(person.track_id, []):
                violation_type = ViolationType(self._ppe_absence[ppe.class_id])
                key = IncidentKey(camera_id, person.track_id, violation_type)

                if key in self._open:
                    incident = self._open[key]
                    incident.last_seen = timestamp
                    incident.confidence = ppe.confidence
                    incident.bbox = person.bbox
                else:
                    self._open[key] = ViolationEvent(
                        track_id=person.track_id,
                        violation_type=violation_type,
                        camera_id=camera_id,
                        zone_id=zone_id,
                        first_seen=timestamp,
                        last_seen=timestamp,
                        confidence=ppe.confidence,
                        bbox=person.bbox,
                    )

        return self._sweep_timeouts(now=timestamp)

    def finalize(self, now: datetime) -> list[ViolationEvent]:
        """Force-close every still-open incident. Call once at the end of a
        video file or when a live stream stops — otherwise the last ongoing
        violation before the feed ends is never closed/persisted.
        """
        closed = []
        for key in list(self._open.keys()):
            incident = self._open.pop(key)
            closed.extend(self._close_if_meaningful(incident))
        return closed

    @property
    def open_incidents(self) -> list[ViolationEvent]:
        """Currently-ongoing violations — useful for a live 'right now' view
        (e.g. a future dashboard), as distinct from the closed incidents this
        engine hands off for persistence.
        """
        return list(self._open.values())

    # -- internals --------------------------------------------------------------

    def _sweep_timeouts(self, now: datetime) -> list[ViolationEvent]:
        closed: list[ViolationEvent] = []
        for key in list(self._open.keys()):
            incident = self._open[key]
            if (now - incident.last_seen) >= self._incident_timeout:
                del self._open[key]
                closed.extend(self._close_if_meaningful(incident))
        return closed

    def _close_if_meaningful(self, incident: ViolationEvent) -> list[ViolationEvent]:
        """min_incident_seconds filters out sub-threshold blips (e.g. a
        single spurious frame) — DESIGN CHOICE: such incidents are silently
        dropped, not persisted at all, since a fraction-of-a-second "incident"
        isn't meaningful evidence of a real violation and would just add
        noise to audit reports.
        """
        if incident.duration_seconds >= self._min_incident.total_seconds():
            incident.status = ViolationStatus.CLOSED
            return [incident]
        return []

    def _zone_for_point(self, camera_id: str, point: tuple[float, float]) -> Optional[str]:
        for zone in self._cfg.zones_for_camera(camera_id):
            polygon = self._zone_polygons[zone.id]
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                return zone.id
        return None

    def _associate_ppe_to_persons(
        self, persons: list[Detection], ppe_boxes: list[Detection]
    ) -> dict[int, list[Detection]]:
        assignment: dict[int, list[Detection]] = defaultdict(list)
        for ppe in ppe_boxes:
            best_track_id: Optional[int] = None
            best_ratio = 0.0
            for person in persons:
                ratio = self._overlap_ratio(small=ppe.bbox, big=person.bbox)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_track_id = person.track_id
            if best_track_id is not None and best_ratio >= self._min_overlap:
                assignment[best_track_id].append(ppe)
        return assignment

    @staticmethod
    def _overlap_ratio(small: BBox, big: BBox) -> float:
        """Fraction of `small`'s AREA that lies inside `big`. Not IoU — IoU
        would be dominated by the huge size difference between a small PPE
        box and a full-body person box and stay low even for a perfect
        containment. This asks the right question: "how much of the PPE box
        is inside this person?", not "how similar in size/position are they?"
        """
        ix1, iy1 = max(small.x1, big.x1), max(small.y1, big.y1)
        ix2, iy2 = min(small.x2, big.x2), min(small.y2, big.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(1e-9, (small.x2 - small.x1) * (small.y2 - small.y1))
        return inter / area
