"""
app/pipeline.py — THE single frame-processing path: frame -> track -> violate
-> persist. Every ingestion route (/detect, /detect/video, /stream/*) drives
frames through THIS class, never its own copy of this logic. That is what
makes "same code path for images, video files, and live streams" true rather
than aspirational.

WHY THIS USES Tracker, NOT Detector, EVEN FOR A SINGLE IMAGE
  app.violations.rules.ViolationEngine only considers a person detection if
  its track_id is not None (see rules.py: `d.track_id is not None`). Plain
  Detector.infer() never sets track_id — only Tracker.track() does. So using
  Detector here would silently produce ZERO violations for every /detect
  image call. Routing every source type through Tracker, even a one-frame
  "session", keeps the pipeline honest to what the violation layer actually
  needs, and keeps the code path genuinely identical across sources — the
  alternative (branch: use Detector for images, Tracker for video/stream)
  is exactly the kind of source-type branching this file exists to avoid.

WHY OPEN INCIDENTS ARE ONLY PERSISTED ONCE THEY CROSS min_incident_seconds
  ViolationEngine deliberately drops sub-min_incident_seconds "blips" as
  noise (see rules.py's _close_if_meaningful) — such an incident is never
  returned as "closed" and therefore never reaches crud.create_violation.
  But engine.open_incidents exposes EVERY currently-open incident, including
  ones still under that threshold. If this pipeline persisted every open
  incident on every frame (for near-real-time /violations visibility), a
  blip would get INSERTed into the DB before the engine ever had the chance
  to filter it out — an orphaned "open" row that never receives a closing
  UPDATE, because the engine silently drops it from its own bookkeeping once
  it times out. Gating persistence on the SAME min_incident_seconds
  threshold (applied to the live, still-growing duration) closes that gap:
  an incident only starts appearing in the DB once it's already long enough
  to be considered real, and the delay before that first write is at most
  min_incident_seconds (currently 1.0s) — a rounding error against "near
  real-time", not a violation of it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.alerting.slack import AlertManager, get_alert_manager
from app.config import settings
from app.db import crud
from app.detection.frame_source import Frame
from app.schemas import FrameDetections, ViolationEvent
from app.tracking.tracker import Tracker
from app.violations.rules import ViolationEngine


class Pipeline:
    """One instance == one processing session for one camera — mirrors the
    session semantics Tracker and ViolationEngine already document for
    themselves. A caller creates one Pipeline per image, per video file, or
    per live-stream connection; frames within that session are fed through
    process_frame() in order, and finalize() is called exactly once, at the
    end, to close out anything still open (mirrors ViolationEngine.finalize
    and Tracker's own per-session state).
    """

    def __init__(
        self,
        camera_id: str,
        tracker: Optional[Tracker] = None,
        engine: Optional[ViolationEngine] = None,
        min_incident_seconds: Optional[float] = None,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        self.camera_id = camera_id
        self.tracker = tracker or Tracker()
        self.engine = engine or ViolationEngine()
        self._min_incident_seconds = (
            min_incident_seconds
            if min_incident_seconds is not None
            else settings.config.violation.min_incident_seconds
        )
        # Unlike tracker/engine (deliberately fresh per session), the alert
        # manager defaults to a PROCESS-WIDE singleton — its cooldown state
        # must survive across separate Pipeline instances (separate uploads,
        # separate stream runs) or anti-spam cooldown would reset every time
        # a new session starts. See app/alerting/slack.py's docstring.
        self.alert_manager = alert_manager or get_alert_manager()

    def process_frame(
        self, frame: Frame, session: Session
    ) -> tuple[FrameDetections, list[ViolationEvent]]:
        """Run ONE frame through detect+track -> violation rules -> persist.
        Returns (raw FrameDetections, incidents that CLOSED on this frame) —
        the second element is what lets a multi-frame caller (e.g.
        /detect/video) accumulate a full violations list across an entire
        session; /detect (single image) is free to ignore it.
        """
        detections = self.tracker.track(frame.image)
        frame_detections = FrameDetections(
            frame_index=frame.index,
            timestamp=frame.timestamp,
            detections=detections,
            camera_id=self.camera_id,
        )

        closed = self.engine.process_frame(self.camera_id, frame.timestamp, detections)
        persistable_open = [
            e for e in self.engine.open_incidents
            if e.duration_seconds >= self._min_incident_seconds
        ]
        self._persist(persistable_open, closed, session)

        return frame_detections, closed

    def finalize(self, session: Session) -> list[ViolationEvent]:
        """Force-close whatever is still open at the end of this session
        (end of video file, or a stream that was stopped) and persist the
        closing UPDATE. Must be called exactly once per Pipeline instance,
        after the last process_frame() call — an un-finalized session leaves
        its last ongoing violation open in the DB forever. Returns the
        incidents closed by this call, same reason as process_frame's second
        return value.
        """
        closed = self.engine.finalize(now=datetime.now(timezone.utc))
        self._persist([], closed, session)
        return closed

    def _persist(
        self,
        open_incidents: list[ViolationEvent],
        closed_incidents: list[ViolationEvent],
        session: Session,
    ) -> None:
        # id is None -> never written before -> INSERT (a brand-new incident
        #   -> counts toward the alert threshold, checked once right after).
        # id is set   -> already has a row -> UPDATE (extends last_seen, or
        #                flips status to closed — same call either way, since
        #                crud.update_violation just overwrites every column).
        #                Does NOT count toward the alert threshold again —
        #                see app/alerting/slack.py's docstring for why.
        for event in open_incidents + closed_incidents:
            is_new = event.id is None
            if is_new:
                crud.create_violation(session, event)
            else:
                crud.update_violation(session, event)
            if is_new:
                self.alert_manager.notify_new_violation(
                    session, zone_id=event.zone_id, camera_id=event.camera_id, now=event.first_seen
                )
        session.commit()
