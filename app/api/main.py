"""
app/api/main.py — FastAPI app. This first pass wires up ONLY `POST /detect`
(single image), to prove the Pipeline works end-to-end over HTTP before
adding video/stream/query endpoints in later passes (each ingestion route is
its own build step per the plan, not bundled into one).

WHY THE IMAGE IS DECODED WITH cv2.imdecode, NOT SAVED TO DISK FIRST
  frame_source.py's _iter_image() reads from a path via cv2.imread because
  that path handles VIDEO FILES and offline images already on disk. An
  uploaded image only exists as bytes in memory (UploadFile) — writing it to
  a temp file just to immediately re-read it would be a pointless disk
  round-trip. cv2.imdecode(np.frombuffer(bytes, uint8), IMREAD_COLOR) decodes
  the same JPEG/PNG bytes directly into the same BGR np.ndarray shape that
  every other frame source already produces, so Pipeline.process_frame()
  never has to know the frame came from an upload instead of a file.

WHY A FRESH Pipeline PER REQUEST
  A single image is a one-frame "session" with no continuity to any other
  request — a fresh Tracker (empty ByteTrack state) and a fresh
  ViolationEngine (no open incidents) are correct here, matching the
  one-instance-per-session contract both classes already document. Video and
  stream endpoints (later passes) will instead construct ONE Pipeline for
  the whole file/connection, reusing it across many process_frame() calls.
"""

from __future__ import annotations

import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.config import settings
from app.db import crud
from app.db.session import get_session
from app.detection.frame_source import Frame, SourceType, iter_frames
from app.pipeline import Pipeline
from app.schemas import (
    AnalyticsResponse,
    DetectResponse,
    StreamActionResponse,
    ViolationEvent,
    ViolationStatus,
    ViolationType,
    VideoDetectResponse,
)
from streaming.stream_worker import StreamWorker

app = FastAPI(title="PPE Compliance Monitoring")

# Registry of currently-running stream workers, keyed by camera_id. Module-
# level (not per-request) because a stream OUTLIVES the request that started
# it — /stream/start returns immediately once the thread launches; /stream/
# stop looks the worker back up by camera_id later, possibly minutes later,
# from a completely different request. The lock guards start/stop racing
# against each other for the same camera_id (e.g. two near-simultaneous
# /stream/start calls), not the worker's internals — each worker has only
# one background thread touching its own state.
_stream_workers: dict[str, StreamWorker] = {}
_stream_lock = threading.Lock()


@app.post("/detect", response_model=DetectResponse)
async def detect_image(camera_id: str, file: UploadFile = File(...)) -> DetectResponse:
    raw = await file.read()
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode uploaded file as an image")

    frame = Frame(index=0, image=image, timestamp=datetime.now(timezone.utc))
    pipeline = Pipeline(camera_id=camera_id)

    with get_session() as session:
        frame_detections, _ = pipeline.process_frame(frame, session)
        pipeline.finalize(session)

    return DetectResponse(frame=frame_detections)


@app.post("/detect/video", response_model=VideoDetectResponse)
async def detect_video(camera_id: str, file: UploadFile = File(...)) -> VideoDetectResponse:
    """Frame-sampled video processing through the SAME Pipeline as /detect —
    the only difference from a single image is that ONE Pipeline instance
    now spans many frames (so track IDs and open incidents persist across
    the whole file), and frames are sampled via config's
    sampling.video_every_n_frames instead of processing every frame (video
    files can be long; this is CPU inference, so sampling is what keeps
    processing time from scaling 1:1 with footage length).

    WHY THE UPLOAD IS WRITTEN TO A TEMP FILE, UNLIKE /detect's in-memory
    cv2.imdecode: cv2.VideoCapture (used by frame_source.py for VIDEO/STREAM
    sources) needs a filesystem path, a webcam index, or a network URL — it
    has no equivalent of imdecode for reading a container format (mp4, avi)
    out of an in-memory buffer. The temp file is deleted in a `finally`
    block regardless of success or failure, so a crash mid-video doesn't
    leak disk space.
    """
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    raw = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        pipeline = Pipeline(camera_id=camera_id)
        frames_processed = 0
        violations = []

        # iter_frames() is a generator — cv2.VideoCapture only actually opens
        # the file on the FIRST next() inside the loop below, so the
        # RuntimeError it can raise (unreadable/corrupt upload) must be
        # caught around the loop itself, not around this call.
        frame_iter = iter_frames(
            SourceType.VIDEO,
            tmp_path,
            every_n_frames=settings.config.sampling.video_every_n_frames,
        )

        try:
            with get_session() as session:
                for frame in frame_iter:
                    _, closed = pipeline.process_frame(frame, session)
                    violations.extend(closed)
                    frames_processed += 1

                violations.extend(pipeline.finalize(session))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)

    if frames_processed == 0:
        raise HTTPException(status_code=400, detail="no frames could be read from the uploaded video")

    return VideoDetectResponse(frames_processed=frames_processed, violations=violations)


@app.post("/stream/start", response_model=StreamActionResponse)
async def stream_start(camera_id: str) -> StreamActionResponse:
    """Launch a background worker that processes this camera's live feed
    through the SAME Pipeline as /detect and /detect/video, until /stream/
    stop is called. camera_id must be one of config.yaml's static `cameras`
    entries — sources are config-defined, not caller-supplied, for the same
    auditability reason zones are static config rather than inferred.
    """
    try:
        camera = settings.config.camera(camera_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown camera_id: {camera_id}")

    with _stream_lock:
        existing = _stream_workers.get(camera_id)
        if existing is not None and existing.is_alive():
            raise HTTPException(status_code=409, detail=f"stream already running for {camera_id}")

        worker = StreamWorker(camera_id=camera_id, source=camera.source)
        _stream_workers[camera_id] = worker
        worker.start()

    return StreamActionResponse(camera_id=camera_id, status="started")


@app.post("/stream/stop", response_model=StreamActionResponse)
async def stream_stop(camera_id: str) -> StreamActionResponse:
    """Signal the worker for this camera to stop, wait for it to exit, and
    finalize (force-close) any incident left open when it did. Idempotent in
    effect but not in status code: calling this with nothing running is a
    caller error (404), not a silent no-op — it usually means the caller
    lost track of stream state.
    """
    with _stream_lock:
        worker = _stream_workers.pop(camera_id, None)

    if worker is None:
        raise HTTPException(status_code=404, detail=f"no running stream for {camera_id}")

    worker.stop()
    return StreamActionResponse(camera_id=camera_id, status="stopped")


@app.get("/violations", response_model=list[ViolationEvent])
async def get_violations(
    camera_id: Optional[str] = None,
    zone_id: Optional[str] = None,
    violation_type: Optional[ViolationType] = None,
    status: Optional[ViolationStatus] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[ViolationEvent]:
    """Thin pass-through to crud.list_violations — all the filtering logic
    already lives there (Step 4), so this endpoint's only job is translating
    query params into that call and returning what it finds. start/end here
    filter on first_seen (an incident STARTED in this window) — this is a
    browsing endpoint, not an aggregate, so "did it start here" is the more
    natural question than /analytics' "did it overlap this window at all."
    """
    with get_session() as session:
        return crud.list_violations(
            session,
            camera_id=camera_id,
            zone_id=zone_id,
            violation_type=violation_type,
            status=status,
            start=start,
            end=end,
            limit=limit,
            offset=offset,
        )


def _merge_intervals(
    intervals: list[tuple[datetime, datetime]]
) -> list[tuple[datetime, datetime]]:
    """Union of possibly-overlapping (start, end) intervals into disjoint
    ones. Needed because two people can be simultaneously non-compliant in
    the same zone — summing their raw durations would double-count that
    overlapping wall-clock time as MORE than 100% non-compliant, which is
    incoherent for a time-based rate."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda pair: pair[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    camera_id: Optional[str] = None,
    zone_id: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> AnalyticsResponse:
    """compliance_rate here is TIME-based, not observation-based:

        1 - (total wall-clock time, in [start, end], during which at least
             one open violation existed) / (length of [start, end])

    WHY NOT "compliant observations / total observations" (the more
    intuitive reading of "compliance rate"): that needs a denominator of
    total person-observations (how many person-frames were seen at all,
    compliant or not), and this system only persists VIOLATIONS — compliant
    workers never generate a row, by design (Step 3/4: only PPE-absence is
    an event worth recording). Computing a true observation-based rate would
    require a NEW counter tracking total person-detections per
    camera/zone/time bucket, which nothing currently persists — a real
    schema addition, deliberately deferred (same category as the session_id
    gap already flagged in db/models.py), not silently assumed away.

    The time-based version answers a related, honestly-computable question
    instead: "of the time monitored, how much of it had zero open
    violations anywhere in scope?" It uses each incident's OWN
    duration_seconds-backing interval (first_seen, last_seen), clipped to
    the window, then merged (see _merge_intervals) so two people violating
    at once don't double-count that overlap as more than 100% non-compliant.

    start/end default to the trailing 24 hours if omitted — a compliance
    rate over an unbounded, ever-growing table has no stable meaning, so
    some window is mandatory; 24h is a readable, stated default rather than
    a silent one.
    """
    end = end or datetime.now(timezone.utc)
    start = start or (end - timedelta(hours=24))
    if start >= end:
        raise HTTPException(status_code=400, detail="start must be before end")

    with get_session() as session:
        intervals = crud.list_violation_intervals(
            session, camera_id=camera_id, zone_id=zone_id, start=start, end=end
        )
        counts = crud.count_by_violation_type(
            session, camera_id=camera_id, zone_id=zone_id, start=start, end=end
        )

    clipped = [(max(s, start), min(e, end)) for s, e in intervals]
    merged = _merge_intervals(clipped)
    violation_seconds = sum((e - s).total_seconds() for s, e in merged)
    window_seconds = (end - start).total_seconds()
    compliance_rate = max(0.0, min(1.0, 1.0 - violation_seconds / window_seconds))

    return AnalyticsResponse(
        total_violations=sum(counts.values()),
        compliance_rate=compliance_rate,
        most_common=counts,
    )
