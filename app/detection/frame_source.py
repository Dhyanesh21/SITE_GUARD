"""
app/detection/frame_source.py — ONE frame iterator abstraction for images,
video files, AND live streams (webcam/RTSP).

Why this exists (ties directly to the "same frame-processing code path for
images, video files, and live streams" non-negotiable): if pipeline.py had to
know "is this an image or a video?", that branching would leak into every
caller. Instead, every source type is reduced to the same shape — a stream of
Frame objects — so pipeline.py (Step 5) never branches on source type at all.

OpenCV detail this exploits: cv2.VideoCapture(source) accepts a file path
(video), an int (webcam index), OR an RTSP/HTTP URL (string) — identically.
That's what makes "video file" and "live stream" collapse into one code path
here; only a still IMAGE needs a different read call (cv2.imread), because
VideoCapture doesn't reliably handle single still-image formats.

FRAME TIMESTAMPING — the one place VIDEO and STREAM genuinely differ, and
deliberately so (this is NOT a violation of "same processing path"; it's
about the metadata attached to a frame, not how the frame is read or
processed downstream):
  * STREAM (webcam/RTSP): a frame arrives roughly when it's captured, so
    wall-clock time (datetime.now()) IS the frame's true timestamp.
  * VIDEO FILE: frames are read as fast as the CPU allows — a 60-second
    clip might be processed in 5 seconds. Stamping frames with
    datetime.now() would compress that 60s of real footage-time into 5s of
    wall-clock time, making every downstream violation DURATION (Step 3)
    wrong by the same factor. Video-file frames are instead timestamped
    from the footage's OWN timeline: frame_index / fps, offset from one
    anchor time captured when iteration starts. Only relative deltas matter
    for duration math — the anchor's absolute value is an arbitrary but
    explicit choice (processing start time), not a claim about when the
    footage was actually recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Union

import cv2
import numpy as np


class SourceType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    STREAM = "stream"   # webcam (int index) or RTSP/HTTP URL (str)


@dataclass
class Frame:
    """One frame, source-agnostic. Detector/tracker only ever see this."""

    index: int                     # 0-based frame index within this source
    image: np.ndarray              # BGR uint8, OpenCV's native layout
    timestamp: datetime


def _iter_image(path: Union[str, Path]) -> Iterator[Frame]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    yield Frame(index=0, image=img, timestamp=datetime.now(timezone.utc))


def _iter_capture(
    source: Union[str, int], every_n_frames: int, use_video_timeline: bool
) -> Iterator[Frame]:
    """Shared by VIDEO and STREAM — both are just a cv2.VideoCapture source.
    The READ loop is identical either way; only timestamp derivation differs
    (see module docstring). Falls back to wall-clock if fps is unavailable
    or reports 0 (some RTSP/webcam feeds don't expose it reliably) — video
    files without usable fps metadata get the STREAM behavior, since
    footage-timeline math is impossible without it.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open capture source: {source!r}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if use_video_timeline and fps and fps > 0:
        anchor = datetime.now(timezone.utc)
    else:
        use_video_timeline = False
        anchor = None

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of file (VIDEO) or dropped connection (STREAM)
            if idx % every_n_frames == 0:
                if use_video_timeline:
                    timestamp = anchor + timedelta(seconds=idx / fps)
                else:
                    timestamp = datetime.now(timezone.utc)
                yield Frame(index=idx, image=frame, timestamp=timestamp)
            idx += 1
    finally:
        cap.release()


def iter_frames(
    source_type: SourceType,
    source: Union[str, int, Path],
    every_n_frames: int = 1,
) -> Iterator[Frame]:
    """The single entry point every caller (image/video/stream endpoints) uses.

    every_n_frames is the config-driven sampling rate (Step 5 concern) — it's
    a parameter here, not a constant, so /detect/video and /stream/start can
    each pass their own configured value (sampling.video_every_n_frames vs
    sampling.stream_every_n_frames) through the SAME function.
    """
    if source_type is SourceType.IMAGE:
        yield from _iter_image(source)
    else:
        yield from _iter_capture(
            source,
            every_n_frames=every_n_frames,
            use_video_timeline=(source_type is SourceType.VIDEO),
        )
