"""
streaming/stream_worker.py — background thread lifecycle for one live camera
stream. This is the ONLY place threading enters the system: /detect and
/detect/video are synchronous request/response (a call comes in, frames get
processed, a response goes out); a live camera/RTSP feed has no natural end,
so it has to run on its own thread, independently of any single HTTP
request, until something explicitly stops it.

WHY A THREAD, NOT A SEPARATE PROCESS
  cv2.VideoCapture.read() and Ultralytics inference both release the GIL
  during their actual native work (OpenCV's C++ capture loop, PyTorch's
  tensor ops), so a background thread here doesn't serialize behind the GIL
  as badly as pure-Python CPU work would. A thread also shares this
  process's already-loaded YOLO weights and Postgres connection pool for
  free — a separate process would need its own copy of the model (a real
  memory/startup cost) plus an IPC channel just to report back what it saw.
  For a small number of demo camera feeds, that overhead buys nothing.

WHY A threading.Event FOR STOP, NOT A KILL
  cap.read() blocks until the next frame arrives; there is no clean way to
  interrupt it from another thread. So "stop" doesn't abort mid-read — it
  sets a flag the worker checks BETWEEN frames, and the loop exits at the
  next opportunity. Stop is therefore not instantaneous (bounded by one
  frame's read latency), which is an explicit, defensible tradeoff against
  os.kill'ing a whole process for what is, at this scale, a demo feature.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Union

from app.config import settings
from app.db.session import get_session
from app.detection.frame_source import SourceType, iter_frames
from app.pipeline import Pipeline

logger = logging.getLogger(__name__)


class StreamWorker:
    """One instance == one running camera stream. Owns its own Pipeline (own
    Tracker + ViolationEngine), mirroring the "one Pipeline per session"
    contract /detect/video already uses — here the session's lifetime is
    bounded by a thread instead of a request.
    """

    def __init__(self, camera_id: str, source: Union[str, int]) -> None:
        self.camera_id = camera_id
        self.source = source
        self.pipeline = Pipeline(camera_id=camera_id)
        self.error: Optional[str] = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"stream-{camera_id}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        # Force-close whatever incident was still open when the stream
        # stopped — same reason /detect/video calls finalize() at EOF.
        with get_session() as session:
            self.pipeline.finalize(session)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            frame_iter = iter_frames(
                SourceType.STREAM,
                self.source,
                every_n_frames=settings.config.sampling.stream_every_n_frames,
            )
            for frame in frame_iter:
                if self._stop_event.is_set():
                    break
                with get_session() as session:
                    self.pipeline.process_frame(frame, session)
        except RuntimeError as exc:
            # Can't open the source (bad RTSP URL, camera unplugged, etc.).
            # There's no HTTP request to raise this to — it happened on a
            # background thread — so it's recorded for /stream/start's
            # caller (or a future /stream/status) to observe instead.
            self.error = str(exc)
            logger.error("stream worker for camera %s failed: %s", self.camera_id, exc)
