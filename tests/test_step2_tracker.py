"""
Step 2 verification: persistent track IDs survive across successive calls on
the same session, and reset() clears state between sessions.

No real video/dataset needed yet: feeding the SAME static image repeatedly
simulates a "video" where nothing moves, so IoU/motion association should
trivially keep the same track_ids call-to-call — the simplest possible proof
that persist=True is doing its job.
"""

import cv2
from ultralytics.utils import ASSETS

from app.tracking.tracker import Tracker


def _load_bus_frame():
    return cv2.imread(str(ASSETS / "bus.jpg"))


def test_track_ids_are_stable_across_consecutive_calls_same_session():
    tracker = Tracker()
    frame = _load_bus_frame()

    # Same static frame "replayed" 3x = a trivial 3-frame video where
    # nothing moves. persist=True should keep identity stable throughout.
    run1 = tracker.track(frame)
    run2 = tracker.track(frame)
    run3 = tracker.track(frame)

    ids1 = {d.track_id for d in run1 if d.track_id is not None}
    ids2 = {d.track_id for d in run2 if d.track_id is not None}
    ids3 = {d.track_id for d in run3 if d.track_id is not None}

    assert ids1, "expected at least one confirmed track on frame 1"
    # Identity persists: the SAME id set reappears on later frames of the
    # same session (this is the whole point of persist=True).
    assert ids1 == ids2 == ids3


def test_reset_clears_state_between_sessions():
    tracker = Tracker()
    frame = _load_bus_frame()

    # "Session 1": a couple of frames.
    tracker.track(frame)
    session1_ids = {d.track_id for d in tracker.track(frame) if d.track_id is not None}

    tracker.reset()

    # "Session 2": a fresh sequence on the SAME Tracker instance.
    tracker.track(frame)
    session2_ids = {d.track_id for d in tracker.track(frame) if d.track_id is not None}

    # After reset, ID assignment restarts from scratch rather than
    # continuing to climb from session 1's counter — proving no state leak.
    assert session2_ids
    assert min(session2_ids) <= min(session1_ids)
