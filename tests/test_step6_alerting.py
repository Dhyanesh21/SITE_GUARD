"""
Step 6 verification: threshold+cooldown alert LOGIC, then its wiring into
Pipeline's persist path. requests.post is monkeypatched everywhere here —
no real webhook is hit by this automated suite, since firing a real Slack
message on every `pytest` run would spam your channel. The one-off PROOF
that a real message lands in Slack is a manual script
(scripts/live_fire_alert.py), run once by hand, not part of this file.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.alerting.slack import AlertManager
from app.db import crud
from app.db.models import Base, ViolationEventORM
from app.db.session import SessionLocal, engine
from app.pipeline import Pipeline
from app.schemas import BBox, Detection, ViolationEvent, ViolationType

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def setup_module(_module):
    Base.metadata.create_all(engine)


def teardown_function(_fn):
    with SessionLocal() as session:
        session.query(ViolationEventORM).delete()
        session.commit()


class _FakeResponse:
    def raise_for_status(self) -> None:
        pass


def _capture_posts(monkeypatch, sent: list) -> None:
    def fake_post(url, json, timeout):
        sent.append((url, json))
        return _FakeResponse()

    monkeypatch.setattr("app.alerting.slack.requests.post", fake_post)


def _seed_closed_violations(session, *, zone_id: str, camera_id: str, count: int, first_seen: datetime) -> None:
    for i in range(count):
        event = ViolationEvent(
            track_id=i,
            violation_type=ViolationType.NO_HARDHAT,
            camera_id=camera_id,
            zone_id=zone_id,
            first_seen=first_seen,
            last_seen=first_seen,
            status="closed",
            confidence=0.9,
            bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        )
        crud.create_violation(session, event)
    session.commit()


# ---------------------------------------------------------------------------
# AlertManager threshold/cooldown logic, in isolation
# ---------------------------------------------------------------------------
def test_alert_fires_once_threshold_crossed(monkeypatch):
    sent: list = []
    _capture_posts(monkeypatch, sent)

    manager = AlertManager(
        webhook_url="https://example.invalid/webhook",
        enabled=True, threshold=3, window_seconds=60, cooldown_seconds=300,
    )

    with SessionLocal() as session:
        _seed_closed_violations(session, zone_id="zone_a", camera_id="cam_01", count=2, first_seen=BASE_TIME)
        assert manager.notify_new_violation(session, zone_id="zone_a", camera_id="cam_01", now=BASE_TIME) is False
        assert sent == []  # only 2 rows exist -> below threshold of 3

        _seed_closed_violations(session, zone_id="zone_a", camera_id="cam_01", count=1, first_seen=BASE_TIME)
        assert manager.notify_new_violation(session, zone_id="zone_a", camera_id="cam_01", now=BASE_TIME) is True
        assert len(sent) == 1
        assert "zone_a" in sent[0][1]["text"]


def test_alert_respects_cooldown(monkeypatch):
    sent: list = []
    _capture_posts(monkeypatch, sent)

    manager = AlertManager(
        webhook_url="https://example.invalid/webhook",
        enabled=True, threshold=1, window_seconds=60, cooldown_seconds=300,
    )

    with SessionLocal() as session:
        _seed_closed_violations(session, zone_id="zone_b", camera_id="cam_01", count=1, first_seen=BASE_TIME)
        assert manager.notify_new_violation(session, zone_id="zone_b", camera_id="cam_01", now=BASE_TIME) is True

        later = BASE_TIME + timedelta(seconds=10)  # well inside the 300s cooldown
        _seed_closed_violations(session, zone_id="zone_b", camera_id="cam_01", count=1, first_seen=later)
        assert manager.notify_new_violation(session, zone_id="zone_b", camera_id="cam_01", now=later) is False

    assert len(sent) == 1  # no second Slack message while cooling down


def test_alert_noops_when_webhook_url_missing():
    # "" (not None) forces "no webhook configured" for this test even though
    # a real SLACK_WEBHOOK_URL is set in .env — passing None here would fall
    # through to that real value (see AlertManager.__init__: None means "use
    # the configured default", the same Optional-override pattern Pipeline
    # uses for tracker/engine).
    manager = AlertManager(webhook_url="", enabled=True, threshold=1, window_seconds=60, cooldown_seconds=300)
    with SessionLocal() as session:
        _seed_closed_violations(session, zone_id="zone_c", camera_id="cam_01", count=5, first_seen=BASE_TIME)
        assert manager.notify_new_violation(session, zone_id="zone_c", camera_id="cam_01", now=BASE_TIME) is False


def test_alert_noops_when_disabled(monkeypatch):
    def fail_post(*args, **kwargs):
        pytest.fail("requests.post should never be called while alert.enabled is False")

    monkeypatch.setattr("app.alerting.slack.requests.post", fail_post)
    manager = AlertManager(
        webhook_url="https://example.invalid/webhook",
        enabled=False, threshold=1, window_seconds=60, cooldown_seconds=300,
    )
    with SessionLocal() as session:
        _seed_closed_violations(session, zone_id="zone_d", camera_id="cam_01", count=5, first_seen=BASE_TIME)
        assert manager.notify_new_violation(session, zone_id="zone_d", camera_id="cam_01", now=BASE_TIME) is False


# ---------------------------------------------------------------------------
# Pipeline integration: real detect->track->violate->persist path triggers
# the alert exactly when the DB-backed threshold is actually crossed.
# ---------------------------------------------------------------------------
def _person(track_id: int, x1=100, y1=100, x2=200, y2=400) -> Detection:
    return Detection(
        class_id=5, class_name="Person", confidence=0.9,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), track_id=track_id,
    )


def _no_hardhat(x1=110, y1=100, x2=190, y2=160) -> Detection:
    return Detection(class_id=2, class_name="NO-Hardhat", confidence=0.8, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))


class _StubTracker:
    """Pipeline normally routes through a real Tracker (YOLO inference); this
    test is about the persist->alert wiring, not detection/tracking, so it
    stubs track() to return pre-built synthetic detections instead — same
    reasoning test_step3_violation_rules.py gives for synthetic Detections.
    """

    def __init__(self, frames: list[list[Detection]]) -> None:
        self._frames = iter(frames)

    def track(self, _image) -> list[Detection]:
        return next(self._frames)


def test_pipeline_fires_alert_when_three_distinct_incidents_open_in_one_zone(monkeypatch):
    sent: list = []
    _capture_posts(monkeypatch, sent)

    manager = AlertManager(
        webhook_url="https://example.invalid/webhook",
        enabled=True, threshold=3, window_seconds=60, cooldown_seconds=300,
    )

    # Three different people (track_ids 7, 8, 9), all NO-Hardhat, all in
    # cam_01/zone_a, present for 1.5s (>= min_incident_seconds=1.0) so they
    # actually get persisted rather than dropped as sub-threshold blips.
    frame_t0 = [_person(7), _no_hardhat(), _person(8, x1=210, x2=310), _no_hardhat(x1=220, x2=300),
                _person(9, x1=320, x2=420), _no_hardhat(x1=330, x2=410)]
    tracker = _StubTracker(frames=[frame_t0, frame_t0])

    from app.detection.frame_source import Frame

    pipeline = Pipeline(camera_id="cam_01", tracker=tracker, alert_manager=manager)

    with SessionLocal() as session:
        pipeline.process_frame(Frame(index=0, image=None, timestamp=BASE_TIME), session)
        pipeline.process_frame(
            Frame(index=1, image=None, timestamp=BASE_TIME + timedelta(seconds=1.5)), session
        )

    assert len(sent) == 1  # exactly one alert, once the 3rd distinct incident opened
    assert "zone_a" in sent[0][1]["text"]
    assert "3" in sent[0][1]["text"]
