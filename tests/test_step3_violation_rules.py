"""
Step 3 verification: zone-scoped violation logic, track-ID dedup, duration,
and the honest boundaries of identity-based dedup (ID switches).

Uses SYNTHETIC Detection objects, not real inference. Reason: the bootstrap
COCO weights (yolov8n.pt) have no PPE classes at all — Step 3's logic is
untestable against real detector output until Step 8 trains PPE-aware
weights. Since this layer is pure logic over the shared Detection schema, a
hand-built sequence of Detections is a legitimate, deterministic way to
verify it precisely (and lets us test edge cases, like ID switches, that
would be hard to reproduce reliably from a real model).

Uses the REAL config/config.yaml (via app.config.settings) rather than a
mock config, so these tests double as an integration check of the actual
zone/class wiring: cam_01/zone_a is a full-frame zone (0,0)-(640,480);
cam_02/zone_b is (100,100)-(500,400); person_class=5; NO-Hardhat=2.
"""

from datetime import datetime, timedelta, timezone

from app.schemas import BBox, Detection, ViolationStatus, ViolationType
from app.violations.rules import ViolationEngine

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _person(track_id: int, x1=100, y1=100, x2=200, y2=400) -> Detection:
    return Detection(
        class_id=5, class_name="Person", confidence=0.9,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), track_id=track_id,
    )


def _no_hardhat(x1=110, y1=100, x2=190, y2=160, confidence=0.8) -> Detection:
    # sits within the top of the person box above -> should associate cleanly
    return Detection(
        class_id=2, class_name="NO-Hardhat", confidence=confidence,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def test_continuous_violation_becomes_one_incident_with_duration():
    engine = ViolationEngine()
    for i in range(5):  # t = 0,1,2,3,4 seconds
        engine.process_frame("cam_01", T0 + timedelta(seconds=i), [_person(7), _no_hardhat()])
        if i < 4:
            assert len(engine.open_incidents) == 1  # still open mid-clip

    closed = engine.finalize(now=T0 + timedelta(seconds=4))
    assert len(closed) == 1
    incident = closed[0]
    assert incident.track_id == 7
    assert incident.violation_type == ViolationType.NO_HARDHAT
    assert incident.status == ViolationStatus.CLOSED
    assert abs(incident.duration_seconds - 4.0) < 1e-6
    assert engine.open_incidents == []  # finalize left nothing dangling


def test_same_track_id_gap_within_timeout_merges_into_one_incident():
    engine = ViolationEngine()
    engine.process_frame("cam_01", T0, [_person(7), _no_hardhat()])
    # a 2-second gap with NO detections at all (e.g. momentary miss) — well
    # under the configured 10s incident_timeout_seconds
    engine.process_frame("cam_01", T0 + timedelta(seconds=2), [])
    engine.process_frame("cam_01", T0 + timedelta(seconds=4), [_person(7), _no_hardhat()])

    closed = engine.finalize(now=T0 + timedelta(seconds=4))
    assert len(closed) == 1
    assert abs(closed[0].duration_seconds - 4.0) < 1e-6  # one continuous incident


def test_id_switch_produces_two_separate_incidents():
    """Honest limitation test: dedup trusts track_id. A switch mid-violation
    (as demonstrated empirically in Step 2's occlusion demo) is NOT merged —
    it becomes two incidents, because the engine has no way to know track 8
    is "the same person" as track 7.
    """
    engine = ViolationEngine()
    engine.process_frame("cam_01", T0, [_person(7), _no_hardhat()])
    engine.process_frame("cam_01", T0 + timedelta(seconds=1), [_person(7), _no_hardhat()])
    engine.process_frame("cam_01", T0 + timedelta(seconds=2), [])  # occlusion, ID lost
    engine.process_frame("cam_01", T0 + timedelta(seconds=3), [_person(8), _no_hardhat()])
    engine.process_frame("cam_01", T0 + timedelta(seconds=4), [_person(8), _no_hardhat()])

    closed = engine.finalize(now=T0 + timedelta(seconds=4))
    assert len(closed) == 2
    track_ids = {c.track_id for c in closed}
    assert track_ids == {7, 8}


def test_zone_scoping_ignores_violations_outside_any_defined_zone():
    engine = ViolationEngine()
    # cam_02's only zone (zone_b) spans (100,100)-(500,400); place the person
    # entirely outside it — bottom_center will be (25, 50).
    outside_person = _person(1, x1=0, y1=0, x2=50, y2=50)
    outside_hardhat = _no_hardhat(x1=5, y1=0, x2=45, y2=20)

    engine.process_frame("cam_02", T0, [outside_person, outside_hardhat])
    assert engine.open_incidents == []
    closed = engine.finalize(now=T0)
    assert closed == []

    # positive control: same camera, person actually inside zone_b
    inside_person = _person(2, x1=150, y1=150, x2=250, y2=380)
    inside_hardhat = _no_hardhat(x1=160, y1=150, x2=230, y2=210)
    engine.process_frame("cam_02", T0, [inside_person, inside_hardhat])
    assert len(engine.open_incidents) == 1


def test_min_incident_seconds_filters_out_sub_threshold_blips():
    engine = ViolationEngine()
    engine.process_frame("cam_01", T0, [_person(7), _no_hardhat()])
    # immediate finalize -> duration is 0s, below configured min_incident_seconds (1.0)
    closed = engine.finalize(now=T0)
    assert closed == []  # silently dropped as noise, not a real incident


def test_unassociated_ppe_detection_is_dropped_not_guessed():
    engine = ViolationEngine()
    person = _person(7, x1=100, y1=100, x2=200, y2=400)
    # far away, zero overlap with the person's box -> can't attribute it
    far_hardhat = _no_hardhat(x1=550, y1=10, x2=600, y2=60)

    engine.process_frame("cam_01", T0, [person, far_hardhat])
    assert engine.open_incidents == []
    closed = engine.finalize(now=T0 + timedelta(seconds=5))
    assert closed == []
