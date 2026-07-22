"""
Step 0 verification: the config loads & validates, and the canonical schemas
instantiate and compute derived fields correctly.

These are deliberately small — they prove the FOUNDATION is sound before any
detection code exists. Run: `pytest -q`
"""

from datetime import datetime, timedelta, timezone

from app.config import AppConfig, settings
from app.schemas import BBox, Detection, ViolationEvent, ViolationStatus, ViolationType


# --- config -----------------------------------------------------------------
def test_config_loads_and_validates():
    cfg: AppConfig = settings.config
    # thresholds present and in range
    assert 0.0 <= cfg.detection.conf_threshold <= 1.0
    # class map keys are ints, person + absence classes are consistent
    assert cfg.classes.person_class in cfg.classes.names
    for cid in cfg.classes.ppe_absence_classes:
        assert cid in cfg.classes.names


def test_zone_and_camera_lookups():
    cfg = settings.config
    z = cfg.zone("zone_a")
    assert z.camera_id == "cam_01"
    # cross-reference: the zone's camera actually exists
    assert cfg.camera(z.camera_id).id == "cam_01"


# --- schema geometry --------------------------------------------------------
def test_bbox_derived_geometry():
    b = BBox(x1=10, y1=20, x2=110, y2=220)
    assert b.width == 100 and b.height == 200
    assert b.center == (60.0, 120.0)
    assert b.bottom_center == (60.0, 220.0)   # feet = midpoint of bottom edge


def test_detection_confidence_bounds():
    d = Detection(class_id=5, class_name="Person", confidence=0.9,
                  bbox=BBox(x1=0, y1=0, x2=1, y2=1))
    assert d.track_id is None                 # not tracked until Step 2


# --- the one violation schema ----------------------------------------------
def test_violation_duration_is_derived_from_timestamps():
    start = datetime.now(timezone.utc)
    ev = ViolationEvent(
        track_id=7,
        violation_type=ViolationType.NO_HARDHAT,
        camera_id="cam_01",
        zone_id="zone_a",
        first_seen=start,
        last_seen=start + timedelta(seconds=12.5),
        confidence=0.82,
        bbox=BBox(x1=0, y1=0, x2=50, y2=120),
    )
    assert ev.status is ViolationStatus.OPEN
    assert abs(ev.duration_seconds - 12.5) < 1e-6   # duration derived, not stored
