"""
Step 5 (pass 4) verification: GET /violations and GET /analytics — real HTTP,
real Postgres, seeded via crud.create_violation/update_violation directly
(no need to run detection to get rows into the table; Step 4's own tests
already prove the persistence layer itself).

The /analytics tests are what actually exercise the compliance_rate design
decision: a time-based rate (1 - non-compliant wall-clock time / window),
with overlapping incidents merged so simultaneous violations don't count as
more than 100% non-compliant.
"""

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.api.main import app
from app.db import crud
from app.db.models import Base, ViolationEventORM
from app.db.session import SessionLocal, engine
from app.schemas import BBox, ViolationEvent, ViolationType

client = TestClient(app)

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def setup_module(_module):
    Base.metadata.create_all(engine)


def teardown_function(_fn):
    with SessionLocal() as session:
        session.query(ViolationEventORM).delete()
        session.commit()


def _make_event(
    track_id: int,
    violation_type: ViolationType,
    camera_id: str,
    zone_id: str,
    first_seen: datetime,
    last_seen: datetime,
) -> ViolationEvent:
    return ViolationEvent(
        track_id=track_id,
        violation_type=violation_type,
        camera_id=camera_id,
        zone_id=zone_id,
        first_seen=first_seen,
        last_seen=last_seen,
        status="closed",
        confidence=0.9,
        bbox=BBox(x1=0, y1=0, x2=10, y2=10),
    )


def test_get_violations_filters_by_camera_and_returns_json():
    with SessionLocal() as session:
        crud.create_violation(
            session,
            _make_event(1, ViolationType.NO_HARDHAT, "cam_01", "zone_a", BASE_TIME, BASE_TIME + timedelta(seconds=5)),
        )
        crud.create_violation(
            session,
            _make_event(2, ViolationType.NO_SAFETY_VEST, "cam_02", "zone_b", BASE_TIME, BASE_TIME + timedelta(seconds=5)),
        )
        session.commit()

    response = client.get("/violations", params={"camera_id": "cam_01"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["camera_id"] == "cam_01"
    assert body[0]["violation_type"] == "NO-Hardhat"


def test_get_analytics_over_non_overlapping_violations():
    window_start = BASE_TIME
    window_end = BASE_TIME + timedelta(seconds=100)

    with SessionLocal() as session:
        # Two SEQUENTIAL (non-overlapping) 10s violations inside a 100s window
        # -> 20s non-compliant, 80s compliant -> compliance_rate == 0.8.
        crud.create_violation(
            session,
            _make_event(
                1, ViolationType.NO_HARDHAT, "cam_01", "zone_a",
                window_start, window_start + timedelta(seconds=10),
            ),
        )
        crud.create_violation(
            session,
            _make_event(
                2, ViolationType.NO_HARDHAT, "cam_01", "zone_a",
                window_start + timedelta(seconds=20), window_start + timedelta(seconds=30),
            ),
        )
        session.commit()

    response = client.get(
        "/analytics",
        params={
            "camera_id": "cam_01",
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_violations"] == 2
    assert body["compliance_rate"] == 0.8
    assert body["most_common"] == {"NO-Hardhat": 2}


def test_get_analytics_merges_overlapping_violations_not_double_counted():
    window_start = BASE_TIME
    window_end = BASE_TIME + timedelta(seconds=100)

    with SessionLocal() as session:
        # Two people violating SIMULTANEOUSLY for the same 10s span. Merged,
        # this is still only 10s of non-compliant wall-clock time, not 20s.
        crud.create_violation(
            session,
            _make_event(
                1, ViolationType.NO_HARDHAT, "cam_01", "zone_a",
                window_start, window_start + timedelta(seconds=10),
            ),
        )
        crud.create_violation(
            session,
            _make_event(
                2, ViolationType.NO_SAFETY_VEST, "cam_01", "zone_a",
                window_start, window_start + timedelta(seconds=10),
            ),
        )
        session.commit()

    response = client.get(
        "/analytics",
        params={
            "camera_id": "cam_01",
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_violations"] == 2
    assert body["compliance_rate"] == 0.9  # 10s non-compliant / 100s, not 20s


def test_get_analytics_rejects_start_after_end():
    response = client.get(
        "/analytics",
        params={"start": "2026-01-02T00:00:00Z", "end": "2026-01-01T00:00:00Z"},
    )
    assert response.status_code == 400


def test_get_analytics_defaults_to_trailing_24h_when_no_window_given():
    response = client.get("/analytics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_violations"] == 0
    assert body["compliance_rate"] == 1.0
