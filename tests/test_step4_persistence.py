"""
Step 4 verification: a REAL Postgres instance (via Docker Compose), not a
mock or SQLite stand-in — insert an event, query it back, extend it,
close it, and confirm the DB-level CHECK constraints actually reject bad
data (defense in depth beyond the Pydantic validation Step 0 already does).

Requires `docker compose up -d` to have been run first (the db service).
Table isolation: each test truncates violation_events afterward via the
`clean_db` fixture, rather than using SQLAlchemy savepoint/rollback tricks —
simpler to read and explain, at the cost of tests needing to run against a
disposable dev DB rather than one holding data you care about (true here).
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.db import crud
from app.db.models import Base, ViolationEventORM
from app.db.session import SessionLocal, engine
from app.schemas import BBox, ViolationEvent, ViolationStatus, ViolationType

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module", autouse=True)
def _create_tables():
    Base.metadata.create_all(engine)
    yield


@pytest.fixture()
def clean_db():
    yield
    with SessionLocal() as session:
        session.query(ViolationEventORM).delete()
        session.commit()


def _sample_event(**overrides) -> ViolationEvent:
    defaults = dict(
        track_id=7,
        violation_type=ViolationType.NO_HARDHAT,
        camera_id="cam_01",
        zone_id="zone_a",
        first_seen=T0,
        last_seen=T0,
        confidence=0.82,
        bbox=BBox(x1=100, y1=100, x2=200, y2=400),
    )
    defaults.update(overrides)
    return ViolationEvent(**defaults)


def test_insert_and_query_back(clean_db):
    with SessionLocal() as session:
        event = _sample_event()
        assert event.id is None

        crud.create_violation(session, event)
        session.commit()
        assert event.id is not None  # DB-assigned id written back onto the schema object

        fetched = crud.get_violation(session, event.id)
        assert fetched is not None
        assert fetched.track_id == 7
        assert fetched.violation_type == ViolationType.NO_HARDHAT
        assert fetched.camera_id == "cam_01"
        assert fetched.status == ViolationStatus.OPEN


def test_update_extends_then_closes_the_same_row(clean_db):
    with SessionLocal() as session:
        event = _sample_event()
        crud.create_violation(session, event)
        session.commit()
        original_id = event.id

    # simulate the incident continuing for 4 more seconds, then closing
    with SessionLocal() as session:
        event.last_seen = T0 + timedelta(seconds=4)
        crud.update_violation(session, event)
        session.commit()

    with SessionLocal() as session:
        event.status = ViolationStatus.CLOSED
        crud.update_violation(session, event)
        session.commit()

    with SessionLocal() as session:
        fetched = crud.get_violation(session, original_id)
        assert fetched.id == original_id  # SAME row throughout, not a new one
        assert fetched.status == ViolationStatus.CLOSED
        assert abs(fetched.duration_seconds - 4.0) < 1e-6


def test_list_violations_filters_by_camera_and_type(clean_db):
    with SessionLocal() as session:
        crud.create_violation(session, _sample_event(track_id=1, camera_id="cam_01",
                                                       violation_type=ViolationType.NO_HARDHAT))
        crud.create_violation(session, _sample_event(track_id=2, camera_id="cam_02",
                                                       violation_type=ViolationType.NO_SAFETY_VEST))
        session.commit()

        cam01_only = crud.list_violations(session, camera_id="cam_01")
        assert len(cam01_only) == 1
        assert cam01_only[0].camera_id == "cam_01"

        vest_only = crud.list_violations(session, violation_type=ViolationType.NO_SAFETY_VEST)
        assert len(vest_only) == 1
        assert vest_only[0].violation_type == ViolationType.NO_SAFETY_VEST


def test_count_by_violation_type(clean_db):
    with SessionLocal() as session:
        crud.create_violation(session, _sample_event(track_id=1, violation_type=ViolationType.NO_HARDHAT))
        crud.create_violation(session, _sample_event(track_id=2, violation_type=ViolationType.NO_HARDHAT))
        crud.create_violation(session, _sample_event(track_id=3, violation_type=ViolationType.NO_SAFETY_VEST))
        session.commit()

        counts = crud.count_by_violation_type(session)
        assert counts["NO-Hardhat"] == 2
        assert counts["NO-Safety Vest"] == 1


def test_db_check_constraint_rejects_out_of_range_confidence(clean_db):
    """Proves the CHECK constraint is real, not just documentation — a
    confidence outside [0,1] is rejected at the DATABASE level even if
    something upstream failed to validate it (defense in depth)."""
    with SessionLocal() as session:
        bad_row = ViolationEventORM(
            track_id=7, violation_type="NO-Hardhat", status="open",
            camera_id="cam_01", zone_id="zone_a",
            first_seen=T0, last_seen=T0, duration_seconds=0.0,
            confidence=1.5,  # invalid: > 1.0
            bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10,
        )
        session.add(bad_row)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
