"""
app/db/crud.py — insert/update/query operations for violation_events.

Two write paths, matching how Step 3's ViolationEngine models an incident's
lifecycle (open -> extended -> closed), NOT a single "insert once when done"
path:

  create_violation(): INSERT a brand-new row when an incident first opens.
    Mutates the given ViolationEvent with its DB-assigned id, so the caller
    (Step 5's pipeline) can hold that id and use it to update the SAME row
    as the incident continues or closes — mirroring a ticketing system
    (open a ticket, update it, close it), not a fire-and-forget log line.

  update_violation(): UPDATE an existing row by id — used both to extend
    last_seen/duration while an incident is still open, and again to flip
    status to "closed". This is what makes an OPEN, still-ongoing violation
    visible via /violations before it ever ends — important for the "near
    real-time intervention" business goal. (A closed-only write model would
    make a worker who stays non-compliant for the whole session invisible
    until the session ends.)

list_violations() is the query foundation for Step 5's GET /violations.
count_by_violation_type() is a PREVIEW aggregate for Step 5's /analytics —
"compliance rate" needs a denominator (total person-observations) this
table doesn't persist, so that specific metric is deliberately left as an
open question for Step 5, not decided here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ViolationEventORM
from app.schemas import BBox, ViolationEvent, ViolationStatus, ViolationType


def _to_orm_fields(event: ViolationEvent) -> dict:
    return dict(
        track_id=event.track_id,
        violation_type=event.violation_type.value,
        status=event.status.value,
        camera_id=event.camera_id,
        zone_id=event.zone_id,
        first_seen=event.first_seen,
        last_seen=event.last_seen,
        duration_seconds=event.duration_seconds,
        confidence=event.confidence,
        bbox_x1=event.bbox.x1,
        bbox_y1=event.bbox.y1,
        bbox_x2=event.bbox.x2,
        bbox_y2=event.bbox.y2,
        snapshot_path=event.snapshot_path,
    )


def _to_schema(row: ViolationEventORM) -> ViolationEvent:
    return ViolationEvent(
        id=row.id,
        track_id=row.track_id,
        violation_type=ViolationType(row.violation_type),
        status=ViolationStatus(row.status),
        camera_id=row.camera_id,
        zone_id=row.zone_id,
        first_seen=row.first_seen,
        last_seen=row.last_seen,
        confidence=row.confidence,
        bbox=BBox(x1=row.bbox_x1, y1=row.bbox_y1, x2=row.bbox_x2, y2=row.bbox_y2),
        snapshot_path=row.snapshot_path,
    )


def create_violation(session: Session, event: ViolationEvent) -> ViolationEvent:
    row = ViolationEventORM(**_to_orm_fields(event))
    session.add(row)
    session.flush()  # assigns row.id without requiring a commit yet
    event.id = row.id
    return event


def update_violation(session: Session, event: ViolationEvent) -> ViolationEvent:
    if event.id is None:
        raise ValueError("update_violation requires event.id — row not yet inserted")
    row = session.get(ViolationEventORM, event.id)
    if row is None:
        raise LookupError(f"no violation_events row with id={event.id}")
    for key, value in _to_orm_fields(event).items():
        setattr(row, key, value)
    return event


def get_violation(session: Session, violation_id: int) -> Optional[ViolationEvent]:
    row = session.get(ViolationEventORM, violation_id)
    return _to_schema(row) if row else None


def list_violations(
    session: Session,
    *,
    camera_id: Optional[str] = None,
    zone_id: Optional[str] = None,
    violation_type: Optional[ViolationType] = None,
    status: Optional[ViolationStatus] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[ViolationEvent]:
    """Filtered query — foundation for Step 5's GET /violations. All filters
    are optional and AND-ed together; unset filters are simply omitted from
    the WHERE clause rather than compared against a sentinel."""
    stmt = select(ViolationEventORM)
    if camera_id is not None:
        stmt = stmt.where(ViolationEventORM.camera_id == camera_id)
    if zone_id is not None:
        stmt = stmt.where(ViolationEventORM.zone_id == zone_id)
    if violation_type is not None:
        stmt = stmt.where(ViolationEventORM.violation_type == violation_type.value)
    if status is not None:
        stmt = stmt.where(ViolationEventORM.status == status.value)
    if start is not None:
        stmt = stmt.where(ViolationEventORM.first_seen >= start)
    if end is not None:
        stmt = stmt.where(ViolationEventORM.first_seen <= end)

    stmt = stmt.order_by(ViolationEventORM.first_seen.desc()).limit(limit).offset(offset)
    rows = session.execute(stmt).scalars().all()
    return [_to_schema(r) for r in rows]


def count_by_violation_type(
    session: Session,
    *,
    camera_id: Optional[str] = None,
    zone_id: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> dict[str, int]:
    """Counts, optionally scoped the same way list_violations() is scoped —
    used unfiltered by early Step 4 tests, and filtered/windowed by Step 5's
    /analytics. start/end here mean OVERLAPS the window (last_seen >= start
    AND first_seen <= end), not "started inside the window" — an incident
    that began before `start` and was still open when the window opened is
    still relevant to that window's analytics."""
    stmt = select(ViolationEventORM.violation_type, func.count())
    if camera_id is not None:
        stmt = stmt.where(ViolationEventORM.camera_id == camera_id)
    if zone_id is not None:
        stmt = stmt.where(ViolationEventORM.zone_id == zone_id)
    if start is not None:
        stmt = stmt.where(ViolationEventORM.last_seen >= start)
    if end is not None:
        stmt = stmt.where(ViolationEventORM.first_seen <= end)
    stmt = stmt.group_by(ViolationEventORM.violation_type)
    return {vtype: count for vtype, count in session.execute(stmt).all()}


def count_recent_violations(session: Session, *, zone_id: str, since: datetime) -> int:
    """Count of DISTINCT incidents (rows) in this zone that OPENED at or
    after `since` — used by app.alerting.slack.AlertManager to evaluate the
    "N violations in T seconds" threshold rule. Counts rows (incidents),
    not updates — first_seen is set once at INSERT and never changes."""
    stmt = (
        select(func.count())
        .select_from(ViolationEventORM)
        .where(ViolationEventORM.zone_id == zone_id, ViolationEventORM.first_seen >= since)
    )
    return session.execute(stmt).scalar_one()


def list_violation_intervals(
    session: Session,
    *,
    camera_id: Optional[str] = None,
    zone_id: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[tuple[datetime, datetime]]:
    """(first_seen, last_seen) pairs only, UNLIMITED (no pagination) — used
    by /analytics to compute compliance_rate, which needs EVERY matching
    incident's interval to merge correctly, not a paginated page of them
    like list_violations() returns for browsing. Same overlap semantics as
    count_by_violation_type()."""
    stmt = select(ViolationEventORM.first_seen, ViolationEventORM.last_seen)
    if camera_id is not None:
        stmt = stmt.where(ViolationEventORM.camera_id == camera_id)
    if zone_id is not None:
        stmt = stmt.where(ViolationEventORM.zone_id == zone_id)
    if start is not None:
        stmt = stmt.where(ViolationEventORM.last_seen >= start)
    if end is not None:
        stmt = stmt.where(ViolationEventORM.first_seen <= end)
    return [(fs, ls) for fs, ls in session.execute(stmt).all()]
