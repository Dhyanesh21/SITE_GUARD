"""
app/db/models.py — SQLAlchemy ORM mapping for the ONE violation schema
(app.schemas.ViolationEvent) onto a real Postgres table.

DESIGN DECISIONS (stated explicitly, not silent):

* violation_type / status are plain VARCHAR + CHECK constraints, NOT native
  Postgres ENUM types. A Postgres ENUM is nice for correctness but adding a
  new value later requires ALTER TYPE ... ADD VALUE, which historically
  can't run inside a transaction — real migration friction for a value set
  that may grow (e.g. adding NO-Mask as in-scope later). A CHECK constraint
  gives the same DB-level rejection of garbage values without that lock-in.

* bbox is stored as four flat FLOAT columns (bbox_x1..y2), not JSON/JSONB.
  The shape is fixed and small (exactly 4 numbers, always present) — a
  relational column per field is simpler to query/index than unpacking JSON,
  and there's no variability here that would justify JSONB's flexibility.

* zone_id / camera_id are plain VARCHAR, NOT foreign keys into zone/camera
  tables. Zones and cameras are STATIC, config-defined (config/config.yaml),
  not a live-editable resource — so there's no DB table to reference. This
  mirrors the plan's explicit call: zones stay in config, DB stores only
  their id as a reference string.

* Event-time vs system-time are DELIBERATELY separate column groups:
    - first_seen / last_seen  -> EVENT time, derived from the footage
      (Step 3's ViolationEngine; for a video FILE this is footage-timeline
      time, not wall-clock — see frame_source.py).
    - created_at / updated_at -> SYSTEM time, assigned by Postgres itself
      via func.now() when a row is written/modified.
  Conflating these is a classic bug: "when it happened" and "when we
  recorded it" are different facts, especially for offline video processing
  where footage-time and processing-time can diverge a lot.

* duration_seconds IS persisted here (denormalized), even though the
  Pydantic ViolationEvent computes it live from timestamps to avoid drift.
  The DB row is different: once a row's last_seen stops changing (violation
  ended), there's no drift risk, and storing the value directly avoids
  recomputing EXTRACT(EPOCH FROM last_seen - first_seen) in every analytics
  query. A deliberate, safe denormalization.

* No session_id column (yet) — track_id is ONLY unique within one
  Tracker/ViolationEngine session (Step 2/3); numbers are reused across
  separate video/stream runs. Known simplification: two different sessions
  could reuse the same track_id, and this table alone can't disambiguate
  them beyond camera_id + rough timing. A future improvement would add a
  session_id (e.g. a UUID minted once per video/stream run) — flagged
  honestly here rather than silently assumed away.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ViolationEventORM(Base):
    __tablename__ = "violation_events"

    # BigInteger, not Integer: this is an append-only audit/event table by
    # design ("audit-ready compliance reports") — the cost of 4 extra bytes
    # per row is negligible next to the cost of an INT primary key
    # overflowing on a long-lived, continuously-sampled deployment.
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    track_id = Column(Integer, nullable=False)
    violation_type = Column(String(32), nullable=False, index=True)
    status = Column(String(16), nullable=False, default="open", index=True)

    camera_id = Column(String(64), nullable=False, index=True)
    zone_id = Column(String(64), nullable=False, index=True)

    first_seen = Column(DateTime(timezone=True), nullable=False, index=True)  # event time
    last_seen = Column(DateTime(timezone=True), nullable=False)               # event time
    duration_seconds = Column(Float, nullable=False, default=0.0)             # denormalized (see docstring)

    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)

    snapshot_path = Column(String(512), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)  # system time
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )  # system time

    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_confidence_range"),
        CheckConstraint("last_seen >= first_seen", name="ck_last_seen_after_first_seen"),
        CheckConstraint("status IN ('open', 'closed')", name="ck_status_values"),
        CheckConstraint(
            "violation_type IN ('NO-Hardhat', 'NO-Safety Vest')", name="ck_violation_type_values"
        ),
    )
