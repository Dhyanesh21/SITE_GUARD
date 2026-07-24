"""
app/db/session.py — engine + session factory for the Postgres connection.

pool_pre_ping=True: before handing out a pooled connection, SQLAlchemy sends
a lightweight liveness check first. Without it, a connection that Postgres
(or a NAT/firewall) silently closed while idle looks fine to the pool but
fails on first real use ("server closed the connection unexpectedly") — an
easy failure mode for a long-running stream-processing service that only
touches the DB occasionally between frames.

get_session() is a context manager wrapping the standard
begin-work/commit-or-rollback/close pattern, so every caller gets consistent
transaction handling without repeating it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
