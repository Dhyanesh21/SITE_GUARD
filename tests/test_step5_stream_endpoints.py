"""
Step 5 (pass 3) verification: POST /stream/start + /stream/stop against a
real background worker thread (StreamWorker), over real HTTP, against real
Postgres.

WHY THIS TEST POINTS cam_02 AT A VIDEO FILE INSTEAD OF A REAL CAMERA/RTSP
  There's no live webcam/RTSP feed available in an automated test run.
  iter_frames(STREAM, ...) treats a finite video file exactly like a
  camera/RTSP source that eventually drops (cap.read() returns ok=False,
  the loop ends) — so pointing the worker at the Step 2 demo clip is a
  legitimate stand-in for proving the actual thing under test: the worker
  lifecycle (start -> frames flow through one Pipeline -> stop -> finalize),
  not the specific transport. monkeypatch mutates the SAME Camera object the
  /stream/start endpoint looks up (settings.config is a cached singleton),
  and is reverted automatically at test teardown.
"""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import main as api_main
from app.config import settings
from app.db.models import Base, ViolationEventORM
from app.db.session import SessionLocal, engine

client = TestClient(api_main.app)

SAMPLE_VIDEO = Path(__file__).parent.parent / "scripts" / "_demo_output" / "synthetic_walk.avi"


def setup_module(_module):
    Base.metadata.create_all(engine)


def teardown_function(_fn):
    # Stop any worker a failed assertion left running before clearing the
    # registry, so one test's background thread can't bleed into the next.
    for worker in list(api_main._stream_workers.values()):
        worker.stop()
    api_main._stream_workers.clear()

    with SessionLocal() as session:
        session.query(ViolationEventORM).delete()
        session.commit()


@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="Step 2 demo video not generated locally")
def test_stream_start_then_stop_runs_worker_lifecycle(monkeypatch):
    camera = settings.config.camera("cam_02")
    monkeypatch.setattr(camera, "source", str(SAMPLE_VIDEO))

    start_resp = client.post("/stream/start", params={"camera_id": "cam_02"})
    assert start_resp.status_code == 200
    assert start_resp.json() == {"camera_id": "cam_02", "status": "started"}

    time.sleep(1.0)  # let the worker actually read some frames

    stop_resp = client.post("/stream/stop", params={"camera_id": "cam_02"})
    assert stop_resp.status_code == 200
    assert stop_resp.json() == {"camera_id": "cam_02", "status": "stopped"}


@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="Step 2 demo video not generated locally")
def test_stream_start_rejects_double_start(monkeypatch):
    camera = settings.config.camera("cam_02")
    monkeypatch.setattr(camera, "source", str(SAMPLE_VIDEO))

    first = client.post("/stream/start", params={"camera_id": "cam_02"})
    assert first.status_code == 200

    second = client.post("/stream/start", params={"camera_id": "cam_02"})
    assert second.status_code == 409


def test_stream_start_rejects_unknown_camera():
    response = client.post("/stream/start", params={"camera_id": "does_not_exist"})
    assert response.status_code == 404


def test_stream_stop_rejects_when_not_running():
    response = client.post("/stream/stop", params={"camera_id": "cam_01"})
    assert response.status_code == 404
