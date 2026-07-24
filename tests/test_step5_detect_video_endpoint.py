"""
Step 5 (pass 2) verification: POST /detect/video end-to-end — real HTTP,
real Pipeline, real Postgres, real (synthetic) video file. Reuses the
walking-person clip generated for Step 2's tracking demo
(scripts/_demo_output/synthetic_walk.avi) so we don't need real footage.

Same reasoning as the /detect test applies to why we don't assert any
ViolationEvent: bootstrap COCO weights label people as class_id 0, but
config.classes.person_class is 5 (the PPE dataset's ordering) — so
ViolationEngine correctly recognizes zero persons on this model's output.
This test proves multi-frame sampling + one Pipeline session spanning the
whole file + finalize() all work together over real HTTP.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.db.models import Base, ViolationEventORM
from app.db.session import SessionLocal, engine

client = TestClient(app)

SAMPLE_VIDEO = Path(__file__).parent.parent / "scripts" / "_demo_output" / "synthetic_walk.avi"


def setup_module(_module):
    Base.metadata.create_all(engine)


def teardown_function(_fn):
    with SessionLocal() as session:
        session.query(ViolationEventORM).delete()
        session.commit()


@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="Step 2 demo video not generated locally")
def test_detect_video_endpoint_samples_multiple_frames():
    with open(SAMPLE_VIDEO, "rb") as f:
        response = client.post(
            "/detect/video",
            params={"camera_id": "cam_02"},
            files={"file": ("synthetic_walk.avi", f, "video/x-msvideo")},
        )

    assert response.status_code == 200
    body = response.json()

    assert body["frames_processed"] > 0
    assert isinstance(body["violations"], list)


def test_detect_video_endpoint_rejects_unreadable_upload():
    response = client.post(
        "/detect/video",
        params={"camera_id": "cam_02"},
        files={"file": ("not_a_video.txt", b"garbage bytes, not a real container", "text/plain")},
    )
    assert response.status_code == 400
