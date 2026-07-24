"""
Step 5 (pass 2) verification: POST /detect/video end-to-end — real HTTP,
real Pipeline, real Postgres, real (synthetic) video file. Reuses the
walking-person clip generated for Step 2's tracking demo
(scripts/_demo_output/synthetic_walk.avi) so we don't need real footage.

WHY THIS TEST DOESN'T ASSERT ANY ViolationEvent WAS CREATED
  As of Step 8, config.yaml points at REAL PPE-trained weights, not the
  earlier bootstrap COCO ones — but this video's frames are a SYNTHETIC
  composite (a real person crop pasted onto a plain background, built for
  Step 2's tracking demo), a scene very unlike the real construction-site
  photos this model was trained on. Whether it fires any PPE-absence
  detections on such an out-of-domain composite isn't something to assert
  either way. This test proves multi-frame sampling + one Pipeline session
  spanning the whole file + finalize() all work together over real HTTP;
  Step 3's own tests already prove the violation LOGIC works, using
  synthetic Detections placed at precise, deliberate zone coordinates.
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
