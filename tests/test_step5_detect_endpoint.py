"""
Step 5 (pass 1) verification: POST /detect end-to-end over real HTTP
(FastAPI TestClient), through the real Pipeline, against the real Postgres
container — not mocked at any layer.

WHY THIS TEST DOESN'T ASSERT ANY ViolationEvent WAS CREATED
  As of Step 8, config.yaml points at REAL PPE-trained weights
  (weights/yolov8n_best.pt) — this image genuinely produces NO-Hardhat/
  NO-Safety Vest detections now (unlike the earlier bootstrap-COCO era,
  where the class-id mismatch made zero violations unconditionally
  correct). Whether a ViolationEvent actually gets CREATED still depends on
  cam_01/zone_a's polygon — a placeholder per config.yaml's own comment
  ("full-frame for now," sized for an assumed 640x480 frame, not bus.jpg's
  real 810x1080) — so asserting a specific violation count here would be
  coupling this test to zone geometry that hasn't been calibrated to real
  camera footage yet, not to anything this test is actually meant to prove.
  This test's job is proving the HTTP -> Pipeline -> DB wiring works end to
  end; Step 3's own tests already prove the violation LOGIC works, using
  synthetic Detections placed at precise, deliberate zone coordinates.
"""

from fastapi.testclient import TestClient
from ultralytics.utils import ASSETS

from app.api.main import app
from app.db.models import Base, ViolationEventORM
from app.db.session import SessionLocal, engine

client = TestClient(app)


def setup_module(_module):
    Base.metadata.create_all(engine)


def teardown_function(_fn):
    with SessionLocal() as session:
        session.query(ViolationEventORM).delete()
        session.commit()


def test_detect_endpoint_returns_structured_detections_for_uploaded_image():
    image_path = ASSETS / "bus.jpg"
    with open(image_path, "rb") as f:
        response = client.post(
            "/detect",
            params={"camera_id": "cam_01"},
            files={"file": ("bus.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()

    frame = body["frame"]
    assert frame["camera_id"] == "cam_01"
    assert frame["frame_index"] == 0
    assert len(frame["detections"]) > 0

    for det in frame["detections"]:
        assert 0.0 <= det["confidence"] <= 1.0
        assert det["bbox"]["x2"] > det["bbox"]["x1"]


def test_detect_endpoint_rejects_undecodable_upload():
    response = client.post(
        "/detect",
        params={"camera_id": "cam_01"},
        files={"file": ("not_an_image.txt", b"this is not image data", "text/plain")},
    )
    assert response.status_code == 400
