"""
Step 5 (pass 1) verification: POST /detect end-to-end over real HTTP
(FastAPI TestClient), through the real Pipeline, against the real Postgres
container — not mocked at any layer.

WHY THIS TEST DOESN'T ASSERT ANY ViolationEvent WAS CREATED
  We're still running the bootstrap COCO weights (yolov8n.pt), not
  PPE-trained ones (Step 8's output). COCO's class_id 0 is "person"; our
  config's person_class is 5 (the PPE dataset's ordering). ViolationEngine
  only recognizes a detection as a "person" when class_id == person_class,
  so on COCO output it never does — zero violations is the CORRECT result
  here, not a test gap. This test proves the HTTP -> Pipeline -> DB wiring
  works; Step 3's own tests already prove the violation LOGIC works, using
  synthetic Detections that stand in for PPE-trained output.
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
