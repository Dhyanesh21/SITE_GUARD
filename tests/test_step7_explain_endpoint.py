"""
Step 7 verification: POST /explain end-to-end over real HTTP, through the
real Eigen-CAM path (no mocking of the CAM/torch layer — this is exactly the
part most likely to break silently, e.g. from an Ultralytics version bump
changing the Detect head's output shape).

No Postgres/DB involvement here — /explain never touches the DB, so
these tests skip the usual setup_module/teardown_function pattern.
"""

import cv2
import numpy as np
from fastapi.testclient import TestClient
from ultralytics.utils import ASSETS

from app.api.main import app

client = TestClient(app)


def test_explain_endpoint_returns_a_same_size_jpeg_heatmap():
    image_path = ASSETS / "bus.jpg"
    original = cv2.imread(str(image_path))

    with open(image_path, "rb") as f:
        response = client.post("/explain", files={"file": ("bus.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

    decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == original.shape  # heatmap is resized back to the input's own size


def test_explain_endpoint_rejects_undecodable_upload():
    response = client.post(
        "/explain",
        files={"file": ("not_an_image.txt", b"this is not image data", "text/plain")},
    )
    assert response.status_code == 400
