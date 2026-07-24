"""
ui/streamlit_app.py — Step 9: a thin client over the FastAPI service.

WHY "THIN" IS THE WHOLE DESIGN, NOT JUST A STYLE CHOICE
  This file contains ZERO business logic — no detection, no violation
  rules, no persistence, nothing that duplicates what app/ already does. It
  only builds HTTP requests and renders HTTP responses. That's deliberate:
  Streamlit and FastAPI are separate processes with no shared Python
  imports between them here — the ONLY thing connecting them is the HTTP
  API surface built in Steps 5-7. If this file imported app.pipeline or
  app.db.crud directly, it would be a second, parallel way to do the same
  work as the API, which is exactly the kind of duplicate code path the
  "one processing path" non-negotiable (app/pipeline.py) exists to prevent.
  A real production UI would talk to the exact same API a mobile app or a
  third-party integration would — this file proves that boundary is real by
  only ever crossing it over HTTP, the same way any other client would.

WHY NO CLIENT-SIDE "IS THIS STREAM RUNNING" STATE
  It would be easy to track "camera X's stream is running" in
  st.session_state after a successful /stream/start call. That's rejected:
  server-side state (main.py's _stream_workers registry) is the only source
  of truth, and Streamlit re-runs this whole script top-to-bottom on every
  interaction — any client-side cache would need constant re-syncing and
  could silently drift from server reality (e.g. if the stream died on its
  own from a bad camera source). Instead, every button click just calls the
  endpoint and displays whatever it says — success, or the 409/404 error
  the API already defines for "already running" / "not running". One
  source of truth, no sync bugs possible.

WHY API_BASE_URL IS A SIDEBAR INPUT, NOT HARDCODED
  Same instinct as config-driven thresholds elsewhere in this project: the
  API might run on a different host/port depending on how it's deployed
  (local uvicorn, Docker Compose service name, etc.) — hardcoding
  "http://localhost:8000" would silently break the moment that's untrue.
"""

from __future__ import annotations

import io

import requests
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="PPE Compliance Monitoring", layout="wide")

with st.sidebar:
    st.header("API connection")
    api_base_url = st.text_input("API base URL", value="http://localhost:8000").rstrip("/")
    st.caption("Start the API separately: `uvicorn app.api.main:app --reload`")


def api_url(path: str) -> str:
    return f"{api_base_url}{path}"


def draw_detections(image_bytes: bytes, detections: list[dict]) -> Image.Image:
    """Draws bboxes client-side from the API's own coordinates — no new
    endpoint needed, /detect already returns everything required. Violation
    classes (name starts with "NO-") are drawn in red to make them visually
    distinct from compliant-equipment/other classes (green) — the same
    absence-is-red framing the rest of this system is built around."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)

    for d in detections:
        bbox = d["bbox"]
        color = "red" if d["class_name"].startswith("NO-") else "lime"
        draw.rectangle(
            [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], outline=color, width=3
        )
        label = f"{d['class_name']} {d['confidence']:.2f}"
        text_y = max(0, bbox["y1"] - 14)
        draw.rectangle([bbox["x1"], text_y, bbox["x1"] + 8 * len(label), text_y + 14], fill=color)
        draw.text((bbox["x1"] + 2, text_y), label, fill="black")

    return image


st.title("PPE Compliance Monitoring")

tab_detect, tab_video, tab_stream, tab_violations, tab_analytics, tab_explain = st.tabs(
    ["Detect Image", "Detect Video", "Live Stream", "Violations", "Analytics", "Explain (CAM)"]
)


# ---------------------------------------------------------------------------
# Detect Image — POST /detect
# ---------------------------------------------------------------------------
with tab_detect:
    st.subheader("Upload one image")
    camera_id = st.text_input("Camera ID", value="cam_01", key="detect_camera_id")
    uploaded_image = st.file_uploader("Image", type=["jpg", "jpeg", "png"], key="detect_image_file")

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded image", width=400)

    if st.button("Run detection", key="detect_button", disabled=uploaded_image is None):
        image_bytes = uploaded_image.getvalue()
        files = {"file": (uploaded_image.name, image_bytes)}
        response = requests.post(api_url("/detect"), params={"camera_id": camera_id}, files=files)

        if response.status_code == 200:
            body = response.json()
            detections = body["frame"]["detections"]
            st.success(f"{len(detections)} detection(s)")
            if detections:
                st.image(
                    draw_detections(image_bytes, detections),
                    caption="Detections (red = PPE-absence violation, green = other)",
                    width=500,
                )
                st.dataframe(
                    [
                        {
                            "class": d["class_name"],
                            "confidence": round(d["confidence"], 3),
                            "bbox": f"({d['bbox']['x1']:.0f}, {d['bbox']['y1']:.0f}) - "
                                    f"({d['bbox']['x2']:.0f}, {d['bbox']['y2']:.0f})",
                        }
                        for d in detections
                    ]
                )
        else:
            st.error(f"{response.status_code}: {response.json().get('detail', response.text)}")


# ---------------------------------------------------------------------------
# Detect Video — POST /detect/video
# ---------------------------------------------------------------------------
with tab_video:
    st.subheader("Upload one video file")
    video_camera_id = st.text_input("Camera ID", value="cam_02", key="video_camera_id")
    uploaded_video = st.file_uploader("Video", type=["mp4", "avi", "mov"], key="detect_video_file")

    if st.button("Run detection on video", key="detect_video_button", disabled=uploaded_video is None):
        files = {"file": (uploaded_video.name, uploaded_video.getvalue())}
        with st.spinner("Processing video — this runs real CPU inference frame by frame, may take a while..."):
            response = requests.post(
                api_url("/detect/video"), params={"camera_id": video_camera_id}, files=files
            )

        if response.status_code == 200:
            body = response.json()
            st.success(f"Processed {body['frames_processed']} sampled frame(s)")
            violations = body["violations"]
            st.metric("Violations", len(violations))
            if violations:
                st.dataframe(
                    [
                        {
                            "track_id": v["track_id"],
                            "type": v["violation_type"],
                            "zone": v["zone_id"],
                            "duration_s": round(v["duration_seconds"], 1),
                            "status": v["status"],
                        }
                        for v in violations
                    ]
                )
        else:
            st.error(f"{response.status_code}: {response.json().get('detail', response.text)}")


# ---------------------------------------------------------------------------
# Live Stream — POST /stream/start, POST /stream/stop
# ---------------------------------------------------------------------------
with tab_stream:
    st.subheader("Start/stop a live camera/RTSP stream")
    st.caption(
        "camera_id must match one of config.yaml's static `cameras` entries — "
        "sources are config-defined, not typed here, for the same auditability "
        "reason zones are static config."
    )
    stream_camera_id = st.text_input("Camera ID", value="cam_01", key="stream_camera_id")

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start stream", key="stream_start_button"):
            response = requests.post(api_url("/stream/start"), params={"camera_id": stream_camera_id})
            if response.status_code == 200:
                st.success(response.json())
            else:
                st.error(f"{response.status_code}: {response.json().get('detail', response.text)}")

    with col_stop:
        if st.button("Stop stream", key="stream_stop_button"):
            response = requests.post(api_url("/stream/stop"), params={"camera_id": stream_camera_id})
            if response.status_code == 200:
                st.success(response.json())
            else:
                st.error(f"{response.status_code}: {response.json().get('detail', response.text)}")


# ---------------------------------------------------------------------------
# Violations — GET /violations
# ---------------------------------------------------------------------------
with tab_violations:
    st.subheader("Browse persisted violations")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_camera_id = st.text_input("Camera ID (optional)", value="", key="viol_camera_id")
    with col2:
        filter_zone_id = st.text_input("Zone ID (optional)", value="", key="viol_zone_id")
    with col3:
        filter_status = st.selectbox("Status", ["(any)", "open", "closed"], key="viol_status")

    limit = st.slider("Max results", min_value=1, max_value=500, value=100, key="viol_limit")

    if st.button("Load violations", key="viol_load_button"):
        params = {"limit": limit}
        if filter_camera_id:
            params["camera_id"] = filter_camera_id
        if filter_zone_id:
            params["zone_id"] = filter_zone_id
        if filter_status != "(any)":
            params["status"] = filter_status

        response = requests.get(api_url("/violations"), params=params)
        if response.status_code == 200:
            rows = response.json()
            st.success(f"{len(rows)} violation(s)")
            if rows:
                st.dataframe(
                    [
                        {
                            "id": r["id"],
                            "track_id": r["track_id"],
                            "type": r["violation_type"],
                            "camera": r["camera_id"],
                            "zone": r["zone_id"],
                            "status": r["status"],
                            "duration_s": round(r["duration_seconds"], 1),
                            "first_seen": r["first_seen"],
                        }
                        for r in rows
                    ]
                )
        else:
            st.error(f"{response.status_code}: {response.json().get('detail', response.text)}")


# ---------------------------------------------------------------------------
# Analytics — GET /analytics
# ---------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Compliance analytics")
    st.caption(
        "compliance_rate is TIME-based (fraction of the window with zero open "
        "violations), not observation-based — see app/api/main.py's get_analytics "
        "docstring for why. Leave the window blank for the default trailing 24h."
    )

    col1, col2 = st.columns(2)
    with col1:
        analytics_camera_id = st.text_input("Camera ID (optional)", value="", key="an_camera_id")
    with col2:
        analytics_zone_id = st.text_input("Zone ID (optional)", value="", key="an_zone_id")

    if st.button("Load analytics", key="an_load_button"):
        params = {}
        if analytics_camera_id:
            params["camera_id"] = analytics_camera_id
        if analytics_zone_id:
            params["zone_id"] = analytics_zone_id

        response = requests.get(api_url("/analytics"), params=params)
        if response.status_code == 200:
            body = response.json()
            m1, m2 = st.columns(2)
            m1.metric("Compliance rate", f"{body['compliance_rate'] * 100:.1f}%")
            m2.metric("Total violations", body["total_violations"])
            if body["most_common"]:
                st.bar_chart(body["most_common"])
        else:
            st.error(f"{response.status_code}: {response.json().get('detail', response.text)}")


# ---------------------------------------------------------------------------
# Explain (CAM) — POST /explain
# ---------------------------------------------------------------------------
with tab_explain:
    st.subheader("Eigen-CAM heatmap")
    st.caption(
        "Shows WHERE the model's convolutional features were most active — "
        "a sanity check, not a per-detection explanation. Class-agnostic: "
        "see app/explain/cam.py for the honest limitation this has for a "
        "safety system."
    )
    uploaded_explain_image = st.file_uploader(
        "Image", type=["jpg", "jpeg", "png"], key="explain_image_file"
    )

    if uploaded_explain_image is not None:
        st.image(uploaded_explain_image, caption="Uploaded image", width=400)

    if st.button("Generate heatmap", key="explain_button", disabled=uploaded_explain_image is None):
        files = {"file": (uploaded_explain_image.name, uploaded_explain_image.getvalue())}
        with st.spinner("Running Eigen-CAM..."):
            response = requests.post(api_url("/explain"), files=files)

        if response.status_code == 200:
            st.image(io.BytesIO(response.content), caption="Eigen-CAM heatmap", width=400)
        else:
            st.error(f"{response.status_code}: {response.text}")
