# PPE Compliance Monitoring System

Turns construction-site camera/video feeds into continuous, **evidenced** PPE
compliance monitoring — near-real-time intervention plus audit-ready reports,
instead of reactive manual spot checks.

**Stack:** YOLOv8 (Ultralytics) + OpenCV · ByteTrack tracking · FastAPI ·
Postgres (Docker Compose) · Slack alerting · Eigen-CAM explainability ·
Streamlit thin client.

> Build is intentionally incremental (one layer at a time) for depth of
> understanding. See the plan for the full build order and rationale.

## Layout (see plan for the "why")

| Path | Role |
|------|------|
| `config/config.yaml` | Single source of truth: thresholds, zones, classes, alert rule |
| `.env` (gitignored)  | Secrets: DB creds, Slack webhook |
| `app/schemas.py`     | The one canonical `ViolationEvent` / `Detection` schema |
| `app/config.py`      | Typed config + settings loader (pydantic) |
| `app/detection/`     | YOLOv8 inference + unified frame source (Step 1) |
| `app/tracking/`      | ByteTrack via `model.track()` (Step 2) |
| `app/violations/`    | Zone-scoped rules, track-ID dedup, duration (Step 3) |
| `app/db/`            | Postgres persistence (Step 4) |
| `app/api/`           | FastAPI endpoints (Step 5) |
| `app/alerting/`      | Slack webhook alerts (Step 6) |
| `app/explain/`       | Eigen-CAM heatmaps (Step 7) |
| `app/pipeline.py`    | The one frame-processing path (image/video/stream) |
| `training/`          | YOLOv8n vs YOLOv8s benchmark — runs on Colab/Kaggle GPU (Step 8) |
| `ui/`                | Streamlit thin client (Step 9) |

## Local setup (Windows, CPU-only)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env
```

_Build status: Step 0 (scaffold + config + schema) complete._
