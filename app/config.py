"""
app/config.py — typed loader that turns config/config.yaml + .env into ONE
validated, importable settings object.

Two sources, deliberately separated:
  * config/config.yaml -> structural, non-secret, version-controlled config
    (thresholds, zones, classes, alert rule). Modelled by `AppConfig` below.
  * .env               -> secrets + machine-specific values (DB creds, Slack
    URL). Modelled by `Settings` (pydantic-settings, reads env vars).

Why typed (pydantic) instead of raw yaml.safe_load into dicts:
  * a malformed threshold or a missing zone field fails LOUDLY at startup with
    a precise error, instead of silently becoming None and blowing up deep in
    the pipeline ("works on my machine" bugs). This is directly interview-
    defensible: config is validated at the boundary, once.

Usage everywhere else:
    from app.config import settings
    settings.config.detection.conf_threshold
    settings.database_url
"""

from __future__ import annotations

import os

# MUST run before ultralytics is imported ANYWHERE in the process — its
# AUTOINSTALL flag is read once, at ultralytics.utils import time, from this
# env var. Left on, a missing optional dependency (e.g. ByteTrack's `lap`
# solver) triggers a silent `pip install` into whatever Python is on PATH,
# which may not be this project's venv. We pin every real dependency in
# requirements.txt instead, so a missing package should fail loudly, not
# vanish into an auto-install. setdefault() so an operator can still opt
# back in by exporting YOLO_AUTOINSTALL=True before running.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Structural config models (mirror config/config.yaml section-for-section)
# ---------------------------------------------------------------------------
class ClassConfig(BaseModel):
    names: dict[int, str]                    # class_id -> name
    person_class: int
    ppe_absence_classes: dict[int, str]      # class_id -> violation type name

    @field_validator("names", "ppe_absence_classes", mode="before")
    @classmethod
    def _coerce_int_keys(cls, v: dict) -> dict:
        # YAML mapping keys may parse as str/int depending on quoting; force int.
        return {int(k): val for k, val in v.items()}


class DetectionConfig(BaseModel):
    # `model_path` starts with "model_", which Pydantic reserves; opting out of
    # the protected namespace keeps the natural field name without a warning.
    model_config = {"protected_namespaces": ()}

    model_path: str
    conf_threshold: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(ge=0.0, le=1.0)
    device: str = "cpu"
    imgsz: int = 640


class TrackingConfig(BaseModel):
    tracker: str = "bytetrack.yaml"
    persist: bool = True


class SamplingConfig(BaseModel):
    video_every_n_frames: int = Field(ge=1)
    stream_every_n_frames: int = Field(ge=1)


class Zone(BaseModel):
    id: str
    name: str
    camera_id: str
    polygon: list[list[float]]               # list of [x, y] vertices

    @field_validator("polygon")
    @classmethod
    def _need_three_points(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) < 3:
            raise ValueError("a zone polygon needs at least 3 vertices")
        return v


class Camera(BaseModel):
    id: str
    name: str
    # int (webcam index) OR str (RTSP/file path). Both valid frame sources.
    source: Union[int, str]


class ViolationLogicConfig(BaseModel):
    incident_timeout_seconds: float = Field(gt=0)
    min_incident_seconds: float = Field(ge=0)
    ppe_person_min_overlap: float = Field(ge=0.0, le=1.0)


class AlertConfig(BaseModel):
    enabled: bool = True
    violations_threshold: int = Field(ge=1)
    window_seconds: float = Field(gt=0)
    cooldown_seconds: float = Field(ge=0)


class SnapshotConfig(BaseModel):
    dir: str = "snapshots"
    save: bool = True


class AppConfig(BaseModel):
    """The whole structural config, validated as one object."""

    classes: ClassConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    sampling: SamplingConfig
    zones: list[Zone]
    cameras: list[Camera]
    violation: ViolationLogicConfig
    alert: AlertConfig
    snapshots: SnapshotConfig

    # --- convenience lookups so callers don't re-scan lists -----------------
    def zone(self, zone_id: str) -> Zone:
        for z in self.zones:
            if z.id == zone_id:
                return z
        raise KeyError(f"unknown zone_id: {zone_id}")

    def camera(self, camera_id: str) -> Camera:
        for c in self.cameras:
            if c.id == camera_id:
                return c
        raise KeyError(f"unknown camera_id: {camera_id}")

    def zones_for_camera(self, camera_id: str) -> list[Zone]:
        return [z for z in self.zones if z.camera_id == camera_id]

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AppConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p.resolve()}")
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Secrets / env (pydantic-settings reads real environment + .env file)
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Secrets and wiring from the environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",              # ignore unrelated env vars
    )

    # Postgres pieces (used from Step 4)
    postgres_user: str = "ppe"
    postgres_password: str = "change_me_locally"
    postgres_db: str = "ppe_compliance"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url_override: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # Slack (used in Step 6; empty until then -> alerting no-ops safely)
    slack_webhook_url: Optional[str] = None

    # Where to find the structural YAML
    ppe_config_path: str = "config/config.yaml"

    @property
    def database_url(self) -> str:
        """Prefer an explicit DATABASE_URL; otherwise build one from parts."""
        if self.database_url_override:
            return self.database_url_override
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def config(self) -> AppConfig:
        # Cached so we parse the YAML once per process (see lru_cache below).
        return _load_app_config(self.ppe_config_path)


@lru_cache(maxsize=1)
def _load_app_config(path: str) -> AppConfig:
    return AppConfig.from_yaml(path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Import-friendly singleton: `from app.config import settings`
settings = get_settings()
