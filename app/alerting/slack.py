"""
app/alerting/slack.py — threshold-based Slack alerting.

WHY THRESHOLD-BASED (N violations in T seconds from ONE zone), NOT
ONE-ALERT-PER-VIOLATION
  A construction site can have dozens of brief violations an hour under
  normal conditions (a worker briefly without a vest while grabbing gear).
  Alerting on every single one would spam the channel into uselessness —
  the classic alert-fatigue failure mode where a real emergency gets lost
  in noise nobody reads anymore. A rate rule instead asks "is this zone
  seeing a SURGE of non-compliance right now?" — which is what a safety
  lead actually needs paged for. N and T are config-driven
  (alert.violations_threshold, alert.window_seconds in config.yaml), not
  hardcoded, so the rate can be tuned per deployment without a code change.

WHY "N violations" COUNTS NEWLY-OPENED INCIDENTS, NOT EVERY DB WRITE
  Pipeline._persist() writes a row on every frame an incident is still
  open (extending last_seen) AND once more when it closes — one long
  violation can generate many UPDATEs. Counting every write would trigger
  an alert from ONE stubborn worker standing still, which isn't what "N
  violations" means. Counting only first-time INSERTs (event.id was None
  before this call) means the threshold reflects N DISTINCT incidents,
  matching the plan's "N violations in T seconds" in the way a human
  reading that sentence would expect.

WHY COOLDOWN STATE IS PER-ZONE AND PROCESS-LIFETIME (A SINGLETON), NOT
PER-Pipeline-INSTANCE
  Tracker/ViolationEngine are deliberately fresh per Pipeline session
  (Step 5) because track IDs and open incidents genuinely don't carry
  meaning across sessions. Alert cooldown is the opposite: if zone_a
  triggers an alert from one /detect/video upload, a SECOND upload five
  seconds later processing the same zone must still respect that cooldown
  — otherwise every new Pipeline (one per upload) resets the anti-spam
  clock to zero. get_alert_manager() below is a process-wide singleton for
  exactly this reason, mirroring how app.config.settings is one process-
  wide singleton rather than reloaded per caller.

WHY IT NO-OPS SAFELY WHEN webhook_url IS EMPTY OR alert.enabled IS FALSE
  Before you create a Slack webhook (or in a test run), the app must keep
  working — a missing/blank secret should silently skip alerting, not
  crash every violation-persisting code path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

import requests
from sqlalchemy.orm import Session

from app.config import settings
from app.db import crud


class AlertManager:
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        enabled: Optional[bool] = None,
        threshold: Optional[int] = None,
        window_seconds: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
    ) -> None:
        cfg = settings.config.alert
        self.webhook_url = webhook_url if webhook_url is not None else settings.slack_webhook_url
        self.enabled = enabled if enabled is not None else cfg.enabled
        self.threshold = threshold if threshold is not None else cfg.violations_threshold
        self.window_seconds = window_seconds if window_seconds is not None else cfg.window_seconds
        self.cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else cfg.cooldown_seconds
        self._last_alert_at: dict[str, datetime] = {}

    def notify_new_violation(
        self, session: Session, *, zone_id: str, camera_id: str, now: Optional[datetime] = None
    ) -> bool:
        """Call once per NEWLY-OPENED incident (not per update). Returns True
        iff an alert was actually sent, so callers/tests can assert on it."""
        if not self.enabled or not self.webhook_url:
            return False

        now = now or datetime.now(timezone.utc)
        last_alert = self._last_alert_at.get(zone_id)
        if last_alert is not None and (now - last_alert).total_seconds() < self.cooldown_seconds:
            return False  # still cooling down from this zone's last alert

        window_start = now - timedelta(seconds=self.window_seconds)
        count = crud.count_recent_violations(session, zone_id=zone_id, since=window_start)
        if count < self.threshold:
            return False

        self._send(zone_id=zone_id, camera_id=camera_id, count=count)
        self._last_alert_at[zone_id] = now
        return True

    def _send(self, *, zone_id: str, camera_id: str, count: int) -> None:
        text = (
            f":rotating_light: *{count} PPE violations* in zone `{zone_id}` "
            f"(camera `{camera_id}`) within the last {int(self.window_seconds)}s. "
            f"Cooling down for {int(self.cooldown_seconds)}s before the next alert."
        )
        response = requests.post(self.webhook_url, json={"text": text}, timeout=5)
        response.raise_for_status()


@lru_cache(maxsize=1)
def get_alert_manager() -> AlertManager:
    return AlertManager()
