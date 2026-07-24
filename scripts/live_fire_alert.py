"""
scripts/live_fire_alert.py — ONE-OFF, MANUAL proof that a real message lands
in your Slack channel. Not part of the automated pytest suite (see
tests/test_step6_alerting.py's module docstring for why: hitting a real
webhook on every test run would spam the channel).

Run directly:  python scripts/live_fire_alert.py

Reads SLACK_WEBHOOK_URL from .env (via app.config.settings) — same code
path production alerting uses (AlertManager._send), just invoked directly
with a fake incident count instead of through the full detect/track/violate
pipeline, so this succeeds even before PPE-trained weights (Step 8) exist.
"""

from __future__ import annotations

import sys

from app.config import settings


def main() -> int:
    if not settings.slack_webhook_url:
        print("SLACK_WEBHOOK_URL is not set in .env — nothing to fire.")
        return 1

    from app.alerting.slack import AlertManager

    manager = AlertManager(
        webhook_url=settings.slack_webhook_url,
        enabled=True,
        threshold=1,          # fire immediately, real threshold math is unit-tested already
        window_seconds=60,
        cooldown_seconds=0,   # no cooldown needed for a single manual test message
    )
    manager._send(zone_id="zone_a", camera_id="cam_01", count=5)
    print("Sent. Check your Slack channel now.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
