"""Progress reports for a guided setup. Active only when ZOROVAL_SETUP is set.

Best effort: short timeout, never raises, never changes a command's result. A 404 or 410 (unknown
or finished code) is ignored without a word. Only event names and app or framework names are sent.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Optional

from .. import _config
from .._transport import USER_AGENT, _default_opener

ENV_VAR = "ZOROVAL_SETUP"
TIMEOUT = 2.0
MAX_NAMES = 10

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+-]{0,39}$")

_sent: set[str] = set()


def setup_code() -> Optional[str]:
    return os.environ.get(ENV_VAR, "").strip() or None


def _base_url(app_url: Optional[str]) -> str:
    return (app_url or os.environ.get("ZOROVAL_APP_URL") or _config.DEFAULT_APP_URL).rstrip("/")


def report(event: str, detail: Optional[dict] = None, *, app_url: Optional[str] = None) -> None:
    code = setup_code()
    if not code:
        return
    try:
        body: dict[str, Any] = {"code": code, "event": event}
        if detail:
            body["detail"] = detail
        request = urllib.request.Request(
            _base_url(app_url) + "/api/agent-setup/progress", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT})
        with _default_opener()(request, timeout=TIMEOUT) as response:
            response.read()
    except Exception:  # noqa: BLE001  (404, 410, no network, anything: ignored)
        return


def started(*, app_url: Optional[str] = None) -> None:
    """Posts `started` once per process."""
    if "started" in _sent or not setup_code():
        return
    _sent.add("started")
    report("started", app_url=app_url)


def app_names(scan_report: dict[str, Any]) -> list[str]:
    """Framework and app names from a detect report. Names only: no paths, no values."""
    names: list[str] = []
    tracing = scan_report.get("tracing") or {}
    for key in ("openai", "anthropic", "langfuse", "otel", "openinference", "openllmetry", "langsmith"):
        if tracing.get(key):
            names.append(key)
    js = (scan_report.get("packages") or {}).get("typescript") or {}
    if "ai" in js:
        names.append("vercel-ai")
    names.extend(str(k) for k in (scan_report.get("servers") or {}))
    seen: list[str] = []
    for name in names:
        if NAME_PATTERN.match(name) and name not in seen:
            seen.append(name)
    return seen[:MAX_NAMES]


def detected(scan_report: dict[str, Any], *, app_url: Optional[str] = None) -> None:
    started(app_url=app_url)
    report("detected", {"apps": app_names(scan_report)}, app_url=app_url)


def doctor_started(*, app_url: Optional[str] = None) -> None:
    started(app_url=app_url)
    report("doctor_started", app_url=app_url)
