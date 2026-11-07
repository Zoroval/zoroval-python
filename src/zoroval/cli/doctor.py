"""`zoroval doctor`: prove the setup works (contracts 5.1 `GET /v1/ingest/status`).

1. Finds the key (environment or env file) and checks its format. Never prints it.
2. Checks the gateway accepts the key.
3. Sends a test trace: two test turns with one conversation id (`zoroval.test=true`).
4. Waits for its batch to reach `done` and checks the two turns became one trace.
5. Reports "first trace received" for real app traffic, and warns about ungrouped turns and
   dropped spans in the last 24 hours. With --wait N it waits up to N seconds for the first
   app trace.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from .. import _config
from .._transport import Transport, uuid7
from .._version import SCOPE_NAME, __version__
from . import envfile

PASS, WARN, FAIL = "pass", "warn", "fail"


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _find_key(root: Path, env_file: Path) -> tuple[Optional[str], str, Optional[str]]:
    """(key, source, ingest_url). The key is used, never shown."""
    env_path = env_file if env_file.is_absolute() else root / env_file
    url = os.environ.get("ZOROVAL_INGEST_URL")
    if os.environ.get(envfile.KEY_VAR):
        return os.environ[envfile.KEY_VAR].strip(), "environment", url
    if env_path.exists():
        key = envfile.read_value(env_path, envfile.KEY_VAR)
        if key:
            return key.strip(), env_path.name, url or envfile.read_value(env_path, envfile.URL_VAR)
    return None, "", url


def _test_spans(service: str) -> tuple[list[Any], str]:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    collected: list[Any] = []

    class _Collect(SpanExporter):
        def export(self, spans: Any) -> Any:
            collected.extend(spans)
            return SpanExportResult.SUCCESS

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(SimpleSpanProcessor(_Collect()))
    tracer = provider.get_tracer(SCOPE_NAME, __version__)
    conversation = f"zoroval-doctor-{uuid.uuid4()}"
    for index in range(2):
        attrs = {"zoroval.observation.type": "turn", "zoroval.conversation.id": conversation,
                 "gen_ai.conversation.id": conversation, "zoroval.test": True, "zoroval.grouping": "explicit",
                 "zoroval.input": json.dumps(f"zoroval doctor test turn {index + 1}")}
        with tracer.start_as_current_span("zoroval doctor turn", attributes=attrs):
            with tracer.start_as_current_span("zoroval doctor step", attributes={
                    "zoroval.observation.type": "step", "zoroval.conversation.id": conversation,
                    "gen_ai.conversation.id": conversation}):
                pass
    provider.shutdown()
    return collected, conversation


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, str]] = []

    def add(self, level: str, name: str, detail: str) -> None:
        self.checks.append({"level": level, "check": name, "detail": detail})

    @property
    def ok(self) -> bool:
        return not any(c["level"] == FAIL for c in self.checks)


def run(*, root: Path, env_file: Path, ingest_url: Optional[str], timeout: float, wait: float,
        skip_test_trace: bool = False) -> Report:
    report = Report()
    report.add(PASS, "sdk", f"zoroval {__version__}")
    for package in ("openai", "anthropic", "langfuse", "opentelemetry-sdk"):
        try:
            from importlib.metadata import version
            report.add(PASS, f"package {package}", version(package))
        except Exception:  # noqa: BLE001
            pass

    key, source, file_url = _find_key(root, env_file)
    if not key:
        report.add(FAIL, "key", f"{envfile.KEY_VAR} not found in the environment or {env_file}. "
                   "Run `zoroval login`.")
        return report
    key_id = _config.key_id_of(key)
    if key_id is None:
        report.add(FAIL, "key", f"{envfile.KEY_VAR} (from {source}) is not a Zoroval ingest key.")
        return report
    report.add(PASS, "key", f"found in {source}, key id {key_id}")
    if source != "environment":
        report.add(WARN, "key loading", f"the key is in {source}, not in this shell's environment. "
                   "Make sure the app loads that file (for example with python-dotenv).")

    cfg = _config.Config(api_key=key, ingest_url=ingest_url or file_url or _config.DEFAULT_INGEST_URL,
                         service_name=os.environ.get("ZOROVAL_SERVICE_NAME") or "zoroval-doctor")
    transport = Transport(cfg)
    status, body = transport.get_json("/v1/ingest/status")
    if status in (401, 403):
        report.add(FAIL, "gateway", f"the gateway refused the key ({status} {body.get('message', '')}). "
                   "Run `zoroval login --replace` or make a new key in the app.")
        return report
    if status != 200:
        report.add(FAIL, "gateway", f"{cfg.ingest_url} answered {status or 'nothing'} "
                   f"({body.get('message', '')}).")
        return report
    report.add(PASS, "gateway", f"{cfg.ingest_url} accepts the key; project {body.get('project_key')}")

    if not skip_test_trace:
        start = datetime.now(timezone.utc) - timedelta(seconds=1)
        spans, _ = _test_spans(cfg.service_name)
        payload = transport.compress(transport.encode_spans(spans, {}))
        batch_id = uuid7()
        result = transport.send(cfg.traces_url, payload, deadline=time.monotonic() + timeout, batch_id=batch_id)
        if not result.ok:
            report.add(FAIL, "test trace", f"not accepted ({result.status} {result.reason})")
            return report
        report.add(PASS, "test trace", f"sent 2 test turns, batch {batch_id}")
        deadline = time.monotonic() + timeout
        batch: dict[str, Any] = {}
        since: dict[str, Any] = {}
        while time.monotonic() < deadline:
            status, body = transport.get_json(f"/v1/ingest/status?batch_id={batch_id}&since={quote(_rfc3339(start))}")
            batch = body.get("batch") or {}
            since = body.get("since") or {}
            if status == 200 and batch.get("status") in ("done", "done_with_errors", "failed"):
                break
            time.sleep(0.5)
        if batch.get("status") not in ("done", "done_with_errors"):
            report.add(FAIL, "test trace processed", f"batch status {batch.get('status') or 'unknown'} "
                       f"after {timeout:.0f}s")
            return report
        report.add(PASS, "test trace processed", f"batch {batch.get('status')}, "
                   f"{batch.get('spans_received', 0)} spans")
        trace_ids = batch.get("trace_ids") or []
        if len(trace_ids) == 1 and int(since.get("turns", 0)) >= 2:
            report.add(PASS, "grouping", "2 test turns with one conversation id became 1 trace")
        else:
            report.add(FAIL, "grouping", f"expected 1 trace from 2 turns, got {len(trace_ids)} trace ids "
                       f"and {since.get('turns', 0)} turns")

    # real traffic
    deadline = time.monotonic() + max(wait, 0.0)
    first = None
    recent: dict[str, Any] = {}
    while True:
        day_ago = quote(_rfc3339(datetime.now(timezone.utc) - timedelta(hours=24)))
        status, body = transport.get_json(f"/v1/ingest/status?since={day_ago}")
        first = body.get("first_trace_received_at") if status == 200 else None
        recent = body.get("since") or {}
        if first or time.monotonic() >= deadline:
            break
        time.sleep(1.0)
    if first:
        report.add(PASS, "first trace received", f"app traces are arriving (first at {first})")
    else:
        report.add(WARN if wait <= 0 else FAIL, "first trace received",
                   "no app traces yet. Run your app once, then `zoroval doctor --wait 60`.")
    ungrouped = int(recent.get("ungrouped_turns") or 0)
    if ungrouped:
        report.add(WARN, "grouping (app)", f"{ungrouped} turns in the last 24 h arrived ungrouped. "
                   "Wrap chat handlers in zoroval.conversation(chat_id) so one chat is one trace.")
    dropped = int(recent.get("dropped_spans_reported") or 0)
    if dropped:
        report.add(WARN, "dropped spans", f"the SDK reported {dropped} dropped spans in the last 24 h "
                   "(queue full or failed sends).")
    for error in recent.get("errors") or []:
        report.add(WARN, "ingest errors", f"{error.get('reason_code')}: {error.get('count')}")
    return report


def main(args: Any) -> int:
    report = run(root=Path(args.root or "."), env_file=Path(args.env_file), ingest_url=args.ingest_url,
                 timeout=float(args.timeout), wait=float(args.wait), skip_test_trace=args.no_test_trace)
    if args.json:
        print(json.dumps({"ok": report.ok, "checks": report.checks}, indent=2))
    else:
        marks = {PASS: "ok  ", WARN: "warn", FAIL: "FAIL"}
        for check in report.checks:
            print(f"[{marks[check['level']]}] {check['check']}: {check['detail']}")
        print("Doctor: " + ("all required checks passed." if report.ok else "failed. Fix the FAIL lines above."))
    return 0 if report.ok else 1
