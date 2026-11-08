"""A fake Zoroval gateway and device-flow server for tests. Localhost only. No real keys.

Implements build contracts section 5 (gateway `/v1/traces`, `/v1/logs`, `/v1/ingest/status`;
device flow `/api/agent-connect`, `/api/agent-connect/token`) and section 6 (headers, resource
and span attributes). It decodes OTLP protobuf or JSON with zstd, gzip, deflate or none, and keeps
every span in memory for assertions.

Run alone: python -m testing.fake_gateway --port 4318 --auto-approve
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import secrets
import select
import socket
import string
import threading
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import zstandard
from google.protobuf import json_format
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (ExportLogsServiceRequest,
                                                                    ExportLogsServiceResponse)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (ExportTraceServiceRequest,
                                                                      ExportTraceServiceResponse)

MIB = 1024 * 1024
WIRE_LIMIT = 16 * MIB
DECODED_LIMIT = 64 * MIB
B32 = "abcdefghijklmnopqrstuvwxyz234567"
B62 = string.digits + string.ascii_letters
CONVERSATION_KEYS = ("zoroval.conversation.id", "gen_ai.conversation.id", "session.id", "langfuse.session.id",
                     "anthropic.session.thread.id")


def make_key() -> tuple[str, str]:
    """A random synthetic key in the contract format (5.3). Returns (key, key_id)."""
    key_id = "".join(secrets.choice(B32) for _ in range(12))
    secret = "".join(secrets.choice(B62) for _ in range(43))
    return f"zv_ingest_{key_id}_{secret}", key_id


def _status_body(message: str, code: int, content_type: str) -> bytes:
    """google.rpc.Status: field 1 code (varint), field 2 message (string)."""
    if "json" in content_type:
        return json.dumps({"code": code, "message": message}).encode()
    msg = message.encode()
    out = bytearray([0x08])
    value = code
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            break
    out.append(0x12)
    length = len(msg)
    while True:
        byte = length & 0x7F
        length >>= 7
        out.append(byte | (0x80 if length else 0))
        if not length:
            break
    return bytes(out) + msg


def _any_value(value: Any) -> Any:
    kind = value.WhichOneof("value")
    if kind == "array_value":
        return [_any_value(v) for v in value.array_value.values]
    if kind == "kvlist_value":
        return {kv.key: _any_value(kv.value) for kv in value.kvlist_value.values}
    if kind is None:
        return None
    return getattr(value, kind)


def _attrs(kvs: Any) -> dict[str, Any]:
    return {kv.key: _any_value(kv.value) for kv in kvs}


def _rfc3339(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class Project:
    project_id: str
    project_key: str


@dataclass
class KeyRecord:
    key: str
    key_id: str
    project: Project
    revoked: bool = False


@dataclass
class Batch:
    id: str
    project_id: str
    status: str
    spans_received: int
    spans_rejected: int
    dropped_reported: int
    trace_ids: list[str]
    received_at: float
    polls_until_done: int = 0


@dataclass
class ConnectRequest:
    device_code: str
    user_code: str
    name: str
    created: float
    status: str = "pending"
    polls: int = 0
    last_poll: float = 0.0
    key: Optional[KeyRecord] = None


@dataclass
class GatewayState:
    project: Project = field(default_factory=lambda: Project("00000000-0000-4000-8000-000000000001", "fake-chat"))
    keys: dict[str, KeyRecord] = field(default_factory=dict)
    spans: list[dict] = field(default_factory=list)
    logs: list[dict] = field(default_factory=list)
    requests: list[dict] = field(default_factory=list)
    batches: dict[str, Batch] = field(default_factory=dict)
    connects: dict[str, ConnectRequest] = field(default_factory=dict)
    fail_next: list[int] = field(default_factory=list)       # statuses to return before accepting
    retry_after: str = "0"
    max_span_bytes: int = 4 * MIB                            # spans over this are rejected (partial success)
    batch_polls_until_done: int = 0
    connect_interval: float = 3.0
    auto_approve_after: Optional[int] = None                 # approve after N token polls
    deny_after: Optional[int] = None
    delay_s: float = 0.0
    uplink_bytes_per_s: float = 0.0                          # slow client uplink: body takes len/rate to arrive
    abandoned: int = 0                                       # bodies the client gave up on mid-upload
    idle_timeout_s: Optional[float] = None                   # close kept-alive connections idle this long
    lock: threading.Lock = field(default_factory=threading.Lock)


class FakeGateway:
    """In-process fake. `with FakeGateway() as gw:` starts it on a free port."""

    def __init__(self, port: int = 0) -> None:
        self.state = GatewayState()
        state = self.state

        class Handler(_Handler):
            gateway_state = state

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeGateway":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def __enter__(self) -> "FakeGateway":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -------------------------------------------------------------- helpers for tests

    def issue_key(self) -> str:
        key, key_id = make_key()
        with self.state.lock:
            self.state.keys[key] = KeyRecord(key, key_id, self.state.project)
        return key

    def revoke(self, key: str) -> None:
        with self.state.lock:
            self.state.keys[key].revoked = True

    def approve(self, user_code: str) -> None:
        with self.state.lock:
            for req in self.state.connects.values():
                if req.user_code == user_code and req.status == "pending":
                    _approve(self.state, req)

    def deny(self, user_code: str) -> None:
        with self.state.lock:
            for req in self.state.connects.values():
                if req.user_code == user_code:
                    req.status = "denied"

    def spans(self, **match: Any) -> list[dict]:
        with self.state.lock:
            items = list(self.state.spans)
        return [s for s in items if all(s["attributes"].get(k) == v for k, v in match.items())]

    def generations(self) -> list[dict]:
        return self.spans(**{"zoroval.observation.type": "generation"})

    def grouped_traces(self, include_test: bool = True) -> dict[str, list[str]]:
        """Zoroval trace id -> OTel trace ids (turns), using the contract's grouping rules."""
        with self.state.lock:
            return _group(self.state, include_test)


def _approve(state: GatewayState, req: ConnectRequest) -> None:
    key, key_id = make_key()
    record = KeyRecord(key, key_id, state.project)
    state.keys[key] = record
    req.key = record
    req.status = "approved"


def _zt_id(project_id: str, kind: str, key: str) -> str:
    return "zt_" + hashlib.sha256(f"ztf1\0{project_id}\0{kind}\0{key}".encode()).hexdigest()[:32]


def _turns(state: GatewayState) -> dict[str, dict]:
    """OTel trace id -> {conversation, test, received_at, response ids, previous ids}."""
    turns: dict[str, dict] = {}
    for span in state.spans:
        turn = turns.setdefault(span["trace_id"], {"conversation": None, "test": False,
                                                   "received_at": span["received_at"], "responses": set(),
                                                   "previous": None})
        attrs = span["attributes"]
        for key in CONVERSATION_KEYS:
            if attrs.get(key) and turn["conversation"] is None:
                turn["conversation"] = str(attrs[key])
        if attrs.get("zoroval.test") is True:
            turn["test"] = True
        if attrs.get("gen_ai.response.id"):
            turn["responses"].add(attrs["gen_ai.response.id"])
        if attrs.get("gen_ai.openai.previous_response_id"):
            turn["previous"] = attrs["gen_ai.openai.previous_response_id"]
        turn["received_at"] = min(turn["received_at"], span["received_at"])
    # response_chain: a turn without an id joins the turn whose response it continues
    by_response = {rid: tid for tid, t in turns.items() for rid in t["responses"]}
    for _ in range(len(turns)):
        changed = False
        for tid, turn in turns.items():
            if turn["conversation"] is None and turn["previous"] in by_response:
                parent = turns[by_response[turn["previous"]]]
                turn["conversation"] = parent["conversation"] or f"chain:{by_response[turn['previous']]}"
                changed = True
        if not changed:
            break
    return turns


def _group(state: GatewayState, include_test: bool) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tid, turn in _turns(state).items():
        if turn["test"] and not include_test:
            continue
        key = turn["conversation"] or tid
        zt = _zt_id(state.project.project_id, "conversation", key)
        out.setdefault(zt, []).append(tid)
    return out


class _Handler(BaseHTTPRequestHandler):
    gateway_state: GatewayState
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # quiet
        pass

    def setup(self) -> None:
        self.timeout = self.gateway_state.idle_timeout_s
        super().setup()

    # -------------------------------------------------------------- plumbing

    def _send(self, status: int, body: bytes = b"", content_type: str = "application/json",
              headers: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload).encode())

    def _read(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _auth(self) -> tuple[Optional[KeyRecord], Optional[tuple[int, str]]]:
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return None, (401, "invalid_key")
        key = header[len("Bearer "):].strip()
        record = self.gateway_state.keys.get(key)
        if record is None:
            return None, (401, "invalid_key")
        if record.revoked:
            return None, (403, "key_revoked")
        return record, None

    # -------------------------------------------------------------- routes

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/v1/traces", "/v1/logs"):
            return self._otlp(path)
        if path == "/api/agent-connect":
            return self._connect()
        if path == "/api/agent-connect/token":
            return self._token()
        self._json(404, {"message": "not_found"})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/v1/ingest/status":
            return self._status(parse_qs(parsed.query))
        if parsed.path == "/connect":
            return self._send(200, b"<html>fake connect page</html>", "text/html")
        self._json(404, {"message": "not_found"})

    def _client_gone(self) -> bool:
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _otlp(self, path: str) -> None:
        st = self.gateway_state
        body = self._read()
        content_type = self.headers.get("Content-Type", "")
        if st.uplink_bytes_per_s:
            # The body trickles in over a slow uplink. A client that timed out and hung up before
            # it arrived gets nothing stored, like the real gateway (client_disconnected).
            time.sleep(len(body) / st.uplink_bytes_per_s)
            if self._client_gone():
                with st.lock:
                    st.abandoned += 1
                self.close_connection = True
                return
        if st.delay_s:
            time.sleep(st.delay_s)
        record, error = self._auth()
        if error:
            return self._send(error[0], _status_body(error[1], 16 if error[0] == 401 else 7, content_type),
                              "application/x-protobuf")
        with st.lock:
            st.requests.append({"path": path, "headers": dict(self.headers.items()), "wire_bytes": len(body),
                                "raw_bytes": None, "encoding": self.headers.get("Content-Encoding"),
                                "client_port": self.client_address[1], "at": time.monotonic()})
            request_record = st.requests[-1]
            if st.fail_next:
                code = st.fail_next.pop(0)
                reason = {429: "rate_limited", 503: "backlog", 408: "timeout"}.get(code, "server_error")
                return self._send(code, _status_body(reason, 14, content_type), "application/x-protobuf",
                                  {"Retry-After": st.retry_after})
        if content_type not in ("application/x-protobuf", "application/json"):
            return self._send(415, _status_body("unsupported_media", 3, ""), "application/x-protobuf")
        encoding = (self.headers.get("Content-Encoding") or "identity").lower()
        if len(body) > WIRE_LIMIT:
            return self._send(413, _status_body("too_large", 8, content_type), content_type)
        try:
            if encoding == "zstd":
                raw = zstandard.ZstdDecompressor().decompress(body, max_output_size=DECODED_LIMIT + 1)
            elif encoding == "gzip":
                raw = gzip.decompress(body)
            elif encoding == "deflate":
                raw = zlib.decompress(body)
            elif encoding == "identity":
                raw = body
            else:
                return self._send(415, _status_body("unsupported_media", 3, content_type), content_type)
        except Exception:  # noqa: BLE001
            return self._send(400, _status_body("bad_encoding", 3, content_type), content_type)
        if len(raw) > DECODED_LIMIT:
            return self._send(413, _status_body("too_large", 8, content_type), content_type)
        batch_id = self.headers.get("X-Zoroval-Batch-Id") or ""
        dropped = int(self.headers.get("X-Zoroval-Dropped-Spans") or 0)
        with st.lock:
            request_record["raw_bytes"] = len(raw)
            duplicate = bool(batch_id) and batch_id in st.batches
        if path == "/v1/logs":
            return self._logs(raw, content_type, duplicate)
        try:
            request = ExportTraceServiceRequest()
            if content_type == "application/json":
                json_format.Parse(raw.decode("utf-8"), request)
            else:
                request.ParseFromString(raw)
        except Exception:  # noqa: BLE001
            return self._send(400, _status_body("bad_otlp", 3, content_type), content_type)
        response = ExportTraceServiceResponse()
        if duplicate:
            return self._send(200, response.SerializeToString(), "application/x-protobuf")
        now = time.time()
        rejected = 0
        received = 0
        new_spans = []
        for rs in request.resource_spans:
            resource = _attrs(rs.resource.attributes)
            for ss in rs.scope_spans:
                for span in ss.spans:
                    received += 1
                    if span.ByteSize() > st.max_span_bytes:
                        rejected += 1
                        continue
                    new_spans.append({
                        "trace_id": span.trace_id.hex(), "span_id": span.span_id.hex(),
                        "parent_span_id": span.parent_span_id.hex() or None, "name": span.name,
                        "kind": span.kind, "attributes": _attrs(span.attributes), "resource": resource,
                        "scope": ss.scope.name, "status_code": span.status.code,
                        "status_message": span.status.message,
                        "events": [{"name": e.name, "attributes": _attrs(e.attributes)} for e in span.events],
                        "start_ns": span.start_time_unix_nano, "end_ns": span.end_time_unix_nano,
                        "received_at": now, "batch_id": batch_id,
                    })
        with st.lock:
            st.spans.extend(new_spans)
            groups = _group(st, include_test=True)
            new_tids = {s["trace_id"] for s in new_spans}
            trace_ids = sorted(zt for zt, tids in groups.items() if new_tids & set(tids))
            if batch_id:
                st.batches[batch_id] = Batch(batch_id, record.project.project_id,
                                             "processing" if st.batch_polls_until_done else "done",
                                             received, rejected, dropped, trace_ids, now,
                                             st.batch_polls_until_done)
        if rejected:
            response.partial_success.rejected_spans = rejected
            response.partial_success.error_message = "span_too_large"
        self._send(200, response.SerializeToString(), "application/x-protobuf")

    def _logs(self, raw: bytes, content_type: str, duplicate: bool) -> None:
        request = ExportLogsServiceRequest()
        try:
            if content_type == "application/json":
                json_format.Parse(raw.decode("utf-8"), request)
            else:
                request.ParseFromString(raw)
        except Exception:  # noqa: BLE001
            return self._send(400, _status_body("bad_otlp", 3, content_type), content_type)
        if not duplicate:
            with self.gateway_state.lock:
                for rl in request.resource_logs:
                    for sl in rl.scope_logs:
                        for rec in sl.log_records:
                            self.gateway_state.logs.append({
                                "event_name": rec.event_name, "trace_id": rec.trace_id.hex(),
                                "span_id": rec.span_id.hex(), "attributes": _attrs(rec.attributes),
                                "body": _any_value(rec.body) if rec.HasField("body") else None})
        self._send(200, ExportLogsServiceResponse().SerializeToString(), "application/x-protobuf")

    def _status(self, query: dict) -> None:
        st = self.gateway_state
        record, error = self._auth()
        if error:
            return self._json(error[0], {"message": error[1]})
        since_raw = (query.get("since") or [""])[0]
        since = 0.0
        if since_raw:
            try:
                since = datetime.fromisoformat(since_raw.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return self._json(400, {"message": "bad_since"})
        with st.lock:
            turns = _turns(st)
            real = [t for t in turns.values() if not t["test"]]
            first = min((t["received_at"] for t in real), default=None)
            last = max((t["received_at"] for t in turns.values()), default=None)
            since_turns = {tid: t for tid, t in turns.items() if t["received_at"] >= since}
            keys = {t["conversation"] or tid for tid, t in since_turns.items()}
            payload: dict[str, Any] = {
                "project_id": st.project.project_id, "project_key": st.project.project_key,
                "first_trace_received_at": _rfc3339(first) if first else None,
                "last_trace_received_at": _rfc3339(last) if last else None,
                "since": {"turns": len(since_turns), "traces": len(keys),
                          "ungrouped_turns": sum(1 for t in since_turns.values() if not t["conversation"]),
                          "dropped_spans_reported": sum(b.dropped_reported for b in st.batches.values()
                                                        if b.received_at >= since),
                          "errors": []},
            }
            batch_id = (query.get("batch_id") or [""])[0]
            if batch_id:
                batch = st.batches.get(batch_id)
                if batch is None:
                    payload["batch"] = {"id": batch_id, "status": "unknown"}
                else:
                    if batch.polls_until_done > 0:
                        batch.polls_until_done -= 1
                        if batch.polls_until_done == 0:
                            batch.status = "done"
                    payload["batch"] = {"id": batch.id, "status": batch.status,
                                        "spans_received": batch.spans_received,
                                        "spans_rejected": batch.spans_rejected,
                                        "trace_ids": batch.trace_ids if batch.status == "done" else []}
        self._json(200, payload)

    def _connect(self) -> None:
        st = self.gateway_state
        try:
            payload = json.loads(self._read() or b"{}")
        except ValueError:
            return self._json(400, {"message": "bad_json"})
        name = str(payload.get("name") or "")[:120]
        if not name:
            return self._json(400, {"message": "name_required"})
        device_code = secrets.token_urlsafe(32)[:43]
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        user_code = "".join(secrets.choice(alphabet) for _ in range(4)) + "-" + \
            "".join(secrets.choice(alphabet) for _ in range(4))
        with st.lock:
            st.connects[device_code] = ConnectRequest(device_code, user_code, name, time.time())
        host, port = self.server.server_address[:2]
        base = f"http://{host}:{port}"
        self._json(200, {"device_code": device_code, "user_code": user_code,
                         "verification_uri": f"{base}/connect",
                         "verification_uri_complete": f"{base}/connect?code={user_code}",
                         "expires_in": 600, "interval": st.connect_interval})

    def _token(self) -> None:
        st = self.gateway_state
        try:
            payload = json.loads(self._read() or b"{}")
        except ValueError:
            return self._json(400, {"message": "bad_json"})
        with st.lock:
            req = st.connects.get(str(payload.get("device_code") or ""))
            if req is None:
                return self._json(410, {"status": "expired"})
            now = time.time()
            if req.last_poll and now - req.last_poll < st.connect_interval * 0.8:
                req.last_poll = now
                return self._json(429, {"status": "slow_down"})
            req.last_poll = now
            req.polls += 1
            if now - req.created > 600:
                req.status = "expired"
            if req.status == "pending":
                if st.deny_after is not None and req.polls >= st.deny_after:
                    req.status = "denied"
                elif st.auto_approve_after is not None and req.polls >= st.auto_approve_after:
                    _approve(st, req)
            if req.status == "pending":
                return self._json(202, {"status": "pending"})
            if req.status == "approved" and req.key is not None:
                req.status = "used"
                key = req.key
                return self._json(200, {"key": key.key, "key_id": key.key_id, "project_id": key.project.project_id,
                                        "project_key": key.project.project_key,
                                        "ingest_url": f"http://{self.server.server_address[0]}:"
                                                      f"{self.server.server_address[1]}",
                                        "env_var": "ZOROVAL_INGEST_KEY"})
            return self._json(410, {"status": req.status})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake Zoroval gateway for local tests")
    parser.add_argument("--port", type=int, default=4318)
    parser.add_argument("--auto-approve", action="store_true", help="approve device-flow requests on first poll")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    gw = FakeGateway(args.port)
    gw.state.connect_interval = args.interval
    if args.auto_approve:
        gw.state.auto_approve_after = 1
    print(gw.url, flush=True)
    gw.server.serve_forever()


if __name__ == "__main__":
    main()
