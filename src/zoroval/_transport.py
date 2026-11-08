"""OTLP/HTTP protobuf transport with zstd, built for the Zoroval gateway (contracts 5.1 and 6).

- Encodes spans with the OpenTelemetry protobuf encoder, then adds Zoroval's resource and span
  attributes on the protobuf message (no span copies on the request path).
- Compresses with zstd level 3. Splits a batch whose compressed size passes the wire limit.
- One `X-Zoroval-Batch-Id` (UUIDv7) per batch, reused on its retries.
- One persistent HTTP/1.1 connection (keep-alive) for the export thread, reopened after a failure
  or a fork. Standard library only; proxies from HTTP(S)_PROXY and NO_PROXY.
- The request timeout grows with the body size, between `request_timeout` and a ceiling below the
  gateway's 60 s body-read deadline.
- Retries 429 and 503 after `Retry-After`, and 5xx, 408, timeouts and connection errors with
  backoff, always inside the caller's deadline. A `Retry-After` that passes the deadline pauses
  every sender on this transport until then. 401 and 403 stop all export for the process.
"""

from __future__ import annotations

import base64
import email.utils
import http.client
import json
import os
import random
import select
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import zstandard
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

from ._config import Config
from ._util import COUNTERS, log_once, logger
from ._version import SDK_NAME, __version__

USER_AGENT = f"{SDK_NAME}/{__version__}"
FATAL_STATUS = {401, 403}
RETRY_AFTER_STATUS = {429, 503}
RETRY_STATUS = {500, 502, 504}
TIMEOUT_STATUS = 408                 # the gateway did not get the whole body within its read deadline
WIRE_MARGIN = 64 * 1024

# Request timeout = request_timeout + wire bytes / MIN_UPLINK, capped at TIMEOUT_CEILING_S.
# 64 KiB/s is a 512 kbit/s uplink, about the slow-3G and flaky-mobile profiles of the 2026-11-09
# low-bandwidth audit. The ceiling stays under the gateway's 60 s whole-body read deadline.
MIN_UPLINK_BYTES_PER_S = 64 * 1024
TIMEOUT_CEILING_S = 55.0
MAX_PAUSE_S = 3600.0                 # longest Retry-After honoured; then one probe request
MIN_PAUSE_S = 1.0                    # shortest pause once a batch has used its attempts on 429/503
MAX_BACKOFF_S = 60.0                 # longest wait between failed batches while the gateway is down
_STALE_ERRORS = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, http.client.BadStatusLine)


def uuid7() -> str:
    """A UUIDv7 string (RFC 9562): 48-bit Unix ms time, version 7, variant 10, random rest."""
    ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= ((rand >> 62) & 0xFFF) << 64
    value |= 0b10 << 62
    value |= rand & ((1 << 62) - 1)
    return str(uuid.UUID(int=value))


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def parse_status_message(body: bytes, content_type: str) -> str:
    """The `message` of a google.rpc.Status body (protobuf or JSON); the gateway puts the reason code there."""
    if not body:
        return ""
    try:
        if "json" in content_type:
            return str(json.loads(body.decode("utf-8")).get("message", ""))
        pos = 0
        while pos < len(body):
            tag, pos = _varint(body, pos)
            field, wire = tag >> 3, tag & 7
            if wire == 0:
                _, pos = _varint(body, pos)
            elif wire == 2:
                length, pos = _varint(body, pos)
                chunk = body[pos:pos + length]
                pos += length
                if field == 2:
                    return chunk.decode("utf-8", "replace")
            elif wire == 5:
                pos += 4
            elif wire == 1:
                pos += 8
            else:
                return ""
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _string_kv(key: str, value: Any) -> KeyValue:
    kv = KeyValue(key=key)
    if isinstance(value, bool):
        kv.value.bool_value = value
    elif isinstance(value, int):
        kv.value.int_value = value
    elif isinstance(value, float):
        kv.value.double_value = value
    elif isinstance(value, (list, tuple)):
        for item in value:
            kv.value.array_value.values.append(AnyValue(string_value=str(item)))
    else:
        kv.value.string_value = str(value)
    return kv


def _default_opener() -> Callable[..., Any]:
    """Proxies from HTTP(S)_PROXY and NO_PROXY only. The default urllib opener also asks macOS
    for system proxies, which crashes a forked child (gunicorn and uvicorn workers on a Mac)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(urllib.request.getproxies_environment()))
    return opener.open


class ConnectTimeout(ConnectionError):
    """The TCP or TLS handshake timed out (the body was never sent)."""


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Seconds from a Retry-After header (delta seconds or HTTP date), clamped to [0, MAX_PAUSE_S]."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        seconds = when.timestamp() - time.time()
    return min(max(seconds, 0.0), MAX_PAUSE_S)


def _connection_dropped(sock: Any) -> bool:
    """An idle keep-alive socket that is readable has been closed by the server (or is broken)."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable)
    except (OSError, ValueError):
        return True


@dataclass
class SendResult:
    ok: bool
    status: int = 0
    reason: str = ""
    rejected: int = 0
    fatal: bool = False
    retryable: bool = False   # failed for a reason that may pass (5xx, 408, timeout, connection)
    timeout: bool = False     # the request timed out or got 408: the caller may split the batch
    paused: bool = False      # the transport is paused (Retry-After) or backing off past the deadline


class Transport:
    """Sends encoded OTLP requests. Thread-safe enough for one export thread plus flush callers."""

    def __init__(self, cfg: Config, *, opener: Optional[Callable[..., Any]] = None) -> None:
        self.cfg = cfg
        self._zstd = zstandard.ZstdCompressor(level=3, write_content_size=True)
        self._opener = opener or _default_opener()   # small GET requests (doctor)
        self._dropped_unreported = 0
        self.fatal_reason: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_batch_id: Optional[str] = None
        # Pause and backoff shared by every sender (export thread, log exporter).
        self._hold_until = 0.0
        self._hold_kind = ""
        self._hold_reason = ""
        self._failures = 0
        self._ssl_context: Optional[ssl.SSLContext] = None
        self._init_runtime()

    def _init_runtime(self) -> None:
        self._lock = threading.Lock()
        self.abort = threading.Event()   # set by shutdown to stop retry sleeps at once
        self._conn_lock = threading.Lock()
        self._conn: Optional[http.client.HTTPConnection] = None
        self._conn_key: Optional[tuple] = None
        self._conn_pid = os.getpid()
        self.connections_opened = 0

    def after_fork(self) -> None:
        """In a forked child: never touch the parent's socket or locks. Open a new connection later.
        A pause stays: it is the gateway's answer for this key, not for one process."""
        self._init_runtime()

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None and self._conn_pid == os.getpid():
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ drop accounting

    def report_dropped(self, count: int) -> None:
        if count > 0:
            with self._lock:
                self._dropped_unreported += count

    def _take_dropped(self) -> int:
        with self._lock:
            return self._dropped_unreported

    def _ack_dropped(self, sent: int) -> None:
        with self._lock:
            self._dropped_unreported = max(0, self._dropped_unreported - sent)

    # ------------------------------------------------------------------ encoding

    def resource_attributes(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {"zoroval.sdk.name": SDK_NAME, "zoroval.sdk.version": __version__}
        if self.cfg.environment:
            attrs["deployment.environment.name"] = self.cfg.environment
        if self.cfg.release:
            attrs["service.version"] = self.cfg.release
        return attrs

    def patch_resources(self, resource_list: Sequence[Any]) -> None:
        extra = self.resource_attributes()
        for item in resource_list:
            existing = {kv.key: kv for kv in item.resource.attributes}
            service = existing.get("service.name")
            if service is None:
                item.resource.attributes.append(_string_kv("service.name", self.cfg.service_name))
            elif service.value.string_value.startswith("unknown_service") and \
                    not self.cfg.service_name.startswith("unknown_service"):
                service.value.string_value = self.cfg.service_name
            for key, value in extra.items():
                if key not in existing:
                    item.resource.attributes.append(_string_kv(key, value))

    def encode_spans(self, spans: Sequence[Any], extras: Mapping[int, Mapping[str, Any]]) -> bytes:
        from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

        request = encode_spans(spans)
        self.patch_resources(request.resource_spans)
        if extras:
            by_id = {span_id.to_bytes(8, "big"): attrs for span_id, attrs in extras.items()}
            for rs in request.resource_spans:
                for ss in rs.scope_spans:
                    for pb_span in ss.spans:
                        attrs = by_id.get(pb_span.span_id)
                        if attrs:
                            present = {kv.key for kv in pb_span.attributes}
                            for key, value in attrs.items():
                                if key not in present:
                                    pb_span.attributes.append(_string_kv(key, value))
        return request.SerializeToString()

    def encode_logs(self, records: Sequence[Any]) -> bytes:
        from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs

        request = encode_logs(records)
        self.patch_resources(request.resource_logs)
        return request.SerializeToString()

    def compress(self, raw: bytes) -> bytes:
        return self._zstd.compress(raw)

    # ------------------------------------------------------------------ pause and backoff

    def request_timeout_for(self, wire_bytes: int) -> float:
        """Per-request timeout: the floor plus time for the body on a slow uplink, capped."""
        floor = self.cfg.request_timeout
        return min(floor + wire_bytes / MIN_UPLINK_BYTES_PER_S, max(floor, TIMEOUT_CEILING_S))

    def hold(self, seconds: float, kind: str, reason: str) -> None:
        """No sends until `seconds` from now. A later hold never shortens an earlier one."""
        until = time.monotonic() + max(seconds, 0.0)
        with self._lock:
            if until > self._hold_until:
                self._hold_until, self._hold_kind, self._hold_reason = until, kind, reason
        if kind == "paused" and seconds >= 5:
            log_once(f"paused:{reason}", "zoroval: the gateway asked to pause export for %.0fs (%s). Spans "
                     "stay queued (oldest dropped when full) and are sent after the pause.", seconds, reason)

    def back_off(self, reason: str) -> float:
        """After a batch failed all its retries: wait longer each time until a send succeeds."""
        with self._lock:
            self._failures += 1
            failures = self._failures
        delay = min(0.5 * (2 ** (failures - 1)), MAX_BACKOFF_S) * (0.8 + 0.4 * random.random())
        self.hold(delay, "backing_off", reason)
        return delay

    def hold_remaining(self) -> float:
        return max(self._hold_until - time.monotonic(), 0.0)

    def export_state(self) -> str:
        """`off` (key refused), `paused` (Retry-After), `backing_off` (gateway failing) or `running`."""
        if self.fatal_reason:
            return "off"
        if self.hold_remaining() > 0:
            return self._hold_kind or "paused"
        return "running"

    def _succeeded(self) -> None:
        with self._lock:
            self._failures = 0
            if self._hold_kind == "backing_off":
                self._hold_until = 0.0

    # ------------------------------------------------------------------ connection

    def _route(self, url: str) -> tuple[tuple, str]:
        """(connection key, request target) for a URL, honouring HTTP(S)_PROXY and NO_PROXY."""
        parts = urllib.parse.urlsplit(url)
        scheme = parts.scheme.lower()
        host = parts.hostname or ""
        port = parts.port or (443 if scheme == "https" else 80)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        proxy = None
        proxies = urllib.request.getproxies_environment()   # never the macOS system lookup (fork crash)
        if proxies.get(scheme) and not urllib.request.proxy_bypass_environment(host, proxies):
            proxy = proxies[scheme]
        if proxy and scheme == "http":
            path = url   # absolute form through an HTTP proxy
        return (scheme, host, port, proxy), path

    def _new_connection(self, key: tuple) -> http.client.HTTPConnection:
        scheme, host, port, proxy = key
        tunnel_headers: dict[str, str] = {}
        target_host, target_port = host, port
        if proxy:
            if "://" not in proxy:
                proxy = "http://" + proxy
            p = urllib.parse.urlsplit(proxy)
            target_host, target_port = p.hostname or "", p.port or (443 if p.scheme == "https" else 80)
            if p.username:
                user = urllib.parse.unquote(p.username) + ":" + urllib.parse.unquote(p.password or "")
                tunnel_headers["Proxy-Authorization"] = "Basic " + base64.b64encode(user.encode()).decode()
        if scheme == "https":
            if self._ssl_context is None:
                self._ssl_context = ssl.create_default_context()
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                target_host, target_port, timeout=self.cfg.request_timeout, context=self._ssl_context)
            if proxy:
                conn.set_tunnel(host, port, headers=tunnel_headers)
        else:
            conn = http.client.HTTPConnection(target_host, target_port, timeout=self.cfg.request_timeout)
            conn._zoroval_proxy_headers = tunnel_headers if proxy else {}  # type: ignore[attr-defined]
        return conn

    def _acquire(self, key: tuple) -> tuple[http.client.HTTPConnection, bool]:
        """The persistent connection when free, else a one-off one (a second sender, such as logs)."""
        if not self._conn_lock.acquire(blocking=False):
            return self._new_connection(key), False
        if self._conn_pid != os.getpid():   # forked without the hook: drop the parent's socket unclosed
            self._conn, self._conn_pid = None, os.getpid()
        conn = self._conn
        if conn is not None and self._conn_key != key:
            conn.close()
            conn = None
        if conn is None:
            conn = self._new_connection(key)
            self._conn, self._conn_key = conn, key
        elif conn.sock is not None and _connection_dropped(conn.sock):
            conn.close()   # the server closed the idle connection; http.client reconnects on use
        return conn, True

    def _release(self, conn: http.client.HTTPConnection, owned: bool) -> None:
        if owned:
            self._conn_lock.release()
        else:
            conn.close()

    def _post_once(self, url: str, body: bytes, headers: dict[str, str],
                   timeout: float) -> tuple[int, Any, bytes]:
        """One POST on the persistent connection. Raises on network errors and timeouts."""
        key, target = self._route(url)
        conn, owned = self._acquire(key)
        try:
            for fresh_try in (False, True):
                reused = conn.sock is not None
                try:
                    started = time.monotonic()
                    conn.timeout = timeout
                    if conn.sock is None:
                        try:
                            conn.connect()
                        except TimeoutError:
                            # Not the body's fault: never a reason to split the batch.
                            raise ConnectTimeout("connect timeout") from None
                        self.connections_opened += 1
                    conn.sock.settimeout(timeout)
                    request_headers = dict(headers)
                    request_headers.update(getattr(conn, "_zoroval_proxy_headers", {}))
                    conn.request("POST", target, body=body, headers=request_headers)
                    left = timeout - (time.monotonic() - started)
                    if left <= 0:
                        raise TimeoutError("request timeout")
                    if conn.sock is not None:
                        conn.sock.settimeout(left)
                    response = conn.getresponse()
                    data = response.read()
                    if response.will_close:
                        conn.close()
                    return response.status, response.headers, data
                except _STALE_ERRORS:
                    conn.close()
                    # A kept-alive connection the server closed between our check and the write:
                    # resend once on a new connection. The gateway dedupes by batch id.
                    if not reused or fresh_try:
                        raise
                except BaseException:
                    conn.close()
                    raise
            raise ConnectionError("unreachable")
        finally:
            self._release(conn, owned)

    # ------------------------------------------------------------------ sending

    def send(self, url: str, body: bytes, *, deadline: float, kind: str = "traces",
             batch_id: Optional[str] = None, timeout: Optional[float] = None,
             split_on_timeout: bool = False) -> SendResult:
        """POST one compressed request with retries inside `deadline` (time.monotonic()).

        With `split_on_timeout`, a timeout or 408 returns at once (`timeout=True`) so the caller can
        send the batch in two halves; otherwise it is retried like a connection error.
        """
        if self.fatal_reason:
            return SendResult(False, reason=self.fatal_reason, fatal=True)
        batch_id = batch_id or uuid7()
        self.last_batch_id = batch_id
        per_request = timeout if timeout is not None else self.request_timeout_for(len(body))
        attempt = 0
        while True:
            held = self.hold_remaining()
            if held > 0:
                if time.monotonic() + held >= deadline:
                    return SendResult(False, reason=self._hold_reason or "paused", retryable=True, paused=True)
                if self.abort.wait(held):
                    return SendResult(False, reason="shutdown", retryable=True)
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0.05:
                return SendResult(False, reason="deadline", retryable=True)
            dropped = self._take_dropped()
            headers = {
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/x-protobuf",
                "Content-Encoding": "zstd",
                "User-Agent": USER_AGENT,
                "X-Zoroval-Batch-Id": batch_id,
            }
            headers.update(self.cfg.extra_headers)
            if dropped:
                headers["X-Zoroval-Dropped-Spans"] = str(dropped)
            status, reason, retry_after, response_body, timed_out = 0, "", None, b"", False
            try:
                status, response_headers, response_body = self._post_once(
                    url, body, headers, min(per_request, remaining))
                if not 200 <= status < 300:
                    reason = parse_status_message(response_body, response_headers.get("Content-Type", ""))
                    retry_after = response_headers.get("Retry-After")
            except TimeoutError:
                status, reason, timed_out = 0, "timeout", True
            except Exception as error:  # noqa: BLE001 - connection refused, DNS, TLS, reset
                status, reason = 0, f"connection:{type(error).__name__}"

            if 200 <= status < 300:
                self._ack_dropped(dropped)
                self._succeeded()
                rejected = self._rejected_count(response_body, kind)
                COUNTERS.add("bytes_sent", len(body))
                self.last_error = None
                return SendResult(True, status=status, rejected=rejected)

            self.last_error = f"{status}:{reason}" if status else reason
            if status in FATAL_STATUS:
                self.fatal_reason = reason or ("invalid_key" if status == 401 else "key_revoked")
                log_once("fatal-key",
                         "zoroval: the gateway refused the ingest key (%s). Export is off for this process. "
                         "Run `zoroval doctor` to check the key.", self.fatal_reason)
                return SendResult(False, status=status, reason=self.fatal_reason, fatal=True)
            timed_out = timed_out or status == TIMEOUT_STATUS
            if status and not timed_out and status not in RETRY_AFTER_STATUS and status not in RETRY_STATUS:
                log_once(f"status:{status}", "zoroval: the gateway refused a batch (%s %s).", status, reason)
                return SendResult(False, status=status, reason=reason)
            if timed_out and split_on_timeout:
                return SendResult(False, status=status, reason=reason or "timeout", retryable=True, timeout=True)
            if status in RETRY_AFTER_STATUS:
                wait = _parse_retry_after(retry_after)
                if wait is not None:
                    # The gateway said when: every sender waits, and a wait past this batch's
                    # deadline (or its last attempt) pauses export instead of failing the batch.
                    if attempt >= self.cfg.max_attempts or time.monotonic() + wait >= deadline:
                        # At least a second, so a gateway answering `Retry-After: 0` cannot cause a loop.
                        self.hold(max(wait, MIN_PAUSE_S), "paused", reason or str(status))
                        return SendResult(False, status=status, reason=reason or "paused",
                                          retryable=True, paused=True)
                    self.hold(wait, "paused", reason or str(status))
                    continue
            if attempt >= self.cfg.max_attempts:
                return SendResult(False, status=status, reason=reason or "retries_exhausted", retryable=True)
            delay = self._backoff(attempt, None)
            if time.monotonic() + delay >= deadline:
                return SendResult(False, status=status, reason=reason or "deadline", retryable=True)
            logger.debug("zoroval: retrying batch %s in %.2fs after %s", batch_id, delay, self.last_error)
            if self.abort.wait(delay):
                return SendResult(False, status=status, reason="shutdown", retryable=True)

    @staticmethod
    def _backoff(attempt: int, retry_after: Optional[str]) -> float:
        wait = _parse_retry_after(retry_after)
        if wait is not None:
            return wait
        base = min(0.5 * (2 ** (attempt - 1)), 8.0)
        return base * (0.8 + 0.4 * random.random())

    @staticmethod
    def _rejected_count(body: bytes, kind: str) -> int:
        if not body:
            return 0
        try:
            if kind == "logs":
                from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceResponse
                msg = ExportLogsServiceResponse()
                msg.ParseFromString(body)
                return int(msg.partial_success.rejected_log_records)
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse
            msg = ExportTraceServiceResponse()
            msg.ParseFromString(body)
            return int(msg.partial_success.rejected_spans)
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------ status (doctor)

    def get_json(self, path: str, *, timeout: float = 10.0) -> tuple[int, Any]:
        url = self.cfg.ingest_url.rstrip("/") + path
        request = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self.cfg.api_key}", "User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with self._opener(request, timeout=timeout) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            body = error.read() if hasattr(error, "read") else b""
            try:
                return error.code, json.loads(body or b"{}")
            except ValueError:
                return error.code, {"message": parse_status_message(body, "")}
        except Exception as error:  # noqa: BLE001
            return 0, {"message": f"connection:{type(error).__name__}"}
