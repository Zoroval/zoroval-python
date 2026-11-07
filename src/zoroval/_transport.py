"""OTLP/HTTP protobuf transport with zstd, built for the Zoroval gateway (contracts 5.1 and 6).

- Encodes spans with the OpenTelemetry protobuf encoder, then adds Zoroval's resource and span
  attributes on the protobuf message (no span copies on the request path).
- Compresses with zstd level 3. Splits a batch whose compressed size passes the wire limit.
- One `X-Zoroval-Batch-Id` (UUIDv7) per batch, reused on its retries.
- Retries 429 and 503 after `Retry-After`, and 5xx and connection errors with backoff, always
  inside the caller's deadline. 401 and 403 stop all export for the process.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
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
WIRE_MARGIN = 64 * 1024


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


@dataclass
class SendResult:
    ok: bool
    status: int = 0
    reason: str = ""
    rejected: int = 0
    fatal: bool = False


class Transport:
    """Sends encoded OTLP requests. Thread-safe enough for one export thread plus flush callers."""

    def __init__(self, cfg: Config, *, opener: Optional[Callable[..., Any]] = None) -> None:
        self.cfg = cfg
        self._zstd = zstandard.ZstdCompressor(level=3, write_content_size=True)
        self._opener = opener or urllib.request.urlopen
        self._dropped_unreported = 0
        self._lock = threading.Lock()
        self.fatal_reason: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_batch_id: Optional[str] = None
        self.abort = threading.Event()   # set by shutdown to stop retry sleeps at once

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

    # ------------------------------------------------------------------ sending

    def send(self, url: str, body: bytes, *, deadline: float, kind: str = "traces",
             batch_id: Optional[str] = None) -> SendResult:
        """POST one compressed request with retries inside `deadline` (time.monotonic())."""
        if self.fatal_reason:
            return SendResult(False, reason=self.fatal_reason, fatal=True)
        batch_id = batch_id or uuid7()
        self.last_batch_id = batch_id
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0.05:
                return SendResult(False, reason="deadline")
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
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            status, reason, retry_after, response_body = 0, "", None, b""
            try:
                with self._opener(request, timeout=min(self.cfg.request_timeout, remaining)) as response:
                    status = response.status
                    response_body = response.read()
            except urllib.error.HTTPError as error:
                status = error.code
                try:
                    response_body = error.read()
                except Exception:  # noqa: BLE001
                    response_body = b""
                reason = parse_status_message(response_body, error.headers.get("Content-Type", "") if error.headers else "")
                retry_after = error.headers.get("Retry-After") if error.headers else None
            except Exception as error:  # noqa: BLE001 - connection refused, timeout, DNS, TLS
                status, reason = 0, f"connection:{type(error).__name__}"

            if 200 <= status < 300:
                self._ack_dropped(dropped)
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
            if status and status not in RETRY_AFTER_STATUS and status not in RETRY_STATUS:
                log_once(f"status:{status}", "zoroval: the gateway refused a batch (%s %s).", status, reason)
                return SendResult(False, status=status, reason=reason)
            if attempt >= self.cfg.max_attempts:
                return SendResult(False, status=status, reason=reason or "retries_exhausted")
            delay = self._backoff(attempt, retry_after)
            if time.monotonic() + delay >= deadline:
                return SendResult(False, status=status, reason=reason or "deadline")
            logger.debug("zoroval: retrying batch %s in %.2fs after %s", batch_id, delay, self.last_error)
            if self.abort.wait(delay):
                return SendResult(False, status=status, reason="shutdown")

    @staticmethod
    def _backoff(attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 60.0)
            except ValueError:
                pass
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
