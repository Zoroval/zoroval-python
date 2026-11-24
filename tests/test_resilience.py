"""Captive portals, redirects, and long outages must not lose spans silently."""

import time
from collections import deque

import pytest
from opentelemetry.sdk.trace import TracerProvider

from zoroval import _processor
from zoroval._config import Config
from zoroval._processor import ZorovalSpanProcessor, _Batch
from zoroval._transport import CAPTIVE_REASON, Transport


def _transport(monkeypatch, status, headers, body):
    cfg = Config(api_key="zv_test", max_attempts=2)
    transport = Transport(cfg)
    monkeypatch.setattr(transport, "_post_once", lambda *a, **k: (status, headers, body))
    monkeypatch.setattr(transport, "_backoff", lambda attempt, retry_after: 0.0)
    return transport


def _send(transport):
    return transport.send(transport.cfg.traces_url, b"x", deadline=time.monotonic() + 5)


@pytest.mark.parametrize("headers,body", [
    ({"Content-Type": "text/html; charset=utf-8"}, b"<html>Please sign in</html>"),
    ({"Content-Type": "text/plain"}, b"ok"),
    ({}, b"<html>Please sign in</html>"),
])
def test_captive_portal_200_is_retryable_not_success(monkeypatch, headers, body):
    result = _send(_transport(monkeypatch, 200, headers, body))
    assert not result.ok
    assert result.retryable
    assert result.reason == CAPTIVE_REASON


@pytest.mark.parametrize("status", [300, 301, 302, 307, 308])
def test_redirect_is_retryable(monkeypatch, status):
    result = _send(_transport(monkeypatch, status, {"Location": "http://login.example/"}, b""))
    assert not result.ok
    assert result.retryable
    assert result.reason == CAPTIVE_REASON


@pytest.mark.parametrize("status,headers,body", [
    (200, {}, b""),
    (204, {}, b""),
    (200, {"Content-Type": "application/x-protobuf"}, b""),
    (200, {"Content-Type": "application/json"}, b"{}"),
    (200, {"Content-Type": "application/x-protobuf"}, b"\xff\xff garbage"),
])
def test_real_gateway_answers_stay_successful(monkeypatch, status, headers, body):
    result = _send(_transport(monkeypatch, status, headers, body))
    assert result.ok
    assert result.rejected == 0


def test_partial_success_still_counts_rejected(monkeypatch):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse
    msg = ExportTraceServiceResponse()
    msg.partial_success.rejected_spans = 3
    result = _send(_transport(monkeypatch, 200, {"Content-Type": "application/x-protobuf"},
                              msg.SerializeToString()))
    assert result.ok and result.rejected == 3


def test_other_4xx_is_still_a_permanent_refusal(monkeypatch):
    result = _send(_transport(monkeypatch, 400, {}, b""))
    assert not result.ok and not result.retryable


def _processor_with_batch(monkeypatch, age):
    cfg = Config(api_key="zv_test", max_attempts=1)
    transport = Transport(cfg)
    monkeypatch.setattr(transport, "_post_once", lambda *a, **k: (302, {}, b""))
    monkeypatch.setattr(transport, "back_off", lambda *a, **k: 0.0)
    processor = ZorovalSpanProcessor(cfg, transport)
    span = TracerProvider().get_tracer("t").start_span("s")
    span.end()
    batch = _Batch(deque([(span, None, 200)]), "b", time.monotonic() - age)
    return processor, batch


def test_outage_keeps_batch_past_old_requeue_limit(monkeypatch):
    processor, batch = _processor_with_batch(monkeypatch, age=0)
    try:
        for _ in range(25):   # the old cap was 10 requeues
            assert processor._export(batch) == 1
            processor._retry.clear()
            processor._retry_spans = 0
            processor._queue_bytes = 0
    finally:
        processor.shutdown()


def test_batch_older_than_max_age_is_dropped(monkeypatch):
    processor, batch = _processor_with_batch(monkeypatch, age=_processor._MAX_BATCH_AGE_S + 1)
    try:
        assert processor._export(batch) == 0
    finally:
        processor.shutdown()
