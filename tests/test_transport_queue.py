"""Transport and queue: zstd, headers, retries, key refusal, partial success, batches cut by bytes,
the queue byte cap, drop counts, bounded flush and shutdown."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import time
import uuid

import pytest
import zstandard

import zoroval
from tests.conftest import ROOT, flush
from zoroval import _config
from zoroval._transport import Transport, parse_status_message, uuid7


def _make_spans(count: int, content_bytes: int = 0, *, random_content: bool = False) -> None:
    for index in range(count):
        with zoroval.step(f"s{index}") as obs:
            if content_bytes:
                payload = os.urandom(content_bytes // 2).hex() if random_content else "x" * content_bytes
                obs.set_output(payload)


@pytest.fixture
def silent_endpoint():
    """A TCP port that accepts connections and never answers."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    sock.close()


def _free_port_url() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


def test_uuid7_format():
    ids = [uuid7() for _ in range(50)]
    for value in ids:
        parsed = uuid.UUID(value)
        assert parsed.version == 7
        assert parsed.variant == uuid.RFC_4122
    assert len(set(ids)) == 50
    assert ids == sorted(ids) or ids[0][:8] <= ids[-1][:8]


def test_zstd_frame_has_content_size():
    transport = Transport(_config.Config(api_key="x"))
    raw = b"hello " * 1000
    body = transport.compress(raw)
    assert zstandard.get_frame_parameters(body).content_size == len(raw)
    assert zstandard.ZstdDecompressor().decompress(body) == raw


def test_status_message_parsing():
    from testing.fake_gateway import _status_body
    assert parse_status_message(_status_body("daily_cap", 8, ""), "application/x-protobuf") == "daily_cap"
    assert parse_status_message(b'{"code":8,"message":"rate_limited"}', "application/json") == "rate_limited"
    assert parse_status_message(b"\xff\xff", "") == ""


def test_request_headers_and_resource(start_sdk, gateway):
    key = start_sdk(environment="staging", release="1.2.3")
    with zoroval.turn():
        pass
    flush()
    request = gateway.state.requests[0]
    headers = {k.lower(): v for k, v in request["headers"].items()}
    assert headers["authorization"] == f"Bearer {key}"
    assert headers["content-type"] == "application/x-protobuf"
    assert headers["content-encoding"] == "zstd"
    assert headers["user-agent"] == f"zoroval-python/{zoroval.__version__}"
    assert uuid.UUID(headers["x-zoroval-batch-id"]).version == 7
    assert "x-zoroval-dropped-spans" not in headers
    resource = gateway.state.spans[0]["resource"]
    assert resource["service.name"] == "test-app"
    assert resource["zoroval.sdk.name"] == "zoroval-python"
    assert resource["zoroval.sdk.version"] == zoroval.__version__
    assert resource["deployment.environment.name"] == "staging"


def test_retry_keeps_batch_id_and_stores_once(start_sdk, gateway):
    gateway.state.fail_next = [503, 429]
    start_sdk()
    _make_spans(3)
    flush()
    ids = [r["headers"].get("X-Zoroval-Batch-Id") for r in gateway.state.requests]
    assert len(ids) == 3 and len(set(ids)) == 1
    assert len(gateway.state.spans) == 3
    assert zoroval.status()["spans_exported"] == 3


def test_resent_batch_id_is_deduplicated_by_gateway(gateway):
    key = gateway.issue_key()
    transport = Transport(_config.Config(api_key=key, ingest_url=gateway.url))
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("zoroval").start_as_current_span("x", attributes={"zoroval.observation.type": "trace"}):
        pass
    body = transport.compress(transport.encode_spans(exporter.get_finished_spans(), {}))
    batch = uuid7()
    assert transport.send(gateway.url + "/v1/traces", body, deadline=time.monotonic() + 5, batch_id=batch).ok
    assert transport.send(gateway.url + "/v1/traces", body, deadline=time.monotonic() + 5, batch_id=batch).ok
    assert len(gateway.state.spans) == 1


def test_invalid_key_stops_export(start_sdk, gateway):
    fake = "zv_ingest_aaaaaaaaaaaa_" + "A" * 43  # right format, unknown to the gateway
    start_sdk(api_key=fake)
    _make_spans(2)
    zoroval.flush(2.0)
    status = zoroval.status()
    assert status["key_refused"] == "invalid_key"
    assert status["spans_failed"] >= 2
    sent = len(gateway.state.requests)
    _make_spans(2)
    zoroval.flush(1.0)
    assert len(gateway.state.requests) == sent  # no more requests after a refused key


def test_revoked_key(start_sdk, gateway):
    key = start_sdk()
    gateway.revoke(key)
    _make_spans(1)
    zoroval.flush(2.0)
    assert zoroval.status()["key_refused"] == "key_revoked"


def test_partial_success_counts_rejected(start_sdk, gateway):
    gateway.state.max_span_bytes = 2000
    start_sdk()
    with zoroval.step("small"):
        pass
    with zoroval.step("big") as obs:
        obs.set_output("y" * 5000)
    flush()
    status = zoroval.status()
    assert status["spans_rejected"] == 1
    assert status["spans_exported"] == 1


def test_batches_cut_by_bytes(start_sdk, gateway):
    start_sdk(max_batch_bytes=1024 * 1024, flush_interval=5.0)
    _make_spans(10, 300_000)
    flush()
    raw_sizes = [r["raw_bytes"] for r in gateway.state.requests]
    assert len(raw_sizes) >= 3
    assert max(raw_sizes) <= 1024 * 1024 + 400_000
    assert len(gateway.state.spans) == 10


def test_compressed_size_over_wire_limit_splits(start_sdk, gateway):
    start_sdk(max_request_bytes=400_000, flush_interval=5.0)
    _make_spans(10, 150_000, random_content=True)
    flush()
    assert all(r["wire_bytes"] <= 400_000 for r in gateway.state.requests)
    assert len(gateway.state.requests) >= 3
    assert len(gateway.state.spans) == 10


def test_queue_is_capped_in_bytes_and_counts_drops(start_sdk, silent_endpoint):
    cap = 1024 * 1024
    start_sdk(ingest_url=silent_endpoint, max_queue_bytes=cap, max_batch_bytes=256 * 1024, request_timeout=30)
    peak = 0
    for _ in range(60):
        _make_spans(1, 100_000)
        peak = max(peak, zoroval.status()["queue_bytes"])
    status = zoroval.status()
    assert peak <= cap
    assert status["spans_dropped"] >= 40
    assert status["spans_queued"] == 60


def test_queue_span_cap(start_sdk, silent_endpoint):
    start_sdk(ingest_url=silent_endpoint, max_queue_spans=20, request_timeout=30)
    _make_spans(100)
    status = zoroval.status()
    assert status["queue_spans"] <= 20
    assert status["spans_dropped"] >= 70


def test_drop_count_is_reported_in_header(start_sdk, gateway):
    gateway.state.delay_s = 0.3
    start_sdk(max_queue_spans=5, flush_interval=0.01)
    _make_spans(1)
    time.sleep(0.05)  # the worker is now inside a slow request
    _make_spans(40)
    gateway.state.delay_s = 0
    flush()
    dropped = zoroval.status()["spans_dropped"]
    assert dropped > 0
    reported = sum(int(r["headers"].get("X-Zoroval-Dropped-Spans") or 0) for r in gateway.state.requests)
    assert reported == dropped


def test_shutdown_is_bounded_when_endpoint_hangs(start_sdk, silent_endpoint):
    start_sdk(ingest_url=silent_endpoint, request_timeout=30, flush_interval=0.01)
    _make_spans(20, 1000)
    time.sleep(0.1)  # a request is now in flight and will never be answered
    started = time.monotonic()
    zoroval.shutdown(timeout=1.0)
    assert time.monotonic() - started < 1.6


def test_shutdown_is_bounded_when_endpoint_is_down(start_sdk):
    start_sdk(ingest_url=_free_port_url(), flush_interval=0.01, max_attempts=50)
    _make_spans(20, 1000)
    started = time.monotonic()
    zoroval.shutdown(timeout=1.0)
    assert time.monotonic() - started < 1.6


def test_flush_returns_false_on_timeout_without_blocking(start_sdk, silent_endpoint):
    start_sdk(ingest_url=silent_endpoint, request_timeout=30)
    _make_spans(5)
    started = time.monotonic()
    assert zoroval.flush(0.5) is False
    assert time.monotonic() - started < 1.0


def test_process_exit_is_bounded_with_hanging_endpoint(silent_endpoint, gateway):
    key = gateway.issue_key()
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(ROOT)!r})
        import zoroval
        zoroval.init(api_key={key!r}, ingest_url={silent_endpoint!r}, shutdown_timeout=1.0,
                     request_timeout=30, flush_interval=0.01)
        for i in range(50):
            with zoroval.step("s"):
                pass
        import time; time.sleep(0.1)
    """)
    started = time.monotonic()
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
    assert time.monotonic() - started < 6.0  # interpreter start plus the 1 s cap, well under the PoC's 30 to 41 s


def test_atexit_flushes_on_normal_exit(gateway):
    key = gateway.issue_key()
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(ROOT)!r})
        import zoroval
        zoroval.init(api_key={key!r}, ingest_url={gateway.url!r}, flush_interval=60)
        with zoroval.trace("job"):
            pass
    """)
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
    assert len(gateway.spans(**{"zoroval.observation.type": "trace"})) == 1
