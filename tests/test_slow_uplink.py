"""Slow uplinks and gateway trouble: split on an upload timeout (never on a slow answer), requeue
after an outage, pause on a long Retry-After, one kept-alive connection, 408 retry, and shutdown
bounded while paused."""

from __future__ import annotations

import http.client
import os
import socket
import time
from typing import Callable

import pytest

import zoroval
from tests.conftest import flush
from testing.fake_gateway import FakeGateway
from zoroval import _config, _processor, _transport
from zoroval._transport import Transport


def _make_spans(count: int, content_bytes: int = 0, prefix: str = "s") -> None:
    for index in range(count):
        with zoroval.step(f"{prefix}{index}") as obs:
            if content_bytes:
                obs.set_output(os.urandom(content_bytes // 2).hex())


def _wait_for(check: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _shape_uplink(monkeypatch: pytest.MonkeyPatch, bytes_per_s: float) -> None:
    """A slow client uplink. Loopback sockets buffer hundreds of KB whatever SO_SNDBUF and SO_RCVBUF
    say, so a slow reader alone cannot make the client's upload slow: throttle the client's writes,
    and time out like a socket would."""
    original = http.client.HTTPConnection.send

    def send(self: http.client.HTTPConnection, data: object) -> None:
        if not isinstance(data, (bytes, bytearray)) or len(data) < 65536:
            return original(self, data)
        timeout = self.sock.gettimeout() if self.sock is not None else None
        started = time.monotonic()
        view = memoryview(data)
        for offset in range(0, len(view), 16384):
            if timeout is not None and time.monotonic() - started > timeout:
                raise TimeoutError("timed out")
            original(self, view[offset:offset + 16384])
            time.sleep(16384 / bytes_per_s)
        return None

    monkeypatch.setattr(http.client.HTTPConnection, "send", send)


def _batch_ids(gateway: FakeGateway) -> set:
    return {r["headers"].get("X-Zoroval-Batch-Id") for r in gateway.state.requests}


def test_request_timeout_scales_with_body_size():
    transport = Transport(_config.Config(api_key="x"))
    assert transport.request_timeout_for(0) == 10.0
    assert transport.request_timeout_for(1024 * 1024) == pytest.approx(10.0 + 16.0)
    assert transport.request_timeout_for(16 * 1024 * 1024) == _transport.TIMEOUT_CEILING_S
    slow = Transport(_config.Config(api_key="x", request_timeout=90.0))
    assert slow.request_timeout_for(16 * 1024 * 1024) == 90.0   # the configured floor always wins
    assert _config.Config().max_batch_bytes == 4 * 1024 * 1024


def test_slow_uplink_timeouts_split_the_batch(start_sdk, gateway, monkeypatch):
    # The body goes up at 300 KB/s; the request timeout is 1 s whatever the size. The full batch
    # (about 800 KB on the wire) and its halves cannot arrive in time; quarters can.
    monkeypatch.setattr(_transport, "MIN_UPLINK_BYTES_PER_S", 10 ** 12)
    _shape_uplink(monkeypatch, 300_000)
    gateway.state.uplink_bytes_per_s = 300_000
    start_sdk(request_timeout=1.0, flush_interval=5.0, max_attempts=2)
    _make_spans(8, 200_000)
    assert zoroval.flush(30.0)
    status = zoroval.status()
    assert sorted(s["name"] for s in gateway.state.spans) == sorted(f"s{i}" for i in range(8))
    assert status["spans_failed"] == 0 and status["spans_exported"] == 8
    assert status["batches_split"] >= 2
    assert gateway.state.abandoned >= 2      # the oversized uploads were reset mid-body, not stored
    assert all(r["wire_bytes"] < 300_000 for r in gateway.state.requests)
    assert len(gateway.state.spans) == 8     # nothing stored twice


def test_slow_answer_after_full_upload_is_resent_under_the_same_id(start_sdk, gateway, monkeypatch):
    # The gateway got the whole body (and stores it) but answers late: never split, never a new id.
    monkeypatch.setattr(_transport, "MIN_UPLINK_BYTES_PER_S", 10 ** 12)
    gateway.state.delay_s = 1.5
    start_sdk(request_timeout=1.0, flush_interval=5.0, max_attempts=2)
    _make_spans(16)
    assert _wait_for(lambda: len(gateway.state.requests) >= 2, 10.0)
    gateway.state.delay_s = 0
    assert zoroval.flush(15.0)
    status = zoroval.status()
    assert status["batches_split"] == 0
    assert len(_batch_ids(gateway)) == 1
    names = [s["name"] for s in gateway.state.spans]
    assert sorted(names) == sorted(f"s{i}" for i in range(16))   # stored once: the gateway skipped resends
    assert status["spans_failed"] == 0


def test_late_answer_with_unacknowledged_bytes_is_an_upload_timeout(start_sdk, gateway, monkeypatch):
    # sendall returns once the body is in kernel buffers. If the kernel still holds unacknowledged
    # bytes when the wait runs out, the upload is what is slow: split.
    monkeypatch.setattr(_transport, "MIN_UPLINK_BYTES_PER_S", 10 ** 12)
    monkeypatch.setattr(_transport, "_unacked_bytes", lambda sock: 4096)
    gateway.state.delay_s = 1.5
    start_sdk(request_timeout=1.0, flush_interval=5.0, max_attempts=2)
    _make_spans(4)
    zoroval.flush(0.01)   # send now, as one batch
    assert _wait_for(lambda: zoroval.status()["batches_split"] >= 1, 5.0)


def test_unacked_bytes_is_known_on_linux_and_macos(gateway):
    import sys
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        pytest.skip("no kernel send-queue query on this platform")
    host, port = gateway.server.server_address[:2]
    with socket.create_connection((host, port)) as sock:
        assert _transport._unacked_bytes(sock) == 0


def test_attempt_cut_short_by_the_deadline_does_not_split(gateway, monkeypatch):
    _shape_uplink(monkeypatch, 300_000)
    transport = Transport(_config.Config(api_key=gateway.issue_key(), ingest_url=gateway.url))
    body = os.urandom(400_000)   # needs about 1.3 s; the per-request timeout allows it, the deadline not
    result = transport.send(transport.cfg.traces_url, body, deadline=time.monotonic() + 0.5,
                            timeout=5.0, split_on_timeout=True)
    assert not result.ok and not result.timeout and result.retryable


def test_outage_longer_than_retry_budget_requeues_then_delivers(gateway, monkeypatch):
    monkeypatch.setattr(_processor, "_BATCH_BUDGET_S", 0.5)
    key = gateway.issue_key()
    port = _free_port()   # nothing listens here yet: connection refused
    assert zoroval.init(api_key=key, ingest_url=f"http://127.0.0.1:{port}", service_name="test-app",
                        flush_interval=0.05, max_attempts=2)
    _make_spans(5)
    assert _wait_for(lambda: zoroval.status()["batches_requeued"] >= 2, 10.0)
    assert zoroval.status()["spans_failed"] == 0
    assert zoroval.status()["queue_spans"] == 5
    with FakeGateway(port) as back:
        back.state.keys.update(gateway.state.keys)
        assert zoroval.flush(15.0)
        assert sorted(s["name"] for s in back.state.spans) == [f"s{i}" for i in range(5)]
        status = zoroval.status()
        assert status["spans_failed"] == 0 and status["spans_exported"] == 5
        assert status["export_state"] == "running"
        zoroval.shutdown(timeout=1.0)


def test_requeue_is_capped(start_sdk, gateway, monkeypatch):
    monkeypatch.setattr(_processor, "_BATCH_BUDGET_S", 0.2)
    monkeypatch.setattr(_processor, "_MAX_REQUEUES", 2)
    gateway.state.fail_next = [500] * 100
    start_sdk(max_attempts=1)
    _make_spans(3)
    assert _wait_for(lambda: zoroval.status()["spans_failed"] == 3, 10.0)
    status = zoroval.status()
    assert status["batches_requeued"] == 2 and status["batches_failed"] == 1
    assert len(gateway.state.requests) == 3   # first try plus two requeues, then given up
    assert status["queue_spans"] == 0


def test_long_retry_after_pauses_export_without_extra_requests(start_sdk, gateway, monkeypatch):
    monkeypatch.setattr(_processor, "_BATCH_BUDGET_S", 0.5)
    gateway.state.fail_next = [429]
    gateway.state.retry_after = "3"
    start_sdk(max_batch_spans=4)
    _make_spans(3, prefix="a")
    # the fake records the request before it answers 429, so wait for the client to take the answer
    assert _wait_for(lambda: zoroval.status()["export_state"] == "paused", 3.0)
    assert len(gateway.state.requests) == 1
    paused_at = gateway.state.requests[0]["at"]
    assert zoroval.status()["export_resumes_in_s"] > 1.5
    _make_spans(10, prefix="b")   # several batches' worth queued during the pause
    assert zoroval.flush(1.0) is False   # honest: nothing can be sent yet
    time.sleep(0.5)
    assert len(gateway.state.requests) == 1
    assert zoroval.flush(8.0)
    times = [r["at"] for r in gateway.state.requests]
    assert min(times[1:]) - paused_at >= 2.9   # nothing sent before Retry-After passed
    assert len(times) <= 5   # the paused batch once more, then the rest in batches of 4
    names = sorted(s["name"] for s in gateway.state.spans)
    assert names == sorted([f"a{i}" for i in range(3)] + [f"b{i}" for i in range(10)])
    assert zoroval.status()["spans_failed"] == 0


def test_keepalive_one_connection_for_several_batches(start_sdk, gateway):
    start_sdk()
    for index in range(4):
        _make_spans(1, prefix=f"r{index}-")
        flush()
    ports = {r["client_port"] for r in gateway.state.requests}
    assert len(gateway.state.requests) == 4 and len(ports) == 1


def test_keepalive_reconnects_after_server_closes_idle_connection(start_sdk, gateway):
    gateway.state.idle_timeout_s = 0.3   # like a proxy or uvicorn keep-alive timeout
    start_sdk(max_attempts=1)   # a write on the closed connection must not cost an attempt
    for index in range(3):
        _make_spans(1, prefix=f"r{index}-")
        flush()
        time.sleep(0.6)   # the server has closed the idle connection by now
    assert sorted(s["name"] for s in gateway.state.spans) == ["r0-0", "r1-0", "r2-0"]
    assert len(gateway.state.requests) == 3   # no lost or failed request on a closed connection
    assert len({r["client_port"] for r in gateway.state.requests}) == 3
    status = zoroval.status()
    assert status["spans_failed"] == 0 and status["batches_requeued"] == 0


def test_408_is_retried_like_a_timeout(start_sdk, gateway):
    gateway.state.fail_next = [408]
    start_sdk()
    _make_spans(1)
    flush()
    ids = [r["headers"].get("X-Zoroval-Batch-Id") for r in gateway.state.requests]
    assert len(ids) == 2 and len(set(ids)) == 1
    assert [s["name"] for s in gateway.state.spans] == ["s0"]


def test_408_splits_a_multi_span_batch(start_sdk, gateway):
    gateway.state.fail_next = [408]
    start_sdk(flush_interval=5.0)
    _make_spans(4)
    flush()
    assert len(gateway.state.spans) == 4
    assert zoroval.status()["batches_split"] == 1


def test_shutdown_does_not_wait_out_a_pause(start_sdk, gateway):
    gateway.state.fail_next = [503]
    gateway.state.retry_after = "60"
    start_sdk()
    _make_spans(3)
    assert _wait_for(lambda: zoroval.status()["export_state"] == "paused", 3.0)
    _make_spans(3)
    started = time.monotonic()
    zoroval.shutdown(timeout=2.0)
    assert time.monotonic() - started < 0.5
    assert len(gateway.state.requests) == 1


def test_nan_retry_after_does_not_loop(start_sdk, gateway):
    gateway.state.fail_next = [429] * 100_000
    gateway.state.retry_after = "nan"
    start_sdk()
    _make_spans(1)
    time.sleep(1.0)
    count = len(gateway.state.requests)
    gateway.state.fail_next = []
    assert count <= 4, count   # normal backoff: 0.5 s, then 1 s


def test_zero_retry_after_waits_like_backoff_and_pauses_grow(start_sdk, gateway):
    gateway.state.fail_next = [503] * 100_000
    gateway.state.retry_after = "0"
    start_sdk(max_attempts=2)
    _make_spans(1)
    time.sleep(4.0)
    count = len(gateway.state.requests)
    gateway.state.fail_next = []
    # attempts at 0 and about 0.5 s, then pauses of at least 1 s, 1 s, 2 s...: never a burst
    assert 3 <= count <= 6, count
    gaps = [b["at"] - a["at"] for a, b in zip(gateway.state.requests, gateway.state.requests[1:])]
    assert min(gaps) >= 0.35


def _gen_ai_log_provider():
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider
    set_logger_provider(LoggerProvider())


def _emit_gen_ai_log() -> None:
    from opentelemetry._logs import LogRecord, get_logger
    get_logger("opentelemetry.instrumentation.openai_v2").emit(
        LogRecord(event_name="gen_ai.user.message", body={"content": "hello"}))


def test_shutdown_with_logs_does_not_wait_out_a_pause(start_sdk, gateway, monkeypatch):
    # Retry-After past the batch budget (15 s here) puts the trace thread in `paused`, so it exits at
    # once on shutdown; the log exporter's own 30 s deadline would still wait the 20 s out.
    monkeypatch.setattr(_processor, "_BATCH_BUDGET_S", 1.0)
    _gen_ai_log_provider()
    gateway.state.fail_next = [503]
    gateway.state.retry_after = "20"
    start_sdk()
    _make_spans(1)
    assert _wait_for(lambda: zoroval.status()["export_state"] == "paused", 3.0)
    _emit_gen_ai_log()
    started = time.monotonic()
    zoroval.shutdown(timeout=1.0)
    assert time.monotonic() - started < 1.6


def test_shutdown_with_logs_does_not_wait_out_a_backoff(start_sdk, gateway):
    _gen_ai_log_provider()
    gateway.state.fail_next = [500] * 1000
    start_sdk(max_attempts=1)
    _make_spans(1)
    assert _wait_for(lambda: zoroval.status()["batches_requeued"] >= 4, 10.0)
    assert zoroval.status()["export_resumes_in_s"] > 1.0
    _emit_gen_ai_log()
    started = time.monotonic()
    zoroval.shutdown(timeout=1.0)
    assert time.monotonic() - started < 1.6


def test_idle_connection_check_works_for_descriptors_over_1024(gateway):
    fcntl = pytest.importorskip("fcntl")
    resource = pytest.importorskip("resource")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 1200:
        if hard != resource.RLIM_INFINITY and hard < 1200:
            pytest.skip("cannot raise the descriptor limit")
        resource.setrlimit(resource.RLIMIT_NOFILE, (1200, hard))
    try:
        host, port = gateway.server.server_address[:2]
        with socket.create_connection((host, port)) as low:
            high = socket.socket(fileno=fcntl.fcntl(low.fileno(), fcntl.F_DUPFD, 1100))
            try:
                assert high.fileno() >= 1024
                assert _transport._connection_dropped(high) is False   # idle and open: keep it
            finally:
                high.close()
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_close_never_closes_a_connection_another_sender_is_using(gateway):
    transport = Transport(_config.Config(api_key="x", ingest_url=gateway.url))
    key, _ = transport._route(transport.cfg.traces_url)
    conn, owned = transport._acquire(key)   # a sender is mid-request on the kept-alive connection
    assert owned
    conn.connect()
    started = time.monotonic()
    transport.close()
    assert time.monotonic() - started < 0.5   # never blocks the caller for long
    assert conn.sock is not None              # still open for the sender using it
    transport._release(conn, owned)
    assert conn.sock is None and transport._conn is None   # closed once the sender is done
