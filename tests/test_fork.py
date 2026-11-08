"""Fork safety: pre-fork servers (gunicorn --preload, uvicorn workers) fork after init().

The child must restart the export thread, must not resend the parent's queued spans, and the
parent must keep exporting.
"""

from __future__ import annotations

import multiprocessing
import os
import sys

import pytest

import zoroval
from tests.conftest import flush

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


def _span(name: str) -> None:
    with zoroval.trace(name):
        pass


def test_child_exports_its_own_spans_and_not_the_parents(start_sdk, gateway):
    start_sdk(flush_interval=30.0)  # keep the parent's span in its queue across the fork
    _span("parent-before-fork")
    assert zoroval.status()["queue_spans"] == 1
    pid = os.fork()
    if pid == 0:  # child
        code = 1
        try:
            queued_at_start = zoroval.status()["queue_spans"]
            _span("child")
            ok = zoroval.flush(5.0)
            code = 0 if ok and queued_at_start == 0 else 3
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
    _span("parent-after-fork")
    flush()
    names = sorted(s["name"] for s in gateway.state.spans)
    assert names == ["child", "parent-after-fork", "parent-before-fork"]


def _worker(index: int) -> int:
    _span(f"worker-{index}")
    return 0 if zoroval.flush(5.0) else 1


def test_prefork_pool_workers_all_export(start_sdk, gateway):
    if sys.platform == "win32":
        pytest.skip("fork start method")
    start_sdk()
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(3) as pool:
        results = pool.map(_worker, range(6))
    assert results == [0] * 6
    names = sorted(s["name"] for s in gateway.state.spans)
    assert names == [f"worker-{i}" for i in range(6)]


def test_fork_without_flush_does_not_crash_child(start_sdk, gateway):
    start_sdk()
    pid = os.fork()
    if pid == 0:
        try:
            _span("child-no-flush")
            zoroval.shutdown(timeout=2.0)
        finally:
            os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
    flush()
    assert [s["name"] for s in gateway.state.spans] == ["child-no-flush"]


def test_child_never_reuses_the_parents_kept_alive_connection(start_sdk, gateway):
    start_sdk()
    _span("parent-1")
    flush()   # the parent now holds an open kept-alive connection
    parent_port = gateway.state.requests[0]["client_port"]
    pid = os.fork()
    if pid == 0:  # child
        code = 1
        try:
            _span("child")
            code = 0 if zoroval.flush(5.0) else 3
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
    _span("parent-2")
    flush()
    port_of = {}
    for request in gateway.state.requests:
        batch = request["headers"].get("X-Zoroval-Batch-Id")
        for span in gateway.state.spans:
            if span["batch_id"] == batch:
                port_of[span["name"]] = request["client_port"]
    assert set(port_of) == {"parent-1", "child", "parent-2"}
    assert port_of["child"] != parent_port   # its own connection, not the parent's socket
    assert port_of["parent-2"] == parent_port   # the parent's connection survived the child
