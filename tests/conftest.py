from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing.fake_gateway import FakeGateway  # noqa: E402
from testing.fake_llm import FakeLLM  # noqa: E402


def reset_otel() -> None:
    """Tests need a fresh global TracerProvider; OpenTelemetry allows setting it only once."""
    import opentelemetry.trace as trace_module
    from opentelemetry.util._once import Once

    trace_module._TRACER_PROVIDER = None
    trace_module._TRACER_PROVIDER_SET_ONCE = Once()
    try:
        import opentelemetry._logs._internal as logs_internal
        logs_internal._LOGGER_PROVIDER = None
        logs_internal._LOGGER_PROVIDER_SET_ONCE = Once()
    except Exception:  # noqa: BLE001
        pass


def reset_zoroval() -> None:
    import zoroval
    from zoroval import _util

    zoroval.shutdown(timeout=1.0)
    _util.COUNTERS.__init__()
    _util.reset_log_once()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in list(os.environ):
        if name.startswith(("ZOROVAL_", "OTEL_", "LANGFUSE_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    reset_zoroval()
    reset_otel()
    yield
    reset_zoroval()
    reset_otel()


@pytest.fixture
def gateway() -> Iterator[FakeGateway]:
    with FakeGateway() as gw:
        yield gw


@pytest.fixture
def llm() -> Iterator[FakeLLM]:
    with FakeLLM() as server:
        yield server


@pytest.fixture
def start_sdk(gateway: FakeGateway) -> Callable[..., Any]:
    import zoroval

    def start(**kwargs: Any) -> str:
        key = kwargs.pop("api_key", None) or gateway.issue_key()
        kwargs.setdefault("flush_interval", 0.05)
        kwargs.setdefault("service_name", "test-app")
        kwargs.setdefault("ingest_url", gateway.url)
        assert zoroval.init(api_key=key, **kwargs)
        return key

    return start


def flush() -> None:
    import zoroval
    assert zoroval.flush(5.0)
