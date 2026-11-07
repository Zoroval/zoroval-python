"""Coexistence with an existing OpenTelemetry or Langfuse setup, in both install orders, with no
duplicate spans; reading OpenInference and OpenLLMetry spans; GenAI log events."""

from __future__ import annotations

import json
import subprocess
import sys

import openai
import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import zoroval
from tests.conftest import ROOT, flush


@pytest.fixture
def client(llm):
    return openai.OpenAI(base_url=llm.openai_base_url, api_key="t", max_retries=0)


def _app_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _app_request(client, conversation: str = "conv-otel") -> None:
    tracer = otel_trace.get_tracer("app.http")
    with tracer.start_as_current_span("POST /chat"):
        with zoroval.conversation(conversation):
            with tracer.start_as_current_span("db.query"):
                pass
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])


def test_app_provider_first(start_sdk, gateway, client):
    provider, app_exporter = _app_provider()
    otel_trace.set_tracer_provider(provider)
    start_sdk()
    assert zoroval.status()["provider"] == "existing"
    _app_request(client)
    flush()
    app_names = sorted(s.name for s in app_exporter.get_finished_spans())
    assert app_names == ["POST /chat", "chat m", "db.query"]           # the app still gets everything, once
    assert [s["name"] for s in gateway.state.spans] == ["chat m"]        # Zoroval gets GenAI spans only
    assert gateway.state.spans[0]["attributes"]["zoroval.conversation.id"] == "conv-otel"
    app_db = [s for s in app_exporter.get_finished_spans() if s.name == "db.query"][0]
    assert "gen_ai.conversation.id" not in (app_db.attributes or {})    # the app's own spans are not tagged


def test_zoroval_first_then_app_sets_provider(start_sdk, gateway, client):
    start_sdk()
    assert zoroval.status()["provider"] == "own"
    provider, app_exporter = _app_provider()
    otel_trace.set_tracer_provider(provider)                              # OTel refuses a second provider...
    _app_request(client)
    flush()
    app_names = sorted(s.name for s in app_exporter.get_finished_spans())
    assert app_names == ["POST /chat", "chat m", "db.query"]           # ...so we bridged the app's processors
    assert [s["name"] for s in gateway.state.spans] == ["chat m"]


def test_explicit_tracer_provider_argument(gateway, client):
    provider, app_exporter = _app_provider()
    key = gateway.issue_key()
    assert zoroval.init(api_key=key, ingest_url=gateway.url, tracer_provider=provider, flush_interval=0.05)
    with zoroval.trace("job"):
        pass
    flush()
    assert [s.name for s in app_exporter.get_finished_spans()] == ["job"]
    assert [s["name"] for s in gateway.state.spans] == ["job"]


def test_openinference_spans_are_exported_and_tagged(start_sdk, gateway, client):
    start_sdk()
    tracer = otel_trace.get_tracer("openinference.instrumentation.langchain")
    with zoroval.conversation("conv-oi"):
        with tracer.start_as_current_span("ChatOpenAI", attributes={
                "openinference.span.kind": "LLM", "llm.model_name": "gpt-x",
                "input.value": "question", "output.value": "answer"}):
            # Their instrumentor already records this LLM call: our wrapper must not add a second span.
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
        with tracer.start_as_current_span("retriever", attributes={"openinference.span.kind": "RETRIEVER"}):
            pass
    flush()
    names = sorted(s["name"] for s in gateway.state.spans)
    assert names == ["ChatOpenAI", "retriever"]
    for span in gateway.state.spans:
        assert span["attributes"]["gen_ai.conversation.id"] == "conv-oi"
        assert span["scope"] == "openinference.instrumentation.langchain"


def test_openllmetry_spans_are_exported_without_double_wrapping(start_sdk, gateway, client):
    start_sdk()
    tracer = otel_trace.get_tracer("opentelemetry.instrumentation.openai.v1")
    with tracer.start_as_current_span("openai.chat", attributes={"llm.request.type": "chat",
                                                                 "gen_ai.system": "openai",
                                                                 "gen_ai.request.model": "m"}):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    flush()
    assert [s["name"] for s in gateway.state.spans] == ["openai.chat"]


def test_otel_contrib_style_spans_and_gen_ai_log_events(start_sdk, gateway):
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider
    set_logger_provider(LoggerProvider())
    start_sdk()
    from opentelemetry._logs import get_logger
    from opentelemetry._logs import LogRecord
    logger = get_logger("opentelemetry.instrumentation.openai_v2")
    tracer = otel_trace.get_tracer("opentelemetry.instrumentation.openai_v2")
    with tracer.start_as_current_span("chat m", attributes={"gen_ai.operation.name": "chat"}):
        logger.emit(LogRecord(event_name="gen_ai.user.message", body={"content": "hello"}))
        logger.emit(LogRecord(event_name="app.audit", body="not ours"))
    zoroval.flush(5.0)
    assert [s["name"] for s in gateway.state.spans] == ["chat m"]
    assert [log["event_name"] for log in gateway.state.logs] == ["gen_ai.user.message"]


def test_non_genai_spans_inside_conversation_do_not_leak(start_sdk, gateway):
    start_sdk()
    tracer = otel_trace.get_tracer("app")
    with zoroval.conversation("c"):
        with tracer.start_as_current_span("cache.get"):
            pass
        with zoroval.turn():
            with tracer.start_as_current_span("http.client"):
                pass
    flush()
    assert [s["name"] for s in gateway.state.spans] == ["turn"]
    assert zoroval.status()["spans_filtered"] == 2


@pytest.mark.parametrize("order", ["langfuse_first", "zoroval_first"])
def test_langfuse_both_install_orders(order):
    out = subprocess.run([sys.executable, str(ROOT / "tests" / "coexist_langfuse.py"), order], capture_output=True,
                         text=True, timeout=120, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr[-2000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["answers"] == ["Echo: first", "Echo: second"]
    assert result["zoroval_generations"] == 2          # one per call, never two
    assert result["langfuse_generations"] == 2         # Langfuse keeps working, no duplicates either
    assert result["zoroval_generation_conversations"] == ["conv-lf"]
    assert result["zoroval_generation_has_output"] is True
    assert result["zoroval_traces"] == 1
