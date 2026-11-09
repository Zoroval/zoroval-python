"""capture_content=False and the redact hook cover other tools' spans and GenAI log events."""

from __future__ import annotations

import json

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import zoroval
from tests.conftest import flush

SECRET = "my-ssn-is-123-45-6789"


def _foreign_spans() -> None:
    oi = otel_trace.get_tracer("openinference.instrumentation.openai")
    with oi.start_as_current_span("ChatCompletion") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", "gpt-test")
        span.set_attribute("input.value", json.dumps({"messages": [{"content": SECRET}]}))
        span.set_attribute("llm.input_messages.0.message.content", SECRET)
        span.set_attribute("output.value", SECRET)
    lf = otel_trace.get_tracer("langfuse-sdk")
    with lf.start_as_current_span("gen") as span:
        span.set_attribute("langfuse.observation.type", "generation")
        span.set_attribute("langfuse.observation.input", SECRET)
        span.set_attribute("langfuse.observation.output", SECRET)
        span.set_attribute("gen_ai.usage.input_tokens", 7)
        span.add_event("gen_ai.user.message", {"content": SECRET})


def _run(start_sdk, gateway, **kwargs):
    provider = TracerProvider()
    app_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(app_exporter))
    otel_trace.set_tracer_provider(provider)
    start_sdk(**kwargs)
    _foreign_spans()
    flush()
    return app_exporter


def test_content_off_drops_other_tools_content(start_sdk, gateway):
    app_exporter = _run(start_sdk, gateway, capture_content=False)
    assert len(gateway.state.spans) == 2
    assert SECRET not in json.dumps(gateway.state.spans)
    attrs = [s["attributes"] for s in gateway.state.spans]
    assert attrs[0]["llm.model_name"] == "gpt-test"
    assert attrs[1]["gen_ai.usage.input_tokens"] == 7
    # The app's own exporter still sees the original spans.
    assert any(s.attributes.get("input.value") for s in app_exporter.get_finished_spans())


def test_redact_hook_applies_to_other_tools_content(start_sdk, gateway):
    _run(start_sdk, gateway, redact=lambda value: value.replace(SECRET, "[removed]") if isinstance(value, str) else value)
    assert SECRET not in json.dumps(gateway.state.spans)
    first = gateway.state.spans[0]["attributes"]
    assert "[removed]" in first["output.value"]
    assert first["zoroval.redacted"] is True


def test_failing_redact_hook_drops_other_tools_content(start_sdk, gateway):
    def boom(value):
        raise RuntimeError("no")

    _run(start_sdk, gateway, redact=boom)
    assert SECRET not in json.dumps(gateway.state.spans)


@pytest.mark.parametrize("options", [{"capture_content": False}, {"redact": lambda v: v}])
def test_genai_log_events_are_not_exported_with_a_content_control(start_sdk, gateway, options):
    from opentelemetry._logs import LogRecord, get_logger, set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider
    set_logger_provider(LoggerProvider())
    start_sdk(**options)
    assert zoroval.status()  # started
    get_logger("opentelemetry.instrumentation.openai_v2").emit(
        LogRecord(event_name="gen_ai.user.message", body={"content": SECRET}))
    flush()
    assert gateway.state.logs == []
