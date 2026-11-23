import json

import pytest
from opentelemetry.sdk.trace import TracerProvider

import zoroval
from zoroval.integrations._common import Generation

KEY = "zv_ingest_abcdefghijkl_" + "x" * 30
CALLS = [{"id": "call_1", "name": "weather", "arguments": {"city": "Paris"}}]


def tool_calls_attr(**init_options):
    zoroval.init(api_key=KEY, ingest_url="http://127.0.0.1:9", tracer_provider=TracerProvider(),
                 instrument=False, **init_options)
    try:
        gen = Generation(provider="openai", model="m")
        gen.finish(tool_calls=CALLS)
        return gen.span.attributes
    finally:
        zoroval.shutdown(0.5)


def test_redact_runs_on_tool_call_arguments():
    attrs = tool_calls_attr(redact=lambda value: "[R]")
    calls = json.loads(attrs["zoroval.tool_calls"])
    assert calls == [{"id": "call_1", "name": "weather", "arguments": "[R]"}]
    assert attrs["zoroval.redacted"] is True


def test_tool_call_arguments_dropped_when_redact_raises():
    def boom(value):
        raise RuntimeError("no")

    attrs = tool_calls_attr(redact=boom)
    assert json.loads(attrs["zoroval.tool_calls"]) == [{"id": "call_1", "name": "weather"}]
    assert "Paris" not in attrs["zoroval.tool_calls"]


def test_capture_content_off_drops_tool_call_arguments():
    attrs = tool_calls_attr(capture_content=False)
    assert json.loads(attrs["zoroval.tool_calls"]) == [{"id": "call_1", "name": "weather"}]


def test_tool_call_arguments_kept_without_redact():
    attrs = tool_calls_attr()
    assert json.loads(attrs["zoroval.tool_calls"])[0]["arguments"] == {"city": "Paris"}
