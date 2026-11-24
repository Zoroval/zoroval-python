"""init(redact=...): off by default, the built-in rules, the developer's function after them, and
every way that function can misbehave. Nothing may reach the app; unscannable content is dropped."""

import json
import time

import pytest
from opentelemetry.sdk.trace import TracerProvider

import zoroval
from zoroval import _redact, _state
from zoroval.integrations._common import Generation

KEY = "zv_ingest_abcdefghijkl_" + "x" * 30
CARD = "4111 1111 1111 1111"
TEXT = f"mail priya@example.com, card {CARD}, phone +91 98765 43210"
CLEAN = "mail [EMAIL], card [CARD], phone [PHONE]"


@pytest.fixture
def start(monkeypatch):
    monkeypatch.delenv("ZOROVAL_CAPTURE_CONTENT", raising=False)
    started = []

    def _start(**options):
        ok = zoroval.init(api_key=KEY, ingest_url="http://127.0.0.1:9", tracer_provider=TracerProvider(),
                          instrument=False, **options)
        started.append(ok)
        return ok

    yield _start
    zoroval.shutdown(0.5)


def output_of(value):
    with zoroval.turn(input="hi") as obs:
        obs.set_output(value)
    return obs.span.attributes


def test_default_is_off_and_sends_content_as_is(start):
    assert start()
    attrs = output_of(TEXT)
    assert json.loads(attrs["zoroval.output"]) == TEXT
    assert "zoroval.redacted" not in attrs
    assert zoroval.status()["redact"] is None


def test_basic_replaces_level1_values(start):
    assert start(redact="basic")
    attrs = output_of({"messages": [{"role": "user", "content": TEXT}], "phone": 9876543210, "n": 9876543210,
                       "pnr": "8123456789"})
    assert json.loads(attrs["zoroval.output"]) == {"messages": [{"role": "user", "content": CLEAN}],
                                                   "phone": "[PHONE]", "n": 9876543210, "pnr": "8123456789"}
    assert attrs["zoroval.redacted"] is True
    assert zoroval.status()["redact"] == "basic"


def test_basic_runs_on_tool_call_arguments(start):
    start(redact="basic")
    gen = Generation(provider="openai", model="m")
    gen.finish(tool_calls=[{"id": "c1", "name": "pay", "arguments": json.dumps({"card": CARD, "amount": 499})}])
    calls = json.loads(gen.span.attributes["zoroval.tool_calls"])
    assert json.loads(calls[0]["arguments"]) == {"card": "[CARD]", "amount": 499}
    assert gen.span.attributes["zoroval.redacted"] is True


def test_basic_runs_on_error_messages_and_metadata(start):
    start(redact="basic")
    gen = Generation(provider="openai", model="m")
    gen.finish(error=RuntimeError(f"declined for priya@example.com card {CARD}"))
    event = gen.span.events[0]
    assert "priya@example.com" not in event.attributes["exception.message"]
    assert "4111" not in event.attributes["exception.stacktrace"]
    assert "[EMAIL]" in gen.span.status.description
    with zoroval.turn(input="x", metadata={"email": "a@b.co", "plan": "pro"}) as obs:
        pass
    assert json.loads(obs.span.attributes["zoroval.metadata"]) == {"email": "[EMAIL]", "plan": "pro"}


def test_function_runs_after_basic_and_sees_labels(start):
    seen = []

    def mine(value):
        seen.append(value)
        return value.replace("phone", "tel") if isinstance(value, str) else value

    start(redact=["basic", mine])
    attrs = output_of(TEXT)
    assert seen == ["hi", CLEAN]
    assert json.loads(attrs["zoroval.output"]) == "mail [EMAIL], card [CARD], tel [PHONE]"
    assert zoroval.status()["redact"] == "basic+function"


def test_order_in_the_list_does_not_matter(start):
    seen = []
    start(redact=[lambda v: seen.append(v) or v, "basic"])
    output_of(TEXT)
    assert seen == ["hi", CLEAN]


@pytest.mark.parametrize("bad", [
    lambda v: 1 / 0,
    lambda v: None,
    lambda v: {}["missing"],
])
def test_function_that_fails_drops_the_field(start, bad):
    start(redact=["basic", bad])
    attrs = output_of(TEXT)
    assert "zoroval.output" not in attrs
    assert attrs["zoroval.redacted"] is True


@pytest.mark.parametrize("returned,expected", [
    (42, 42), ({"k": [1, "x"]}, {"k": [1, "x"]}), (b"raw", "[3 bytes omitted]"), (True, True), ("", ""),
])
def test_function_returning_non_strings_is_sent_as_json(start, returned, expected):
    start(redact=["basic", lambda v: returned])
    assert json.loads(output_of(TEXT)["zoroval.output"]) == expected


def test_function_returning_a_hostile_object_never_reaches_the_app(start):
    class Hostile:
        def __repr__(self):
            raise RuntimeError("repr")

        def __eq__(self, other):
            raise RuntimeError("eq")

        __hash__ = object.__hash__

    start(redact=["basic", lambda v: Hostile()])
    attrs = output_of(TEXT)  # must not raise
    assert "zoroval.output" not in attrs and "zoroval.input" not in attrs
    assert zoroval.status()["unserializable_fields"] >= 1


def test_function_receiving_unscannable_value_is_not_called_and_field_dropped(start):
    calls = []
    start(redact=["basic", lambda v: calls.append(v) or v])
    deep = current = []
    for _ in range(_redact.MAX_DEPTH + 5):
        current.append([])
        current = current[0]
    attrs = output_of(deep)
    assert "zoroval.output" not in attrs and calls == ["hi"]


def test_slow_function_runs_once_per_field(start):
    calls = []

    def slow(value):
        calls.append(value)
        time.sleep(0.05)
        return value

    start(redact=["basic", slow])
    began = time.perf_counter()
    output_of(TEXT)
    assert len(calls) == 2  # input and output
    assert time.perf_counter() - began < 1.0


@pytest.mark.parametrize("bad", ["pii", "BASIC", 123, ["basic", "basic2"], [print, len], object()])
def test_invalid_redact_values_raise_typeerror(start, bad):
    with pytest.raises(TypeError):
        start(redact=bad)


def test_capture_content_false_keeps_tracing_off(start):
    assert start(capture_content=False) is False
    assert _state.get() is None


def test_capture_content_true_is_ignored(start):
    assert start(capture_content=True) is True
    assert _state.get() is not None


def test_env_capture_content_off_keeps_tracing_off(start, monkeypatch):
    monkeypatch.setenv("ZOROVAL_CAPTURE_CONTENT", "false")
    assert start() is False
    assert _state.get() is None


def test_broken_rule_file_keeps_tracing_off(start, monkeypatch):
    def broken():
        raise OSError("missing")

    monkeypatch.setattr(_redact, "_RULES", None)
    monkeypatch.setattr(_redact, "load_rules", broken)
    assert start(redact="basic") is False
    assert _state.get() is None


def test_other_tools_spans_are_scrubbed_on_a_copy(start):
    start(redact="basic")
    processor = _state.get().processor
    provider = TracerProvider()
    span = provider.get_tracer("opentelemetry.instrumentation.openai").start_span(
        "chat", attributes={"gen_ai.prompt.0.content": TEXT, "gen_ai.request.model": "m"})
    span.end()
    clean = processor._scrub(span)
    assert clean.attributes["gen_ai.prompt.0.content"] == CLEAN
    assert clean.attributes["zoroval.redacted"] is True
    assert span.attributes["gen_ai.prompt.0.content"] == TEXT


def test_log_events_stay_off_with_basic(start):
    start(redact="basic")
    assert _state.get().log_processor is None
