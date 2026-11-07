"""Public API: conversation grouping, turn and trace, user, tags and metadata, @step and @tool,
tool-call matching, errors, redact, capture_content, size caps, head sampling, never raising."""

from __future__ import annotations

import asyncio
import json

import openai
import pytest

import zoroval
from tests.conftest import flush
from zoroval import _context


def roots(gateway, kind):
    return [s["attributes"] for s in gateway.spans(**{"zoroval.observation.type": kind})]


def test_two_turns_one_conversation_group_into_one_trace(start_sdk, gateway, llm):
    start_sdk()
    client = openai.OpenAI(base_url=llm.openai_base_url, api_key="t", max_retries=0)
    for text in ("first", "second"):
        with zoroval.conversation("chat-42", user_id="user-7", tags=["beta"]):
            with zoroval.turn(input=text) as turn:
                reply = client.chat.completions.create(model="m", messages=[{"role": "user", "content": text}])
                turn.set_output(reply.choices[0].message.content)
    flush()
    groups = gateway.grouped_traces()
    assert len(groups) == 1 and len(next(iter(groups.values()))) == 2
    turns = roots(gateway, "turn")
    assert all(t["zoroval.conversation.id"] == "chat-42" and t["gen_ai.conversation.id"] == "chat-42" for t in turns)
    assert all(t["zoroval.user.id"] == "user-7" and t["user.id"] == "user-7" for t in turns)
    assert all(t["zoroval.tags"] == ["beta"] for t in turns)
    assert all(t["zoroval.grouping"] == "explicit" for t in turns)
    assert all(g["attributes"]["zoroval.conversation.id"] == "chat-42" for g in gateway.generations())
    assert json.loads(turns[0]["zoroval.input"]) == "first"


def test_turn_without_conversation_is_flagged_ungrouped(start_sdk, gateway):
    start_sdk()
    with zoroval.turn():
        pass
    with zoroval.trace("nightly-job"):
        pass
    flush()
    assert roots(gateway, "turn")[0]["zoroval.grouping"] == "ungrouped"
    assert roots(gateway, "trace")[0]["zoroval.grouping"] == "not_chat"
    assert zoroval.status()["ungrouped_turns"] == 1


def test_trace_inside_conversation_acts_as_turn(start_sdk, gateway):
    start_sdk()
    with zoroval.conversation("c1"):
        with zoroval.trace("chat"):
            pass
    flush()
    assert roots(gateway, "turn")[0]["zoroval.conversation.id"] == "c1"


def test_inherited_conversation_from_parent_span(start_sdk, gateway):
    start_sdk()
    from opentelemetry import trace as otel_trace
    tracer = otel_trace.get_tracer("my.web.framework")
    with tracer.start_as_current_span("POST /chat", attributes={"session.id": "sess-9"}):
        with zoroval.turn():
            with zoroval.step("retrieve"):
                pass
    flush()
    turn = roots(gateway, "turn")[0]
    assert turn["gen_ai.conversation.id"] == "sess-9"
    assert "zoroval.conversation.id" not in turn
    assert turn["zoroval.grouping"] == "inherited"
    assert not gateway.spans(**{"session.id": "sess-9"})  # the framework span itself is not exported


def test_long_conversation_id_is_hashed_not_cut():
    long_a, long_b = "a" * 300 + "x", "a" * 300 + "y"
    assert _context.normalize_conversation_id(long_a) != _context.normalize_conversation_id(long_b)
    assert len(_context.normalize_conversation_id(long_a)) <= 256
    assert _context.normalize_conversation_id("short") == "short"


def test_set_user_tags_metadata_on_current_turn(start_sdk, gateway):
    start_sdk()
    with zoroval.turn():
        zoroval.set_user("u-1")
        zoroval.set_tags("a", ["b", "a"])
        zoroval.set_metadata({"plan": "pro"}, region="eu")
    flush()
    turn = roots(gateway, "turn")[0]
    assert turn["zoroval.user.id"] == "u-1"
    assert turn["zoroval.tags"] == ["a", "b"]
    assert json.loads(turn["zoroval.metadata"]) == {"plan": "pro", "region": "eu"}


def test_metadata_is_capped_at_16kb(start_sdk, gateway):
    start_sdk()
    with zoroval.trace("t", metadata={"blob": "z" * 50_000}):
        pass
    flush()
    trace = roots(gateway, "trace")[0]
    assert len(trace["zoroval.metadata"].encode()) <= 16 * 1024
    assert trace["zoroval.truncated_bytes"] > 30_000
    json.loads(trace["zoroval.metadata"])  # still valid JSON


def test_step_and_tool_decorators_sync_and_async(start_sdk, gateway):
    start_sdk()

    @zoroval.step
    def plan(query: str, limit: int = 3) -> list:
        return [query] * limit

    @zoroval.step(name="rank", type="chain")
    async def rank(items: list) -> list:
        return sorted(items)

    @zoroval.tool(name="search_docs")
    async def search(q: str) -> str:
        return "found " + q

    class Agent:
        @zoroval.step
        def act(self, x: int) -> int:
            return x + 1

    with zoroval.trace("run"):
        plan("q", limit=2)
        asyncio.run(rank(["b", "a"]))
        asyncio.run(search("x"))
        Agent().act(1)
    flush()
    steps = {s["name"]: s["attributes"] for s in gateway.state.spans}
    assert json.loads(steps["plan"]["zoroval.input"]) == {"query": "q", "limit": 2}
    assert json.loads(steps["plan"]["zoroval.output"]) == ["q", "q"]
    assert steps["rank"]["zoroval.observation.type"] == "chain"
    assert json.loads(steps["rank"]["zoroval.output"]) == ["a", "b"]
    assert steps["execute_tool search_docs"]["gen_ai.tool.name"] == "search_docs"
    assert json.loads(steps["execute_tool search_docs"]["gen_ai.tool.call.arguments"]) == {"q": "x"}
    assert json.loads(steps["act"]["zoroval.input"]) == {"x": 1}  # self is not captured
    trace_id = gateway.spans(**{"zoroval.observation.type": "trace"})[0]["trace_id"]
    assert all(s["trace_id"] == trace_id for s in gateway.state.spans)


def test_decorator_error_is_recorded_and_reraised_unchanged(start_sdk, gateway):
    start_sdk()

    class AppError(Exception):
        pass

    @zoroval.step
    def fails() -> None:
        raise AppError("boom")

    with pytest.raises(AppError, match="boom"):
        fails()
    flush()
    span = gateway.state.spans[0]
    assert span["status_code"] == 2 and "AppError: boom" in span["status_message"]
    event = span["events"][0]
    assert event["name"] == "exception"
    assert event["attributes"]["exception.type"].endswith("AppError")
    assert "raise AppError" in event["attributes"]["exception.stacktrace"]


def test_tool_matching_ambiguous_calls_stay_unlinked(start_sdk, gateway):
    start_sdk()

    @zoroval.tool
    def lookup(city: str) -> str:
        return city

    with zoroval.trace("run") as run:
        span = run.span
        _context.register_tool_calls(span, [{"id": "c1", "name": "lookup", "arguments": '{"city": "Oslo"}'},
                                            {"id": "c2", "name": "lookup", "arguments": '{"city": "Rome"}'},
                                            {"id": "c3", "name": "other", "arguments": "{}"},
                                            {"id": "c4", "name": "other", "arguments": "{}"}])
        lookup("Rome")                      # two pending, arguments pick c2
        lookup("Oslo")                      # one pending left: c1
        with zoroval.tool("other"):         # c3 and c4 identical: ambiguous, no id
            pass
        explicit = zoroval.tool("other", call_id="c4")
        with explicit:
            pass
    flush()
    tools = [s["attributes"] for s in gateway.spans(**{"zoroval.observation.type": "tool"})]
    ids = [t.get("gen_ai.tool.call.id") for t in tools]
    assert ids == ["c2", "c1", None, "c4"]


def test_tool_matching_does_not_cross_conversations(start_sdk, gateway):
    start_sdk()
    with zoroval.conversation("user-a"):
        with zoroval.turn() as turn:
            _context.register_tool_calls(turn.span, [{"id": "a1", "name": "pay", "arguments": "{}"}])
    with zoroval.conversation("user-b"):
        with zoroval.turn():
            with zoroval.tool("pay"):
                pass
    flush()
    tool = gateway.spans(**{"zoroval.observation.type": "tool"})[0]["attributes"]
    assert "gen_ai.tool.call.id" not in tool


def test_redact_hook_changes_content_and_sets_flag(start_sdk, gateway, llm):
    def redact(value):
        return json.loads(json.dumps(value).replace("4111-1111", "[card]"))

    start_sdk(redact=redact)
    client = openai.OpenAI(base_url=llm.openai_base_url, api_key="t", max_retries=0)
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "my card 4111-1111"}])
    flush()
    attrs = gateway.generations()[0]["attributes"]
    assert "4111" not in attrs["gen_ai.input.messages"] and "4111" not in attrs["gen_ai.output.messages"]
    assert "[card]" in attrs["gen_ai.input.messages"]
    assert attrs["zoroval.redacted"] is True


def test_redact_hook_that_raises_drops_content_not_the_call(start_sdk, gateway, llm):
    def redact(value):
        raise RuntimeError("bad hook")

    start_sdk(redact=redact)
    client = openai.OpenAI(base_url=llm.openai_base_url, api_key="t", max_retries=0)
    reply = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "secret"}])
    assert reply.choices[0].message.content == "Echo: secret"
    flush()
    attrs = gateway.generations()[0]["attributes"]
    assert "gen_ai.input.messages" not in attrs and "gen_ai.output.messages" not in attrs
    assert zoroval.status()["redact_errors"] >= 1


def test_content_is_truncated_to_valid_json(start_sdk, gateway):
    start_sdk(max_content_bytes=10_000)
    with zoroval.step("big", input={"doc": "w" * 100_000, "keep": "small"}):
        pass
    flush()
    attrs = gateway.state.spans[0]["attributes"]
    value = json.loads(attrs["zoroval.input"])
    assert value["keep"] == "small"
    assert len(attrs["zoroval.input"].encode()) <= 10_000
    assert attrs["zoroval.truncated_bytes"] > 80_000


def test_capture_content_false_for_steps(start_sdk, gateway):
    start_sdk(capture_content=False)

    @zoroval.step
    def secret(x: str) -> str:
        return x

    secret("pii")
    flush()
    attrs = gateway.state.spans[0]["attributes"]
    assert "zoroval.input" not in attrs and "zoroval.output" not in attrs


def test_head_sampling_keeps_whole_conversations(start_sdk, gateway):
    start_sdk(sample_rate=0.5)
    kept, dropped = [], []
    from zoroval._processor import sample_keep
    for index in range(40):
        conv = f"conv-{index}"
        (kept if sample_keep("c:" + conv, 0.5) else dropped).append(conv)
        for turn in range(2):
            with zoroval.conversation(conv):
                with zoroval.turn():
                    with zoroval.step("s"):
                        pass
    flush()
    seen = {s["attributes"]["zoroval.conversation.id"] for s in gateway.state.spans}
    assert seen == set(kept)
    assert 5 < len(kept) < 35
    per_conv = {}
    for s in gateway.state.spans:
        per_conv.setdefault(s["attributes"]["zoroval.conversation.id"], 0)
        per_conv[s["attributes"]["zoroval.conversation.id"]] += 1
    assert set(per_conv.values()) == {4}  # 2 turns x (turn + step), never partial
    assert zoroval.status()["spans_sampled_out"] == 4 * len(dropped)


def test_sample_rate_zero_sends_nothing(start_sdk, gateway):
    start_sdk(sample_rate=0.0)
    with zoroval.trace("t"):
        pass
    zoroval.flush(1.0)
    assert gateway.state.spans == []


def test_internal_errors_never_reach_the_app(start_sdk, gateway, llm, monkeypatch):
    start_sdk()
    from zoroval import _json

    def broken(*args, **kwargs):
        raise ValueError("serializer bug")

    monkeypatch.setattr(_json, "dumps", broken)
    client = openai.OpenAI(base_url=llm.openai_base_url, api_key="t", max_retries=0)
    reply = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "still works"}])
    assert reply.choices[0].message.content == "Echo: still works"

    @zoroval.step
    def f(x):
        return x * 2

    assert f(2) == 4
    with zoroval.turn() as turn:
        turn.set_output(object())
        turn.set_metadata({"a": 1})
    assert zoroval.status()["internal_errors"] >= 1


def test_no_key_means_noop_but_api_works(gateway):
    assert zoroval.init() is False
    with zoroval.conversation("c"):
        with zoroval.turn() as turn:
            turn.set_output("x")
            zoroval.set_user("u")

    @zoroval.step
    def f() -> int:
        return 1

    assert f() == 1
    assert zoroval.flush() is True
    assert zoroval.status()["active"] is False


def test_bad_key_format_is_refused_without_printing_it(caplog):
    with caplog.at_level("WARNING", logger="zoroval"):
        assert zoroval.init(api_key="sk-not-ours-abc123") is False
    assert "sk-not-ours-abc123" not in caplog.text


def test_init_twice_keeps_first(start_sdk):
    start_sdk()
    assert zoroval.init(api_key="zv_ingest_bbbbbbbbbbbb_" + "B" * 43) is True
    assert zoroval.status()["key_id"] != "bbbbbbbbbbbb"


def test_key_from_environment(gateway, monkeypatch):
    key = gateway.issue_key()
    monkeypatch.setenv("ZOROVAL_INGEST_KEY", key)
    monkeypatch.setenv("ZOROVAL_INGEST_URL", gateway.url)
    assert zoroval.init(flush_interval=0.05) is True
    with zoroval.trace("env"):
        pass
    flush()
    assert gateway.spans(**{"zoroval.observation.type": "trace"})


def test_status_never_contains_the_key(start_sdk):
    key = start_sdk()
    assert key not in json.dumps(zoroval.status())
