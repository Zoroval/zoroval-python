"""Anthropic wrapper against the fake Anthropic server: create, stream=True, .stream(), sync and
async, tool_use ids, the client's built-in span not duplicated, errors."""

from __future__ import annotations

import asyncio
import json

import anthropic
import pytest

import zoroval
from tests.conftest import flush

TOOLS = [{"name": "get_weather", "description": "weather",
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]


@pytest.fixture
def client(llm):
    return anthropic.Anthropic(base_url=llm.url, api_key="test", max_retries=0)


@pytest.fixture
def aclient(llm):
    return anthropic.AsyncAnthropic(base_url=llm.url, api_key="test", max_retries=0)


def _gens(gateway):
    flush()
    return [g["attributes"] for g in gateway.generations()]


def test_create_sync(start_sdk, gateway, client):
    start_sdk()
    message = client.messages.create(model="claude-test", max_tokens=50, system="be brief",
                                     messages=[{"role": "user", "content": "hello"}])
    assert message.content[0].text == "Echo: hello"
    gens = _gens(gateway)
    assert len(gens) == 1  # the client's own OTel span is not exported as a second generation
    attrs = gens[0]
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.request.max_tokens"] == 50
    assert attrs["gen_ai.response.id"] == message.id
    assert attrs["gen_ai.usage.input_tokens"] == 13
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 2
    assert attrs["gen_ai.response.finish_reasons"] == ["end_turn"]
    assert json.loads(attrs["gen_ai.system_instructions"]) == [{"type": "text", "content": "be brief"}]
    assert json.loads(attrs["gen_ai.output.messages"])[0]["parts"] == [{"type": "text", "content": "Echo: hello"}]


def test_native_client_span_is_suppressed_not_lost_for_other_exporters(start_sdk, gateway, client):
    start_sdk()
    client.messages.create(model="c", max_tokens=5, messages=[{"role": "user", "content": "x"}])
    flush()
    assert zoroval.status()["spans_filtered"] >= 1
    assert all(s["scope"] == "zoroval" for s in gateway.state.spans)


def test_tool_use_ids_and_tool_result(start_sdk, gateway, client):
    start_sdk()
    with zoroval.conversation("c-anthropic"):
        with zoroval.turn():
            first = client.messages.create(model="c", max_tokens=50, tools=TOOLS,
                                           messages=[{"role": "user", "content": "weather in Paris?"}])
            use = [b for b in first.content if b.type == "tool_use"][0]
            with zoroval.tool("get_weather", arguments=use.input) as run:
                run.set_output({"temp_c": 21})
            client.messages.create(model="c", max_tokens=50, tools=TOOLS, messages=[
                {"role": "user", "content": "weather in Paris?"},
                {"role": "assistant", "content": [b.model_dump() for b in first.content]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "21C"}]}])
    gens = _gens(gateway)
    calls = json.loads(gens[0]["zoroval.tool_calls"])
    assert calls == [{"id": use.id, "name": "get_weather", "arguments": json.dumps({"city": "Paris"},
                                                                                   separators=(",", ":"))}]
    tool = gateway.spans(**{"zoroval.observation.type": "tool"})[0]["attributes"]
    assert tool["gen_ai.tool.call.id"] == use.id
    second = json.loads(gens[1]["gen_ai.input.messages"])
    assert second[2]["parts"] == [{"type": "tool_call_response", "id": use.id, "response": "21C"}]
    assert second[1]["parts"][1]["id"] == use.id


def test_stream_true_sync(start_sdk, gateway, client):
    start_sdk()
    stream = client.messages.create(model="c", max_tokens=50, stream=True, tools=TOOLS,
                                    messages=[{"role": "user", "content": "weather?"}])
    events = [e.type for e in stream]
    assert events[0] == "message_start" and events[-1] == "message_stop"
    attrs = _gens(gateway)[0]
    calls = json.loads(attrs["zoroval.tool_calls"])
    assert calls[0]["name"] == "get_weather" and json.loads(calls[0]["arguments"]) == {"city": "Paris"}
    assert attrs["gen_ai.usage.output_tokens"] == 8 and attrs["gen_ai.usage.input_tokens"] == 13
    assert attrs["gen_ai.response.finish_reasons"] == ["tool_use"]


def test_stream_helper_sync(start_sdk, gateway, client):
    start_sdk()
    with client.messages.stream(model="c", max_tokens=50, messages=[{"role": "user", "content": "helper"}]) as s:
        text = "".join(s.text_stream)
        final = s.get_final_message()
    assert text.strip() == "Echo: helper" and final.content[0].text.strip() == "Echo: helper"
    gens = _gens(gateway)
    assert len(gens) == 1
    assert json.loads(gens[0]["gen_ai.output.messages"])[0]["parts"][0]["content"].strip() == "Echo: helper"


def test_async_create_stream_and_helper(start_sdk, gateway, aclient):
    start_sdk()

    async def run():
        await aclient.messages.create(model="c", max_tokens=5, messages=[{"role": "user", "content": "a1"}])
        stream = await aclient.messages.create(model="c", max_tokens=5, stream=True,
                                               messages=[{"role": "user", "content": "a2"}])
        async for _ in stream:
            pass
        async with aclient.messages.stream(model="c", max_tokens=5,
                                           messages=[{"role": "user", "content": "a3"}]) as s:
            async for _ in s.text_stream:
                pass

    asyncio.run(run())
    texts = sorted(json.loads(g["gen_ai.output.messages"])[0]["parts"][0]["content"].strip() for g in _gens(gateway))
    assert texts == ["Echo: a1", "Echo: a2", "Echo: a3"]


def test_error_passes_through(start_sdk, gateway, client):
    start_sdk()
    with pytest.raises(anthropic.BadRequestError):
        client.messages.create(model="error-400", max_tokens=5, messages=[{"role": "user", "content": "x"}])
    flush()
    span = gateway.generations()[0]
    assert span["status_code"] == 2
    assert span["attributes"]["error.type"] == "BadRequestError"


def test_image_blocks_become_placeholders(start_sdk, gateway, client):
    start_sdk()
    client.messages.create(model="c", max_tokens=5, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo" * 100}},
        {"type": "text", "text": "what is this"}]}])
    attrs = _gens(gateway)[0]
    parts = json.loads(attrs["gen_ai.input.messages"])[0]["parts"]
    assert parts[0] == {"type": "image", "content": "[image omitted]"}
    assert "iVBOR" not in attrs["gen_ai.input.messages"]
