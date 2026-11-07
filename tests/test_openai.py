"""OpenAI wrapper against the fake OpenAI server: Chat Completions and Responses, sync, async,
streaming, tool-call ids, errors."""

from __future__ import annotations

import asyncio
import json

import openai
import pytest

import zoroval
from tests.conftest import flush

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "weather",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
RESP_TOOLS = [{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}}}}]


@pytest.fixture
def client(llm):
    return openai.OpenAI(base_url=llm.openai_base_url, api_key="test", max_retries=0)


@pytest.fixture
def aclient(llm):
    return openai.AsyncOpenAI(base_url=llm.openai_base_url, api_key="test", max_retries=0)


def _one_generation(gateway):
    flush()
    gens = gateway.generations()
    assert len(gens) == 1, [g["name"] for g in gateway.state.spans]
    return gens[0]["attributes"]


def _messages(attrs, key):
    return json.loads(attrs[key])


def test_chat_sync(start_sdk, gateway, client):
    start_sdk()
    reply = client.chat.completions.create(model="gpt-test", messages=[
        {"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}], temperature=0.2)
    assert reply.choices[0].message.content == "Echo: hello"
    attrs = _one_generation(gateway)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-test"
    assert attrs["gen_ai.request.temperature"] == 0.2
    assert attrs["gen_ai.response.id"] == reply.id
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 3
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]
    inputs = _messages(attrs, "gen_ai.input.messages")
    assert [m["role"] for m in inputs] == ["system", "user"]
    assert inputs[1]["parts"] == [{"type": "text", "content": "hello"}]
    output = _messages(attrs, "gen_ai.output.messages")
    assert output[0]["parts"][0]["content"] == "Echo: hello"


def test_chat_tool_calls_have_ids_and_pair_with_tool_span(start_sdk, gateway, client):
    start_sdk()

    @zoroval.tool
    def get_weather(city: str) -> dict:
        return {"city": city, "temp_c": 20}

    with zoroval.conversation("conv-tools"):
        with zoroval.turn():
            reply = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "weather?"}],
                                                   tools=TOOLS)
            call = reply.choices[0].message.tool_calls[0]
            result = get_weather(**json.loads(call.function.arguments))
            client.chat.completions.create(model="m", tools=TOOLS, messages=[
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": None, "tool_calls": [call.model_dump()]},
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}])
    flush()
    gens = gateway.generations()
    assert len(gens) == 2
    first = gens[0]["attributes"]
    calls = json.loads(first["zoroval.tool_calls"])
    assert calls == [{"id": call.id, "name": "get_weather", "arguments": call.function.arguments}]
    tool = gateway.spans(**{"zoroval.observation.type": "tool"})[0]["attributes"]
    assert tool["gen_ai.tool.call.id"] == call.id
    assert tool["gen_ai.tool.name"] == "get_weather"
    assert json.loads(tool["gen_ai.tool.call.result"]) == {"city": "Paris", "temp_c": 20}
    second_input = json.loads(gens[1]["attributes"]["gen_ai.input.messages"])
    assert second_input[2] == {"role": "tool", "parts": [{"type": "tool_call_response", "id": call.id,
                                                          "response": json.dumps(result)}]}
    assert second_input[1]["parts"][0]["type"] == "tool_call" and second_input[1]["parts"][0]["id"] == call.id


def test_chat_stream_sync(start_sdk, gateway, client):
    start_sdk()
    stream = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "stream me"}],
                                            stream=True, stream_options={"include_usage": True})
    assert isinstance(stream, openai.Stream)  # the proxy keeps isinstance() working
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text.strip() == "Echo: stream me"
    attrs = _one_generation(gateway)
    assert attrs["gen_ai.request.stream"] is True
    assert _messages(attrs, "gen_ai.output.messages")[0]["parts"][0]["content"].strip() == "Echo: stream me"
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert "gen_ai.response.time_to_first_chunk" in attrs


def test_chat_stream_tool_call(start_sdk, gateway, client):
    start_sdk()
    chunks = list(client.chat.completions.create(model="m", messages=[{"role": "user", "content": "weather?"}],
                                                 tools=TOOLS, stream=True))
    assert chunks
    attrs = _one_generation(gateway)
    calls = json.loads(attrs["zoroval.tool_calls"])
    assert calls[0]["name"] == "get_weather" and calls[0]["id"].startswith("call_")
    assert json.loads(calls[0]["arguments"]) == {"city": "Paris"}
    assert attrs["gen_ai.response.finish_reasons"] == ["tool_calls"]


def test_chat_stream_helper(start_sdk, gateway, client):
    start_sdk()
    with client.chat.completions.stream(model="m", messages=[{"role": "user", "content": "helper"}]) as stream:
        for _ in stream:
            pass
        final = stream.get_final_completion()
    assert final.choices[0].message.content.strip() == "Echo: helper"
    attrs = _one_generation(gateway)
    assert _messages(attrs, "gen_ai.output.messages")[0]["parts"][0]["content"].strip() == "Echo: helper"


def test_chat_async_and_async_stream(start_sdk, gateway, aclient):
    start_sdk()

    async def run():
        reply = await aclient.chat.completions.create(model="m", messages=[{"role": "user", "content": "a1"}])
        stream = await aclient.chat.completions.create(model="m", messages=[{"role": "user", "content": "a2"}],
                                                       stream=True)
        text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
        return reply.choices[0].message.content, text

    first, second = asyncio.run(run())
    assert first == "Echo: a1" and second.strip() == "Echo: a2"
    flush()
    outputs = sorted(json.loads(g["attributes"]["gen_ai.output.messages"])[0]["parts"][0]["content"].strip()
                     for g in gateway.generations())
    assert outputs == ["Echo: a1", "Echo: a2"]


def test_stream_closed_early_still_ends_span(start_sdk, gateway, client):
    start_sdk()
    stream = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "one two three"}],
                                            stream=True)
    next(iter(stream))
    stream.close()
    attrs = _one_generation(gateway)
    assert attrs["gen_ai.request.stream"] is True


def test_responses_sync_and_chain(start_sdk, gateway, client):
    start_sdk()
    with zoroval.turn():
        first = client.responses.create(model="m", input="hi there", instructions="be kind")
        second = client.responses.create(model="m", input="again", previous_response_id=first.id)
    assert first.output_text == "Echo: hi there"
    flush()
    gens = gateway.generations()
    assert len(gens) == 2
    a, b = gens[0]["attributes"], gens[1]["attributes"]
    assert a["gen_ai.response.id"] == first.id
    assert json.loads(a["gen_ai.system_instructions"]) == [{"type": "text", "content": "be kind"}]
    assert json.loads(a["gen_ai.input.messages"]) == [{"role": "user", "parts": [{"type": "text",
                                                                                   "content": "hi there"}]}]
    assert b["gen_ai.openai.previous_response_id"] == first.id
    assert a["gen_ai.usage.reasoning.output_tokens"] == 2
    root = gateway.spans(**{"zoroval.observation.type": "turn"})[0]["attributes"]
    assert root["zoroval.grouping"] == "response_chain"
    assert second.id != first.id


def test_responses_function_call_ids(start_sdk, gateway, client):
    start_sdk()
    response = client.responses.create(model="m", input=[{"role": "user", "content": "weather in Paris?"}],
                                       tools=RESP_TOOLS)
    item = response.output[0]
    client.responses.create(model="m", tools=RESP_TOOLS, input=[
        {"role": "user", "content": "weather in Paris?"},
        {"type": "function_call", "call_id": item.call_id, "name": item.name, "arguments": item.arguments},
        {"type": "function_call_output", "call_id": item.call_id, "output": "20C"}])
    flush()
    gens = gateway.generations()
    calls = json.loads(gens[0]["attributes"]["zoroval.tool_calls"])
    assert calls[0]["id"] == item.call_id
    second = json.loads(gens[1]["attributes"]["gen_ai.input.messages"])
    assert second[1]["parts"][0] == {"type": "tool_call", "id": item.call_id, "name": "get_weather",
                                     "arguments": item.arguments}
    assert second[2] == {"role": "tool", "parts": [{"type": "tool_call_response", "id": item.call_id,
                                                    "response": "20C"}]}


def test_responses_stream_sync_async_and_helper(start_sdk, gateway, client, aclient):
    start_sdk()
    events = list(client.responses.create(model="m", input="s1", stream=True))
    assert events[-1].type == "response.completed"
    with client.responses.stream(model="m", input="s2") as stream:
        for _ in stream:
            pass

    async def run():
        stream = await aclient.responses.create(model="m", input="s3", stream=True)
        async for _ in stream:
            pass

    asyncio.run(run())
    flush()
    texts = sorted(json.loads(g["attributes"]["gen_ai.output.messages"])[0]["parts"][0]["content"]
                   for g in gateway.generations())
    assert texts == ["Echo: s1", "Echo: s2", "Echo: s3"]
    assert all(g["attributes"]["gen_ai.usage.input_tokens"] == 12 for g in gateway.generations())


def test_responses_async_non_stream(start_sdk, gateway, aclient):
    start_sdk()
    response = asyncio.run(aclient.responses.create(model="m", input="async resp"))
    assert response.output_text == "Echo: async resp"
    attrs = _one_generation(gateway)
    assert attrs["gen_ai.response.id"] == response.id


def test_provider_error_passes_through_and_is_recorded(start_sdk, gateway, client):
    start_sdk()
    with pytest.raises(openai.InternalServerError):
        client.chat.completions.create(model="error-500", messages=[{"role": "user", "content": "x"}])
    flush()
    span = gateway.generations()[0]
    assert span["status_code"] == 2  # ERROR
    assert span["attributes"]["error.type"] == "InternalServerError"
    assert any(e["name"] == "exception" for e in span["events"])


def test_azure_and_compatible_base_url(start_sdk, gateway, llm):
    start_sdk()
    azure = openai.AzureOpenAI(azure_endpoint=llm.url, api_key="test", api_version="2024-06-01", max_retries=0)
    try:
        azure.chat.completions.create(model="deploy", messages=[{"role": "user", "content": "az"}])
    except openai.APIError:
        pass  # the fake server does not serve Azure paths; the span still records the attempt
    flush()
    attrs = gateway.generations()[0]["attributes"]
    assert attrs["gen_ai.provider.name"] == "azure.ai.openai"
    assert attrs["server.address"] == "127.0.0.1"


def test_capture_content_off_keeps_ids_and_usage(start_sdk, gateway, client):
    start_sdk(capture_content=False)
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "weather secret?"}], tools=TOOLS)
    attrs = _one_generation(gateway)
    assert "gen_ai.input.messages" not in attrs and "gen_ai.output.messages" not in attrs
    assert "gen_ai.tool.definitions" not in attrs
    calls = json.loads(attrs["zoroval.tool_calls"])
    assert set(calls[0]) == {"id", "name"}
    assert attrs["gen_ai.usage.input_tokens"] == 11


def test_raw_response_is_traced(start_sdk, gateway, client):
    start_sdk()
    raw = client.chat.completions.with_raw_response.create(model="m", messages=[{"role": "user", "content": "raw"}])
    assert raw.parse().choices[0].message.content == "Echo: raw"
    attrs = _one_generation(gateway)
    assert json.loads(attrs["gen_ai.output.messages"])[0]["parts"][0]["content"] == "Echo: raw"


def test_not_initialized_means_untouched(client, gateway):
    reply = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "plain"}])
    assert reply.choices[0].message.content == "Echo: plain"
    assert gateway.state.spans == []
