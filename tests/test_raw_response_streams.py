"""Streams read through `with_raw_response` / `with_streaming_response` keep their reply (found
2026-11-25: OpenAI Agents SDK run_streamed and LangChain ChatAnthropic streams lost it)."""

import asyncio
import gc
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import zoroval
from zoroval.integrations._common import Generation, hook_raw_response, needs_parse_hook
from zoroval.integrations.openai import _chat_finish, _ChatStreamState

KEY = "zv_ingest_abcdefghijkl_" + "x" * 30


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    zoroval.init(api_key=KEY, ingest_url="http://127.0.0.1:9", tracer_provider=provider, instrument=["openai", "anthropic"])
    try:
        yield exporter
    finally:
        zoroval.shutdown(0.5)


def _llm_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.instrumentation_scope.name == "zoroval"]


def _output(span):
    return json.loads(span.attributes["gen_ai.output.messages"])


# --------------------------------------------------------------- without the provider SDKs (runs in CI)

class Stream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        return iter(self.chunks)


Stream.__module__ = "openai._streaming"

CHUNKS = [{"id": "c", "choices": [{"index": 0, "delta": {"content": "Hello"}}]},
          {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]


class RawResponse:
    http_response = object()

    def __init__(self, parsed):
        self.parsed = parsed
        self.closed = False

    def parse(self):
        return self.parsed

    def close(self):
        self.closed = True


def test_parse_hook_wraps_the_stream(spans):
    gen = Generation(provider="openai", model="m")
    raw = RawResponse(Stream(CHUNKS))
    assert needs_parse_hook(raw, True)
    assert hook_raw_response(raw, gen, _ChatStreamState, False, _chat_finish)
    stream = raw.parse()
    assert raw.parse() is stream  # same proxy on a second parse()
    assert [c["id"] for c in stream] == ["c", "c"]
    (span,) = _llm_spans(spans)
    assert _output(span)[0]["parts"] == [{"type": "text", "content": "Hello"}]


def test_parse_hook_final_object(spans):
    gen = Generation(provider="openai", model="m")
    raw = RawResponse({"id": "c1", "choices": [{"finish_reason": "stop", "message": {"content": "Hi"}}]})
    raw.__class__.__name__  # an APIResponse-like object: not LegacyAPIResponse
    assert needs_parse_hook(raw, False)
    hook_raw_response(raw, gen, _ChatStreamState, False, _chat_finish)
    raw.parse()
    (span,) = _llm_spans(spans)
    assert _output(span)[0]["parts"] == [{"type": "text", "content": "Hi"}]


def test_unparsed_response_ends_span_on_close(spans):
    gen = Generation(provider="openai", model="m")
    raw = RawResponse(Stream(CHUNKS))
    hook_raw_response(raw, gen, _ChatStreamState, False, _chat_finish)
    raw.close()
    assert raw.closed and len(_llm_spans(spans)) == 1


def test_unparsed_response_ends_span_when_collected(spans):
    gen = Generation(provider="openai", model="m")
    raw = RawResponse(Stream(CHUNKS))
    hook_raw_response(raw, gen, _ChatStreamState, False, _chat_finish)
    del raw
    gc.collect()
    assert len(_llm_spans(spans)) == 1


# --------------------------------------------------------------- real clients over a mock transport

def _sse(events):
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return body.encode()


def _anthropic_sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


OPENAI_CHUNKS = [
    {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-x",
     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}]},
    {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-x",
     "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
]
OPENAI_JSON = {"id": "chatcmpl-2", "object": "chat.completion", "created": 1, "model": "gpt-x",
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "Plain hello"}}],
               "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
ANTHROPIC_EVENTS = [
    {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
                                          "content": [], "stop_reason": None, "stop_sequence": None,
                                          "usage": {"input_tokens": 10, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Bonjour"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
     "usage": {"output_tokens": 3}},
    {"type": "message_stop"},
]


def _httpx():
    """The HTTP library the installed provider SDKs build on (httpx2 in current releases, httpx before)."""
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    return httpx


def _openai_handler(request):
    httpx = _httpx()
    body = json.loads(request.content)
    if body.get("stream"):
        return httpx.Response(200, content=_sse(OPENAI_CHUNKS), headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=OPENAI_JSON)


def _anthropic_handler(request):
    httpx = _httpx()
    return httpx.Response(200, content=_anthropic_sse(ANTHROPIC_EVENTS), headers={"content-type": "text/event-stream"})


def _openai_client(is_async=False):
    openai = pytest.importorskip("openai")
    httpx = _httpx()
    if is_async:
        return openai.AsyncOpenAI(api_key="sk-test", base_url="http://fake.test/v1",
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(_openai_handler)))
    return openai.OpenAI(api_key="sk-test", base_url="http://fake.test/v1",
                         http_client=httpx.Client(transport=httpx.MockTransport(_openai_handler)))


MSGS = [{"role": "user", "content": "hi"}]


def test_openai_with_raw_response_stream(spans):
    client = _openai_client()
    raw = client.chat.completions.with_raw_response.create(model="gpt-x", messages=MSGS, stream=True)
    text = "".join(c.choices[0].delta.content or "" for c in raw.parse())
    assert text == "Hello"
    (span,) = _llm_spans(spans)
    assert _output(span)[0]["parts"] == [{"type": "text", "content": "Hello"}]


def test_openai_with_streaming_response_stream(spans):
    client = _openai_client()
    with client.chat.completions.with_streaming_response.create(model="gpt-x", messages=MSGS, stream=True) as resp:
        text = "".join(c.choices[0].delta.content or "" for c in resp.parse())
    assert text == "Hello"
    (span,) = _llm_spans(spans)
    assert _output(span)[0]["parts"] == [{"type": "text", "content": "Hello"}]


def test_openai_with_streaming_response_bytes_only(spans):
    client = _openai_client()
    with client.chat.completions.with_streaming_response.create(model="gpt-x", messages=MSGS, stream=True) as resp:
        data = b"".join(resp.iter_bytes())
    assert b"Hel" in data
    (span,) = _llm_spans(spans)
    assert "gen_ai.output.messages" not in span.attributes


def test_openai_with_streaming_response_not_streamed(spans):
    client = _openai_client()
    with client.chat.completions.with_streaming_response.create(model="gpt-x", messages=MSGS) as resp:
        completion = resp.parse()
    assert completion.choices[0].message.content == "Plain hello"
    (span,) = _llm_spans(spans)
    assert _output(span)[0]["parts"] == [{"type": "text", "content": "Plain hello"}]


def test_openai_async_raw_and_streaming(spans):
    async def run():
        client = _openai_client(is_async=True)
        raw = await client.chat.completions.with_raw_response.create(model="gpt-x", messages=MSGS, stream=True)
        first = "".join([c.choices[0].delta.content or "" async for c in raw.parse()])
        async with client.chat.completions.with_streaming_response.create(
                model="gpt-x", messages=MSGS, stream=True) as resp:
            stream = await resp.parse()
            second = "".join([c.choices[0].delta.content or "" async for c in stream])
        async with client.chat.completions.with_streaming_response.create(model="gpt-x", messages=MSGS) as resp:
            third = (await resp.parse()).choices[0].message.content
        return first, second, third

    assert asyncio.run(run()) == ("Hello", "Hello", "Plain hello")
    outputs = [_output(s)[0]["parts"][0]["content"] for s in _llm_spans(spans)]
    assert outputs == ["Hello", "Hello", "Plain hello"]


def test_anthropic_with_raw_response_stream(spans):
    anthropic = pytest.importorskip("anthropic")
    httpx = _httpx()
    client = anthropic.Anthropic(api_key="sk-test", base_url="http://fake.test",
                                 http_client=httpx.Client(transport=httpx.MockTransport(_anthropic_handler)))
    raw = client.messages.with_raw_response.create(model="claude-x", max_tokens=10, messages=MSGS, stream=True)
    kinds = [e.type for e in raw.parse()]
    assert kinds[0] == "message_start"
    (span,) = _llm_spans(spans)
    assert _output(span)[0]["parts"] == [{"type": "text", "content": "Bonjour"}]
