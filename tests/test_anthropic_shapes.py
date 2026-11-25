"""Anthropic capture shapes found by the 2026-11-25 SDK simulations: server tools (web search, web fetch)
with their results and citations, streamed and not; input tokens as the total; tool_result text joined."""

import json

from zoroval.integrations.anthropic import _StreamState, _finish, _provider_name, _usage, input_messages

CITE = {"type": "web_search_result_location", "url": "https://www.louvre.fr/en/visit/hours", "title": "Hours",
        "encrypted_index": "EpMBCioIARgB", "cited_text": "Closed on Tuesdays."}
RESULTS = [{"type": "web_search_result", "url": "https://www.louvre.fr/en/visit/hours", "title": "Hours",
            "encrypted_content": "EqgfCioIARgBIiQ3", "page_age": "November 2, 2026"}]
FETCH = {"type": "web_fetch_result", "url": "https://www.louvre.fr/en/visit/hours", "content": {
    "type": "document", "title": "Hours", "source": {"type": "base64", "media_type": "application/pdf",
                                                     "data": "JVBERi0xLjcK" * 20}}}
CONTENT = [
    {"type": "text", "text": "I'll look it up."},
    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "Louvre hours"}},
    {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": RESULTS},
    {"type": "server_tool_use", "id": "srvtoolu_2", "name": "web_fetch", "input": {"url": RESULTS[0]["url"]}},
    {"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_2", "content": FETCH},
    {"type": "text", "text": "It is closed on Tuesdays.", "citations": [CITE]},
]


class Gen:
    def __init__(self):
        self.done = False
        self.kwargs = None

    def finish(self, **kwargs):
        self.kwargs = kwargs
        self.done = True


def _finished(message):
    gen = Gen()
    _finish(gen, message)
    return gen.kwargs


def test_server_tools_not_streamed():
    out = _finished({"id": "msg_1", "model": "m", "content": CONTENT, "stop_reason": "end_turn"})
    parts = out["output_messages"][0]["parts"]
    assert [p["type"] for p in parts] == ["text", "tool_call", "tool_call_response", "tool_call",
                                         "tool_call_response", "text"]
    assert parts[1] == {"type": "tool_call", "id": "srvtoolu_1", "name": "web_search",
                        "arguments": '{"query":"Louvre hours"}'}
    assert parts[2]["id"] == "srvtoolu_1" and parts[2]["response"][0]["url"] == RESULTS[0]["url"]
    assert parts[4]["id"] == "srvtoolu_2"
    dumped = json.dumps(parts)
    assert "encrypted" not in dumped and "JVBERi0" not in dumped and "[base64 omitted]" in dumped
    assert parts[5]["citations"] == [{k: v for k, v in CITE.items() if k != "encrypted_index"}]
    assert [c["name"] for c in out["tool_calls"]] == ["web_search", "web_fetch"]


def test_history_matches_output():
    out = _finished({"content": CONTENT, "stop_reason": "end_turn"})
    history = input_messages([{"role": "assistant", "content": CONTENT}])
    assert history[0]["parts"] == out["output_messages"][0]["parts"]


def test_server_tools_streamed():
    events = [
        {"type": "message_start", "message": {"id": "msg_2", "model": "m", "usage": {
            "input_tokens": 900, "output_tokens": 1, "cache_read_input_tokens": 100}}},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "server_tool_use", "id": "srvtoolu_3", "name": "web_search", "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"query": "dress'}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": ' code"}'}},
        {"type": "content_block_start", "index": 1, "content_block": {
            "type": "web_search_tool_result", "tool_use_id": "srvtoolu_3", "content": RESULTS}},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "citations_delta", "citation": CITE}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "Smart attire."}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 40}},
    ]
    gen = Gen()
    state = _StreamState(gen)
    for event in events:
        state.observe(event)
    state.finish(None)
    parts = gen.kwargs["output_messages"][0]["parts"]
    assert parts[0] == {"type": "tool_call", "id": "srvtoolu_3", "name": "web_search",
                        "arguments": '{"query":"dress code"}'}
    assert parts[1]["type"] == "tool_call_response" and parts[1]["id"] == "srvtoolu_3"
    assert parts[2]["content"] == "Smart attire." and parts[2]["citations"][0]["cited_text"] == "Closed on Tuesdays."
    assert gen.kwargs["tool_calls"][0]["name"] == "web_search"
    assert gen.kwargs["usage"]["input_tokens"] == 1000


def test_input_tokens_include_cache():
    usage = _usage({"input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 1024,
                    "cache_creation_input_tokens": 300})
    assert usage == {"input_tokens": 1336, "output_tokens": 5, "cache_read.input_tokens": 1024,
                     "cache_creation.input_tokens": 300}
    assert _usage({"input_tokens": 12, "output_tokens": 5})["input_tokens"] == 12


def test_tool_result_text_blocks_joined():
    history = input_messages([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
        {"type": "text", "text": "line one"}, {"type": "text", "text": "line two"},
        {"type": "image", "source": {"type": "base64", "data": "AAAA"}}]}]}])
    assert history[0]["parts"][0]["response"] == "line one\nline two\n[image omitted]"


def test_server_tool_error_result():
    out = _finished({"content": [
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "x"}},
        {"type": "web_search_tool_result", "tool_use_id": "s1",
         "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}}]})
    assert out["output_messages"][0]["parts"][1]["is_error"] is True


class _URL:
    def __init__(self, host):
        self.host = host
        self.port = None


def _resource(host, cls_name="Anthropic"):
    client = type(cls_name, (), {})()
    client.base_url = _URL(host)
    return type("R", (), {"_client": client})()


def test_provider_name():
    assert _provider_name(_resource("api.anthropic.com")) == "anthropic"
    assert _provider_name(_resource("127.0.0.1")) == "anthropic"
    assert _provider_name(_resource("bedrock-runtime.us-east-1.amazonaws.com", "AnthropicBedrock")) == "aws.bedrock"
    assert _provider_name(_resource("us-east5-aiplatform.googleapis.com", "AnthropicVertex")) == "gcp.vertex_ai"
