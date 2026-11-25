"""OpenAI capture shapes found by the 2026-11-25 SDK simulations: one message per Responses output item,
built-in tool results, reasoning from compatible providers, annotations, Groq usage, provider names."""

import json

from zoroval.integrations.openai import (_ChatStreamState, _chat_finish, _provider_name, _responses_output,
                                         chat_input_messages, responses_input_messages)

WEB_SOURCES = [{"type": "url", "url": "https://science.nasa.gov/webb"}]
CITE = {"type": "url_citation", "start_index": 0, "end_index": 4, "url": "https://science.nasa.gov/webb",
        "title": "Webb"}
OUTPUT = [
    {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Search first."}]},
    {"type": "web_search_call", "id": "ws_1", "status": "completed",
     "action": {"type": "search", "query": "webb news", "sources": WEB_SOURCES}},
    {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "save", "arguments": "{}"},
    {"type": "message", "role": "assistant", "id": "msg_1",
     "content": [{"type": "output_text", "text": "Webb found ice.", "annotations": [CITE]}]},
]


class Gen:
    """Stands in for Generation: keeps what finish() was given."""

    def __init__(self):
        self.done = False
        self.kwargs = None

    def finish(self, **kwargs):
        self.kwargs = kwargs
        self.done = True


def test_one_message_per_output_item():
    messages, calls = _responses_output({"output": OUTPUT})
    assert [m["role"] for m in messages] == ["assistant"] * 4
    assert messages[0]["parts"] == [{"type": "reasoning", "content": "Search first."}]
    assert [p["type"] for p in messages[1]["parts"]] == ["tool_call", "tool_call_response"]
    assert messages[2]["parts"] == [{"type": "tool_call", "id": "call_1", "name": "save", "arguments": "{}"}]
    assert [c["name"] for c in calls] == ["web_search", "save"]


def test_output_matches_input_history():
    # The app sends the output items back as history: the messages must be the same, or the reader
    # cannot tell they are the reply it already showed.
    messages, _ = _responses_output({"output": OUTPUT})
    history = responses_input_messages([{"role": "user", "content": "news?"}] + OUTPUT)
    assert history[1:] == messages


def test_web_search_result_and_query_only_arguments():
    messages, calls = _responses_output({"output": OUTPUT[1:2]})
    call, result = messages[0]["parts"]
    assert json.loads(call["arguments"]) == {"query": "webb news"}
    assert result == {"type": "tool_call_response", "id": "ws_1", "response": WEB_SOURCES}
    assert json.loads(calls[0]["arguments"]) == {"query": "webb news"}


def test_open_page_action_keeps_its_url():
    item = {"type": "web_search_call", "id": "ws_2", "action": {"type": "open_page", "url": "https://x.test"}}
    messages, _ = _responses_output({"output": [item]})
    assert json.loads(messages[0]["parts"][0]["arguments"]) == {"type": "open_page", "url": "https://x.test"}
    assert len(messages[0]["parts"]) == 1  # no sources: no result part


def test_file_search_code_and_mcp_results():
    output = [
        {"type": "file_search_call", "id": "fs_1", "queries": ["policy"],
         "results": [{"file_id": "f1", "filename": "h.pdf", "score": 0.9, "text": "Three days."}]},
        {"type": "code_interpreter_call", "id": "ci_1", "code": "print(1)",
         "outputs": [{"type": "logs", "logs": "1\n"}, {"type": "image", "url": "data:image/png;base64,AAAA"}]},
        {"type": "mcp_call", "id": "mcp_1", "name": "lookup", "arguments": "{}", "server_label": "s",
         "output": None, "error": "server down"},
        {"type": "image_generation_call", "id": "ig_1", "result": "iVBORw0KGgo" * 50},
    ]
    messages, _ = _responses_output({"output": output})
    results = [m["parts"][1] for m in messages]
    assert results[0]["response"][0]["text"] == "Three days."
    assert results[1]["response"] == [{"type": "logs", "logs": "1\n"}, {"type": "image", "url": "[media omitted]"}]
    assert results[2] == {"type": "tool_call_response", "id": "mcp_1", "response": "server down", "is_error": True}
    assert results[3]["response"] == "[image omitted]"
    assert "iVBOR" not in json.dumps(messages)


def test_computer_call_output_pairs_by_call_id():
    history = [{"type": "computer_call", "id": "cu_1", "call_id": "call_cu", "action": {"type": "click"}},
               {"type": "computer_call_output", "call_id": "call_cu",
                "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AAAA"}}]
    out = responses_input_messages(history)
    assert out[0]["parts"][0]["id"] == out[1]["parts"][0]["id"] == "call_cu"
    assert out[1]["parts"][0]["response"]["image_url"] == "[media omitted]"


def test_annotations_kept_on_both_sides():
    messages, _ = _responses_output({"output": OUTPUT[3:]})
    assert messages[0]["parts"][0]["annotations"] == [CITE]
    history = responses_input_messages([OUTPUT[3]])
    assert history[0]["parts"][0]["annotations"] == [CITE]
    plain = responses_input_messages([{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}])
    assert plain[0]["parts"] == [{"type": "text", "content": "hi"}]


def test_chat_annotations_on_message():
    result = {"id": "c1", "model": "m", "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": "Webb found ice.", "annotations": [CITE]}}]}
    gen = Gen()
    _chat_finish(gen, result)
    part = gen.kwargs["output_messages"][0]["parts"][0]
    assert part["annotations"] == [CITE]
    back = chat_input_messages([{"role": "assistant", "content": "Webb found ice.", "annotations": [CITE]}])
    assert back[0]["parts"] == gen.kwargs["output_messages"][0]["parts"]


def test_deepseek_reasoning_content():
    result = {"id": "c1", "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": "Peak is 8 to 10.", "reasoning_content": "Western line peaks early."}}]}
    gen = Gen()
    _chat_finish(gen, result)
    reasoning, reply = gen.kwargs["output_messages"]
    assert reasoning == {"role": "assistant", "parts": [{"type": "reasoning", "content": "Western line peaks early."}]}
    assert reply["parts"] == [{"type": "text", "content": "Peak is 8 to 10."}] and reply["finish_reason"] == "stop"
    # DeepSeek refuses reasoning_content in history: the app re-sends the reply alone, which still matches
    back = chat_input_messages([{"role": "assistant", "content": "Peak is 8 to 10."}])
    assert back[0]["parts"] == reply["parts"]
    with_it = chat_input_messages([{"role": "assistant", "content": "Peak is 8 to 10.",
                                    "reasoning_content": "Western line peaks early."}])
    assert [m["parts"] for m in with_it] == [reasoning["parts"], reply["parts"]]


def test_openrouter_reasoning_read_once():
    message = {"role": "assistant", "content": "Five, seven, five.", "reasoning": "Simple question.",
               "reasoning_details": [{"type": "reasoning.text", "text": "Simple question.", "index": 0}]}
    gen = Gen()
    _chat_finish(gen, {"choices": [{"finish_reason": "stop", "message": message}]})
    messages = gen.kwargs["output_messages"]
    assert [m["parts"][0]["type"] for m in messages] == ["reasoning", "text"]
    assert messages[0]["parts"] == [{"type": "reasoning", "content": "Simple question."}]
    only_details = dict(message, reasoning=None)
    gen = Gen()
    _chat_finish(gen, {"choices": [{"finish_reason": "stop", "message": only_details}]})
    assert gen.kwargs["output_messages"][0]["parts"][0] == {"type": "reasoning", "content": "Simple question."}


def _stream(chunks):
    gen = Gen()
    state = _ChatStreamState(gen)
    for chunk in chunks:
        state.observe(chunk)
    state.finish(None)
    return gen.kwargs


def test_stream_reasoning_deltas():
    out = _stream([
        {"id": "c", "choices": [{"index": 0, "delta": {"reasoning_content": "Fast "}}]},
        {"id": "c", "choices": [{"index": 0, "delta": {"reasoning_content": "trains."}}]},
        {"id": "c", "choices": [{"index": 0, "delta": {"content": "Take the 7:10."}}]},
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ])
    assert [m["parts"] for m in out["output_messages"]] == [[{"type": "reasoning", "content": "Fast trains."}],
                                                            [{"type": "text", "content": "Take the 7:10."}]]
    out = _stream([  # OpenRouter sends the same text in reasoning and reasoning_details
        {"choices": [{"index": 0, "delta": {"reasoning": "Short ", "reasoning_details": [
            {"type": "reasoning.text", "text": "Short ", "index": 0}]}}]},
        {"choices": [{"index": 0, "delta": {"reasoning": "one.", "reasoning_details": [
            {"type": "reasoning.text", "text": "one.", "index": 0}]}}]},
        {"choices": [{"index": 0, "delta": {"content": "Yes."}, "finish_reason": "stop"}]},
    ])
    assert out["output_messages"][0]["parts"][0] == {"type": "reasoning", "content": "Short one."}


def test_groq_usage_in_x_groq():
    out = _stream([
        {"id": "c", "choices": [{"index": 0, "delta": {"content": "Hi"}}]},
        {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "x_groq": {"id": "req_1", "usage": {"prompt_tokens": 12, "completion_tokens": 3}}},
    ])
    assert out["usage"] == {"input_tokens": 12, "output_tokens": 3}
    out = _stream([{"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    "x_groq": {"usage": {"prompt_tokens": 99, "completion_tokens": 99}}}])
    assert out["usage"]["input_tokens"] == 5  # top-level usage wins


class _URL:
    def __init__(self, host):
        self.host = host
        self.port = None


class _Client:
    def __init__(self, host):
        self.base_url = _URL(host)


class _Resource:
    def __init__(self, host):
        self._client = _Client(host)


def test_provider_name_from_base_url():
    assert _provider_name(_Resource("api.openai.com")) == "openai"
    assert _provider_name(_Resource("api.groq.com")) == "groq"
    assert _provider_name(_Resource("api.deepseek.com")) == "deepseek"
    assert _provider_name(_Resource("openrouter.ai")) == "openrouter"
    assert _provider_name(_Resource("api.anthropic.com")) == "anthropic"
    assert _provider_name(_Resource("127.0.0.1")) == "openai"
    assert _provider_name(_Resource("evil-api.groq.com.example")) == "openai"
    assert _provider_name(_Resource("bedrock-runtime.us-west-2.amazonaws.com")) == "aws.bedrock"
    assert _provider_name(_Resource("bedrock-mantle.us-east-1.api.aws")) == "aws.bedrock"
    assert _provider_name(_Resource("abc123.execute-api.us-east-1.amazonaws.com")) == "openai"
