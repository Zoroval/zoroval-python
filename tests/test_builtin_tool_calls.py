"""OpenAI's built-in tools (web search, file search, MCP) are recorded as tool calls with their query."""

import json

from zoroval.integrations.openai import _responses_output, responses_input_messages


def test_builtin_tool_calls_in_output():
    response = {"output": [
        {"type": "web_search_call", "id": "ws_1", "status": "completed",
         "action": {"type": "search", "query": "Lord's stadium events"}},
        {"type": "file_search_call", "id": "fs_1", "status": "completed", "queries": ["ticket policy"]},
        {"type": "mcp_call", "id": "mcp_1", "name": "lookup", "arguments": '{"q":"x"}', "server_label": "s"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Check with the organiser."}]},
    ]}
    messages, calls = _responses_output(response)
    parts = [m["parts"][0] for m in messages]
    assert parts[:3] == [
        {"type": "tool_call", "id": "ws_1", "name": "web_search", "arguments": '{"query":"Lord\'s stadium events"}'},
        {"type": "tool_call", "id": "fs_1", "name": "file_search", "arguments": '["ticket policy"]'},
        {"type": "tool_call", "id": "mcp_1", "name": "lookup", "arguments": '{"q":"x"}'},
    ]
    assert [c["name"] for c in calls] == ["web_search", "file_search", "lookup"]


def test_builtin_tool_call_in_input_history():
    history = [{"role": "user", "content": "hi"},
               {"type": "web_search_call", "id": "ws_0", "status": "completed", "action": {"type": "search", "query": "earlier"}}]
    out = responses_input_messages(history)
    assert out[1] == {"role": "assistant", "parts": [
        {"type": "tool_call", "id": "ws_0", "name": "web_search", "arguments": json.dumps({"query": "earlier"}, separators=(",", ":"))}]}


def test_unknown_items_still_show_their_type():
    messages, calls = _responses_output({"output": [{"type": "something_new"}]})
    assert messages == [{"role": "assistant", "parts": [{"type": "something_new"}]}] and calls == []
