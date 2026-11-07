"""Fake OpenAI and Anthropic HTTP servers for tests. Localhost only, deterministic, free.

Routes:
- POST /v1/chat/completions  (JSON or SSE stream; tool calls when the last user message asks
  about "weather" and no tool result follows it)
- POST /v1/responses         (JSON or SSE event stream; function_call items the same way)
- POST /v1/messages          (Anthropic Messages, JSON or SSE stream; tool_use the same way)
A model named "error-500" or "error-400" returns that status.

Run alone: python -m testing.fake_llm --port 18080
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, Optional


def _last_user_text(messages: Iterable[Any]) -> str:
    text = ""
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user" or msg.get("type") == "message" and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and
                                p.get("type") in ("text", "input_text"))
    return text


def _has_tool_result_after_user(messages: list) -> bool:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool" or msg.get("type") == "function_call_output":
            return True
        content = msg.get("content")
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "tool_result" for p in content):
            return True
        if msg.get("role") == "user":
            return False
    return False


def _wants_tool(body: dict, messages: list) -> bool:
    return bool(body.get("tools")) and "weather" in _last_user_text(messages).lower() and \
        not _has_tool_result_after_user(messages)


class FakeLLM:
    def __init__(self, port: int = 0, latency_s: float = 0.0) -> None:
        self.calls: list[dict] = []
        self.latency_s = latency_s
        owner = self

        class Handler(_Handler):
            server_owner = owner

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._counter = 0
        self._lock = threading.Lock()

    def next_id(self, prefix: str) -> str:
        with self._lock:
            self._counter += 1
            return f"{prefix}{self._counter:06d}"

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def openai_base_url(self) -> str:
        return self.url + "/v1"

    def start(self) -> "FakeLLM":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def __enter__(self) -> "FakeLLM":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


class _Handler(BaseHTTPRequestHandler):
    server_owner: FakeLLM
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, events: Iterable[tuple[Optional[str], Any]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for name, data in events:
            chunk = ""
            if name:
                chunk += f"event: {name}\n"
            chunk += f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"
            self.wfile.write(chunk.encode())
            self.wfile.flush()
        self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        owner = self.server_owner
        owner.calls.append({"path": self.path, "body": body, "headers": dict(self.headers.items())})
        if owner.latency_s:
            time.sleep(owner.latency_s)
        model = str(body.get("model", ""))
        if model.startswith("error-"):
            code = int(model.split("-")[1])
            if self.path.endswith("/messages"):
                return self._json(code, {"type": "error", "error": {"type": "api_error", "message": "fake error"}})
            return self._json(code, {"error": {"message": "fake error", "type": "server_error", "code": None}})
        if self.path.endswith("/chat/completions"):
            return self._chat(body)
        if self.path.endswith("/responses"):
            return self._responses(body)
        if self.path.endswith("/messages"):
            return self._messages(body)
        self._json(404, {"error": {"message": "not found"}})

    # ------------------------------------------------------------------ chat completions

    def _chat(self, body: dict) -> None:
        owner = self.server_owner
        messages = body.get("messages") or []
        cid = owner.next_id("chatcmpl-")
        model = body.get("model", "fake-model")
        tool = _wants_tool(body, messages)
        text = None if tool else "Echo: " + _last_user_text(messages)
        call = {"id": owner.next_id("call_"), "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}
        usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
                 "prompt_tokens_details": {"cached_tokens": 3}}
        finish = "tool_calls" if tool else "stop"
        if not body.get("stream"):
            message = {"role": "assistant", "content": text}
            if tool:
                message["tool_calls"] = [call]
            return self._json(200, {"id": cid, "object": "chat.completion", "created": int(time.time()),
                                    "model": model, "choices": [{"index": 0, "message": message,
                                                                 "finish_reason": finish, "logprobs": None}],
                                    "usage": usage})
        base = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
        events: list[tuple[Optional[str], Any]] = [
            (None, {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                         "finish_reason": None}]})]
        if tool:
            events.append((None, {**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": call["id"], "type": "function",
                 "function": {"name": "get_weather", "arguments": ""}}]}, "finish_reason": None}]}))
            args = call["function"]["arguments"]
            for piece in (args[:5], args[5:]):
                events.append((None, {**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": piece}}]}, "finish_reason": None}]}))
        else:
            for word in text.split(" "):
                events.append((None, {**base, "choices": [{"index": 0, "delta": {"content": word + " "},
                                                           "finish_reason": None}]}))
        events.append((None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}))
        if (body.get("stream_options") or {}).get("include_usage"):
            events.append((None, {**base, "choices": [], "usage": usage}))
        events.append((None, "[DONE]"))
        self._sse(events)

    # ------------------------------------------------------------------ responses

    def _responses(self, body: dict) -> None:
        owner = self.server_owner
        raw_input = body.get("input")
        items = [{"role": "user", "content": raw_input}] if isinstance(raw_input, str) else (raw_input or [])
        rid = owner.next_id("resp_")
        model = body.get("model", "fake-model")
        tool = _wants_tool(body, items)
        if tool:
            output = [{"type": "function_call", "id": owner.next_id("fc_"), "call_id": owner.next_id("call_"),
                       "name": "get_weather", "arguments": json.dumps({"city": "Paris"}), "status": "completed"}]
        else:
            output = [{"type": "message", "id": owner.next_id("msg_"), "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": "Echo: " + _last_user_text(items),
                                    "annotations": []}]}]
        response = {"id": rid, "object": "response", "created_at": int(time.time()), "model": model,
                    "status": "completed", "output": output, "parallel_tool_calls": True, "tool_choice": "auto",
                    "tools": body.get("tools") or [], "previous_response_id": body.get("previous_response_id"),
                    "usage": {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18,
                              "input_tokens_details": {"cached_tokens": 0},
                              "output_tokens_details": {"reasoning_tokens": 2}}}
        if not body.get("stream"):
            return self._json(200, response)
        seq = 0

        def ev(kind: str, **fields: Any) -> tuple[str, dict]:
            nonlocal seq
            seq += 1
            return kind, {"type": kind, "sequence_number": seq, **fields}

        pending = {**response, "status": "in_progress", "output": [], "usage": None}
        events = [ev("response.created", response=pending), ev("response.in_progress", response=pending)]
        for index, item in enumerate(output):
            events.append(ev("response.output_item.added", output_index=index,
                             item={**item, "status": "in_progress"} if item["type"] == "function_call"
                             else {**item, "content": [], "status": "in_progress"}))
            if item["type"] == "message":
                text = item["content"][0]["text"]
                events.append(ev("response.content_part.added", item_id=item["id"], output_index=index,
                                 content_index=0, part={"type": "output_text", "text": "", "annotations": []}))
                for word in text.split(" "):
                    events.append(ev("response.output_text.delta", item_id=item["id"], output_index=index,
                                     content_index=0, delta=word + " ", logprobs=[]))
                events.append(ev("response.content_part.done", item_id=item["id"], output_index=index,
                                 content_index=0, part=item["content"][0]))
            else:
                events.append(ev("response.function_call_arguments.delta", item_id=item["id"],
                                 output_index=index, delta=item["arguments"]))
            events.append(ev("response.output_item.done", output_index=index, item=item))
        events.append(ev("response.completed", response=response))
        self._sse(events)

    # ------------------------------------------------------------------ anthropic messages

    def _messages(self, body: dict) -> None:
        owner = self.server_owner
        messages = body.get("messages") or []
        mid = owner.next_id("msg_")
        model = body.get("model", "fake-claude")
        tool = _wants_tool(body, messages)
        if tool:
            content = [{"type": "text", "text": "Let me check."},
                       {"type": "tool_use", "id": owner.next_id("toolu_"), "name": "get_weather",
                        "input": {"city": "Paris"}}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": "Echo: " + _last_user_text(messages)}]
            stop = "end_turn"
        usage = {"input_tokens": 13, "output_tokens": 8, "cache_read_input_tokens": 2}
        message = {"id": mid, "type": "message", "role": "assistant", "model": model, "content": content,
                   "stop_reason": stop, "stop_sequence": None, "usage": usage}
        if not body.get("stream"):
            return self._json(200, message)
        events: list[tuple[Optional[str], Any]] = [
            ("message_start", {"type": "message_start", "message": {**message, "content": [], "stop_reason": None,
                                                                     "usage": {"input_tokens": 13,
                                                                               "output_tokens": 1}}})]
        for index, block in enumerate(content):
            if block["type"] == "text":
                events.append(("content_block_start", {"type": "content_block_start", "index": index,
                                                       "content_block": {"type": "text", "text": ""}}))
                for word in block["text"].split(" "):
                    events.append(("content_block_delta", {"type": "content_block_delta", "index": index,
                                                           "delta": {"type": "text_delta", "text": word + " "}}))
            else:
                events.append(("content_block_start", {"type": "content_block_start", "index": index,
                                                       "content_block": {"type": "tool_use", "id": block["id"],
                                                                         "name": block["name"], "input": {}}}))
                raw = json.dumps(block["input"])
                for piece in (raw[:4], raw[4:]):
                    events.append(("content_block_delta", {"type": "content_block_delta", "index": index,
                                                           "delta": {"type": "input_json_delta",
                                                                     "partial_json": piece}}))
            events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
        events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop,
                                                                            "stop_sequence": None},
                                         "usage": {"output_tokens": 8}}))
        events.append(("message_stop", {"type": "message_stop"}))
        self._sse(events)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake OpenAI and Anthropic servers")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    llm = FakeLLM(args.port)
    print(llm.url, flush=True)
    llm.server.serve_forever()


if __name__ == "__main__":
    main()
