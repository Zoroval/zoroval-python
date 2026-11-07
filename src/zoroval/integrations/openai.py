"""OpenAI wrapper: Chat Completions and Responses, sync and async, streaming, tool-call ids.

Covers `create`, `parse` and the `stream()` helpers (they call `create(stream=True)`), for the
official client, OpenAI-compatible base URLs and Azure OpenAI. Spans follow the OpenTelemetry
GenAI semantic conventions; content goes to gen_ai.input.messages and gen_ai.output.messages.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

from .. import _context, _json
from .._util import internal_error
from ._common import (MARK, AsyncStream, Generation, SyncStream, foreign_wrapper, get, given, media_part,
                      is_ours, safe, text_part, wrapped_over)

REQUEST_PARAMS = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "max_tokens": "gen_ai.request.max_tokens",
    "max_completion_tokens": "gen_ai.request.max_tokens",
    "max_output_tokens": "gen_ai.request.max_tokens",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
    "seed": "gen_ai.request.seed",
    "n": "gen_ai.request.choice.count",
}

_originals: dict[tuple[type, str], Any] = {}


# ---------------------------------------------------------------------------- shared request bits

def _client_request_wrapped(resource: Any) -> bool:
    """OpenInference's OpenAI instrumentor wraps `OpenAI.request` (below `create`). When it is
    active, its span records the call, so we make none (no double spans in either backend)."""
    client = getattr(resource, "_client", None)
    if client is None:
        return False
    try:
        import inspect
        import types
        func = inspect.getattr_static(type(client), "request")
    except AttributeError:
        return False
    return type(func) is not types.FunctionType



def _provider_name(resource: Any) -> str:
    client = getattr(resource, "_client", None)
    name = type(client).__name__ if client is not None else ""
    return "azure.ai.openai" if "Azure" in name else "openai"


def _request_attributes(resource: Any, kwargs: dict) -> dict[str, Any]:
    attrs: dict[str, Any] = {"gen_ai.request.stream": bool(get(kwargs, "stream", False))}
    base_url = getattr(getattr(resource, "_client", None), "base_url", None)
    host = getattr(base_url, "host", None)
    if host:
        attrs["server.address"] = str(host)
        port = getattr(base_url, "port", None)
        if port:
            attrs["server.port"] = int(port)
    for key, name in REQUEST_PARAMS.items():
        value = get(kwargs, key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attrs[name] = value
    stop = get(kwargs, "stop")
    if stop:
        attrs["gen_ai.request.stop_sequences"] = [stop] if isinstance(stop, str) else [str(s) for s in stop]
    return attrs


def _content_parts(content: Any) -> list[dict]:
    if content is None or not given(content):
        return []
    if isinstance(content, str):
        return [text_part(content)]
    parts = []
    for item in content:
        kind = get(item, "type")
        if kind in ("text", "input_text", "output_text"):
            parts.append(text_part(get(item, "text")))
        elif kind == "refusal":
            parts.append({"type": "refusal", "content": get(item, "refusal")})
        elif kind in ("image_url", "input_image"):
            image = get(item, "image_url")
            url = get(image, "url") if not isinstance(image, str) else image
            parts.append(media_part("image", url))
        elif kind in ("input_audio", "audio"):
            parts.append(media_part("audio"))
        elif kind in ("file", "input_file"):
            parts.append(media_part("file"))
        else:
            parts.append({"type": str(kind or "unknown")})
    return parts


# ---------------------------------------------------------------------------- Chat Completions

def chat_input_messages(messages: Any) -> list[dict]:
    out = []
    for msg in messages or []:
        role = get(msg, "role")
        if role in ("tool", "function"):
            parts = [{"type": "tool_call_response", "id": get(msg, "tool_call_id"),
                      "response": get(msg, "content")}]
        else:
            parts = _content_parts(get(msg, "content"))
            for call in get(msg, "tool_calls") or []:
                fn = get(call, "function")
                parts.append({"type": "tool_call", "id": get(call, "id"), "name": get(fn, "name"),
                              "arguments": get(fn, "arguments")})
        item: dict[str, Any] = {"role": role, "parts": parts}
        if get(msg, "name"):
            item["name"] = get(msg, "name")
        out.append(item)
    return out


def _chat_tool_calls(message: Any) -> list[dict]:
    calls = []
    for call in get(message, "tool_calls") or []:
        fn = get(call, "function")
        calls.append({"id": get(call, "id"), "name": get(fn, "name") or get(call, "name"),
                      "arguments": get(fn, "arguments")})
    return calls


def _chat_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    out = {"input_tokens": get(usage, "prompt_tokens"), "output_tokens": get(usage, "completion_tokens")}
    cached = get(get(usage, "prompt_tokens_details"), "cached_tokens")
    if cached:
        out["cache_read.input_tokens"] = cached
    reasoning = get(get(usage, "completion_tokens_details"), "reasoning_tokens")
    if reasoning:
        out["reasoning.output_tokens"] = reasoning
    return out


def _chat_start(resource: Any, kwargs: dict) -> Generation:
    gen = Generation(provider=_provider_name(resource), model=get(kwargs, "model"),
                     request_attributes=_request_attributes(resource, kwargs))
    if gen.active:
        gen.content("gen_ai.input.messages", chat_input_messages(get(kwargs, "messages")))
        tools = get(kwargs, "tools")
        if tools:
            gen.content("gen_ai.tool.definitions", _json.to_plain(list(tools)))
    return gen


def _chat_finish(gen: Generation, result: Any) -> None:
    if not hasattr(result, "choices") and hasattr(result, "parse"):
        result = result.parse()  # with_raw_response: LegacyAPIResponse
    outputs, calls, reasons = [], [], []
    for choice in get(result, "choices") or []:
        message = get(choice, "message")
        choice_calls = _chat_tool_calls(message)
        parts = _content_parts(get(message, "content"))
        if get(message, "refusal"):
            parts.append({"type": "refusal", "content": get(message, "refusal")})
        parts += [{"type": "tool_call", **c} for c in choice_calls]
        reason = get(choice, "finish_reason")
        outputs.append({"role": "assistant", "parts": parts, "finish_reason": reason})
        calls += choice_calls
        reasons.append(reason)
    gen.finish(response_id=get(result, "id"), model=get(result, "model"), finish_reasons=reasons,
               output_messages=outputs, tool_calls=calls, usage=_chat_usage(get(result, "usage")))


class _ChatStreamState:
    def __init__(self, gen: Generation) -> None:
        self.gen = gen
        self.id: Optional[str] = None
        self.model: Optional[str] = None
        self.usage: Any = None
        self.choices: dict[int, dict] = {}

    def observe(self, chunk: Any) -> None:
        self.id = self.id or get(chunk, "id")
        self.model = self.model or get(chunk, "model")
        if get(chunk, "usage"):
            self.usage = get(chunk, "usage")
        for choice in get(chunk, "choices") or []:
            slot = self.choices.setdefault(get(choice, "index", 0) or 0,
                                           {"text": [], "refusal": [], "calls": {}, "finish": None})
            delta = get(choice, "delta")
            if delta is not None:
                if get(delta, "content"):
                    slot["text"].append(get(delta, "content"))
                if get(delta, "refusal"):
                    slot["refusal"].append(get(delta, "refusal"))
                for call in get(delta, "tool_calls") or []:
                    entry = slot["calls"].setdefault(get(call, "index", 0) or 0,
                                                     {"id": None, "name": None, "arguments": ""})
                    if get(call, "id"):
                        entry["id"] = get(call, "id")
                    fn = get(call, "function")
                    if get(fn, "name"):
                        entry["name"] = get(fn, "name")
                    if get(fn, "arguments"):
                        entry["arguments"] += get(fn, "arguments")
            if get(choice, "finish_reason"):
                slot["finish"] = get(choice, "finish_reason")

    def finish(self, error: Optional[BaseException]) -> None:
        outputs, calls, reasons = [], [], []
        for index in sorted(self.choices):
            slot = self.choices[index]
            choice_calls = [slot["calls"][i] for i in sorted(slot["calls"])]
            parts = [text_part("".join(slot["text"]))] if slot["text"] else []
            if slot["refusal"]:
                parts.append({"type": "refusal", "content": "".join(slot["refusal"])})
            parts += [{"type": "tool_call", **c} for c in choice_calls]
            outputs.append({"role": "assistant", "parts": parts, "finish_reason": slot["finish"]})
            calls += choice_calls
            reasons.append(slot["finish"])
        self.gen.finish(response_id=self.id, model=self.model, finish_reasons=reasons, output_messages=outputs,
                        tool_calls=calls, usage=_chat_usage(self.usage), error=error)


# ---------------------------------------------------------------------------- Responses

def responses_input_messages(value: Any) -> list[dict]:
    if value is None or not given(value):
        return []
    if isinstance(value, str):
        return [{"role": "user", "parts": [text_part(value)]}]
    out = []
    for item in value:
        kind = get(item, "type")
        if kind == "function_call":
            out.append({"role": "assistant", "parts": [{"type": "tool_call", "id": get(item, "call_id"),
                                                        "name": get(item, "name"),
                                                        "arguments": get(item, "arguments")}]})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            out.append({"role": "tool", "parts": [{"type": "tool_call_response", "id": get(item, "call_id"),
                                                   "response": _json.to_plain(get(item, "output"))}]})
        elif kind == "custom_tool_call":
            out.append({"role": "assistant", "parts": [{"type": "tool_call", "id": get(item, "call_id"),
                                                        "name": get(item, "name"),
                                                        "arguments": get(item, "input")}]})
        elif kind in (None, "message") or get(item, "role"):
            out.append({"role": get(item, "role", "user"), "parts": _content_parts(get(item, "content"))})
        elif kind == "reasoning":
            continue
        else:
            out.append({"role": "unknown", "parts": [{"type": str(kind)}]})
    return out


def _responses_output(response: Any) -> tuple[list[dict], list[dict]]:
    parts: list[dict] = []
    calls: list[dict] = []
    for item in get(response, "output") or []:
        kind = get(item, "type")
        if kind == "message":
            parts += _content_parts(get(item, "content"))
        elif kind in ("function_call", "custom_tool_call"):
            call = {"id": get(item, "call_id"), "name": get(item, "name"),
                    "arguments": get(item, "arguments") if kind == "function_call" else get(item, "input")}
            calls.append(call)
            parts.append({"type": "tool_call", **call})
        elif kind == "reasoning":
            summary = " ".join(get(s, "text") or "" for s in get(item, "summary") or [])
            if summary:
                parts.append({"type": "reasoning", "content": summary})
        elif kind:
            parts.append({"type": str(kind)})
    return parts, calls


def _responses_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    out = {"input_tokens": get(usage, "input_tokens"), "output_tokens": get(usage, "output_tokens")}
    cached = get(get(usage, "input_tokens_details"), "cached_tokens")
    if cached:
        out["cache_read.input_tokens"] = cached
    reasoning = get(get(usage, "output_tokens_details"), "reasoning_tokens")
    if reasoning:
        out["reasoning.output_tokens"] = reasoning
    return out


def _responses_start(resource: Any, kwargs: dict) -> Generation:
    attrs = _request_attributes(resource, kwargs)
    previous = get(kwargs, "previous_response_id")
    if previous:
        attrs["gen_ai.openai.previous_response_id"] = str(previous)
        _context.note_previous_response()
    conv = get(kwargs, "conversation")
    conv_id = conv if isinstance(conv, str) else get(conv, "id")
    if conv_id and _context.current_conversation_id() is None:
        attrs["gen_ai.conversation.id"] = str(conv_id)  # OpenAI Conversations API id: an inherited id
    gen = Generation(provider=_provider_name(resource), model=get(kwargs, "model"), request_attributes=attrs)
    if gen.active:
        instructions = get(kwargs, "instructions")
        if instructions:
            gen.content("gen_ai.system_instructions", [text_part(instructions)])
        gen.content("gen_ai.input.messages", responses_input_messages(get(kwargs, "input")))
        tools = get(kwargs, "tools")
        if tools:
            gen.content("gen_ai.tool.definitions", _json.to_plain(list(tools)))
    return gen


def _responses_finish(gen: Generation, response: Any, error: Optional[BaseException] = None) -> None:
    if not hasattr(response, "output") and hasattr(response, "parse"):
        response = response.parse()
    parts, calls = _responses_output(response)
    status = get(response, "status")
    reason = get(get(response, "incomplete_details"), "reason") or status
    outputs = [{"role": "assistant", "parts": parts, "finish_reason": reason}] if parts or reason else []
    if status == "failed" and error is None:
        err = get(response, "error")
        gen.set("error.type", str(get(err, "code") or "response_failed"))
        from opentelemetry.trace import Status, StatusCode
        if gen.span is not None:
            gen.span.set_status(Status(StatusCode.ERROR, str(get(err, "message") or "response failed")[:32768]))
    gen.finish(response_id=get(response, "id"), model=get(response, "model"),
               finish_reasons=[reason] if reason else [], output_messages=outputs, tool_calls=calls,
               usage=_responses_usage(get(response, "usage")), error=error)


class _ResponsesStreamState:
    def __init__(self, gen: Generation) -> None:
        self.gen = gen
        self.response: Any = None
        self.id: Optional[str] = None
        self.model: Optional[str] = None
        self.text: list[str] = []
        self.items: list[Any] = []
        self.error: Any = None

    def observe(self, event: Any) -> None:
        kind = get(event, "type")
        if kind in ("response.created", "response.in_progress"):
            response = get(event, "response")
            self.id = self.id or get(response, "id")
            self.model = self.model or get(response, "model")
        elif kind == "response.output_text.delta":
            self.text.append(get(event, "delta") or "")
        elif kind == "response.output_item.done":
            self.items.append(get(event, "item"))
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            self.response = get(event, "response")
        elif kind == "error":
            self.error = event

    def finish(self, error: Optional[BaseException]) -> None:
        if self.response is not None:
            _responses_finish(self.gen, self.response, error)
            return
        parts: list[dict] = []
        calls: list[dict] = []
        if self.items:
            parts, calls = _responses_output({"output": self.items})
        elif self.text:
            parts = [text_part("".join(self.text))]
        if self.error is not None and error is None and self.gen.span is not None:
            from opentelemetry.trace import Status, StatusCode
            self.gen.span.set_status(Status(StatusCode.ERROR, str(get(self.error, "message") or "stream error")))
        self.gen.finish(response_id=self.id, model=self.model, finish_reasons=["incomplete"],
                        output_messages=[{"role": "assistant", "parts": parts, "finish_reason": "incomplete"}],
                        tool_calls=calls, error=error)


# ---------------------------------------------------------------------------- wrappers

def _wrap_sync(original: Any, start: Any, finish: Any, state_cls: Any, owner: type, method: str) -> Any:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if wrapped_over(owner, method, wrapper) or _client_request_wrapped(self):
            return original(self, *args, **kwargs)
        gen = safe("openai.start", start, self, kwargs)
        if gen is None or not gen.active:
            return original(self, *args, **kwargs)
        gen.attach()
        try:
            result = original(self, *args, **kwargs)
        except BaseException as error:
            gen.detach()
            safe("openai.fail", gen.fail, error)
            raise
        gen.detach()
        try:
            if get(kwargs, "stream", False) and not hasattr(result, "choices") and not hasattr(result, "output"):
                if hasattr(result, "http_response"):
                    gen.finish()  # raw streaming response: the app reads the bytes itself
                    return result
                state = state_cls(gen)
                return SyncStream(result, gen, state.observe, state.finish)
            finish(gen, result)
        except Exception as error:  # noqa: BLE001
            internal_error("openai.finish", error)
            safe("openai.end", gen.finish)
        return result

    setattr(wrapper, MARK, True)
    return wrapper


def _wrap_async(original: Any, start: Any, finish: Any, state_cls: Any, owner: type, method: str) -> Any:
    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if wrapped_over(owner, method, wrapper) or _client_request_wrapped(self):
            return await original(self, *args, **kwargs)
        gen = safe("openai.start", start, self, kwargs)
        if gen is None or not gen.active:
            return await original(self, *args, **kwargs)
        gen.attach()
        try:
            result = await original(self, *args, **kwargs)
        except BaseException as error:
            gen.detach()
            safe("openai.fail", gen.fail, error)
            raise
        gen.detach()
        try:
            if get(kwargs, "stream", False) and not hasattr(result, "choices") and not hasattr(result, "output"):
                if hasattr(result, "http_response"):
                    gen.finish()  # raw streaming response: the app reads the bytes itself
                    return result
                state = state_cls(gen)
                return AsyncStream(result, gen, state.observe, state.finish)
            finish(gen, result)
        except Exception as error:  # noqa: BLE001
            internal_error("openai.finish", error)
            safe("openai.end", gen.finish)
        return result

    setattr(wrapper, MARK, True)
    return wrapper


def _targets() -> list[tuple[type, str, bool, Any, Any, Any]]:
    from openai.resources.chat.completions import AsyncCompletions, Completions
    out = []
    for cls, is_async in ((Completions, False), (AsyncCompletions, True)):
        for method in ("create", "parse"):
            out.append((cls, method, is_async, _chat_start, _chat_finish, _ChatStreamState))
    try:
        from openai.resources.responses import AsyncResponses, Responses
        for cls, is_async in ((Responses, False), (AsyncResponses, True)):
            for method in ("create", "parse"):
                out.append((cls, method, is_async, _responses_start, _responses_finish, _ResponsesStreamState))
    except ImportError:
        pass
    return out


def instrument() -> bool:
    """Patch the OpenAI client classes. Returns False when openai is missing or someone else wraps it."""
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    wrapped_any = False
    for cls, method, is_async, start, finish, state_cls in _targets():
        current = cls.__dict__.get(method)
        if current is None:
            continue
        if is_ours(current):
            wrapped_any = True
            continue
        if foreign_wrapper(current, "openai"):
            continue  # Langfuse, contrib or OpenInference owns this call; we read their spans
        _originals[(cls, method)] = current
        factory = _wrap_async if is_async else _wrap_sync
        setattr(cls, method, factory(current, start, finish, state_cls, cls, method))
        wrapped_any = True
    return wrapped_any


def uninstrument() -> None:
    for (cls, method), original in list(_originals.items()):
        setattr(cls, method, original)
    _originals.clear()
