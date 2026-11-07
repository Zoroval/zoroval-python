"""Anthropic wrapper: Messages `create`, `parse` and `stream()`, sync and async, with tool_use ids.

The Anthropic client (2.x) has its own OpenTelemetry span per call, metadata only. When our
wrapper is active, that span is a child of our generation span and the processor does not
export it, so each call gives one generation.
"""

from __future__ import annotations

import functools
import json
from typing import Any, Optional

from .. import _json
from .._util import internal_error
from ._common import (MARK, AsyncStream, Generation, SyncStream, foreign_wrapper, get, given, media_part,
                      is_ours, safe, text_part, wrapped_over)

REQUEST_PARAMS = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "top_k": "gen_ai.request.top_k",
    "max_tokens": "gen_ai.request.max_tokens",
}

_originals: dict[tuple[type, str], Any] = {}


def _blocks(content: Any) -> list[dict]:
    if content is None or not given(content):
        return []
    if isinstance(content, str):
        return [text_part(content)]
    parts = []
    for block in content:
        kind = get(block, "type")
        if kind == "text":
            parts.append(text_part(get(block, "text")))
        elif kind == "tool_use" or kind == "server_tool_use":
            parts.append({"type": "tool_call", "id": get(block, "id"), "name": get(block, "name"),
                          "arguments": _json.dumps(_json.to_plain(get(block, "input")))})
        elif kind == "tool_result":
            inner = get(block, "content")
            response = inner if isinstance(inner, str) else [
                get(b, "text") if get(b, "type") == "text" else f"[{get(b, 'type')} omitted]" for b in inner or []]
            part = {"type": "tool_call_response", "id": get(block, "tool_use_id"), "response": response}
            if get(block, "is_error"):
                part["is_error"] = True
            parts.append(part)
        elif kind == "thinking":
            parts.append({"type": "reasoning", "content": get(block, "thinking")})
        elif kind in ("image", "document"):
            source = get(block, "source")
            parts.append(media_part(kind, get(source, "url")))
        elif kind == "redacted_thinking":
            parts.append({"type": "reasoning", "content": "[redacted]"})
        else:
            parts.append({"type": str(kind or "unknown")})
    return parts


def input_messages(messages: Any) -> list[dict]:
    return [{"role": get(m, "role"), "parts": _blocks(get(m, "content"))} for m in messages or []]


def _tool_calls(content: Any) -> list[dict]:
    calls = []
    for block in content or []:
        if get(block, "type") == "tool_use":
            calls.append({"id": get(block, "id"), "name": get(block, "name"),
                          "arguments": _json.dumps(_json.to_plain(get(block, "input")))})
    return calls


def _usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    out = {"input_tokens": get(usage, "input_tokens"), "output_tokens": get(usage, "output_tokens")}
    if get(usage, "cache_read_input_tokens"):
        out["cache_read.input_tokens"] = get(usage, "cache_read_input_tokens")
    if get(usage, "cache_creation_input_tokens"):
        out["cache_creation.input_tokens"] = get(usage, "cache_creation_input_tokens")
    return out


def _start(resource: Any, kwargs: dict, stream: Optional[bool] = None) -> Generation:
    attrs: dict[str, Any] = {"gen_ai.request.stream": bool(get(kwargs, "stream", False) if stream is None else stream)}
    base_url = getattr(getattr(resource, "_client", None), "base_url", None)
    if getattr(base_url, "host", None):
        attrs["server.address"] = str(base_url.host)
        if getattr(base_url, "port", None):
            attrs["server.port"] = int(base_url.port)
    for key, name in REQUEST_PARAMS.items():
        value = get(kwargs, key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attrs[name] = value
    if get(kwargs, "stop_sequences"):
        attrs["gen_ai.request.stop_sequences"] = [str(s) for s in get(kwargs, "stop_sequences")]
    gen = Generation(provider="anthropic", model=get(kwargs, "model"), request_attributes=attrs)
    if gen.active:
        system = get(kwargs, "system")
        if system:
            gen.content("gen_ai.system_instructions", _blocks(system))
        gen.content("gen_ai.input.messages", input_messages(get(kwargs, "messages")))
        tools = get(kwargs, "tools")
        if tools:
            gen.content("gen_ai.tool.definitions", _json.to_plain(list(tools)))
    return gen


def _finish(gen: Generation, message: Any, error: Optional[BaseException] = None) -> None:
    if not hasattr(message, "content") and hasattr(message, "parse"):
        message = message.parse()
    content = get(message, "content")
    reason = get(message, "stop_reason")
    gen.finish(response_id=get(message, "id"), model=get(message, "model"), finish_reasons=[reason],
               output_messages=[{"role": "assistant", "parts": _blocks(content), "finish_reason": reason}],
               tool_calls=_tool_calls(content), usage=_usage(get(message, "usage")), error=error)


class _StreamState:
    """Rebuilds the message from raw stream events."""

    def __init__(self, gen: Generation) -> None:
        self.gen = gen
        self.id: Optional[str] = None
        self.model: Optional[str] = None
        self.usage: dict[str, Any] = {}
        self.blocks: dict[int, dict] = {}
        self.stop_reason: Optional[str] = None

    def observe(self, event: Any) -> None:
        kind = get(event, "type")
        if kind == "message_start":
            message = get(event, "message")
            self.id = get(message, "id")
            self.model = get(message, "model")
            self._take_usage(get(message, "usage"))
        elif kind == "content_block_start":
            block = get(event, "content_block")
            self.blocks[get(event, "index", 0) or 0] = {
                "type": get(block, "type"), "text": get(block, "text") or "", "id": get(block, "id"),
                "name": get(block, "name"), "json": "", "thinking": get(block, "thinking") or "",
                "input": get(block, "input")}
        elif kind == "content_block_delta":
            slot = self.blocks.setdefault(get(event, "index", 0) or 0, {"type": "text", "text": "", "json": "",
                                                                        "thinking": ""})
            delta = get(event, "delta")
            dtype = get(delta, "type")
            if dtype == "text_delta":
                slot["text"] += get(delta, "text") or ""
            elif dtype == "input_json_delta":
                slot["json"] += get(delta, "partial_json") or ""
            elif dtype == "thinking_delta":
                slot["thinking"] += get(delta, "thinking") or ""
        elif kind == "message_delta":
            delta = get(event, "delta")
            if get(delta, "stop_reason"):
                self.stop_reason = get(delta, "stop_reason")
            self._take_usage(get(event, "usage"))

    def _take_usage(self, usage: Any) -> None:
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            value = get(usage, key)
            if value is not None:
                self.usage[key] = value

    def finish(self, error: Optional[BaseException]) -> None:
        content = []
        for index in sorted(self.blocks):
            slot = self.blocks[index]
            kind = slot.get("type")
            if kind == "tool_use":
                raw = slot.get("json") or ""
                try:
                    tool_input = json.loads(raw) if raw else (slot.get("input") or {})
                except ValueError:
                    tool_input = raw
                content.append({"type": "tool_use", "id": slot.get("id"), "name": slot.get("name"),
                                "input": tool_input})
            elif kind == "thinking":
                content.append({"type": "thinking", "thinking": slot.get("thinking")})
            elif kind == "text":
                content.append({"type": "text", "text": slot.get("text")})
            else:
                content.append({"type": kind})
        _finish(self.gen, {"id": self.id, "model": self.model, "content": content,
                           "stop_reason": self.stop_reason, "usage": self.usage}, error)


# ---------------------------------------------------------------------------- wrappers

def _wrap_create(original: Any, is_async: bool, owner: type, method: str) -> Any:
    if is_async:
        @functools.wraps(original)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if wrapped_over(owner, method, async_wrapper):
                return await original(self, *args, **kwargs)
            gen = safe("anthropic.start", _start, self, kwargs)
            if gen is None or not gen.active:
                return await original(self, *args, **kwargs)
            gen.attach()
            try:
                result = await original(self, *args, **kwargs)
            except BaseException as error:
                gen.detach()
                safe("anthropic.fail", gen.fail, error)
                raise
            gen.detach()
            try:
                if get(kwargs, "stream", False) and not hasattr(result, "content"):
                    if hasattr(result, "http_response"):
                        gen.finish()
                        return result
                    state = _StreamState(gen)
                    return AsyncStream(result, gen, state.observe, state.finish)
                _finish(gen, result)
            except Exception as error:  # noqa: BLE001
                internal_error("anthropic.finish", error)
                safe("anthropic.end", gen.finish)
            return result
        setattr(async_wrapper, MARK, True)
        return async_wrapper

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if wrapped_over(owner, method, wrapper):
            return original(self, *args, **kwargs)
        gen = safe("anthropic.start", _start, self, kwargs)
        if gen is None or not gen.active:
            return original(self, *args, **kwargs)
        gen.attach()
        try:
            result = original(self, *args, **kwargs)
        except BaseException as error:
            gen.detach()
            safe("anthropic.fail", gen.fail, error)
            raise
        gen.detach()
        try:
            if get(kwargs, "stream", False) and not hasattr(result, "content"):
                if hasattr(result, "http_response"):
                    gen.finish()
                    return result
                state = _StreamState(gen)
                return SyncStream(result, gen, state.observe, state.finish)
            _finish(gen, result)
        except Exception as error:  # noqa: BLE001
            internal_error("anthropic.finish", error)
            safe("anthropic.end", gen.finish)
        return result
    setattr(wrapper, MARK, True)
    return wrapper


class _StreamManager:
    """Wraps MessageStreamManager: the request (and our span) starts on enter."""

    def __init__(self, manager: Any, resource: Any, kwargs: dict) -> None:
        self._manager = manager
        self._resource = resource
        self._kwargs = kwargs
        self._proxy: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    def _hook(self, message_stream: Any, gen: Generation, is_async: bool) -> None:
        state = _StreamState(gen)
        raw = message_stream._raw_stream
        proxy = (AsyncStream if is_async else SyncStream)(raw, gen, state.observe, state.finish)
        message_stream._raw_stream = proxy
        self._proxy = proxy

    def __enter__(self) -> Any:
        gen = safe("anthropic.start", _start, self._resource, self._kwargs, True)
        if gen is None or not gen.active:
            return self._manager.__enter__()
        gen.attach()
        try:
            stream = self._manager.__enter__()
        except BaseException as error:
            gen.detach()
            safe("anthropic.fail", gen.fail, error)
            raise
        gen.detach()
        safe("anthropic.hook", self._hook, stream, gen, False)
        return stream

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self._manager.__exit__(*exc)
        finally:
            if self._proxy is not None:
                safe("anthropic.stream_end", self._proxy._end, exc[1] if isinstance(exc[1], Exception) else None)

    async def __aenter__(self) -> Any:
        gen = safe("anthropic.start", _start, self._resource, self._kwargs, True)
        if gen is None or not gen.active:
            return await self._manager.__aenter__()
        gen.attach()
        try:
            stream = await self._manager.__aenter__()
        except BaseException as error:
            gen.detach()
            safe("anthropic.fail", gen.fail, error)
            raise
        gen.detach()
        safe("anthropic.hook", self._hook, stream, gen, True)
        return stream

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            return await self._manager.__aexit__(*exc)
        finally:
            if self._proxy is not None:
                safe("anthropic.stream_end", self._proxy._end, exc[1] if isinstance(exc[1], Exception) else None)


def _wrap_stream(original: Any, owner: type) -> Any:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        manager = original(self, *args, **kwargs)
        if wrapped_over(owner, "stream", wrapper):
            return manager
        try:
            return _StreamManager(manager, self, kwargs)
        except Exception as error:  # noqa: BLE001
            internal_error("anthropic.stream", error)
            return manager
    setattr(wrapper, MARK, True)
    return wrapper


def instrument() -> bool:
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
    except ImportError:
        return False
    wrapped_any = False
    for cls, is_async in ((Messages, False), (AsyncMessages, True)):
        for method in ("create", "parse", "stream"):
            current = cls.__dict__.get(method)
            if current is None:
                continue
            if is_ours(current):
                wrapped_any = True
                continue
            if foreign_wrapper(current, "anthropic"):
                continue
            _originals[(cls, method)] = current
            if method == "stream":
                setattr(cls, method, _wrap_stream(current, cls))
            else:
                setattr(cls, method, _wrap_create(current, is_async, cls, method))
            wrapped_any = True
    return wrapped_any


def uninstrument() -> None:
    for (cls, method), original in list(_originals.items()):
        setattr(cls, method, original)
    _originals.clear()
