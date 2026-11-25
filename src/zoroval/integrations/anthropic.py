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
from ._common import (MARK, AsyncStream, Generation, SyncStream, client_host, foreign_wrapper, get, given,
                      hook_raw_response, is_ours, media_part, needs_parse_hook, provider_from_host, safe,
                      strip_inline_media, text_part, wrapped_over)

REQUEST_PARAMS = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "top_k": "gen_ai.request.top_k",
    "max_tokens": "gen_ai.request.max_tokens",
}

_originals: dict[tuple[type, str], Any] = {}


# Tool calls the model makes: the app's own tools, and server tools Anthropic runs (web search, web fetch,
# code execution) or an MCP server runs.
_CALL_BLOCKS = ("tool_use", "server_tool_use", "mcp_tool_use")


def _is_call_block(kind: Any) -> bool:
    return isinstance(kind, str) and (kind in _CALL_BLOCKS or kind.endswith("_tool_use"))


def _tool_result_text(inner: Any) -> Any:
    """tool_result content: a string, or blocks. Text blocks are joined into one string."""
    if isinstance(inner, str) or inner is None:
        return inner
    pieces = [get(b, "text") if get(b, "type") == "text" else f"[{get(b, 'type')} omitted]" for b in inner]
    return "\n".join(p for p in pieces if isinstance(p, str))


def _server_result(block: Any) -> dict:
    """web_search_tool_result, web_fetch_tool_result, code_execution_tool_result, mcp_tool_result, ...:
    the result of a server tool call, with the call's id. Base64 and encrypted blobs are left out."""
    inner = get(block, "content")
    response = _tool_result_text(inner) if get(block, "type") == "mcp_tool_result" else \
        strip_inline_media(_json.to_plain(inner))
    part = {"type": "tool_call_response", "id": get(block, "tool_use_id"), "response": response}
    error_type = get(inner, "type") if not isinstance(inner, (list, str)) else None
    if get(block, "is_error") or (isinstance(error_type, str) and error_type.endswith("_error")):
        part["is_error"] = True
    return part


def _blocks(content: Any) -> list[dict]:
    if content is None or not given(content):
        return []
    if isinstance(content, str):
        return [text_part(content)]
    parts = []
    for block in content:
        kind = get(block, "type")
        if kind == "text":
            parts.append(text_part(get(block, "text"), "citations", get(block, "citations")))
        elif _is_call_block(kind):
            parts.append({"type": "tool_call", "id": get(block, "id"), "name": get(block, "name"),
                          "arguments": _json.dumps(_json.to_plain(get(block, "input")))})
        elif kind == "tool_result":
            part = {"type": "tool_call_response", "id": get(block, "tool_use_id"),
                    "response": _tool_result_text(get(block, "content"))}
            if get(block, "is_error"):
                part["is_error"] = True
            parts.append(part)
        elif isinstance(kind, str) and kind.endswith("_tool_result"):
            parts.append(_server_result(block))
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
        if get(block, "type") == "tool_use":  # server and MCP tools run outside the app: no tool span to link
            calls.append({"id": get(block, "id"), "name": get(block, "name"),
                          "arguments": _json.dumps(_json.to_plain(get(block, "input")))})
    return calls


def _usage(usage: Any) -> dict[str, Any]:
    """input_tokens is the whole prompt (GenAI semconv): Anthropic's input_tokens counts only the part
    after the last cache breakpoint, so cache reads and writes are added. They are also kept apart."""
    if usage is None:
        return {}
    cache_read = get(usage, "cache_read_input_tokens")
    cache_write = get(usage, "cache_creation_input_tokens")
    total = get(usage, "input_tokens")
    if isinstance(total, int):
        total += sum(v for v in (cache_read, cache_write) if isinstance(v, int) and not isinstance(v, bool))
    out = {"input_tokens": total, "output_tokens": get(usage, "output_tokens")}
    if cache_read:
        out["cache_read.input_tokens"] = cache_read
    if cache_write:
        out["cache_creation.input_tokens"] = cache_write
    return out


def _provider_name(resource: Any) -> str:
    client = type(getattr(resource, "_client", None)).__name__
    if "Bedrock" in client:
        return "aws.bedrock"
    if "Vertex" in client:
        return "gcp.vertex_ai"
    return provider_from_host(client_host(resource), "anthropic")


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
    gen = Generation(provider=_provider_name(resource), model=get(kwargs, "model"), request_attributes=attrs)
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
                "input": get(block, "input"), "citations": list(get(block, "citations") or []), "block": block}
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
            elif dtype == "citations_delta" and get(delta, "citation") is not None:
                slot.setdefault("citations", []).append(get(delta, "citation"))
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
            if _is_call_block(kind):
                raw = slot.get("json") or ""
                try:
                    tool_input = json.loads(raw) if raw else (slot.get("input") or {})
                except ValueError:
                    tool_input = raw
                content.append({"type": kind, "id": slot.get("id"), "name": slot.get("name"),
                                "input": tool_input})
            elif kind == "thinking":
                content.append({"type": "thinking", "thinking": slot.get("thinking")})
            elif kind == "text":
                content.append({"type": "text", "text": slot.get("text"), "citations": slot.get("citations")})
            elif isinstance(kind, str) and kind.endswith("_tool_result") and slot.get("block") is not None:
                content.append(slot["block"])  # server tool results arrive whole in content_block_start
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
                if needs_parse_hook(result, get(kwargs, "stream", False)):
                    if not safe("anthropic.raw_response", hook_raw_response, result, gen, _StreamState,
                                True, _finish):
                        gen.finish()
                    return result
                if get(kwargs, "stream", False) and not hasattr(result, "content"):
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
            if needs_parse_hook(result, get(kwargs, "stream", False)):
                if not safe("anthropic.raw_response", hook_raw_response, result, gen, _StreamState,
                            False, _finish):
                    gen.finish()
                return result
            if get(kwargs, "stream", False) and not hasattr(result, "content"):
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
