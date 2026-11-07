"""Shared pieces of the provider wrappers: the generation span, stream proxies, patch helpers.

Rules every wrapper follows:
- Never raise into the app. Our code runs in try/except; the provider's own errors pass
  through unchanged.
- No duplicate spans. If another LLM instrumentation already has an open span for this call
  (Langfuse's OpenAI wrapper, the OTel contrib instrumentors, OpenInference, OpenLLMetry), we make
  none. The provider SDK's own child span (the Anthropic client's built-in span) is suppressed in
  the processor.
"""

from __future__ import annotations

import time
import types
from typing import Any, Callable, Optional, Sequence

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.trace import SpanKind

from .. import _context, _json, _state
from .._util import internal_error
from .._version import SCOPE_NAME

MARK = "__zoroval_wrapped__"
NOT_GIVEN_NAMES = {"NotGiven", "Omit", "_NotGiven", "_Omit"}


def given(value: Any) -> bool:
    """False for None and the provider SDKs' NOT_GIVEN / omit sentinels."""
    return value is not None and type(value).__name__ not in NOT_GIVEN_NAMES


def get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    return value if given(value) else default


def foreign_llm_span_active() -> bool:
    """True when the current span is an open LLM-call span from another instrumentation."""
    span = otel_trace.get_current_span()
    if not span.is_recording() or getattr(span, "end_time", None) is not None:
        return False
    scope = getattr(span, "instrumentation_scope", None)
    scope_name = getattr(scope, "name", "") or ""
    if scope_name == SCOPE_NAME:
        return False
    attrs = getattr(span, "attributes", None) or {}
    if attrs.get("langfuse.observation.type") == "generation":
        return True
    if attrs.get("openinference.span.kind") == "LLM":
        return True
    if attrs.get("llm.request.type") in ("chat", "completion"):
        return True
    if attrs.get("gen_ai.operation.name") in ("chat", "text_completion", "generate_content"):
        return True
    return False


def is_ours(func: Any) -> bool:
    """wrapt proxies forward attribute reads, so check the real type before the marker."""
    return type(func) is types.FunctionType and bool(getattr(func, MARK, False))


def foreign_wrapper(func: Any, package: str) -> bool:
    """Is `func` already wrapped by someone else? wrapt proxies fake __class__, so test the type."""
    if is_ours(func):
        return False
    if type(func) is not types.FunctionType:
        return True
    return not (getattr(func, "__module__", "") or "").startswith(package + ".")


def wrapped_over(owner: type, method: str, ours: Any) -> bool:
    """True when another instrumentation wrapped the method after us (install order: Zoroval first).
    That wrapper records this call, so ours must not record it again."""
    current = owner.__dict__.get(method)
    return current is not None and current is not ours and not is_ours(current)


class Generation:
    """One LLM call as a GenAI semconv CLIENT span."""

    def __init__(self, *, provider: str, model: Any, operation: str = "chat",
                 request_attributes: Optional[dict] = None) -> None:
        self.span: Any = None
        self.token: Any = None
        self.started = time.monotonic()
        self.first_chunk: Optional[float] = None
        self.done = False
        state = _state.get()
        if state is None or foreign_llm_span_active():
            return
        model_name = str(model) if given(model) else ""
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model_name,
            _context.OBS_TYPE: "generation",
        }
        attrs.update(_context.conversation_attributes())
        parent = otel_trace.get_current_span()
        if not parent.get_span_context().is_valid:
            attrs.update(_context.root_attributes(None, None))
        for key, value in (request_attributes or {}).items():
            if value is not None:
                attrs[key] = value
        self.span = state.tracer.start_span(f"{operation} {model_name}".strip(), kind=SpanKind.CLIENT,
                                            attributes=attrs)
        try:
            state.processor.register_generation(self.span)
        except Exception as error:  # noqa: BLE001
            internal_error("register_generation", error)

    @property
    def active(self) -> bool:
        return self.span is not None

    # context of the provider call, so its own spans and HTTP spans become children
    def attach(self) -> None:
        if self.span is not None:
            self.token = otel_context.attach(otel_trace.set_span_in_context(self.span))

    def detach(self) -> None:
        if self.token is not None:
            try:
                otel_context.detach(self.token)
            except Exception:  # noqa: BLE001
                pass
            self.token = None

    def content(self, key: str, value: Any) -> None:
        if self.span is not None and value is not None:
            _context.set_content(self.span, key, value)

    def set(self, key: str, value: Any) -> None:
        if self.span is not None and value is not None:
            self.span.set_attribute(key, value)

    def mark_first_chunk(self) -> None:
        if self.first_chunk is None:
            self.first_chunk = time.monotonic()

    def finish(self, *, response_id: Any = None, model: Any = None, finish_reasons: Sequence[Any] = (),
               output_messages: Optional[list] = None, tool_calls: Optional[list] = None,
               usage: Optional[dict] = None, error: Optional[BaseException] = None) -> None:
        if self.span is None or self.done:
            return
        self.done = True
        span = self.span
        try:
            if response_id:
                span.set_attribute("gen_ai.response.id", str(response_id))
            if model:
                span.set_attribute("gen_ai.response.model", str(model))
            reasons = [str(r) for r in finish_reasons if r]
            if reasons:
                span.set_attribute("gen_ai.response.finish_reasons", reasons)
            for key, value in (usage or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    span.set_attribute(f"gen_ai.usage.{key}", int(value))
            if self.first_chunk is not None:
                span.set_attribute("gen_ai.response.time_to_first_chunk", self.first_chunk - self.started)
            if output_messages:
                _context.set_content(span, "gen_ai.output.messages", output_messages)
            if tool_calls:
                calls = [{"id": c.get("id"), "name": c.get("name"),
                          **({"arguments": c.get("arguments")} if _context.content_allowed(span)
                             and c.get("arguments") is not None else {})}
                         for c in tool_calls]
                state = _state.get()
                limit = state.cfg.max_content_bytes if state else 1 << 20
                text, cut = _json.fit(_json.dumps(calls), limit)
                span.set_attribute("zoroval.tool_calls", text)
                if cut:
                    _context.add_truncated(span, cut)
                _context.register_tool_calls(span, tool_calls)
            if error is not None:
                _context.record_error(span, error)
        except Exception as inner:  # noqa: BLE001
            internal_error("generation.finish", inner)
        finally:
            try:
                span.end()
            except Exception as inner:  # noqa: BLE001
                internal_error("generation.end", inner)

    def fail(self, error: BaseException) -> None:
        self.finish(error=error)


def safe(where: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return func(*args, **kwargs)
    except Exception as error:  # noqa: BLE001
        internal_error(where, error)
        return None


class _ProxyBase:
    """Forwards every attribute to the wrapped stream; reports the wrapped class for isinstance()."""

    _wrapped: Any

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    @property  # type: ignore[misc]
    def __class__(self) -> type:  # noqa: D401 - isinstance(stream, openai.Stream) keeps working
        return type(object.__getattribute__(self, "_wrapped"))


class SyncStream(_ProxyBase):
    """Passes events through to the app and gives each to `observe`; ends the span at the end."""

    def __init__(self, wrapped: Any, gen: Generation, observe: Callable[[Any], None],
                 finish: Callable[[Optional[BaseException]], None]) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_gen", gen)
        object.__setattr__(self, "_observe", observe)
        object.__setattr__(self, "_finish_cb", finish)
        object.__setattr__(self, "_iter", None)
        object.__setattr__(self, "_ended", False)

    def _end(self, error: Optional[BaseException] = None) -> None:
        if object.__getattribute__(self, "_ended"):
            return
        object.__setattr__(self, "_ended", True)
        safe("stream.finish", object.__getattribute__(self, "_finish_cb"), error)

    def __iter__(self) -> "SyncStream":
        return self

    def __next__(self) -> Any:
        it = object.__getattribute__(self, "_iter")
        if it is None:
            it = iter(object.__getattribute__(self, "_wrapped"))
            object.__setattr__(self, "_iter", it)
        try:
            event = next(it)
        except StopIteration:
            self._end(None)
            raise
        except Exception as error:
            self._end(error)
            raise
        object.__getattribute__(self, "_gen").mark_first_chunk()
        safe("stream.observe", object.__getattribute__(self, "_observe"), event)
        return event

    def __enter__(self) -> "SyncStream":
        wrapped = object.__getattribute__(self, "_wrapped")
        if hasattr(wrapped, "__enter__"):
            wrapped.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        wrapped = object.__getattribute__(self, "_wrapped")
        try:
            if hasattr(wrapped, "__exit__"):
                return wrapped.__exit__(*exc)
            return False
        finally:
            self._end(exc[1] if exc and isinstance(exc[1], Exception) else None)

    def close(self) -> None:
        try:
            object.__getattribute__(self, "_wrapped").close()
        finally:
            self._end(None)

    def __del__(self) -> None:
        try:
            self._end(None)
        except Exception:  # noqa: BLE001
            pass


class AsyncStream(_ProxyBase):
    def __init__(self, wrapped: Any, gen: Generation, observe: Callable[[Any], None],
                 finish: Callable[[Optional[BaseException]], None]) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_gen", gen)
        object.__setattr__(self, "_observe", observe)
        object.__setattr__(self, "_finish_cb", finish)
        object.__setattr__(self, "_iter", None)
        object.__setattr__(self, "_ended", False)

    def _end(self, error: Optional[BaseException] = None) -> None:
        if object.__getattribute__(self, "_ended"):
            return
        object.__setattr__(self, "_ended", True)
        safe("stream.finish", object.__getattribute__(self, "_finish_cb"), error)

    def __aiter__(self) -> "AsyncStream":
        return self

    async def __anext__(self) -> Any:
        it = object.__getattribute__(self, "_iter")
        if it is None:
            it = object.__getattribute__(self, "_wrapped").__aiter__()
            object.__setattr__(self, "_iter", it)
        try:
            event = await it.__anext__()
        except StopAsyncIteration:
            self._end(None)
            raise
        except Exception as error:
            self._end(error)
            raise
        object.__getattribute__(self, "_gen").mark_first_chunk()
        safe("stream.observe", object.__getattribute__(self, "_observe"), event)
        return event

    async def __aenter__(self) -> "AsyncStream":
        wrapped = object.__getattribute__(self, "_wrapped")
        if hasattr(wrapped, "__aenter__"):
            await wrapped.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        wrapped = object.__getattribute__(self, "_wrapped")
        try:
            if hasattr(wrapped, "__aexit__"):
                return await wrapped.__aexit__(*exc)
            return False
        finally:
            self._end(exc[1] if exc and isinstance(exc[1], Exception) else None)

    async def close(self) -> None:
        try:
            await object.__getattribute__(self, "_wrapped").close()
        finally:
            self._end(None)

    def __del__(self) -> None:
        try:
            self._end(None)
        except Exception:  # noqa: BLE001
            pass


def text_part(text: Any) -> dict:
    return {"type": "text", "content": text}


def media_part(kind: str, detail: Any = None) -> dict:
    """Media is not captured in v1 (design section 3): a placeholder keeps the message shape."""
    part = {"type": kind, "content": f"[{kind} omitted]"}
    url = detail if isinstance(detail, str) else None
    if url and not url.startswith("data:"):
        part["uri"] = url
    return part
