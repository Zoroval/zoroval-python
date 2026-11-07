"""Public context API: conversation(), turn(), trace(), step(), tool(), and trace attributes.

- conversation(id) groups every turn inside it into one Zoroval trace (design 6.5). It makes no
  span itself: a conversation outlives one request.
- turn() is the root span of one user turn. trace() is the root of one run (agents, batch jobs);
  inside a conversation it acts as a turn.
- step() and tool() are child spans, usable as context managers and as decorators (sync and async).
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Optional, Sequence

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from . import _json, _state
from ._util import COUNTERS, internal_error, log_once

MAX_CONVERSATION_ID = 256
MAX_TAGS = 64
MAX_TAG_CHARS = 128

# Span attribute names (contracts section 6).
CONVERSATION_ID = "zoroval.conversation.id"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
OBS_TYPE = "zoroval.observation.type"
USER_ID = "zoroval.user.id"
TAGS = "zoroval.tags"
METADATA = "zoroval.metadata"
INPUT = "zoroval.input"
OUTPUT = "zoroval.output"
GROUPING = "zoroval.grouping"
TEST = "zoroval.test"
REDACTED = "zoroval.redacted"
TRUNCATED = "zoroval.truncated_bytes"


class _ConversationScope:
    __slots__ = ("id", "user_id", "tags", "metadata")

    def __init__(self, conversation_id: str, user_id: Optional[str], tags: Optional[Sequence[str]],
                 metadata: Optional[dict]) -> None:
        self.id = conversation_id
        self.user_id = user_id
        self.tags = list(tags or [])
        self.metadata = dict(metadata or {})


_conversation: contextvars.ContextVar[Optional[_ConversationScope]] = \
    contextvars.ContextVar("zoroval_conversation", default=None)
_root: contextvars.ContextVar[Optional["Observation"]] = contextvars.ContextVar("zoroval_root", default=None)
_test: contextvars.ContextVar[bool] = contextvars.ContextVar("zoroval_test", default=False)


def current_conversation_id() -> Optional[str]:
    scope = _conversation.get()
    return scope.id if scope is not None else None


def current_conversation() -> Optional[_ConversationScope]:
    return _conversation.get()


def current_root() -> Optional["Observation"]:
    return _root.get()


def normalize_conversation_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    if len(text) > MAX_CONVERSATION_ID:
        import hashlib
        # Cutting would merge different long ids; a hash keeps them apart and stays stable.
        return "sha256:" + hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
    return text


def clean_tags(tags: Optional[Iterable[Any]]) -> list[str]:
    out: list[str] = []
    for tag in tags or ():
        text = str(tag)[:MAX_TAG_CHARS]
        if text and text not in out:
            out.append(text)
        if len(out) >= MAX_TAGS:
            break
    return out


# ---------------------------------------------------------------------------- capture helpers

def tracer() -> Any:
    state = _state.get()
    if state is None:
        return otel_trace.NoOpTracer()
    return state.tracer


def capture(value: Any, *, limit: Optional[int] = None) -> _json.Captured:
    state = _state.get()
    if state is None:
        return _json.Captured(None, False, 0)
    cfg = state.cfg
    return _json.capture(value, capture_content=cfg.capture_content, redact=cfg.redact,
                         limit=limit or cfg.max_content_bytes)


def content_allowed(span: Any) -> bool:
    """False when content must not be serialized for this span (capture off, or sampled out
    while Zoroval owns the provider so no other exporter needs the content)."""
    state = _state.get()
    if state is None or not state.cfg.capture_content:
        return False
    if not span.is_recording():
        return False
    if state.owns_provider and state.cfg.sample_rate < 1.0:
        from ._processor import sample_key, sample_keep
        ctx = span.get_span_context()
        return sample_keep(sample_key(current_conversation_id(), ctx.trace_id), state.cfg.sample_rate)
    return True


def set_content(span: Any, key: str, value: Any, *, limit: Optional[int] = None) -> None:
    """Capture `value` into span attribute `key`, with redact and size cap flags."""
    if value is None or not content_allowed(span):
        return
    captured = capture(value, limit=limit)
    if captured.text is not None:
        span.set_attribute(key, captured.text)
    if captured.redacted:
        span.set_attribute(REDACTED, True)
    if captured.truncated_bytes:
        add_truncated(span, captured.truncated_bytes)


def add_truncated(span: Any, count: int) -> None:
    attrs = getattr(span, "attributes", None) or {}
    previous = attrs.get(TRUNCATED, 0) if hasattr(attrs, "get") else 0
    span.set_attribute(TRUNCATED, int(previous or 0) + int(count))


def set_metadata_attr(span: Any, metadata: Optional[dict]) -> None:
    if not metadata:
        return
    state = _state.get()
    limit = state.cfg.max_metadata_bytes if state else 16 * 1024
    text, cut = _json.fit(_json.dumps(metadata), limit)
    span.set_attribute(METADATA, text)
    if cut:
        add_truncated(span, cut)


def conversation_attributes() -> dict[str, Any]:
    """Explicit ids from conversation(); else an id inherited from the current (parent) span."""
    scope = _conversation.get()
    if scope is not None:
        return {CONVERSATION_ID: scope.id, GEN_AI_CONVERSATION_ID: scope.id}
    from ._processor import inherited_conversation
    parent = otel_trace.get_current_span()
    inherited = inherited_conversation(getattr(parent, "attributes", None))
    if inherited:
        return {GEN_AI_CONVERSATION_ID: inherited}
    return {}


def root_attributes(user_id: Optional[str], tags: Optional[Sequence[str]]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    scope = _conversation.get()
    user = user_id or (scope.user_id if scope else None)
    if user:
        attrs[USER_ID] = str(user)
        attrs["user.id"] = str(user)
    all_tags = clean_tags(list(scope.tags if scope else []) + list(tags or []))
    if all_tags:
        attrs[TAGS] = all_tags
    if _test.get():
        attrs[TEST] = True
    return attrs


def record_error(span: Any, error: BaseException) -> None:
    """Span status ERROR plus a standard `exception` event with a capped stack."""
    try:
        state = _state.get()
        stack_limit = state.cfg.max_stack_bytes if state else 16 * 1024
        message = f"{type(error).__name__}: {error}"[:32768]
        span.set_status(Status(StatusCode.ERROR, message))
        stack = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        if len(stack) > stack_limit:
            stack = stack[-stack_limit:]
        span.add_event("exception", {
            "exception.type": type(error).__qualname__,
            "exception.message": str(error)[:32768],
            "exception.stacktrace": stack,
            "exception.escaped": True,
        })
        span.set_attribute("error.type", type(error).__qualname__)
    except Exception as inner:  # noqa: BLE001
        internal_error("record_error", inner)


# ---------------------------------------------------------------------------- pending tool calls

class _PendingCalls:
    """Tool calls the model asked for and no @tool run has claimed yet.

    Keyed by trace id and conversation id, so a match never crosses users or requests.
    """

    TTL_S = 600.0
    LIMIT = 1024

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[dict[str, Any]] = []

    def add(self, calls: Sequence[dict[str, Any]], trace_id: int, conversation: Optional[str]) -> None:
        now = time.monotonic()
        with self._lock:
            for call in calls:
                if call.get("id") and call.get("name"):
                    self._items.append({"id": call["id"], "name": call["name"], "arguments": call.get("arguments"),
                                        "trace_id": trace_id, "conversation": conversation, "at": now})
            self._prune(now)

    def _prune(self, now: float) -> None:
        self._items = [i for i in self._items if now - i["at"] < self.TTL_S][-self.LIMIT:]

    def claim(self, name: str, arguments: Any, trace_id: int, conversation: Optional[str]) -> Optional[str]:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            candidates = [i for i in self._items if i["name"] == name and (
                i["trace_id"] == trace_id or (conversation is not None and i["conversation"] == conversation))]
            if len(candidates) > 1 and arguments is not None:
                candidates = [i for i in candidates if _same_arguments(i["arguments"], arguments)]
            if len(candidates) != 1:
                return None  # none, or ambiguous: never pair by position
            self._items.remove(candidates[0])
            return candidates[0]["id"]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def _same_arguments(raw: Any, arguments: Any) -> bool:
    import json
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return _json.to_plain(parsed) == _json.to_plain(arguments)
    except Exception:  # noqa: BLE001
        return False


PENDING = _PendingCalls()


def register_tool_calls(span: Any, calls: Sequence[dict[str, Any]]) -> None:
    if not calls:
        return
    try:
        PENDING.add(calls, span.get_span_context().trace_id, current_conversation_id())
    except Exception as error:  # noqa: BLE001
        internal_error("register_tool_calls", error)


def note_previous_response() -> None:
    root = _root.get()
    if root is not None:
        root._response_chain = True


# ---------------------------------------------------------------------------- Observation

class Observation:
    """Handle to an open span. Every method is safe: it never raises into the app."""

    def __init__(self, span: Any, kind: str, *, is_root: bool = False) -> None:
        self.span = span
        self.kind = kind
        self.is_root = is_root
        self._response_chain = False
        self._inherited = False
        self._metadata: dict[str, Any] = {}

    @property
    def trace_id(self) -> str:
        return format(self.span.get_span_context().trace_id, "032x")

    @property
    def span_id(self) -> str:
        return format(self.span.get_span_context().span_id, "016x")

    def set_input(self, value: Any) -> None:
        try:
            key = "gen_ai.tool.call.arguments" if self.kind == "tool" else INPUT
            set_content(self.span, key, value)
        except Exception as error:  # noqa: BLE001
            internal_error("set_input", error)

    def set_output(self, value: Any) -> None:
        try:
            key = "gen_ai.tool.call.result" if self.kind == "tool" else OUTPUT
            set_content(self.span, key, value)
        except Exception as error:  # noqa: BLE001
            internal_error("set_output", error)

    def set_metadata(self, metadata: Optional[dict] = None, **values: Any) -> None:
        try:
            self._metadata.update(metadata or {})
            self._metadata.update(values)
            set_metadata_attr(self.span, self._metadata)
        except Exception as error:  # noqa: BLE001
            internal_error("set_metadata", error)

    def set_user(self, user_id: Any) -> None:
        try:
            if user_id is not None:
                self.span.set_attribute(USER_ID, str(user_id))
                self.span.set_attribute("user.id", str(user_id))
        except Exception as error:  # noqa: BLE001
            internal_error("set_user", error)

    def set_tags(self, tags: Iterable[Any]) -> None:
        try:
            existing = list((getattr(self.span, "attributes", None) or {}).get(TAGS, ()) or ())
            merged = clean_tags(existing + list(tags))
            if merged:
                self.span.set_attribute(TAGS, merged)
        except Exception as error:  # noqa: BLE001
            internal_error("set_tags", error)

    def set_attribute(self, key: str, value: Any) -> None:
        try:
            self.span.set_attribute(key, value)
        except Exception as error:  # noqa: BLE001
            internal_error("set_attribute", error)

    def set_error(self, error: Any) -> None:
        try:
            if isinstance(error, BaseException):
                record_error(self.span, error)
            else:
                self.span.set_status(Status(StatusCode.ERROR, str(error)[:32768]))
        except Exception as inner:  # noqa: BLE001
            internal_error("set_error", inner)

    def update(self, *, output: Any = None, metadata: Optional[dict] = None, user_id: Any = None,
               tags: Optional[Iterable[Any]] = None) -> None:
        if output is not None:
            self.set_output(output)
        if metadata:
            self.set_metadata(metadata)
        if user_id is not None:
            self.set_user(user_id)
        if tags:
            self.set_tags(tags)


class _NoopObservation(Observation):
    def __init__(self) -> None:
        super().__init__(otel_trace.INVALID_SPAN, "noop")


# ---------------------------------------------------------------------------- span scopes

class _SpanScope:
    """A span as a context manager (sync or async) or as a decorator."""

    def __init__(self, kind: str, name: Optional[str], *, input: Any = None, metadata: Optional[dict] = None,
                 user_id: Any = None, tags: Optional[Sequence[str]] = None, call_id: Optional[str] = None,
                 arguments: Any = None, capture_input: bool = True, capture_output: bool = True,
                 obs_type: Optional[str] = None) -> None:
        self.kind = kind
        self.name = name
        self.input = input
        self.metadata = metadata
        self.user_id = user_id
        self.tags = tags
        self.call_id = call_id
        self.arguments = arguments
        self.capture_input = capture_input
        self.capture_output = capture_output
        self.obs_type = obs_type
        self._token: Any = None
        self._root_token: Any = None
        self._obs: Optional[Observation] = None
        self._span: Any = None

    # ---------------------------------------------------------------- span start and end

    def _start(self) -> Observation:
        try:
            return self._start_inner()
        except Exception as error:  # noqa: BLE001
            internal_error(f"{self.kind}.start", error)
            self._obs = _NoopObservation()
            return self._obs

    def _start_inner(self) -> Observation:
        state = _state.get()
        if state is None:
            self._obs = _NoopObservation()
            return self._obs
        scope = _conversation.get()
        kind = self.kind
        if kind == "trace" and scope is not None:
            kind = "turn"
        name = self.name or kind
        is_root = kind in ("turn", "trace")
        attrs: dict[str, Any] = {OBS_TYPE: self.obs_type or kind}
        attrs.update(conversation_attributes())
        if is_root:
            attrs.update(root_attributes(self.user_id, self.tags))
        if kind == "tool":
            attrs["gen_ai.operation.name"] = "execute_tool"
            attrs["gen_ai.tool.name"] = name
            attrs["gen_ai.tool.type"] = "function"
            name = f"execute_tool {name}"
        span = state.tracer.start_span(name, kind=SpanKind.INTERNAL, attributes=attrs)
        self._span = span
        self._token = otel_context.attach(otel_trace.set_span_in_context(span))
        obs = Observation(span, kind, is_root=is_root)
        self._obs = obs
        if is_root:
            self._root_token = _root.set(obs)
            obs._inherited = scope is None and GEN_AI_CONVERSATION_ID in attrs
        if self.metadata:
            obs.set_metadata(self.metadata)
        if kind == "tool":
            call_id = self.call_id
            if call_id is None:
                call_id = PENDING.claim(self.name or "", self.arguments,
                                        span.get_span_context().trace_id, current_conversation_id())
            if call_id:
                span.set_attribute("gen_ai.tool.call.id", str(call_id))
            if self.capture_input and self.arguments is not None:
                set_content(span, "gen_ai.tool.call.arguments", self.arguments)
        elif self.capture_input and self.input is not None:
            set_content(span, INPUT, self.input)
        return obs

    def _end(self, error: Optional[BaseException]) -> None:
        try:
            span = self._span
            obs = self._obs
            if span is None or obs is None:
                return
            if error is not None:
                record_error(span, error)
            if obs.is_root:
                span.set_attribute(GROUPING, self._grouping(obs))
            span.end()
        except Exception as inner:  # noqa: BLE001
            internal_error(f"{self.kind}.end", inner)
        finally:
            if self._token is not None:
                try:
                    otel_context.detach(self._token)
                except Exception:  # noqa: BLE001
                    pass
                self._token = None
            if self._root_token is not None:
                try:
                    _root.reset(self._root_token)
                except Exception:  # noqa: BLE001
                    pass
                self._root_token = None

    def _grouping(self, obs: Observation) -> str:
        if _conversation.get() is not None:
            return "explicit"
        if obs._inherited:
            return "inherited"
        if obs._response_chain:
            return "response_chain"
        if obs.kind == "trace":
            return "not_chat"
        COUNTERS.add("ungrouped_turns")
        log_once("ungrouped-turn", "zoroval: turn() ran outside conversation(); each such turn becomes its "
                 "own trace (ungrouped). Wrap the chat handler in zoroval.conversation(chat_id).")
        return "ungrouped"

    # ---------------------------------------------------------------- context manager protocol

    def __enter__(self) -> Observation:
        return self._start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._end(exc if isinstance(exc, BaseException) and not isinstance(exc, GeneratorExit) else None)
        return False

    async def __aenter__(self) -> Observation:
        return self._start()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self.__exit__(exc_type, exc, tb)

    # ---------------------------------------------------------------- decorator

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        return _decorate(func, self)

    def _fresh(self, name: str, arguments: Any) -> "_SpanScope":
        scope = _SpanScope(self.kind, self.name or name, input=arguments if self.kind != "tool" else None,
                           metadata=self.metadata, user_id=self.user_id, tags=self.tags, call_id=self.call_id,
                           arguments=arguments if self.kind == "tool" else None,
                           capture_input=self.capture_input, capture_output=self.capture_output,
                           obs_type=self.obs_type)
        return scope


def _bound_arguments(func: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        out = {}
        for key, value in bound.arguments.items():
            if key in ("self", "cls"):
                continue
            param = sig.parameters.get(key)
            if param is not None and param.kind is inspect.Parameter.VAR_KEYWORD:
                out.update(value)
            else:
                out[key] = value
        return out
    except Exception:  # noqa: BLE001
        return {"args": list(args), "kwargs": kwargs}


def _decorate(func: Callable[..., Any], template: _SpanScope) -> Callable[..., Any]:
    name = getattr(func, "__name__", "function")

    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            scope = template._fresh(name, _bound_arguments(func, args, kwargs) if template.capture_input else None)
            obs = scope._start()
            try:
                result = await func(*args, **kwargs)
            except BaseException as error:
                scope._end(error if isinstance(error, Exception) else None)
                raise
            if template.capture_output:
                obs.set_output(result)
            scope._end(None)
            return result
        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        scope = template._fresh(name, _bound_arguments(func, args, kwargs) if template.capture_input else None)
        obs = scope._start()
        try:
            result = func(*args, **kwargs)
        except BaseException as error:
            scope._end(error if isinstance(error, Exception) else None)
            raise
        if template.capture_output:
            obs.set_output(result)
        scope._end(None)
        return result
    return wrapper


# ---------------------------------------------------------------------------- public functions

class conversation:  # noqa: N801 - reads as a function at the call site
    """Group every turn inside this block into one conversation (one Zoroval trace).

        with zoroval.conversation(chat_id, user_id=user.id):
            reply = client.chat.completions.create(...)

    Works with `with` and `async with`. Makes no span itself.
    """

    def __init__(self, conversation_id: Any, *, user_id: Any = None, tags: Optional[Sequence[str]] = None,
                 metadata: Optional[dict] = None) -> None:
        self._id = normalize_conversation_id(conversation_id)
        if self._id is None:
            log_once("conversation-empty", "zoroval: conversation() got an empty id; turns stay ungrouped.")
        self._scope = _ConversationScope(self._id, None if user_id is None else str(user_id), tags, metadata) \
            if self._id else None
        self._token: Any = None

    def __enter__(self) -> "conversation":
        if self._scope is not None:
            self._token = _conversation.set(self._scope)
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._token is not None:
            try:
                _conversation.reset(self._token)
            except ValueError:
                _conversation.set(None)
            self._token = None
        return False

    async def __aenter__(self) -> "conversation":
        return self.__enter__()

    async def __aexit__(self, *exc: Any) -> bool:
        return self.__exit__(*exc)

    @property
    def id(self) -> Optional[str]:
        return self._id


def turn(name: Optional[str] = None, *, input: Any = None, user_id: Any = None,
         tags: Optional[Sequence[str]] = None, metadata: Optional[dict] = None) -> _SpanScope:
    """The root span of one user turn. Use inside conversation()."""
    if callable(name):
        return _SpanScope("turn", None)(name)  # type: ignore[return-value]
    return _SpanScope("turn", name, input=input, user_id=user_id, tags=tags, metadata=metadata)


def trace(name: Optional[str] = None, *, input: Any = None, user_id: Any = None,
          tags: Optional[Sequence[str]] = None, metadata: Optional[dict] = None) -> _SpanScope:
    """The root span of one run (agent or batch job). Inside conversation() it acts as a turn."""
    if callable(name):
        return _SpanScope("trace", None)(name)  # type: ignore[return-value]
    return _SpanScope("trace", name, input=input, user_id=user_id, tags=tags, metadata=metadata)


def step(name: Any = None, *, input: Any = None, metadata: Optional[dict] = None, type: str = "step",
         capture_input: bool = True, capture_output: bool = True) -> Any:
    """A custom step. `@zoroval.step`, `@zoroval.step(name=...)` or `with zoroval.step("plan"):`.

    type: "step" (default), "chain" or "agent".
    """
    if callable(name):
        return _decorate(name, _SpanScope("step", None))
    if type not in ("step", "chain", "agent"):
        type = "step"
    return _SpanScope("step", name, input=input, metadata=metadata, capture_input=capture_input,
                      capture_output=capture_output, obs_type=type)


def tool(name: Any = None, *, call_id: Optional[str] = None, arguments: Any = None,
         metadata: Optional[dict] = None, capture_input: bool = True, capture_output: bool = True) -> Any:
    """A tool run, linked to the model's tool-call id.

    The id is `call_id` when given; otherwise the one pending call of the same name (and the same
    arguments, if several) in this trace or conversation. Ambiguous calls stay unlinked.
    """
    if callable(name):
        return _decorate(name, _SpanScope("tool", None))
    return _SpanScope("tool", name, call_id=call_id, arguments=arguments, metadata=metadata,
                      capture_input=capture_input, capture_output=capture_output)


def _target() -> Any:
    root = _root.get()
    if root is not None:
        return root
    return _conversation.get()


def set_user(user_id: Any) -> None:
    """Set the user id on the current turn or trace, or on the current conversation."""
    target = _target()
    if isinstance(target, Observation):
        target.set_user(user_id)
    elif isinstance(target, _ConversationScope):
        target.user_id = None if user_id is None else str(user_id)


def set_tags(*tags: Any) -> None:
    flat: list[Any] = []
    for tag in tags:
        flat.extend(tag if isinstance(tag, (list, tuple, set)) else [tag])
    target = _target()
    if isinstance(target, Observation):
        target.set_tags(flat)
    elif isinstance(target, _ConversationScope):
        target.tags = clean_tags(target.tags + flat)


def set_metadata(metadata: Optional[dict] = None, **values: Any) -> None:
    target = _target()
    if isinstance(target, Observation):
        target.set_metadata(metadata, **values)
    elif isinstance(target, _ConversationScope):
        target.metadata.update(metadata or {})
        target.metadata.update(values)


def current_observation() -> Observation:
    """The innermost turn or trace handle, or a no-op handle."""
    return _root.get() or _NoopObservation()


class test_mode:  # noqa: N801
    """Marks root spans with zoroval.test=true. Used by `zoroval doctor` only."""

    def __enter__(self) -> None:
        self._token = _test.set(True)

    def __exit__(self, *exc: Any) -> bool:
        _test.reset(self._token)
        return False
