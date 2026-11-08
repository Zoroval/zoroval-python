"""The Zoroval span processor: GenAI filter, conversation stamping, head sampling, a queue capped
in bytes, a background export thread, bounded flush and shutdown, and fork safety.

Batch lifecycle: queued, sending, then sent; split in two (413, or a timeout on a slow uplink);
put back at the front of the queue (retries used up while the gateway is down, at most
`_MAX_REQUEUES` times, or the gateway asked for a pause); or dropped. The exporter itself is
`running`, `paused(until)` after a long Retry-After, or `backing_off(until)` after a failed batch:
while held it sends nothing and the queue keeps filling under its caps (oldest dropped first).

It replaces OpenTelemetry's BatchSpanProcessor for Zoroval's export. It sits on the app's
TracerProvider beside any other processors, so the app's own exporters are untouched.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import weakref
from collections import OrderedDict, deque
from typing import Any, Deque, Iterable, Optional

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from . import _context
from ._config import Config
from ._transport import WIRE_MARGIN, Transport, uuid7
from ._util import COUNTERS, internal_error, log_once
from ._version import SCOPE_NAME

GENAI_SCOPE_PREFIXES = (
    SCOPE_NAME,
    "langfuse",
    "openinference",
    "traceloop",
    "opentelemetry.instrumentation.openai",
    "opentelemetry.instrumentation.anthropic",
    "opentelemetry.instrumentation.langchain",
    "opentelemetry.instrumentation.llamaindex",
    "opentelemetry.instrumentation.bedrock",
    "opentelemetry.instrumentation.vertexai",
    "opentelemetry.instrumentation.google_genai",
    "opentelemetry.instrumentation.mistralai",
    "opentelemetry.instrumentation.groq",
    "opentelemetry.instrumentation.cohere",
    "opentelemetry.instrumentation.ollama",
    "opentelemetry.instrumentation.crewai",
    "opentelemetry.instrumentation.mcp",
    "com.anthropic.sdk",
    "openai.agents",
    "litellm",
)
GENAI_ATTR_PREFIXES = ("gen_ai.", "zoroval.", "langfuse.", "llm.", "openinference.", "traceloop.")

# Conversation ids other instrumentations set (design 6.5 "inherited").
INHERITED_CONVERSATION_KEYS = (
    "zoroval.conversation.id", "gen_ai.conversation.id", "session.id", "langfuse.session.id",
    "anthropic.session.thread.id",
)

_MAX_NOTES = 50_000
_BATCH_BUDGET_S = 30.0          # retries for one batch; grows to 1.5 request timeouts for big bodies
_MAX_REQUEUES = 10              # a failing batch goes back to the queue at most this many times
_MIN_BATCH_BYTES = 64 * 1024    # timeouts halve the batch size down to this
_GROW_AFTER_OK = 4              # sends in a row before a halved batch size doubles again
_LIVE: "weakref.WeakSet[ZorovalSpanProcessor]" = weakref.WeakSet()


def is_genai_span(span: ReadableSpan) -> bool:
    scope = span.instrumentation_scope.name if span.instrumentation_scope else ""
    if scope and scope.startswith(GENAI_SCOPE_PREFIXES):
        return True
    attrs = span.attributes or {}
    return any(key.startswith(GENAI_ATTR_PREFIXES) for key in attrs)


def sample_keep(key: str, rate: float) -> bool:
    """Deterministic head sampling: the same key always gets the same answer."""
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(key.encode("utf-8", "surrogatepass")).digest()
    return int.from_bytes(digest[:8], "big") < int(rate * (1 << 64))


def sample_key(conversation_id: Optional[str], trace_id: int) -> str:
    if conversation_id:
        return "c:" + conversation_id
    return "t:" + format(trace_id, "032x")


def inherited_conversation(attributes: Any) -> Optional[str]:
    if not attributes:
        return None
    for key in INHERITED_CONVERSATION_KEYS:
        value = attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def estimate_size(span: ReadableSpan) -> int:
    size = 160 + len(span.name or "")
    for key, value in (span.attributes or {}).items():
        size += len(key) + 8
        if isinstance(value, str):
            size += len(value)
        elif isinstance(value, (list, tuple)):
            size += sum(len(v) if isinstance(v, str) else 8 for v in value)
        else:
            size += 8
    for event in span.events or ():
        size += 32 + len(event.name)
        for key, value in (event.attributes or {}).items():
            size += len(key) + (len(value) if isinstance(value, str) else 8)
    return size


class _Bounded(OrderedDict):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def put(self, key: Any, value: Any) -> None:
        self[key] = value
        if len(self) > self.limit:
            self.popitem(last=False)


_Item = tuple  # (span, extras, estimated size)


class _Batch:
    """Spans sent together under one batch id. A requeued batch keeps its id: the gateway dedupes
    a resend, and a subset under the same id is safe because spans are deduped by id downstream."""

    __slots__ = ("items", "batch_id", "requeues", "size")

    def __init__(self, items: Deque[_Item], batch_id: str, requeues: int = 0) -> None:
        self.items = items
        self.batch_id = batch_id
        self.requeues = requeues
        self.size = sum(item[2] for item in items)

    def halves(self) -> list["_Batch"]:
        items = list(self.items)
        half = len(items) // 2
        return [_Batch(deque(items[:half]), uuid7(), self.requeues),
                _Batch(deque(items[half:]), uuid7(), self.requeues)]


class ZorovalSpanProcessor(SpanProcessor):
    def __init__(self, cfg: Config, transport: Transport, *, filter_spans: bool = True,
                 skip_content_when_sampled_out: bool = False) -> None:
        self.cfg = cfg
        self.transport = transport
        self.filter_spans = filter_spans
        self.skip_content_when_sampled_out = skip_content_when_sampled_out
        self._notes = _Bounded(_MAX_NOTES)          # span id -> conversation id (inherited or explicit)
        self._suppressed = _Bounded(_MAX_NOTES)     # span id -> True (native child of our generation)
        self._generations = _Bounded(_MAX_NOTES)    # our generation span ids
        self._init_runtime()
        _LIVE.add(self)

    # ------------------------------------------------------------------ lifecycle

    def _init_runtime(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._queue: Deque[tuple[ReadableSpan, Optional[dict], int]] = deque()
        self._retry: Deque[_Batch] = deque()       # batches put back; older than everything in _queue
        self._retry_spans = 0
        self._queue_bytes = 0                       # both deques
        self._batch_limit = self.cfg.max_batch_bytes
        self._ok_streak = 0
        self._enqueued = 0
        self._completed = 0
        self._flush_requested = False
        self._stopping = False
        self._stopped = False
        self._hard_deadline = float("inf")
        self._thread: Optional[threading.Thread] = None
        self._pid = os.getpid()

    def _after_fork_in_child(self) -> None:
        """Called in a forked child. The parent's queued spans stay with the parent."""
        self._init_runtime()
        self.transport.after_fork()

    def _ensure_worker(self) -> None:
        if self._pid != os.getpid():
            self._after_fork_in_child()
        if self._thread is None or not self._thread.is_alive():
            if self._stopping:
                return
            self._thread = threading.Thread(target=self._run, name="zoroval-export", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ SpanProcessor hooks

    def register_generation(self, span: Any) -> None:
        try:
            self._generations.put(span.get_span_context().span_id, True)
        except Exception as error:  # noqa: BLE001
            internal_error("register_generation", error)

    def on_start(self, span: Span, parent_context: Any = None) -> None:
        try:
            scope = span.instrumentation_scope.name if span.instrumentation_scope else ""
            span_id = span.context.span_id
            parent = span.parent
            if scope != SCOPE_NAME and parent is not None and parent.span_id in self._generations:
                # The provider SDK's own span for the same call (for example the Anthropic
                # client's built-in span). Our generation span already covers it.
                self._suppressed.put(span_id, True)
                return
            if scope == SCOPE_NAME:
                return
            conversation = _context.current_conversation_id()
            if conversation is None:
                parent_span = otel_trace.get_current_span(parent_context)
                conversation = inherited_conversation(getattr(parent_span, "attributes", None))
            if conversation is None:
                conversation = inherited_conversation(span.attributes)
            if conversation is not None:
                self._notes.put(span_id, conversation)
        except Exception as error:  # noqa: BLE001
            internal_error("on_start", error)

    def on_end(self, span: ReadableSpan) -> None:
        try:
            self._on_end(span)
        except Exception as error:  # noqa: BLE001
            internal_error("on_end", error)

    def _on_end(self, span: ReadableSpan) -> None:
        if self._stopped:
            return
        COUNTERS.add("spans_seen")
        span_id = span.context.span_id
        note = self._notes.pop(span_id, None)
        if self._suppressed.pop(span_id, None):
            COUNTERS.add("spans_filtered")
            return
        if self.filter_spans and not is_genai_span(span):
            COUNTERS.add("spans_filtered")
            return
        attrs = span.attributes or {}
        explicit = attrs.get("zoroval.conversation.id")
        conversation = explicit if isinstance(explicit, str) else (note or inherited_conversation(attrs))
        if not sample_keep(sample_key(conversation, span.context.trace_id), self.cfg.sample_rate):
            COUNTERS.add("spans_sampled_out")
            return
        extras = None
        if note and "zoroval.conversation.id" not in attrs and "gen_ai.conversation.id" not in attrs:
            extras = {"gen_ai.conversation.id": note}
        self._enqueue(span, extras, estimate_size(span))

    def _enqueue(self, span: ReadableSpan, extras: Optional[dict], size: int) -> None:
        if self._pid != os.getpid():
            self._after_fork_in_child()
        with self._cond:
            self._queue.append((span, extras, size))
            self._queue_bytes += size
            self._enqueued += 1
            dropped = self._drop_oldest_locked()
            if self._queue_bytes >= self._batch_limit or len(self._queue) >= self.cfg.max_batch_spans:
                self._cond.notify_all()
        COUNTERS.add("spans_queued")
        self._report_dropped(dropped)
        self._ensure_worker()

    def _pending_spans(self) -> int:
        return len(self._queue) + self._retry_spans

    def _drop_oldest_locked(self) -> int:
        """Enforce the byte and span caps by dropping the oldest spans (requeued batches first)."""
        dropped = 0
        while (self._queue or self._retry) and (self._queue_bytes > self.cfg.max_queue_bytes
                                                or self._pending_spans() > self.cfg.max_queue_spans):
            if self._retry:
                batch = self._retry[0]
                item = batch.items.popleft()
                batch.size -= item[2]
                self._retry_spans -= 1
                if not batch.items:
                    self._retry.popleft()
            else:
                item = self._queue.popleft()
            self._queue_bytes -= item[2]
            self._completed += 1
            dropped += 1
        return dropped

    def _report_dropped(self, dropped: int) -> None:
        if dropped:
            COUNTERS.add("spans_dropped", dropped)
            self.transport.report_dropped(dropped)
            log_once("queue-full", "zoroval: the export queue is full; the oldest spans are being dropped. "
                     "See zoroval.status() for counts.")

    def _put_back(self, *batches: _Batch) -> int:
        """Put batches back at the front of the queue, under the caps. Returns the spans put back."""
        count = 0
        with self._cond:
            for batch in reversed(batches):
                if batch.items:
                    self._retry.appendleft(batch)
                    self._retry_spans += len(batch.items)
                    self._queue_bytes += batch.size
                    count += len(batch.items)
            dropped = self._drop_oldest_locked()
        self._report_dropped(dropped)
        return count

    # ------------------------------------------------------------------ export thread

    def _take_batch(self) -> Optional[_Batch]:
        if self._retry:
            batch = self._retry.popleft()
            self._retry_spans -= len(batch.items)
            self._queue_bytes -= batch.size
            return batch
        items: Deque[_Item] = deque()
        size = 0
        while self._queue and len(items) < self.cfg.max_batch_spans:
            item = self._queue[0]
            if items and size + item[2] > self._batch_limit:
                break
            self._queue.popleft()
            self._queue_bytes -= item[2]
            size += item[2]
            items.append(item)
        return _Batch(items, uuid7()) if items else None

    def _next_batch(self) -> Optional[_Batch]:
        """Wait for work the exporter may send now. None means stop."""
        waited = False
        with self._cond:
            while True:
                pending = bool(self._queue or self._retry)
                if self._stopping and (not pending or time.monotonic() >= self._hard_deadline - 0.05):
                    self._cond.notify_all()
                    return None
                held = self.transport.hold_remaining()
                if held > 0:
                    # paused or backing off: send nothing; the queue keeps filling under its caps
                    if self._stopping and time.monotonic() + held >= self._hard_deadline:
                        self._cond.notify_all()
                        return None
                    self._cond.wait(held)
                    continue
                if (not waited and not self._stopping and not self._flush_requested and not self._retry
                        and self._queue_bytes < self._batch_limit
                        and len(self._queue) < self.cfg.max_batch_spans):
                    self._cond.wait(self.cfg.flush_interval)
                    waited = True
                    continue
                batch = self._take_batch()
                if not self._queue and not self._retry:
                    self._flush_requested = False
                if batch is not None:
                    return batch
                waited = False

    def _run(self) -> None:
        while True:
            batch = self._next_batch()
            if batch is None:
                return
            size = len(batch.items)
            put_back = 0
            try:
                put_back = self._export(batch)
            except Exception as error:  # noqa: BLE001
                internal_error("export", error)
                COUNTERS.add("spans_failed", size)
            with self._cond:
                self._completed += size - put_back
                self._cond.notify_all()

    def _shrink(self, size: int) -> None:
        self._batch_limit = max(_MIN_BATCH_BYTES, min(self._batch_limit, size) // 2)
        self._ok_streak = 0

    def _sent_ok(self) -> None:
        self._ok_streak += 1
        if self._batch_limit < self.cfg.max_batch_bytes and self._ok_streak >= _GROW_AFTER_OK:
            self._batch_limit = min(self.cfg.max_batch_bytes, self._batch_limit * 2)
            self._ok_streak = 0

    def _export(self, batch: _Batch) -> int:
        """Send one batch. Returns how many of its spans went back to the queue."""
        count = len(batch.items)
        if not self.cfg.api_key or self.transport.fatal_reason:
            COUNTERS.add("spans_failed", count)
            return 0
        spans = [item[0] for item in batch.items]
        extras = {item[0].context.span_id: item[1] for item in batch.items if item[1]}
        raw = self.transport.encode_spans(spans, extras)
        body = self.transport.compress(raw)
        if len(body) > self.cfg.max_request_bytes - WIRE_MARGIN:
            if count == 1:
                COUNTERS.add("spans_failed")
                self.transport.report_dropped(1)
                log_once("span-too-large", "zoroval: one span was larger than the gateway limit after "
                         "compression and was dropped. Lower max_content_bytes.")
                return 0
            COUNTERS.add("batches_split")
            return self._put_back(*batch.halves())
        timeout = self.transport.request_timeout_for(len(body))
        budget = max(_BATCH_BUDGET_S, 1.5 * timeout)
        deadline = min(time.monotonic() + budget, self._hard_deadline)
        result = self.transport.send(self.cfg.traces_url, body, deadline=deadline, batch_id=batch.batch_id,
                                     timeout=timeout, split_on_timeout=count > 1)
        if result.ok:
            self._sent_ok()
            COUNTERS.add("batches_sent")
            COUNTERS.add("spans_exported", count - result.rejected)
            if result.rejected:
                COUNTERS.add("spans_rejected", result.rejected)
            return 0
        if (result.status == 413 or result.timeout) and count > 1:
            if result.timeout:
                # A slow uplink: send halves now, and smaller batches until sends succeed again.
                self._shrink(batch.size)
            COUNTERS.add("batches_split")
            return self._put_back(*batch.halves())
        if result.fatal:
            COUNTERS.add("batches_failed")
            COUNTERS.add("spans_failed", count)
            self.transport.report_dropped(count)
            with self._cond:
                lost = self._pending_spans()
                self._queue.clear()
                self._retry.clear()
                self._retry_spans = 0
                self._queue_bytes = 0
                self._completed += lost
            if lost:
                COUNTERS.add("spans_failed", lost)
            return 0
        if result.paused:
            return self._put_back(batch)   # the gateway asked for a pause: not this batch's fault
        if result.retryable and self._stopping:
            # Shutdown cut the retries short; shutdown counts what is left. Wait before trying again.
            if result.reason not in ("shutdown", "deadline"):
                self.transport.back_off(result.reason)
            return self._put_back(batch)
        if result.retryable and batch.requeues < _MAX_REQUEUES:
            batch.requeues += 1
            COUNTERS.add("batches_requeued")
            delay = self.transport.back_off(result.reason or "unreachable")
            log_once("requeued", "zoroval: could not deliver a batch (%s). Its spans stay queued and are "
                     "retried in %.1fs, then with growing waits.", result.reason, delay)
            return self._put_back(batch)
        COUNTERS.add("batches_failed")
        COUNTERS.add("spans_failed", count)
        self.transport.report_dropped(count)
        return 0

    # ------------------------------------------------------------------ flush and shutdown

    def force_flush(self, timeout_millis: int = 5000) -> bool:
        return self.flush(timeout_millis / 1000.0)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every span queued before this call is sent or given up. Never raises."""
        try:
            deadline = time.monotonic() + max(timeout, 0.0)
            with self._cond:
                target = self._enqueued
                if self._completed >= target:
                    return True
                self._flush_requested = True
                self._cond.notify_all()
            self._ensure_worker()
            with self._cond:
                while self._completed < target:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(min(remaining, 0.25))
            return True
        except Exception as error:  # noqa: BLE001
            internal_error("flush", error)
            return False

    def shutdown(self, timeout: Optional[float] = None) -> None:
        """Send what is queued, then stop. Returns within `timeout` seconds (default 5)."""
        if self._stopped:
            return
        timeout = self.cfg.shutdown_timeout if timeout is None else timeout
        deadline = time.monotonic() + max(timeout, 0.0)
        try:
            with self._cond:
                self._hard_deadline = deadline
                self._stopping = True
                self._cond.notify_all()
            thread = self._thread
            if thread is not None and thread.is_alive() and self._pid == os.getpid():
                thread.join(max(deadline - time.monotonic() - 0.05, 0.0))
                if thread.is_alive():
                    self.transport.abort.set()
                    thread.join(0.1)
            with self._cond:
                left = self._pending_spans()
                self._queue.clear()
                self._retry.clear()
                self._retry_spans = 0
                self._queue_bytes = 0
            if left:
                COUNTERS.add("spans_failed", left)
            if thread is None or not thread.is_alive():
                self.transport.close()
        except Exception as error:  # noqa: BLE001
            internal_error("shutdown", error)
        finally:
            self._stopped = True

    def queue_state(self) -> dict[str, int]:
        with self._cond:
            return {"queue_spans": self._pending_spans(), "queue_bytes": self._queue_bytes}


def _after_fork_child() -> None:
    for processor in list(_LIVE):
        try:
            processor._after_fork_in_child()
        except Exception as error:  # noqa: BLE001
            internal_error("fork", error)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def live_processors() -> Iterable[ZorovalSpanProcessor]:
    return list(_LIVE)
