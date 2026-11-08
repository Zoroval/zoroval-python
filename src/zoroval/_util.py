"""Small internal helpers: one-time logging, the never-raise guard, counters."""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable, TypeVar

logger = logging.getLogger("zoroval")

_logged: set[str] = set()
_logged_lock = threading.Lock()

F = TypeVar("F", bound=Callable[..., Any])


def log_once(key: str, message: str, *args: Any, level: int = logging.WARNING) -> None:
    """Log a message the first time `key` is seen in this process."""
    with _logged_lock:
        if key in _logged:
            return
        _logged.add(key)
    logger.log(level, message, *args)


def reset_log_once() -> None:
    with _logged_lock:
        _logged.clear()


class Counters:
    """Process-wide counters. Increments are not locked: under heavy contention a count can be off by one."""

    FIELDS = (
        "spans_seen",           # spans that ended while the SDK was active
        "spans_queued",         # spans accepted into the export queue
        "spans_filtered",       # non-GenAI spans of the host app, never sent
        "spans_sampled_out",    # dropped by head sampling
        "spans_dropped",        # pushed out of a full queue (oldest first)
        "spans_exported",       # in batches the gateway accepted
        "spans_rejected",       # refused by the gateway inside an accepted batch (partial success)
        "spans_failed",         # in batches that failed after retries
        "batches_sent",
        "batches_failed",
        "batches_split",        # halved after a 413 or a timeout on a slow uplink
        "batches_requeued",     # put back on the queue after retries ran out while the gateway was down
        "bytes_sent",           # compressed bytes on the wire
        "internal_errors",      # SDK bugs caught before they reached the app
        "redact_errors",        # redact hook raised; content was dropped
        "truncated_fields",     # content fields cut by size caps
        "ungrouped_turns",      # turn() roots with no conversation id
        "logs_exported",
        "logs_failed",
    )

    def __init__(self) -> None:
        for name in self.FIELDS:
            setattr(self, name, 0)

    def add(self, name: str, value: int = 1) -> None:
        setattr(self, name, getattr(self, name) + value)

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.FIELDS}


COUNTERS = Counters()


def internal_error(where: str, error: BaseException) -> None:
    COUNTERS.add("internal_errors")
    log_once(f"internal:{where}:{type(error).__name__}",
             "zoroval: internal error in %s (%s). Tracing continues; this is logged once.",
             where, type(error).__name__)
    logger.debug("zoroval internal error in %s", where, exc_info=error)


def never_raise(where: str, default: Any = None) -> Callable[[F], F]:
    """Decorator for SDK internals: an exception is counted and logged once, never raised."""

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as error:  # noqa: BLE001 - the point is to catch everything
                internal_error(where, error)
                return default
        return wrapper  # type: ignore[return-value]

    return decorate
