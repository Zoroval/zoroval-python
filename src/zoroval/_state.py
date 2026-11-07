"""The one process-wide SDK state. `None` until init() succeeds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ._config import Config


@dataclass
class State:
    cfg: Config
    tracer: Any
    provider: Any
    owns_provider: bool
    processor: Any
    transport: Any
    log_processor: Any = None
    instrumented: dict[str, bool] = field(default_factory=dict)
    pid: int = 0


STATE: Optional[State] = None


def get() -> Optional[State]:
    return STATE


def set_state(state: Optional[State]) -> None:
    global STATE
    STATE = state
