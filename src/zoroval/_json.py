"""Content capture: JSON serialization, the redact hook and size caps.

Every captured value goes through `capture()`. It applies the redact hook, serializes to JSON
and cuts long strings until the result fits the cap, so the JSON stays valid. If the redact
hook raises, the content is dropped (fail closed): unredacted content never leaves the app.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import json
from typing import Any, Callable, NamedTuple, Optional

from ._util import COUNTERS, log_once

_NO_REDACT = object()


class Captured(NamedTuple):
    text: Optional[str]      # JSON text, or None when nothing should be sent
    redacted: bool           # the redact hook changed the value
    truncated_bytes: int     # bytes removed by the size cap


def _default(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except TypeError:
            return dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"[{len(value)} bytes omitted]"
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # noqa: BLE001
            pass
    return repr(value)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_default, separators=(",", ":"))


def to_plain(value: Any) -> Any:
    """Turn SDK objects (pydantic models, dataclasses) into plain JSON values."""
    return json.loads(dumps(value))


def _utf8_len(text: str) -> int:
    if text.isascii():
        return len(text)
    return len(text.encode("utf-8", "surrogatepass"))


def _shrink(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) > max_chars:
            cut = len(value) - max_chars
            return value[:max_chars] + f"...[{cut} chars truncated]"
        return value
    if isinstance(value, list):
        return [_shrink(v, max_chars) for v in value]
    if isinstance(value, dict):
        return {k: _shrink(v, max_chars) for k, v in value.items()}
    return value


def fit(text: str, limit: int) -> tuple[str, int]:
    """Cut a JSON text to at most `limit` UTF-8 bytes, keeping it valid JSON."""
    size = _utf8_len(text)
    if size <= limit:
        return text, 0
    try:
        value = json.loads(text)
    except ValueError:
        value = text
    max_chars = max(limit // 2, 64)
    for _ in range(24):
        candidate = dumps(_shrink(value, max_chars))
        new_size = _utf8_len(candidate)
        if new_size <= limit:
            COUNTERS.add("truncated_fields")
            return candidate, size - new_size
        if max_chars <= 16:
            break
        max_chars //= 2
    # Too many small fields: keep the head as one string.
    head = text[: max(limit // 4 - 64, 0)]
    candidate = dumps({"zoroval.truncated": True, "head": head})
    COUNTERS.add("truncated_fields")
    return candidate, size - _utf8_len(candidate)


def apply_redact(redact: Optional[Callable[[Any], Any]], value: Any) -> tuple[Any, bool, bool]:
    """Returns (value, redacted, ok). ok=False means drop the content."""
    if redact is None:
        return value, False, True
    try:
        new = redact(value)
    except Exception as error:  # noqa: BLE001
        COUNTERS.add("redact_errors")
        log_once(f"redact:{type(error).__name__}",
                 "zoroval: the redact hook raised %s; content was dropped for that field.",
                 type(error).__name__)
        return None, True, False
    if new is None:
        return None, True, False
    changed = new is not value
    if changed:
        try:
            changed = new != value
        except Exception:  # noqa: BLE001
            changed = True
    return new, bool(changed), True


def capture(value: Any, *, capture_content: bool, redact: Optional[Callable[[Any], Any]],
            limit: int) -> Captured:
    """Serialize content for a span attribute. Returns text=None when nothing is sent."""
    if not capture_content or value is None:
        return Captured(None, False, 0)
    value, redacted, ok = apply_redact(redact, value)
    if not ok:
        return Captured(None, True, 0)
    text = dumps(value)
    text, cut = fit(text, limit)
    return Captured(text, redacted, cut)


def data_url_placeholder(url: Any) -> Any:
    """Media is not captured in v1: base64 data URLs become a short placeholder."""
    if isinstance(url, str) and url.startswith("data:"):
        kind = url[5:url.find(";")] if ";" in url[:64] else "media"
        return f"[{kind} omitted]"
    return url
