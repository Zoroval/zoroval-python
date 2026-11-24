"""Built-in redaction: level-1 rules (cards, emails, phones, Aadhaar, PAN, keys and tokens).

The rules live in `redaction/level1.json`, shared with the TypeScript SDK and Zoroval's ingest. This
module only applies them: fold, prefilter, match, validate, replace. Nothing here runs unless the
developer turns redaction on.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from typing import Any, Callable, Optional

BUILTIN_NAMES = ("basic",)
MAX_DEPTH = 200          # nested containers deeper than this are not scanned: the field is dropped
MIN_INT_DIGITS = 10      # smaller integers cannot hold any level-1 value


class RedactionError(Exception):
    """The value could not be scanned. The caller drops the field."""


# ---------------------------------------------------------------------------- validators

def luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = ord(ch) - 48
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


_V_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5), (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7), (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3), (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_V_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4), (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7), (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _V_D[c][_V_P[i % 8][ord(ch) - 48]]
    return c == 0


VALIDATORS: dict[str, Callable[[str], bool]] = {"luhn": luhn, "verhoeff": verhoeff}
_NOT_DIGIT = re.compile(r"[^0-9]")


# ---------------------------------------------------------------------------- rules

class _Rule:
    __slots__ = ("id", "label", "regex", "prefilter", "validators", "deny", "window")

    def __init__(self, spec: dict[str, Any]) -> None:
        self.id = spec["id"]
        self.label = spec["label"]
        self.regex = re.compile(spec["pattern"])
        self.prefilter = re.compile(spec["prefilter"]) if spec.get("prefilter") else None
        self.validators = [VALIDATORS[name] for name in spec.get("validators", ())]
        deny = spec.get("deny_prefix")
        self.deny = re.compile(deny["pattern"]) if deny else None
        self.window = int(deny["window"]) if deny else 0


class Rules:
    def __init__(self, doc: dict[str, Any]) -> None:
        fold: dict[int, int] = {}
        for entry in doc["fold"]:
            first, last = entry["from"]
            for cp in range(first, last + 1):
                fold[cp] = entry["to"] + (cp - first) if entry.get("shift") else entry["to"]
        self.fold = fold
        # One fast scan decides whether the slower per-character fold is needed at all.
        ranges = "".join(f"\\u{e['from'][0]:04x}-\\u{e['from'][1]:04x}" for e in doc["fold"])
        self.foldable = re.compile(f"[{ranges}]")
        self.rules = [_Rule(spec) for spec in doc["rules"]]
        self.labels = tuple(doc["labels"])


def load_rules() -> Rules:
    text = resources.files("zoroval").joinpath("redaction/level1.json").read_text(encoding="utf-8")
    return Rules(json.loads(text))


_RULES: Optional[Rules] = None


def rules() -> Rules:
    global _RULES
    if _RULES is None:
        _RULES = load_rules()
    return _RULES


# ---------------------------------------------------------------------------- strings

def redact_text(text: str, rs: Optional[Rules] = None) -> str:
    """Replace level-1 values in one string with labels. Returns the same object when nothing matched."""
    if not text:
        return text
    rs = rs or rules()
    shadow = text if text.isascii() or not rs.foldable.search(text) else text.translate(rs.fold)
    same = shadow is text
    passed: dict[Any, bool] = {}
    for rule in rs.rules:
        pre = rule.prefilter
        if pre is not None:
            hit = passed.get(pre.pattern)
            if hit is None:
                hit = passed[pre.pattern] = pre.search(shadow) is not None
            if not hit:
                continue
        out: list[str] = []
        sh: list[str] = []
        last = 0
        for match in rule.regex.finditer(shadow):
            start, end = match.span()
            if start == end:
                continue
            if rule.validators:
                digits = _NOT_DIGIT.sub("", match.group())
                if not all(check(digits) for check in rule.validators):
                    continue
            if rule.deny is not None and rule.deny.search(shadow[max(0, start - rule.window):start].lower()):
                continue
            out.append(text[last:start])
            out.append(rule.label)
            if not same:
                sh.append(shadow[last:start])
                sh.append(rule.label)
            last = end
        if not out:
            continue
        out.append(text[last:])
        text = "".join(out)
        if same:
            shadow = text
        else:
            sh.append(shadow[last:])
            shadow = "".join(sh)
        # Labels may contain characters the prefilters look for; check again next time.
        passed.clear()
    return text


# ---------------------------------------------------------------------------- values

_UNCHANGED = object()


def _walk(value: Any, depth: int, rs: Rules) -> Any:
    """Returns a redacted copy, or _UNCHANGED. Never mutates `value`."""
    if isinstance(value, str):
        new = redact_text(value, rs)
        return _UNCHANGED if new is value else new
    if value is None or isinstance(value, (bool, float)):
        return _UNCHANGED
    if isinstance(value, int):
        if -10 ** (MIN_INT_DIGITS - 1) < value < 10 ** (MIN_INT_DIGITS - 1):
            return _UNCHANGED
        text = str(value)
        new = redact_text(text, rs)
        return _UNCHANGED if new == text else new
    if depth >= MAX_DEPTH:
        raise RedactionError("value nested too deep to scan")
    if isinstance(value, dict):
        changed: Optional[dict] = None
        for key, item in value.items():
            new = _walk(item, depth + 1, rs)
            if new is not _UNCHANGED:
                if changed is None:
                    changed = dict(value)
                changed[key] = new
        return _UNCHANGED if changed is None else changed
    if isinstance(value, (list, tuple)):
        items: Optional[list] = None
        for i, item in enumerate(value):
            new = _walk(item, depth + 1, rs)
            if new is not _UNCHANGED:
                if items is None:
                    items = list(value)
                items[i] = new
        if items is None:
            return _UNCHANGED
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _UNCHANGED  # never sent: serialized as "[N bytes omitted]"
    # SDK objects (pydantic models, dataclasses, sets): scan what will be sent.
    from ._json import to_plain
    plain = to_plain(value)
    if isinstance(plain, (dict, list, str, int)) and not isinstance(plain, bool):
        new = _walk(plain, depth + 1, rs)
        return _UNCHANGED if new is _UNCHANGED else new
    return _UNCHANGED


def redact_value(value: Any) -> Any:
    """The built-in redact hook. Same object back when nothing matched. Raises RedactionError when
    the value cannot be scanned, so the caller drops the field (fail closed)."""
    rs = rules()
    try:
        new = _walk(value, 0, rs)
    except RecursionError as error:
        raise RedactionError("value nested too deep to scan") from error
    return value if new is _UNCHANGED else new


# ---------------------------------------------------------------------------- init(redact=...)

def build_hook(redact: Any) -> tuple[Optional[Callable[[Any], Any]], Optional[str]]:
    """Turn init(redact=...) into one callable, plus a short description for status().

    Accepts "basic", a callable, or a list or tuple of both. The built-in always runs first, then
    the developer's function on its output. Anything else is a coding error: TypeError.
    """
    if redact is None:
        return None, None
    parts = list(redact) if isinstance(redact, (list, tuple)) else [redact]
    builtin = False
    functions: list[Callable[[Any], Any]] = []
    for part in parts:
        if isinstance(part, str):
            if part not in BUILTIN_NAMES:
                raise TypeError(f"zoroval.init(redact=...) got {part!r}; use \"basic\" or a function")
            builtin = True
        elif callable(part):
            functions.append(part)
        else:
            raise TypeError("zoroval.init(redact=...) takes \"basic\", a function, or a list of both")
    if len(functions) > 1:
        raise TypeError("zoroval.init(redact=...) takes at most one function")
    if builtin:
        rules()  # load now, so a broken install shows at init and not on the first call
    fn = functions[0] if functions else None
    if builtin and fn is None:
        return redact_value, "basic"
    if fn is None:
        return None, None
    if not builtin:
        return fn, "function"

    def chained(value: Any) -> Any:
        return fn(redact_value(value))

    return chained, "basic+function"
