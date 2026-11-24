"""The shared level-1 rule file and corpus: exact results, language-neutral syntax, JavaScript parity."""

import json
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import pytest

from zoroval import _redact

HERE = Path(__file__).parent
CORPUS = json.loads((HERE / "level1_corpus.json").read_text(encoding="utf-8"))
RULES_PATH = Path(str(resources.files("zoroval").joinpath("redaction/level1.json")))
RULES = json.loads(RULES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CORPUS["must_redact"], ids=lambda c: c["id"])
def test_must_redact(case):
    assert _redact.redact_text(case["text"]) == case["expect"]


@pytest.mark.parametrize("case", CORPUS["must_keep"], ids=lambda c: c["id"])
def test_must_keep(case):
    assert _redact.redact_text(case["text"]) is case["text"]


def test_corpus_is_big_enough_and_ids_unique():
    ids = [c["id"] for c in CORPUS["must_redact"] + CORPUS["must_keep"]]
    assert len(ids) == len(set(ids))
    assert len(CORPUS["must_redact"]) >= 80 and len(CORPUS["must_keep"]) >= 70
    labels_used = {label for c in CORPUS["must_redact"] for label in re.findall(r"\[[A-Z]+\]", c["expect"])}
    assert labels_used == set(RULES["labels"])


# ---------------------------------------------------------------------------- language-neutral syntax

PATTERNS = [(r["id"], key, value) for r in RULES["rules"]
            for key, value in (("pattern", r["pattern"]), ("prefilter", r.get("prefilter")),
                               ("deny_prefix", (r.get("deny_prefix") or {}).get("pattern"))) if value]


@pytest.mark.parametrize("rule_id,key,pattern", PATTERNS, ids=lambda v: str(v)[:40])
def test_pattern_uses_only_portable_syntax(rule_id, key, pattern):
    # \d \w \b \s-free character classes would differ between Python (Unicode) and JS (ASCII)
    for token in (r"\d", r"\D", r"\w", r"\W", r"\b", r"\B", "(?P", "(?i", "(?m", "(?s", "(?x", "(?>", "*+", "++", "?+"):
        assert token not in pattern, f"{rule_id}.{key} uses {token}"
    re.compile(pattern)  # Python accepts only fixed-width lookbehinds


def test_rules_are_well_formed():
    assert [r["id"] for r in RULES["rules"]] == ["secret", "email", "card", "phone_international", "aadhaar",
                                                  "phone_india", "pan"]
    assert {r["label"] for r in RULES["rules"]} == set(RULES["labels"])
    for rule in RULES["rules"]:
        assert set(rule.get("validators", [])) <= set(_redact.VALIDATORS)
        assert set(rule) <= {"id", "label", "covers", "prefilter", "pattern", "validators", "deny_prefix"}
    for entry in RULES["fold"]:
        first, last = entry["from"]
        assert 0 < first <= last <= 0xFFFF, "fold stays in the BMP so offsets match in UTF-16"
        assert not 0xD800 <= first <= 0xDFFF and not 0xD800 <= last <= 0xDFFF
        top = entry["to"] + (last - first if entry.get("shift") else 0)
        assert 0x20 <= entry["to"] <= top <= 0x7E, "fold targets are printable ASCII"
    sources = [cp for e in RULES["fold"] for cp in range(e["from"][0], e["from"][1] + 1)]
    assert len(sources) == len(set(sources)), "fold ranges overlap"


def test_labels_never_match_any_rule():
    for label in RULES["labels"]:
        text = f"{label} {label}{label} x{label}1"
        assert _redact.redact_text(text) == text


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_reference_gives_the_same_results():
    out = subprocess.run(["node", str(HERE / "reference.mjs"), str(RULES_PATH), str(HERE / "level1_corpus.json")],
                         capture_output=True, text=True, timeout=60, check=True)
    result = json.loads(out.stdout)
    assert result["failures"] == []
    assert result["checked"] == len(CORPUS["must_redact"]) + len(CORPUS["must_keep"])


# ---------------------------------------------------------------------------- validators

def test_luhn_and_verhoeff_known_values():
    assert _redact.luhn("4111111111111111") and not _redact.luhn("4111111111111112")
    assert _redact.luhn("378282246310005")
    assert _redact.verhoeff("2363") and not _redact.verhoeff("2364")  # the classic Verhoeff example
