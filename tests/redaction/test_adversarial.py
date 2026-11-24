"""Attacks on the built-in redactor: slow matching, exceptions, memory, unicode, structures, threads.

Budgets are about three times what a 2024 laptop measures, so a slower CI runner passes and a
regression to super-linear matching fails. Measured numbers are in the PR description.
"""

import copy
import dataclasses
import json
import random
import sys
import threading
import time
import tracemalloc

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from zoroval import _redact
from zoroval._redact import RedactionError, redact_text, redact_value

MIB = 1 << 20
BUDGET_MS_PER_MIB = 1500      # worst adversarial input, per MiB (dense candidates: about 450 locally)
TYPICAL_BUDGET_MS_PER_MIB = 400  # chat-shaped JSON with numbers, emails and ids
settings.register_profile("zoroval", derandomize=True, deadline=None, max_examples=300,
                          suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("zoroval")


def _rng():
    return random.Random(20261124)


def typical_payload(size: int) -> str:
    """Chat-shaped JSON: messages with prose, prices, dates, order ids and a few level-1 values."""
    rng = _rng()
    lines = [
        "Hi, my order 403-5926381-7740324 placed on 2026-11-24 has not arrived.",
        "The total was Rs. 4,999 including 18% GST, invoice INV-2026-000123.",
        "Sure! I can see tracking number 789456123012 with Delhivery, ETA 26/11/2026.",
        "Please call me on +91 98765 43210 or mail priya.sharma@example.com.",
        "मेरा ऑर्डर अभी तक नहीं आया, कृपया मदद करें। कुल ₹1,299",
        "என் ஆர்டர் இன்னும் வரவில்லை, தயவுசெய்து உதவுங்கள்.",
        "Model gpt-4o-2024-08-06 replied in 1.2s; trace 4bf92f3577b34da6a3ce929d0e0e4736.",
        "I paid with card 4111 1111 1111 1111 but got charged twice 😞",
    ]
    messages, total = [], 0
    while total < size:
        content = " ".join(rng.choice(lines) for _ in range(rng.randint(1, 6)))
        messages.append({"role": rng.choice(["user", "assistant"]), "content": content})
        total += len(content) + 32
    return json.dumps({"messages": messages}, ensure_ascii=False)[:size]


ADVERSARIAL = {
    "digits": lambda n: "7" * n,
    "digits_spaced": lambda n: "1 " * (n // 2),
    "digits_dashed": lambda n: "4-" * (n // 2),
    "card_groups": lambda n: "4111 " * (n // 5),
    "newline_digits": lambda n: "4111\n" * (n // 5),
    "at_signs": lambda n: "@" * n,
    "a_at": lambda n: "a@" * (n // 2),
    "a_at_dot": lambda n: "a@a." * (n // 4),
    "local_part_no_at": lambda n: "a.b-c_d+" * (n // 8),
    "long_domain": lambda n: "x@" + "a." * (n // 2 - 1),
    "fullwidth_at": lambda n: "ａ＠" * (n // 2),
    "fullwidth_digits": lambda n: "４" * n,
    "zero_width_digits": lambda n: "4​" * (n // 2),
    "plus_digits": lambda n: "+1" * (n // 2),
    "plus_spaced": lambda n: "+1 2 3 4 5 6 7 8 " * (n // 17),
    "paren_digits": lambda n: "+(1)" * (n // 4),
    "sk_prefix": lambda n: "sk-" * (n // 3),
    "jwt_dots": lambda n: "eyJabcdefgh." * (n // 12),
    "pem_begin": lambda n: "-----BEGIN PRIVATE KEY-----" * (n // 27),
    "bearer": lambda n: "Bearer " * (n // 7),
    "pan_like": lambda n: "ABCPE1234 " * (n // 10),
    "deny_words": lambda n: "order no 9876543210 " * (n // 20),
    "hindi_digits": lambda n: "मेरा नंबर ९८७६५४३२१० है " * (n // 24),
    "tamil_mixed": lambda n: "என் எண் 98765 43210 😀 " * (n // 21),
    "rtl_mixed": lambda n: "‮رقم 4111 1111 1111 1111‬ " * (n // 28),
    "emoji": lambda n: "😀👍🏽🇮🇳" * (n // 7),
    "lone_surrogates": lambda n: "\ud800 1\udfff" * (n // 4),
    "json_numbers": lambda n: ",".join(str(10 ** 15 + i * 7919) for i in range(n // 17)),
    "random_chars": lambda n: "".join(map(chr, _rng().choices(range(32, 0x3000), k=n))),
}


def _time_ms(fn, *args):
    best = None
    for _ in range(2):
        began = time.perf_counter()
        fn(*args)
        took = (time.perf_counter() - began) * 1000
        best = took if best is None else min(best, took)
    return best


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_4mib_stays_within_budget(name):
    text = ADVERSARIAL[name](4 * MIB)
    mib = len(text) / MIB
    took = _time_ms(redact_text, text)
    assert took / mib < BUDGET_MS_PER_MIB, f"{name}: {took / mib:.0f} ms per MiB"


@pytest.mark.parametrize("name", ["digits_spaced", "card_groups", "a_at_dot", "plus_spaced", "hindi_digits",
                                  "deny_words", "long_domain"])
def test_matching_time_grows_linearly(name):
    small = _time_ms(redact_text, ADVERSARIAL[name](MIB))
    large = _time_ms(redact_text, ADVERSARIAL[name](4 * MIB))
    assert large < small * 4 * 2 + 50, f"{name}: 1 MiB {small:.0f} ms, 4 MiB {large:.0f} ms"


def test_typical_payload_latency_budget():
    text = typical_payload(4 * MIB)
    took = _time_ms(redact_text, text)
    assert took / (len(text) / MIB) < TYPICAL_BUDGET_MS_PER_MIB
    out = redact_text(text)
    assert "4111 1111" not in out and "98765 43210" not in out and "priya.sharma@" not in out
    assert "403-5926381-7740324" in out and "INV-2026-000123" in out and "789456123012" in out


def test_off_path_costs_nothing_when_nothing_matches():
    text = "plain words only, nothing to find here. " * (MIB // 40)
    assert redact_text(text) is text
    assert _time_ms(redact_text, text) < 60


# ---------------------------------------------------------------------------- memory

def test_no_memory_growth_over_repeated_calls():
    text = typical_payload(MIB)
    redact_text(text)
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for _ in range(10):
            redact_text(text)
        after, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert after - before < 64 * 1024
    assert peak - before < 10 * sys.getsizeof(text)  # a few working copies, nothing kept


def test_structure_memory_is_copy_on_change():
    clean = [{"role": "user", "content": f"message {i}"} for i in range(50_000)]
    assert redact_value(clean) is clean
    dirty = clean + [{"role": "user", "content": "x@y.co"}]
    out = redact_value(dirty)
    assert out[0] is dirty[0] and out[-1] == {"role": "user", "content": "[EMAIL]"}


# ---------------------------------------------------------------------------- exceptions and unicode

TRICKY = st.sampled_from(list("0123456789 -+@.()\n\r\t_/#:=") + [
    "４", "＠", "＋", "－", "​", "‍", "﻿", " ", "–", "९", "௯", "٣", "‮", "‏",
    "😀", "🇮🇳", "\ud800", "a", "Z", "sk-", "eyJ", "AKIA", "Bearer ", "-----BEGIN PRIVATE KEY-----", "[CARD]",
    "4111", "9876543210", "+91 ", "aadhaar ", "pnr ", "order ", "मेरा", "என்", "رقم",
])
TRICKY_TEXT = st.lists(TRICKY, max_size=80).map("".join)


@given(st.one_of(st.text(max_size=400), TRICKY_TEXT))
def test_never_raises_and_is_idempotent(text):
    once = redact_text(text)
    assert isinstance(once, str)
    assert redact_text(once) == once


@given(TRICKY_TEXT)
def test_labels_only_replace_never_invent(text):
    out = redact_text(text)
    if out != text:
        assert any(label in out for label in _redact.rules().labels)


PLANTS = st.sampled_from([
    ("4111 1111 1111 1111", "[CARD]"), ("4111111111111111", "[CARD]"), ("378282246310005", "[CARD]"),
    ("priya@example.com", "[EMAIL]"), ("+91 98765 43210", "[PHONE]"), ("phone 9876543210", "[PHONE]"),
    ("+1 (415) 555-2671", "[PHONE]"), ("ABCPE1234F", "[PAN]"), ("AKIAIOSFODNN7EXAMPLE", "[SECRET]"),
    ("sk-proj-abcdefghijklmnopqrstuvwxyz012345", "[SECRET]"),
])
SAFE_FILLER = st.lists(st.sampled_from(["hello", "नमस्ते", "வணக்கம்", "مرحبا", "😀", "price ₹499 only",
                                        "v1.2.3 ok", "on 2026-11-24 ok", "ok."]), max_size=6).map(" ".join)


@given(SAFE_FILLER, PLANTS, SAFE_FILLER, st.sampled_from([" ", "\n", ", ", ": ", " (", "\t", " ‏"]))
def test_planted_values_are_always_removed(before, plant, after, gap):
    value, label = plant
    text = f"{before}{gap}{value} {after}"
    out = redact_text(text)
    assert value not in out and label in out


@given(st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=True), TRICKY_TEXT, st.text(max_size=50)),
    lambda children: st.one_of(st.lists(children, max_size=6), st.dictionaries(st.text(max_size=8), children, max_size=6),
                               st.tuples(children, children)),
    max_leaves=40))
def test_structures_never_raise_and_are_not_mutated(value):
    before = copy.deepcopy(value)
    out = redact_value(value)
    assert repr(value) == repr(before)
    json.dumps(out, default=repr)
    if out is not value:
        assert redact_value(out) == out


def test_unicode_survives_untouched_around_matches():
    text = "\u202eरुको 😀 a@b.co 🇮🇳 \ud800 phone ９८७६५४३२१० சரி ௦"
    out = redact_text(text)
    assert out == "\u202eरुको 😀 [EMAIL] 🇮🇳 \ud800 phone [PHONE] சரி ௦"


def test_mixed_script_digits_inside_one_number_are_folded():
    assert redact_text("4111 １１１１ ११११ ௧௧௧௧") == "[CARD]"


# ---------------------------------------------------------------------------- structured values

def test_json_inside_strings_and_tool_arguments():
    args = json.dumps({"customer": {"email": "a@b.co", "phones": ["9876543210", "+44 20 7946 0958"]},
                       "order": "403-5926381-7740324"})
    out = json.loads(redact_value(args))
    assert out == {"customer": {"email": "[EMAIL]", "phones": ["[PHONE]", "[PHONE]"]},
                   "order": "403-5926381-7740324"}
    nested = json.dumps({"inner": json.dumps({"card": "4111111111111111"})})
    assert "4111" not in redact_value(nested)


def test_numbers_as_integers():
    assert redact_value({"phone": 9876543210, "card": 4111111111111111, "qty": 3, "ts": 1764000000000}) == \
        {"phone": "[PHONE]", "card": "[CARD]", "qty": 3, "ts": 1764000000000}
    assert redact_value(True) is True and redact_value(1.5) == 1.5


def test_keys_are_not_changed():
    assert redact_value({"a@b.co": "x@y.co"}) == {"a@b.co": "[EMAIL]"}


def test_tuples_sets_bytes_and_sdk_objects():
    @dataclasses.dataclass
    class Msg:
        role: str
        content: str

    class Model:
        def model_dump(self, **_):
            return {"content": "call 9876543210"}

    assert redact_value(("a@b.co", 1)) == ("[EMAIL]", 1)
    assert redact_value({"a@b.co"}) == ["[EMAIL]"]
    assert redact_value(b"a@b.co") == b"a@b.co"  # bytes are never sent, only their length
    assert redact_value(Msg("user", "a@b.co")) == {"role": "user", "content": "[EMAIL]"}
    assert redact_value(Model()) == {"content": "call [PHONE]"}
    clean = Msg("user", "hello")
    assert redact_value(clean) is clean


def test_deep_nesting_and_cycles_fail_closed():
    deep = current = {}
    for _ in range(5000):
        current["x"] = {}
        current = current["x"]
    with pytest.raises(RedactionError):
        redact_value(deep)
    loop = []
    loop.append(loop)
    with pytest.raises((RedactionError, ValueError)):
        redact_value(loop)


def test_big_arrays_are_fast_enough():
    items = [f"line {i} of the report" for i in range(200_000)] + ["mail a@b.co"]
    began = time.perf_counter()
    out = redact_value(items)
    assert time.perf_counter() - began < 6.0
    assert out[-1] == "mail [EMAIL]" and out[0] is items[0]


def test_threads_get_the_same_answers():
    texts = [typical_payload(64 * 1024) + str(i) for i in range(8)]
    expected = [redact_text(t) for t in texts]
    results = [None] * 32
    errors = []

    def work(i):
        try:
            results[i] = redact_text(texts[i % 8])
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and results == [expected[i % 8] for i in range(32)]
