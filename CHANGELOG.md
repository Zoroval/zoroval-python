# Changelog

## 0.3.0

- Breaking: `capture_content` is removed. `zoroval.init(capture_content=...)` now raises `TypeError`, and if `ZOROVAL_CAPTURE_CONTENT` is set to off, tracing stays off until you remove it. Use `redact` to clean content before it is sent.
- New: built-in redaction, off by default. `zoroval.init(redact="basic")` replaces card numbers, emails, phone numbers, Aadhaar, PAN and API keys with labels. `redact=["basic", fn]` runs it before your own function.
- `zoroval.status()` reports `redact` in place of `capture_content`.
- A value that cannot be turned into JSON is now dropped from that field instead of skipping the span.
