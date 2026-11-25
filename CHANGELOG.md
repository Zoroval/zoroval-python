# Changelog

## 0.3.1

- OpenAI Responses: built-in tool calls (web search, file search, code interpreter, MCP and others) are recorded as tool calls with their query or arguments, in the output and in the input history. Before, only their type was kept.

## 0.3.0

- Breaking: `capture_content` is removed. If an app still passes `capture_content=False` or sets `ZOROVAL_CAPTURE_CONTENT` to off, tracing stays off with a log line until you remove it, so an upgrade never starts sending content the app had turned off. `capture_content=True` has no effect. Use `redact` to clean content before it is sent.
- New: built-in redaction, off by default. `zoroval.init(redact="basic")` replaces card numbers, emails, phone numbers, Aadhaar, PAN and API keys with labels. `redact=["basic", fn]` runs it before your own function.
- `zoroval.status()` reports `redact` in place of `capture_content`.
- A value that cannot be turned into JSON is now dropped from that field instead of skipping the span.
