# Changelog

## 0.3.1

- OpenAI Responses: built-in tool calls (web search, file search, code interpreter, MCP and others) are recorded as tool calls with their query or arguments, in the output and in the input history. Before, only their type was kept.
- OpenAI Responses: one output message per output item, the same messages the app's history makes when it sends those items back, so a reply is not shown twice.
- OpenAI built-in tools: their results (web search sources, file search results, code output, MCP output) are recorded as tool results with the call's id. Web search arguments keep the query, not the sources. Images and other inline base64 are left out.
- Reasoning from OpenAI-compatible providers is recorded: DeepSeek `reasoning_content`, OpenRouter `reasoning` and `reasoning_details`, in replies and in streams.
- Citations are kept on text: OpenAI `annotations` and Anthropic `citations`, in the output and in the history.
- Groq streams: token usage is read from `x_groq` when the chunk has no usage of its own.
- `gen_ai.provider.name` follows the base URL for compatible providers (Groq, DeepSeek, OpenRouter, Mistral, xAI, Gemini, Bedrock and others) instead of always `openai`. Anthropic on Bedrock and Vertex is named for them.
- Anthropic server tools (web search, web fetch, code execution) and MCP tools: calls and their results are recorded, streamed and not.
- Anthropic: `input_tokens` is now the whole prompt, including cache reads and cache writes, as in the OpenTelemetry GenAI conventions. Cache reads and writes are still sent on their own.
- Anthropic: a tool result made of text blocks is recorded as one text.
- Streams read through `with_raw_response` or `with_streaming_response` (OpenAI and Anthropic) now record their reply. The OpenAI Agents SDK's streamed runs and LangChain's ChatAnthropic streams used this path and lost it.
- No second span when OpenInference's LangChain or LlamaIndex instrumentor already records the call. A direct client call inside a tool is still recorded.
- `zoroval.init()` creates an OpenTelemetry LoggerProvider when the app has none, so instrumentors that send content as log events (OpenTelemetry botocore, google-genai) are exported. An app that sets its own LoggerProvider later keeps its exporters. Not done when `redact` is set.
- LiteLLM with `callbacks=["otel"]`: streamed calls keep their conversation, so one chat is no longer split into several traces.
- `zoroval detect`: LangChain, LlamaIndex, OpenAI Agents SDK and Pydantic AI apps get `zoroval.init()` in their plan; OpenLLMetry instrumentors for more providers are recognised; a Gemini (google-genai) app gets `zoroval.init()` and the instrumentor to add.

## 0.3.0

- Breaking: `capture_content` is removed. If an app still passes `capture_content=False` or sets `ZOROVAL_CAPTURE_CONTENT` to off, tracing stays off with a log line until you remove it, so an upgrade never starts sending content the app had turned off. `capture_content=True` has no effect. Use `redact` to clean content before it is sent.
- New: built-in redaction, off by default. `zoroval.init(redact="basic")` replaces card numbers, emails, phone numbers, Aadhaar, PAN and API keys with labels. `redact=["basic", fn]` runs it before your own function.
- `zoroval.status()` reports `redact` in place of `capture_content`.
- A value that cannot be turned into JSON is now dropped from that field instead of skipping the span.
