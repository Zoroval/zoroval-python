# zoroval

The Zoroval Python SDK. It sends your LLM app's traces to Zoroval, grouped as whole conversations, so you can find and fix what goes wrong for your users.

- One `init()` line. OpenAI (Chat Completions and Responses) and Anthropic (Messages) are traced automatically: sync, async and streaming, with tool-call ids.
- One trace per conversation: wrap a chat in `zoroval.conversation(chat_id)`.
- Safe for production: it never raises into your code, never blocks a request, caps its memory in bytes, and exits within 5 seconds even when the network is down.
- Works beside Langfuse, OpenTelemetry, OpenInference and OpenLLMetry, in either install order, without duplicate spans.
- Apache-2.0.

Status: 0.1.0, not yet published. The gateway it talks to is being built.

## Install and set up

```bash
pip install zoroval
zoroval login      # opens the browser; writes ZOROVAL_INGEST_KEY to your git-ignored .env
zoroval doctor     # sends a test trace and checks it arrived
```

Using a coding agent? Point it at [get-started.md](get-started.md). The setup skill is in [skills/setup/SKILL.md](skills/setup/SKILL.md).

## Use

```python
import zoroval
from openai import OpenAI

zoroval.init()  # reads ZOROVAL_INGEST_KEY
client = OpenAI()

def handle_message(chat_id: str, user_id: str, text: str) -> str:
    with zoroval.conversation(chat_id, user_id=user_id):
        with zoroval.turn(input=text) as turn:
            reply = client.chat.completions.create(model="gpt-5", messages=history(chat_id, text))
            answer = reply.choices[0].message.content
            turn.set_output(answer)
            return answer
```

Agents and batch jobs that are not chats use `zoroval.trace()`:

```python
@zoroval.tool
def search_orders(customer_id: str) -> list: ...

@zoroval.step
def plan(goal: str) -> list: ...

with zoroval.trace("nightly-report", user_id="ops"):
    plan("summarize yesterday")
```

A `@zoroval.tool` run links to the model's tool-call id: pass `call_id=` to `zoroval.tool(...)`, or let the SDK match the one pending call with the same name (and arguments, when several are pending). Ambiguous calls stay unlinked; the SDK never pairs by position.

## API

| Call | What it does |
|---|---|
| `zoroval.init(api_key=None, *, service_name=None, environment=None, release=None, capture_content=True, redact=None, sample_rate=1.0, instrument=True, tracer_provider=None)` | Starts tracing. Safe to call twice. Returns `False` (and does nothing) without a valid key. |
| `zoroval.conversation(id, *, user_id=None, tags=None, metadata=None)` | Every turn inside joins one Zoroval trace. `with` or `async with`. |
| `zoroval.turn(name=None, *, input=None, ...)` | Root span of one user turn. |
| `zoroval.trace(name=None, *, input=None, user_id=None, tags=None, metadata=None)` | Root span of one run. Inside a conversation it acts as a turn. |
| `@zoroval.step`, `@zoroval.step(name=..., type="chain")`, `with zoroval.step("x"):` | A custom step; captures arguments and the return value (capped). |
| `@zoroval.tool`, `with zoroval.tool("name", call_id=..., arguments=...) as t:` | A tool run with its call id. |
| `zoroval.set_user()`, `set_tags()`, `set_metadata()` | On the current turn or trace, or on the current conversation. |
| `zoroval.flush(timeout=5.0)` | Send what is queued. Call it at the end of serverless handlers. |
| `zoroval.shutdown(timeout=5.0)` | Flush and stop. Runs at exit automatically. |
| `zoroval.status()` | Counters: queued, exported, dropped, failed, rejected, filtered, sampled out, internal errors, ungrouped turns. |

Environment variables: `ZOROVAL_INGEST_KEY`, `ZOROVAL_INGEST_URL`, `ZOROVAL_SERVICE_NAME` (or `OTEL_SERVICE_NAME`), `ZOROVAL_ENVIRONMENT`, `ZOROVAL_RELEASE`, `ZOROVAL_CAPTURE_CONTENT`, `ZOROVAL_SAMPLE_RATE`, `ZOROVAL_ENABLED`.

## Privacy

- Prompts and outputs are captured by default. `capture_content=False` keeps model names, usage, timings, errors and tool-call ids, and drops content.
- `redact=fn` gets each captured value and returns the value to send. If `fn` raises, that field is dropped, never sent raw. Changed spans carry `zoroval.redacted=true`.
- Both controls also cover spans from other tools (OpenInference, Langfuse, OpenLLMetry, OTel GenAI). Their content attributes (for example `input.value`, `llm.input_messages.*`, `langfuse.observation.input`, `gen_ai.prompt.*`, `gen_ai.input.messages`) are dropped, or passed through `redact` as the raw attribute value, and GenAI span events are dropped. Your other exporters still get the original spans.
- GenAI log events from the OTel contrib instrumentors are sent only when `capture_content=True` and no `redact` hook is set.
- `@zoroval.step` and `@zoroval.tool` capture all arguments and the return value. Use `capture_input=False` on functions that take secrets, or `capture_content=False`.
- Images and files in messages are replaced by placeholders.
- Only GenAI spans leave the process. Your app's HTTP, database and other spans are never sent to Zoroval.
- When Zoroval attaches to your existing OpenTelemetry provider, its LLM spans also reach your other exporters, with content. Use `capture_content=False` or a filter on that side if needed.

## How it behaves under load and failure

| Situation | Behavior |
|---|---|
| Gateway slow or down | Your requests are not slowed. Spans wait in a queue capped at 64 MiB (`max_queue_bytes`) and 20,000 spans; the oldest are dropped first and counted. The next accepted request reports the count in `X-Zoroval-Dropped-Spans`. |
| Process exit with the gateway down | Exit waits at most `shutdown_timeout` (5 s). |
| Large prompts | Each content field is capped at 4 MiB (`max_content_bytes`), cut so the JSON stays valid, with `zoroval.truncated_bytes` set. Metadata is capped at 16 KB. |
| Batches | Cut by bytes (16 MiB raw), zstd level 3, split again if the compressed body passes the 16 MiB gateway limit. Retries 429 and 503 after `Retry-After`, and 5xx with backoff, reusing the batch id. |
| Invalid or revoked key | Export stops for the process; `zoroval.status()["key_refused"]` says why. |
| gunicorn, uvicorn workers, Celery prefork | The export thread restarts in each forked worker. The parent's queued spans are not sent twice. |
| Serverless | Call `zoroval.flush()` at the end of each handler. |
| Head sampling | `sample_rate=0.2` keeps 20% of conversations, each kept whole. |

## Coexistence

| You already use | What `init()` does |
|---|---|
| Your own OpenTelemetry `TracerProvider` | Adds Zoroval's processor to it. If you set yours after `init()`, your processors are attached to the global provider so both keep receiving spans. |
| Langfuse v3+ (`@observe`, `langfuse.openai`) | Attaches to Langfuse's provider and reads its generation spans; does not add a second OpenAI span. Langfuse's `session_id` groups conversations. |
| OpenInference, OpenLLMetry, OTel contrib instrumentors | Reads their spans; does not wrap a client they already wrap. GenAI log events from the contrib instrumentors are sent too. |
| The Anthropic client's built-in OpenTelemetry span | Not sent twice: Zoroval's generation span covers the call. |

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]' openinference-instrumentation-openai python-dotenv
.venv/bin/python -m pytest
```

Tests run against a fake Zoroval gateway (`testing/fake_gateway.py`) and fake OpenAI and Anthropic servers (`testing/fake_llm.py`). No real API calls.

## License

Apache-2.0. See [LICENSE](LICENSE).
