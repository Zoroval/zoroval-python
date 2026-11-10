# zoroval

The Zoroval Python SDK. It sends your LLM app's traces to Zoroval as whole conversations.

- One line to start. OpenAI and Anthropic calls are traced automatically.
- One trace per conversation.
- Safe for production: it never raises into your code and never slows a request.
- Keeps your current tracing. No rip and replace.

## Install and set up

```bash
pip install zoroval
zoroval login      # opens the browser; writes your key to the git-ignored .env
zoroval doctor     # checks that a trace arrived
```

Using a coding agent? Point it at [get-started.md](https://github.com/Zoroval/zoroval-python/blob/main/get-started.md).

## Works with what you already use

Add Zoroval beside your current tools. They keep working as before.

| You use | With Zoroval |
|---|---|
| Langfuse | Keep it. Zoroval reads the same traces, with no second wrapper and no duplicate spans. Langfuse sessions become Zoroval conversations. You can also upload a Langfuse export in the app. |
| OpenTelemetry | Zoroval joins your existing setup. Only LLM spans come to Zoroval. You can also add Zoroval as a second exporter in your Collector. |
| LangChain, LlamaIndex and other OpenInference frameworks | Keep your instrumentors. Their sessions group into conversations. |
| OpenLLMetry (Traceloop) | Keep it. Zoroval reads its spans. |
| LangSmith | Runs beside it. |
| Vercel AI SDK and other TypeScript apps | Send OpenTelemetry traces to Zoroval from `instrumentation.ts`. [get-started.md](https://github.com/Zoroval/zoroval-python/blob/main/get-started.md) shows how. |
| gunicorn, uvicorn, Celery, serverless | Works with no extra setup. In serverless handlers, call `zoroval.flush()` at the end. |

Your coding agent finds what you use and sets up the right path.

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

with zoroval.trace("nightly-report"):
    plan("summarize yesterday")
```

## API

| Call | What it does |
|---|---|
| `zoroval.init()` | Starts tracing. Safe to call twice. |
| `zoroval.conversation(id, *, user_id=None)` | Every turn inside joins one trace. |
| `zoroval.turn(input=None)` | One user turn. |
| `zoroval.trace(name=None)` | One run of an agent or job. |
| `@zoroval.step`, `@zoroval.tool` | A custom step or tool run. |
| `zoroval.flush()` | Sends what is queued. Call it at the end of serverless handlers. |

Settings: `ZOROVAL_INGEST_KEY`, `ZOROVAL_SERVICE_NAME`, `ZOROVAL_ENVIRONMENT`, `ZOROVAL_RELEASE`, `ZOROVAL_CAPTURE_CONTENT`, `ZOROVAL_SAMPLE_RATE`, `ZOROVAL_ENABLED`.

## Privacy

- Prompts and outputs are sent by default. `zoroval.init(capture_content=False)` keeps model names, usage, timings and errors, and drops content.
- `zoroval.init(redact=fn)` passes each captured value through your function before it is sent. If `fn` raises, that field is dropped.
- Only LLM spans are sent. Your app's HTTP, database and other spans stay with you.

## License

Apache-2.0. See [LICENSE](https://github.com/Zoroval/zoroval-python/blob/main/LICENSE).
