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
| TypeScript or JavaScript apps that call OpenAI | Use the npm package: `npm install zoroval`, then `npx zoroval login`. Works on Node.js, Next.js, Vercel and Cloudflare Workers. |
| Vercel AI SDK and other TypeScript providers | Send OpenTelemetry traces to Zoroval from `instrumentation.ts`. [get-started.md](https://github.com/Zoroval/zoroval-python/blob/main/get-started.md) shows how. |
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

Settings: `ZOROVAL_INGEST_KEY`, `ZOROVAL_SERVICE_NAME`, `ZOROVAL_ENVIRONMENT`, `ZOROVAL_RELEASE`, `ZOROVAL_SAMPLE_RATE`, `ZOROVAL_ENABLED`.

## Privacy

Prompts, outputs and tool calls are sent as they are unless you turn on redaction.

**Built-in redaction** is off by default. Turn it on with:

```python
zoroval.init(redact="basic")
```

Before anything leaves your app, it replaces these values with a label:

| Value | Label |
|---|---|
| Payment card numbers: 13 to 19 digits that start like a card and pass the card check digit | `[CARD]` |
| Email addresses | `[EMAIL]` |
| Phone numbers written with + and a country code, such as +91 98765 43210. Indian mobile numbers without + only when a phone word comes right before them (phone, mobile, mob, cell, contact, WhatsApp, call, ph, tel, "my number is") or they sit under a key such as phone, mobile, phone_number, contact or whatsapp | `[PHONE]` |
| Aadhaar numbers: 12 digits that pass the Aadhaar check digit | `[AADHAAR]` |
| PAN numbers | `[PAN]` |
| API keys and tokens with a known prefix (OpenAI, Anthropic, Stripe, Razorpay, GitHub, GitLab, AWS, Slack, Google, Hugging Face, npm, SendGrid, Zoroval), JWTs, bearer tokens and private keys | `[SECRET]` |

It finds these with or without spaces, dashes or line breaks, and in full-width or Indian-script digits. A 10-digit number on its own, or with only 0 or 91 in front, is kept: `8123456789` and `{"id": "9876543210"}` stay as they are, `call me on 98765 43210` and `{"phone": "9876543210"}` do not. A 12-digit number that could be Aadhaar is kept when a word such as order, PNR, invoice, AWB or tracking comes right before it or is its key, as in `{"order_id": "234567890124"}`. Numbers stored as integers are left alone unless the key names a phone, mobile, card or Aadhaar field, as in `{"phone": 9876543210}`.

It does not find names, addresses, health details, bank account or UPI IDs, landline numbers without +, or anything not in the table. It matches patterns, so it can miss a value written an unusual way, and it can replace an ID that happens to look like a card or Aadhaar number.

**Your own function:** `zoroval.init(redact=fn)` passes each prompt, output and tool call through `fn`. To use both, `zoroval.init(redact=["basic", fn])`: the built-in runs first and `fn` gets its output. If `fn` raises or returns `None`, that field is dropped, never sent raw.

- Redaction also covers the standard prompt and output fields of other tools' spans. The built-in also cleans metadata and error messages. User ids, conversation ids, tags and span names are sent as you set them.
- While redaction is on, GenAI log events are not sent.
- Redaction runs in your process. The built-in adds about 15 ms per 100 KB of text on a laptop, and nothing when it is off.
- Only LLM spans are sent. Your app's HTTP, database and other spans stay with you.

## License

Apache-2.0. See [LICENSE](https://github.com/Zoroval/zoroval-python/blob/main/LICENSE).
