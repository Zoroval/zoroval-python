---
name: zoroval-setup
description: Set up Zoroval tracing in a Python or TypeScript/JavaScript repo end to end. Detects the starting point (no tracing, Langfuse, OpenTelemetry only, OpenInference or OpenLLMetry, LangSmith, Vercel AI SDK, serverless or pre-fork servers, monorepos, env files and secrets managers), shows a plan and a diff, signs in with the device flow, proves the first trace arrived grouped as whole conversations, and leaves a rollback list. Use when the user asks to add Zoroval, send traces to Zoroval, or set up LLM monitoring with Zoroval.
---

# Set up Zoroval

Goal: the app's LLM calls show up in Zoroval as whole conversations (one trace per chat), and you prove it with `zoroval doctor`.

## Which SDK

Use the SDK for the language of the code that calls the model, not for the repo's other files. A Python app behind a Cloudflare Worker that only forwards requests is a Python service; leave the Worker alone.

| The model is called from | SDK | Commands |
|---|---|---|
| Python (OpenAI, Anthropic) | `pip install zoroval` | `zoroval detect\|login\|doctor\|logout` |
| TypeScript or JavaScript (OpenAI through the `openai` package or `fetch`) | `npm install zoroval` | `npx zoroval detect\|login\|doctor\|logout` |
| The Vercel AI SDK (`ai`) | no SDK, OpenTelemetry (case 6) | `npx zoroval login\|doctor` |

Both CLIs take the same flags and print the same report. Below, `zoroval <command>` means the one for the service you are working on, so a TypeScript service runs `npx zoroval <command>`. Never ask a TypeScript service to install Python, or a Python service to install Node packages. For several services, or both languages, set up each service with its own SDK (case 8). The npm package covers OpenAI; for another provider in TypeScript, tell the user it is not covered yet and do not wire it.

## Rules (never break these)

1. Never print, log or paste an ingest key. Never write it into a tracked file. Never commit it.
2. Read env var names only, never values. `zoroval detect` already does this.
3. Show the diff before every edit and wait for the user's yes.
4. Ask before touching production config: deploy files, CI, hosting dashboards, secrets managers, OTel Collector config. For production, print the production login command (section 9); the user runs it themselves.
5. Do not commit. List every file you changed, so rollback is "restore these files and run `zoroval logout`".
6. Never copy another vendor's secret (Langfuse, LangSmith, Phoenix) into Zoroval config, and never send it anywhere.
7. When you cannot tell (two possible conversation ids, an unknown framework), stop and ask.
8. If the prompt gave a `ZOROVAL_SETUP` value, keep it set for every `zoroval` command you run.

## Order

1. **Detect.** Run `pip install zoroval` (or the project's package manager) in the project's environment, then `zoroval detect --json` (for TypeScript: `npm install zoroval`, or the project's package manager, then `npx zoroval detect --json`). Read `cases`, `explain_only`, `llm_calls`, `chat_id_candidates`, `entry_points`, `env` and `notes`. If `env.ZOROVAL_INGEST_KEY` is `present`, skip the login step.
2. **Plan.** Tell the user, in a short list: which case(s) apply, which files you will change, and what the conversation id will be. Use the case sections below. Start from `plan` in the detect output.
3. **Ask.** Get a yes on the plan. Ask which variable is the conversation id when `notes` says there are several, or none.
4. **Edit.** For each file: show the diff, wait for yes, apply. Keep edits minimal. The app must load the env file at startup (Python: for example python-dotenv; TypeScript: `.env.local` is loaded by Next.js and `.dev.vars` by `wrangler dev`, other Node apps need dotenv or `node --env-file=.env` on Node 20.6 or newer) so `ZOROVAL_INGEST_KEY` is set; add that if it does not already, after asking.
5. **Log in.** Run `zoroval login`. With a `ZOROVAL_SETUP` value it connects on its own. Without one it prints a link and a code; ask the user to open it, sign in, check the code and approve. The command writes the key to the git-ignored env file with mode 600 and never shows it (TypeScript: `.dev.vars` for a Worker, `.env.local` for Next.js, `.env` otherwise). If it refuses because the env file is not git-ignored, show the `.gitignore` diff that adds it, wait for yes, then run it again.
6. **Prove.** Run `zoroval doctor`. Then run the app once (or ask the user to send two messages in one chat) and run `zoroval doctor --wait 60`. Success: every required check is `ok`, "first trace received" passes, and for chat apps there is no "ungrouped" warning (TypeScript on a Worker: no "worker context" warning either).
7. **Report.** Say what changed (file list), what was proven, and the rollback steps. Say what you did not do (for example: production key not set).

## Rollback (every case)

1. Restore the files you changed (`git checkout -- <files>`, or undo your edits if the user had uncommitted work in them; check `git status` first).
2. Run `zoroval logout` (`npx zoroval logout` in TypeScript). It removes the lines `zoroval login` wrote and deletes the env file if login created it.
3. If the key will not be used again, tell the user to revoke it on the project's Keys page (the logout output names its key id).

## Snippets

Entry point (after any existing tracing setup, before the first LLM call; client creation order does not matter):

```python
import zoroval

zoroval.init()  # reads ZOROVAL_INGEST_KEY; wraps OpenAI and Anthropic; never raises
```

One chat turn (chat apps). The conversation id is the app's own chat or session id, never a per-request id:

```python
with zoroval.conversation(session_id, user_id=user_id):
    with zoroval.turn(input=text) as turn:
        ...existing handler body...
        turn.set_output(answer)
```

TypeScript. `init()` once at app start, wrap the client, and wrap one chat turn. The conversation id is the app's own chat or session id:

```ts
import * as zoroval from "zoroval";
import OpenAI from "openai";

zoroval.init(); // reads ZOROVAL_INGEST_KEY; never throws
const openai = zoroval.wrapOpenAI(new OpenAI());

export async function handleMessage(chatId: string, userId: string, text: string) {
  return zoroval.conversation(chatId, { userId }, () =>
    zoroval.turn({ input: text }, async (turn) => {
      // ...existing handler body, using `openai`...
      turn.setOutput(answer);
      return answer;
    }));
}
```

For plain `fetch` calls to OpenAI, use `const tracedFetch = zoroval.wrapFetch();` in place of `fetch`. For an OpenAI-compatible endpoint: `zoroval.wrapFetch(fetch, { baseURLs: ["https://llm.example.com/v1"] })`. Non-chat runs: `zoroval.trace("job-name", fn)`. Custom steps and tools: `zoroval.step(name, fn)` and `zoroval.tool(name, { arguments }, fn)`. Content off: `zoroval.init({ captureContent: false })`. Redaction: `zoroval.init({ redact: fn })`; if `fn` throws, that field is dropped.

Non-chat runs in Python (agents, batch jobs): `with zoroval.trace("job-name"):`. Custom steps and tools: `@zoroval.step` and `@zoroval.tool` (sync or async). Content off: `zoroval.init(capture_content=False)`. Redaction: `zoroval.init(redact=fn)`; if `fn` raises, that field is dropped, never sent raw. Both also apply to the standard prompt, output and document fields of other tools' spans (Langfuse, OpenInference, OTel GenAI), and GenAI log events are not sent when either is set.

## Cases

### 1. No tracing

Detect: `openai` or `anthropic` in dependencies, LLM calls in code, no tracing package.

Change (TypeScript): add `zoroval` to `package.json` (`npm install zoroval`); add `init()` at the app start and wrap the client with `wrapOpenAI` (or use `wrapFetch`); wrap the chat handler in `conversation()` and `turn()`; flush as in case 7. The chat id comes from the request, the session or the chat record (`chat_id_candidates`).

Change (Python): add `zoroval` to the dependency file; add the entry-point snippet to the module that creates the LLM client or the app entry; wrap the chat handler in `conversation(<chat id>)` and `turn()`. The chat id comes from the request, the session or the chat record (`chat_id_candidates`). If none is clear, ask.

Rollback: restore the files, `zoroval logout`.

### 2. Already on Langfuse (SDK v3 or later, OpenTelemetry based)

Detect: `langfuse` with major version 3 or more (`tracing.langfuse.major`), `@observe` or `langfuse.openai`, `LANGFUSE_*` names.

Change: add `zoroval` to dependencies and the entry-point snippet right after `Langfuse()` is created (or after the first Langfuse import if it is created lazily). Zoroval attaches to Langfuse's provider. Do not add another OpenAI wrapper: when `langfuse.openai` is in use, Zoroval reads Langfuse's spans and makes none of its own (no duplicates, in either install order). Langfuse's `session_id` (through `propagate_attributes(session_id=...)`) is the conversation id; add `zoroval.conversation()` only if the app sets no session id. Langfuse keeps working; no Langfuse key is read or copied.

Rollback: remove the two lines and the dependency, `zoroval logout`.

### 2b. Langfuse SDK v2 (not OpenTelemetry) - explain only

Detect: `langfuse` below version 3.

Do not edit. Explain the three choices and let the user pick: (a) upgrade Langfuse to v3 or later, then follow case 2; (b) add Zoroval beside it, but both wrap OpenAI, so the user must keep only one wrapper (Zoroval's or `langfuse.openai`); (c) export from Langfuse and upload the JSON in the Zoroval app.

### 3. OpenTelemetry only (own TracerProvider or a Collector)

Detect: `opentelemetry-sdk`, `TracerProvider(` or `set_tracer_provider` in code, `OTEL_EXPORTER_OTLP_*` names, a Collector config.

Change (default): the entry-point snippet right after `trace.set_tracer_provider(provider)`. Zoroval adds its own processor to that provider, exports only GenAI spans (never the app's HTTP or database spans), and adds the OpenAI and Anthropic wrappers. Wrap chat handlers in `conversation()`. Tell the user: Zoroval's LLM spans also go to their existing exporter, with prompt content; use `capture_content=False` or a filter on their side if that backend must not hold content.

Option (only with explicit consent, production config): a second OTLP exporter in the Collector pointing at `https://ingest.zoroval.com` with header `Authorization: Bearer ${env:ZOROVAL_INGEST_KEY}`. Show the YAML diff; do not apply it without a yes.

Rollback: remove the lines (or the exporter block), `zoroval logout`.

### 4. OpenInference or OpenLLMetry present (LangChain, LlamaIndex, others)

Detect: `openinference-instrumentation-*`, `traceloop-sdk`, or OpenLLMetry's `opentelemetry-instrumentation-<provider>` packages.

Change: the entry-point snippet after their instrumentor is set up. Keep their instrumentor. Zoroval does not wrap a client their instrumentor already covers, and reads their spans as they are. The conversation id: their session attribute (`using_session(id)` in OpenInference, `association_properties` in OpenLLMetry) or `zoroval.conversation(id)`; both work.

Rollback: remove the lines, `zoroval logout`.

### 5. LangSmith in use - explain only

Detect: `langsmith`, `@traceable`, `LANGSMITH_*` or `LANGCHAIN_TRACING*` names.

Do not edit unless the user asks. Explain: Zoroval runs beside LangSmith (case 1 changes, LangSmith untouched). For LangChain apps, add the OpenInference LangChain instrumentor so chain steps reach Zoroval.

### 6. Vercel AI SDK (TypeScript)

Detect: `package.json` with `ai` or `@vercel/otel`. A TypeScript app that only calls OpenAI directly is case 1, even if it has `@opentelemetry/*` packages.

Change: send OTLP to Zoroval from `instrumentation.ts`, with the key read from the environment:

```ts
import { OTLPHttpJsonTraceExporter, registerOTel } from "@vercel/otel";

export function register() {
  registerOTel({
    serviceName: "<service>",
    traceExporter: new OTLPHttpJsonTraceExporter({
      url: `${process.env.ZOROVAL_INGEST_URL ?? "https://ingest.zoroval.com"}/v1/traces`,
      headers: { Authorization: `Bearer ${process.env.ZOROVAL_INGEST_KEY}` },
    }),
  });
}
```

Turn on AI SDK telemetry per call, with the chat id as metadata: `experimental_telemetry: { isEnabled: true, functionId: "chat", metadata: { sessionId: chatId } }`. Write the key with `npx zoroval login` (Next.js apps get `.env.local`, which Next.js loads). Prove with `npx zoroval doctor`; no Python is needed.

Rollback: restore the two files, `npx zoroval logout`.

### 7. Serverless or pre-fork servers

Detect: Lambda handlers (`handler(event, context)`), `serverless.yml`, SAM or Vercel Python functions; gunicorn or uvicorn with workers, `--preload`, Celery.

Change:
- Serverless: the entry-point snippet at module level (once per cold start) and `zoroval.flush()` at the end of every handler, in a `finally`, because the container can freeze or stop right after it returns:

  ```python
  def handler(event, context):
      try:
          ...existing body...
      finally:
          zoroval.flush()
  ```
- TypeScript on Node servers (Express, Fastify, long-running processes): nothing extra; spans are sent in the background and when the process exits. In a short script, `await zoroval.flush()` before it ends.
- TypeScript on Cloudflare Workers: wrap the export, `export default zoroval.withZoroval({ async fetch(request, env, ctx) { ... } })`. It reads `ZOROVAL_INGEST_KEY` from the Worker's bindings and sends the trace after the response. Add `nodejs_als` to `compatibility_flags` in `wrangler.toml` or `wrangler.jsonc` (a deploy file: diff and yes first), or turns are not grouped into conversations; `detect` notes it and `doctor` warns. A Worker that only forwards requests to a container or another service needs neither.
- TypeScript on Next.js: `after(() => zoroval.flush());` (from `next/server`) at the start of each route that calls the model. Other Vercel functions: `waitUntil(zoroval.flush())` from `@vercel/functions`.
- Pre-fork servers (gunicorn, uvicorn workers, Celery prefork): nothing extra. The SDK restarts its export thread in each worker after fork, also with `--preload`. Do not add a `post_fork` hook.

Rollback: restore the files, `zoroval logout`.

### 8. Monorepo or several services - choose services only

Detect: more than one `pyproject.toml`, `requirements.txt` or `package.json`; compose or deploy files with several services.

Ask which services send traces. Then apply the matching case to each chosen service, one at a time, with the SDK of that service's language: `zoroval login --root <service folder>` (Python) or `npx zoroval login --root <service folder>` (TypeScript) per service (one key per service, all bound to the same project), and a distinct `service_name=` in each Python `init()` (`serviceName` in TypeScript). A repo with a Python service and a TypeScript service gets `pip install zoroval` in the first and `npm install zoroval` in the second.

### 9. Env files and secrets managers

Detect: `.env*` files (with tracked and ignored status), a dotenv loader, Dockerfile, `railway.json`, `vercel.json`, `fly.toml`, Kubernetes manifests, CI files, secrets-manager references.

Development: `zoroval login` writes the key to the git-ignored env file (it checks: inside the project, not a symlink, untracked, git-ignored; mode 600). In TypeScript the file is `.dev.vars` for a Worker, `.env.local` for Next.js and `.env` otherwise; pick another with `--file <path>`. If the app does not load that file, tell the user to run it with their dotenv loader (for example `dotenv run -- python app.py`, or `node --env-file=.env` for Node) or export the variable; do not add a loader without asking.

Production: write nothing. A production key comes from a second `zoroval login`, which the user runs themselves in their own terminal, with a name that says it is for production:

```bash
zoroval login --name "<service> production" --env-file .env.production --replace
```

In TypeScript, the same with `npx zoroval login --name "<service> production" --file .env.production --replace` (use `--file`: Node 20.6 or newer reads `--env-file` itself and stops when the file does not exist yet). Tell the user to approve it in the browser, copy the value from `.env.production` into the host's secret settings (`npx wrangler secret put ZOROVAL_INGEST_KEY` for a Worker; the host's environment variables, for example the Vercel project's settings, otherwise), then delete `.env.production`. The file must be git-ignored, or the command refuses. Never read that file or print its value. Do not copy the development key (`.env`, `.dev.vars` or `.env.local`) to production. To rotate or remove a key, use the Keys page, which lists and revokes keys.

Never put the key in Dockerfiles, CI files or committed config.

## What success looks like

- `zoroval doctor` prints only `ok` and `warn` lines, and "first trace received" is `ok` after the app ran.
- For chat apps: two messages in one chat give one trace with two turns, and doctor shows no ungrouped warning. If the app is not a chat app, say so and use `zoroval.trace()` (Python) or `zoroval.trace(name, fn)` (TypeScript).
- The user has the changed-file list and the rollback steps.
