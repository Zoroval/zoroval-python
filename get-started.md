# Set up Zoroval (for coding agents)

You are helping a developer send their LLM app's traces to Zoroval. Follow these steps. The full procedure, with one section per starting point, is the setup skill: `skills/setup/SKILL.md` in the `zoroval` package repository.

## Rules

- Never print, log or paste the ingest key. Never write it into a tracked file. Never commit it.
- Read env var names only, never their values.
- Show a diff before every edit and wait for the developer's yes.
- Ask before touching deploy files, CI, hosting dashboards, secrets managers or Collector config.
- Do not commit. Keep a list of changed files for rollback.
- If the prompt gave a `ZOROVAL_SETUP` value, keep it set for every `zoroval` command you run (`npx zoroval` too).

## Which SDK

Use the SDK for the language of the code that calls the model. Do not choose by the other files in the repo: a Python app behind a Cloudflare Worker that only forwards requests is a Python service.

- Python code calls OpenAI or Anthropic: follow **Steps (Python)**.
- TypeScript or JavaScript code calls OpenAI (the `openai` package or `fetch`): follow **Steps (TypeScript)**. The npm package wraps OpenAI.
- The Vercel AI SDK (`ai`), or TypeScript or JavaScript code that calls another provider (for example Anthropic): use case 6 of the setup skill, the OpenTelemetry route. It needs no SDK.
- Both languages: set up each service with its own SDK, one after the other. Do not install Python for a TypeScript service, or Node packages for a Python one.

## Steps (Python)

1. Install the package in the project's environment: `pip install zoroval` (or `uv add zoroval`, `poetry add zoroval`).
2. Run `zoroval detect --json`. It reports the starting point (`cases`), LLM calls, possible conversation ids and env file names. It never prints values.
3. Show the developer a short plan from the detect output and the skill. Ask which variable is the chat or conversation id if it is unclear.
4. Make the edits, each after a diff and a yes:
   - `import zoroval` and `zoroval.init()` at the app entry, after any existing tracing setup and after the env file is loaded. If the app does not already load `.env` (for example with python-dotenv), add that so `ZOROVAL_INGEST_KEY` is set at startup.
   - For chat apps: `with zoroval.conversation(chat_id):` around each chat turn, so one chat becomes one trace.
   - For serverless handlers: `zoroval.flush()` in a `finally` at the end of each handler.
5. Run `zoroval login`. With a `ZOROVAL_SETUP` value it connects on its own. Without one, ask the developer to open the printed link, sign in, check the code and approve. The key goes into the git-ignored `.env` with mode 600. If `.env` is not git-ignored, the command refuses: show the `.gitignore` diff, get a yes, run it again.
6. Run `zoroval doctor`. Run the app once (two messages in one chat), then `zoroval doctor --wait 60`. Every required check must pass, "first trace received" must pass, and there must be no ungrouped warning for a chat app.
7. Report the changed files, what doctor proved, and the rollback: restore the files and run `zoroval logout`.

## Steps (TypeScript)

1. Install the package with the project's package manager: `npm install zoroval` (or `pnpm add zoroval`, `yarn add zoroval`, `bun add zoroval`).
2. Run `npx zoroval detect --json`. It reports the starting point (`cases`), LLM calls, possible conversation ids, env file names and a `plan`. It never prints values.
3. Show the developer a short plan from the detect output and the skill. Ask which variable is the chat or conversation id if it is unclear.
4. Make the edits, each after a diff and a yes:
   - `import * as zoroval from "zoroval"` and `zoroval.init()` once at app start, before the first model call. Next.js loads `.env.local` itself. For other Node apps, make sure `.env` is loaded at startup (`node --env-file=.env` on Node 20.6 or newer, or dotenv); ask before changing a start script.
   - Wrap the client: `const openai = zoroval.wrapOpenAI(new OpenAI())`. For plain `fetch` calls to OpenAI, use `zoroval.wrapFetch()` as the `fetch`.
   - For chat apps: `zoroval.conversation(chatId, () => zoroval.turn({ input }, async (turn) => { ...; turn.setOutput(answer); }))` around each chat turn, so one chat becomes one trace.
   - Send spans before the runtime stops. Long-running Node servers need nothing. Cloudflare Workers: `export default zoroval.withZoroval({ fetch })` (no `init()` needed), and add `nodejs_als` to `compatibility_flags` in `wrangler.toml` or `wrangler.jsonc`, or turns are not grouped into chats (this is a deploy file: show the diff first). Next.js routes: `after(() => zoroval.flush())`. Other Vercel functions: `waitUntil(zoroval.flush())` from `@vercel/functions`.
5. Run `npx zoroval login`. With a `ZOROVAL_SETUP` value it connects on its own. Without one, ask the developer to open the printed link, sign in, check the code and approve. The key goes into a git-ignored file with mode 600: `.dev.vars` for a Worker, `.env.local` for Next.js, `.env` otherwise. If that file is not git-ignored, the command refuses: show the `.gitignore` diff, get a yes, run it again.
6. Run `npx zoroval doctor`. Run the app once (`wrangler dev`, `next dev` or your start command; two messages in one chat), then `npx zoroval doctor --wait 60`. Every required check must pass, "first trace received" must pass, and there must be no ungrouped or worker context warning for a chat app.
7. Report the changed files, what doctor proved, and the rollback: restore the files and run `npx zoroval logout`.

## Production (Python)

Write nothing to production config. Tell the developer to run `zoroval login --name "<service> production" --env-file .env.production --replace` themselves, approve it in the browser, copy the value into their host's secret settings as `ZOROVAL_INGEST_KEY`, and delete that file. Do not copy the development key from `.env`.

## Production (TypeScript)

Write nothing to production config. Tell the developer to run `npx zoroval login --name "<service> production" --file .env.production --replace` themselves and approve it in the browser. Then they add the value as `ZOROVAL_INGEST_KEY` in production themselves (`npx wrangler secret put ZOROVAL_INGEST_KEY` for a Worker, or the host's environment variables, for example the Vercel project's settings) and delete that file. Do not copy the development key from `.dev.vars`, `.env.local` or `.env`.

## Without the SDK

Any other language or tool: any OpenTelemetry exporter can send OTLP/HTTP (protobuf or JSON, zstd or gzip) to `https://ingest.zoroval.com/v1/traces` with the header `Authorization: Bearer <ingest key>`. Set `gen_ai.conversation.id` (or `session.id`) on spans to group a chat.
