# Set up Zoroval (for coding agents)

You are helping a developer send their LLM app's traces to Zoroval. Follow these steps. The full procedure, with one section per starting point, is the setup skill: `skills/setup/SKILL.md` in the `zoroval` package repository.

## Rules

- Never print, log or paste the ingest key. Never write it into a tracked file. Never commit it.
- Read env var names only, never their values.
- Show a diff before every edit and wait for the developer's yes.
- Ask before touching deploy files, CI, hosting dashboards, secrets managers or Collector config.
- Do not commit. Keep a list of changed files for rollback.

## Steps

1. Install the package in the project's environment: `pip install zoroval` (or `uv add zoroval`, `poetry add zoroval`).
2. Run `zoroval detect --json`. It reports the starting point (`cases`), LLM calls, possible conversation ids and env file names. It never prints values.
3. Show the developer a short plan from the detect output and the skill. Ask which variable is the chat or conversation id if it is unclear.
4. Make the edits, each after a diff and a yes:
   - `import zoroval` and `zoroval.init()` at the app entry, after any existing tracing setup and after the env file is loaded.
   - For chat apps: `with zoroval.conversation(chat_id):` around each chat turn, so one chat becomes one trace.
   - For serverless handlers: `zoroval.flush()` in a `finally` at the end of each handler.
5. Run `zoroval login`. Ask the developer to open the printed link, sign in, check the code and approve. The key goes into the git-ignored `.env` with mode 600. If `.env` is not git-ignored, the command refuses: show the `.gitignore` diff, get a yes, run it again.
6. Run `zoroval doctor`. Run the app once (two messages in one chat), then `zoroval doctor --wait 60`. Every required check must pass, "first trace received" must pass, and there must be no ungrouped warning for a chat app.
7. Report the changed files, what doctor proved, and the rollback: restore the files and run `zoroval logout`.

## Production

Write nothing to production config. Tell the developer to run `zoroval login --name "<service> production" --env-file .env.production --replace` themselves, approve it in the browser, copy the value into their host's secret settings as `ZOROVAL_INGEST_KEY`, and delete that file. Do not copy the development key from `.env`.

## Without the Python SDK

Any OpenTelemetry exporter can send OTLP/HTTP (protobuf or JSON, zstd or gzip) to `https://ingest.zoroval.com/v1/traces` with the header `Authorization: Bearer <ingest key>`. Set `gen_ai.conversation.id` (or `session.id`) on spans to group a chat.
