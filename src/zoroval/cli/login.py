"""`zoroval login`: the device flow (contracts 5.2). Writes the key to a git-ignored env file.

The key is never printed, logged or written to a tracked file. The device code is never written
to disk.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Optional

from .. import _config
from .._transport import USER_AGENT
from . import envfile

from .._transport import _default_opener

_OPEN = _default_opener()
SLOW_DOWN_STEP = 5.0     # RFC 8628: add 5 seconds after slow_down
MIN_INTERVAL = 0.05


def _post(url: str, payload: dict, timeout: float = 15.0) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={
        "Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT})
    try:
        with _OPEN(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read() or b"{}")
        except ValueError:
            return error.code, {}
    except Exception as error:  # noqa: BLE001
        return 0, {"status": f"connection:{type(error).__name__}"}


def _say(message: str) -> None:
    print(message, flush=True)


def run(*, root: Path, env_file: Path, name: Optional[str], app_url: str, no_browser: bool, replace: bool,
        allow_no_git: bool, timeout: float) -> int:
    env_path = env_file if env_file.is_absolute() else root / env_file
    # 1. Already set? Report presence only.
    if not replace:
        if os.environ.get(envfile.KEY_VAR):
            _say(f"{envfile.KEY_VAR} is already set in the environment. Nothing to do (use --replace for a new key).")
            return 0
        if env_path.exists() and envfile.KEY_VAR in envfile.read_names(env_path):
            _say(f"{envfile.KEY_VAR} is already in {env_path.name}. Nothing to do (use --replace for a new key).")
            return 0
    # 2. Check the file before asking anyone to approve anything.
    verdict = envfile.check(env_path, root, allow_no_git=allow_no_git)
    if not verdict.ok:
        _say(f"Cannot write the key to {env_path.name}: {verdict.reason}.")
        return 1
    # 3. Start the device flow.
    client_name = (name or f"{root.resolve().name} (agent)")[:120]
    status, body = _post(app_url.rstrip("/") + "/api/agent-connect", {"name": client_name})
    if status != 200 or "device_code" not in body:
        _say(f"Could not start sign-in at {app_url} (status {status or 'no connection'}).")
        return 1
    device_code = body["device_code"]
    interval = max(float(body.get("interval") or 3), MIN_INTERVAL)
    expires_at = time.monotonic() + min(float(body.get("expires_in") or 600), timeout)
    link = body.get("verification_uri_complete") or body.get("verification_uri")
    _say(f"Open this link and approve the request named \"{client_name}\":\n  {link}\n"
         f"Check that the page shows the code {body.get('user_code')}.")
    if not no_browser and sys.stdout.isatty() and not os.environ.get("CI"):
        try:
            webbrowser.open(link)
        except Exception:  # noqa: BLE001
            pass
    # 4. Poll.
    token_url = app_url.rstrip("/") + "/api/agent-connect/token"
    while True:
        if time.monotonic() >= expires_at:
            _say("The request expired before it was approved. Run `zoroval login` again.")
            return 1
        time.sleep(interval)
        status, body = _post(token_url, {"device_code": device_code})
        if status == 202:
            continue
        if status == 429:
            interval += SLOW_DOWN_STEP
            continue
        if status == 410:
            state = body.get("status", "expired")
            _say({"denied": "The request was denied in the app.",
                  "used": "This request was already used. Run `zoroval login` again.",
                  }.get(state, "The request expired. Run `zoroval login` again."))
            return 1
        if status == 200 and body.get("key"):
            break
        # 5xx, connection errors: wait and try again (design 4.1).
    key = str(body["key"])
    if _config.key_id_of(key) is None:
        _say("The server returned a key in an unknown format; nothing was written.")
        return 1
    ingest_url = body.get("ingest_url")
    write_url = ingest_url if ingest_url and ingest_url.rstrip("/") != _config.DEFAULT_INGEST_URL else None
    # 5. Re-check (the file could have changed while we waited), then write.
    verdict = envfile.check(env_path, root, allow_no_git=allow_no_git)
    if not verdict.ok:
        _say(f"Cannot write the key to {env_path.name}: {verdict.reason}. Nothing was written; "
             f"revoke key {body.get('key_id')} in the app.")
        return 1
    try:
        envfile.write_key(env_path, key, write_url, replace=replace)
    except FileExistsError as error:
        _say(f"{error}. Nothing was written; revoke key {body.get('key_id')} in the app.")
        return 1
    _say(f"Key written to {env_path.name} as {envfile.KEY_VAR} (key id {body.get('key_id')}, project "
         f"{body.get('project_key')}). The key is not shown. File mode is 600.")
    if write_url:
        _say(f"{envfile.URL_VAR} was written too ({write_url}).")
    _say("Next: load the env file in your app (for example python-dotenv) and run `zoroval doctor`.")
    return 0


def logout(*, root: Path, env_file: Path) -> int:
    env_path = env_file if env_file.is_absolute() else root / env_file
    key = envfile.read_value(env_path, envfile.KEY_VAR) if env_path.exists() else None
    removed = envfile.remove_key(env_path)
    if not removed:
        _say(f"No Zoroval lines in {env_path.name}.")
        return 0
    key_id = _config.key_id_of(key)
    _say(f"Removed {', '.join(removed)} from {env_path.name}." +
         (f" If key {key_id} is no longer used, revoke it in the app." if key_id else ""))
    return 0


def main(args: Any) -> int:
    root = Path(args.root or ".")
    app_url = args.app_url or os.environ.get("ZOROVAL_APP_URL") or _config.DEFAULT_APP_URL
    return run(root=root, env_file=Path(args.env_file), name=args.name, app_url=app_url,
               no_browser=args.no_browser, replace=args.replace, allow_no_git=args.allow_no_git,
               timeout=float(args.timeout))
