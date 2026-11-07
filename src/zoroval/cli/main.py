"""The `zoroval` command: detect, login, logout, doctor."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .._version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zoroval", description="Set up and check Zoroval tracing.")
    parser.add_argument("--version", action="version", version=f"zoroval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="report what this repo uses (JSON with --json); never prints secrets")
    detect.add_argument("path", nargs="?", default=".")
    detect.add_argument("--json", action="store_true")

    login = sub.add_parser("login", help="sign in through the browser and write the ingest key to an env file")
    login.add_argument("--root", default=".", help="project folder (default: current folder)")
    login.add_argument("--env-file", default=".env", help="git-ignored env file for the key (default: .env)")
    login.add_argument("--name", help="name shown on the approve page (default: '<folder> (agent)')")
    login.add_argument("--app-url", help="default: ZOROVAL_APP_URL or https://app.zoroval.com")
    login.add_argument("--no-browser", action="store_true", help="print the link only")
    login.add_argument("--replace", action="store_true", help="replace a key that is already set")
    login.add_argument("--allow-no-git", action="store_true", help="allow a folder that is not a git repository")
    login.add_argument("--timeout", default="600", help="seconds to wait for approval (default 600)")

    logout = sub.add_parser("logout", help="remove the lines `zoroval login` wrote (does not revoke the key)")
    logout.add_argument("--root", default=".")
    logout.add_argument("--env-file", default=".env")

    doctor = sub.add_parser("doctor", help="send a test trace and check the setup")
    doctor.add_argument("--root", default=".")
    doctor.add_argument("--env-file", default=".env")
    doctor.add_argument("--ingest-url", help="default: ZOROVAL_INGEST_URL or https://ingest.zoroval.com")
    doctor.add_argument("--timeout", default="30", help="seconds to wait for the test batch (default 30)")
    doctor.add_argument("--wait", default="0", help="seconds to wait for the first app trace (default 0)")
    doctor.add_argument("--no-test-trace", action="store_true", help="only read status")
    doctor.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "detect":
        from . import detect
        return detect.main(args)
    if args.command == "login":
        from . import login
        return login.main(args)
    if args.command == "logout":
        from pathlib import Path

        from . import login
        return login.logout(root=Path(args.root), env_file=Path(args.env_file))
    if args.command == "doctor":
        from . import doctor
        return doctor.main(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
