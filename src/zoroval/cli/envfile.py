"""Writing the ingest key into an env file, under fal-style checks (design 4.1 step 5).

The file must be inside the project, not a symlink, not tracked by git, and git-ignored. The
key is written with mode 600 and never printed.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

KEY_VAR = "ZOROVAL_INGEST_KEY"
URL_VAR = "ZOROVAL_INGEST_URL"
MANAGED = (KEY_VAR, URL_VAR)
MARKER = "# Added by `zoroval login`. Remove with `zoroval logout`."


@dataclass
class Check:
    ok: bool
    reason: str = ""
    git_root: Optional[str] = None


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=15)


def git_root(path: Path) -> Optional[Path]:
    try:
        out = _git(["rev-parse", "--show-toplevel"], path)
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None


def is_tracked(path: Path, repo: Path) -> bool:
    out = _git(["ls-files", "--error-unmatch", "--", str(path)], repo)
    return out.returncode == 0


def is_ignored(path: Path, repo: Path) -> bool:
    out = _git(["check-ignore", "-q", "--", str(path)], repo)
    return out.returncode == 0


def check(path: Path, root: Path, *, allow_no_git: bool = False) -> Check:
    """Can the key go into `path`? Never reads or prints values."""
    root = root.resolve()
    absolute = path if path.is_absolute() else (root / path)
    parent = absolute.parent
    if not parent.exists():
        return Check(False, f"folder {parent} does not exist")
    real_parent = parent.resolve()
    try:
        real_parent.relative_to(root)
    except ValueError:
        return Check(False, f"{absolute} is outside the project folder {root}")
    target = real_parent / absolute.name
    if os.path.islink(absolute) or os.path.islink(target):
        return Check(False, f"{absolute.name} is a symlink; write the key to a regular file")
    if target.exists() and not target.is_file():
        return Check(False, f"{absolute.name} is not a regular file")
    repo = git_root(real_parent)
    if repo is None:
        if allow_no_git:
            return Check(True, "not a git repository (allowed by --allow-no-git)")
        return Check(False, "not inside a git repository, so the file cannot be shown to be git-ignored; "
                     "use --allow-no-git if this folder is never committed")
    if is_tracked(target, repo):
        return Check(False, f"{absolute.name} is tracked by git; a key must never be committed")
    if not is_ignored(target, repo):
        return Check(False, f"{absolute.name} is not git-ignored; add it to .gitignore first")
    return Check(True, "untracked, git-ignored, not a symlink", str(repo))


def read_names(path: Path) -> list[str]:
    """Variable names in an env file. Values are never returned."""
    names = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return names
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        if name:
            names.append(name)
    return names


def read_value(path: Path, name: str) -> Optional[str]:
    """One value, for the SDK and doctor to use. Callers must never print it."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[len("export "):]
        if stripped.startswith(name + "="):
            value = stripped.split("=", 1)[1].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value
    return None


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".zoroval-tmp-", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)


def write_key(path: Path, key: str, ingest_url: Optional[str], *, replace: bool = False) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = existing.splitlines()
    present = [n for n in read_names(path) if n in MANAGED] if path.exists() else []
    if present and not replace:
        raise FileExistsError(f"{path.name} already sets {', '.join(present)}; use --replace")
    kept = [ln for ln in lines if ln.strip() != MARKER and not any(
        ln.strip().removeprefix("export ").startswith(name + "=") for name in MANAGED)]
    out = "\n".join(kept)
    if out and not out.endswith("\n"):
        out += "\n"
    out += MARKER + "\n" + f"{KEY_VAR}={key}\n"
    if ingest_url:
        out += f"{URL_VAR}={ingest_url}\n"
    _write_atomic(path, out)


def remove_key(path: Path) -> list[str]:
    """Remove what `zoroval login` wrote. Deletes the file if nothing else is left in it."""
    if not path.exists() or os.path.islink(path):
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    removed = []
    kept = []
    for line in lines:
        bare = line.strip().removeprefix("export ")
        name = next((n for n in MANAGED if bare.startswith(n + "=")), None)
        if name:
            removed.append(name)
        elif line.strip() != MARKER:
            kept.append(line)
    if not removed and len(kept) == len(lines):
        return []
    if not any(line.strip() for line in kept):
        path.unlink()
    else:
        _write_atomic(path, "\n".join(kept) + "\n")
    return removed
