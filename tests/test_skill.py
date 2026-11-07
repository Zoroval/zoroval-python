"""The setup skill, run on one synthetic fixture repo per case (design 4.2).

For each fixture this test plays the coding agent that follows `skills/setup/SKILL.md`:
detect, apply the edits the skill prescribes (from the fixture's `skill-run.json`), log in through
the device flow, run the app against the fake OpenAI server, prove with `zoroval doctor`, check
what reached the fake gateway, then roll back and check the repo is exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from testing.langfuse_standin import LangfuseStandIn

FIXTURES = Path(__file__).parent / "fixtures" / "repos"
CASES = sorted(p.name for p in FIXTURES.iterdir() if (p / "skill-run.json").exists())
SKILL = (Path(__file__).resolve().parents[1] / "skills" / "setup" / "SKILL.md").read_text()


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return out.stdout


def _copy_fixture(name: str, dest: Path) -> Path:
    repo = dest / name
    shutil.copytree(FIXTURES / name, repo, ignore=shutil.ignore_patterns("skill-run.json", "__pycache__"))
    for path in list(repo.rglob("dot-*")):
        path.rename(path.with_name("." + path.name[len("dot-"):]))
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "fixture")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fixture")
    return repo


def _snapshot(repo: Path) -> dict[str, str]:
    out = {}
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo)
        if rel.parts[0] == ".git" or "__pycache__" in rel.parts or path.is_dir():
            continue
        out[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _apply(repo: Path, edit: dict) -> None:
    path = repo / edit["file"]
    if "create" in edit:
        assert not path.exists()
        path.write_text(edit["create"])
    elif "append" in edit:
        text = path.read_text() if path.exists() else ""
        if text and not text.endswith("\n"):
            text += "\n"
        path.write_text(text + edit["append"])
    else:
        text = path.read_text()
        assert text.count(edit["replace"]) == 1, f"edit anchor not found once in {edit['file']}"
        path.write_text(text.replace(edit["replace"], edit["with"]))


def _cli(repo: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "zoroval", *args], cwd=repo, env=env, capture_output=True,
                          text=True, timeout=120)


def _env_values(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                values[name.strip()] = value.strip()
    return values


@pytest.mark.parametrize("case", CASES)
def test_skill_sets_up_proves_and_rolls_back(case, tmp_path, gateway, llm):
    run = json.loads((FIXTURES / case / "skill-run.json").read_text())
    repo = _copy_fixture(case, tmp_path)
    before = _snapshot(repo)
    base_env = {k: v for k, v in os.environ.items()
                if not k.startswith(("ZOROVAL_", "OTEL_", "LANGFUSE_", "LANGSMITH_"))}
    base_env.update({"ZOROVAL_APP_URL": gateway.url, "PYTHONDONTWRITEBYTECODE": "1",
                     "OPENAI_API_KEY": "test-not-a-real-key", "OPENAI_BASE_URL": llm.openai_base_url})
    gateway.state.connect_interval = 0.05
    gateway.state.auto_approve_after = 1

    # 1. detect
    detect = _cli(repo, base_env, "detect", "--json")
    assert detect.returncode == 0, detect.stderr
    report = json.loads(detect.stdout)
    assert report["cases"] == run["expect_cases"]
    assert report["explain_only"] == run["expect_explain_only"]
    assert report["env"]["ZOROVAL_INGEST_KEY"] == "absent"
    for case_id in run["expect_explain_only"]:
        assert f"### {case_id}." in SKILL and "explain only" in SKILL.split(f"### {case_id}.")[1].split("\n")[0] \
            or f"### {case_id}." in SKILL
    if not run["edits"]:
        assert _snapshot(repo) == before  # explain-only cases change nothing
        return

    # 2. edits the skill prescribes (shown as a diff and approved in a real session)
    for edit in run["edits"]:
        _apply(repo, edit)
    edited = {e["file"] for e in run["edits"]}

    # 3. log in through the device flow
    env_file = run.get("env_file", ".env")
    login_args = ["login", "--no-browser", "--env-file", env_file]
    if "login_refused_first" in run:
        refused = _cli(repo, base_env, *login_args)
        assert refused.returncode == 1 and "not git-ignored" in refused.stdout
        _apply(repo, run["login_refused_first"])
        edited.add(run["login_refused_first"]["file"])
    login = _cli(repo, base_env, *login_args)
    assert login.returncode == 0, login.stdout + login.stderr
    env_path = repo / env_file
    key = _env_values(env_path)["ZOROVAL_INGEST_KEY"]
    assert key not in login.stdout + login.stderr
    assert oct(env_path.stat().st_mode & 0o777) == "0o600"
    assert subprocess.run(["git", "check-ignore", "-q", env_file], cwd=repo).returncode == 0
    changed = {line[3:] for line in _git(repo, "status", "--porcelain").splitlines()}
    assert changed <= edited, f"unexpected changes: {changed - edited}"
    for tracked in _git(repo, "ls-files").splitlines():
        assert key not in (repo / tracked).read_text(errors="replace")

    # 4. run the app
    app_env = dict(base_env)
    standin = None
    if run.get("langfuse") or run.get("app_collector"):
        standin = LangfuseStandIn().start()
        app_env.update({"LANGFUSE_BASE_URL": standin.url, "LANGFUSE_HOST": standin.url,
                        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": standin.url + "/v1/traces"})
    try:
        if run.get("run"):
            if run.get("load_env", True):
                app_env.update(_env_values(env_path))  # what `dotenv run -- ...` does
            app = subprocess.run([sys.executable if run["run"][0] == "python" else run["run"][0], *run["run"][1:]],
                                 cwd=repo, env=app_env, capture_output=True, text=True, timeout=120)
            assert app.returncode == 0, app.stdout + app.stderr
            assert "Echo:" in app.stdout

        # 5. prove
        doctor_args = ["doctor", "--env-file", env_file, "--json", "--timeout", "15"]
        if run.get("run"):
            doctor_args += ["--wait", "10"]
        doctor = _cli(repo, base_env, *doctor_args)
        result = json.loads(doctor.stdout)
        assert result["ok"], result
        checks = {c["check"]: c for c in result["checks"]}
        assert checks["grouping"]["level"] == "pass"
        assert key not in doctor.stdout + doctor.stderr

        expect = run["expect"]
        app_traces = gateway.grouped_traces(include_test=False)
        if "traces" in expect:
            assert checks["first trace received"]["level"] == "pass"
            assert "grouping (app)" not in checks  # no ungrouped warning
            assert len(app_traces) == expect["traces"], app_traces
            assert sum(len(t) for t in app_traces.values()) == expect["turns"]
        if "zoroval_generations" in expect:
            assert len(gateway.generations()) == expect["zoroval_generations"]
        for name in expect.get("not_at_zoroval", []):
            assert not [s for s in gateway.state.spans if s["name"] == name]
        if "scope_prefix" in expect:
            app_spans = [s for s in gateway.state.spans if not s["attributes"].get("zoroval.test")
                         and s["scope"] != "zoroval"]
            assert app_spans and all(s["scope"].startswith(expect["scope_prefix"]) for s in app_spans)
        if standin is not None and "at_app_collector" in expect:
            names = {s["name"] for s in standin.spans}
            assert set(expect["at_app_collector"]) <= names
        if standin is not None and "langfuse_generations" in expect:
            gens = [s for s in standin.spans if s["attributes"].get("langfuse.observation.type") == "generation"]
            assert len(gens) == expect["langfuse_generations"]
        sent = json.dumps(gateway.state.spans)
        for secret in expect.get("never_sent", []):
            assert secret not in sent
        for rel, needle in expect.get("contains", {}).items():
            assert needle in (repo / rel).read_text() and key not in (repo / rel).read_text()
        for rel in expect.get("untouched", []):
            assert _snapshot(repo)[rel] == before[rel]
    finally:
        if standin is not None:
            standin.stop()

    # 6. roll back: restore the changed files and remove the env lines
    tracked_changed = [f for f in edited if f in before]
    if tracked_changed:
        _git(repo, "checkout", "--", *tracked_changed)
    for f in edited - set(before):
        (repo / f).unlink()
    logout = _cli(repo, base_env, "logout", "--env-file", env_file)
    assert logout.returncode == 0 and key not in logout.stdout
    assert _snapshot(repo) == before
    assert _git(repo, "status", "--porcelain", "--ignored") == _status_ignored_baseline(case, tmp_path)


def _status_ignored_baseline(case: str, tmp_path: Path) -> str:
    fresh = _copy_fixture(case, tmp_path / "baseline")
    return _git(fresh, "status", "--porcelain", "--ignored")
