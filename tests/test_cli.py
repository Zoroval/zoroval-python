"""CLI: login under the fal-style file checks, device-flow states, logout, doctor failures, detect
never printing values."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from zoroval.cli import doctor, envfile, login
from zoroval.cli.main import main


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(".env\n")
    return tmp_path


@pytest.fixture
def device(gateway):
    gateway.state.connect_interval = 0.05
    return gateway


def _login(repo: Path, gateway, **kwargs) -> int:
    args = dict(root=repo, env_file=Path(".env"), name=None, app_url=gateway.url, no_browser=True, replace=False,
                allow_no_git=False, timeout=20)
    args.update(kwargs)
    return login.run(**args)


def test_login_writes_key_once_and_never_prints_it(repo, device, capsys):
    device.state.auto_approve_after = 2
    assert _login(repo, device) == 0
    out = capsys.readouterr().out
    key = envfile.read_value(repo / ".env", "ZOROVAL_INGEST_KEY")
    assert key and key.startswith("zv_ingest_") and key not in out
    assert oct((repo / ".env").stat().st_mode & 0o777) == "0o600"
    assert "ZOROVAL_INGEST_URL" in envfile.read_names(repo / ".env")
    assert _login(repo, device) == 0  # second run: already set, nothing to do
    assert "already" in capsys.readouterr().out
    assert envfile.read_value(repo / ".env", "ZOROVAL_INGEST_KEY") == key


def test_login_keeps_existing_env_lines(repo, device):
    device.state.auto_approve_after = 1
    (repo / ".env").write_text("OPENAI_MODEL=gpt-test\n")
    assert _login(repo, device) == 0
    assert envfile.read_names(repo / ".env")[0] == "OPENAI_MODEL"
    assert login.logout(root=repo, env_file=Path(".env")) == 0
    assert (repo / ".env").read_text() == "OPENAI_MODEL=gpt-test\n"


def test_login_replace(repo, device):
    device.state.auto_approve_after = 1
    assert _login(repo, device) == 0
    first = envfile.read_value(repo / ".env", "ZOROVAL_INGEST_KEY")
    assert _login(repo, device, replace=True) == 0
    assert envfile.read_value(repo / ".env", "ZOROVAL_INGEST_KEY") != first
    assert envfile.read_names(repo / ".env").count("ZOROVAL_INGEST_KEY") == 1


def test_refuses_tracked_file_before_starting_flow(repo, device, capsys):
    (repo / ".gitignore").write_text("")
    (repo / ".env").write_text("A=1\n")
    subprocess.run(["git", "add", ".env"], cwd=repo, check=True)
    assert _login(repo, device) == 1
    assert "tracked by git" in capsys.readouterr().out
    assert device.state.connects == {}  # nobody was asked to approve anything


def test_refuses_not_ignored(repo, device, capsys):
    (repo / ".gitignore").write_text("")
    assert _login(repo, device) == 1
    assert "not git-ignored" in capsys.readouterr().out


def test_refuses_symlink(repo, device, tmp_path_factory, capsys):
    outside = tmp_path_factory.mktemp("outside") / "real.env"
    outside.write_text("")
    os.symlink(outside, repo / ".env")
    assert _login(repo, device) == 1
    assert "symlink" in capsys.readouterr().out
    assert outside.read_text() == ""


def test_refuses_outside_project(repo, device, tmp_path_factory, capsys):
    other = tmp_path_factory.mktemp("other")
    assert _login(repo, device, env_file=other / ".env") == 1
    assert "outside the project" in capsys.readouterr().out


def test_refuses_without_git_unless_allowed(tmp_path, device, capsys):
    device.state.auto_approve_after = 1
    assert _login(tmp_path, device) == 1
    assert "not inside a git repository" in capsys.readouterr().out
    assert _login(tmp_path, device, allow_no_git=True) == 0


def test_denied_and_slow_down(repo, device, capsys, monkeypatch):
    monkeypatch.setattr(login, "SLOW_DOWN_STEP", 0.05)
    device.state.connect_interval = 0.2
    monkeypatch.setattr(login, "MIN_INTERVAL", 0.01)
    original_post = login._post
    calls = {"n": 0}

    def fast_first_poll(url, payload, timeout=15.0):
        if url.endswith("/token") and calls["n"] == 0:
            calls["n"] += 1
            original_post(url, payload, timeout)          # first poll
            return original_post(url, payload, timeout)   # immediate second poll: slow_down
        return original_post(url, payload, timeout)

    monkeypatch.setattr(login, "_post", fast_first_poll)
    device.state.deny_after = 3
    assert _login(repo, device) == 1
    assert "denied" in capsys.readouterr().out
    assert not (repo / ".env").exists()


def test_connect_server_down(repo, capsys):
    assert _login(repo, type("G", (), {"url": "http://127.0.0.1:9"})()) == 1
    assert "Could not start sign-in" in capsys.readouterr().out


def test_logout_without_lines(repo, capsys):
    assert login.logout(root=repo, env_file=Path(".env")) == 0
    assert "No Zoroval lines" in capsys.readouterr().out


def _doctor(repo: Path, **kwargs):
    args = dict(root=repo, env_file=Path(".env"), ingest_url=None, timeout=10, wait=0)
    args.update(kwargs)
    return doctor.run(**args)


def test_doctor_without_key(repo):
    report = _doctor(repo)
    assert not report.ok and report.checks[-1]["check"] == "key"


def test_doctor_bad_key_format(repo, monkeypatch):
    monkeypatch.setenv("ZOROVAL_INGEST_KEY", "sk-some-other-vendor")
    report = _doctor(repo)
    assert not report.ok
    assert "sk-some-other-vendor" not in json.dumps(report.checks)


def test_doctor_revoked_key(repo, gateway, monkeypatch):
    key = gateway.issue_key()
    gateway.revoke(key)
    monkeypatch.setenv("ZOROVAL_INGEST_KEY", key)
    report = _doctor(repo, ingest_url=gateway.url)
    assert not report.ok
    assert any(c["check"] == "gateway" and "403" in c["detail"] for c in report.checks)
    assert key not in json.dumps(report.checks)


def test_doctor_waits_for_processing_batch(repo, gateway, monkeypatch):
    gateway.state.batch_polls_until_done = 2
    monkeypatch.setenv("ZOROVAL_INGEST_KEY", gateway.issue_key())
    report = _doctor(repo, ingest_url=gateway.url)
    assert report.ok
    levels = {c["check"]: c["level"] for c in report.checks}
    assert levels["grouping"] == "pass"
    assert levels["first trace received"] == "warn"  # test traces do not count as app traffic


def test_doctor_gateway_down(repo, monkeypatch):
    monkeypatch.setenv("ZOROVAL_INGEST_KEY", "zv_ingest_aaaaaaaaaaaa_" + "A" * 43)
    report = _doctor(repo, ingest_url="http://127.0.0.1:9")
    assert not report.ok


def test_doctor_warns_on_ungrouped_app_turns(repo, gateway, monkeypatch, start_sdk):
    key = start_sdk()
    import zoroval
    with zoroval.turn():
        pass
    zoroval.flush(5)
    monkeypatch.setenv("ZOROVAL_INGEST_KEY", key)
    report = _doctor(repo, ingest_url=gateway.url)
    levels = {c["check"]: c["level"] for c in report.checks}
    assert levels["first trace received"] == "pass"
    assert levels["grouping (app)"] == "warn"


def test_detect_reads_names_never_values(tmp_path, capsys):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=value-that-must-not-appear\nLANGFUSE_SECRET_KEY=also-hidden\n")
    (tmp_path / "requirements.txt").write_text("openai\n")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nOpenAI().chat.completions.create(model='m', "
                                     "messages=[])\n")
    assert main(["detect", "--json", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "value-that-must-not-appear" not in out and "also-hidden" not in out
    report = json.loads(out)
    assert report["llm_calls"][0]["api"] == "chat.completions.create"
    assert "LANGFUSE_SECRET_KEY" in report["env"]["vendor_vars"]
    assert main(["detect", str(tmp_path)]) == 0
    assert "value-that-must-not-appear" not in capsys.readouterr().out
