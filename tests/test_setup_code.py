import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from zoroval.cli import login, setup_report
from zoroval.cli.main import main

KEY = "zv_ingest_abcdefghijkl_" + "x" * 30


class Server:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                outer.calls.append((self.path, body))
                status, payload = outer.routes.get(self.path, (404, {}))
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def serve():
    servers = []

    def make(routes):
        servers.append(Server(routes))
        return servers[-1]

    yield make
    for server in servers:
        server.close()


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    setup_report._sent.clear()
    for name in ("ZOROVAL_SETUP", "ZOROVAL_APP_URL", "ZOROVAL_INGEST_KEY", "CI"):
        monkeypatch.delenv(name, raising=False)


def paths(server):
    return [p for p, _ in server.calls]


def test_reporter_does_nothing_without_env(serve, monkeypatch):
    server = serve({})
    monkeypatch.setenv("ZOROVAL_APP_URL", server.url)
    setup_report.started()
    setup_report.report("detected", {"apps": ["openai"]})
    assert server.calls == []


@pytest.mark.parametrize("status", [404, 410, 429, 500])
def test_reporter_is_silent_on_errors(serve, monkeypatch, capsys, status):
    server = serve({"/api/agent-setup/progress": (status, {"message": "nope"})})
    monkeypatch.setenv("ZOROVAL_APP_URL", server.url)
    monkeypatch.setenv("ZOROVAL_SETUP", "zvs_tree_secret")
    setup_report.started()
    setup_report.report("doctor_started")
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_reporter_never_raises_without_network(monkeypatch):
    monkeypatch.setenv("ZOROVAL_APP_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ZOROVAL_SETUP", "zvs_tree_secret")
    setup_report.started()
    setup_report.report("detected", {"apps": []})


def test_started_is_sent_once_per_process(serve, monkeypatch):
    server = serve({"/api/agent-setup/progress": (200, {})})
    monkeypatch.setenv("ZOROVAL_APP_URL", server.url)
    monkeypatch.setenv("ZOROVAL_SETUP", "zvs_tree_secret")
    setup_report.started()
    setup_report.started()
    setup_report.doctor_started()
    events = [b["event"] for _, b in server.calls]
    assert events == ["started", "doctor_started"]
    assert all(b["code"] == "zvs_tree_secret" for _, b in server.calls)


def test_detect_sends_names_only(serve, monkeypatch, tmp_path):
    server = serve({"/api/agent-setup/progress": (200, {})})
    monkeypatch.setenv("ZOROVAL_APP_URL", server.url)
    monkeypatch.setenv("ZOROVAL_SETUP", "zvs_tree_secret")
    (tmp_path / "requirements.txt").write_text("openai>=1.0\n")
    (tmp_path / "app.py").write_text("import openai\nclient = openai.OpenAI()\n")
    assert main(["detect", str(tmp_path), "--json"]) == 0
    detected = [b for _, b in server.calls if b["event"] == "detected"]
    assert detected and detected[0]["detail"]["apps"] == ["openai"]
    assert str(tmp_path) not in json.dumps(server.calls)


def approved_routes(**extra):
    connect = {"device_code": "dc1", "interval": 3, "expires_in": 600, "approved": True, "handle": "#ZVtree"}
    connect.update(extra)
    token = {"key": KEY, "key_id": "abc123", "project_key": "proj_1", "project_name": "Shop bot"}
    return {"/api/agent-connect": (200, connect), "/api/agent-connect/token": (200, token),
            "/api/agent-setup/progress": (200, {})}


def run_login(server, tmp_path: Path):
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / ".gitignore").write_text(".env\n")
    return login.run(root=tmp_path, env_file=Path(".env"), name=None, app_url=server.url, no_browser=False,
                     replace=False, allow_no_git=True, timeout=30)


def test_login_with_approved_code(serve, monkeypatch, tmp_path, capsys):
    server = serve(approved_routes())
    monkeypatch.setenv("ZOROVAL_SETUP", "zvs_tree_secret")
    opened = []
    monkeypatch.setattr(login.webbrowser, "open", lambda url: opened.append(url))
    assert run_login(server, tmp_path) == 0
    out = capsys.readouterr().out
    assert "#ZVtree connected to Shop bot" in out
    assert "Open this link" not in out and opened == []
    assert KEY not in out and KEY in (tmp_path / ".env").read_text()
    connect_body = dict(server.calls)["/api/agent-connect"]
    assert connect_body["setup_code"] == "zvs_tree_secret"
    assert paths(server).count("/api/agent-connect/token") == 1


@pytest.mark.parametrize("status", [400, 403, 404, 409, 410])
def test_login_with_rejected_code(serve, monkeypatch, tmp_path, capsys, status):
    server = serve({"/api/agent-connect": (status, {"message": "This setup code has expired."}),
                    "/api/agent-setup/progress": (200, {})})
    monkeypatch.setenv("ZOROVAL_SETUP", "zvs_tree_secret")
    assert run_login(server, tmp_path) == 1
    out = capsys.readouterr().out
    assert "This setup code has expired." in out and "Open this link" not in out
    assert "/api/agent-connect/token" not in paths(server)
    assert not (tmp_path / ".env").exists()


def test_login_without_code_is_unchanged(serve, monkeypatch, tmp_path, capsys):
    routes = approved_routes(approved=False, verification_uri_complete="https://x.test/c", user_code="AB-CD")
    server = serve(routes)
    monkeypatch.setattr(login.time, "sleep", lambda s: None)
    monkeypatch.setattr(login.webbrowser, "open", lambda url: None)
    assert run_login(server, tmp_path) == 0
    out = capsys.readouterr().out
    assert "Open this link" in out and "connected to" not in out
    assert "setup_code" not in dict(server.calls)["/api/agent-connect"]
    assert "/api/agent-setup/progress" not in paths(server)
