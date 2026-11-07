"""`zoroval detect`: what is in this repo, as JSON. Reads names, never secret values.

It reads dependency files, Python imports and calls (with `ast`), TypeScript imports (text
search), env file variable names, deploy and CI file names. It maps what it finds to the setup
skill's cases (design 4.2) and suggests a plan. It never prints an env value.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from . import envfile

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", "dist", "build", ".tox", ".mypy_cache",
             ".pytest_cache", "site-packages", ".next", ".idea", ".vscode", ".ruff_cache"}
MAX_FILES = 3000
MAX_BYTES = 1_000_000

CHAT_ID_NAMES = ("conversation_id", "session_id", "chat_id", "thread_id", "conversationId", "sessionId", "chatId",
                 "threadId", "dialog_id", "room_id")
LLM_CALLS = {
    "chat.completions.create": "openai", "chat.completions.parse": "openai", "chat.completions.stream": "openai",
    "responses.create": "openai", "responses.parse": "openai", "responses.stream": "openai",
    "messages.create": "anthropic", "messages.stream": "anthropic", "messages.parse": "anthropic",
}
DEPLOY_FILES = ("Dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml", "railway.json",
                "railway.toml", "vercel.json", "fly.toml", "render.yaml", "app.yaml", "Procfile", "netlify.toml",
                "serverless.yml", "serverless.yaml", "template.yaml", "samconfig.toml", "cdk.json", "Chart.yaml")
SECRETS_HINTS = ("aws secretsmanager", "secretsmanager", "vault ", "doppler", "1password", "op run", "infisical",
                 "gcloud secrets", "azure keyvault", "sops")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _walk(root: Path) -> Iterable[Path]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for filename in sorted(filenames):
            count += 1
            if count > MAX_FILES:
                return
            yield Path(dirpath) / filename


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------- manifests

_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([<>=!~]=?\s*[^;#\s]+)?")


def _parse_requirements(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "git+", "http")):
            continue
        match = _REQ_LINE.match(line)
        if match:
            out[_norm(match.group(1))] = (match.group(3) or "").replace(" ", "")
    return out


def _parse_pyproject(text: str) -> dict[str, str]:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        return _parse_requirements("\n".join(re.findall(r'"([A-Za-z0-9][^"]*)"', text)))
    try:
        data = tomllib.loads(text)
    except Exception:  # noqa: BLE001
        return {}
    deps: dict[str, str] = {}
    project = data.get("project") or {}
    for spec in project.get("dependencies") or []:
        deps.update(_parse_requirements(spec))
    for group in (project.get("optional-dependencies") or {}).values():
        for spec in group:
            deps.update(_parse_requirements(spec))
    for group in (data.get("dependency-groups") or {}).values():
        for spec in group:
            if isinstance(spec, str):
                deps.update(_parse_requirements(spec))
    poetry = (data.get("tool") or {}).get("poetry") or {}
    for table in [poetry.get("dependencies") or {}] + [g.get("dependencies") or {} for g in
                                                        (poetry.get("group") or {}).values()]:
        for name, spec in table.items():
            if name.lower() != "python":
                deps[_norm(name)] = spec if isinstance(spec, str) else str(spec.get("version", "")) \
                    if isinstance(spec, dict) else ""
    return deps


def _parse_lock(text: str) -> dict[str, str]:
    """uv.lock and poetry.lock: exact installed versions."""
    out = {}
    for name, version in re.findall(r'\[\[package\]\]\s*\nname\s*=\s*"([^"]+)"\s*\nversion\s*=\s*"([^"]+)"', text):
        out[_norm(name)] = "==" + version
    return out


def _parse_package_json(text: str) -> dict[str, str]:
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    out = {}
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        for name, spec in (data.get(key) or {}).items():
            out[name] = str(spec)
    return out


def _major(spec: str) -> Optional[int]:
    match = re.search(r"(\d+)", spec or "")
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------- python code scan

class _PyScan(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: set[str] = set()
        self.calls: list[tuple[int, str]] = []
        self.signals: list[tuple[int, str]] = []
        self.chat_ids: list[tuple[int, str]] = []
        self.has_main = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module)
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")

    def _dotted(self, node: ast.AST) -> str:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Call):
            parts.append(self._dotted(node.func) + "()")
        return ".".join(reversed(parts))

    def visit_Call(self, node: ast.Call) -> None:
        name = self._dotted(node.func)
        for suffix, provider in LLM_CALLS.items():
            if name.endswith(suffix):
                self.calls.append((node.lineno, f"{provider}:{suffix}"))
        last = name.rsplit(".", 1)[-1]
        if last in ("FastAPI", "Flask", "Starlette", "Quart", "Sanic", "Litestar"):
            self.signals.append((node.lineno, f"web_app:{last}"))
        elif last in ("TracerProvider",):
            self.signals.append((node.lineno, "otel_provider"))
        elif last == "set_tracer_provider":
            self.signals.append((node.lineno, "otel_set_provider"))
        elif last == "load_dotenv":
            self.signals.append((node.lineno, "dotenv"))
        elif name.endswith("Traceloop.init"):
            self.signals.append((node.lineno, "traceloop_init"))
        elif last == "register" and "phoenix" in name:
            self.signals.append((node.lineno, "phoenix_register"))
        elif last == "instrument" and "Instrumentor" in name:
            self.signals.append((node.lineno, f"instrumentor:{name}"))
        elif last in ("Langfuse", "get_client") and "langfuse" in name.lower():
            self.signals.append((node.lineno, "langfuse_client"))
        elif last == "Celery":
            self.signals.append((node.lineno, "celery"))
        for kw in node.keywords:
            if kw.arg in CHAT_ID_NAMES:
                self.chat_ids.append((node.lineno, kw.arg))
        self.generic_visit(node)

    def _function(self, node: Any) -> None:
        for deco in node.decorator_list:
            name = self._dotted(deco.func if isinstance(deco, ast.Call) else deco)
            last = name.rsplit(".", 1)[-1]
            if last == "observe":
                self.signals.append((node.lineno, "langfuse_observe"))
            elif last == "traceable":
                self.signals.append((node.lineno, "langsmith_traceable"))
            elif last in ("get", "post", "put", "route", "websocket", "api_route"):
                self.signals.append((node.lineno, f"route:{node.name}"))
        args = [a.arg for a in node.args.args + node.args.kwonlyargs]
        if node.name in ("lambda_handler", "handler") and args[:2] == ["event", "context"]:
            self.signals.append((node.lineno, "lambda_handler"))
        for arg in args:
            if arg in CHAT_ID_NAMES:
                self.chat_ids.append((node.lineno, arg))
        self.generic_visit(node)

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in CHAT_ID_NAMES:
            self.chat_ids.append((node.lineno, node.attr))
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        key = node.slice
        if isinstance(key, ast.Constant) and key.value in CHAT_ID_NAMES:
            self.chat_ids.append((node.lineno, str(key.value)))
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        test = node.test
        if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__":
            self.has_main = True
        self.generic_visit(node)


# ---------------------------------------------------------------------------- main scan

def scan(root: Path) -> dict[str, Any]:
    root = root.resolve()
    report: dict[str, Any] = {
        "root": str(root), "services": [], "packages": {}, "imports": {}, "llm_calls": [], "tracing": {},
        "entry_points": [], "chat_id_candidates": [], "servers": {}, "serverless": [], "env": {},
        "deploy_files": [], "ci_files": [], "collector_configs": [], "secrets_managers": [], "typescript": {},
        "cases": [], "explain_only": [], "notes": [], "plan": [],
    }
    py_packages: dict[str, str] = {}
    js_packages: dict[str, str] = {}
    env_files: list[Path] = []
    py_files: list[Path] = []
    ts_files: list[Path] = []
    texts_for_hints: list[str] = []

    for path in _walk(root):
        name = path.name
        rel = _rel(path, root)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_BYTES:
            continue
        if name in ("pyproject.toml", "setup.py", "setup.cfg", "Pipfile") or (
                name.startswith("requirements") and name.endswith((".txt", ".in"))):
            text = path.read_text(encoding="utf-8", errors="replace")
            deps = _parse_pyproject(text) if name == "pyproject.toml" else _parse_requirements(
                text if name.endswith((".txt", ".in")) else "\n".join(re.findall(r"['\"]([A-Za-z][^'\"]+)['\"]", text)))
            py_packages.update({k: v for k, v in deps.items() if v or k not in py_packages})
            if name in ("pyproject.toml", "setup.py") or name == "requirements.txt":
                service = str(Path(rel).parent)
                if not any(s["path"] == service for s in report["services"]):
                    report["services"].append({"path": service, "kind": "python", "manifest": rel})
        elif name in ("uv.lock", "poetry.lock"):
            for k, v in _parse_lock(path.read_text(encoding="utf-8", errors="replace")).items():
                py_packages[k] = v
        elif name == "package.json":
            js_packages.update(_parse_package_json(path.read_text(encoding="utf-8", errors="replace")))
            report["services"].append({"path": str(Path(rel).parent), "kind": "typescript", "manifest": rel})
        elif name == ".env" or name.startswith(".env.") or name.endswith(".env"):
            env_files.append(path)
        elif name.endswith(".py"):
            py_files.append(path)
        elif name.endswith((".ts", ".tsx", ".js", ".mjs")):
            ts_files.append(path)
        if name in DEPLOY_FILES or rel.startswith(("k8s/", "kubernetes/", "helm/")):
            report["deploy_files"].append(rel)
            texts_for_hints.append(path.read_text(encoding="utf-8", errors="replace"))
        if rel.startswith((".github/workflows/", ".gitlab-ci", ".circleci/")) or name in (".gitlab-ci.yml",
                                                                                          "Jenkinsfile"):
            report["ci_files"].append(rel)
            texts_for_hints.append(path.read_text(encoding="utf-8", errors="replace"))
        if re.match(r"(otel|otelcol|collector)[-_.]?.*\.(ya?ml)$", name) or name in ("otel-collector-config.yaml",):
            text = path.read_text(encoding="utf-8", errors="replace")
            if "receivers" in text and "exporters" in text:
                report["collector_configs"].append(rel)
        if name in ("gunicorn.conf.py", "gunicorn_config.py", "Procfile") or name.endswith(".sh"):
            texts_for_hints.append(path.read_text(encoding="utf-8", errors="replace"))

    report["packages"] = {"python": dict(sorted(py_packages.items())), "typescript": dict(sorted(js_packages.items()))}

    # python code
    imports: dict[str, list[str]] = {}
    for path in py_files:
        rel = _rel(path, root)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (SyntaxError, ValueError, OSError):
            continue
        visitor = _PyScan()
        visitor.visit(tree)
        for module in visitor.imports:
            top = module.split(".")[0]
            imports.setdefault(top, [])
            if rel not in imports[top]:
                imports[top].append(rel)
            if module.startswith(("langfuse.openai", "langfuse.decorators")) or module == "langfuse.openai":
                imports.setdefault("langfuse.openai" if "openai" in module else module, []).append(rel)
        for line, call in visitor.calls:
            provider, api = call.split(":", 1)
            report["llm_calls"].append({"file": rel, "line": line, "provider": provider, "api": api})
        for line, signal in visitor.signals:
            kind, _, detail = signal.partition(":")
            if kind in ("web_app", "lambda_handler"):
                report["entry_points"].append({"file": rel, "line": line, "kind": detail or kind})
            elif kind == "route":
                report.setdefault("routes", []).append({"file": rel, "line": line, "function": detail})
            else:
                report["tracing"].setdefault("code_signals", []).append({"file": rel, "line": line, "signal": signal})
        if visitor.has_main and not any(e["file"] == rel for e in report["entry_points"]):
            report["entry_points"].append({"file": rel, "line": None, "kind": "script"})
        seen = set()
        for line, name in visitor.chat_ids:
            if (line, name) not in seen:
                seen.add((line, name))
                report["chat_id_candidates"].append({"file": rel, "line": line, "name": name})
        texts_for_hints.append(source if "gunicorn" in source or "uvicorn" in source else "")
    report["imports"] = {k: sorted(set(v)) for k, v in sorted(imports.items())}

    # typescript
    ts_signals = []
    for path in ts_files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = _rel(path, root)
        if re.search(r"from\s+['\"]ai['\"]|require\(['\"]ai['\"]\)", text):
            ts_signals.append({"file": rel, "signal": "vercel_ai_sdk"})
        if "registerOTel" in text or "@vercel/otel" in text:
            ts_signals.append({"file": rel, "signal": "vercel_otel"})
        if "experimental_telemetry" in text:
            ts_signals.append({"file": rel, "signal": "ai_sdk_telemetry"})
        if "NodeSDK" in text or "@opentelemetry/" in text:
            ts_signals.append({"file": rel, "signal": "otel_js"})
        if rel.endswith(("instrumentation.ts", "instrumentation.js")):
            ts_signals.append({"file": rel, "signal": "next_instrumentation_hook"})
    report["typescript"] = {"signals": ts_signals}

    # servers and serverless
    hint_text = "\n".join(texts_for_hints)
    servers: dict[str, Any] = {}
    if "gunicorn" in hint_text or "gunicorn" in py_packages:
        servers["gunicorn"] = {"preload": "--preload" in hint_text or "preload_app" in hint_text,
                               "workers": bool(re.search(r"(-w|--workers)[ =]\d+|workers\s*=", hint_text))}
    if re.search(r"uvicorn[^\n]*--workers", hint_text):
        servers["uvicorn_workers"] = True
    if "celery" in py_packages or "celery" in imports:
        servers["celery"] = True
    report["servers"] = servers
    for entry in report["entry_points"]:
        if entry["kind"] == "lambda_handler":
            report["serverless"].append({"kind": "aws_lambda", "file": entry["file"]})
    for rel in report["deploy_files"]:
        if Path(rel).name.startswith(("serverless.", "template.yaml", "samconfig")):
            report["serverless"].append({"kind": "serverless_config", "file": rel})
        if Path(rel).name == "vercel.json" and py_files:
            report["serverless"].append({"kind": "vercel_python", "file": rel})
    report["secrets_managers"] = sorted({h.strip() for h in SECRETS_HINTS if h in hint_text.lower()})

    # env files: names only
    env_report = []
    for path in env_files:
        rel = _rel(path, root)
        info: dict[str, Any] = {"path": rel, "symlink": os.path.islink(path)}
        repo = envfile.git_root(path.parent)
        if repo is not None:
            info["tracked"] = envfile.is_tracked(path, repo)
            info["ignored"] = envfile.is_ignored(path, repo)
        else:
            info["tracked"] = None
            info["ignored"] = None
        try:
            info["names"] = envfile.read_names(path)
        except OSError:
            info["names"] = None
            info["inaccessible"] = True
        env_report.append(info)
    key_state = "present" if os.environ.get(envfile.KEY_VAR) else "absent"
    if key_state == "absent":
        for info in env_report:
            if info.get("inaccessible"):
                key_state = "inaccessible"
            elif envfile.KEY_VAR in (info.get("names") or []):
                key_state = f"present in {info['path']}"
                break
    names_in_files = {n for info in env_report for n in (info.get("names") or [])}
    report["env"] = {
        envfile.KEY_VAR: key_state,
        "env_files": env_report,
        "dotenv_loader": any(s.get("signal") == "dotenv" for s in report["tracing"].get("code_signals", []))
        or "python-dotenv" in py_packages,
        "vendor_vars": sorted(n for n in names_in_files | set(os.environ)
                              if n.startswith(("LANGFUSE_", "LANGSMITH_", "LANGCHAIN_", "OTEL_EXPORTER_OTLP",
                                               "TRACELOOP_", "PHOENIX_", "ARIZE_"))),
        "repo_is_git": envfile.git_root(root) is not None,
    }

    _classify(report, py_packages, js_packages, imports)
    return report


def _has(packages: dict[str, str], *names: str) -> bool:
    return any(_norm(n) in packages for n in names)


def _classify(report: dict[str, Any], py: dict[str, str], js: dict[str, str], imports: dict[str, list[str]]) -> None:
    cases: list[Any] = []
    explain: list[Any] = []
    notes: list[str] = report["notes"]
    tracing = report["tracing"]
    signals = {s["signal"].split(":")[0] for s in tracing.get("code_signals", [])}

    uses_openai = _has(py, "openai") or "openai" in imports
    uses_anthropic = _has(py, "anthropic") or "anthropic" in imports
    tracing["openai"] = uses_openai
    tracing["anthropic"] = uses_anthropic

    langfuse_spec = py.get("langfuse")
    if langfuse_spec is not None or "langfuse" in imports:
        major = _major(langfuse_spec or "")
        if major is None:
            try:
                from importlib.metadata import version
                major = _major(version("langfuse"))
            except Exception:  # noqa: BLE001
                major = None
        tracing["langfuse"] = {"spec": langfuse_spec, "major": major, "otel_based": None if major is None
                               else major >= 3, "openai_wrapper": "langfuse" in imports and any(
                                   k.startswith("langfuse.openai") for k in imports)}
        if major is None:
            notes.append("Langfuse version unknown: pin it or check `pip show langfuse` (v3+ is OTel based).")
            cases.append(2)
        elif major >= 3:
            cases.append(2)
        else:
            explain.append("2b")
    contrib = [p for p in py if p.startswith("opentelemetry-instrumentation-") and p.endswith("-v2")]
    openllmetry = [p for p in py if p == "traceloop-sdk" or (p.startswith("opentelemetry-instrumentation-") and
                                                           not p.endswith("-v2") and any(
                                                               x in p for x in ("openai", "anthropic", "langchain",
                                                                                "llamaindex", "bedrock")))]
    openinference = [p for p in py if p.startswith("openinference-")]
    if _has(py, "opentelemetry-sdk") or signals & {"otel_provider", "otel_set_provider"} or \
            report["collector_configs"] or any(n.startswith("OTEL_EXPORTER_OTLP") for n in
                                               report["env"].get("vendor_vars", [])):
        tracing["otel"] = {"sdk": _has(py, "opentelemetry-sdk"), "contrib_instrumentors": contrib,
                           "collector_configs": report["collector_configs"]}
        if 2 not in cases or signals & {"otel_provider"}:
            cases.append(3)
    if openinference or openllmetry:
        tracing["openinference"] = openinference
        tracing["openllmetry"] = openllmetry
        cases.append(4)
    if _has(py, "langsmith") or "langsmith_traceable" in signals or any(
            n.startswith(("LANGSMITH_", "LANGCHAIN_TRACING")) for n in report["env"].get("vendor_vars", [])):
        tracing["langsmith"] = True
        explain.append(5)
    if any(k in js for k in ("ai", "@vercel/otel")) or any(k.startswith("@opentelemetry/") for k in js):
        cases.append(6)
    if report["serverless"] or report["servers"]:
        cases.append(7)
    if len(report["services"]) > 1:
        explain.append(8)
    if (uses_openai or uses_anthropic) and not any(c in cases for c in (2, 3, 4)) and "2b" not in explain:
        cases.append(1)
    cases.append(9)
    report["cases"] = [c for c in [1, 2, 3, 4, 6, 7, 9] if c in cases]
    report["explain_only"] = [c for c in ["2b", 5, 8] if c in explain]
    if not (uses_openai or uses_anthropic) and 6 not in cases and 4 not in cases:
        notes.append("No OpenAI or Anthropic client found. Other providers need OTel GenAI instrumentation.")
    if report["chat_id_candidates"]:
        names = sorted({c["name"] for c in report["chat_id_candidates"]})
        if len(names) > 1:
            notes.append(f"Several possible conversation ids ({', '.join(names)}): ask the user which one.")
    elif uses_openai or uses_anthropic:
        notes.append("No conversation id found. Ask whether this is a chat app and where its chat id lives.")
    report["plan"] = _plan(report)


def _plan(report: dict[str, Any]) -> list[str]:
    cases = report["cases"]
    plan = []
    entry = report["entry_points"][0]["file"] if report["entry_points"] else None
    if any(s["kind"] == "python" for s in report["services"]):
        plan.append("Add `zoroval` to the Python dependencies.")
    if 2 in cases or 3 in cases or 4 in cases:
        plan.append("Add `import zoroval` and `zoroval.init()` after the existing tracing setup "
                    f"({'in ' + entry if entry else 'at the app entry'}); it attaches to the existing provider.")
    elif 1 in cases:
        plan.append(f"Add `import zoroval` and `zoroval.init()` at the top of {entry or 'the app entry'}.")
    if 4 in cases:
        plan.append("Keep their instrumentor; do not add a second OpenAI wrapper (zoroval detects theirs).")
    if report["chat_id_candidates"] and 2 not in cases:
        first = report["chat_id_candidates"][0]
        plan.append(f"Wrap each chat turn in `with zoroval.conversation({first['name']}):` "
                    f"(candidate at {first['file']}:{first['line']}).")
    if 7 in cases:
        if report["serverless"]:
            plan.append("Call `zoroval.flush()` at the end of each serverless handler.")
        if report["servers"].get("gunicorn"):
            plan.append("Pre-fork servers: nothing extra; the SDK restarts its export thread after fork.")
    if 6 in cases:
        plan.append("TypeScript: register an OTLP exporter to the Zoroval endpoint with the key header; "
                    "pass the chat id in experimental_telemetry metadata.")
    plan.append("Run `zoroval login` (writes the key to the git-ignored env file), then `zoroval doctor`.")
    if report["deploy_files"]:
        plan.append("Production: make a deploy key in the app and set ZOROVAL_INGEST_KEY in the host's "
                    "secret settings yourself; the skill writes nothing there.")
    return plan


def main(args: Any) -> int:
    root = Path(args.path or ".")
    if not root.is_dir():
        print(f"zoroval detect: {root} is not a folder", file=sys.stderr)
        return 2
    report = scan(root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=False))
        return 0
    print(f"Folder: {report['root']}")
    print(f"Cases: {', '.join(str(c) for c in report['cases']) or 'none'}"
          + (f"; explain only: {', '.join(str(c) for c in report['explain_only'])}" if report["explain_only"] else ""))
    print(f"ZOROVAL_INGEST_KEY: {report['env'][envfile.KEY_VAR]}")
    for call in report["llm_calls"][:10]:
        print(f"LLM call: {call['file']}:{call['line']} {call['provider']} {call['api']}")
    for cand in report["chat_id_candidates"][:10]:
        print(f"Conversation id candidate: {cand['file']}:{cand['line']} {cand['name']}")
    for note in report["notes"]:
        print(f"Note: {note}")
    print("Plan:")
    for step in report["plan"]:
        print(f"- {step}")
    return 0
