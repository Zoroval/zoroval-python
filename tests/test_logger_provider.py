"""init() creates a LoggerProvider when none exists, so contrib instrumentors that put content in log
events (botocore, google-genai) are exported; an app that sets its own LoggerProvider, before or after
init(), keeps its exporters. Each case runs in a fresh process: OpenTelemetry sets the global once."""

import os
import subprocess
import sys
import textwrap

PRELUDE = """
import zoroval
from zoroval import _state
from opentelemetry._logs import get_logger_provider, set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
try:
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter as Mem
except ImportError:
    from opentelemetry.sdk._logs.export import InMemoryLogExporter as Mem
KEY = "zv_ingest_abcdefghijkl_" + "x" * 30
def init(**kw):
    return zoroval.init(api_key=KEY, ingest_url="http://127.0.0.1:9", instrument=False, **kw)
def processors(provider):
    return list(provider._multi_log_record_processor._log_record_processors)
"""


def run(body):
    code = PRELUDE + textwrap.dedent(body)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ZOROVAL_", "OTEL_"))}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr


def test_creates_provider_when_none():
    run("""
    assert init()
    provider = get_logger_provider()
    assert isinstance(provider, LoggerProvider)
    assert _state.get().log_processor in processors(provider)
    zoroval.shutdown(0.5)
    print("OK")
    """)


def test_app_provider_set_after_init_keeps_its_exporter():
    run("""
    assert init()
    ours = get_logger_provider()
    mem = Mem()
    theirs = LoggerProvider()
    app_processor = SimpleLogRecordProcessor(mem)
    theirs.add_log_record_processor(app_processor)
    set_logger_provider(theirs)
    assert get_logger_provider() is ours
    assert theirs._multi_log_record_processor in processors(ours)
    zoroval.shutdown(0.5)
    print("OK")
    """)


def test_app_provider_set_before_init_is_used():
    run("""
    theirs = LoggerProvider()
    set_logger_provider(theirs)
    assert init()
    assert get_logger_provider() is theirs
    assert _state.get().log_processor in processors(theirs)
    zoroval.shutdown(0.5)
    print("OK")
    """)


def test_no_provider_when_redact_is_set_or_no_key():
    run("""
    assert not zoroval.init(api_key=None, instrument=False)
    assert not isinstance(get_logger_provider(), LoggerProvider)
    assert init(redact="basic")
    assert not isinstance(get_logger_provider(), LoggerProvider)
    zoroval.shutdown(0.5)
    print("OK")
    """)


def test_init_again_after_shutdown():
    run("""
    assert init()
    ours = get_logger_provider()
    zoroval.shutdown(0.5)
    assert init()
    assert get_logger_provider() is ours
    assert _state.get().log_processor in processors(ours)
    theirs = LoggerProvider()
    set_logger_provider(theirs)
    assert theirs._multi_log_record_processor in processors(ours)
    zoroval.shutdown(0.5)
    print("OK")
    """)
