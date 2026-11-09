"""init(), shutdown(), flush() and status(): provider setup and coexistence.

Coexistence (design section 3, PoC section 5):
- If an OpenTelemetry SDK TracerProvider is already global (the app's own, Langfuse's,
  OpenInference's), Zoroval adds its processor to it and creates no provider.
- If none is global, Zoroval creates one and makes it global, so instrumentations that start later
  (Langfuse creates its provider only when none is set) share it.
- If the app sets its own provider after Zoroval did, OpenTelemetry refuses the second one. Zoroval
  then moves the app's span processors onto the global provider, so the app's exporters keep
  receiving spans (other install order).
- Zoroval exports only GenAI spans; the app's HTTP and database spans never leave through us.
"""

from __future__ import annotations

import atexit
import os
import threading
from typing import Any, Callable, Optional, Sequence

from opentelemetry import trace as otel_trace

from . import _config, _state
from ._processor import ZorovalSpanProcessor
from ._transport import Transport
from ._util import COUNTERS, internal_error, log_once, logger
from ._version import SCOPE_NAME, __version__

_LOCK = threading.Lock()
_hook_installed = False
_original_set_provider: Optional[Callable[..., Any]] = None
_atexit_registered = False


def _sdk_provider_class() -> Any:
    from opentelemetry.sdk.trace import TracerProvider
    return TracerProvider


def _install_set_provider_hook() -> None:
    """Bridge a provider the app sets after us onto the global one (second install order)."""
    global _hook_installed, _original_set_provider
    if _hook_installed:
        return
    import opentelemetry.trace as trace_module

    original = trace_module._set_tracer_provider
    _original_set_provider = original

    def _set_tracer_provider(tracer_provider: Any, log: bool) -> None:
        state = _state.get()
        current = trace_module._TRACER_PROVIDER
        if (state is not None and state.owns_provider and current is state.provider
                and tracer_provider is not state.provider
                and isinstance(tracer_provider, _sdk_provider_class())):
            try:
                theirs = tracer_provider._active_span_processor
                state.provider.add_span_processor(theirs)
                logger.info("zoroval: an app TracerProvider was set after zoroval.init(); its span processors "
                            "were attached to the global provider so both receive spans.")
                return
            except Exception as error:  # noqa: BLE001
                internal_error("bridge_provider", error)
        original(tracer_provider, log)

    trace_module._set_tracer_provider = _set_tracer_provider
    _hook_installed = True


def _remove_set_provider_hook() -> None:
    global _hook_installed, _original_set_provider
    if not _hook_installed or _original_set_provider is None:
        return
    import opentelemetry.trace as trace_module
    trace_module._set_tracer_provider = _original_set_provider
    _hook_installed = False
    _original_set_provider = None


def _attach_logs(cfg: _config.Config, transport: Transport) -> Any:
    """The OTel contrib instrumentors put message content in GenAI log events. If the app has an SDK
    LoggerProvider, export those events (and only those) to /v1/logs."""
    try:
        from opentelemetry._logs import get_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    except ImportError:
        return None
    try:  # newer SDKs renamed the exporter base class
        from opentelemetry.sdk._logs.export import LogRecordExporter as LogExporter
        from opentelemetry.sdk._logs.export import LogRecordExportResult as LogExportResult
    except ImportError:
        from opentelemetry.sdk._logs.export import LogExporter, LogExportResult  # type: ignore[no-redef]
    provider = get_logger_provider()
    if not isinstance(provider, LoggerProvider):
        return None

    import time

    class _ZorovalLogExporter(LogExporter):  # type: ignore[misc]
        def export(self, batch: Sequence[Any]) -> Any:
            try:
                if not cfg.api_key:
                    return LogExportResult.FAILURE
                body = transport.compress(transport.encode_logs(batch))
                result = transport.send(cfg.logs_url, body, deadline=time.monotonic() + 30, kind="logs")
                if result.ok:
                    COUNTERS.add("logs_exported", len(batch) - result.rejected)
                    return LogExportResult.SUCCESS
                COUNTERS.add("logs_failed", len(batch))
            except Exception as error:  # noqa: BLE001
                internal_error("log_export", error)
            return LogExportResult.FAILURE

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    class _GenAIOnly(BatchLogRecordProcessor):  # type: ignore[misc]
        def on_emit(self, log_data: Any) -> None:
            try:
                record = getattr(log_data, "log_record", log_data)
                name = getattr(record, "event_name", None) or (getattr(record, "attributes", None) or {}).get(
                    "event.name") or ""
                if str(name).startswith("gen_ai."):
                    super().on_emit(log_data)
            except Exception as error:  # noqa: BLE001
                internal_error("log_emit", error)

        emit = on_emit

    processor = _GenAIOnly(_ZorovalLogExporter(), schedule_delay_millis=1000)
    provider.add_log_record_processor(processor)
    return processor


def init(api_key: Optional[str] = None, *, ingest_url: Optional[str] = None, service_name: Optional[str] = None,
         environment: Optional[str] = None, release: Optional[str] = None, capture_content: Optional[bool] = None,
         redact: Optional[Callable[[Any], Any]] = None, sample_rate: Optional[float] = None,
         instrument: Any = True, tracer_provider: Any = None, enabled: Optional[bool] = None,
         **options: Any) -> bool:
    """Start Zoroval tracing. Safe to call more than once; never raises.

    api_key: defaults to the ZOROVAL_INGEST_KEY environment variable.
    instrument: True (OpenAI and Anthropic when installed), False, or a list such as ["openai"].
    capture_content: send prompts and outputs (default True). redact: callable(value) -> value.
    sample_rate: head sampling, 0.0 to 1.0, kept whole per conversation (or per trace).
    Returns True when tracing is active.
    """
    try:
        return _init(api_key=api_key, ingest_url=ingest_url, service_name=service_name, environment=environment,
                     release=release, capture_content=capture_content, redact=redact, sample_rate=sample_rate,
                     instrument=instrument, tracer_provider=tracer_provider, enabled=enabled, **options)
    except TypeError:
        raise  # a wrong argument name is a coding error the developer must see
    except Exception as error:  # noqa: BLE001
        internal_error("init", error)
        return False


def _init(*, instrument: Any, tracer_provider: Any, **kwargs: Any) -> bool:
    global _atexit_registered
    with _LOCK:
        if _state.get() is not None:
            log_once("init-twice", "zoroval: init() was already called; the first configuration stays.")
            return True
        cfg = _config.resolve(**kwargs)
        if not cfg.enabled:
            return False
        if not cfg.api_key:
            log_once("no-key", "zoroval: no ingest key (set ZOROVAL_INGEST_KEY or run `zoroval login`). "
                     "Tracing is off.")
            return False
        if _config.key_id_of(cfg.api_key) is None:
            log_once("bad-key", "zoroval: ZOROVAL_INGEST_KEY does not look like a Zoroval ingest key "
                     "(zv_ingest_...). Tracing is off. Run `zoroval doctor`.")
            return False

        if not _config.ingest_url_ok(cfg.ingest_url):
            log_once("bad-url", "zoroval: the ingest URL must be https (http only for localhost). "
                     "Tracing is off so the key is not sent in clear text.")
            return False

        TracerProvider = _sdk_provider_class()
        provider = tracer_provider
        if provider is None:
            current = otel_trace.get_tracer_provider()
            if isinstance(current, TracerProvider):
                provider = current
        owns = provider is None
        if owns:
            from opentelemetry.sdk.resources import Resource
            attrs = {"service.name": cfg.service_name}
            if cfg.environment:
                attrs["deployment.environment.name"] = cfg.environment
            provider = TracerProvider(resource=Resource.create(attrs))
            otel_trace.set_tracer_provider(provider)
            if otel_trace.get_tracer_provider() is not provider:
                # Someone else won the race (or OTEL_PYTHON_TRACER_PROVIDER is set): attach to theirs.
                current = otel_trace.get_tracer_provider()
                if isinstance(current, TracerProvider):
                    provider, owns = current, False

        transport = Transport(cfg)
        processor = ZorovalSpanProcessor(cfg, transport, skip_content_when_sampled_out=owns)
        provider.add_span_processor(processor)
        tracer = provider.get_tracer(SCOPE_NAME, __version__)
        state = _state.State(cfg=cfg, tracer=tracer, provider=provider, owns_provider=owns,
                             processor=processor, transport=transport, pid=os.getpid())
        _state.set_state(state)
        if owns:
            _install_set_provider_hook()
        # Log events hold raw content and cannot be redacted here, so they stay off with either control.
        if cfg.export_logs and cfg.capture_content and cfg.redact is None:
            state.log_processor = _attach_logs(cfg, transport)
        _instrument(state, instrument)
        if not _atexit_registered:
            atexit.register(_atexit)
            _atexit_registered = True
        logger.debug("zoroval: tracing on (key id %s, provider %s)", _config.key_id_of(cfg.api_key),
                     "own" if owns else "existing")
        return True


def _instrument(state: _state.State, instrument: Any) -> None:
    if instrument is False or instrument is None:
        return
    names = ("openai", "anthropic") if instrument is True else tuple(instrument)
    for name in names:
        try:
            if name == "openai":
                from .integrations import openai as module
            elif name == "anthropic":
                from .integrations import anthropic as module  # type: ignore[no-redef]
            else:
                log_once(f"unknown-integration:{name}", "zoroval: unknown integration %r ignored.", name)
                continue
            state.instrumented[name] = module.instrument()
        except Exception as error:  # noqa: BLE001
            internal_error(f"instrument.{name}", error)


def instrument_openai() -> bool:
    from .integrations import openai as module
    return module.instrument()


def instrument_anthropic() -> bool:
    from .integrations import anthropic as module
    return module.instrument()


def flush(timeout: float = 5.0) -> bool:
    """Send everything queued so far. For serverless handlers and scripts. Never raises."""
    state = _state.get()
    if state is None:
        return True
    ok = state.processor.flush(timeout)
    if state.log_processor is not None:
        try:
            state.log_processor.force_flush(int(max(timeout, 0) * 1000))
        except Exception as error:  # noqa: BLE001
            internal_error("log_flush", error)
    return ok


def shutdown(timeout: Optional[float] = None) -> dict[str, Any]:
    """Flush and stop, within `timeout` seconds (default 5). Returns status(). Never raises."""
    state = _state.get()
    if state is None:
        return status()
    try:
        state.processor.shutdown(timeout)
        if state.log_processor is not None:
            # Logs get one try each: no waiting out a pause, a backoff or retry sleeps at exit.
            state.transport.abort.set()
            try:
                state.log_processor.shutdown()
            except Exception as error:  # noqa: BLE001
                internal_error("log_shutdown", error)
        from .integrations import anthropic as anthropic_module
        from .integrations import openai as openai_module
        openai_module.uninstrument()
        anthropic_module.uninstrument()
        if state.owns_provider:
            _remove_set_provider_hook()
    except Exception as error:  # noqa: BLE001
        internal_error("shutdown", error)
    result = status()
    _state.set_state(None)
    from ._context import PENDING
    PENDING.clear()
    return result


def _atexit() -> None:
    state = _state.get()
    if state is not None and state.pid == os.getpid():
        shutdown()


def status() -> dict[str, Any]:
    """Counters and health: sent, dropped, failed, queue size, last error. Never shows the key."""
    state = _state.get()
    out: dict[str, Any] = {"active": state is not None}
    out.update(COUNTERS.as_dict())
    if state is not None:
        out.update(state.processor.queue_state())
        out["key_id"] = _config.key_id_of(state.cfg.api_key)
        out["ingest_url"] = state.cfg.ingest_url
        out["provider"] = "own" if state.owns_provider else "existing"
        out["instrumented"] = dict(state.instrumented)
        out["last_error"] = state.transport.last_error
        out["key_refused"] = state.transport.fatal_reason
        out["export_state"] = state.transport.export_state()
        out["export_resumes_in_s"] = round(state.transport.hold_remaining(), 1)
        out["sample_rate"] = state.cfg.sample_rate
        out["capture_content"] = state.cfg.capture_content
    return out
