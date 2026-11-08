"""Configuration: explicit arguments first, then environment variables, then defaults."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

DEFAULT_INGEST_URL = "https://ingest.zoroval.com"
DEFAULT_APP_URL = "https://app.zoroval.com"
ENV_KEY = "ZOROVAL_INGEST_KEY"

KEY_PATTERN = re.compile(r"^zv_ingest_([a-z2-7]{12})_([0-9A-Za-z]{20,64})$")

MIB = 1024 * 1024


def key_id_of(key: Optional[str]) -> Optional[str]:
    """The public key id inside an ingest key, or None. Never returns the secret."""
    if not key:
        return None
    match = KEY_PATTERN.match(key.strip())
    return match.group(1) if match else None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def _default_service_name() -> str:
    for name in ("ZOROVAL_SERVICE_NAME", "OTEL_SERVICE_NAME"):
        if os.environ.get(name):
            return os.environ[name]
    argv0 = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    stem = os.path.splitext(argv0)[0]
    if stem and stem not in ("-c", "python", "python3", "__main__", "-m"):
        return stem
    return "unknown_service:python"


@dataclass
class Config:
    api_key: Optional[str] = None
    ingest_url: str = DEFAULT_INGEST_URL
    service_name: str = "unknown_service:python"
    environment: Optional[str] = None
    release: Optional[str] = None
    capture_content: bool = True
    redact: Optional[Callable[[Any], Any]] = None
    sample_rate: float = 1.0
    enabled: bool = True
    # Transport and queue. Defaults follow the PoC traps (REPORT.md sections 4.3 to 4.5).
    max_queue_bytes: int = 64 * MIB
    max_queue_spans: int = 20_000
    # Raw protobuf bytes per batch before zstd. 4 MiB is the gateway's per-span cap, so the largest
    # span still goes alone, and about 1 MiB after zstd fits the slow-uplink timeout (_transport.py).
    max_batch_bytes: int = 4 * MIB
    max_batch_spans: int = 4096
    max_request_bytes: int = 16 * MIB       # gateway wire limit (contracts 5.1)
    flush_interval: float = 1.0
    shutdown_timeout: float = 5.0
    request_timeout: float = 10.0            # floor; grows with the body size (_transport.request_timeout_for)
    max_attempts: int = 5
    # Content caps.
    max_content_bytes: int = 4 * MIB        # per content attribute (messages, inputs, outputs)
    max_metadata_bytes: int = 16 * 1024     # contracts section 6: zoroval.metadata at most 16 KB
    max_stack_bytes: int = 16 * 1024
    export_logs: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def traces_url(self) -> str:
        return self.ingest_url.rstrip("/") + "/v1/traces"

    @property
    def logs_url(self) -> str:
        return self.ingest_url.rstrip("/") + "/v1/logs"


def resolve(**kwargs: Any) -> Config:
    """Build a Config from init() arguments (None means "not given") and the environment."""
    given = {k: v for k, v in kwargs.items() if v is not None}
    cfg = Config()
    cfg.api_key = given.pop("api_key", None) or os.environ.get(ENV_KEY) or None
    if cfg.api_key:
        cfg.api_key = cfg.api_key.strip()
    cfg.ingest_url = given.pop("ingest_url", None) or os.environ.get("ZOROVAL_INGEST_URL") or DEFAULT_INGEST_URL
    cfg.service_name = given.pop("service_name", None) or _default_service_name()
    cfg.environment = given.pop("environment", None) or os.environ.get("ZOROVAL_ENVIRONMENT") or None
    cfg.release = given.pop("release", None) or os.environ.get("ZOROVAL_RELEASE") or None
    cfg.capture_content = given.pop("capture_content", _env_bool("ZOROVAL_CAPTURE_CONTENT", True))
    cfg.sample_rate = float(given.pop("sample_rate", _env_float("ZOROVAL_SAMPLE_RATE", 1.0)))
    cfg.sample_rate = min(1.0, max(0.0, cfg.sample_rate))
    cfg.enabled = given.pop("enabled", _env_bool("ZOROVAL_ENABLED", True))
    for name, value in given.items():
        if not hasattr(cfg, name):
            raise TypeError(f"zoroval.init() got an unexpected argument {name!r}")
        setattr(cfg, name, value)
    return cfg
