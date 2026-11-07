"""A stand-in for a Langfuse server: accepts the Langfuse SDK's OTLP export and any other call.
Localhost only. Used to prove coexistence (no duplicate spans, Langfuse keeps working)."""

from __future__ import annotations

import gzip
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from .fake_gateway import _attrs


class LangfuseStandIn:
    def __init__(self) -> None:
        self.spans: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _ok(self, body: bytes = b"{}", content_type: str = "application/json") -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                self._ok()

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                if self.path.endswith("/otel/v1/traces"):
                    if (self.headers.get("Content-Encoding") or "") == "gzip":
                        body = gzip.decompress(body)
                    request = ExportTraceServiceRequest()
                    request.ParseFromString(body)
                    for rs in request.resource_spans:
                        for ss in rs.scope_spans:
                            for span in ss.spans:
                                owner.spans.append({"name": span.name, "scope": ss.scope.name,
                                                    "trace_id": span.trace_id.hex(),
                                                    "attributes": _attrs(span.attributes)})
                    return self._ok(b"", "application/x-protobuf")
                self._ok()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "LangfuseStandIn":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
