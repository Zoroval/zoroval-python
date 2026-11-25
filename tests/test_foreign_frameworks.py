"""No second span when OpenInference's LangChain or LlamaIndex instrumentor already records the call
(found 2026-11-25: both doubled every LLM call), and still a span for a raw client call in a tool."""

import contextvars
import sys
import types
import uuid

import pytest
from opentelemetry.sdk.trace import TracerProvider

from zoroval.integrations._common import foreign_llm_span_active

_var = contextvars.ContextVar("child_runnable_config", default=None)


@pytest.fixture
def langchain(monkeypatch):
    module = types.ModuleType("langchain_core.runnables.config")
    module.var_child_runnable_config = _var
    monkeypatch.setitem(sys.modules, "langchain_core.runnables.config", module)
    return module


class Run:
    def __init__(self, run_type, parent, end_time=None):
        self.id = uuid.uuid4()
        self.run_type = run_type
        self.parent_run_id = parent
        self.end_time = end_time


def _tracer(module_name, runs):
    cls = type("OpenInferenceTracer", (), {})
    cls.__module__ = module_name
    tracer = cls()
    tracer.run_map = {str(r.id): r for r in runs}
    return tracer


def _under(parent, handlers):
    callbacks = types.SimpleNamespace(parent_run_id=parent, handlers=handlers, inheritable_handlers=[])
    return _var.set({"callbacks": callbacks})


def test_chat_model_run_traced_by_openinference(langchain):
    node = uuid.uuid4()
    token = _under(node, [_tracer("openinference.instrumentation.langchain._tracer", [Run("llm", node)])])
    try:
        assert foreign_llm_span_active() is True
    finally:
        _var.reset(token)


def test_raw_call_inside_a_tool_is_ours(langchain):
    node, tool_run = uuid.uuid4(), uuid.uuid4()
    tracer = _tracer("openinference.instrumentation.langchain._tracer", [Run("llm", node)])
    token = _under(tool_run, [tracer])  # the tool's run has no open LLM run under it
    try:
        assert foreign_llm_span_active() is False
    finally:
        _var.reset(token)


def test_finished_llm_run_and_other_tracers_ignored(langchain):
    node = uuid.uuid4()
    done = _tracer("openinference.instrumentation.langchain._tracer", [Run("llm", node, end_time=1)])
    langsmith = _tracer("langchain_core.tracers.langchain", [Run("llm", node)])  # not exported via OTel
    token = _under(node, [done, langsmith])
    try:
        assert foreign_llm_span_active() is False
    finally:
        _var.reset(token)


@pytest.fixture
def llama_index(monkeypatch):
    module = types.ModuleType("llama_index.core.base.llms.base")
    module.BaseLLM = type("BaseLLM", (), {})
    module.OpenAI = type("OpenAI", (module.BaseLLM,), {})
    monkeypatch.setitem(sys.modules, "llama_index.core.base.llms.base", module)
    return module


def test_llama_index_llm_span_before_its_kind_is_set(llama_index):
    tracer = TracerProvider().get_tracer("openinference.instrumentation.llama_index")
    with tracer.start_as_current_span("CondensePlusContextChatEngine.stream_chat"):
        assert foreign_llm_span_active() is False  # a chat engine, not an LLM: a raw call under it is ours
    with tracer.start_as_current_span("OpenAI.stream_chat"):
        assert foreign_llm_span_active() is True
    with tracer.start_as_current_span("FunctionTool.call"):
        assert foreign_llm_span_active() is False
    with tracer.start_as_current_span("OpenAI.chat") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        assert foreign_llm_span_active() is False
