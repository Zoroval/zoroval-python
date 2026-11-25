"""LiteLLM's logging pool runs each task in a copy of the caller's context, so a streamed call's span
keeps its zoroval.conversation id (found 2026-11-25: one LiteLLM chat split into three traces)."""

import contextvars
import sys
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from zoroval import _context
from zoroval.integrations import litellm as integration

VAR = contextvars.ContextVar("sim_var", default=None)


@pytest.fixture
def pool(monkeypatch):
    module = types.ModuleType(integration.MODULE)
    module.executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setitem(sys.modules, integration.MODULE, module)
    yield module.executor
    integration.uninstrument()
    module.executor.shutdown()


def test_task_sees_the_callers_conversation(pool):
    assert integration.instrument()
    with _context.conversation("chat-7f3a"):
        future = pool.submit(_context.current_conversation_id)
    assert future.result() == "chat-7f3a"


def test_without_patch_the_task_loses_it(pool):
    with _context.conversation("chat-7f3a"):
        future = pool.submit(_context.current_conversation_id)
    assert future.result() is None


def test_litellm_own_copy_is_passed_through(pool):
    integration.instrument()
    token = VAR.set("theirs")
    ctx = contextvars.copy_context()
    VAR.reset(token)
    assert pool.submit(ctx.run, VAR.get).result() == "theirs"


def test_idempotent_and_removable(pool):
    assert integration.instrument() and integration.instrument()
    integration.uninstrument()
    assert "submit" not in pool.__dict__


def test_not_loaded_is_a_no_op(monkeypatch):
    monkeypatch.delitem(sys.modules, integration.MODULE, raising=False)
    assert integration.instrument() is False


def test_patched_when_litellm_is_imported_after_init(tmp_path, monkeypatch):
    pkg = tmp_path / "litellm" / "litellm_core_utils"
    pkg.mkdir(parents=True)
    (tmp_path / "litellm" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "thread_pool_executor.py").write_text(
        "from concurrent.futures import ThreadPoolExecutor\nexecutor = ThreadPoolExecutor(max_workers=1)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("litellm", "litellm.litellm_core_utils", integration.MODULE):
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        assert integration.instrument() is False  # not loaded yet: waits for the import
        from litellm.litellm_core_utils.thread_pool_executor import executor
        with _context.conversation("chat-late"):
            assert executor.submit(_context.current_conversation_id).result() == "chat-late"
        assert integration._finder not in sys.meta_path
    finally:
        integration.uninstrument()
        for name in ("litellm", "litellm.litellm_core_utils", integration.MODULE):
            sys.modules.pop(name, None)
