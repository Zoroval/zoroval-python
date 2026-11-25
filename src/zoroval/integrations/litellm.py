"""LiteLLM keeps its conversation: `callbacks=["otel"]` makes LiteLLM's spans in its logging thread pool.
For a streamed call that work was handed over without the caller's context, so the span had no parent
and no `zoroval.conversation` id, and one chat split into several traces (found 2026-11-25). LiteLLM
already copies the context for non-streamed calls; this does the same for every task its logging pool
runs. The caller's context is copied per task, never shared or taken from another thread.
"""

from __future__ import annotations

import contextvars
import sys
from typing import Any

from .._util import internal_error

MODULE = "litellm.litellm_core_utils.thread_pool_executor"
MARK = "__zoroval_context__"
_patched: list[Any] = []


class _PatchOnImport:
    """A finder for LiteLLM's pool module only: when the app imports LiteLLM after init(), the pool is
    patched as soon as its module has run. Every other import is left to the other finders."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != MODULE:
            return None
        try:
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                        spec.loader = _Loader(spec.loader)
                    return spec
        except Exception as error:  # noqa: BLE001
            internal_error("litellm.find_spec", error)
        return None


class _Loader:
    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def create_module(self, spec: Any) -> Any:
        return self._loader.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._loader.exec_module(module)
        try:
            _remove_finder()
            _patch(module)
        except Exception as error:  # noqa: BLE001
            internal_error("litellm.patch_on_import", error)


_finder = _PatchOnImport()


def _remove_finder() -> None:
    while _finder in sys.meta_path:
        sys.meta_path.remove(_finder)


def instrument() -> bool:
    """Patch LiteLLM's logging pool now if LiteLLM is loaded, or when it is imported. Never imports it."""
    module = sys.modules.get(MODULE)
    if module is None:
        if _finder not in sys.meta_path:
            sys.meta_path.insert(0, _finder)
        return False
    return _patch(module)


def _patch(module: Any) -> bool:
    executor = getattr(module, "executor", None)
    submit = getattr(executor, "submit", None)
    if submit is None:
        return False
    if getattr(submit, MARK, False):
        return True

    def submit_in_context(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        if isinstance(getattr(fn, "__self__", None), contextvars.Context):
            return submit(fn, *args, **kwargs)  # LiteLLM copied the context itself
        try:
            ctx = contextvars.copy_context()
        except Exception as error:  # noqa: BLE001
            internal_error("litellm.copy_context", error)
            return submit(fn, *args, **kwargs)
        return submit(ctx.run, fn, *args, **kwargs)

    setattr(submit_in_context, MARK, True)
    try:
        executor.submit = submit_in_context
    except Exception as error:  # noqa: BLE001
        internal_error("litellm.instrument", error)
        return False
    _patched.append(executor)
    return True


def uninstrument() -> None:
    _remove_finder()
    while _patched:
        executor = _patched.pop()
        try:
            if getattr(executor.__dict__.get("submit"), MARK, False):
                del executor.submit
        except Exception as error:  # noqa: BLE001
            internal_error("litellm.uninstrument", error)
