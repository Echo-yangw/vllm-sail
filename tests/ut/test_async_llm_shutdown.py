# SPDX-License-Identifier: Apache-2.0
"""Abort shutdown cancels its output consumer before intentional core death."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.support.source import function
from tests.support.upstream import SUPPORTED_RANGE
from vllm_sail.patch.utils import PATCH_REGISTRY, original_of, patch

MODULE_PATH = Path(__file__).parents[2] / "vllm_sail/patch/bugfix/async_llm_shutdown.py"
TARGET = "vllm.v1.engine.async_llm.AsyncLLM.shutdown"


@pytest.fixture
def shutdown_module():
    spec = importlib.util.spec_from_file_location("_shutdown_leaf_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    count = len(PATCH_REGISTRY)
    try:
        yield module
    finally:
        del PATCH_REGISTRY[count:]


def modules(monkeypatch, *, ppu=True, cancel=None, engine=None):
    leaves = {
        "vllm.platforms": NS(
            current_platform=NS(is_ppu=lambda: ppu, is_cuda=lambda: True)
        ),
        "vllm.utils.async_utils": NS(cancel_task_threadsafe=cancel),
        "vllm.v1.engine.async_llm": NS(AsyncLLM=engine),
    }
    names = set(leaves)
    for name in leaves:
        parts = name.split(".")
        names.update(".".join(parts[:i]) for i in range(1, len(parts)))
    for name in sorted(names, key=lambda value: value.count(".")):
        module = types.ModuleType(name)
        module.__path__ = []
        if name in leaves:
            module.__dict__.update(vars(leaves[name]))
        monkeypatch.setitem(sys.modules, name, module)


def test_bugfix_and_leaf_import_without_installing(monkeypatch):
    import vllm_sail.patch.bugfix as bugfix

    def unexpected():
        pytest.fail("importing a cached category must not install patches")

    monkeypatch.setattr(bugfix, "install", unexpected)
    assert importlib.import_module("vllm_sail.patch.bugfix") is bugfix
    assert callable(bugfix.install)


def test_cached_category_is_explicitly_installed_once(monkeypatch):
    import vllm_sail.patch as package
    import vllm_sail.patch.bugfix as bugfix

    calls = []
    monkeypatch.setattr(bugfix, "install", lambda: calls.append("bugfix"))
    monkeypatch.setattr(package, "_CATEGORIES", ("bugfix",))
    monkeypatch.setattr(package, "_installed", False)
    package.install()
    package.install()
    assert calls == ["bugfix"]


@pytest.mark.parametrize(
    "ppu,timeout", [(True, 0), (False, 0), (True, 3), (True, None)]
)
def test_only_ppu_abort_cancels_before_delegating(
    monkeypatch, shutdown_module, ppu, timeout
):
    calls = []
    handler = object()
    instance = NS(output_handler=handler)
    token = object()

    def original(self, timeout=None):
        assert self is instance
        calls.append(("upstream", timeout))
        return token

    modules(monkeypatch, ppu=ppu, cancel=lambda value: calls.append(("cancel", value)))
    result = shutdown_module._wrap_shutdown(original)(instance, timeout=timeout)
    expected = [("cancel", handler)] if ppu and timeout == 0 else []
    assert calls == [*expected, ("upstream", timeout)]
    assert result is token


@pytest.mark.parametrize("handler_present", [False, True])
def test_partial_construction_and_upstream_errors_are_preserved(
    monkeypatch, shutdown_module, handler_present
):
    calls = []
    error = RuntimeError("upstream shutdown failure")
    instance = NS(output_handler=None) if handler_present else NS()
    modules(monkeypatch, cancel=lambda task: calls.append(task))

    def original(self, timeout=None):
        assert self is instance and timeout == 0
        raise error

    with pytest.raises(RuntimeError) as raised:
        shutdown_module._wrap_shutdown(original)(instance, timeout=0)
    assert raised.value is error and calls == []


def test_install_records_real_target_and_reapplication_guard(
    monkeypatch, shutdown_module
):
    class Engine:
        def shutdown(self, timeout=None):
            pass

    original = Engine.shutdown
    modules(monkeypatch, engine=Engine, cancel=lambda task: None)
    shutdown_module.install()
    replacement = Engine.shutdown
    assert replacement is not original
    assert original_of(replacement, TARGET) is original
    assert PATCH_REGISTRY[-1].target == TARGET
    assert PATCH_REGISTRY[-1].reason and PATCH_REGISTRY[-1].remove_when
    assert PATCH_REGISTRY[-1].affected_versions == SUPPORTED_RANGE
    count = len(PATCH_REGISTRY)
    shutdown_module.install()
    assert len(PATCH_REGISTRY) == count
    with pytest.raises(RuntimeError, match="already patched"):
        patch(
            "vllm.v1.engine.async_llm",
            "AsyncLLM.shutdown",
            reason="test",
            affected_versions="test",
            remove_when="test",
        )(shutdown_module._wrap_shutdown(replacement))


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "unexpected_death,rank_retired,engine_error",
    [
        (False, False, True),
        (True, False, True),
        (True, True, True),
        (True, True, False),
    ],
)
def test_real_output_loop_distinguishes_shutdown_from_engine_failure(
    monkeypatch,
    shutdown_module,
    upstream_source_root,
    unexpected_death,
    rank_retired,
    engine_error,
):
    namespace = {"asyncio": asyncio}
    utils = upstream_source_root / "vllm/utils/async_utils.py"
    for name in ("cancel_task_threadsafe", "run_in_loop", "in_loop"):
        namespace[name] = function(utils, name, namespace)
    path = upstream_source_root / "vllm/v1/engine/async_llm.py"
    shutdown = function(path, "AsyncLLM.shutdown", namespace)
    start = function(path, "AsyncLLM._run_output_handler", namespace)
    modules(monkeypatch, cancel=namespace["cancel_task_threadsafe"])
    middleware = (
        upstream_source_root / "vllm/entrypoints/serve/elastic_ep/middleware.py"
    )
    for name in ("get_elastic_ep_rank_retired", "set_elastic_ep_rank_retired"):
        namespace[name] = function(middleware, name, namespace)
    namespace["set_elastic_ep_rank_retired"](rank_retired)
    assert namespace["get_elastic_ep_rank_retired"]() is rank_retired
    for name in ("VLLMError", "VLLMServerError"):
        namespace[name] = function(
            upstream_source_root / "vllm/exceptions.py", name, namespace
        )
    EngineDeadError = function(
        upstream_source_root / "vllm/v1/engine/exceptions.py",
        "EngineDeadError",
        namespace,
    )

    async def scenario():
        loop = asyncio.get_running_loop()
        errors, notices, propagated = [], [], []
        failure = (
            EngineDeadError() if engine_error else ValueError("unexpected failure")
        )
        namespace.update(
            logger=NS(
                exception=lambda message: errors.append(message),
                info=lambda message: notices.append(message),
            ),
            envs=NS(VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=1),
            shutdown_prometheus=lambda: None,
        )

        class Core:
            def __init__(self):
                self.future = loop.create_future()

            async def get_output_async(self):
                return await self.future

            def shutdown(self, timeout=None):
                assert timeout == 0
                settled = threading.Event()

                def close():
                    if not self.future.done():
                        self.future.set_exception(EngineDeadError())
                    instance.output_handler.add_done_callback(
                        lambda task: settled.set()
                    )

                loop.call_soon_threadsafe(close)
                assert settled.wait(2), "the output loop did not observe core teardown"

        core = Core()
        instance = NS(
            output_handler=None,
            engine_core=core,
            output_processor=NS(propagate_error=lambda error: propagated.append(error)),
            log_stats=False,
            logger_manager=None,
            renderer=NS(mm_processor_cache=None, shutdown=lambda: None),
        )
        start(instance)
        await asyncio.sleep(0)
        if unexpected_death:
            core.future.set_exception(failure)
        else:
            await asyncio.to_thread(
                shutdown_module._wrap_shutdown(shutdown), instance, 0
            )
        try:
            await instance.output_handler
        except asyncio.CancelledError:
            pass
        if unexpected_death:
            assert propagated == [failure]
            if rank_retired and engine_error:
                assert errors == []
                assert notices == [
                    "[Elastic EP] Retired rank stopped receiving EngineCore outputs"
                ]
            else:
                assert errors == ["AsyncLLM output_handler failed."]
                assert notices == []
        else:
            assert errors == [] and notices == [] and propagated == []
            assert instance.output_handler.cancelled()

    asyncio.run(scenario())
