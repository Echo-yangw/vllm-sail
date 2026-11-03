# SPDX-License-Identifier: Apache-2.0
"""Exclude CuTeDSL on PPU, register the sampler JIT warmups, run PPU DeepGEMM.

Upstream ``warmup/kernel_warmup.py`` ends with a DeepGEMM warmup block gated
on ``is_deep_gemm_supported()``, which upstream defines via
``support_deep_gemm()`` — Hopper and Blackwell only. PPU devices report
sm80-class capability, so that block never fires on PPU, and PPU needs its own
warmup anyway: ``vllm_sail.model_executor.warmup.deep_gemm_warmup`` exercises
the PPU DeepGEMM wrapper's int8/bf16/fp4 grouped GEMMs and its tuned-config
lookup, none of which CUDA's warmup reaches.

vLLM 0.31.0's V2 runner only registers native top-k/top-p sampler warmups on
ROCm. PPU misses that branch, so register them before delegating to upstream.

PPU shares CUDA identity but cannot compile CuTeDSL. Before delegating to
upstream, discard CuTeDSL registrations and guard the legacy warmup provider
and its imported alias. Other registered JIT kernels retain their warmup.
The PPU DeepGEMM warmup is appended with the fork's gate — never when
``VLLM_DEEP_GEMM_WARMUP=skip``,
only when the PPU DeepGEMM wrapper reports support, and only when at least one
of the MoE / dense backend selectors still allows DeepGEMM.

PPU 1.0 (810E) reports sm80-class capability, where Triton rejects
``tl.float8e4nv`` outright ("type fp8e4nv not supported in this architecture")
regardless of whether the cast or a bitcast is traced. Some DeepSeek V4 upstream
kernels take that dtype unconditionally — directly in their ``@triton.jit`` body
or in an inlined helper — so upstream registers warmups that cannot compile on
that generation, and whose runtime call sites the plugin routes to INT8 or
software-E4M3 ops anyway. Registrations are therefore filtered by reading the
kernel source instead of a fixed deny-list: any registration whose Triton
function names ``float8e4nv`` is dropped. Devices with native FP8 (PPU 890P and
up) keep the upstream warmup. A registration whose *signature* needs the dtype
(passed as a ``torch.float8_e4m3fn`` tensor) is invisible to source inspection,
so a narrow per-kernel guard turns exactly that compile error into a warning
rather than a dead worker.

The target module imports the full worker/warmup stack at module scope. On
hosts where that chain cannot run (e.g. a minimal test venv without vLLM's
engine dependencies) there is also no warmup entry point for the patch to act
on, so this module skips itself with a warning instead of taking down the rest
of the plugin install. On a real PPU host the import always succeeds — vLLM's
own runtime requires the same dependencies.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
import textwrap

from vllm.logger import init_logger

from vllm_sail.patch.utils import patch
from vllm_sail.patch.bodies import bind_body
from vllm_sail.model_executor.warmup.ppu_kernel_warmup import (
    kernel_warmup as _ppu_warmup_body,
)

logger = init_logger(__name__)

_MODULE = "vllm.model_executor.warmup.kernel_warmup"
_CUTEDSL_MODULE = "vllm.model_executor.warmup.cutedsl_warmup"

# Triton refuses the OCP E4M3 dtype on compute capability < 8.9, so a kernel that
# names it can never be warmed up on PPU 1.0. Matching on the dtype symbol keeps
# the filter independent of how a kernel was registered (class, decorated
# function, or a subclass/wrapper defined by this plugin).
_UNSUPPORTED_FP8_DTYPE = "float8e4nv"
_UNSUPPORTED_FP8_ERRORS = ("fp8e4nv", "not supported in this architecture")


def _called_name(node: ast.expr) -> str | None:
    """Return the trailing name of a call target (``pkg.mod.fn`` -> ``fn``)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _names_unsupported_fp8(jit_fn, seen: set[int] | None = None) -> bool:
    """Whether a ``@triton.jit`` function uses E4M3 in a live code path.

    Triton inlines other ``@triton.jit`` helpers at trace time, so a kernel can
    pull the dtype in without naming it itself; follow those calls through the
    function's globals.
    """
    seen = set() if seen is None else seen
    if id(jit_fn) in seen:
        return False
    seen.add(id(jit_fn))
    source_fn = getattr(jit_fn, "fn", jit_fn)
    try:
        source = textwrap.dedent(inspect.getsource(source_fn))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return False
    if any(
        (isinstance(node, ast.Attribute) and node.attr == _UNSUPPORTED_FP8_DTYPE)
        or (isinstance(node, ast.Name) and node.id == _UNSUPPORTED_FP8_DTYPE)
        for node in ast.walk(tree)
    ):
        return True
    globals_ = getattr(source_fn, "__globals__", {})
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        helper = globals_.get(_called_name(node.func) or "")
        # Only JIT functions are inlined into the kernel body.
        if helper is None or helper is jit_fn or not hasattr(helper, "fn"):
            continue
        if _names_unsupported_fp8(helper, seen):
            return True
    return False


def _kernel_needs_unsupported_fp8(kernel) -> bool:
    jit_fn = getattr(kernel, "kernel", None)
    return jit_fn is not None and _names_unsupported_fp8(jit_fn)


def _is_unsupported_fp8_error(exc: BaseException) -> bool:
    message = str(exc)
    return all(marker in message for marker in _UNSUPPORTED_FP8_ERRORS)


def _tolerate_unsupported_fp8(kernel) -> None:
    """Skip a warmup whose compile key needs E4M3 as an argument dtype.

    Source inspection cannot see this case: the rejected dtype only appears in
    the signature Triton derives from a ``torch.float8_e4m3fn`` tensor, and the
    error is raised while tracing the function definition. Warmup is optional,
    so warn with the kernel name and let the worker continue.
    """
    compile_many = kernel.compile_many

    def guarded(compile_keys) -> None:
        try:
            compile_many(compile_keys)
        except Exception as exc:
            if not _is_unsupported_fp8_error(exc):
                raise
            logger.warning(
                "Skipping JIT warmup for %s: this PPU architecture cannot "
                "compile fp8e4nv. %s",
                type(kernel).__name__,
                exc,
            )

    try:
        kernel.compile_many = guarded
    except AttributeError:  # frozen kernel wrappers
        pass


def _exclude_ppu_cutedsl_registrations(worker) -> None:
    from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
        VllmCuTeDSLJitKernel,
    )

    registry = getattr(worker.model_runner, "jit_warmup_registry", None)
    if registry is None:
        return
    # vLLM 0.30 has no public backend filter. Keep the existing mapping and
    # registration arguments for every supported kernel; remove only CuTeDSL.
    registrations = registry._registrations
    unsupported = [
        kernel for kernel in registrations if isinstance(kernel, VllmCuTeDSLJitKernel)
    ]
    for kernel in unsupported:
        del registrations[kernel]
    if unsupported:
        logger.warning(
            "Skipping unsupported CuTeDSL JIT warmup on PPU: %s",
            ", ".join(type(kernel).__name__ for kernel in unsupported),
        )


def _exclude_sm80_fp8_registrations(worker) -> None:
    """Drop warmups Triton cannot compile without native FP8."""
    from vllm.platforms import current_platform

    if current_platform.has_device_capability(89):
        return
    registry = getattr(worker.model_runner, "jit_warmup_registry", None)
    if registry is None:
        return
    registrations = registry._registrations
    unsupported = [
        kernel for kernel in registrations if _kernel_needs_unsupported_fp8(kernel)
    ]
    for kernel in unsupported:
        del registrations[kernel]
    if unsupported:
        logger.warning(
            "Skipping JIT warmups that use tl.%s on this PPU architecture, "
            "which Triton cannot compile: %s",
            _UNSUPPORTED_FP8_DTYPE,
            ", ".join(type(kernel).__name__ for kernel in unsupported),
        )
    for kernel in registrations:
        _tolerate_unsupported_fp8(kernel)


def _register_ppu_sampler_warmups(worker) -> None:
    """Register native sampler warmups missing from the V2 runner on PPU.

    Only acts on the last PP rank of non-pooling models.  Duplicates are
    deduplicated by the upstream registry.  See patch ``remove_when``.
    """
    runner = worker.model_runner
    if not getattr(runner, "is_last_pp_rank", True):
        return
    if getattr(runner, "is_pooling_model", False):
        return
    registry = getattr(runner, "jit_warmup_registry", None)
    if registry is None:
        return
    from vllm.v1.sample.ops.topk_topp_sampler import register_top_k_top_p_warmups

    with registry.activate():
        register_top_k_top_p_warmups()


def _ppu_deep_gemm_warmup_enabled() -> bool:
    """The fork's gate for running DeepGEMM warmup on PPU."""
    import vllm.envs as envs

    import vllm_sail.envs as ppu_envs
    from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

    if envs.VLLM_DEEP_GEMM_WARMUP == "skip" or not is_deep_gemm_supported():
        return False
    moe_backend = ppu_envs.VLLM_SAIL_MOE_BACKEND
    dense_backend = ppu_envs.VLLM_SAIL_DENSE_BACKEND
    return (
        (not moe_backend or moe_backend == "deepgemm")
        or (not dense_backend or dense_backend == "deepgemm")
        or (ppu_envs.VLLM_SAIL_DENSE_BF16_DEEPGEMM)
    )


try:
    _upstream_module = importlib.import_module(_MODULE)
    _upstream_kernel_warmup = _upstream_module.kernel_warmup
except ImportError as exc:
    logger.warning(
        "vllm-sail: skipping the %s patch because the target module did not "
        "import (%s). This is expected in minimal test environments; on a PPU "
        "host the import succeeds.",
        _MODULE,
        exc,
    )
else:
    _ppu_upstream_warmup = bind_body(_ppu_warmup_body, _upstream_module)
    _upstream_cutedsl_warmup = importlib.import_module(_CUTEDSL_MODULE).cutedsl_warmup

    @patch(
        _MODULE,
        "cutedsl_warmup",
        reason="Rebind the preloaded legacy CuTeDSL warmup alias to the PPU guard.",
        affected_versions=">=0.32.0,<0.33.0",
        remove_when="The legacy CuTeDSL warmup gates execution on backend capabilities.",
    )
    @patch(
        _CUTEDSL_MODULE,
        "cutedsl_warmup",
        reason="Legacy CuTeDSL warmup checks is_cuda(), which also accepts PPU.",
        affected_versions=">=0.32.0,<0.33.0",
        remove_when="The legacy CuTeDSL warmup gates execution on backend capabilities.",
    )
    def cutedsl_warmup() -> None:
        from vllm.platforms import current_platform

        if not current_platform.is_ppu():
            return _upstream_cutedsl_warmup()

    @patch(
        _MODULE,
        "kernel_warmup",
        reason=(
            "Exclude NVIDIA-only MiniMax MSA imports and unsupported CuTeDSL JIT registrations on PPU, and the "
            "JIT warmups whose Triton kernels use fp8e4nv, which PPU 1.0's "
            "sm80-class capability cannot compile. "
            "Register native top-k/top-p sampler warmups that V2 only "
            "registers on ROCm. "
            "Upstream's DeepGEMM warmup is gated on support_deep_gemm() "
            "(sm90/sm100/sm120), so it never runs on PPU; and CUDA's warmup "
            "would not exercise the PPU DeepGEMM wrapper's kernels or tuned "
            "configs anyway. Appends the PPU DeepGEMM warmup with the fork's "
            "env gates after the upstream body."
        ),
        affected_versions=">=0.32.0,<0.33.0",
        remove_when=(
            "upstream gives kernel_warmup a per-platform warmup registry (or "
            "vllm.utils.deep_gemm dispatches on current_platform), so the PPU "
            "warmup can register itself. Sampler warmups: remove once upstream "
            "V2 registers native top-k/top-p on PPU for non-speculative paths. "
            "FP8 warmups: remove once upstream skips JIT warmups whose "
            "kernels cannot be compiled for the device's FP8 support."
        ),
    )
    def kernel_warmup(worker, *, process_local_only: bool = False):
        from vllm.platforms import current_platform

        if current_platform.is_ppu():
            _exclude_ppu_cutedsl_registrations(worker)
            _register_ppu_sampler_warmups(worker)
            _exclude_sm80_fp8_registrations(worker)
        upstream = (
            _ppu_upstream_warmup
            if current_platform.is_ppu()
            else _upstream_kernel_warmup
        )
        result = upstream(worker, process_local_only=process_local_only)

        if process_local_only or not current_platform.is_ppu():
            return result

        if _ppu_deep_gemm_warmup_enabled():
            from vllm_sail.model_executor.warmup.deep_gemm_warmup import (
                deep_gemm_warmup,
            )

            deep_gemm_warmup(
                worker.get_model(),
                worker.scheduler_config.max_num_batched_tokens,
            )
        return result

    _worker_module = sys.modules.get("vllm.v1.worker.gpu_worker")
    if (
        _worker_module is not None
        and getattr(_worker_module, "kernel_warmup", None) is _upstream_kernel_warmup
    ):
        patch(
            "vllm.v1.worker.gpu_worker",
            "kernel_warmup",
            reason="Rebind the preloaded worker alias so PPU avoids NVIDIA-only MSA warmup imports.",
            affected_versions=">=0.32.0,<0.33.0",
            remove_when="The worker resolves warmups through a platform-aware registry.",
        )(kernel_warmup)
