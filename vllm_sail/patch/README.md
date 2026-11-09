# Runtime patches

A patch replaces a piece of upstream vLLM at runtime. Each patch adds an upstream
implementation dependency that must be reviewed on upgrades. Treat adding one
as a last resort.

## Escalation order — try these first, in order

1. **A registry / registration API.** vLLM already exposes real extension points.
   Prefer them to reduce dependence on upstream implementation details.
   - Dense GEMM kernels → `vllm.model_executor.kernels.linear.register_linear_kernel`
   - Custom ops → `vllm.utils.torch_utils.direct_register_custom_op`
   - Models → `vllm.ModelRegistry.register_model`
2. **A platform hook.** `PPUPlatform.apply_config_platform_defaults`,
   `get_valid_backends`, `use_custom_allreduce`, … are virtual by design. Most
   config-level differences belong here.
3. **An attention backend.** Register a backend class rather than patching the
   selection logic.
4. **Only then, a patch.**

## Writing a patch

Copy a template from `template/`, put it in the right category, add it to that
category's `__init__.py`, and fill in every field.

Patches are applied by `vllm_sail.patch.install()`, called once from
`vllm_sail.register_out_of_tree()`. Importing `vllm_sail.patch` itself does
**nothing** — that keeps `vllm_sail.patch.utils` importable without vLLM (so the
framework is unit-testable on a bare CPU runner) and avoids the circular import
that an install-on-import package would create, since every patch module does
`from vllm_sail.patch.utils import patch`.

```python
from vllm_sail.patch.utils import patch


@patch(
    "vllm.some_module",
    "target_function",
    reason="PPU's foo kernel needs bar; upstream hardcodes baz.",
    affected_versions=">=0.31.0,<0.32.0",
    remove_when="upstream PR #12345 lands, which makes baz configurable.",
)
def target_function(arg: int) -> int:
    ...
```

`reason`, `affected_versions` and `remove_when` are **mandatory** and validated at
import. Generate an on-demand report from actual installation results with
`python tools/patch_manifest.py`, or save it using `--output /tmp/patch-report.md`.
Reports are not committed or used as test input. Each report covers only the
current process and environment, including the conditional patches installed
there; it is not an inventory of every possible platform path.
`remove_when` must be a *verifiable condition* — "when upstream fixes it" is not
acceptable; name the PR, the API, or the capability.

### Prefer delegation over copying a body

If you can express the change as "call upstream, then adjust", do that instead of
copying the upstream body:

```python
_upstream = module.target          # captured before patching

@patch("vllm.some_module", "target", reason=..., affected_versions=..., remove_when=...)
def target(*args, **kwargs):
    result = _upstream(*args, **kwargs)
    return adjust(result) if current_platform.is_ppu() else result
```

A delegating patch survives upstream refactors of the parts it does not care
about. `patch/enhancement/custom_op_dispatch.py` is the worked example: upstream's
`dispatch_forward` is ~35 lines of enable bookkeeping plus a platform ladder, and
we only need one branch of the ladder.

When you *must* copy a body (Triton kernels, tightly interleaved logic), copy it
**verbatim** — same name, signature, decorators, comments and unchanged lines —
and mark only the changed lines:

```python
    # PPU MODIFICATION: begin
    ...
    # PPU MODIFICATION: end
```

That is what makes the next upstream bump a diff rather than an investigation.

### Borrowed bodies and direct source contracts

The runtime pilot separates `enhancement/runtime.py` (installation, PPU dispatch
and loaded-consumer rebinding) from `_upstream/runtime.py` (two maintained vLLM
bodies with limited PPU changes). These bodies are not deterministically
generated. Their source revision records reviewed provenance and does not select
a runtime version. Bind each body to its own provider with `bind_body` so globals
remain live; do not combine the sampler and model runner namespaces.

CPU tests verify framework and feature behavior without importing the complete
patch set. With `VLLM_SOURCE_ROOT` set to the pinned checkout, source tests check
required targets, representative calls, registration APIs and consumer imports
directly. Static discovery handles named constants, module aliases and finite
literal loops; opaque installers require feature-specific source and behavior
checks. Unknown forms fail with their location. An additive target is declared
explicitly with `allow_missing=True`, never inferred from unavailable source.

There is no interface signature lock to refresh. The existing
`python tools/check_patch_drift.py` and `tools/patch_baseline.json` remain required
upgrade review signals: binding compatibility does not establish unchanged
upstream behavior. Do not refresh that baseline merely for a body relocation.
The drift checker does not follow helper dependencies, its records depend on
the installation environment, and missing captured source can still be reported
as additive. Retiring it needs a separate, reviewed source-provenance and upgrade
comparison workflow that distinguishes unresolved source from additive targets.

### Target forms

| Form | Path | Notes |
|---|---|---|
| Module function / class / Triton kernel | `@patch("vllm.mod")` or `@patch("vllm.mod", "name")` | Installs the object **unchanged**. |
| Class method / property | `@patch("vllm.mod", "Cls.method")` | Resolves `Cls` internally; preserves `staticmethod` / `classmethod` / `property`. |
| Data constant / lookup table | `patch_value("vllm.mod", "NAME", value, ...)` | Imperative, not a decorator. |

**Triton kernels: `@patch` must be the outermost decorator**, so that the fully
decorated Triton object is what gets installed:

```python
@patch("vllm.some_module", "target_kernel", reason=..., affected_versions=..., remove_when=...)
@triton.heuristics({...})
@triton.autotune(configs=[...], key=[...])
@triton.jit
def target_kernel(...):
    ...
```

Keep `@patch` outside method descriptors too:

```python
@patch("vllm.some_module", "TargetClass.from_config", reason=..., ...)
@classmethod
def from_config(cls, config): ...
```

### `allow_missing`

Use `allow_missing=True` **only** when deliberately adding a new attribute
(`Platform.is_ppu`, `CustomOp.forward_ppu`, the int8 quant constants). Normal
patches require the target to exist, so a renamed upstream symbol fails loudly at
startup instead of silently doing nothing.

## Categories

| Directory | Contents |
| --- | --- |
| `bugfix/` | Upstream defects that affect PPU; remove as the corresponding fixes land. |
| `enhancement/` | PPU integration that requires adapting upstream behavior. |
| `performance/` | Functionally equivalent optimizations for PPU. |

## Rules

- Lazy platform access. A patch module must not call `current_platform` at import
  time; patches are installed before the platform is necessarily resolved. Do it
  inside the function body.
- Use package-qualified or explicit relative imports. Test that patches actually
  replace their targets in the installed plugin environment.
- Every patch needs a test asserting the target was replaced, the marker records
  the original, and re-application raises.
- Never assume idempotency. `vllm.general_plugins` is loaded in every process;
  the guards in `vllm_sail/__init__.py` are what make that safe.

## Maintenance

`tools/check_patch_drift.py` hashes the upstream source each patch replaced and
compares it against `tools/patch_baseline.json`. On an upstream bump it reports
exactly which patches now sit on changed code, with each patch's `reason` and
`remove_when`, so realignment is a checklist. Run `--update` after reviewing.
