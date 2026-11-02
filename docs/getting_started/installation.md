# Installation

vLLM SAIL runs on Linux hosts with a PPU accelerator and a compatible SAIL SDK.
For documentation and unit-test development on a machine without PPU hardware,
follow [Contributing](../../CONTRIBUTING.md#development-environment).

## 1. Prepare the SAIL software stack

Use an SDK environment with the following components installed together:

| Component | Requirement |
| --- | --- |
| SAIL SDK | Driver, HGGC headers, runtime libraries and `hgcc` compiler |
| SAIL PyTorch | A PPU build of PyTorch from the SAIL fork |
| Triton | The [Triton for SAIL](https://github.com/t-head/triton-for-sail) build matching your SDK |
| Compute libraries | PPU builds of `acext`, `deep_gemm`, `flash_attn` and `flash_attn_3`, as needed by the selected backend |
| Model-specific libraries | PPU FlashMLA, PLA, TileLang or DeepEP when required by the workload |
| Python | 3.10–3.13, with matching SDK wheels |

Obtain the SDK and its Python packages through your PPU software distribution.
Use a matched set of libraries for the active SAIL PyTorch build. The plugin's
Python requirements do not install the SDK.

Activate that environment, then set paths for your installation:

```bash
export PPU_SDK=/path/to/ppu-sdk
export HGCC="$PPU_SDK/bin/hgcc"
export PYTORCH_SAIL_ARCH='ppu_15;ppu_10'
unset VLLM_SAIL_HG_ARCH
unset VLLM_SAIL_SKIP_EXT
```

CMake first uses an explicit `CMAKE_HGCC_DIR`, configured module path or
installed package. When none is available, it downloads the pinned
[cmake-hgcc](https://github.com/t-head/cmake-hgcc) revision and reuses it from
the build directory. For an offline build or a local mirror, set
`CMAKE_HGCC_DIR` to an existing checkout. Ensure the SDK runtime libraries are
on the system loader's search path, following your SDK's environment setup.

`PYTORCH_SAIL_ARCH` selects the device images included in the build. Use
`ppu_10`, `ppu_15`, or the quoted list above. The default includes both targets;
quoting prevents the shell from interpreting the semicolon as a command boundary.

## 2. Install matching vLLM

### Build both wheels

In the activated SAIL SDK / SAIL PyTorch environment, run from the vLLM SAIL
checkout on Linux (Bash 4+, Python 3.10–3.13):

```bash
BUILD_JOBS=8 bash scripts/build_wheels.sh --output-dir /path/to/new-wheelhouse
```

The script builds the commit in `.github/vllm-main-verified.commit` with
`VLLM_TARGET_DEVICE=empty`, including its Rust frontend and parser, then builds
vllm-sail. It installs build dependencies and the new vLLM wheel into the active
environment while preserving SDK package versions; runtime dependencies and the
final vllm-sail wheel are not installed.

The output directory must be empty; the default is `artifacts/local-<run>/`.
It contains both wheels under `wheels/`, source commits and hashes in
`wheels/wheel-manifest.json`, and build logs under `logs/`. See `--help` for
SDK, mirror and parallelism options. Continue with step 4 for runtime checks.

### Install a separately supplied vLLM build

Install a vLLM build with `VLLM_TARGET_DEVICE=empty` in the same environment.
This lets the PPU plugin provide the native operations used by the engine.
The supported release line is `>=0.32.0,<0.33.0`. Build the exact upstream commit
recorded in [the verified pin](../../.github/vllm-main-verified.commit) and
[kernel manifest](../../csrc/upstream/manifest.toml); both records must agree.
**v0.32.0** is the release baseline for this line.

For a commit-pinned main build, the full upstream SHA establishes compatibility.
The numeric prefix produced by setuptools-scm or `git describe` can depend on
the tags present in the build checkout; it does not change the supported release
range. Keep the full SHA in the wheel version and leave version overrides unset.

Use a matching wheel supplied with your PPU environment, or build that upstream
source pin. For a source build, prepare vLLM's build and runtime dependencies
using the SDK's package constraints, including the Rust toolchain required by
that checkout. Then run from the vLLM checkout:

```bash
python tools/use_existing_torch.py
VLLM_REQUIRE_RUST_FRONTEND=1 VLLM_TARGET_DEVICE=empty \
  python -m pip install --no-build-isolation --no-deps .
```

The upstream helper removes dependency pins that would replace the active torch
installation. `--no-deps` assumes the runtime dependencies are already installed.
Keep the actual vLLM revision with your environment records.

## 3. Build and install the plugin

Clone the repository, then install its build dependencies in the same environment:

```bash
git clone https://github.com/t-head/vllm-sail.git
cd vllm-sail
python -m pip install -r requirements/build.txt -r requirements/ppu.txt
```

Check that the build will include native extensions:

```bash
python - <<'PY'
from vllm_sail.native.toolchain import decide, probe

decision = decide(probe())
print(decision.reason)
if not decision.build:
    raise SystemExit(1)
PY
```

Resolve any prerequisite reported by the check before continuing:

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

Build isolation is disabled so the build can use the active SAIL PyTorch and
SDK. Native builds compile the authored plugin HGGC sources and committed
upstream corpus; installation requires neither sailify nor an upstream source
checkout.

For a distributable wheel and build controls, see [HGGC kernels](../developer_guide/kernels.md).

## 4. Check the installation

```bash
python -c "from vllm_sail import collect_env; collect_env()"
python - <<'PY'
import importlib
import torch
import vllm_sail
from vllm_sail.native.extensions import import_kernels

print('Plugin:', vllm_sail.__file__)
for name in ('_upstream_C', '_upstream_moe_C', '_C', '_moe_C'):
    extension = importlib.import_module(f'vllm_sail.{name}')
    print(name, extension.__file__)
print('Native preflight:', import_kernels(strict=True))
PY
```

The four libraries must load from the intended installation. Strict preflight
also checks representative operation registrations. These checks do not execute
a model; continue with the [quickstart](quickstart.md) and
[device verification](../user_guide/verification_guide.md).

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Build fails before CMake configuration | Check that PyTorch is importable and that `hgcc` and CMake are available. Native extensions are required by default. |
| Build unexpectedly produces a `py3-none-any` wheel | Unset `VLLM_SAIL_SKIP_EXT`; it is the only Python-only build path. |
| CMake cannot fetch the HG language modules | Restore network access or set `CMAKE_HGCC_DIR` to an existing cmake-hgcc checkout. |
| Architectures disagree | Set `PYTORCH_SAIL_ARCH`; unset `VLLM_SAIL_HG_ARCH` or make both settings use the same targets. |
| A shared library reports an undefined symbol | Match the SDK library to the active SAIL PyTorch and runtime, then restart the process. |
| Native extensions are missing after installing a wheel | Check `vllm_sail.__file__`. A checkout can shadow the installed package; use an editable native build there or run outside it. |
| Plugin is not selected | If `VLLM_PLUGINS` is set, include `ppu`; see [Configuration](../user_guide/configuration.md). |
| vLLM version is rejected | Use the supported release range. For a development build with ambiguous metadata, see [Configuration](../user_guide/configuration.md#version-metadata). |
