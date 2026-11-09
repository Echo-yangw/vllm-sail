# Agent Guide

User-facing overview: [README.md](README.md). Contribution workflow:
[CONTRIBUTING.md](CONTRIBUTING.md).

## Project terminology

- Display name: **vLLM SAIL**. **SAIL** is the software brand; **Zhenwu (真武)**
  is the hardware product name. Document **Zhenwu 810E** and **Zhenwu M890**
  T-Head AI chips.
- Python distribution: `vllm-sail`; import package: `vllm_sail`.
- Use **SAIL SDK** in prose. Preserve `PPU_SDK`, the `ppu` plugin name,
  `PPUPlatform`, and code-facing device identifiers. Explain identifiers such as
  `PPU-ZW810E` (Zhenwu 810E) and `ZW-M890P` (Zhenwu M890) when documenting
  configuration names or program output.
- Plugin runtime and build variables use `VLLM_SAIL_*`. Runtime variables retain
  corresponding `VLLM_PPU_*` aliases, with SAIL taking precedence. Build variables
  accept only the `VLLM_SAIL_*` names.
- Declare runtime variables in `vllm_sail/envs.py` and keep their access lazy.

## Verification

From the repository root:

```bash
.venv-test/bin/python -m pytest tests/ut -q
.venv-test/bin/pre-commit run --files path/to/changed_file.py
```

CI uses `python -m pytest tests/ut -q` with only `requirements/dev.txt` installed,
across Python 3.10–3.13. This merge gate must pass without torch, vLLM or a device.
Never add imports or test dependencies that break that contract.

`tests/e2e` and `@pytest.mark.ppu` require device verification. Skips are not
failures and are not runtime evidence. The uninstalled `_version` warning is
expected. Follow [Verification](docs/user_guide/verification_guide.md) before
making hardware or model-support claims.

## Source ownership

- Never clean up copied upstream bodies in `vllm_sail/patch/**`. Preserve their
  structure and express behavior changes only inside `PPU MODIFICATION:
  begin/end` markers.
- Runtime patch reports are generated on demand by `tools/patch_manifest.py`;
  do not commit generated reports. The pilot borrowed bodies in
  `vllm_sail/patch/_upstream/runtime.py` are maintained source, not generated
  output. Preserve their attribution and PPU modification markers.
- Never hand-edit `csrc/upstream/**`, except its input `manifest.toml`.
  Edit inputs or transformations and regenerate.
- `csrc/plugin/` contains authored HGGC sources and host bindings. Edit these
  directly; its manifest lists the files compiled into the plugin extensions.
- Preserve lazy imports and entry-point idempotency: vLLM loads general plugins
  in every process. Importing the package must not install patches or registries.
- Preserve dirty and untracked work. Use topic branches and pull requests for
  code publication; never force-push or push directly to shared branches.

## Routes

- Module and dependency boundaries: [Architecture](docs/developer_guide/architecture.md).
  PPU reuses CUDA paths; use `current_platform.is_ppu()` for PPU-only behavior.
- Patches: [Runtime patch guide](vllm_sail/patch/README.md). Try registration APIs,
  platform hooks and attention extensions first. Every patch requires `reason`,
  `affected_versions` and `remove_when`.
- Native code and toolchains: [HGGC kernels](docs/developer_guide/kernels.md).
  Preserve upstream interface identifiers used by SAIL PyTorch.
- Installation: [Installation guide](docs/getting_started/installation.md).
- Capability questions: [Supported features](docs/user_guide/supported_features.md).

## Maintenance commands

Patch maintenance needs a real vLLM environment:

```bash
python tools/patch_manifest.py
python tools/patch_manifest.py --output /tmp/patch-report.md
python tools/check_patch_drift.py
```

The report defaults to stdout and reflects actual installation in that process
and environment. Update the drift baseline only after reviewing changed upstream
code.

Upstream kernel generation needs sailify and the manifest-pinned source:

```bash
python tools/port_upstream_kernels.py --vllm-src /path/to/vllm --check
```

Native compilation requires the SAIL SDK and SAIL PyTorch. Follow the installation
guide and use `--no-build-isolation` so the build sees the prepared environment.

## Formatting and documentation

Use the pinned pre-commit hooks on changed files; do not reformat unrelated
code. Keep public documentation focused on PPU usage and development. Internal
specifications, phase reports and personal run records belong outside the
repository. Use maintained guide paths for new cross-references.

Follow `CONTRIBUTING.md` for PR titles and complete the
[PR template](.github/pull_request_template.md).
