---
name: vllm-main-upgrade
description: Update the upstream vLLM pin in vLLM SAIL through sampled commits while preserving tested checkpoint history and reviewable pull requests.
---

# Upgrade the vLLM main pin

Use this for upstream pin migrations in vLLM SAIL. Read the repository's contribution, patch, kernel and verification guides for the interfaces being changed.

## Freeze the run

Resolve the starting SAIL commit, release endpoints and target main SHA once. Keep plans and run records outside the repository. Use [sample_commits.py](scripts/sample_commits.py) with explicit start, release and frozen target refs to write a new immutable sampling plan. Sample the frozen main first-parent history using the chosen calendar timezone and intervals. Include release tag commits as mandatory endpoints even when they belong to a release branch. Later upstream commits belong to a subsequent run.

If the latest release predates the starting main pin, pass that release explicitly and sample forward from the baseline using the post-release interval; do not insert an older release checkpoint. Mandatory release endpoints apply to the selected upgrade interval.

Bind authorized source roots, node, existing build/test environments and model profiles, including each profile's ordinary request indices and one SSE request. Capture the SAIL SDK and both Torch identities. Preserve dirty, untracked and needed ignored content plus live work before source changes. Continue within already authorized bounds without repeating approvals; additional nodes, replacement environments and publication need their own scope authorization.

Establish the current pin's baseline before adapting a candidate. Use the repository's [model sanity runner](../../../scripts/eval/ppu_model_sanity.py) and [structured profiles](../../../scripts/eval/model_sanity_profiles.json) for a short launch/request/cleanup feedback loop; read [sanity.md](references/sanity.md). Freeze the selected profiles, checkpoint paths, argv, environment overrides and request budgets in the external run plan. Keep baseline failures, resource waits and transport failures distinct. Never use a container's preinstalled vLLM as evidence for the repository pin without verifying its exact revision and installation.

Record the sailify revision and SDK/runtime-selection inputs for generation checks. Reproduce the baseline with the same translator configuration before attributing generated drift to a candidate; do not overwrite the baseline corpus to hide a toolchain mismatch.

Before compiling, probe Torch and its device-availability query under the exact container execution identity. Even an empty-target source build may run that query while importing Torch's C++ extension helpers; it must return normally, but need not report a device for compilation. Preserve a working container identity rather than arbitrarily dropping to an unmapped UID. Isolate package writes with venvs and task-owned caches. When the user chooses image dependency reuse, install paired wheels with `--no-deps`, retain inherited versions and declaration conflicts, and verify real imports and behavior before qualifying that configuration.

## Advance one checkpoint

The SDK can generate kernels under its root `rtccache` directory during inference.
Inventory that mutable cache separately. If a full SDK digest changes, localize
the difference and require the cache-excluding immutable digest to match the
original exactly; do not replace the baseline merely because the new run differs.

Do not advance the formal pin until its predecessor is accepted. Independent investigation or private drafts may proceed while hardware work runs; label their evidence accordingly.

1. Inventory changed copied bodies, delegated providers, signatures, native inputs and separately cached by-value consumers. Review real source bodies; an unchanged signature or successful import does not prove compatibility.
2. Prefer registration and platform interfaces. Preserve PPU modification regions when refreshing copied upstream bodies. Regenerate native outputs, patch inventory, interface locks and reviewed drift baselines with maintained tools.
3. Add behavioral regressions at changed call boundaries using real upstream helpers where feasible. If an existing fixture becomes stale, retain its semantic assertions and preserve the failing evidence.
   Run CPU/source UT before an expensive build, then installed UT and baseline sanity before the other selected sanity profiles, sequentially on the agreed devices. Use one ordinary request for a fast development probe and `--with-stream` for checkpoint qualification. A failed baseline blocks checkpoint advancement; preserve the attempt and diagnose it without silently reducing context, disabling graphs/speculation or changing parallelism. Sanity success is a necessary early gate and does not replace device tests, model-path evidence, answer review or composite acceptance.
4. Where metadata prerequisites allow it, run the installed UT selector against the preliminary wheel before freezing the final candidate. Always set the exact checked-out VLLM_SOURCE_ROOT, including for selectors that deselect upstream_source: unmarked inventory tests also need it. If generated inventory is a prerequisite, record the deferred gate explicitly. Immediately after metadata adoption, run the complete installed UT selector against the WIP tree before freezing or committing it; a deferred gate is never a pass. Build the Rust outputs declared by that exact pin through its official source build entry. Verify the full SHA in the frontend version, reject undeclared stale modules and use separate per-attempt build/packaging directories. Build SAIL with prepared SAIL Torch; install without dependency replacement in the separate compatible Torch environment.
   Distinguish checkout compatibility tests beside installed upstream from tests
   that import the installed plugin wheel. Prove origins inside the actual
   pytest process; a separate isolated preflight does not constrain pytest's
   `pythonpath` configuration. Any import isolation must preserve strict test
   doubles, parent-package search paths and missing-module exception identities.
5. Review captured runtime provider bodies before updating fingerprints. After adopting real generated metadata, regenerate the interface lock and run the complete WIP CPU/source/generation gates again. Assert every patch declaration agrees with the supported release range. Then freeze and independently review the complete tree, save a clean signed checkpoint commit and rerun its CPU/source/generation gates. Build its clean SAIL wheel and perform all gates in [acceptance.md](references/acceptance.md).
6. Accept only after the composite evidence check succeeds. Reverify the accepted raw evidence with [verify_checkpoint.py](scripts/verify_checkpoint.py) and a frozen external schema-2 policy; read [evidence-policy.md](references/evidence-policy.md) for its input contract. Collect model commit/wheel identities at execution time and the complete real-Torch schema parsing receipt before final postproof. Preserve failed attempts and supplements separately. Record acceptance before preparing the next checkpoint.

Reserve each attempt directory atomically. Check both source trees' exact HEAD
and tracked/untracked cleanliness before and after source gates. Wait for any
asynchronous log writer and propagate its failure before publishing a successful
stage receipt; preserve each earlier executor version and attempt.

Read commit identities from the live repositories when producing each source
receipt, and bind the actual producer hash and required ignored inputs. A
hardcoded expected SHA alone is not a live identity observation. Consumers of
stage exit files must wait for a complete status line, or use an atomically
published receipt; file existence alone can race the writer.

An isolated CPU container can verify wheel bytes, schema parsing and native
imports while its installed-vLLM tests still require driver metadata during
collection. Preserve such environment failures and rerun the complete selector
in the prepared device environment; do not fabricate NVML or add mode skips.

Capture conditional aliases in both cold and preloaded consumer processes when the generated target set changes. Resolve the captured callable's actual provider, including inherited methods, compilation decorators and unavailable-backend fallbacks, before assigning a source fingerprint. Read the provider guidance in [acceptance.md](references/acceptance.md).

Recheck release-branch API ownership at mandatory off-main endpoints; do not assume main and the release tag provide the same helper. Audit real-environment fixtures as well as production adapters: a bare-CPU or source-only pass can skip a fixture that imports the real provider. Exercise that provider in the compatible Torch environment, preserve a focused failing case and retain all behavioral assertions when adapting the fixture. At a release-range transition update patch declarations, coarse compatibility policy, tests and maintained user guides together. The full-SHA pin remains authoritative.

For source relocations, represent deleted tracked files separately from present and new files. Preserve their original Git bytes, file modes and the patch before cleanup; [owned_source.py](scripts/owned_source.py) supplies these schema-2 captures. [source_transition.py](scripts/source_transition.py) verifies backup bytes and modes before restoring an exact reviewed snapshot. Recapture older snapshots that lack modes before cleanup. Prove target-added ignored paths cannot collide before using Git no-overwrite-ignore for the final switch. Do not rebuild a generated directory over unowned files. Distinguish intentional older-layout fixtures from current-corpus assertions.

## Package and publish

Keep upstream follow-up candidates separate from plugin fixes. Preserve each accepted pin as a checkoutable commit. Publish only to owned topic branches and create the requested PR sequence; do not merge without authorization. State exact checkpoint ranges and validation actually completed.

Bind model requests, metrics and shutdown logs to the same final wheel pair. Review both pre-shutdown and post-shutdown ERROR counts; a runner exit of zero does not waive either. Distinguish abort ordering from draining and retain genuine failure reporting in regressions.

After a completed milestone, reconcile this workflow with observed failures and validate reusable scripts. Keep machine paths, run logs and attempt receipts outside the project skill.
