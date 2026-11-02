# Checkpoint evidence contract

Every required receipt binds the same clean SAIL commit, exact upstream SHA and final wheel pair. Record producer/input identities, raw artifact hashes and exit status. Preserve original logs/XML with counts.

## CPU and source

- Run full tests/ut on Python3.10-3.13 with only development requirements; verify no Torch or vLLM is available.
- Run required exact-source tests. Bind XML/logs, actual clean commit pair, source root and verifier; WIP counts cannot qualify a later clean commit.
- Check native regeneration and pinned hooks on changed files. Generate inventory and locks through maintained tools. After actual metadata adoption run the complete WIP gates before freezing, then all clean gates after committing. Keep new patch ranges, generated inventory and the official interface lock mutually consistent.
- Review compatibility and standards independently, binding actual diff and provider/file hashes.
- Resolve the callable actually captured by runtime metadata. A class method can capture a nested decorator or inherited provider; unavailable backends can expose a fallback. Compare full bodies and actual import edges, including cached consumers. A target count change requires inspection of cold and preloaded registry/consumer identities, rather than forcing the previous count.

## Wheels and installed tests

- Verify declared Rust outputs, frontend full-SHA/version, four SAIL native extensions and wheel/source identities. A dirty preliminary wheel may supply metadata but cannot replace the clean checkpoint wheel.
- Parse every generated excluded-op schema with the installed compatible Torch before final qualification. Derive that count from the generated resource and bind its full file hash, parser producer/log, Torch identity, exact commit pair and final wheel digests in the schema-2 parsing receipt. The copied resource must match `vllm_sail/native/data/excluded_ops.toml` in the installed-file map. Bind the parsing receipt's hash into final postproof and acceptance; compiled operation claims are a separate inventory. Source extraction regressions should exercise porting, C++ literal decoding, pruning, rendering and the actual resource reader. Compilation, native imports and a lossless TOML round trip alone do not verify schema registration.
- Verify imports resolve to installed packages. Compare every package file with wheel bytes; retain both complete maps, including native/Rust outputs. Reject undeclared obsolete bridges.
- After generated metadata becomes usable, run full installed WIP UT before the final freeze or clean commit. Bind its exact test identities and fixed mode exclusions; a focused regression pass alone does not meet this prerequisite.
- At release-branch endpoints, verify real-provider fixtures against the exact API owner. Adapt stale setup to the actual loader or factory while preserving tensor identity, slicing, padding and filtering assertions. Do not create absent helpers or add skips to hide API mismatches.
- Run all installed UT with an explicit exact-source VLLM_SOURCE_ROOT. Do not silently turn an avoidable missing-source skip into a mode exclusion. Distinguish known mode exclusions and source-only deselection from passing tests.
- Recheck fixed SDK and Torch identities before/after installation and verification. Track allowed additive caches separately from immutable files.

## Device and models

Run every applicable device E2E test in the agreed profile. Hardware skips are not runtime evidence. Expected failures must be agreed for that environment and freshly reproduced at each pin with raw causes retained. Bind causes to the exact commits and device XML hash. Partition the actual excluded case IDs into disjoint expected-failure and hardware-inapplicable sets; matching totals alone are insufficient. Unexpected failures block acceptance.

For a focused regression gate, compare the actual XML case identities with the
frozen selection and reject unexpected skips, failures and errors. A zero pytest
exit can also mean that every selected regression was skipped.

For every agreed model retain exact invocation/checkpoint/versions, readiness and elapsed timings; every correct complete final answer for the policy's request indices and one SSE answer with natural finish; positive configured quantization/parallel/attention/graph/speculation paths; generation/speculation metrics, memory, capture completion and error logs; owned handles, cleanup result and final absence of live owned services. Record commit and wheel identities in the raw invocation/result, complete-answer receipt, manual audit and per-model postproof. Capture these identities during the run, rather than labeling an old model directory from the final summary.

HTTP success, runner status and keyword matches alone do not qualify answers. Review the complete answers against the prompts. Preserve truncated attempts and supplement affected cases within the authorized protocol; do not silently change service/model settings. Name the qualifying receipt for each case. Preserve shutdown warnings rather than calling warningful shutdown entirely clean.

Wait when other tasks occupy devices unless the human explicitly authorizes resource termination on the named node. With that authority, verify fresh same-namespace PID/start-time/argv/cgroup identities and pidfds before signalling; exclude owned services and infrastructure. Preserve containers/data and record actual signals. Do not extend authority to other nodes. Resource waiting is distinct from test failure.

## Controllers and acceptance

Publish host success only after its stage receipt is durable. Container startup, missing receipts and receipt-write failures must propagate. Exercise slow/failing writes and downstream failure paths with local fixtures; synthetic receipts are not runtime proof.

Dispatch a dependent stage only after its prerequisite runtime directory and
successful preparation receipt exist. An asynchronous executor's successful
launch does not establish that the child started or completed; retain failed
dispatches separately when no child receipt was created.

The final verifier requires CPU/source/generation reviews, installed UT, device results/exclusions, every model's reviewed receipt, complete installed maps, native/Rust provenance, stack invariants and owned cleanup. Exercise its complete late path and missing/tampered evidence cases. Write acceptance only after it passes, preserving earlier records.

A Rust wheel may be reused only for the same exact upstream commit/tree and verified source-build provenance; a new pin requires its own official source compilation. A maintenance commit needs its own agreed qualification at the same pin: an earlier accepted SAIL commit and wheel pair do not qualify a later commit.
