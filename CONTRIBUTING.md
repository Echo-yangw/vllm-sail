# Contributing to vLLM SAIL

Contributions to documentation, tests, PPU kernels and performance are welcome.
Check existing issues and pull requests before starting. Discuss substantial
API, dependency or architecture changes in an issue first.

## Development environment

The CPU test suite runs with Python 3.10–3.13 and does not require torch, vLLM,
the SAIL SDK or a device. From a checkout of this repository:

```bash
python3 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install -r requirements/dev.txt
python -m pytest tests/ut -q
```

The package is imported directly from the checkout for these tests. The expected
`vllm_sail._version is missing` warning can appear before the package is installed.
Optional integration tests skip when their dependencies are unavailable.

For native compilation or device testing, use the separate
[PPU installation instructions](docs/getting_started/installation.md).

## Coding guidelines

- Keep changes focused and describe the behavior they change.
- Prefer explicit interfaces and shared helpers when they remove real duplication.
- Preserve lazy imports and entry-point idempotency. Core imports and CPU tests
  must remain independent of optional SDK packages.
- Use the existing registration APIs and platform hooks before adding a patch.
  Read the [runtime patch guide](vllm_sail/patch/README.md) for metadata and copied
  source rules.
- Edit plugin-owned HGGC kernels directly in `csrc/plugin/`. For upstream
  kernels, edit generation inputs and regenerate; follow
  [HGGC kernel development](docs/developer_guide/kernels.md).
- Preserve upstream attribution and license headers. New Python files should
  use `# SPDX-License-Identifier: Apache-2.0`.
- Keep user documentation focused on reproducible PPU workflows. Personal
  run logs, implementation phase records and private environment notes belong
  outside the repository.

See [Architecture](docs/developer_guide/architecture.md) for module boundaries.

## Checks before review

Install the pre-commit and commit-msg hooks once with `pre-commit install`.
Every commit requires a `Signed-off-by` trailer; use `git commit -s` to add it
with your configured Git name and email.

Run the CPU tests and formatting checks:

```bash
python -m pytest tests/ut -q
pre-commit run --files path/to/changed_file.py
```

For staged files or a full repository check:

```bash
pre-commit run
pre-commit run --all-files --hook-stage manual
```

Add tests for behavior changes and keep formatting changes focused. For native
or runtime changes, follow [Verification](docs/user_guide/verification_guide.md)
and report any checks that were not run. Verify links and commands in docs.

## Pull request workflow

1. Fork the repository and create a topic branch for the change.
2. Implement the change and update the relevant tests and documentation.
3. Run the applicable checks and inspect the diff.
4. Push the topic branch to your fork and open a pull request.
5. Address review feedback; merge through the platform after review.

Use vLLM-style PR titles: `[Type][Scope] Concise behavior-oriented summary`,
with an optional scope. For example, `[Bugfix][MoE] Honor explicit backend
selection` or `[Doc] Clarify native build prerequisites`.

Complete the [PR template](.github/pull_request_template.md), link related issues
and include test commands and results. Performance changes need baseline and
candidate measurements under the same workload. Avoid force pushes and direct
pushes to shared branches; use follow-up commits for review changes.

## PPU CI commands

On an open PR, repository writers and users in `CI_TRUSTED_USERS` can request
CI for CPU and PPU checks. The PR author can also request it once the non-draft
PR has `ready` or `ready-run-all-tests`, or an approved review decision with a
trusted approval. Post one of these exact commands as a new PR comment:

| Command | Behavior |
| --- | --- |
| `/ci run` | Run CPU unit tests, source/interface checks and upstream reuse tests, build wheels, and run PPU pin unit tests and model smoke tests for the current head. |
| `/ci run --allow-stale` | Run even when the PR is behind its base branch; no automatic merge is performed. |
| `/ci retry` | Retry failed jobs in the latest run for the current head, falling back to a full rerun if GitHub rejects the partial rerun. |
| `/ci cancel` | Request cancellation of active PPU CI runs for this PR, including older heads. |

Push a new commit and post `/ci run` again to test it. Once a head has completed
CI, only repository writers can request another full run; others should use
`/ci retry` for failures. Successful builds are reused by partial retries while
their wheel artifacts remain available (14-day retention). If they have expired,
request a new build from a repository writer or push a new commit. Repository
writers may also use GitHub's native Re-run controls. Retry results update the
`PPU CI` status and the bot's original `/ci run` reply, not the `/ci retry`
acknowledgement.

The on-demand workflow is dispatched from the default branch and reports the
`PPU CI` commit status on the tested head, with links and attempt numbers. PR
code runs with read-only permissions, no inherited secrets, disabled cache
access and an explicitly verified checkout SHA. Authorization and reporting
run separately on GitHub-hosted runners using default-branch code.

CPU unit tests, vLLM source/interface checks and upstream reuse tests run through
the same commands in one CPU job and contribute to the formal `PPU CI` status.
The job runs Python 3.10–3.13 UT, source contracts and upstream reuse sequentially
in separate virtual environments. Interface and backend tests are included in
the UT/source suites. Each phase still runs after another phase fails; any
phase failure fails the job. Retrying a failed CPU job reruns the whole CPU suite.
Opening a PR or pushing a commit does not start these tests. CPU checks still
run when PPU jobs are skipped for non-impacting changes. Manually dispatch
`ut.yaml` to run the consolidated CPU suite, or `upstream-reuse.yaml` to run it
alongside PPU reuse tests. Its PPU nightly schedule is retained.
Manually dispatch PPU ops tests (`ppu-ops-tests.yaml`) in GitHub Actions for
operator tests. Dispatch PPU debug (`ppu-debug.yaml`) to rerun pin UT and/or
model tests on existing builds.

The formal command path intentionally loads its executor from the default
branch, so `/ci run` tests PR source but cannot exercise workflow definitions
changed by that PR. For changes under `.github/workflows/**` or `scripts/ci/**`,
a repository maintainer can add the `ci-preview` label to opt the PR into the
`pull_request` route in `ppu-build-and-test.yaml`. This label is separate from
`ready` and `ready-run-all-tests`: those labels only authorize the PR author to
use the trusted default-branch `/ci` executor.

The preview route uses the PR merge commit for both workflow definitions and
source, builds fresh wheels, and runs pin UT and model tests. While the
`ci-preview` label remains present, each matching push starts a new preview and
cancels the older preview for the same PR. Removing the label cancels an active
preview and skips the preview jobs. Preview results do not publish or replace
the formal `PPU CI` commit status, and `/ci retry` and `/ci cancel` do not manage
them; repository writers can use GitHub's Re-run controls instead.

The pull-request workflow definition is PR-controlled. Repository and
organization fork-run approval plus runner admission policies are the actual
trust boundary. The `ci-preview` label is an operational resource gate, not an
independent sandbox. Preview reuses the team-controlled runner and device pools
under that resource trust model.


Repository administrators must require the `PPU CI` commit status from GitHub
Actions instead of the individual PPU build, pin UT, smoke and gate job checks.
Remove the individual CPU, vLLM source/interface and CPU reuse checks from required
checks; they now contribute to `PPU CI` and are no longer automatically created
for every head. Coordinate this ruleset change with the switch to unified
on-demand CI, after verifying the commands on the default branch.

## Reporting issues

Use the bug or feature request template in the repository's Issues tab. Bug
reports should include a minimal reproducer, expected and actual results, the
full relevant error and `collect_env` output. Remove credentials, private host
addresses and confidential model paths from logs before posting.

Keep discussions respectful and focused on the work. Explain technical
disagreements with examples, measurements or source evidence.

## License

Contributions are made under the project's [Apache License 2.0](LICENSE).
Retain the applicable notices when adapting third-party code.
