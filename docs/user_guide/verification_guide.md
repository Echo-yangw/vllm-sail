# Verification Guide

Validate the same package installation, SDK and hardware that will run the
workload. Keep the results of each check separate:

| Check | What it establishes |
| --- | --- |
| CPU unit tests | Python behavior, registration contracts and packaging invariants |
| Source regeneration checks | Generated upstream kernels match their declared inputs |
| HGGC build | Selected sources compile and link for the requested targets |
| Extension imports and strict preflight | Libraries load and representative operations are registered |
| Device correctness tests | Tested kernels match their numerical references on that device |
| Model run | The tested checkpoint and configuration complete inference |
| Benchmark | Performance under the recorded workload and environment |

Passing one check does not imply the later checks pass. A skipped test provides
no execution evidence for that feature.

## 1. Record the environment

After following [Installation](../getting_started/installation.md), run:

```bash
git rev-parse HEAD
python -c "from vllm_sail import collect_env; collect_env()"
python -c "import vllm_sail; print(vllm_sail.__file__)"
```

Record the PPU model and count, SDK version, SAIL PyTorch version, vLLM revision
and plugin revision. Confirm the printed package path points to the installation
being tested. A source checkout can shadow a separately installed native wheel.

Run the four-library import and strict preflight commands in
[Check the installation](../getting_started/installation.md#4-check-the-installation).
Resolve shared-library errors before running a model.

## 2. Run unit tests

From the plugin checkout, using its test environment:

```bash
python -m pytest tests/ut -q
```

The merge gate runs with only `requirements/dev.txt` installed, without torch,
vLLM or a device. Additional integration tests become available in a PPU
environment with their dependencies installed. Investigate failures and report
skip reasons; do not compare test counts from different environments as if they
cover the same paths.

Compatibility UT may intentionally import the plugin checkout beside installed
vLLM and torch. Record those import paths and set `VLLM_SOURCE_ROOT` to the same
exact upstream revision even when source-marked cases are deselected. This UT
mode does not verify the installed plugin wheel: separately compare its full
package contents with the wheel and run native imports, device tests and models
from that verified installation.

### Check upstream source compatibility

Tests live with the feature they cover, independently of the vLLM release that
introduced them. Tests marked `upstream_source` read a selected vLLM checkout
without importing vLLM, torch or the SDK:

```bash
VLLM_SOURCE_ROOT=/path/to/vllm python -m pytest tests/ut -q -rs \
  -m upstream_source --require-upstream-source
```

`VLLM_SOURCE_ROOT` names the checkout root containing `vllm/__init__.py`.
Without it, ordinary CPU runs skip source-dependent checks. The required-source
option makes a missing or invalid checkout an error, so CI cannot silently
skip these checks. The source tests cover selected upstream call sites, keyword
compatibility, quantization behavior and imported consumer inventories. They do
not establish that module imports, patch installation or device inference work.

### CI jobs

The [unit-test workflow](../../.github/workflows/ut.yaml) runs for pull requests,
pushes to `main` or `master`, and manual dispatch. Its jobs run independently:

| Job | Environment and checks | Failure handling |
| --- | --- | --- |
| CPU UT | Python 3.10–3.13; only `requirements/dev.txt`; UT and branch coverage | Test failure fails the workflow; coverage-upload failure does not |
| CI control contracts | Python 3.12；独立运行 Preview 控制工具的标准库测试、现有算子测试定义校验和 shell 语法检查 | 不分配 PPU；不访问 Registry 或发布资源；任一校验失败则工作流失败 |
| vLLM source (main-verified) | Python 3.12; checks out the exact verified main commit; runs `upstream_source` tests with a required source path | Checkout or test failure fails the workflow |

The source job reads the full SHA in
[verified main commit](../../.github/vllm-main-verified.commit) and logs the
resolved commit. Update this pin only after qualifying that exact revision.
Keep feature tests named after their behavior. The workflow does not install
vLLM or run PPU workloads; installed compatibility and device checks require
the prepared environment below. Repository branch protection determines which
job results are required for merging.

The separate [upstream canary](../../.github/workflows/upstream-canary.yaml) runs
weekly on Tuesday at 03:17 UTC and on manual dispatch. It installs upstream
`main` and checks patch drift. A drift failure opens or updates the tracking
issue and fails the job. Installation failure stops before that report step.
This scheduled check does not replace PR checks, and none of these public CPU
jobs runs PPU inference or `tests/e2e`.

### Preview 镜像与最小回归

Preview 是独立的镜像验证与分发通道，不替代正式 `PPU CI`、模型精度、性能、
双芯片算子验证或 `release.yaml`。它不会修改
[verified main commit](../../.github/vllm-main-verified.commit)，也不会把最小回归
结果升级为完整软件栈认证；KMD 和 firmware 仍标记为 `not_measured`。

| 工作流 | 输入与输出 |
| --- | --- |
| [Nightly regression (minimum)](../../.github/workflows/nightly-tests.yaml) | 每日 UTC 00:17 或默认分支手动触发；下载 `vllm-project/vllm-sail` 默认分支 `main` 的 `ppu-accuracy-nightly.yaml` 成功产物，组装并测试 candidate 镜像 |
| [Distribute nightly previews](../../.github/workflows/promote-nightly.yaml) | 成功 nightly 的 `workflow_run` 或默认分支手动指定 `test_run_id`；复核当前 attempt 的 passed bundle，分发原始 wheel 和相同镜像 digest |
| [Dockerfile CI](../../.github/workflows/dockerfile-ci.yaml) | PR 自动运行 hosted 静态检查；受信任维护者从默认分支手动申请镜像构建及单卡 smoke，单独报告 `dockerfile-ppu` |

来源必须是整个工作流成功的 `schedule` 或 `workflow_dispatch` 运行，而不仅是
wheel 构建 job 成功。来源运行创建时间必须在 48 小时内，且
`ppu-wheels-<run_id>` 未过期。缺少符合条件的产物会失败，不会退回源码编译、
PR preview 或失败运行。精度任务较长，定时时间不保证上一轮已经完成；必要时手动
选择合格运行，不能把缺失或过期的产物当作成功。

三个提交身份分别记录：默认分支控制脚本的 control SHA、wheel 的 SAIL SHA、
wheel 的上游 vLLM SHA。测试源码按 wheel 的 SAIL SHA 检出；上游 vLLM SHA 必须
匹配控制 checkout 的 reviewed pin。测试旧产物不代表覆盖最新 `main`。

[Dockerfile.ci](../../docker/Dockerfile.ci) 仅将两个原始 wheel 安装到指定运行镜像，
使用 `--no-deps --force-reinstall` 并执行 `pip check`，不下载模型或编译 vLLM。
测试阶段先校验已安装 Python/native 文件与原始 wheel 的字节及导入路径，再按
[固定目录](../../.github/ci/tests/catalog.json) 执行 2 项 patch UT 和 9 项
`silu_and_mul` 数值测试，使用一张真武 M890。JUnit 和节点报告必须证明全部
11 项正常执行；skip 不算通过。扩展目录需要同步修改执行与证据校验，不能只增加 JSON 项。

#### 管理员配置与首次启用

默认不提供真实镜像地址或凭证。未配置启用变量时，hosted preflight 失败，
下游自托管任务不会分配。先将控制文件通过 PR 合入受保护默认分支，再配置：

| Actions Variable | 要求 |
| --- | --- |
| `PPU_TRANSITION_ENABLED` | runner 负责人批准持久化主机用途后设为 `true` |
| `PREVIEW_REGISTRY_HOST` | Registry 主机名，可带端口；不含协议或命名空间 |
| `PREVIEW_IMAGE_REPOSITORY` | 该 Registry 下专用的无 tag Preview 镜像路径 |
| `RUNTIME_BASE_IMAGE` | 可读取的 `repository@sha256:<64 hex>`；不能与 Preview 输出仓库相同 |
| `TRUSTED_PPU_TESTS_ENABLED` | 首次 nightly/分发验证完成且受信任 PR 执行获准后才设为 `true` |

`RUNTIME_BASE_IMAGE` 必须已包含匹配的 SAIL SDK、SAIL PyTorch、CUDA-compatible
运行依赖、Triton，以及所测源码 `requirements/dev.txt` 的依赖（包括 pytest）。
现有 runner 的 `PPU_CI_RUNTIME_IMAGE` 不会自动传入此变量。工作流只登录
`PREVIEW_REGISTRY_HOST`，其他私有 Registry 的 base image 访问需另行安排。

创建以下 Environments，并限制到默认分支；声明 `environment:` 不会自动安装审批规则：

| Environment | Secrets 与保护 |
| --- | --- |
| `nightly-registry` | `REGISTRY_USER`、`REGISTRY_TOKEN`：Preview push 权限；跨仓库读取时可配置 `UPSTREAM_ACTIONS_TOKEN` |
| `nightly-test` | `REGISTRY_PULL_USER`、`REGISTRY_PULL_TOKEN`：只读拉取权限 |
| `preview-distribution` | `REGISTRY_USER`、`REGISTRY_TOKEN`：仅 Preview tagging/push 权限 |
| `dockerfile-ci` | `REGISTRY_PULL_USER`、`REGISTRY_PULL_TOKEN`；配置 required reviewers，并在支持时禁止自行批准 |

在本仓库消费本仓库产物通常使用 `GITHUB_TOKEN` 即可。跨仓库 token 需能读取来源
Actions 产物及本地 Preview Release。Registry 写凭证只放在对应 Environment，
不向 PR 设备任务或测试容器传入。保护工作流文件、runner group 与默认分支；
Environment 审批不是恶意代码隔离机制，不能用于放行外部不受信任 PR。

设备 runner 标签要求 `[self-hosted, Linux, X64, 8-ppu-m890p]`，并需满足：

- Linux x86-64、Bash、Git、Python 3.11+、Docker/runc、flock、coreutils。
- GitHub/API/Actions 产物及 Registry 连通性、足够磁盘空间。
- 可用 PPU 驱动及 `/dev/alixpu`、`/dev/alixpu_ctl`、`/dev/alixpu_sep`、
  `/dev/alixpu_ppu0`，runner 用户和容器用户具备设备访问权限。
- 协调所有共用设备的任务；标签本身不是独占设备预留。

新脚本沿用 `/tmp/vllm-sail-ppu-ci.lock`，最多等待 3600 秒；现有 PR runtime
仍使用 `flock -n`，遇到占锁会失败。上线前应安排独立 runner 或无冲突时段，
若统一锁策略，应单独调整所有调用方及超时。该路线不是 Kubernetes 隔离方案。

首次启用按以下顺序验证：

1. 先运行 hosted 控制测试，配置 base image、Registry、Environments 和 runner。
2. 批准并启用 `PPU_TRANSITION_ENABLED`，手动运行 nightly；可填写合格的来源 run ID。
3. 确认 candidate 镜像、11 项实际执行结果及当前 attempt 的 passed bundle。
4. 确认自动分发后的镜像 digest 和 wheel SHA256 与被测值一致。
5. 再启用受信任 Dockerfile PR 测试并观察后续定时运行。

Dockerfile 手动测试要求开放、非草稿、同仓库且目标为默认分支的 PR；触发者和
作者都必须具有仓库写权限。申请时勾选 `approve_host_execution`，审批后再次
校验 head/base SHA；发生变化必须重新申请。测试的是 PR 的 `docker/` 与已有
Preview wheel，不是该 PR 修改后的 SAIL 源码。`dockerfile-static` 成功不代表
`dockerfile-ppu` 成功，也不能代替正式 `PPU CI`。取消后若留下 pending status，
仍应视为未通过；手动重新验证。按路径触发的检查不宜直接设为所有 PR 的必需状态。

#### 分发、重试与保留策略

发布器只维护以下 Preview 命名空间：

- 镜像 tag：`preview-<upstream_run_id>-test-<local_test_run_id>`，不重建被测镜像。
- 不可变 prerelease：`preview-nightly-<upstream_run_id>-test-<local_test_run_id>`。
- 固定 prerelease：`preview-baseline`，包含 `preview-wheels.zip` 和
  `preview-baseline.json`，保留原始 wheel 版本与元数据。

正式发布的 push 触发器与 `setuptools-scm` 版本推导均排除 `preview-*`，
避免 Preview run ID 被解释为包版本。最小回归不更新稳定 pin、BOM、
`ci-baseline` 或正式 Release。每日可以重测 48 小时内已分发的产物，但相同或更旧
来源 run 不会推进 Preview 指针；只改 Dockerfile 后重测旧 wheel 也不会自动推进。

固定 Release 的两份资源不是原子更新。发布器串行运行，先上传 wheel，再写 JSON
标记；读取者遇到缺失或哈希不一致会拒绝使用。中断后从已校验的不可变 Preview
资源恢复，旧资源以归档重命名保留。不要同时启用其他发布器写入同一指针。
回退应由维护者暂停发布器后，选择历史不可变 Preview 的 wheel 和 digest。

Actions 产物保留 14 天；没有自动清理 Registry、Release 或持久化工作目录。
上线时配置独立的保留与磁盘监控策略。

本地控制验证不需要 torch、vLLM、设备或网络。在项目根目录使用 Python 3.11+：

```bash
python3.12 -B -m unittest discover -s tests -p 'test_nightly_assets.py' -q
python3.12 -B -m unittest discover -s tests -p 'test_deployment.py' -q
python3.12 -B -m unittest discover -s tests -p 'test_verify_installed.py' -q
python3.12 -B -m unittest discover -s tests -p 'test_workflow_contracts.py' -q
python3.12 -B scripts/ci/deployment.py validate-test-source --source .
bash -n scripts/ci/run_existing_tests.sh
bash -n scripts/ci/ppu_smoke.sh
```

## 3. Run device correctness tests

In the PPU environment, install `requirements/dev.txt`. Use an editable native
plugin build in the checkout, or ensure the test process imports the installed
wheel as described above. Start with activation and the affected feature:

```bash
pytest tests/e2e/test_native_activation.py -q -rs
pytest tests/e2e/test_deepseek_v4_native.py -q -rs
pytest tests/e2e/test_mxfp4_prepare.py -q -rs
pytest tests/e2e/tier_b/test_norm_quant.py -q -rs
```

The [`tests/e2e`](../../tests/e2e) directory also contains linear-attention,
indexer, cache and model-specific kernel tests. Select the relevant tests for
the change; run the full suite when qualifying an environment:

```bash
pytest tests/e2e -q -rs
```

These tests require a real PPU and the applicable SDK libraries. Verify that the
intended tests actually ran. Include dtype, shape, tolerance and device in a
correctness report, and test both architecture targets when claiming both.

Capability markers restrict tests to one chip: `cap80` runs only on compute
capability 8.0 (Zhenwu 810E) and `cap89` only on 8.9 (Zhenwu M890); other
devices skip them. `perf` marks performance benchmarks and `model_e2e` marks
offline model end-to-end tests. Select or exclude them with `-m`, for example
`pytest tests/e2e -q -rs -m "not perf"`.

### GDN fused MTP decode

Run the native GDN MTP accuracy matrix with a native plugin build:

```bash
python -m pytest tests/e2e/gdn/test_gdn_fused_mtp.py -q -rs \
  -o junit_family=xunit1 --junitxml=gdn-mtp-accuracy.xml
```

The tests call the native fused op directly and force the unfused reference onto
Triton, bypassing PLA. Selected cases also compare with an independent eager
PyTorch reference. On PPU, a missing native dispatch registration fails the
suite rather than skipping it.

Coverage includes all supported head ratios, BF16/FP32 recurrent state and norm
weights, FP16/BF16/FP32 gate bias, SiLU/sigmoid output gates, MTP lengths 1–8,
strided rows and state slots, padding and NULL slots, and repeated accepted-token
rewind with changing request lengths. Every round checks outputs, candidate
state writes, unused state storage and output-buffer guards. The suite also
tests numerical boundaries and the layer's post-convolution entry point.

The native-versus-Triton output keeps upstream's relative-L2 limit of `5e-4`;
pointwise output/state comparisons retain the upstream model-path tolerances
`atol=rtol=3e-2`. Each written slot additionally has a relative-L2 limit of `3e-2`.
Untouched slots and buffer guards must remain bitwise identical. The JUnit file
records relative-L2 and maximum absolute errors per round. These limits are
source-derived, not calibrated from a PPU run; investigate failures before
changing them.

The independent oracle has worked-example CPU tests that run when torch is
available:

```bash
python -m pytest tests/ut/test_gdn_mtp_reference.py -q
```

CPU oracle tests and device-test collection do not establish native accuracy.
Record actual execution counts and skips from the PPU command above.

## 4. Run a model

### Quick serving sanity

The standalone sanity runner launches one installed model, checks readiness
against a unique served-model identity, sends a deterministic arithmetic request,
checks its complete final answer and stops its owned server process group:

```bash
python scripts/eval/ppu_model_sanity.py --list
python scripts/eval/ppu_model_sanity.py \
  --profile deepseek-v4-community-tp8 \
  --checkpoint-root /path/to/datasets \
  --output /absolute/new-sanity-attempt \
  --with-stream
```

Profiles cover community DeepSeek V4 Flash 0731 TP8, its MXFP4/FP8-channel TP8
and DP8/EP8 low-latency variants, GLM 5.2 MXFP4/FP8 TP8, and MiniMax M3
FP8-channel TP8. Checkpoint suffixes and serving arguments are recorded in
[`model_sanity_profiles.json`](../../scripts/eval/model_sanity_profiles.json).
The dataset root defaults to `/ppusw/datasets`; set it to the actual mounted
root. Each attempt requires a new output directory. Run profiles sequentially
on free devices in the selected environment, preserving existing work.

The default sends one ordinary request; `--with-stream` adds a streamed request.
Both must return the final answer `42` and finish naturally. Truncated output,
reasoning without a final answer, a wrong answer, HTTP failure, serving or
shutdown ERROR logs, and failed cleanup fail the attempt. The process exits
nonzero on failure and retains its invocation, installed package metadata,
responses, metrics, serving log and result. Adjust readiness, request and token
budgets explicitly with `--ready-timeout`, `--request-timeout`, and `--max-tokens`.
Each request budget covers connection setup, response headers and the complete
body, including a streamed response that never finishes a line. Expiration
fails the attempt and enters service cleanup; cleanup has its own bounded wait.

This is a short inference check for the selected configuration. It does not
replace the accuracy evaluation below, kernel correctness checks, package-byte
verification or evidence that every configured optimization was active.

### Model verification

Follow the [quickstart](../getting_started/quickstart.md) with a checkpoint
appropriate for the available memory and supported precision. Confirm that:

- The PPU platform and intended compute backend are selected.
- Model loading and warm-up complete.
- The readiness endpoint succeeds and generation returns sensible output.
- Repeated generation and the intended context lengths complete successfully.

For MoE, quantized or multi-device workloads, repeat the check using the intended
expert backend, checkpoint layout and parallelism. A small dense-model smoke
test does not cover those paths.

## 5. Report correctness and performance

Include the exact command, checkpoint identifier and revision, dtype,
quantization, context length, device count and parallelism. For benchmarks, also
include the workload or dataset, input/output lengths, concurrency, warm-up,
throughput and latency measurements.

Useful tools include vLLM's `vllm bench --help` and the repository's
[`benchmarks/kernels`](../../benchmarks/kernels). Compare baseline and candidate
with the same software, hardware and workload. Explain the numerical reference
and tolerance used for kernel comparisons.

Report which checks passed, failed, skipped or were not run. Attach concise
logs and a minimal reproducer through the repository's issue or pull request
template; remove credentials and private model or host details before posting.
