# SPDX-License-Identifier: Apache-2.0
"""Run one CPU CI phase in its own virtual environment on a shared runner."""

import argparse
import os
import subprocess
from pathlib import Path

VERSIONS = ("3.10", "3.11", "3.12", "3.13")


def run_tests(stage, work_dir, version="3.12", upstream=None):
    if stage not in {"unit", "source", "reuse"} or version not in VERSIONS:
        raise ValueError("Invalid CPU test phase or Python version")
    if stage != "unit" and (version != "3.12" or upstream is None):
        raise ValueError(
            "Source and reuse tests require Python 3.12 and upstream source"
        )
    directory = Path(work_dir).resolve() / (
        f"unit-{version}" if stage == "unit" else stage
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "VLLM_SOURCE_ROOT"}
    }
    env.update(PYTHONNOUSERSITE="1", PIP_NO_CACHE_DIR="1")
    subprocess.run(
        [f"python{version}", "-m", "venv", "--clear", str(directory)],
        check=True,
        env=env,
    )
    python = str(directory / "bin/python")

    def run(*args, extra_env=None):
        subprocess.run([python, *args], check=True, env={**env, **(extra_env or {})})

    requirements = ["-r", "requirements/dev.txt"]
    if stage == "reuse":
        requirements += ["-r", "requirements/build.txt"]
    run("-m", "pip", "install", *requirements)
    if stage != "reuse":
        run(
            "-c",
            "import importlib.util; assert all(importlib.util.find_spec(name) is None for name in ('torch', 'vllm'))",
        )
    if stage == "unit":
        run(
            "-m",
            "pytest",
            "tests/ut",
            "-q",
            "--cov=vllm_sail",
            "--cov-branch",
            "--cov-report=term-missing",
            f"--cov-report=xml:{directory / 'coverage.xml'}",
            extra_env={"COVERAGE_FILE": str(directory / ".coverage")},
        )
        return
    upstream = Path(upstream).resolve()
    env["VLLM_SOURCE_ROOT"] = str(upstream)
    if stage == "source":
        run(
            "-m",
            "pytest",
            "tests/ut",
            "-q",
            "-rs",
            "-m",
            "upstream_source",
            "--require-upstream-source",
        )
        return
    run(
        "-m",
        "pip",
        "install",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "torch",
    )
    run(
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "-e",
        str(upstream),
        extra_env={"VLLM_TARGET_DEVICE": "empty"},
    )
    run(
        "-c",
        "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)",
    )
    run(
        "-m",
        "pytest",
        "tests/upstream_reuse",
        "-q",
        "-rs",
        "-m",
        "upstream_source and not ppu",
        "--require-upstream-source",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("unit", "source", "reuse"))
    parser.add_argument("--version", choices=VERSIONS, default="3.12")
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--upstream", type=Path)
    args = parser.parse_args(argv)
    run_tests(args.stage, args.work_dir, args.version, args.upstream)


if __name__ == "__main__":
    main()
