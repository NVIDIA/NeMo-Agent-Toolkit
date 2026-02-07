#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

from dotenv import dotenv_values
from dotenv import find_dotenv

REPO = Path(__file__).resolve().parents[2]
ART = REPO / ".artifacts"
JUNIT_DIR = ART / "junit"
COV_DIR = ART / "coverage"
VENV_DIR = ART / "venvs"
MAX_PROJECT_DEPTH = 5
SKIP_DIRS = {"__pycache__", "node_modules"}


def sh(cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    return subprocess.run(cmd, check=False, cwd=REPO, env=env).returncode


def slug(path: Path) -> str:
    rel = path.relative_to(REPO).as_posix()
    return re.sub(r"[^A-Za-z0-9._-]+", "__", rel).strip("_")


def discover_projects(max_depth: int = MAX_PROJECT_DEPTH) -> list[Path]:
    projects: list[Path] = []
    locations = [REPO / "packages", REPO / "examples"]
    for location in locations:
        if location.exists():
            curr_projects = []
            for root, dirs, files in os.walk(location, topdown=True):
                rel_depth = len(Path(root).relative_to(location).parts)
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
                if rel_depth >= max_depth:
                    dirs[:] = []
                if "pyproject.toml" in files:
                    curr_projects.append(Path(root))
            projects.extend(sorted(curr_projects))
    return projects


def make_env() -> dict[str, str]:
    env = os.environ.copy()

    # One environment per worker process (runner), reused across projects handled by that process.
    venv_path = VENV_DIR / f"pid-{os.getpid()}"
    env["UV_PROJECT_ENVIRONMENT"] = str(venv_path)

    if env_values := dotenv_values():
        env.update({k: v for k, v in env_values.items() if v is not None})

    # Optional: keep for downstream tooling that reads it.
    # Note: uv itself primarily uses --env-file/UV_ENV_FILE for `uv run`-spawned commands.
    if dotenv_path := find_dotenv():
        env["UV_ENV_FILE"] = dotenv_path

    return env


def run_one(
    project_dir: Path,
    *,
    enable_coverage: bool,
    enable_junit: bool,
    run_slow: bool,
    run_integration: bool,
) -> int:
    logger = logging.getLogger("testing")
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler(sys.stdout))

    env = make_env()

    display_project_dir = project_dir.relative_to(REPO).as_posix()

    name = slug(project_dir)
    junit = JUNIT_DIR / f"{name}.xml"
    covfile = COV_DIR / f".coverage.{name}"
    env["COVERAGE_FILE"] = str(covfile)

    # 1) Sync exact environment for this project into the worker’s venv.
    # uv sync is exact by default for the project environment (removes extraneous packages).
    cmd = [
        "uv",
        "sync",
        "--active",
        "-q",
        "--project",
        str(project_dir),
        "--all-groups",
        "--all-extras",
        "--no-progress",
    ]
    if rc := sh(cmd, env=env):
        logger.error(f"{display_project_dir} (sync failed)")
        return rc
    else:
        logger.info(f"{display_project_dir} (synced)")

    if not (project_dir / "tests").exists():
        logger.info(f"{display_project_dir} (no tests)")
        return 0

    # 2) Run pytest in that environment.
    cmd = ["uv", "run", "--active", "--", "pytest", "-q", str(project_dir)]
    if run_slow:
        cmd.append("--run_slow")
    if run_integration:
        cmd.append("--run_integration")
    if enable_junit:
        cmd.append(f"--junitxml={junit}")
    if enable_coverage:
        # always include nat module in the coverage report
        cmd.append("--cov=nat")
        # if the project has a src directory, include it in the coverage report
        source_dir = project_dir / "src"
        if source_dir.exists():
            cmd.append(f"--cov={str(source_dir)}")
        cmd.append("--cov-report=")

    if rc := sh(cmd, env=env):
        logger.error(f"{display_project_dir} (test failed)")
        return rc
    else:
        logger.info(f"{display_project_dir} (tested)")
    return 0


def main(junit_xml: str | None, cov_xml: str | None, run_slow: bool, run_integration: bool, jobs: int) -> int:
    projects = discover_projects()
    if not projects:
        print("No projects found under packages/ or examples/")
        return 2

    for d in (ART, JUNIT_DIR, COV_DIR, VENV_DIR):
        d.mkdir(parents=True, exist_ok=True)

    failures = 0

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        ex = executor

        def shutdown_executor(signum, frame):
            nonlocal ex
            if ex is not None:
                print("Shutting down executor...")
                ex.shutdown(wait=False, cancel_futures=True)
            else:
                print("Executor not found")

        signal.signal(signal.SIGINT, shutdown_executor)
        futs = [
            ex.submit(run_one,
                      p,
                      enable_coverage=cov_xml is not None,
                      enable_junit=junit_xml is not None,
                      run_slow=run_slow,
                      run_integration=run_integration) for p in projects
        ]
        try:
            for fut in as_completed(futs):
                if fut.result() != 0:
                    failures += 1
        finally:
            ex = None

    if cov_xml is not None:
        sh(["uv", "tool", "install", "coverage[toml]"])
        sh(["coverage", "combine", "--keep", str(COV_DIR)])
        sh(["coverage", "xml", "-o", str(cov_xml)])
        sh(["coverage", "report"])

    if junit_xml is not None:
        sh(["uv", "tool", "install", "junitparser"])
        sh(["junitparser", "merge", "--glob", str(JUNIT_DIR / "*.xml"), str(junit_xml)])

    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit_xml", action="store", default=None)
    parser.add_argument("--cov_xml", action="store", default=None)
    parser.add_argument("--run_slow", action="store_true", default=False)
    parser.add_argument("--run_integration", action="store_true", default=False)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(
        main(junit_xml=args.junit_xml,
             cov_xml=args.cov_xml,
             run_slow=args.run_slow,
             run_integration=args.run_integration,
             jobs=args.jobs))
