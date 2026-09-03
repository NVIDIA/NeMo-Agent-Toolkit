# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Runs the framework-parity check for a single entry in ``matrix.FRAMEWORK_MATRIX``.

Intended to be invoked with the *target example's own isolated interpreter* (i.e. the
venv that `uv pip install -e <example_dir>` was run into), so that `import nat` and
`import <underlying framework>` resolve to exactly what that example depends on --
not whatever happens to be on the orchestrator's own Python path.

Usage:
    python run_check.py <key> --repo-root <path> [--live]

Run directly (not with `-m`) so Python's automatic sys.path[0] makes the sibling
`matrix` module importable regardless of the caller's working directory.

Prints one JSON object to stdout and exits 0 on pass, 1 on fail. A missing live
credential is not a failure: the entry is reported with live_status="skipped" and the
process still exits 0, since the structural tier is what CI enforces on every run.
"""

import argparse
import asyncio
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matrix import get_entry  # noqa: E402


def _framework_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_structural(repo_root: Path, config_file: Path) -> tuple[bool, str]:
    """Runs the real `nat validate` CLI against the example's config file."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"  # nat validate prints unicode check/cross marks
    proc = subprocess.run(
        [sys.executable, "-m", "nat.cli.main", "validate", "--config_file", str(config_file)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        encoding="utf-8",  # the child writes utf-8 (PYTHONIOENCODING above); decode it as such,
        errors="replace",  # not the parent's locale-preferred encoding, or checkmarks get mangled
        check=False,
    )
    if proc.returncode != 0:
        return False, (proc.stdout + proc.stderr).strip()[-4000:]
    return True, proc.stdout.strip()[-2000:]


async def _run_live(config_file: Path, question: str) -> dict:
    """
    Runs the workflow for real and inspects the emitted IntermediateStep stream.

    Returns a dict with the raw counts so the caller can decide pass/fail -- kept
    separate from the assertions themselves so a failure here always comes with the
    actual span/token counts observed, not just a bare "it didn't work".
    """
    from nat.builder.context import Context
    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.runtime.loader import load_workflow

    collected = []
    ctx = Context.get()
    subscription = ctx.intermediate_step_manager.subscribe(on_next=collected.append)

    try:
        async with load_workflow(config_file) as workflow:
            async with workflow.run(question) as runner:
                answer = await runner.result()
    finally:
        subscription.unsubscribe()

    has_workflow_start = any(s.event_type == IntermediateStepType.WORKFLOW_START for s in collected)
    has_workflow_end = any(s.event_type == IntermediateStepType.WORKFLOW_END for s in collected)
    llm_end_steps = [s for s in collected if s.event_type == IntermediateStepType.LLM_END]
    total_tokens = sum(
        s.usage_info.token_usage.total_tokens for s in llm_end_steps if s.usage_info is not None)

    return {
        "answer_preview": str(answer)[:200],
        "total_steps": len(collected),
        "has_workflow_start": has_workflow_start,
        "has_workflow_end": has_workflow_end,
        "llm_end_span_count": len(llm_end_steps),
        "total_tokens": total_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", help="Framework matrix key, e.g. 'crewai'")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--live", action="store_true", help="Attempt the live tier if credentials are present")
    args = parser.parse_args()

    entry = get_entry(args.key)
    repo_root: Path = args.repo_root.resolve()
    config_file = repo_root / entry.config_file

    result = {
        "key": entry.key,
        "display_name": entry.display_name,
        "framework_package": entry.framework_package,
        "framework_version": _framework_version(entry.framework_package),
        "structural_status": "fail",
        "structural_detail": "",
        "live_status": "skipped",
        "live_detail": "",
    }

    structural_ok, structural_detail = _run_structural(repo_root, config_file)
    result["structural_status"] = "pass" if structural_ok else "fail"
    result["structural_detail"] = structural_detail

    missing_env = [v for v in entry.required_live_env if not os.environ.get(v)]

    if args.live and entry.required_live_env and not missing_env:
        try:
            live_result = asyncio.run(_run_live(config_file, entry.question))
            ok = (live_result["has_workflow_start"] and live_result["has_workflow_end"]
                 and live_result["llm_end_span_count"] > 0 and live_result["total_tokens"] > 0)
            result["live_status"] = "pass" if ok else "fail"
            result["live_detail"] = json.dumps(live_result)
        except Exception:  # noqa: BLE001 - report the real failure, don't swallow it
            result["live_status"] = "fail"
            result["live_detail"] = traceback.format_exc()[-4000:]
    elif entry.required_live_env and missing_env:
        result["live_status"] = "skipped"
        result["live_detail"] = f"missing env vars: {', '.join(missing_env)}"
    elif not entry.required_live_env:
        result["live_status"] = "not_wired"
        result["live_detail"] = "no single-credential live path yet; see matrix.py"

    print(json.dumps(result))

    failed = result["structural_status"] == "fail" or result["live_status"] == "fail"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
