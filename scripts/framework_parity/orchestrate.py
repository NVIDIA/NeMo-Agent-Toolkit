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
Drives the full framework-parity matrix: for each entry, creates a throwaway `uv`
venv, installs that example's own package into it (exactly as the example's own
README tells a user to), and runs `run_check.py` inside that venv's interpreter.

This is the piece that makes the check honest: every framework gets its own
environment, so a dependency pulled in by the CrewAI example can't silently make the
ADK example's import succeed (or fail) for the wrong reason.

Usage:
    python scripts/framework_parity/orchestrate.py [--live] [--only KEY [KEY ...]]

Writes one <repo_root>/.framework_parity_results/<key>.json per entry and a summary
to stdout. Exits 1 if any entry's structural or (attempted) live tier failed.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matrix import FRAMEWORK_MATRIX  # noqa: E402
from matrix import FrameworkEntry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / ".framework_parity_results"


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_entry(entry: FrameworkEntry, live: bool, keep_venvs: bool) -> dict:
    print(f"== {entry.display_name} ({entry.key}) ==", file=sys.stderr)

    work_dir = Path(tempfile.mkdtemp(prefix=f"nat_parity_{entry.key}_"))
    venv_dir = work_dir / "venv"

    try:
        subprocess.run(["uv", "venv", str(venv_dir)], check=True, cwd=REPO_ROOT)
        python = _venv_python(venv_dir)

        install = subprocess.run(
            ["uv", "pip", "install", "-e", entry.example_dir, "--python", str(python)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if install.returncode != 0:
            return {
                "key": entry.key,
                "display_name": entry.display_name,
                "framework_package": entry.framework_package,
                "framework_version": None,
                "structural_status": "fail",
                "structural_detail": f"uv pip install failed:\n{install.stdout}\n{install.stderr}"[-4000:],
                "live_status": "skipped",
                "live_detail": "install failed before the live tier could run",
            }

        check_cmd = [str(python), str(Path(__file__).resolve().parent / "run_check.py"), entry.key, "--repo-root",
                    str(REPO_ROOT)]
        if live:
            check_cmd.append("--live")

        check = subprocess.run(check_cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        if not check.stdout.strip():
            return {
                "key": entry.key,
                "display_name": entry.display_name,
                "framework_package": entry.framework_package,
                "framework_version": None,
                "structural_status": "fail",
                "structural_detail": f"run_check.py produced no output:\n{check.stderr}"[-4000:],
                "live_status": "skipped",
                "live_detail": "",
            }
        return json.loads(check.stdout.strip().splitlines()[-1])
    finally:
        if not keep_venvs:
            shutil.rmtree(work_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Attempt the live tier for entries that support it")
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to these matrix keys")
    parser.add_argument("--keep-venvs", action="store_true", help="Don't delete the per-framework venvs (debugging)")
    args = parser.parse_args()

    entries = [e for e in FRAMEWORK_MATRIX if args.only is None or e.key in args.only]

    RESULTS_DIR.mkdir(exist_ok=True)
    results = []
    for entry in entries:
        result = run_entry(entry, live=args.live, keep_venvs=args.keep_venvs)
        results.append(result)
        (RESULTS_DIR / f"{entry.key}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))

    any_failed = any(r["structural_status"] == "fail" or r["live_status"] == "fail" for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
