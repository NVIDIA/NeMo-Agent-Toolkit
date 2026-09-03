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
Renders the framework-parity result JSON files into the Markdown table between the
`<!-- FRAMEWORK_PARITY_TABLE:START -->` / `:END` markers in a target README, and
rewrites the file in place. Re-running this is idempotent -- everything outside the
markers is left untouched.

Usage:
    python scripts/framework_parity/render_table.py --results-dir <dir> --readme <path>
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matrix import FRAMEWORK_MATRIX  # noqa: E402

START_MARKER = "<!-- FRAMEWORK_PARITY_TABLE:START -->"
END_MARKER = "<!-- FRAMEWORK_PARITY_TABLE:END -->"

_STATUS_BADGE = {
    "pass": "✅ pass",
    "fail": "❌ fail",
    "skipped": "⚪ skipped (no credentials)",
    "not_wired": "⚪ not wired (multi-credential example)",
}


def _row(result: dict) -> str:
    structural = _STATUS_BADGE.get(result["structural_status"], result["structural_status"])
    live = _STATUS_BADGE.get(result["live_status"], result["live_status"])
    version = result.get("framework_version") or "not installed"
    return (f"| {result['display_name']} | {result['framework_package']} `{version}` "
           f"| {structural} | {live} |")


def render(results_dir: Path) -> str:
    rows = []
    for entry in FRAMEWORK_MATRIX:
        result_file = results_dir / f"{entry.key}.json"
        if result_file.exists():
            result = json.loads(result_file.read_text())
        else:
            result = {
                "display_name": entry.display_name,
                "framework_package": entry.framework_package,
                "framework_version": None,
                "structural_status": "fail",
                "live_status": "skipped",
            }
        rows.append(_row(result))

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        START_MARKER,
        f"_Last generated {timestamp} by [`framework_parity.yml`]"
        f"(../../.github/workflows/framework_parity.yml)._",
        "",
        "| Framework | Version tested | Install + validate | Live workflow run |",
        "| --- | --- | --- | --- |",
        *rows,
        "",
        END_MARKER,
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--readme", required=True, type=Path)
    args = parser.parse_args()

    table = render(args.results_dir)
    content = args.readme.read_text(encoding="utf-8")

    if START_MARKER not in content or END_MARKER not in content:
        raise SystemExit(f"{args.readme} is missing the {START_MARKER}/{END_MARKER} markers")

    before, _, rest = content.partition(START_MARKER)
    _, _, after = rest.partition(END_MARKER)
    new_content = before + table + after

    args.readme.write_text(new_content, encoding="utf-8")
    print(f"Updated {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
