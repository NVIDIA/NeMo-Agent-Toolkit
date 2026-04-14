# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Generate a self-contained HTML viewer for an ATIF trajectory JSON file.

The HTML template, CSS, and JS are stored as sibling files in this package
and assembled at generation time into a single self-contained HTML file
(no external dependencies, works offline).

Usage:

    python -m nat.atif.scripts.viewer --input atif_trajectory.json
    python -m nat.atif.scripts.viewer --input atif_trajectory.json --output my_viewer.html --open
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent


def _build_html(atif_json: str) -> str:
    """Assemble the self-contained HTML from template parts and embedded ATIF data."""
    template = (_TEMPLATE_DIR / "template.html").read_text()
    css = (_TEMPLATE_DIR / "style.css").read_text()
    js = (_TEMPLATE_DIR / "app.js").read_text()

    html = template.replace("__CSS_PLACEHOLDER__", css)
    html = html.replace("__JS_PLACEHOLDER__", js)
    html = html.replace("__ATIF_DATA_PLACEHOLDER__", atif_json)
    return html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML viewer for an ATIF trajectory JSON file.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the ATIF trajectory JSON file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output HTML file path (default: {input_stem}_viewer.html).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        default=False,
        help="Open the generated HTML in the default browser.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    atif_data = json.loads(input_path.read_text())

    steps_count = len(atif_data.get("steps", []))
    sub_count = len((atif_data.get("extra") or {}).get("subagent_trajectories", {}))

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_viewer.html")

    html = _build_html(json.dumps(atif_data))

    output_path.write_text(html)
    print(f"ATIF viewer written to: {output_path}")
    print(f"  Steps: {steps_count}, Subagent trajectories: {sub_count}")

    if args.open:
        url = output_path.resolve().as_uri()
        webbrowser.open(url)
        print(f"  Opened in browser: {url}")


if __name__ == "__main__":
    main()
