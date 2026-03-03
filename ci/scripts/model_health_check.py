#!/usr/bin/env python3
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
"""Check that NIM model endpoints referenced in example configs are reachable.

Scans config*.yml files under examples/ for LLM blocks with _type: nim,
extracts the model_name values (including optimizer search_space entries),
and makes a minimal inference call to each unique model. Reports any models
that return non-200 responses.
"""

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPO = Path(__file__).resolve().parents[2]
NIM_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
REQUEST_TIMEOUT = 30


def find_nim_models(examples_dir: Path) -> dict[str, list[str]]:
    """Scan example configs for model_name values under _type: nim LLMs.

    NIMModelConfig accepts both ``model_name`` and ``model`` as the field name
    (via pydantic AliasChoices), so we check both. Embedders also use _type: nim
    but hit the /v1/embeddings endpoint, not /v1/chat/completions, so we only
    scan the ``llms`` section here.

    Returns a dict mapping model name to list of config file paths that reference it.
    """
    models: dict[str, list[str]] = {}

    for config_path in sorted(examples_dir.rglob("config*.yml")):
        with open(config_path) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                try:
                    rel = str(config_path.relative_to(REPO))
                except ValueError:
                    rel = str(config_path)
                print(f"  WARNING: could not parse {rel}: {exc}", file=sys.stderr)
                continue

        if not isinstance(cfg, dict):
            continue

        llms = cfg.get("llms")
        if not isinstance(llms, dict):
            continue

        rel = str(config_path.relative_to(REPO))

        for _llm_name, block in llms.items():
            if not isinstance(block, dict):
                continue
            if block.get("_type") != "nim":
                continue

            model = block.get("model_name") or block.get("model")
            if model:
                models.setdefault(model, []).append(rel)

            search_space = block.get("search_space", {})
            if isinstance(search_space, dict):
                for key in ("model_name", "model"):
                    space_entry = search_space.get(key, {})
                    if isinstance(space_entry, dict):
                        for val in space_entry.get("values", []):
                            if isinstance(val, str):
                                models.setdefault(val, []).append(rel)

    return models


def check_model(model: str, api_key: str) -> tuple[int, str]:
    """Make a minimal inference call and return (status_code, detail).

    Returns (200, "") on success, (status_code, error_detail) on API error,
    or (0, error_message) on connection/timeout failure.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{
            "role": "user", "content": "hi"
        }],
        "max_tokens": 1,
    }).encode()

    req = urllib.request.Request(
        NIM_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    ctx = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=ctx) as resp:
            return resp.status, ""
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            body = json.loads(e.read().decode())
            detail = body.get("detail", str(body))
        except Exception:
            detail = str(e)
        return e.code, detail
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, f"Connection error: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=REPO / "examples",
        help="Directory to scan for config files (default: examples/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan configs and list models without making API calls",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show which config files reference each model",
    )
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: NVIDIA_API_KEY environment variable is not set", file=sys.stderr)
        print("Set it or use --dry-run to just list discovered models", file=sys.stderr)
        return 1

    if not args.examples_dir.is_dir():
        print(f"ERROR: {args.examples_dir} is not a directory", file=sys.stderr)
        return 1

    models = find_nim_models(args.examples_dir)

    if not models:
        print("No NIM models found in config files")
        return 0

    print(f"Found {len(models)} unique NIM model(s) across example configs\n")

    if args.dry_run:
        for model, files in sorted(models.items()):
            print(f"  {model}")
            if args.verbose:
                for f in sorted(set(files)):
                    print(f"    - {f}")
        return 0

    failures: list[tuple[str, int, str, list[str]]] = []

    for model, files in sorted(models.items()):
        status, detail = check_model(model, api_key)
        if status in (401, 403):
            print(f"\n  ERROR: API key is invalid or expired (HTTP {status}): {detail}", file=sys.stderr)
            return 1
        if status == 200:
            print(f"  OK    {model}")
            if args.verbose:
                for f in sorted(set(files)):
                    print(f"    - {f}")
        else:
            label = f"HTTP {status}" if status > 0 else "ERROR"
            print(f"  FAIL  {model} -> {label}: {detail}")
            failures.append((model, status, detail, files))

    print()

    if failures:
        print(f"{len(failures)} model(s) unreachable:\n")
        for model, status, detail, files in failures:
            label = f"HTTP {status}" if status > 0 else "ERROR"
            print(f"  {model} ({label})")
            for f in sorted(set(files)):
                print(f"    - {f}")
            print()
        return 1

    print(f"All {len(models)} model(s) are reachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
