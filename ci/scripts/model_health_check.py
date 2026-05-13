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

Scans config*.yml files under examples/ for LLM and embedder blocks with
_type: nim, extracts model references (including optimizer search_space),
and checks each model in two passes:

  1. Catalog check  -- models missing from /v1/models have been removed.
     Applies to both LLMs and embedders.
  2. Inference check -- models present in the catalog but returning non-200
     on a minimal API call are temporarily down.  LLMs are tested via
     /v1/chat/completions, embedders via /v1/embeddings.

Reports removed and down models separately so the team can tell whether a
config needs a model swap (removed) or just needs to wait (down).
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    from gitutils import GitWrapper
    _FALLBACK_REPO = GitWrapper.get_repo_dir()
except Exception:
    _FALLBACK_REPO = str(Path(__file__).resolve().parents[2])

REPO = Path(os.environ.get('PROJECT_ROOT', _FALLBACK_REPO))
NIM_API_BASE = "https://integrate.api.nvidia.com/v1"
REQUEST_TIMEOUT = 30
INTER_REQUEST_DELAY = 1.0


def find_nim_models(examples_dir: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Scan example configs for NIM model references in both llms and embedders.

    NIMModelConfig accepts both ``model_name`` and ``model`` as the field name
    (via pydantic AliasChoices), so we check both. LLMs and embedders are
    returned separately because they use different endpoints for inference.

    Returns (llm_models, embedder_models), each mapping model name to config paths.
    """
    llm_models: dict[str, list[str]] = {}
    embedder_models: dict[str, list[str]] = {}

    for config_path in sorted(examples_dir.rglob("config*.yml")):
        with open(config_path, encoding="utf-8") as f:
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

        try:
            rel = str(config_path.relative_to(REPO))
        except ValueError:
            rel = str(config_path)

        for section_key, target in (("llms", llm_models), ("embedders", embedder_models)):
            section = cfg.get(section_key)
            if not isinstance(section, dict):
                continue

            for _name, block in section.items():
                if not isinstance(block, dict):
                    continue
                if block.get("_type") != "nim":
                    continue

                model = block.get("model_name") or block.get("model")
                if model:
                    target.setdefault(model, []).append(rel)

                search_space = block.get("search_space", {})
                if isinstance(search_space, dict):
                    for key in ("model_name", "model"):
                        space_entry = search_space.get(key, {})
                        if isinstance(space_entry, dict):
                            for val in space_entry.get("values", []):
                                if isinstance(val, str):
                                    target.setdefault(val, []).append(rel)

    return llm_models, embedder_models


def get_catalog_models(api_key: str) -> set[str]:
    """Fetch the set of model IDs currently listed in the NIM catalog.

    Calls GET /v1/models and returns the ``id`` field of each entry.
    Returns an empty set on any network or parsing failure so the caller
    can fall back to inference-only checks.
    """
    req = urllib.request.Request(
        f"{NIM_API_BASE}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    ctx = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=ctx) as resp:
            body = json.loads(resp.read().decode())
            return {m["id"] for m in body.get("data", []) if isinstance(m, dict) and "id" in m}
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError) as e:
        print(f"  WARNING: could not fetch /v1/models catalog: {e}", file=sys.stderr)
        return set()


def _nim_post(endpoint: str, payload: bytes, api_key: str) -> tuple[int, str]:
    """POST *payload* to NIM_API_BASE/*endpoint* and return (status, detail)."""
    req = urllib.request.Request(
        f"{NIM_API_BASE}/{endpoint}",
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
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
            detail = str(e)
        return e.code, detail
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, f"Connection error: {e}"


def check_model(model: str, api_key: str) -> tuple[int, str]:
    """Make a minimal chat/completions call and return (status_code, detail)."""
    payload = json.dumps({
        "model": model,
        "messages": [{
            "role": "user", "content": "hi"
        }],
        "max_tokens": 1,
    }).encode()
    return _nim_post("chat/completions", payload, api_key)


def check_embedder(model: str, api_key: str) -> tuple[int, str]:
    """Make a minimal embeddings call and return (status_code, detail)."""
    payload = json.dumps({
        "model": model,
        "input": ["hi"],
        "input_type": "query",
    }).encode()
    return _nim_post("embeddings", payload, api_key)


def main() -> int:
    """Parse CLI args, discover NIM models from configs, and health-check each one."""
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write structured results to a JSON file for downstream reporting",
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

    llm_models, embedder_models = find_nim_models(args.examples_dir)

    # Merge into a single lookup for config file references
    all_configs: dict[str, list[str]] = {}
    for m, files in llm_models.items():
        all_configs.setdefault(m, []).extend(files)
    for m, files in embedder_models.items():
        all_configs.setdefault(m, []).extend(files)

    if not all_configs:
        print("No NIM models found in config files")
        return 0

    print(f"Found {len(llm_models)} LLM(s) and {len(embedder_models)} embedder(s) "
          f"({len(all_configs)} unique model(s)) across example configs\n")

    if args.dry_run:
        for label, section in (("LLMs", llm_models), ("Embedders", embedder_models)):
            if not section:
                continue
            print(f"  {label}:")
            for model, files in sorted(section.items()):
                print(f"    {model}")
                if args.verbose:
                    for f in sorted(set(files)):
                        print(f"      - {f}")
        return 0

    # -- Pass 1: catalog check for ALL models (LLMs + embedders) -------------
    print("Pass 1: checking /v1/models catalog...")
    catalog = get_catalog_models(api_key)

    all_model_names = set(all_configs.keys())

    if catalog:
        removed = sorted(all_model_names - catalog)
        catalog_ok = all_model_names & catalog

        for model in removed:
            mtype = "embedder" if model in embedder_models else "llm"
            print(f"  REMOVED  {model}  ({mtype})")
    else:
        print("  WARNING: catalog unavailable, falling back to inference-only checks")
        removed = []
        catalog_ok = all_model_names

    print()

    # -- Pass 2: inference check on models still in catalog ------------------
    llm_to_test = sorted(set(llm_models.keys()) & catalog_ok)
    embedder_to_test = sorted(set(embedder_models.keys()) & catalog_ok)

    if llm_to_test or embedder_to_test:
        print("Pass 2: inference check on catalog-listed models...")
    down: list[tuple[str, int, str]] = []
    call_count = 0

    for model in llm_to_test:
        if call_count > 0:
            time.sleep(INTER_REQUEST_DELAY)
        call_count += 1

        status, detail = check_model(model, api_key)
        if status in (401, 403):
            print(f"\n  ERROR: API key is invalid or expired (HTTP {status}): {detail}", file=sys.stderr)
            return 1
        if status == 200:
            print(f"  OK      {model}")
        else:
            label = f"HTTP {status}" if status > 0 else "ERROR"
            print(f"  DOWN    {model} -> {label}: {detail}")
            down.append((model, status, detail))

    for model in embedder_to_test:
        if call_count > 0:
            time.sleep(INTER_REQUEST_DELAY)
        call_count += 1

        status, detail = check_embedder(model, api_key)
        if status in (401, 403):
            print(f"\n  ERROR: API key is invalid or expired (HTTP {status}): {detail}", file=sys.stderr)
            return 1
        if status == 200:
            print(f"  OK      {model}  (embedder)")
        else:
            label = f"HTTP {status}" if status > 0 else "ERROR"
            print(f"  DOWN    {model} -> {label} (embedder): {detail}")
            down.append((model, status, detail))

    print()

    # -- Summary -------------------------------------------------------------
    has_failures = bool(removed) or bool(down)

    if removed:
        print(f"{len(removed)} model(s) REMOVED from catalog (need config update):\n")
        for model in removed:
            print(f"  {model}")
            for f in sorted(set(all_configs[model])):
                print(f"    - {f}")
            print()

    if down:
        print(f"{len(down)} model(s) DOWN (in catalog but unreachable):\n")
        for model, status, _detail in down:
            label = f"HTTP {status}" if status > 0 else "ERROR"
            print(f"  {model} ({label})")
            for f in sorted(set(all_configs[model])):
                print(f"    - {f}")
            print()

    if not has_failures:
        print(f"All {len(all_configs)} model(s) are reachable.")

    if args.output_json:
        down_models = {m for m, _s, _d in down}
        report = {
            "removed": [{
                "model": m,
                "type": "embedder" if m in embedder_models else "llm",
                "configs": sorted(set(all_configs[m])),
            } for m in removed],
            "down": [{
                "model": m,
                "type": "embedder" if m in embedder_models else "llm",
                "status": s,
                "detail": d,
                "configs": sorted(set(all_configs[m])),
            } for m, s, d in down],
            "ok": [{
                "model": m,
                "type": "embedder" if m in embedder_models else "llm",
                "configs": sorted(set(all_configs[m])),
            } for m in sorted(all_model_names) if m not in removed and m not in down_models],
        }
        with open(args.output_json, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)
        print(f"Results written to {args.output_json}")

    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
