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

Scans YAML files under examples/ for LLM and embedder blocks with
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
import logging
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    logging.basicConfig(format="%(message)s", level=logging.ERROR)
    _logger.error("ERROR: pyyaml is required. Install with: pip install pyyaml")
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
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
SCHEMA_VERSION = "1.0"

EXCLUDE_YAMLS = ("examples/documentation_guides/locally_hosted_llms/nim_config.yml", )


def find_nim_models(examples_dir: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Scan example configs for NIM model references in both llms and embedders.

    NIMModelConfig accepts both ``model_name`` and ``model`` as the field name
    (via pydantic AliasChoices), so we check both. LLMs and embedders are
    returned separately because they use different endpoints for inference.

    Returns (llm_models, embedder_models), each mapping model name to config paths.
    """
    llm_models: dict[str, list[str]] = {}
    embedder_models: dict[str, list[str]] = {}

    config_paths = []
    for pattern in ("*.yml", "*.yaml"):
        config_paths.extend(examples_dir.rglob(pattern))

    for config_path in sorted(config_paths):
        relative_path = str(config_path.resolve().relative_to(REPO))
        if relative_path in EXCLUDE_YAMLS:
            _logger.debug("Skipping excluded config: %s", relative_path)
            continue

        with open(config_path, encoding="utf-8") as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                _logger.warning("  WARNING: could not parse %s: %s", relative_path, exc)
                continue

        if not isinstance(cfg, dict):
            continue

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
                    target.setdefault(model, set()).add(relative_path)

                search_space = block.get("search_space", {})
                if isinstance(search_space, dict):
                    for key in ("model_name", "model"):
                        space_entry = search_space.get(key, {})
                        if isinstance(space_entry, dict):
                            for val in space_entry.get("values", []):
                                if isinstance(val, str):
                                    target.setdefault(val, set()).add(relative_path)

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
        _logger.warning("  WARNING: could not fetch /v1/models catalog: %s", e)
        return set()


def _nim_post(endpoint: str, payload: bytes, api_key: str) -> tuple[int, str, str]:
    """POST *payload* to NIM_API_BASE/*endpoint* and return (status, detail, deprecation)."""

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
            _logger.info("    Received response: HTTP %s", resp.status)
            deprecated = resp.headers.get("Deprecation", "")
            return resp.status, "", deprecated
    except urllib.error.HTTPError as e:
        _logger.error("    Received response: HTTP %s", e.code)
        detail = ""
        try:
            body = json.loads(e.read().decode())
            detail = body.get("detail", str(body))
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
            detail = str(e)
        return e.code, detail, ""
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, f"Connection error: {e}", ""


def check_model(model: str, api_key: str) -> tuple[int, str, str]:
    """Make a minimal chat/completions call and return (status_code, detail, deprecation)."""
    payload = json.dumps({
        "model": model,
        "messages": [{
            "role": "user", "content": "hi"
        }],
        "max_tokens": 1,
    }).encode()
    return _nim_post("chat/completions", payload, api_key)


def check_embedder(model: str, api_key: str) -> tuple[int, str, str]:
    """Make a minimal embeddings call and return (status_code, detail, deprecation)."""
    payload = json.dumps({
        "model": model,
        "input": ["hi"],
        "input_type": "query",
    }).encode()
    return _nim_post("embeddings", payload, api_key)


def write_json_report(output_file: Path, report: dict) -> None:
    """Write the report dict to a JSON file at the given path."""
    with open(output_file, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2)
    _logger.info("Results written to %s", output_file)


def _handle_dry_run(llm_models: dict[str, list[str]],
                    embedder_models: dict[str, list[str]],
                    output_file: Path | None,
                    verbose: bool) -> int:
    try:
        report = {"schema_version": SCHEMA_VERSION}
        for label, section in (("LLMs", llm_models), ("Embedders", embedder_models)):
            report_rows: list[dict[str, str | int]] = []
            model_by_usage = sorted(((len(files), model) for model, files in section.items()), reverse=True)
            _logger.info("  %s: Usage count", label)
            for count, model in model_by_usage:
                _logger.info("    %s: %s", model, count)
                report_rows.append({"model": model, "num_configs": count})
                if verbose:
                    files = section[model]
                    for f in sorted(set(files)):
                        _logger.info("      - %s", f)

            report[label.lower()] = report_rows

        if output_file is not None:
            write_json_report(output_file=output_file, report=report)

        return 0
    except Exception as e:
        _logger.error("ERROR during dry run: %s", e)
        return 1


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
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=os.environ.get("NAT_LOG_LEVEL", "INFO").upper(),
        help="Set logging level (default: INFO, or NAT_LOG_LEVEL if set)",
        type=str.upper,
    )
    args = parser.parse_args()
    if args.log_level not in LOG_LEVELS:
        parser.error(f"invalid log level: {args.log_level}. Choose from {', '.join(LOG_LEVELS)}")
    logging.basicConfig(format="%(message)s", level=getattr(logging, args.log_level))

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key and not args.dry_run:
        _logger.error("ERROR: NVIDIA_API_KEY environment variable is not set")
        _logger.error("Set it or use --dry-run to just list discovered models")
        return 1

    if not args.examples_dir.is_dir():
        _logger.error("ERROR: %s is not a directory", args.examples_dir)
        return 1

    llm_models, embedder_models = find_nim_models(args.examples_dir)

    # Merge into a single lookup for config file references
    all_configs: dict[str, list[str]] = llm_models.copy()
    # Assume that LLMs and embedders are distinct sets of models
    all_configs.update(embedder_models)

    if not all_configs:
        _logger.info("No NIM models found in config files")
        return 0

    _logger.info("Found %s LLM(s) and %s embedder(s) (%s unique model(s)) across example configs\n",
                 len(llm_models),
                 len(embedder_models),
                 len(all_configs))

    if args.dry_run:
        return _handle_dry_run(llm_models, embedder_models, args.output_json, args.verbose)

    # -- Pass 1: catalog check for ALL models (LLMs + embedders) -------------
    _logger.info("Pass 1: checking /v1/models catalog...")
    catalog = get_catalog_models(api_key)

    all_model_names = set(all_configs.keys())

    if catalog:
        removed = sorted(all_model_names - catalog)
        catalog_ok = all_model_names & catalog

        for model in removed:
            mtype = "embedder" if model in embedder_models else "llm"
            _logger.info("  REMOVED  %s  (%s)", model, mtype)
    else:
        _logger.warning("  WARNING: catalog unavailable, falling back to inference-only checks")
        removed = []
        catalog_ok = all_model_names

    _logger.info("")

    # -- Pass 2: inference check on models still in catalog ------------------
    llm_to_test = sorted(set(llm_models.keys()) & catalog_ok)
    embedder_to_test = sorted(set(embedder_models.keys()) & catalog_ok)

    if llm_to_test or embedder_to_test:
        _logger.info("Pass 2: inference check on catalog-listed models...")

    down: list[tuple[str, int, str]] = []
    deprecation: list[tuple[str, str]] = []
    call_count = 0

    for model in llm_to_test:
        if call_count > 0:
            time.sleep(INTER_REQUEST_DELAY)
        call_count += 1

        status, detail, deprecation_detail = check_model(model, api_key)
        if status in (401, 403):
            _logger.error("\n  ERROR: API key is invalid or expired (HTTP %s): %s", status, detail)
            return 1
        elif deprecation_detail != "":
            _logger.info("  Deprecation: %s", deprecation_detail)
            deprecation.append((model, deprecation_detail))
        elif status == 200:
            _logger.info("  OK      %s", model)
        else:
            label = f"HTTP {status}" if status > 0 else "ERROR"
            _logger.info("  DOWN    %s -> %s: %s", model, label, detail)
            down.append((model, status, detail))

    for model in embedder_to_test:
        if call_count > 0:
            time.sleep(INTER_REQUEST_DELAY)
        call_count += 1

        status, detail, deprecation_detail = check_embedder(model, api_key)
        if status in (401, 403):
            _logger.error("\n  ERROR: API key is invalid or expired (HTTP %s): %s", status, detail)
            return 1
        elif deprecation_detail != "":
            _logger.info("  Deprecation: %s", deprecation_detail)
            deprecation.append((model, deprecation_detail))
        elif status == 200:
            _logger.info("  OK      %s  (embedder)", model)
        else:
            label = f"HTTP {status}" if status > 0 else "ERROR"
            _logger.info("  DOWN    %s -> %s (embedder): %s", model, label, detail)
            down.append((model, status, detail))

    _logger.info("")

    # -- Summary -------------------------------------------------------------
    has_failures = bool(removed) or bool(down) or bool(deprecation)

    if removed:
        _logger.info("%s model(s) REMOVED from catalog (need config update):\n", len(removed))
        for model in removed:
            _logger.info("  %s", model)
            for f in sorted(set(all_configs[model])):
                _logger.info("    - %s", f)
            _logger.info("")

    if deprecation:
        _logger.info("%s model(s) DEPRECATED (in catalog but deprecated):\n", len(deprecation))
        for model, detail in deprecation:
            _logger.info("  %s (%s)", model, detail)
            for f in sorted(set(all_configs[model])):
                _logger.info("    - %s", f)
            _logger.info("")

    if down:
        _logger.info("%s model(s) DOWN (in catalog but unreachable):\n", len(down))
        for model, status, _detail in down:
            label = f"HTTP {status}" if status > 0 else "ERROR"
            _logger.info("  %s (%s)", model, label)
            for f in sorted(set(all_configs[model])):
                _logger.info("    - %s", f)
            _logger.info("")

    if not has_failures:
        _logger.info("All %s model(s) are reachable.", len(all_configs))

    if args.output_json:
        down_models = {m for m, _s, _d in down}
        deprecated_models = {m for m, _d in deprecation}
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
            "deprecated": [{
                "model": m,
                "type": "embedder" if m in embedder_models else "llm",
                "detail": d,
                "configs": sorted(set(all_configs[m])),
            } for m, d in deprecation],
            "ok": [{
                "model": m,
                "type": "embedder" if m in embedder_models else "llm",
                "configs": sorted(set(all_configs[m])),
            } for m in sorted(all_model_names)
                   if m not in removed and m not in down_models and m not in deprecated_models],
        }

        write_json_report(output_file=args.output_json, report=report)

    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
