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

"""CLI entry points for NAT Harbor adapter execution."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from nat_harbor.adapters import BaseTaskAdapter


def parse_source_ids(raw_ids: str | None) -> list[str] | None:
    if raw_ids is None:
        return None
    parsed = [source_id.strip() for source_id in raw_ids.split(",") if source_id.strip()]
    return parsed if parsed else None


def load_adapter_class(adapter_class_path: str) -> type[BaseTaskAdapter]:
    """Load an adapter class from module path format `module.path:ClassName`."""
    module_path, separator, class_name = adapter_class_path.partition(":")
    if not separator or not module_path or not class_name:
        raise ValueError(
            "Invalid --adapter-class value. Expected format: 'module.path:ClassName'."
        )

    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name, None)
    if adapter_cls is None:
        raise ValueError(f"Class '{class_name}' not found in module '{module_path}'.")
    if not isinstance(adapter_cls, type) or not issubclass(adapter_cls, BaseTaskAdapter):
        raise ValueError(f"'{adapter_class_path}' must resolve to a BaseTaskAdapter subclass.")
    return adapter_cls


def build_parser() -> argparse.ArgumentParser:
    """Build top-level CLI parser."""
    parser = argparse.ArgumentParser(description="Generate Harbor tasks using adapter classes.")
    parser.add_argument(
        "--adapter-class",
        required=True,
        help="Adapter class import path in format 'module.path:ClassName'.",
    )
    parser.add_argument(
        "--source-file",
        required=True,
        help="Path to source dataset JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Target directory for generated Harbor tasks.",
    )
    parser.add_argument(
        "--source-ids",
        default=None,
        help="Comma-separated source IDs to generate. Defaults to all entries in source dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated task directories.",
    )
    return parser


def main() -> None:
    """Execute selected adapter."""
    parser = build_parser()
    args = parser.parse_args()

    adapter_cls = load_adapter_class(args.adapter_class)
    adapter = adapter_cls(Path(args.source_file))

    parsed_ids = parse_source_ids(args.source_ids)
    source_ids = parsed_ids if parsed_ids is not None else adapter.list_available_tasks()
    generated, requested = adapter.generate_many(
        source_ids=source_ids,
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
    )
    print(
        f"[nat-harbor-adapt] adapter_class={args.adapter_class} generated={generated} requested={requested} output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()

