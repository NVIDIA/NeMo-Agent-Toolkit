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
"""CLI runner for ATIF evaluator bridge in Harbor verifier scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from nat_harbor.verifier.inline_verifier import InlineVerifierError
from nat_harbor.verifier.inline_verifier import InlineVerifierRequest
from nat_harbor.verifier.inline_verifier import verify_inline_sync


def run_bridge(
    *,
    artifact_path: str,
    evaluator_kind: str,
    evaluator_ref: str | None,
    output_dir: str,
    fallback_mode: str,
    config_file: str | None,
    evaluator_name: str | None,
) -> int:
    """Run bridge evaluation and emit Harbor verifier artifacts."""
    request = InlineVerifierRequest(
        trajectory_path=Path(artifact_path),
        evaluator_kind=evaluator_kind,
        evaluator_ref=evaluator_ref,
        config_file=config_file,
        evaluator_name=evaluator_name,
        verifier_output_dir=Path(output_dir),
        fallback_mode=fallback_mode,
    )
    try:
        verify_inline_sync(request)
    except InlineVerifierError:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build bridge runner command line parser."""
    parser = argparse.ArgumentParser(description="Run NAT ATIF evaluator bridge from Harbor verifier script.")
    parser.add_argument("--artifact-path", default="trajectory.json", help="ATIF artifact JSON path.")
    parser.add_argument(
        "--evaluator-kind",
        required=True,
        help=("Evaluator dispatch kind. Use `custom` with --evaluator-ref, "
              "or provide a builtin evaluator name (for example `trajectory`, `tunable_rag`, `ragas`)."),
    )
    parser.add_argument("--evaluator-ref", default=None, help="Custom evaluator ref in module:function format.")
    parser.add_argument("--output-dir", default="/logs/verifier", help="Verifier output directory.")
    parser.add_argument(
        "--fallback-mode",
        default="fail",
        choices=["fail", "raw_output"],
        help="Fallback behavior when artifacts/evaluation fail.",
    )
    parser.add_argument("--config-file", default=None, help="NAT config path for builtin evaluators.")
    parser.add_argument("--evaluator-name", default=None, help="Evaluator name inside --config-file.")
    return parser


def main() -> int:
    """CLI entrypoint for bridge runner."""
    args = build_parser().parse_args()
    return run_bridge(
        artifact_path=args.artifact_path,
        evaluator_kind=args.evaluator_kind,
        evaluator_ref=args.evaluator_ref,
        output_dir=args.output_dir,
        fallback_mode=args.fallback_mode,
        config_file=args.config_file,
        evaluator_name=args.evaluator_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
