# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
import re
import sys
import textwrap
from dataclasses import dataclass

from gitutils import all_files

# File path pairs to ignore -- first is the file path, second is the path in the file
IGNORED_FILE_PATH_PAIRS: set[tuple[str, str]] = {

    # allow references from tests to examples
    (r"^tests/", r"^examples/"),

    # allow references to docs from examples
    (r"^examples/", r"^(\.\./)*docs/"),

    # allow references to examples from docs
    (r"^docs/", r"^(\.\./)*examples/"),

    # allow relative references within docs
    (r"^docs/", r"^(\.?\./)*[A-Za-z0-9_\-\.]+"),

    # allow references to subdirectories from examples
    (r"^examples/", r"^\./[A-Za-z0-9_\-\.]+/"),

    # allow references to text_file_ingest from create-a-new-workflow
    (
        r"^docs/source/tutorials/create-a-new-workflow.md",
        r"^src/text_file_ingest/data",
    ),

    # allow references from some examples to other examples
    (
        r"^examples/advanced_agents/alert_triage_agent/README.md",
        r"^examples/advanced_agents/alert_triage_agent/.your_custom_env",
    ),
    (
        r"^examples/advanced_agents/profiler_agent/README.md",
        r"^examples/observability/simple_calculator_observability",
    ),
    (
        r"^examples/advanced_agents/profiler_agent/README.md",
        r"^examples/observability/simple_calculator_observability/configs/config-tracing.yml",
    ),
    (
        r"^examples/advanced_agents/profiler_agent/README.md",
        r"^examples/observability/simple_calculator_observability/configs/config-tracing.yml",
    ),
    (
        r"^examples/advanced_agents/profiler_agent/README.md",
        r"^examples/observability/simple_calculator_observability/configs/config-tracing.yml",
    ),
    (
        r"^examples/advanced_agents/profiler_agent/README.md",
        r"^examples/observability/simple_calculator_observability/configs/config-tracing.yml",
    ),
    (
        r"^examples/custom_functions/automated_description_generation/README.md",
        r"^\.\./\.\./RAG/simple_rag/README.md",
    ),
    (
        r"^examples/documentation_guides/workflows/text_file_ingest/src/text_file_ingest/configs/config.yml",
        r"^examples/evaluation_and_profiling/simple_web_query_eval/data/langsmith.json",
    ),
    (
        r"^examples/evaluation_and_profiling/simple_calculator_eval/README.md",
        r"^\.\./\.\./getting_started/simple_calculator",
    ),
    (
        r"^examples/evaluation_and_profiling/simple_calculator_eval/README.md",
        r"^examples/getting_started/simple_calculator/data/simple_calculator.json",
    ),
    (
        r"^examples/evaluation_and_profiling/simple_calculator_eval/configs/config-sizing-calc.yml",
        r"^examples/getting_started/simple_calculator/data/simple_calculator.json",
    ),
    (
        r"^examples/evaluation_and_profiling/simple_calculator_eval/configs/config-tunable-rag-eval.yml",
        r"^examples/getting_started/simple_calculator/data/simple_calculator.json",
    ),
    (
        r"^examples/evaluation_and_profiling/simple_web_query_eval/README.md",
        r"^\.\./\.\./getting_started/simple_web_query",
    ),
    (
        r"^examples/memory/redis/README.md",
        r"^examples/deploy/docker-compose\..*\.yml",
    ),
    (
        r"^examples/MCP/simple_calculator_mcp/README.md",
        r"^\.\./\.\./getting_started/simple_calculator",
    ),
    (
        r"^examples/MCP/simple_calculator_mcp/README.md",
        r"^examples/getting_started/simple_calculator/configs/config.yml",
    ),
    (
        r"^examples/MCP/simple_calculator_mcp/README.md",
        r"^examples/getting_started/simple_calculator/configs/config.yml",
    ),
    (
        r"^examples/observability/simple_calculator_observability/README.md",
        r"^\.\./\.\./getting_started/simple_calculator",
    ),

    # allow references to repo source code from examples
    (
        r"^examples/front_ends/simple_calculator_custom_routes/README.md",
        r"^src/aiq/tool/server_tools.py",
    ),
    (
        r"^examples/evaluation_and_profiling/swe_bench/README.md",
        r"^src/aiq/data_models/swe_bench_model.py",
    ),
    # allow short names for references to files within the swe_bench example
    (
        r"^examples/evaluation_and_profiling/swe_bench/README.md",
        r"^(predict_gold_stub\.py|predict_skeleton\.py|predictors/predict_skeleton|predictors/register\.py)$",
    ),
    # blacklist remote files
    (
        r"^examples/evaluation_and_profiling/simple_web_query_eval/.*configs/eval_upload.yml",
        r"^input/langsmith.json",
    ),
}

IGNORED_FILE_PATH_PAIRS_REGEX = list(map(lambda x: (re.compile(x[0]), re.compile(x[1])), IGNORED_FILE_PATH_PAIRS))

# Paths to ignore -- regex pattern
IGNORED_PATHS: set[str] = {
    r"^(\./)?\.tmp/",  #
    # files that are located in the directory of the file being checked
    r"^\./upload_to_minio\.sh$",
    r"^\./upload_to_mysql\.sh$",
    r"^\./start_local_sandbox\.sh$",  #
    # script files that exist in the root of the repo
    r"^scripts/langchain_web_ingest\.py$",
    r"^scripts/bootstrap_milvus\.sh$",  #
    # generated files
    r"^\.venv/bin/activate$",
    r"^\./run_service\.sh$",
    r"^outputs/line_chart_\d+\.png$",  #
    # false positives
    r"^N/A$",
    r"^and/or$",
    r"^input/output$",
}

IGNORED_PATHS_REGEX = list(map(re.compile, IGNORED_PATHS))

# Files to ignore -- regex pattern
IGNORED_FILES: set[str] = {
    r"^\..*",
    r"^ci/",
    r"pyproject\.toml$",
    r"Dockerfile",
    r"docker-compose([A-Za-z0-9_\-\.]+)?\.ya?ml$",
    r"(CHANGELOG|CONTRIBUTING|LICENSE|SECURITY)\.md",
    r"^manifest.yaml$",
    r"data/.*$"
}

IGNORED_FILES_REGEX = list(map(re.compile, IGNORED_FILES))

# Paths to consider referential -- string
# referential paths are ones that should not only be checked for existence, but also for referential integrity
# (i.e. that the path exists in the same directory as the file)
REFERENTIAL_PATHS: set[str] = {
    "examples",
    "docs",
}

# File extensions to check paths
EXTENSIONS: tuple[str, ...] = ('.md', '.rst', '.yml', '.yaml', '.json', '.toml', '.ini', '.conf', '.cfg')

PATH_REGEX = re.compile(r'((\w+://[^\s\'"<>]+)|(\.?\.?/?)(([$A-Za-z0-9_\-\.]+/)*[$A-Za-z0-9_\-\.]+)|)')

YAML_BLOCK_REGEX = re.compile(r":\s*\|\s*$")


def list_broken_symlinks() -> list[str]:
    """
    Lists all broken symbolic links found within the repo.

    Returns:
        A list of paths to broken symlinks.
    """
    broken_symlinks = []
    for f in all_files():
        if os.path.islink(f):
            if not os.path.exists(f):
                broken_symlinks.append(f)
    return broken_symlinks


@dataclass
class PathInfo:
    line_number: int
    column: int
    path: str


def extract_paths_from_file(filename: str) -> list[PathInfo]:
    """
    Extracts paths from a file. Skips absolute paths, "." and ".." paths, and paths that match any of the ignored paths.
    Args:
        filename: The path to the file to extract paths from.
    Returns:
        A list of PathInfo objects.
    """
    paths = []
    with open(filename, 'r', encoding='utf-8') as f:
        in_skipped_section: int | bool = False
        for line_number, line in enumerate(f, start=1):
            if filename.endswith(".md") and line.lstrip().startswith("```"):
                in_skipped_section = not in_skipped_section
            elif filename.endswith(".yml") or filename.endswith(".yaml"):
                # YAML blocks are delimited by a line that ends with a pipe
                if YAML_BLOCK_REGEX.search(line):
                    # keep track of the number of leading spaces
                    in_skipped_section = len(line) - len(line.lstrip())
                elif in_skipped_section:
                    # if we are in a skipped section, and the number of leading spaces is the same as the number of
                    # leading spaces in the skipped section, then we are done
                    if len(line) - len(line.lstrip()) == in_skipped_section:
                        in_skipped_section = False
            if in_skipped_section:
                continue
            for match in PATH_REGEX.finditer(line):
                column, _ = match.span()
                path = match.group(0)
                # Exclude URIs
                if "://" in path:
                    continue
                # Exclude absolute paths
                if path.startswith('/'):
                    continue
                # Exclude paths that don't contain a slash
                if '/' not in path:
                    continue
                # Exclude "." and ".."
                if path in ('.', '..'):
                    continue
                # Exclude empty after stripping
                if not path:
                    continue
                if any(r.search(path) for r in IGNORED_PATHS_REGEX):
                    continue
                if any(r[0].search(filename) and r[1].search(path) for r in IGNORED_FILE_PATH_PAIRS_REGEX):
                    continue
                paths.append(PathInfo(line_number, column + 1, path))
    return paths


def check_files() -> list[tuple[str, PathInfo]]:
    """
    Checks files in the repo for paths that don't exist. Skips files that match any of the ignored files.

    Returns:
        A list of tuples of (filename, path) that don't exist.
    """
    filenames_with_broken_paths = []

    skipped_paths: set[str] = set()

    for f in all_files(path_filter=lambda x: x.endswith(EXTENSIONS)):
        if any(r.search(f) for r in IGNORED_FILES_REGEX):
            continue
        paths = extract_paths_from_file(f)
        for path_info in paths:
            # If the path does not exist, then it is broken
            if not os.path.exists(path_info.path):
                if path_info.path.startswith("."):
                    # if it is a relative path, then it is broken
                    filenames_with_broken_paths.append((f, path_info))
                    continue
                else:
                    # attempt to resolve the path first
                    if not os.path.exists(os.path.join(os.path.dirname(f), path_info.path)):
                        # if it still doesn't exist then assume that it isn't actually a path
                        if any(p in path_info.path for p in REFERENTIAL_PATHS):
                            # if it is a referential path, then it is likely broken
                            filenames_with_broken_paths.append((f, path_info))
                        else:
                            skipped_paths.add(path_info.path)
                            continue
            path = os.path.commonpath([f, path_info.path])
            if path == "" or path == ".":
                # try to resolve the path relative to the file
                if not os.path.exists(os.path.join(os.path.dirname(f), path_info.path)):
                    filenames_with_broken_paths.append((f, path_info))
            else:
                for r in REFERENTIAL_PATHS:
                    if r in path:
                        # If the path is a single directory (which would be a referential path),
                        # then it is likely broken since it would be referring to outside the current
                        # directory
                        if len(path.split(os.path.sep)) == 1:
                            filenames_with_broken_paths.append((f, path_info))
                            break

    if skipped_paths:
        print("Warning: skipped the following paths:")
        for path in sorted(skipped_paths):
            print(f"- {path}")
        print("")

    return filenames_with_broken_paths


def main():
    """Main function to handle command line arguments and execute checks."""
    parser = argparse.ArgumentParser(description='Check for broken symlinks and paths in files')
    parser.add_argument('--check-broken-symlinks', action='store_true', help='Check for broken symbolic links')
    parser.add_argument('--check-paths-in-files', action='store_true', help='Check for broken paths in files')

    args = parser.parse_args()

    return_code: int = 0

    if args.check_broken_symlinks:
        print("Checking for broken symlinks...")
        broken_symlinks: list[str] = list_broken_symlinks()
        if broken_symlinks:
            return_code = 1
            print("Found broken symlinks:")
            for symlink in broken_symlinks:
                print(f"  {symlink}")
        print("Done checking for broken symlinks.")

    if args.check_paths_in_files:
        print("Checking for broken paths in files...")

        broken_paths: list[tuple[str, PathInfo]] = check_files()
        if broken_paths:
            return_code = 1
            print("Found \"broken\" paths in files:")
            for filename, path_info in broken_paths:
                print(f"- {filename}:{path_info.line_number}:{path_info.column} -> {path_info.path}")
        else:
            print("No broken paths found!")

        print(
            textwrap.dedent(
                """
                Note: some paths may be ignored due to rules
                - IGNORED_FILES: files that should be ignored
                - IGNORED_PATHS: paths that should be ignored
                - IGNORED_FILE_PATH_PAIRS: file-path pairs that should be ignored
                """, ))

        print("Done checking for broken paths in files.")

    sys.exit(return_code)


if __name__ == "__main__":
    main()
