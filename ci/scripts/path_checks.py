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

# File path pairs to allowlist -- first is the file path, second is the path in the file
ALLOWLISTED_FILE_PATH_PAIRS: set[tuple[str, str]] = {
    # allow references to data from configs
    (
        r"^examples/agents/.*/configs/config.yml",
        r"^examples/agents/data/",
    ),
    (
        r"^examples/",
        r"^examples/deploy/",
    ),
    (
        r"^examples/advanced_agents/alert_triage_agent/.*configs/config.*\.yml",
        r"^examples/advanced_agents/alert_triage_agent/data/",
    ),
    (
        r"^examples/advanced_agents/profiler_agent/README.md",
        r"^examples/observability/simple_calculator_observability",
    ),
    (
        r"^examples/documentation_guides/workflows/text_file_ingest/.*/config.yml",
        r"^examples/evaluation_and_profiling/simple_web_query_eval/data/langsmith.json",
    ),
    (
        r"^examples/evaluation_and_profiling/email_phishing_analyzer/configs",
        r"^examples/evaluation_and_profiling/email_phishing_analyzer/data",
    ),
    (
        r"^examples/evaluation_and_profiling/simple_web_query_eval/.*configs/",
        r"^examples/evaluation_and_profiling/simple_web_query_eval/data/",
    ),
    (
        r"^examples/evaluation_and_profiling/simple_calculator_eval/.*configs/",
        r"^examples/evaluation_and_profiling/simple_calculator_eval/data/",
    ),
    (
        r"^examples/evaluation_and_profiling/swe_bench/.*configs/",
        r"^examples/evaluation_and_profiling/swe_bench/data/",
    ),
    (
        r"^examples/evaluation_and_profiling/simple_calculator_eval/.*configs/",
        r"^examples/getting_started/simple_calculator/data/simple_calculator.json",
    ),
    (
        r"^examples/evaluation_and_profiling/simple_web_query_eval/.*configs",
        r"^examples/evaluation_and_profiling/simple_web_query_eval/.*/workflow_to_csv.py",
    ),
    (
        r"^examples/MCP/simple_calculator_mcp/README.md",
        r"^examples/getting_started/simple_calculator/configs/config.yml",
    ),
    (
        r"^examples/evaluation_and_profiling/simple_calculator_eval/README.md",
        r"^examples/getting_started/simple_calculator/data/simple_calculator.json",
    ),
    (
        r"^docs/source/",
        r"^docs/source/_static",
    ),
}

ALLOWLISTED_WORDS: set[str] = {
    "and/or",
    "application/json",
    "CI/CD",
    "commit/push",
    "Continue/Cancel",
    "conversation/chat",
    "create/reinstall/delete",
    "copy/paste",
    "edit/score",
    "file/console",
    "files/functions",
    "I/O",
    "Input/Observation",
    "input/output",
    "inputs/outputs",
    "JavaScript/TypeScript",
    "JSON/YAML",
    "LTE/5G",
    "output/jobs/job_",
    "predictions/forecasts",
    "provider/method.",
    "RagaAI/Catalyst",
    "read/write",
    "run/host",
    "run/serve",
    "search/edit/score/select",
    "size/time",
    "string/array",
    "string/object",
    "success/failure",
    "Thought/Action/Action",
    "thinking/reasoning",
    "tool/workflow",
    "tooling/vector",
    "true/false",
    "try/except",
    "user/assistant",
    "validate/sanitize",
    "Workflows/tools",
    "Yes/No",  #
    # numbers
    r"\d+/\d+(/\d+)*",  #
    # LLM model names
    "meta/[Ll]lama.*",
    "nvidia/([Ll]lama|[Nn][Vv]-).*",
    "mistralai/[Mm]ixtral.*",
    "microsoft/[Pp]hi.*",
    "ssmits/[Qq]wen.*",  #
    # MIME types
    "(application|text|image|video|audio|model|dataset|token|other)/.*",  #
    # Time zones
    "[A-Z][a-z]+(_[A-Z][a-z]+)*/[A-Z][a-z]+(_[A-Z][a-z]+)*",
}

IGNORED_FILE_PATH_PAIRS: set[tuple[str, str]] = {
    # ignore remote files
    (
        r"^examples/evaluation_and_profiling/simple_web_query_eval/.*configs/eval_upload.yml",
        r"^input/langsmith.json",
    ),
    # ignore notebook-relative paths
    (
        r"^examples/notebooks/retail_sales_agent/.*configs/",
        r"^\./retail_sales_agent/data/",
    ),
    # ignore generated files
    (
        r"^docs/",
        r"\.rst$",
    )
}

# Files to ignore -- regex pattern
IGNORED_FILES: set[str] = {
    # hidden files
    r"^\.",  #
    # CI files
    r"^ci/",  #
    # project files
    r"pyproject\.toml$",  #
    # docker files
    r"Dockerfile",  #
    r"docker-compose([A-Za-z0-9_\-\.]+)?\.ya?ml$",  #
    # top-level markdown files with no related content
    r"(CHANGELOG|CONTRIBUTING|LICENSE|SECURITY)\.md",
    r"^manifest.yaml$",  #
    # files located within data directories
    r"data/.*$",  #
    # Versions json file for the documentation version switcher button
    r"^docs/source/versions1.json$",
}

# Paths to ignore -- regex pattern
IGNORED_PATHS: set[str] = {
    # temporary files
    r"\.tmp/",  #
    # files that are located in the directory of the file being checked
    r"^\./upload_to_minio\.sh$",
    r"^\./upload_to_mysql\.sh$",
    r"^\./start_local_sandbox\.sh$",  #
    # script files that exist in the root of the repo
    r"^scripts/langchain_web_ingest\.py$",
    r"^scripts/bootstrap_milvus\.sh$",  #
    # generated files
    r"^\./run_service\.sh$",
    r"^outputs/line_chart_\d+\.png$",  #
    # virtual environment directories
    r"(\.[a-z_]*env$|^\.[a-z_]*env)",
}

ALLOWLISTED_FILE_PATH_PAIRS_REGEX = list(
    map(lambda x: (re.compile(x[0]), re.compile(x[1])), ALLOWLISTED_FILE_PATH_PAIRS))
ALLOWLISTED_WORDS_REGEX = re.compile(r"^(" + "|".join(ALLOWLISTED_WORDS) + r")$")

IGNORED_FILE_PATH_PAIRS_REGEX = list(map(lambda x: (re.compile(x[0]), re.compile(x[1])), IGNORED_FILE_PATH_PAIRS))
IGNORED_FILES_REGEX = list(map(re.compile, IGNORED_FILES))
IGNORED_PATHS_REGEX = list(map(re.compile, IGNORED_PATHS))

YAML_WHITELISTED_KEYS: set[str] = {
    "model_name",
    "llm_name",
    "tool_name",
    "_type",
    "remote_file_path",
}

# Paths to consider referential -- string
# referential paths are ones that should not only be checked for existence, but also for referential integrity
# (i.e. that the path exists in the same directory as the file)
REFERENTIAL_PATHS: set[str] = {
    "examples",
    "docs",
}

# File extensions to check paths
EXTENSIONS: tuple[str, ...] = ('.ipynb', '.md', '.rst', '.yml', '.yaml', '.json', '.toml', '.ini', '.conf', '.cfg')

URI_OR_PATH_REGEX = re.compile(r'((([^:/?# ]+):)?(//([^/?# ]*))([^?# ]*)(\?([^# ]*))?(#([^ ]*))?'
                               r'|(\.?\.?/?)(([^ \t`=\'"]+/)+[^ \t`=\'"]+))')

PATH_REGEX = re.compile(r'^(\.?\.?/?)(([^ \t`=\'"]+/)+[^ \t`=\'"]+)$')

VALID_PATH_REGEX = re.compile(r'^[A-Za-z0-9_\-\./]+$')


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
    with open(filename, "r", encoding="utf-8") as f:
        section: list[str] = []
        in_skipped_section: bool = False
        skip_next_line: bool = False
        for line_number, line in enumerate(f, start=1):
            if skip_next_line:
                skip_next_line = False
                continue
            if "path-check-skip-file" in line:
                return []
            if "path-check-skip-next-line" in line:
                skip_next_line = True
                continue
            if "path-check-skip-end" in line:
                in_skipped_section = False
            elif "path-check-skip-begin" in line:
                in_skipped_section = True
                continue

            # Handle code blocks in markdown files
            if filename.endswith(".md") and "```" in line:
                index = line.index("```")
                block_type = line[index + 3:].strip()
                # if we have a block type
                if block_type or not section:
                    # ensure that we don't push a single-line block
                    if "```" not in block_type:
                        section.append(block_type)
                else:
                    # if it's empty, then we're done with the section
                    if section:
                        section.pop()

            if filename.endswith("yml") or filename.endswith("yaml") or (section and section[-1] in ["yml", "yaml"]):
                if any((key in line) for key in YAML_WHITELISTED_KEYS):
                    continue
            if in_skipped_section:
                continue
            for match in URI_OR_PATH_REGEX.finditer(line):
                column, _ = match.span()
                path = match.group(0).strip()
                # Exclude URIs and other non-path-like strings
                if not PATH_REGEX.search(path):
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
                if not VALID_PATH_REGEX.match(path):
                    continue
                if ALLOWLISTED_WORDS_REGEX.search(path):
                    continue
                if any(r.search(path) for r in IGNORED_PATHS_REGEX):
                    continue
                if any(r[0].search(filename) and r[1].search(path) for r in IGNORED_FILE_PATH_PAIRS_REGEX):
                    continue
                paths.append(PathInfo(line_number, column + 1, path))
    return paths


def check_files() -> list[tuple[str, PathInfo]]:
    """
    Checks files in the repo for paths that don't exist.

    Skips files that:
    - match any of the ignored files.

    Skips paths that:
    - are absolute paths
    - are URIs
    - are empty
    - are "." or ".."
    - match any of the ignored paths
    - match any of the ignored file-path pairs

    Skips sections of files that:
    - all remaining lines of a file after marked with `path-check-skip-file`
    - are marked with `path-check-skip-begin` / `path-check-skip-end` region
    - are marked on a line after `path-check-skip-next-line`
    - are within a code block
    - are within a YAML block

    Returns:
        A list of tuples of (filename, path) that don't exist.
    """
    filenames_with_broken_paths = []

    skipped_paths: set[str] = set()

    for f in all_files(path_filter=lambda x: x.endswith(EXTENSIONS)):
        if any(r.search(f) for r in IGNORED_FILES_REGEX):
            continue
        paths = extract_paths_from_file(f)

        def check_path(path: str, path_info: PathInfo, f: str) -> bool:
            """
            Checks if a path is valid.

            Args:
                path: The path to check.
                path_info: The path info object.
                f: The filename of the file being checked.

            Returns:
                True if we performed an action based on the path
            """
            path = os.path.normpath(path)
            if not os.path.exists(path):
                return False
            for p in REFERENTIAL_PATHS:
                if p in f and p in path:
                    common = os.path.commonprefix([f, path])[:-1]
                    if (os.path.dirname(f) == common or os.path.dirname(path) == common or os.path.dirname(path) in f):
                        break
                    if not any(r[0].search(f) and r[1].search(path) for r in ALLOWLISTED_FILE_PATH_PAIRS_REGEX):
                        filenames_with_broken_paths.append((f, path_info))
                        break
            return True

        for path_info in paths:
            # attempt to resolve the path relative to the file
            resolved_path = os.path.join(os.path.dirname(f), path_info.path)
            if check_path(resolved_path, path_info, f):
                continue
            # attempt to use the path as-is
            if check_path(path_info.path, path_info, f):
                continue

            # if it still doesn't exist then it's broken
            filenames_with_broken_paths.append((f, path_info))

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
        print("Checking for broken symbolic links...")
        broken_symlinks: list[str] = list_broken_symlinks()
        if broken_symlinks:
            return_code = 1
            print("Found broken symlinks:")
            for symlink in broken_symlinks:
                print(f"  {symlink}")
        print("Done checking for broken symbolic links.")

    if args.check_paths_in_files:
        print("Checking paths within files...")

        broken_paths: list[tuple[str, PathInfo]] = check_files()
        if broken_paths:
            return_code = 1
            print("Failed path checks:")
            for filename, path_info in broken_paths:
                print(f"- {filename}:{path_info.line_number}:{path_info.column} -> {path_info.path}")
            print(
                textwrap.dedent("""
                    Note: If a path exists but is identified here as broken, then it is likely due to the
                          referential integrity check failing. This check is designed to ensure that paths
                          are valid and that they exist in the same directory tree as the file being checked.

                          If you believe this is a false positive, please add the path to the
                          ALLOWLISTED_FILE_PATH_PAIRS set in the path_checks.py file.

                    Note: Some paths may be ignored due to rules:
                        - IGNORED_FILES: files that should be ignored
                        - IGNORED_PATHS: paths that should be ignored
                        - IGNORED_FILE_PATH_PAIRS: file-path pairs that should be ignored
                        - ALLOWLISTED_WORDS: common word groups that should be ignored (and/or, input/output)

                    See ./docs/source/resources/contributing.md#path-checks for more information about path checks.
                    """))
        else:
            print("No failed path checks encountered!")

        print("Done checking paths within files.")

    sys.exit(return_code)


if __name__ == "__main__":
    main()
