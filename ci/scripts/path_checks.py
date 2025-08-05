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

from gitutils import all_files

# File path pairs to ignore -- first is the file path, second is the path in the file
IGNORED_FILE_PATH_PAIRS: set[tuple[str, str]] = {
    # allow references to docs from examples
    (r"^examples/", r"^(\.\./)*docs/"),
    # allow references to examples from docs
    (r"^docs/", r"^(\.\./)*examples/"),
    # allow relative references within docs
    (r"^docs/", r"^(\.?\./)*[A-Za-z0-9_\-\.]+"),
    # allow references to subdirectories from examples
    (r"^examples/", r"^\./[A-Za-z0-9_\-\.]+/"),

    # allow references from some examples to other examples
    (r"^examples/MCP/simple_calculator_mcp/README.md", r"^\.\./\.\./getting_started/simple_calculator"),
    (r"^examples/custom_functions/automated_description_generation/README.md", r"^\.\./\.\./RAG/simple_rag/README.md"),
    (r"^examples/evaluation_and_profiling/simple_calculator_eval/README.md",
     r"^\.\./\.\./getting_started/simple_calculator"),
    (r"^examples/evaluation_and_profiling/simple_web_query_eval/README.md",
     r"^\.\./\.\./getting_started/simple_web_query"),
    (r"^examples/observability/simple_calculator_observability/README.md",
     r"^\.\./\.\./getting_started/simple_calculator"),
}

# Paths to ignore -- regex pattern
IGNORED_PATHS: set[str] = {
    r"^(\./)?\.tmp/",
    r"^\./upload_to_minio\.sh$",
    r"^\./upload_to_mysql\.sh$",
    r"^\./start_local_sandbox\.sh$",
    r"^\.venv/bin/activate$",
    r"^\./run_service\.sh$"
}

# Files to ignore -- regex pattern
IGNORED_FILES: set[str] = {
    # ignore github and gitlab files
    r"^\.github",
    r"^\.gitlab-ci.yml",  # ignore swe_bench dataset files
    r"^examples/evaluation_and_profiling/swe_bench/src/aiq_swe_bench/data/test_dataset_lite\.json$",
    r"^examples/evaluation_and_profiling/swe_bench/src/aiq_swe_bench/data/test_dataset_verified\.json$",
    r"^examples/evaluation_and_profiling/swe_bench/src/aiq_swe_bench/data/dev_dataset_lite\.json$",
    # ignore pyproject.toml files
    r"pyproject\.toml$",
}

# Paths to consider referential -- string
# referential paths are ones that should not only be checked for existence, but also for referential integrity
# (i.e. that the path exists in the same directory as the file)
REFERENTIAL_PATHS: set[str] = {
    "examples",
    "docs",
}

# File extensions to check paths
EXTENSIONS: tuple[str, ...] = ('.md', '.rst', '.yml', '.yaml', '.json', '.toml', '.ini', '.conf', '.cfg')


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


def extract_paths_from_file(filename: str) -> list[str]:
    """
    Extracts paths from a file. Skips absolute paths and "." and ".." paths.
    Args:
        filename: The path to the file to extract paths from.
    Returns:
        A list of paths.
    """
    # This is a regex to match URIs or file paths
    path_regex = re.compile(r'((\w+://[^\s\'"<>]+)|(\.?\.?/?)(([A-Za-z0-9_\-\.]+/)*[A-Za-z0-9_\-\.]+)|)')
    paths = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            for match in path_regex.finditer(line):
                path = match.group(0)
                # Exclude URIs
                if "://" in path:
                    continue
                # Exclude absolute paths
                if path.startswith('/'):
                    continue
                # Exclude paths that don't start with "."
                if not path.startswith('.'):
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
                paths.append(path)
    return paths


def check_modified_files(extensions: tuple[str, ...]) -> list[tuple[str, str]]:
    """
    Checks all files in the repo for paths that don't exist.

    Args:
        extensions: A tuple of file extensions to check.
        path_filter: Optional regex pattern to filter file paths found within the files

    Returns:
        A list of tuples of (filename, path) that don't exist.
    """
    filenames_with_broken_paths = []

    # TODO: use modified_files instead of all_files
    modified: list[str] = all_files(path_filter=lambda x: x.endswith(extensions))

    for f in modified:
        paths = extract_paths_from_file(f)
        for p in paths:
            # If the path does not exist, then it is broken
            if not os.path.exists(p):
                filenames_with_broken_paths.append((f, p))
                continue
            path = os.path.commonpath([f, p])
            if path == "" or path == ".":
                # try to resolve the path relative to the file
                if not os.path.exists(os.path.join(os.path.dirname(f), p)):
                    filenames_with_broken_paths.append((f, p))
            else:
                for r in REFERENTIAL_PATHS:
                    if r in path:
                        # If the path is a single directory (which would be a referential path),
                        # then it is likely broken since it would be referring to outside the current
                        # directory
                        if len(path.split(os.path.sep)) == 1:
                            filenames_with_broken_paths.append((f, p))
                            break

    return filenames_with_broken_paths


def main():
    """Main function to handle command line arguments and execute checks."""
    parser = argparse.ArgumentParser(description='Check for broken symlinks and paths in files')
    parser.add_argument('--check-broken-symlinks', action='store_true', help='Check for broken symbolic links')
    parser.add_argument('--check-paths-in-files', action='store_true', help='Check for broken paths in files')

    args = parser.parse_args()

    return_code: int = 0

    # Check for broken symlinks
    if args.check_broken_symlinks:
        print("Checking for broken symlinks...")
        broken_symlinks: list[str] = list_broken_symlinks()
        if broken_symlinks:
            return_code = 1
            print("Found broken symlinks:")
            for symlink in broken_symlinks:
                print(f"  {symlink}")
        print("Done checking for broken symlinks.")

    # Check for broken paths in files
    if args.check_paths_in_files:
        print("Checking for broken paths in files...")

        def ignored_files(filename: str) -> bool:
            return any(re.search(r, filename) for r in IGNORED_FILES)

        def ignored_paths(path: str) -> bool:
            return any(re.search(r, path) for r in IGNORED_PATHS)

        def ignored_file_path_pairs(pair: tuple[str, str]) -> bool:
            return any(re.search(r[0], pair[0]) and re.search(r[1], pair[1]) for r in IGNORED_FILE_PATH_PAIRS)

        def all_filters(pair: tuple[str, str]) -> bool:
            return ignored_file_path_pairs(pair) or ignored_files(pair[0]) or ignored_paths(pair[1])

        broken_paths: list[tuple[str, str]] = check_modified_files(EXTENSIONS)
        if broken_paths:
            broken_paths = list(filter(lambda x: not all_filters(x), broken_paths))
            if broken_paths:
                return_code = 1
                print("Found \"broken\" paths in files:")
                for filename, path in broken_paths:
                    print(f"  {filename}: {path}")

        print("Done checking for broken paths in files.")

    sys.exit(return_code)


if __name__ == "__main__":
    main()
