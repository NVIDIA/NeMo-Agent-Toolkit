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

"""Advanced replacement strategies used by file-edit tooling."""

from __future__ import annotations

import re
from collections.abc import Callable, Generator
from dataclasses import dataclass

from nat.workspace_actions.workspace.utils.levenshtein import levenshtein_distance


Replacer = Callable[[str, str], Generator[str, None, None]]


SINGLE_CANDIDATE_SIMILARITY_THRESHOLD = 0.0
MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD = 0.3


def _substring_from_line_range(
    content: str,
    lines: list[str],
    start_line: int,
    end_line: int,
) -> str:
    start_index = 0
    for idx in range(start_line):
        start_index += len(lines[idx]) + 1

    end_index = start_index
    for idx in range(start_line, end_line + 1):
        end_index += len(lines[idx])
        if idx < end_line:
            end_index += 1

    return content[start_index:end_index]


def simple_replacer(_content: str, find: str) -> Generator[str, None, None]:
    """Yield the search string as-is."""
    yield find


def line_trimmed_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Match a block by comparing line content after trimming whitespace."""
    original_lines = content.split("\n")
    search_lines = find.split("\n")

    if search_lines and search_lines[-1] == "":
        search_lines.pop()

    if not search_lines:
        return

    for i in range(0, len(original_lines) - len(search_lines) + 1):
        matches = True
        for j in range(len(search_lines)):
            if original_lines[i + j].strip() != search_lines[j].strip():
                matches = False
                break

        if matches:
            end_line = i + len(search_lines) - 1
            yield _substring_from_line_range(content, original_lines, i, end_line)


def block_anchor_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Use first/last lines as anchors, fuzzy-matching middle lines."""
    original_lines = content.split("\n")
    search_lines = find.split("\n")

    if len(search_lines) < 3:
        return

    if search_lines and search_lines[-1] == "":
        search_lines.pop()

    if len(search_lines) < 3:
        return

    first_line_search = search_lines[0].strip()
    last_line_search = search_lines[-1].strip()
    search_block_size = len(search_lines)

    candidates: list[tuple[int, int]] = []
    for i, line in enumerate(original_lines):
        if line.strip() != first_line_search:
            continue
        for j in range(i + 2, len(original_lines)):
            if original_lines[j].strip() == last_line_search:
                candidates.append((i, j))
                break

    if not candidates:
        return

    if len(candidates) == 1:
        start_line, end_line = candidates[0]
        actual_block_size = end_line - start_line + 1

        similarity = 0.0
        lines_to_check = min(search_block_size - 2, actual_block_size - 2)

        if lines_to_check > 0:
            upper = min(search_block_size - 1, actual_block_size - 1)
            for j in range(1, upper):
                original_line = original_lines[start_line + j].strip()
                search_line = search_lines[j].strip()
                max_len = max(len(original_line), len(search_line))
                if max_len == 0:
                    continue
                distance = levenshtein_distance(original_line, search_line)
                similarity += (1 - distance / max_len) / lines_to_check
                if similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD:
                    break
        else:
            similarity = 1.0

        if similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD:
            yield _substring_from_line_range(
                content,
                original_lines,
                start_line,
                end_line,
            )
        return

    best_match: tuple[int, int] | None = None
    max_similarity = -1.0

    for start_line, end_line in candidates:
        actual_block_size = end_line - start_line + 1
        similarity = 0.0
        lines_to_check = min(search_block_size - 2, actual_block_size - 2)

        if lines_to_check > 0:
            upper = min(search_block_size - 1, actual_block_size - 1)
            for j in range(1, upper):
                original_line = original_lines[start_line + j].strip()
                search_line = search_lines[j].strip()
                max_len = max(len(original_line), len(search_line))
                if max_len == 0:
                    continue
                distance = levenshtein_distance(original_line, search_line)
                similarity += 1 - distance / max_len
            similarity /= lines_to_check
        else:
            similarity = 1.0

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = (start_line, end_line)

    if max_similarity >= MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD and best_match:
        start_line, end_line = best_match
        yield _substring_from_line_range(content, original_lines, start_line, end_line)


def whitespace_normalized_replacer(
    content: str,
    find: str,
) -> Generator[str, None, None]:
    """Match with normalized whitespace."""

    def normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    normalized_find = normalize_whitespace(find)
    lines = content.split("\n")

    for line in lines:
        if normalize_whitespace(line) == normalized_find:
            yield line
        else:
            normalized_line = normalize_whitespace(line)
            if normalized_find in normalized_line:
                words = re.split(r"\s+", find.strip())
                if words and words[0]:
                    pattern = r"\s+".join(re.escape(word) for word in words)
                    try:
                        regex = re.compile(pattern)
                    except re.error:
                        regex = None
                    if regex:
                        match = regex.search(line)
                        if match:
                            yield match.group(0)

    find_lines = find.split("\n")
    if len(find_lines) > 1:
        for i in range(0, len(lines) - len(find_lines) + 1):
            block = "\n".join(lines[i : i + len(find_lines)])
            if normalize_whitespace(block) == normalized_find:
                yield block


def indentation_flexible_replacer(
    content: str,
    find: str,
) -> Generator[str, None, None]:
    """Match a block by removing shared indentation."""

    def remove_indentation(text: str) -> str:
        lines = text.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            return text

        min_indent = min(
            len(re.match(r"^(\s*)", line).group(1)) for line in non_empty_lines
        )
        return "\n".join(
            line if not line.strip() else line[min_indent:] for line in lines
        )

    normalized_find = remove_indentation(find)
    content_lines = content.split("\n")
    find_lines = find.split("\n")

    if not find_lines:
        return

    for i in range(0, len(content_lines) - len(find_lines) + 1):
        block = "\n".join(content_lines[i : i + len(find_lines)])
        if remove_indentation(block) == normalized_find:
            yield block


def escape_normalized_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Unescape common escape sequences in search text, then match."""

    def unescape_string(value: str) -> str:
        def replace_match(match: re.Match[str]) -> str:
            captured = match.group(1)
            mapping = {
                "n": "\n",
                "t": "\t",
                "r": "\r",
                "'": "'",
                '"': '"',
                "`": "`",
                "\\": "\\",
                "\n": "\n",
                "$": "$",
            }
            return mapping.get(captured, match.group(0))

        return re.sub(r"\\(n|t|r|'|\"|`|\\|\n|\$)", replace_match, value)

    unescaped_find = unescape_string(find)

    if unescaped_find in content:
        yield unescaped_find

    lines = content.split("\n")
    find_lines = unescaped_find.split("\n")
    if not find_lines:
        return

    for i in range(0, len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        unescaped_block = unescape_string(block)
        if unescaped_block == unescaped_find:
            yield block


def trimmed_boundary_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Trim search boundaries and try matching."""
    trimmed_find = find.strip()
    if trimmed_find == find:
        return

    if trimmed_find in content:
        yield trimmed_find

    lines = content.split("\n")
    find_lines = find.split("\n")
    if not find_lines:
        return

    for i in range(0, len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        if block.strip() == trimmed_find:
            yield block


def context_aware_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Context-line matching heuristic using first/last lines as anchors."""
    find_lines = find.split("\n")
    if len(find_lines) < 3:
        return

    if find_lines and find_lines[-1] == "":
        find_lines.pop()

    if len(find_lines) < 3:
        return

    content_lines = content.split("\n")
    first_line = find_lines[0].strip()
    last_line = find_lines[-1].strip()

    for i, line in enumerate(content_lines):
        if line.strip() != first_line:
            continue

        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() == last_line:
                block_lines = content_lines[i : j + 1]
                block = "\n".join(block_lines)

                if len(block_lines) == len(find_lines):
                    matching_lines = 0
                    total_non_empty_lines = 0

                    for k in range(1, len(block_lines) - 1):
                        block_line = block_lines[k].strip()
                        find_line = find_lines[k].strip()
                        if block_line or find_line:
                            total_non_empty_lines += 1
                            if block_line == find_line:
                                matching_lines += 1

                    if (
                        total_non_empty_lines == 0
                        or matching_lines / total_non_empty_lines >= 0.5
                    ):
                        yield block
                        break
                break


def multi_occurrence_replacer(content: str, find: str) -> Generator[str, None, None]:
    """Yield a candidate for each non-overlapping exact occurrence."""
    if not find:
        return

    start_index = 0
    while True:
        index = content.find(find, start_index)
        if index == -1:
            break
        yield find
        start_index = index + len(find)


ALL_REPLACERS: list[Replacer] = [
    simple_replacer,
    line_trimmed_replacer,
    block_anchor_replacer,
    whitespace_normalized_replacer,
    indentation_flexible_replacer,
    escape_normalized_replacer,
    trimmed_boundary_replacer,
    context_aware_replacer,
    multi_occurrence_replacer,
]


@dataclass(frozen=True)
class ReplacementResult:
    """Result payload for an advanced replacement attempt."""

    new_content: str
    replaced_count: int
    replacer_used: str
    matched_string: str


def perform_advanced_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
) -> ReplacementResult | None:
    """Try replacement with cascading fallback strategies."""
    if old_string == new_string:
        return None

    replacer_names = [
        "SimpleReplacer",
        "LineTrimmedReplacer",
        "BlockAnchorReplacer",
        "WhitespaceNormalizedReplacer",
        "IndentationFlexibleReplacer",
        "EscapeNormalizedReplacer",
        "TrimmedBoundaryReplacer",
        "ContextAwareReplacer",
        "MultiOccurrenceReplacer",
    ]

    for replacer, replacer_name in zip(ALL_REPLACERS, replacer_names, strict=True):
        for search in replacer(content, old_string):
            index = content.find(search)
            if index == -1:
                continue

            if replace_all:
                if not search:
                    continue
                return ReplacementResult(
                    new_content=content.replace(search, new_string),
                    replaced_count=content.count(search),
                    replacer_used=replacer_name,
                    matched_string=search,
                )

            last_index = content.rfind(search)
            if index != last_index:
                continue

            return ReplacementResult(
                new_content=(
                    content[:index] + new_string + content[index + len(search) :]
                ),
                replaced_count=1,
                replacer_used=replacer_name,
                matched_string=search,
            )

    return None


def has_multiple_matches(content: str, old_string: str) -> bool:
    """Return whether search appears multiple times via any replacer."""
    for replacer in ALL_REPLACERS:
        match_count = 0
        for search in replacer(content, old_string):
            first_index = content.find(search)
            if first_index == -1:
                continue

            last_index = content.rfind(search)
            if first_index != last_index:
                return True

            match_count += 1
            if match_count > 1:
                return True

    return False
