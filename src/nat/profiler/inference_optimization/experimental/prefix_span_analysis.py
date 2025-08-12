# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
An advanced script that:

1. Builds chronological call sequences (LLM or TOOL) from a DataFrame of events.
2. Incorporates llm_text_input for LLM calls into the token used by PrefixSpan.
3. Runs PrefixSpan to discover frequent sub-sequences (patterns) across examples.
4. Computes coverage (fraction of examples containing each pattern) and average sub-sequence duration.
5. Returns a Pydantic model with the top patterns plus a textual report.

Main use case:

- Identify recurring sequences of calls + repeated LLM text inputs, which can help with caching or further optimization
  (deduplicate repeated calls or pre-load certain tokens).
"""

import logging

import numpy as np
import pandas as pd

from nat.data_models.intermediate_step import IntermediateStep
from nat.profiler.inference_optimization.data_models import FrequentPattern
from nat.profiler.inference_optimization.data_models import PrefixCallNode
from nat.profiler.inference_optimization.data_models import PrefixSpanSubworkflowResult
from nat.profiler.utils import create_standardized_dataframe

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# 1) Building Sequences (Including llm_text_input)
# --------------------------------------------------------------------------------


def parse_op_type(evt: str) -> str | None:
    """Map event_type => 'LLM' or 'TOOL' if it starts with those prefixes."""
    et = evt.upper()
    if et.startswith("LLM_"):
        return "LLM"
    if et.startswith("TOOL_"):
        return "TOOL"
    return None


def get_op_name(row: pd.Series, op_type: str) -> str:
    """Pick the operation_name from either llm_name or tool_name based on op_type."""
    if op_type == "LLM":
        return row.get("llm_name") or "unknown_llm"
    if op_type == "TOOL":
        return row.get("tool_name") or "unknown_tool"
    return "unknown_op"


def build_call_sequence_for_example(example_df: pd.DataFrame) -> list[PrefixCallNode]:
    """
    For a single example's events, pair START/END calls and build a chronological list of PrefixCallNodes,
    storing llm_text_input if op_type=LLM and it's available at START or END.

    """
    example_df = example_df.sort_values("event_timestamp")
    example_num = int(example_df["example_number"].iloc[0])

    partial_map: dict[str, dict] = {}
    calls_list: list[PrefixCallNode] = []

    for _, row in example_df.iterrows():
        evt_type = row["event_type"].value.upper()
        uuid = str(row["UUID"])
        ts = float(row["event_timestamp"])

        op_type = parse_op_type(evt_type)
        if not op_type:
            # ignore events that are not LLM_/TOOL_
            continue

        if evt_type.endswith("_START"):
            op_name = get_op_name(row, op_type)
            call_info = {
                "uuid": uuid,
                "example_number": example_num,
                "operation_type": op_type,
                "operation_name": op_name,
                "start_time": ts,
                "llm_text_input": None
            }
            # If llm_text_input is present in START
            if op_type == "LLM" and "llm_text_input" in row and pd.notna(row["llm_text_input"]):
                call_info["llm_text_input"] = str(row["llm_text_input"])
            partial_map[uuid] = call_info

        elif evt_type.endswith("_END"):
            if uuid in partial_map:
                # finalize
                start_info = partial_map[uuid]
                end_time = ts
                duration = max(0.0, end_time - start_info["start_time"])
                # If we only have llm_text_input at END, override if not present
                if op_type == "LLM" and "llm_text_input" in row and pd.notna(row["llm_text_input"]):
                    start_info["llm_text_input"] = str(row["llm_text_input"])

                node = PrefixCallNode(uuid=uuid,
                                      example_number=example_num,
                                      operation_type=start_info["operation_type"],
                                      operation_name=start_info["operation_name"],
                                      start_time=start_info["start_time"],
                                      end_time=end_time,
                                      duration=duration,
                                      llm_text_input=start_info["llm_text_input"])
                calls_list.append(node)
                del partial_map[uuid]

    # Sort final calls by start_time
    calls_list.sort(key=lambda c: c.start_time)
    return calls_list


def build_sequences(df: pd.DataFrame) -> dict[int, list[PrefixCallNode]]:
    """
    Group events by example_number, build a chronological list of PrefixCallNode for each example,
    including the LLM text input if present.
    """
    dfc = df.copy()
    dfc.sort_values(["example_number", "event_timestamp"], inplace=True)

    sequences_map = {}
    for ex_num, group_df in dfc.groupby("example_number"):
        seq_calls = build_call_sequence_for_example(group_df)
        sequences_map[ex_num] = seq_calls
    return sequences_map


# --------------------------------------------------------------------------------
# 2) Token Construction & PrefixSpan
# --------------------------------------------------------------------------------


def build_token(call: PrefixCallNode, max_text_len: int = 20, prefix_list: list[str] = None) -> str:
    """
    Construct a token for prefixspan from a PrefixCallNode.
    - We do "LLM:{operation_name}|{text}" if it's an LLM call and text is available
    - We optionally truncate or hash the text for length. Here we just do naive truncation
    - For a tool call, we do "TOOL:{operation_name}"
    """
    if call.operation_type == "LLM":
        text_part = ""
        if call.llm_text_input:
            # naive truncation
            truncated = call.llm_text_input

            # Check truncated text for an exact match of any string in prefix_list
            # Does not have to be in just the prefix, but anywhere
            # Replaces the matched string with <common_prefix>
            if prefix_list:
                for prefix in prefix_list:
                    for i in range(len(prefix), 0, -1):
                        if truncated.startswith(prefix[:i]):
                            truncated = truncated.replace(prefix[:i], "<common_prefix>")
                            break

            truncated = truncated[:max_text_len].replace("\n", " ")
            text_part = f"|{truncated}"
        return f"LLM:{call.operation_name}{text_part}"

    return f"TOOL:{call.operation_name}"


def convert_sequences_for_prefixspan(sequences_map: dict[int, list[PrefixCallNode]],
                                     max_text_len: int = 20,
                                     prefix_list: list[str] = None) -> list[list[str]]:
    """
    Convert each example's list of PrefixCallNode into a list of tokens. Return a list-of-lists
    suitable for prefixspan. E.g.::

        [
        ["LLM:llama-3|Hello", "TOOL:internet-search", "LLM:llama-3|How are you?"],
        ["LLM:davinci|some prompt", "TOOL:vector-db"]
        ...
        ]

    """
    result = []
    for _, call_list in sequences_map.items():
        token_list = [build_token(c, max_text_len, prefix_list) for c in call_list]
        result.append(token_list)
    return result


def run_prefixspan(sequences_map: dict[int, list[PrefixCallNode]],
                   min_support: int | float,
                   max_text_len: int = 20,
                   prefix_list: list[str] = None) -> list[tuple[list[str], int]]:
    """
    1) Convert all example sequences => tokens
    2) Run prefixspan with min_support
    3) Return (pattern, freq) list
    """

    try:
        from prefixspan import PrefixSpan
    except ImportError:
        logger.error("prefixspan is not installed. Please install prefixspan to run the prefix analysis in the "
                     "profiler or install `aiq[profiler]` to install all necessary profiling packages.")

        raise

    token_seqs = convert_sequences_for_prefixspan(sequences_map, max_text_len, prefix_list)

    ps = PrefixSpan(token_seqs)

    # Convert min_support if float => absolute freq
    # prefixspan interprets min_support as an absolute occurrence count
    if isinstance(min_support, float):
        total_seq_count = len(token_seqs)
        abs_min_support = max(1, int(round(min_support * total_seq_count)))
    else:
        abs_min_support = min_support

    freq_patterns = ps.frequent(abs_min_support)  # pylint: disable=not-callable
    # freq_patterns => [(count, [item1, item2, ...])]

    results = []
    for (count, pat) in freq_patterns:
        results.append((pat, count))
    return results


# --------------------------------------------------------------------------------
# 3) Coverage & Duration Computation
# --------------------------------------------------------------------------------


def find_contiguous_matches(pattern: list[str], seq: list[str]) -> list[tuple[int, int]]:
    """
    Look for contiguous matches of 'pattern' in 'seq' by naive scanning.
    e.g. pattern=["LLM:llama-3|Hello", "TOOL:internet-search"], seq=...
    Return list of (start_idx, end_idx).
    """
    matches = []
    plen = len(pattern)
    slen = len(seq)
    for start in range(slen - plen + 1):
        if seq[start:start + plen] == pattern:
            matches.append((start, start + plen - 1))
    return matches


def compute_coverage_and_duration(sequences_map: dict[int, list[PrefixCallNode]],
                                  prefixspan_patterns: list[tuple[list[str], int]],
                                  top_k: int,
                                  min_coverage: float = 0.0,
                                  max_text_len: int = 20) -> list[FrequentPattern]:
    """
    For each pattern from prefixspan, compute:

    - coverage: fraction of examples that contain it
    - average_duration: sum of durations of calls in sub-sequence / total occurrences

    Then filter by min_coverage and pick top_k, sorted by frequency, coverage, avg_duration desc.
    """
    # We'll also rebuild token sequences for matching
    token_sequences = {}
    call_sequences = {}
    for ex_num, call_list in sequences_map.items():
        token_seq = [build_token(c, max_text_len) for c in call_list]
        token_sequences[ex_num] = token_seq
        call_sequences[ex_num] = call_list

    total_examples = len(token_sequences)
    results: list[FrequentPattern] = []

    for (pat, freq) in prefixspan_patterns:
        # coverage => how many distinct example_num have at least one contiguous match
        examples_with_pattern = []
        total_occ = 0
        total_dur = 0.0

        for ex_num, token_seq in token_sequences.items():
            matches = find_contiguous_matches(pat, token_seq)
            if matches:
                examples_with_pattern.append(ex_num)
                # sum durations for each occurrence
                calls = call_sequences[ex_num]
                for (start_idx, end_idx) in matches:
                    dur_sum = float(np.sum([calls[i].duration for i in range(start_idx, end_idx + 1)]))
                    total_dur += dur_sum
                    total_occ += 1

        coverage_val = len(examples_with_pattern) / total_examples if total_examples > 0 else 0.0
        if coverage_val < min_coverage:
            continue

        avg_dur = total_dur / total_occ if total_occ > 0 else 0.0

        fp = FrequentPattern(pattern=pat,
                             frequency=freq,
                             coverage=coverage_val,
                             average_duration=avg_dur,
                             examples_containing=sorted(examples_with_pattern))
        results.append(fp)

    # sort & top_k
    results.sort(key=lambda p: (p.frequency, p.coverage, p.average_duration), reverse=True)
    return results[:top_k]


# --------------------------------------------------------------------------------
# 4) Main Entry Function
# --------------------------------------------------------------------------------


def prefixspan_subworkflow_with_text(  # pylint: disable=too-many-positional-arguments
        all_steps: list[list[IntermediateStep]],
        min_support: int | float = 2,
        top_k: int = 10,
        min_coverage: float = 0.0,
        max_text_len: int = 700,
        prefix_list: list[str] = None) -> PrefixSpanSubworkflowResult:
    """
    1) Build sequences of calls for each example (with llm_text_input).
    2) Convert to token lists, run PrefixSpan with min_support.
    3) Compute coverage & average duration for each pattern, filter by min_coverage, pick top_k.
    4) Return Pydantic model with final patterns & textual report.

    :param all_steps: Intermediate steps
    :param min_support: minimal # of times (int) or fraction (float) for prefixspan
    :param top_k: how many patterns to keep
    :param min_coverage: discard patterns that appear in fewer than this fraction of examples
    :param max_text_len: how many chars of llm_text_input to incorporate in the token
    :param prefix_list: list of prefixes to filter on and exclude from pattern matching
    """
    df = create_standardized_dataframe(all_steps)
    # Validate columns
    required_cols = {
        "framework",
        "tool_name",
        "llm_name",
        "llm_text_input",
        "llm_text_output",
        "event_timestamp",
        "event_type",
        "UUID",
        "example_number",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # 1) Build sequences
    sequences_map = build_sequences(df)
    total_examples = len(sequences_map)

    # 2) prefixspan
    prefixspan_patterns = run_prefixspan(sequences_map,
                                         min_support=min_support,
                                         max_text_len=max_text_len,
                                         prefix_list=prefix_list)
    if not prefixspan_patterns:
        return PrefixSpanSubworkflowResult(
            patterns=[], textual_report="No frequent patterns found by PrefixSpan with the given min_support.")

    # 3) coverage & duration
    final_patterns = compute_coverage_and_duration(sequences_map,
                                                   prefixspan_patterns,
                                                   top_k=top_k,
                                                   min_coverage=min_coverage,
                                                   max_text_len=max_text_len)
    if not final_patterns:
        return PrefixSpanSubworkflowResult(patterns=[],
                                           textual_report="No patterns passed coverage/duration thresholds.")

    # 4) Build textual report
    lines = []
    lines.append("=== PrefixSpan Sub-Workflow Mining w/ LLM Text ===")
    lines.append(f"Total examples: {total_examples}")
    lines.append(f"min_support={min_support}, top_k={top_k}, min_coverage={min_coverage}, max_text_len={max_text_len}")
    lines.append(f"Patterns discovered: {len(final_patterns)}")

    for i, pat in enumerate(final_patterns, start=1):
        chain_str = " -> ".join(pat.pattern)
        lines.append(f"\n{i}) Pattern: {chain_str}")
        lines.append(f"   Frequency: {pat.frequency}")
        lines.append(f"   Coverage: {pat.coverage:.2f}  (appears in {len(pat.examples_containing)} examples)")
        lines.append(f"   Avg Duration: {pat.average_duration:.2f} seconds")
        lines.append(f"   Examples containing: {pat.examples_containing}")

    report_text = "\n".join(lines)

    # 5) Return final model
    return PrefixSpanSubworkflowResult(patterns=final_patterns, textual_report=report_text)
