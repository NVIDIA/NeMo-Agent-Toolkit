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

from nat.data_models.intermediate_step import IntermediateStep
from nat.profiler.inference_optimization.data_models import CommonPrefixesOutput
from nat.profiler.inference_optimization.data_models import FrameworkLLMPrefixData
from nat.profiler.inference_optimization.data_models import PrefixInfo
from nat.profiler.utils import create_standardized_dataframe


# -----------------------------------------------------------
# 1. Helper: Build a prefix trie
# -----------------------------------------------------------
def build_prefix_trie(strings: list[str]) -> dict:
    """
    Build a trie from a list of strings.

    Returns a nested dictionary with::

        {
            'count': int,         # number of strings passing through this node
            'children': dict[str, TrieNode]
        }

    """
    root = {'count': 0, 'children': {}}
    for s in strings:
        node = root
        node['count'] += 1  # every string passes through the root
        for ch in s:
            if ch not in node['children']:
                node['children'][ch] = {'count': 0, 'children': {}}
            node = node['children'][ch]
            node['count'] += 1
    return root


# -----------------------------------------------------------
# 2. Helper: Iterative traversal of the trie
# -----------------------------------------------------------
def collect_prefixes_iterative(root: dict, total_calls: int) -> list[dict]:
    """
    Iteratively traverse the trie to collect prefix statistics,
    avoiding recursion depth limits.

    :param root: Trie node with 'count' and 'children'
    :param total_calls: Number of total calls in this group (denominator for percentages)
    :return: A list of dicts, each dict containing prefix info
    """
    results = []
    # stack holds (node, prefix_so_far)
    stack = [(root, "")]

    while stack:
        node, prefix = stack.pop()

        # Skip storing the empty root prefix
        if prefix:
            calls_count = node['count']
            calls_percentage = calls_count / total_calls
            results.append({
                'prefix': prefix,
                'prefix_length': len(prefix),
                'calls_count': calls_count,
                'calls_percentage': calls_percentage
            })

        # Add children to the stack
        for ch, child_node in node['children'].items():
            stack.append((child_node, prefix + ch))

    return results


# -----------------------------------------------------------
# 3. Main Function
# -----------------------------------------------------------
def get_common_prefixes(all_steps: list[list[IntermediateStep]],
                        min_call_percentage: float = 0.0) -> CommonPrefixesOutput:
    """
    Given a pandas DataFrame with columns 'framework', 'llm_name',
    and 'llm_text_input', return a Pydantic-validated RootModel
    keyed by "<llm_name>" with a sorted list of
    common prefix statistics.

    1) Only includes prefixes with calls_percentage >= `min_call_percentage`.
    2) Excludes any prefix that is a substring of another (longer) prefix
       that already meets the threshold and is retained.
    3) Optionally writes the resulting dictionary to JSON if `output_path` is provided.

    :param all_steps: Intermediate Steps
    :param min_call_percentage: Exclude prefixes that appear in fewer than this fraction
                                of total calls. (Default 0.0 = no filtering)

    Sorting: primarily by prefix length (descending),
             secondarily by frequency (descending).
    """
    # Validate necessary columns
    df = create_standardized_dataframe(all_steps)

    required_cols = {'framework', 'llm_name', 'llm_text_input'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    output_data: dict[str, FrameworkLLMPrefixData] = {}

    # Group DataFrame by (framework, llm_name)
    grouped = df.groupby(['llm_name'])
    for llm_name, group_df in grouped:
        # Unpack llm_name Tuple
        llm_name = llm_name[0]

        text_inputs = group_df['llm_text_input'].astype(str).tolist()
        total_calls = len(text_inputs)

        # Build trie for all text inputs
        trie = build_prefix_trie(text_inputs)

        # Collect prefix info using iterative traversal
        results = collect_prefixes_iterative(trie, total_calls=total_calls)

        # 1) Filter out prefixes below min_call_percentage
        results_filtered = [r for r in results if r['calls_percentage'] >= min_call_percentage]

        # 2) Sort results: prefix_length desc, then calls_count desc
        results_sorted = sorted(results_filtered, key=lambda x: (x['prefix_length'], x['calls_count']), reverse=True)

        # 3) Substring filtering:
        #    Because results_sorted is in descending length order,
        #    if we keep a prefix, we exclude any shorter prefix that
        #    is a substring of that already-kept prefix.
        final_results = []
        for r in results_sorted:
            pfx = r['prefix']
            # Check if this prefix is contained in any longer prefix we have kept
            if not any(pfx in kept['prefix'] for kept in final_results):
                final_results.append(r)

        # Convert each dict to a PrefixInfo model
        prefix_info_list = [PrefixInfo(**res) for res in final_results]

        # Construct the dictionary key
        framework_llm_key = f"{llm_name}"

        # Save the data for this group
        output_data[framework_llm_key] = FrameworkLLMPrefixData(total_calls=total_calls, prefix_info=prefix_info_list)

    # Package the final result in a validated RootModel
    result_model = CommonPrefixesOutput(root=output_data)
    return result_model
