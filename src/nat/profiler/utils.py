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

import inspect
import logging
import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.type_registry import RegisteredFunctionInfo
from nat.data_models.intermediate_step import IntermediateStep
from nat.profiler.data_frame_row import DataFrameRow

# A simple set of regex patterns to scan for direct references to LLMFrameworkEnum
_FRAMEWORK_REGEX_MAP = {t: fr'\b{t._name_}\b' for t in LLMFrameworkEnum}

logger = logging.getLogger(__name__)


def detect_llm_frameworks_in_build_fn(registration: RegisteredFunctionInfo) -> list[LLMFrameworkEnum]:
    """
    Analyze a function's source (the build_fn) to see which LLM frameworks it uses. Also recurses
    into any additional Python functions that the build_fn calls while passing `builder`, so that
    references to LLMFrameworkEnum in those helper calls are also detected.

    1. If `registration.framework_wrappers` is non-empty, we return that first.
       (We do convert them to LLMFrameworkEnum if possible.)
    2. Otherwise, we attempt to:

       - Get the build_fn's source via `inspect.getsource(...)`
       - Parse it for references to LLMFrameworkEnum
       - Find any function calls that include the word "builder" in the arguments

         - Recursively parse those functions' source code for frameworks

    3. If we cannot parse the source at all (e.g. OSError), we return a list of all frameworks.
    """
    # ----------------------------------------------------------------
    # 1) If frameworks were explicitly declared in registration.framework_wrappers, use them:
    if registration.framework_wrappers:
        results: list[LLMFrameworkEnum] = []
        for fw_str in registration.framework_wrappers:
            try:
                results.append(LLMFrameworkEnum(fw_str))
            except ValueError:
                # If it's not recognized, ignore or log
                logger.warning("Unrecognized framework %s in registration.framework_wrappers", fw_str)

        return list(set(results))  # unique
    # ----------------------------------------------------------------

    # Because we want to recursively parse code, we'll keep track of visited function objects
    visited_fns: set[Callable[..., Any]] = set()
    # We also need a place to store discovered frameworks
    discovered: set[LLMFrameworkEnum] = set()

    def _parse_source_for_frameworks(src: str) -> None:
        """Check lines for any direct references to LLMFrameworkEnum.* or placeholders in the map."""
        for fw_enum_member, pattern in _FRAMEWORK_REGEX_MAP.items():
            if re.search(pattern, src):
                discovered.add(fw_enum_member)

    def _find_builder_func_calls(src: str) -> list[str]:
        """
        Look for calls of the form:   some_func(..., builder, ...)
        or   some_func(..., builder=..., ...)

        This returns the name of each function we found being called, e.g. 'some_func'.
        It's a naive best-effort approach
        and group(1) is the function name.
        """
        # E.g.  foo(builder) or foo( param=..., builder=builder )
        pattern = r'(\w+)\s*\([^)]*\bbuilder\b[^)]*\)'
        return re.findall(pattern, src)

    def _recurse_parse(fn: Callable[..., Any], visited: set[Callable[..., Any]]) -> None:
        """Recursively parse the source code of `fn`, add discovered frameworks,
           and parse any new functions that get called with 'builder'."""
        if fn in visited:
            return
        visited.add(fn)

        try:
            src = inspect.getsource(fn)
        except OSError:
            # If we can't parse source, we add all frameworks and bail
            discovered.update([k for k, v in _FRAMEWORK_REGEX_MAP.items()])
            return

        # parse direct references
        _parse_source_for_frameworks(src)

        # parse any function calls that pass in "builder"
        child_func_names = _find_builder_func_calls(src)
        if not child_func_names:
            return

        # We'll try to find these child functions in the same module as `fn`
        mod = inspect.getmodule(fn)
        if not mod:
            return
        # We'll see if the child function is a top-level in that module
        for child_name in child_func_names:
            # get the function object if it exists in the module
            child_obj = getattr(mod, child_name, None)
            if callable(child_obj):
                _recurse_parse(child_obj, visited)

    # ----------------------------------------------------------------
    # 2) Actually do the BFS/DFS parse on `registration.build_fn`
    main_fn = registration.build_fn

    try:
        _recurse_parse(main_fn, visited_fns)
    except Exception:
        # If an unexpected error occurs, fallback to "all frameworks"
        discovered.update([k for k, v in _FRAMEWORK_REGEX_MAP.items()])
    # ----------------------------------------------------------------
    if len(discovered) > 0:
        logger.warning(
            "Discovered frameworks: %s in function %s by inspecting "
            "source. It is recommended and more reliable to instead add the used LLMFrameworkEnum "
            "types in the framework_wrappers argument when calling @register_function.",
            discovered,
            main_fn.__name__)

    return list(discovered)


# -------------------------------------------------------------------
# Create a single standardized DataFrame for all usage stats
# -------------------------------------------------------------------
def create_standardized_dataframe(requests_data: list[list[IntermediateStep]]) -> pd.DataFrame:
    """
    Merge usage stats for *all* requests into one DataFrame, each row representing a usage_stats entry.
    - Include a column 'example_number' to mark which request it originated from.
    """
    all_rows = []
    try:
        for i, steps in enumerate(requests_data):
            for step in steps:
                # Create a DataFrameRow
                all_rows.append(
                    DataFrameRow(event_timestamp=step.event_timestamp,
                                 example_number=i,
                                 prompt_tokens=step.token_usage.prompt_tokens,
                                 completion_tokens=step.token_usage.completion_tokens,
                                 total_tokens=step.token_usage.total_tokens,
                                 llm_text_input=step.llm_text_input,
                                 llm_text_output=step.llm_text_output,
                                 llm_new_token=step.llm_text_chunk,
                                 llm_name=step.llm_name,
                                 tool_name=step.tool_name,
                                 function_name=step.function_name,
                                 function_id=step.function_id,
                                 parent_function_name=step.parent_function_name,
                                 parent_function_id=step.parent_function_id,
                                 UUID=step.payload.UUID,
                                 framework=step.framework,
                                 event_type=step.event_type).model_dump(), )

    except Exception as e:
        logger.exception("Error creating standardized DataFrame: %s", e, exc_info=True)
        return pd.DataFrame()

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame.from_records(all_rows)
