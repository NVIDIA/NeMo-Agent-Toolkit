# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Attach each LLM framework's callback handler to the workflow, so its LLM and tool calls become intermediate steps.

This lives in core so tracing works without the optional profiler package.
"""

from __future__ import annotations

import functools
import inspect
import logging
import re
from collections.abc import AsyncIterator
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager as AsyncContextManager
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.type_registry import RegisteredFunctionGroupInfo
from nat.cli.type_registry import RegisteredFunctionInfo

logger = logging.getLogger(__name__)

_FRAMEWORK_REGEX_MAP = {t: fr'\b{t._name_}\b' for t in LLMFrameworkEnum}

_library_instrumented = {
    "langchain": False,
    "crewai": False,
    "semantic_kernel": False,
    "agno": False,
    "adk": False,
    "strands": False,
    "autogen": False,
}

callback_handler_var: ContextVar[Any | None] = ContextVar("callback_handler_var", default=None)


def set_framework_profiler_handler(
    workflow_llms: dict | None = None,
    frameworks: list[LLMFrameworkEnum] | None = None,
) -> Callable[[Callable[..., AsyncContextManager[Any]]], Callable[..., AsyncContextManager[Any]]]:
    """
    Decorator that wraps an async context manager function to set up framework-specific profiling.

    Args:
        workflow_llms (dict | None): A dictionary of workflow LLM configurations.
        frameworks (list[LLMFrameworkEnum] | None): A list of LLM frameworks used in the workflow functions.

    Returns:
        Callable[[Callable[..., AsyncContextManager[Any]]], Callable[..., AsyncContextManager[Any]]]:
        A decorator that wraps the original function with profiling setup.
    """

    def decorator(func: Callable[..., AsyncContextManager[Any]]) -> Callable[..., AsyncContextManager[Any]]:
        """The actual decorator that wraps the function.

        Args:
            func (Callable[..., AsyncContextManager[Any]]): The function to wrap.

        Returns:
            Callable[..., AsyncContextManager[Any]]: The wrapped function.
        """

        @functools.wraps(func)
        @asynccontextmanager
        async def wrapper(workflow_config, builder):

            if LLMFrameworkEnum.LANGCHAIN in frameworks:
                try:
                    # Always set a fresh handler in the current context so callbacks
                    # route to the active run. Only register the hook once globally.
                    from nat.plugins.langchain.callback_handler import LangchainProfilerHandler

                    handler = LangchainProfilerHandler()
                    callback_handler_var.set(handler)

                    if not _library_instrumented["langchain"]:
                        from langchain_core.tracers.context import register_configure_hook
                        register_configure_hook(callback_handler_var, inheritable=True)
                        _library_instrumented["langchain"] = True
                        logger.debug("LangChain/LangGraph callback hook registered")
                except ImportError as e:
                    logger.warning(
                        "LangChain profiler not available. " +
                        "Install NAT with LangChain extras: pip install \"nvidia-nat[langchain]\". Error: %s",
                        e)

            if LLMFrameworkEnum.LLAMA_INDEX in frameworks:
                try:
                    from llama_index.core import Settings
                    from llama_index.core.callbacks import CallbackManager

                    from nat.plugins.llama_index.callback_handler import LlamaIndexProfilerHandler

                    handler = LlamaIndexProfilerHandler()
                    Settings.callback_manager = CallbackManager([handler])
                    logger.debug("LlamaIndex callback handler registered")
                except ImportError as e:
                    logger.warning(
                        "LlamaIndex profiler not available. " +
                        "Install NAT with LlamaIndex extras: pip install \"nvidia-nat[llama_index]\". Error: %s",
                        e)

            if LLMFrameworkEnum.CREWAI in frameworks and not _library_instrumented["crewai"]:
                try:
                    from nat.plugins.crewai.crewai_callback_handler import CrewAIProfilerHandler
                    handler = CrewAIProfilerHandler()
                    handler.instrument()
                    _library_instrumented["crewai"] = True
                    logger.debug("CrewAI callback handler registered")
                except ImportError as e:
                    logger.warning(
                        "CrewAI profiler not available. " +
                        "Install NAT with CrewAI extras: pip install \"nvidia-nat[crewai]\". Error: %s",
                        e)

            if LLMFrameworkEnum.SEMANTIC_KERNEL in frameworks and not _library_instrumented["semantic_kernel"]:
                try:
                    from nat.plugins.semantic_kernel.callback_handler import SemanticKernelProfilerHandler
                    handler = SemanticKernelProfilerHandler(workflow_llms=workflow_llms)
                    handler.instrument()
                    _library_instrumented["semantic_kernel"] = True
                    logger.debug("SemanticKernel callback handler registered")
                except ImportError as e:
                    logger.warning(
                        "SemanticKernel profiler not available. Install NAT with SemanticKernel extras: " +
                        "pip install \"nvidia-nat[semantic_kernel]\". Error: %s",
                        e)

            if LLMFrameworkEnum.AGNO in frameworks and not _library_instrumented["agno"]:
                try:
                    from nat.plugins.agno.callback_handler import AgnoProfilerHandler
                    handler = AgnoProfilerHandler()
                    handler.instrument()
                    _library_instrumented["agno"] = True
                    logger.info("Agno callback handler registered")
                except ImportError as e:
                    logger.warning(
                        "ADK profiler not available. " +
                        "Install NAT with Agno extras: pip install \"nvidia-nat[agno]\". Error: %s",
                        e)

            if LLMFrameworkEnum.ADK in frameworks and not _library_instrumented["adk"]:
                try:
                    from nat.plugins.adk.callback_handler import ADKProfilerHandler
                    handler = ADKProfilerHandler()
                    handler.instrument()
                    _library_instrumented["adk"] = True
                    logger.debug("ADK callback handler registered")
                except ImportError as e:
                    logger.warning(
                        "ADK profiler not available. " +
                        "Install NAT with ADK extras: pip install \"nvidia-nat[adk]\". Error: %s",
                        e)

            if LLMFrameworkEnum.STRANDS in frameworks and not _library_instrumented["strands"]:
                try:
                    from nat.plugins.strands.callback_handler import StrandsProfilerHandler
                    handler = StrandsProfilerHandler()
                    handler.instrument()
                    _library_instrumented["strands"] = True
                    logger.debug("Strands callback handler registered")
                except ImportError as e:
                    logger.warning(
                        "Strands profiler not available. Install NAT with Strands extras: "
                        "pip install \"nvidia-nat[strands]\". Error: %s",
                        e,
                    )

            if LLMFrameworkEnum.AUTOGEN in frameworks and not _library_instrumented["autogen"]:
                try:
                    from nat.plugins.autogen.callback_handler import AutoGenProfilerHandler
                    handler = AutoGenProfilerHandler()
                    handler.instrument()
                    _library_instrumented["autogen"] = True
                    logger.debug("AutoGen callback handler registered")
                except ImportError as e:
                    logger.warning(
                        "AutoGen profiler not available. " +
                        "Install NAT with AutoGen extras: pip install 'nvidia-nat[autogen]'. Error: %s",
                        e)

            # IMPORTANT: actually call the wrapped function as an async context manager
            async with func(workflow_config, builder) as result:
                yield result

        return wrapper

    return decorator


def chain_wrapped_build_fn(
    original_build_fn: Callable[..., AsyncContextManager],
    workflow_llms: dict,
    function_frameworks: list[LLMFrameworkEnum],
) -> Callable[..., AsyncContextManager]:
    """
    Convert an original build function into an async context manager that
    wraps it with a single call to set_framework_profiler_handler, passing
    all frameworks at once.

    Args:
        original_build_fn (Callable[..., AsyncContextManager]): The original build function to wrap.
        workflow_llms (dict): A dictionary of workflow LLM configurations.
        function_frameworks (list[LLMFrameworkEnum]): A list of LLM frameworks used in the workflow functions.

    Returns:
        Callable[..., AsyncContextManager]: The wrapped build function.
    """

    # Define a base async context manager that simply calls the original build function.
    @asynccontextmanager
    async def base_fn(*args, **kwargs) -> AsyncIterator[Any]:
        """Base async context manager that calls the original build function.

        Args:
            *args: Positional arguments to pass to the original build function.
            **kwargs: Keyword arguments to pass to the original build function.

        Yields:
            The result of the original build function.
        """
        async with original_build_fn(*args, **kwargs) as w:
            yield w

    # Instead of wrapping iteratively, we now call the decorator once,
    # passing the entire list of frameworks along with the workflow_llms.
    wrapped_fn = set_framework_profiler_handler(workflow_llms, function_frameworks)(base_fn)
    return wrapped_fn


def detect_llm_frameworks_in_build_fn(
        registration: RegisteredFunctionInfo | RegisteredFunctionGroupInfo) -> list[LLMFrameworkEnum]:
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
