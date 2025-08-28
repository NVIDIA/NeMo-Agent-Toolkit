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

import functools
import inspect
import logging
import types
from abc import abstractmethod
from collections.abc import AsyncGenerator
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import TypeVar

ModelType = TypeVar("ModelType")
MessagesType = TypeVar("MessagesType")

logger = logging.getLogger(__name__)


class FunctionArgumentWrapper:
    """
    Wrapper for the arguments and keyword arguments of a function.

    The arguments and keyword arguments are stored in the args and kwargs attributes, respectively.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Initialize the FunctionArgumentWrapper.

        Args:
            args: The arguments to the function.
            kwargs: The keyword arguments to the function.
        """
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"FunctionArgumentWrapper(args={self.args}, kwargs={self.kwargs})"


@dataclass
class BaseThinkingInjector:
    """
    Base class for thinking injectors.

    Args:
        system_prompt: The system prompt to inject.
        function_names: The function names to inject the system prompt into.
    """

    system_prompt: str
    function_names: list[str]

    @abstractmethod
    def inject(self, *args, **kwargs) -> FunctionArgumentWrapper:
        """
        Inject the system prompt into the arguments.

        Args:
            args: The arguments to inject the system prompt into.
            kwargs: The keyword arguments to inject the system prompt into.

        Returns:
            FunctionArgumentWrapper: An object that contains the transformed args and kwargs.
        """
        pass


def _make_thinking_decorator(injector: BaseThinkingInjector):

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:

        async def _call_async(obj: object, *call_args, **call_kwargs) -> Any:
            new_args = injector.inject(*call_args, **call_kwargs)
            return await fn(obj, *new_args.args, **new_args.kwargs)

        async def _agen(obj: object, *call_args, **call_kwargs) -> AsyncGenerator[Any, None]:
            new_args = injector.inject(*call_args, **call_kwargs)
            async for item in fn(obj, *new_args.args, **new_args.kwargs):
                yield item

        def _gen(obj: object, *call_args, **call_kwargs) -> Iterable[Any]:
            new_args = injector.inject(*call_args, **call_kwargs)
            yield from fn(obj, *new_args.args, **new_args.kwargs)
            return

        def _sync(obj: object, *call_args, **call_kwargs) -> Any:
            new_args = injector.inject(*call_args, **call_kwargs)
            return fn(obj, *new_args.args, **new_args.kwargs)

        # Decide which wrapper to return
        if inspect.iscoroutinefunction(fn):
            wrapper = _call_async
        elif inspect.isasyncgenfunction(fn):
            wrapper = _agen
        elif inspect.isgeneratorfunction(fn):
            wrapper = _gen
        else:
            wrapper = _sync

        return functools.wraps(fn)(wrapper)

    return decorate


def patch_with_thinking(obj: ModelType, injector: BaseThinkingInjector) -> ModelType:
    """
    Patch the given object with a decorator that injects a system prompt into the supplied messages.
    There is an assumption that the first non-object argument is the messages.

    Args:
        obj: The object to patch.
        injector: The injector to use.

    Returns:
        The patched object.

    Examples:
        >>> from nat.llm.utils.thinking import BaseThinkingInjector
        >>> from nat.llm.utils.thinking import FunctionArgumentWrapper
        >>> from nat.llm.utils.thinking import patch_with_thinking
        >>>
        >>> class MockClass:
        ...     def sync_method(self, *args, **kwargs):
        ...         return (args, kwargs)
        ...
        >>> mock_obj_1 = MockClass()
        >>> class AddThinking(BaseThinkingInjector):
        ...     def inject(self, x: str, *args, **kwargs) -> FunctionArgumentWrapper:
        ...         return FunctionArgumentWrapper(("thinking " + x), *args, **kwargs)
        >>>
        >>> patched_obj = patch_with_thinking(mock_obj_1, AddThinking(
        ...     system_prompt="thinking",
        ...     function_names=["sync_method"],
        ... ))
        >>> patched_obj.sync_method("test", 1, 2, 3, foo="bar")
        (('thinking test', 1, 2, 3), {'foo': 'bar'})
        >>>
        >>> mock_obj_2 = MockClass()
        >>> class AddThinkingWithArgs(BaseThinkingInjector):
        ...     def inject(self, *args, **kwargs) -> FunctionArgumentWrapper:
        ...         return FunctionArgumentWrapper("thinking", *args, **kwargs)
        >>>
        >>> patched_obj = patch_with_thinking(mock_obj_2, AddThinkingWithArgs(
        ...     system_prompt="thinking",
        ...     function_names=["sync_method"],
        ... ))
        >>> patched_obj.sync_method("test", 1, 2, 3, foo="bar")
        (('thinking', 'test', 1, 2, 3), {'foo': 'bar'})
        >>>
        >>> mock_obj_3 = MockClass()
        >>> class AddThinkingWithKwargs(BaseThinkingInjector):
        ...     def inject(self, *args, **kwargs) -> FunctionArgumentWrapper:
        ...         return FunctionArgumentWrapper(*args, thinking=True, **kwargs)
        >>>
        >>> patched_obj = patch_with_thinking(mock_obj_3, AddThinkingWithKwargs(
        ...     system_prompt="thinking",
        ...     function_names=["sync_method"],
        ... ))
        >>> patched_obj.sync_method("test", 1, 2, 3, foo="bar")
        (('test', 1, 2, 3), {'thinking': True, 'foo': 'bar'})
    """

    decorator = _make_thinking_decorator(injector)

    cls = obj if inspect.isclass(obj) else type(obj)
    cls_name = getattr(cls, "__name__", str(cls))

    for name, _ in inspect.getmembers(cls, callable):
        if name not in injector.function_names:
            continue

        descriptor = inspect.getattr_static(cls, name)
        original = descriptor.__func__ if isinstance(descriptor, types.MethodType) else descriptor
        wrapped = decorator(original)

        try:  # instance‑level first
            if not inspect.isclass(obj):
                object.__setattr__(obj, name, types.MethodType(wrapped, obj))
                continue
        except Exception as exc:
            logger.info(
                "Instance‑level patch failed for %s.%s (%s); "
                "falling back to class‑level patch.",
                cls_name,
                name,
                exc,
            )

        try:  # class‑level fallback
            setattr(cls, name, wrapped)
        except Exception as exc:
            logger.info(
                "Cannot patch method %s.%s with thinking: %s",
                cls_name,
                name,
                exc,
            )

    return obj
