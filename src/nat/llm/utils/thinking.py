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
from collections.abc import AsyncGenerator
from collections.abc import Iterable
from typing import Any
from typing import Callable
from typing import TypeVar

ModelType = TypeVar("ModelType")
MessagesType = TypeVar("MessagesType")

logger = logging.getLogger(__name__)


def _thinking_injector(system_prompt_injector: Callable[[MessagesType], MessagesType], ) -> Callable[..., Any]:
    """
    Inject a system prompt into the messages by returning a decorator that can be be wrapped around a function.

    Args:
        system_prompt_injector: A function that injects a system prompt into the messages.

    Returns:
        A decorator that can be be wrapped around a function.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:

        async def _call_async(obj: object, message: MessagesType, *call_args, **call_kwargs) -> Any:
            new_messages = system_prompt_injector(message)
            return await fn(obj, new_messages, *call_args, **call_kwargs)

        async def _agen(obj: object, message: MessagesType, *call_args, **call_kwargs) -> AsyncGenerator[Any, None]:
            new_messages = system_prompt_injector(message)
            async for item in fn(obj, new_messages, *call_args, **call_kwargs):
                yield item

        def _gen(obj: object, message: MessagesType, *call_args, **call_kwargs) -> Iterable[Any]:
            new_messages = system_prompt_injector(message)
            yield from fn(obj, new_messages, *call_args, **call_kwargs)
            return

        def _sync(obj: object, message: MessagesType, *call_args, **call_kwargs) -> Any:
            new_messages = system_prompt_injector(message)
            return fn(obj, new_messages, *call_args, **call_kwargs)

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


def patch_with_thinking(obj: ModelType,
                        function_names: list[str],
                        system_prompt_injector: Callable[[MessagesType], MessagesType]) -> ModelType:
    """
    Patch the given object with a decorator that injects a system prompt into the supplied messages.
    There is an assumption that the first non-object argument is the messages.

    Args:
        obj: The object to patch.
        function_names: The names of the functions to patch.
        system_prompt_injector: A function that injects a system prompt into the messages.

    Returns:
        The patched object.
    """

    decorator = _thinking_injector(system_prompt_injector)

    cls = obj if inspect.isclass(obj) else type(obj)
    cls_name = getattr(cls, "__name__", str(cls))

    for name, _ in inspect.getmembers(cls, callable):
        if name not in function_names:
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
