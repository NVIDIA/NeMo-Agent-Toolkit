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
"""NeMo Guardrails policy middleware for NAT function boundaries."""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any

from nemoguardrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationLogOptions
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.rails.llm.options import RailStatus
from nemoguardrails.rails.llm.options import RailType

from nat.builder.builder import Builder
from nat.middleware.dynamic.dynamic_function_middleware import DynamicFunctionMiddleware
from nat.middleware.function_middleware import CallNext
from nat.middleware.function_middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext
from nat.plugins.security.middleware.guardrails.exceptions import PostInvokeBlockedError
from nat.plugins.security.middleware.guardrails.nemo_guardrails_middleware_config import GuardrailFunctionFields
from nat.plugins.security.middleware.guardrails.nemo_guardrails_middleware_config import GuardrailsMiddlewareConfig


class GuardrailsMiddleware(DynamicFunctionMiddleware):
    """Hosts NeMo Guardrails as a policy engine at configured function boundaries."""

    def __init__(
        self,
        config: GuardrailsMiddlewareConfig,
        builder: Builder,
    ) -> None:
        """Initialize Guardrails middleware and register configured function targets.

        Args:
            config: Guardrails middleware configuration with required ``RailsConfig`` on ``config.guardrails``.
            builder: Workflow builder used for rail LLM bindings.
        """
        self._llm_rails: LLMRails = LLMRails(config.guardrails)
        self._guardrails_config: GuardrailsMiddlewareConfig = config
        self._rail_llms: set[str] = set((config.llm_bindings or {}).values())
        self._rail_llms_bound: bool = False
        super().__init__(config, builder)

    def _set_modified_rail_value(self, obj: Any, name: str) -> Callable[[str], None]:
        """Build a setter that writes a modified rail value back to an object attribute.

        Args:
            obj: Object whose attribute is reassigned.
            name: Attribute name to write.

        Returns:
            A callable assigning its argument to ``obj.name``.
        """

        def setter(new_value: str) -> None:
            setattr(obj, name, new_value)

        return setter

    def _set_modified_rail_value_in_list(self, items: list[Any], index: int) -> Callable[[str], None]:
        """Build a setter that writes a modified rail value back to a list element.

        Args:
            items: List whose element is reassigned.
            index: Position to write.

        Returns:
            A callable assigning its argument to ``items[index]``.
        """

        def setter(new_value: str) -> None:
            items[index] = new_value

        return setter

    def _iter_targets_at_path(self, value: Any, path: str) -> Iterator[tuple[str, Callable[[str], None]]]:
        """Yield each string reached by a dotted path with a setter to rewrite it in place.
        Args:
            value: Root object to traverse (model instance or list of them).
            path: Dotted attribute path, e.g. ``reviews.review``.

        Yields:
            ``(text, setter)`` pairs where ``setter(new_text)`` rewrites that leaf.
        """
        *prefix, last = path.split(".")
        parents: list[Any] = list(value) if isinstance(value, list) else [value]
        for segment in prefix:
            next_parents: list[Any] = []
            for node in parents:
                attr: Any = getattr(node, segment, None)
                if attr is None:
                    continue
                next_parents.extend(attr if isinstance(attr, list) else [attr])
            parents = next_parents
        for parent in parents:
            leaf: Any = getattr(parent, last, None)
            if isinstance(leaf, str):
                yield leaf, self._set_modified_rail_value(parent, last)
            elif isinstance(leaf, list):
                for index, item in enumerate(leaf):
                    if isinstance(item, str):
                        yield item, self._set_modified_rail_value_in_list(leaf, index)

    def _handle_modified_rail_response(self, response: GenerationResponse, *, fallback: str) -> str:
        """Resolve the output text from a rail response that passed or was modified.

        Args:
            response: Rail generation response from a passed or modified evaluation.
            fallback: Value returned when the response carries no extractable text.

        Returns:
            The response text, preferring the last assistant-role message when the
            response is a message list; ``fallback`` when nothing can be extracted.
        """
        if isinstance(response.response, str) and response.response:
            return response.response

        if isinstance(response.response, list):
            for message in reversed(response.response):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    content: Any = message.get("content")
                    if isinstance(content, str) and content:
                        return content
            for message in reversed(response.response):
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        return content

        return fallback

    def _handle_blocked_rail_response(self, response: GenerationResponse) -> str:
        """Return the safe refusal message from a blocked NeMo Guardrails response.

        Args:
            response: Generation response from a rail that returned a blocked verdict.

        Returns:
            The refusal text.

        Raises:
            ValueError: If the blocked response does not include a refusal message.
        """
        output_data: dict[str, Any] = response.output_data or {}
        bot_message: Any = output_data.get("bot_message")
        if isinstance(bot_message, str) and bot_message:
            return bot_message

        response_text: str = self._handle_modified_rail_response(response, fallback="")
        if response_text:
            return response_text

        raise ValueError("Blocked NeMo Guardrails response did not include a refusal message.")

    def _apply_modified_input(self, context: InvocationContext, text: str) -> None:
        """Write the rail's modified input text back to the invocation context.


        Args:
            context: Invocation context whose ``modified_args[0]`` is updated in-place.
            text: Modified input text returned by the rail.
        """
        args: list[Any] = list(context.modified_args)
        raw_input: Any = args[0]
        if getattr(raw_input, "input_message", None) is not None and hasattr(raw_input, "model_copy"):
            args[0] = raw_input.model_copy(update={"input_message": text})
        else:
            args[0] = text
        context.modified_args = tuple(args)

    def _resolve_guarded_field_paths(self, name: str) -> list[str]:
        """Expand the ``workflow_functions`` config entry for a function into dotted field paths.



        Args:
            name: Fully-qualified function name (e.g. ``retail_tools__get_product_info``).

        Returns:
            Dotted field paths for the function; empty when ``workflow_functions`` is a list
            or the function has no explicit field selection.
        """
        configured: Any = self._guardrails_config.workflow_functions
        if not isinstance(configured, dict):
            return []
        selection: GuardrailFunctionFields | None = configured.get(name)
        if selection is None:
            return []
        paths: list[str] = []
        for field, subpaths in selection.root.items():
            paths.extend([field] if not subpaths else [f"{field}.{subpath}" for subpath in subpaths])
        return paths

    def _iter_guard_targets(
        self,
        value: Any,
        paths: list[str],
        whole_setter: Callable[[str], None],
    ) -> Iterator[tuple[str, Callable[[str], None]]]:
        """Yield ``(text, setter)`` rail targets for one boundary value.

        With configured field paths, yields one target per string leaf, each with a setter that
        rewrites that leaf in place. With no paths, yields a single whole-value target guarding
        ``input_message`` when present, otherwise the stringified value.

        Args:
            value: Boundary value (input argument for pre, output for post).
            paths: Configured dotted field paths; empty selects the whole value.
            whole_setter: Setter applied when guarding the whole value.

        Yields:
            ``(text, setter)`` pairs to evaluate against a rail.
        """
        if not paths:
            text: str = getattr(value, "input_message", None) or (value if isinstance(value, str) else str(value))
            if text:
                yield text, whole_setter
            return
        for path in paths:
            yield from self._iter_targets_at_path(value, path)

    def _rail_blocked(self, response: GenerationResponse) -> bool:
        """Return whether any activated rail signaled a block.

        Args:
            response: Rail generation response.

        Returns:
            True when an activated rail set ``stop``.
        """
        return any(r.stop for r in (response.log.activated_rails if response.log else []))

    async def bind_llms_to_rail(self) -> None:
        """Register NAT-configured LLMs as NeMo Guardrails rail action parameters.


        """
        if self._rail_llms_bound:
            return
        if not self._config.llm_bindings:
            self._rail_llms_bound = True
            return

        from langchain_core.language_models import BaseLanguageModel

        from nat.builder.framework_enum import LLMFrameworkEnum

        for rail_type, llms_key in self._config.llm_bindings.items():
            param: str = "llm" if rail_type in {"default", "main"} else f"{rail_type}_llm"
            llm: BaseLanguageModel = await self._builder.get_llm(
                llms_key,
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            if not isinstance(llm, BaseLanguageModel):
                raise TypeError(f"llm_bindings['{llms_key}'] must resolve to a LangChain BaseLanguageModel "
                                f"(NeMo Guardrails requires LangChain); got {type(llm).__name__}")
            self._llm_rails.register_action_param(param, llm)
        self._rail_llms_bound = True

    def _should_intercept_llm(self, llm_name: str) -> bool:
        """Return whether the middleware should wrap LLM creation for the given name.

        Args:
            llm_name: NAT LLM component name.

        Returns:
            False for rail-bound LLMs so bindings are not double-wrapped.
        """
        if llm_name in self._rail_llms:
            return False
        return super()._should_intercept_llm(llm_name)

    def on_post_invoke_blocked(self, context: InvocationContext, block_message: str) -> Any:
        """Extension point called when a post_invoke rail blocks the function output.

        The default returns the rail's ``block_message`` as the function output —
        the policy's own response to the blocked content.  Override in a subclass
        to raise or return a different value:

            class StrictGuardrails(GuardrailsMiddleware):
                def on_post_invoke_blocked(self, context, block_message):
                    raise PostInvokeBlockedError(block_message)

        Args:
            context: Invocation context at the time of the block.
            block_message: Message from the blocking rail.

        Returns:
            Value to use as the function output.  The rail's block message by default.
        """
        return block_message

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> Any:
        """Run input and output Guardrails rails around a non-streaming function call.

        Args:
            args: Positional arguments for the wrapped function.
            call_next: Next middleware or target function in the chain.
            context: Static metadata for the wrapped function.
            kwargs: Keyword arguments for the wrapped function.

        Returns:
            Function output, possibly rewritten or replaced by policy.
        """
        ctx = InvocationContext(
            function_context=context,
            original_args=args,
            original_kwargs=dict(kwargs),
            modified_args=args,
            modified_kwargs=dict(kwargs),
            output=None,
        )
        result = await self.pre_invoke(ctx)
        if result is not None:
            ctx = result
        blocked_on_input: bool = ctx.output is not None
        if not blocked_on_input:
            ctx.output = await call_next(*ctx.modified_args, **ctx.modified_kwargs)
            result = await self.post_invoke(ctx)
            if result is not None:
                ctx = result
        return ctx.output

    async def function_middleware_stream(
        self,
        *args: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Run Guardrails rails around a streaming call by buffering the full payload.

        Args:
            args: Positional arguments for the wrapped function.
            call_next: Next middleware or target stream in the chain.
            context: Static metadata for the wrapped function.
            kwargs: Keyword arguments for the wrapped function.

        Yields:
            Stream output after output rails evaluate the assembled payload.
        """
        ctx = InvocationContext(
            function_context=context,
            original_args=args,
            original_kwargs=dict(kwargs),
            modified_args=args,
            modified_kwargs=dict(kwargs),
            output=None,
        )
        result = await self.pre_invoke(ctx)
        if result is not None:
            ctx = result
        if ctx.output is not None:
            yield ctx.output
            return

        buffered: list[Any] = [chunk async for chunk in call_next(*ctx.modified_args, **ctx.modified_kwargs)]
        ctx.output = "".join(str(chunk) for chunk in buffered)
        result = await self.post_invoke(ctx)
        if result is not None:
            ctx = result
        yield ctx.output

    async def pre_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Run input rails over the configured input fields (or whole input) and block on refusal.

        Field-selected paths are evaluated one string leaf at a time so a rail rewrite is written
        back into the original input structure; a block sets ``context.output`` so the wrapped
        call is skipped.

        Args:
            context: Invocation context for the current boundary.

        Returns:
            Updated context when an input is blocked or rewritten; otherwise None.
        """
        await self.bind_llms_to_rail()
        if not context.modified_args or context.modified_args[0] is None:
            return None

        value: Any = context.modified_args[0]
        paths: list[str] = self._resolve_guarded_field_paths(context.function_context.name)

        def write_whole(new_value: str) -> None:
            self._apply_modified_input(context, new_value)

        modified: bool = False
        for text, write_back in self._iter_guard_targets(value, paths, write_whole):
            response: GenerationResponse = await self._llm_rails.generate_async(
                prompt=text,
                options=GenerationOptions(
                    rails=["input"],
                    log=GenerationLogOptions(activated_rails=True),
                    output_vars=["user_message", "bot_message"],
                ),
            )
            if self._rail_blocked(response):
                context.output = self._handle_modified_rail_response(response, fallback=text)
                return context
            result_text: str = self._handle_modified_rail_response(response, fallback=text)
            if result_text != text:
                write_back(result_text)
                modified = True
        return context if modified else None

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Run output rails over the configured output fields (or whole output) and block on refusal.

        Field-selected paths are evaluated one string leaf at a time so a rail rewrite is written
        back into the original output structure; a block replaces ``context.output`` via
        ``on_post_invoke_blocked``.

        Args:
            context: Invocation context including function output.

        Returns:
            Updated context when an output is blocked or rewritten; otherwise None.
        """
        await self.bind_llms_to_rail()
        if context.output is None:
            return None

        input_text: str = ""
        if context.original_args:
            raw: Any = context.original_args[0]
            input_text = getattr(raw, "input_message", None) or (raw if isinstance(raw, str) else str(raw))

        value: Any = context.output
        paths: list[str] = self._resolve_guarded_field_paths(context.function_context.name)

        def write_whole(new_value: str) -> None:
            context.output = new_value

        modified: bool = False
        for text, write_back in self._iter_guard_targets(value, paths, write_whole):
            messages: list[dict[str, str]] = ([{"role": "user", "content": input_text}] if input_text else [])
            messages.append({"role": "assistant", "content": text})
            response: GenerationResponse = await self._llm_rails.generate_async(
                messages=messages,
                options=GenerationOptions(
                    rails=["output"],
                    log=GenerationLogOptions(activated_rails=True),
                    output_vars=["bot_message", "user_message"],
                ),
            )
            if self._rail_blocked(response):
                context.output = self.on_post_invoke_blocked(context, self._handle_blocked_rail_response(response))
                return context
            result_text: str = self._handle_modified_rail_response(response, fallback=text)
            if result_text != text:
                write_back(result_text)
                modified = True
        return context if modified else None


__all__ = ["GuardrailsMiddleware", "PostInvokeBlockedError", "RailStatus", "RailType"]
