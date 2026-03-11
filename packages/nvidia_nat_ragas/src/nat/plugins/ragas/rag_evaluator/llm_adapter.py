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

from collections.abc import Awaitable
from collections.abc import Callable

from pydantic import BaseModel

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.llms.base import InstructorTypeVar


class NatLangChainRagasLLMAdapter(InstructorBaseRagasLLM):
    """Expose a NAT-managed LangChain LLM through ragas' native LLM contract.

    Why this adapter exists instead of a new ``LLMFrameworkEnum`` entry:

    - Framework enums model agent/runtime ecosystems (LangChain, LlamaIndex, etc).
    - ragas' ``InstructorBaseRagasLLM`` is a library-specific scoring interface, not a workflow framework.
    - Keeping the adaptation local avoids expanding global builder/registry surface area for a
      ragas-only concern while preserving the front-facing LLM configuration model.
    """

    def __init__(self, langchain_llm: object, llm_name: str | None = None):
        self._langchain_llm = langchain_llm
        self._llm_name = llm_name

    def _llm_context(self) -> str:
        if self._llm_name:
            return f" for configured LLM `{self._llm_name}`"
        return ""

    @staticmethod
    def _coerce_output(result: object, response_model: type[InstructorTypeVar]) -> InstructorTypeVar:
        if isinstance(result, response_model):
            return result
        if isinstance(result, BaseModel):
            return response_model.model_validate(result.model_dump())
        if isinstance(result, dict):
            return response_model.model_validate(result)
        raise TypeError(f"Unsupported structured output type: {type(result).__name__}")

    def _structured_llm(self, response_model: type[InstructorTypeVar]) -> object:
        with_structured_output = getattr(self._langchain_llm, "with_structured_output", None)
        if not callable(with_structured_output):
            raise TypeError("NAT LLM does not support `with_structured_output`, required for ragas collections metrics"
                            f"{self._llm_context()}.")
        return with_structured_output(response_model)

    def generate(self, prompt: str, response_model: type[InstructorTypeVar]) -> InstructorTypeVar:
        structured_llm = self._structured_llm(response_model)
        invoke = getattr(structured_llm, "invoke", None)
        if not callable(invoke):
            raise TypeError(f"Structured LLM wrapper does not implement sync `invoke`{self._llm_context()}.")
        return self._coerce_output(invoke(prompt), response_model)

    async def agenerate(self, prompt: str, response_model: type[InstructorTypeVar]) -> InstructorTypeVar:
        structured_llm = self._structured_llm(response_model)
        ainvoke = getattr(structured_llm, "ainvoke", None)
        if not callable(ainvoke):
            raise TypeError(f"Structured LLM wrapper does not implement async `ainvoke`{self._llm_context()}.")
        ainvoke_typed = ainvoke
        ainvoke_fn = ainvoke_typed if isinstance(ainvoke_typed, Callable) else None
        if ainvoke_fn is None:
            raise TypeError(f"Structured LLM wrapper has invalid async `ainvoke`{self._llm_context()}.")
        awaitable = ainvoke_fn(prompt)
        if not isinstance(awaitable, Awaitable):
            raise TypeError(f"Structured LLM wrapper `ainvoke` must return an awaitable{self._llm_context()}.")
        return self._coerce_output(await awaitable, response_model)
