<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Advanced NeMo Agent toolkit Python Patterns

These patterns are used by individual production projects but are not common across the ecosystem.

## Custom Agent Class (`BaseAgent` subclass)

When built-in agents (ReAct, Router, etc.) don't fit your routing logic, subclass `BaseAgent` to build a custom agent with full control over the reasoning loop.

```python
from typing import Sequence
from nat.agent.base import BaseAgent
from nat.builder.builder import Builder


class MyCustomAgent(BaseAgent):
    def __init__(
        self,
        *,
        llm,
        tools,
        builder: Builder | None = None,
        prompt: str,
        detailed_logs: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            detailed_logs=detailed_logs,
        )
        self.tools_by_name = {tool.name: tool for tool in tools}

        from langchain_core.prompts import PromptTemplate
        self.prompt_template = PromptTemplate(
            input_variables=["query", "tool_list"],
            template=prompt,
        )
        self.chain = self.prompt_template | llm

    async def _build_graph(self, state_schema: type):
        # Only needed if using LangGraph state graphs
        raise NotImplementedError("Uses direct LangChain ainvoke")
```

The TPM Assistant uses this to implement a custom tool router that classifies requests and dispatches to one of 20+ specialized tools, with forced-tool overrides and chat history awareness.

## Custom FastAPI Worker (`FastApiFrontEndPluginWorker` subclass)

Extend the built-in FastAPI frontend to add custom routes, middleware, SSL, or authentication:

```python
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import (
    FastApiFrontEndPluginWorker
)
from nat.data_models.config import AIQConfig
from fastapi import FastAPI


class MyFastAPIWorker(FastApiFrontEndPluginWorker):
    def __init__(self, full_config: AIQConfig):
        super().__init__(full_config)
        # Custom initialization (SSL, auth, etc.)

    def _configure_app(self, app: FastAPI) -> None:
        """Override to add custom routes and middleware."""
        super()._configure_app(app)

        # Add custom API routes
        from my_app.endpoints import my_router
        app.include_router(my_router)
```

Reference the custom worker in the YAML config via `runner_class`:

```yaml
general:
  front_end:
    _type: fastapi
    runner_class: my_package.worker.MyFastAPIWorker
    host: "0.0.0.0"
    port: 8080
```

## Custom Evaluator (`BaseEvaluator` subclass)

Create domain-specific evaluation logic by subclassing `BaseEvaluator`:

```python
from typing import override
from nat.eval.evaluator.base_evaluator import BaseEvaluator
from nat.eval.evaluator.evaluator_model import EvalInputItem, EvalOutputItem


class ConsistencyEvaluator(BaseEvaluator):
    """Evaluates generated output for logical consistency."""

    @override
    async def evaluate(self, item: EvalInputItem) -> EvalOutputItem:
        # item.input contains the original prompt
        # item.output contains the agent's response
        # item.expected contains the expected output (if any)

        score = self._check_consistency(item.output, item.expected)

        return EvalOutputItem(
            score=score,
            reasoning="Consistency check passed" if score >= 0.8 else "Inconsistencies found",
        )

    def _check_consistency(self, output: str, expected: str) -> float:
        # Domain-specific validation logic
        ...
```

Register the evaluator the same way as tools — via `@register_function` or entry points — and reference it in the `eval:` section of the workflow YAML.
