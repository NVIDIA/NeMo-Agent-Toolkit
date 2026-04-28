<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tavily Weather (LangChain ReAct + Anthropic)

Smoke-test workflow for the `nvidia-nat-tavily` package. A LangChain ReAct agent runs against Claude Sonnet 4.6 (via LiteLLM) with the Tavily tools (`search`, `extract`).

## Run

```bash
export TAVILY_API_KEY=...
export ANTHROPIC_API_KEY=...

uv sync --extra langchain --extra tavily

uv run nat run \
  --config_file examples/tavily_weather/src/nat_tavily_weather/configs/config.yml \
  --input "What is the weather in San Francisco right now?"
```

## What to expect

The agent should make a single `tavily__search` tool call with a query like `"current weather San Francisco"`, receive structured search results, and produce a natural-language answer with the temperature/conditions.
