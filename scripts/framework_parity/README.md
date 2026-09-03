<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Framework Parity Harness

Backs the badge table in [`examples/frameworks/README.md`](../../examples/frameworks/README.md).
Runs nightly via [`.github/workflows/framework_parity.yml`](../../.github/workflows/framework_parity.yml).

## What it actually checks

Each entry in [`matrix.py`](matrix.py) gets **two** independent tiers, run in its own
throwaway `uv` venv so one framework's dependencies can never leak into another's
result:

1. **Structural** (always runs, no credentials): `uv pip install -e <example>` into a
   fresh venv, then the real `nat validate` CLI against the example's shipped config
   file. Confirms the example installs the way its own README says to, and that its
   config/registration graph is valid.

2. **Live** (only for entries with `required_live_env` set, and only when those env
   vars are actually present): loads the workflow for real, runs its canonical
   question through the real framework and a real LLM, subscribes to the
   `IntermediateStepManager` event stream, and asserts a `WORKFLOW_START`/`WORKFLOW_END`
   pair and at least one `LLM_END` span with non-zero `usage_info.token_usage.total_tokens`
   were actually emitted. This is the part that backs the "framework-agnostic" claim --
   a config that parses is not proof an integration executes.

## Why most rows are structural-only today

`agno_personal_finance`, `nat_autogen_demo`, `strands_demo`, `multi_frameworks`, and
`haystack_deep_research_agent` each reach across several external services in their
*normal, realistic* configuration (NIM, SerpAPI, Tavily, AWS Bedrock, MCP servers).
Wiring a live check for those means either provisioning that whole service list as CI
secrets, or forking a second "minimal" config that no longer matches what a user
actually installs -- both worse than being honest that the live tier isn't wired yet.
`crewai_demo`, `semantic_kernel_demo`, and `adk_demo`'s `config_oai.yml` were verified
by hand to build and run to the point of a real (auth-failing, without a key) LLM call
using only `OPENAI_API_KEY` (`semantic_kernel_demo` additionally needs `MEM0_API_KEY`
for its memory tools), so those three are live-wired now.

Notably, NAT's own built-in `nat_test_llm` mock provider
(`nvidia_nat_test/src/nat/test/llm.py`) was **not** used for the live tier: its
per-framework client classes are bare stand-ins that don't inherit from each
framework's real base LLM class, so they don't go through that framework's real
callback/instrumentation path -- using it would let this harness pass without ever
proving the profiler's span/token wiring works.

## Running it locally

```bash
# One framework:
python scripts/framework_parity/orchestrate.py --only crewai --live

# The whole matrix:
python scripts/framework_parity/orchestrate.py --live

# Regenerate the table from whatever's in .framework_parity_results/:
python scripts/framework_parity/render_table.py \
    --results-dir .framework_parity_results \
    --readme examples/frameworks/README.md
```

`--live` is a no-op for an entry unless every env var in its `required_live_env` is
already set -- it degrades to the structural tier and reports `live_status: skipped`
rather than failing.
