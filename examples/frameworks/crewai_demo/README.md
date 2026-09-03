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

# CrewAI Example

**Complexity:** 🟢 Beginner

A minimal example using CrewAI showcasing a multi-agent travel planning system where an Itinerary Expert plans activities, a Budget Advisor prices lodging and totals the trip cost, and a Summarizer compiles the final plan. This is the CrewAI counterpart to the [Semantic Kernel example](../semantic_kernel_demo/README.md) — same task, same two tools, different framework — intended to be run side by side with it as a framework-parity check.

## Table of Contents

- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Run the Workflow](#run-the-workflow)

## Key Features

- **CrewAI Framework Integration:** Demonstrates NeMo Agent Toolkit support for CrewAI alongside other frameworks like LangChain/LangGraph and Semantic Kernel.
- **Multi-Agent Travel Planning:** Three role-based agents (Itinerary Expert, Budget Advisor, Summarizer) running as a sequential `Crew`, each contributing its own specialized `Task`.
- **Shared Tools Across Agents:** Both the itinerary and budget agents draw on the same two NAT-registered tools (`hotel_price`, `local_events`), demonstrating that a single tool implementation is reusable across every supported framework, not rewritten per integration.
- **Task Context Chaining:** The budget and summary tasks receive prior tasks' output via CrewAI's `context=[...]`, so later agents build on earlier agents' work rather than repeating it.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install this Workflow

From the root directory of the NeMo Agent Toolkit library, run the following commands:

```bash
uv pip install -e examples/frameworks/crewai_demo
```

### Set Up API Keys

You need to set your OpenAI API key as an environment variable to access OpenAI AI services:

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
```

## Run the Workflow

```bash
nat run --config_file examples/frameworks/crewai_demo/configs/config.yml --input "Create a 3-day travel itinerary for Tokyo in April, covering hotels and activities within a USD 2000 budget."
```

**Expected Workflow Output**

The workflow produces a day-by-day itinerary, a lodging + activity cost breakdown, and a total estimated cost, compiled by the Summarizer agent into a single final plan — structurally the same kind of output the Semantic Kernel example produces for the same prompt, which is the point: the two examples are meant to be comparable, not just individually correct.
