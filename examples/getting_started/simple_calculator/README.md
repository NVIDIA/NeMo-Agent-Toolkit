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

# A Simple LLM Calculator

**Complexity:** 🟢 Beginner

This example demonstrates an end-to-end (E2E) agentic workflow using the NeMo Agent Toolkit library, fully configured through a YAML file. It showcases the NeMo Agent Toolkit plugin system and `Builder` to seamlessly integrate pre-built and custom tools into workflows.

## Table of Contents

- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow:](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
  - [Run the Workflow](#run-the-workflow)

---

## Key Features

- **Custom Calculator Tools:** Demonstrates six tools - `calculator__add`, `calculator__subtract`, `calculator__multiply`, `calculator__divide`, `calculator__compare`, and `current_datetime` for mathematical operations and time-based comparisons.
- **ReAct Agent Integration:** Uses a `react_agent` that performs reasoning between tool calls to solve complex mathematical queries requiring multiple steps.
- **Multi-step Problem Solving:** Shows how an agent can break down complex questions like "Is the product of 2 * 4 greater than the current hour?" into sequential tool calls.
- **Custom Function Registration:** Demonstrates the NeMo Agent Toolkit plugin system for registering custom mathematical functions with proper validation and error handling.
- **YAML-based Configuration:** Fully configurable workflow that showcases how to orchestrate multiple tools through simple configuration.

---

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install this Workflow:

From the root directory of the NeMo Agent Toolkit library, run the following commands:

```bash
uv pip install -e examples/getting_started/simple_calculator
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/get-started/quick-start.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
export OPENAI_API_KEY=<YOUR_API_KEY>  # OPTIONAL
```

### Run the Workflow

Return to your original terminal, and run the following command from the root of the NeMo Agent Toolkit repo to execute this workflow with the specified input:

```bash
nat run --config_file examples/getting_started/simple_calculator/configs/config.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```

**Expected Workflow Output**
Note that the output is subject to the time of day when the workflow was run. For this example output, it was run in the afternoon.
```
No, the product of 2 * 4 (which is 8) is less than the current hour of the day (which is 15).
```
