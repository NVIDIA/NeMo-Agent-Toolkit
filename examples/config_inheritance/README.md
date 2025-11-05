<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Configuration Inheritance

This example demonstrates how to use YAML configuration inheritance in the NeMo Agent toolkit to reduce duplication across similar configuration files.

## Table of Contents

- [Key Features](#key-features)
- [How It Works](#how-it-works)
  - [Base Configuration](#base-configuration)
  - [Variant Configuration](#variant-configuration)
  - [Chained Inheritance](#chained-inheritance)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
  - [Run the Workflows with Variant Configs](#run-the-workflows-with-variant-configs)
- [Real-World Use Cases](#real-world-use-cases)

---

## Key Features

- **Reduce Configuration Duplication**: Define common settings once in a base configuration and reuse across multiple variants
- **Selective Overrides**: Override specific values at any nesting level while inheriting all other settings
- **Multi-Level Inheritance**: Chain multiple configuration files together for progressive customization
- **Flexible File Organization**: Reference base configurations using relative or absolute paths

## How It Works

### Base Configuration

The base config (`base-config.yml`) contains all common settings:

```yaml
general:
  telemetry:
    logging:
      console:
        level: INFO

llms:
  nim_llm:
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 1024

workflow:
  _type: react_agent
  tool_names: [calculator, current_datetime]
  llm_name: nim_llm
  verbose: true
```

### Variant Configuration

Each variant specifies the base and overrides only what's different:

```yaml
# config-high-temp.yml
base: base-config.yml

llms:
  nim_llm:
    temperature: 0.9  # Override just this one value
```

The result is a fully merged configuration with:
- `temperature: 0.9` (overridden)
- All other settings inherited from base

### Chained Inheritance

You can chain inheritance multiple levels deep:

```yaml
# base-config.yml
llms:
  nim_llm:
    temperature: 0.0
general:
  telemetry:
    logging:
      console:
        level: INFO

# config-high-temp.yml
base: base-config.yml
llms:
  nim_llm:
    temperature: 0.9

# config-high-temp-debug.yml
base: config-high-temp.yml  # Inherits from variant, not base!
general:
  telemetry:
    logging:
      console:
        level: DEBUG
```

**Result for `config-high-temp-debug.yml`:**
- `temperature: 0.9` (from config-high-temp.yml)
- `console.level: DEBUG` (from config-high-temp-debug.yml)
- All other settings inherited from base-config.yml

This creates a 3-level inheritance chain:
```
base-config.yml → config-high-temp.yml → config-high-temp-debug.yml
```

---

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install the NeMo Agent toolkit.

### Install this Workflow

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e examples/config_inheritance
```

### Set Up API Keys

If you have not already done so, follow the [Obtaining API Keys](../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Run the Workflows with Variant Configs

This example shows a simple calculator workflow with several configuration variants:

- **`base-config.yml`** - Base configuration with common settings
- **`config-high-temp.yml`** - Variant with higher temperature for creative responses
- **`config-debug.yml`** - Variant with verbose logging for debugging
- **`config-different-model.yml`** - Variant using a different LLM model
- **`config-with-tracing.yml`** - Variant with Weave tracing enabled
- **`config-high-temp-debug.yml`** - Chained inheritance example (base → high-temp → high-temp-debug)

From the root directory of the NeMo Agent toolkit library, run the workflow with different configuration variants:

```bash
# Test basic inheritance
nat run --config_file examples/config_inheritance/configs/config-high-temp.yml --input "What is 25 * 4?"

# Test chained inheritance
nat run --config_file examples/config_inheritance/configs/config-high-temp-debug.yml --input "What is 25 * 4?"

# Compare with debug variant
nat run --config_file examples/config_inheritance/configs/config-debug.yml --input "What is 25 * 4?"
```

---

## Real-World Use Cases

Configuration inheritance is particularly useful for:

- **Environment-specific configs**: Create separate variants for development, staging, and production environments
- **Evaluation configurations**: Define different evaluation scenarios while maintaining consistent base workflow settings
- **Model experiments**: Test different hyperparameters while keeping the workflow structure unchanged
- **LLM provider variations**: Switch between different LLM backends without duplicating configuration
- **Feature toggles**: Enable or disable features through small configuration overrides
- **Team member configs**: Allow team members to overlay personal preferences on shared defaults
- **Progressive customization**: Start with a base configuration and incrementally add features
