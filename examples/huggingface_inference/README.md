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

# HuggingFace Inference Providers

**Complexity:** 🟢 Beginner

Demonstrates how to use HuggingFace models for LLM inference and text embeddings via the NeMo Agent Toolkit provider system.

## Key Features

- **HuggingFace Inference API LLM**: Remote model inference via the Serverless Inference API, Dedicated Inference Endpoints, or self-hosted TGI servers.
- **HuggingFace Embedder (Local)**: Local embedding generation using sentence-transformers models (e.g., BGE, E5, GTE, all-MiniLM).
- **HuggingFace Embedder (Remote)**: Remote embedding generation via TEI servers or HuggingFace Inference Endpoints.
- **YAML Configuration**: Fully configurable via YAML with support for model selection, generation parameters, and endpoint routing.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install Dependencies

```bash
uv pip install -e '.[langchain]'
uv pip install huggingface_hub langchain-huggingface sentence-transformers
```

### Set Up API Keys

Set your HuggingFace API token as an environment variable:

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
```

You can obtain a token from [HuggingFace Settings](https://huggingface.co/settings/tokens). Ensure the token has Inference API permissions.

## Configuration

See [configs/config.yaml](configs/config.yaml) for a reference configuration showing:

- **LLMs**: Three deployment modes (Serverless API, Custom Endpoint, Self-hosted TGI)
- **Embedders**: Local sentence-transformers and remote TEI server configurations

This config defines provider entries that can be referenced from your own workflow configurations. To use these providers in a workflow, add the relevant `llms` or `embedders` sections to your workflow's config file.
