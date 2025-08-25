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

# NVIDIA NeMo Agent Toolkit Troubleshooting

## Workflow Issues

- **Workflow Not Found**: Ensure that your workflow is correctly registered and that the `_type` in your configuration file matches the workflow's `_type`.

- **Requested {category} type is ambiguous**: This error might arise when the `_type` in your configuration file is not unique. Please ensure that the `_type` is unique for each workflow. It can also occur after upgrading the toolkit from a previous version in-place when developing. To fix this issue, run the following commands:

    <!-- path-check-skip-begin -->
    ```bash
    # Remove all __pycache__ directories -- the existing __pycache__ directories contain the old aiqtoolkit packages
    find . -name __pycache__ -type d -exec rm -rf {} +
    # Remove references to the old aiqtoolkit packages
    rm -rf packages/aiqtoolkit*
    # Remove references to the old aiq tests
    rm -rf tests/aiq
    # Remove the current environment since we are going to recreate it
    deactivate; rm -rf .venv
    # Reinstall the environment
    uv sync --all-groups --all-extras
    ```
    <!-- path-check-skip-end -->

## Runtime Issues

- **[429] Too Many Requests**: This error might arise during executing workflows that involve LLM calls because of rate limiting on the LLM models. It is recommended to pause briefly and then attempt the operation again a few times. For warm fix set the `parse_agent_response_max_retries: 1` in `config.yaml` for the `react_agent`. Usually happens that the `react_agent` exhausts the available LLM rate with entire error stack trace.

- **Environment Variables**: Double-check that your `NVIDIA_API_KEY` is correctly set if using NVIDIA NIMs. For other LLM providers, you may need to set other environment variables.

## Dependency Issues

- **Requested type not found**: Verify that all required dependencies are listed in your `pyproject.toml` file and installed. If in doubt run `uv sync --all-groups --all-extras` from the root of the repository.
