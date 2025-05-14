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

# NVIDIA Agent Intelligence Toolkit Troubleshooting

If you encounter any issues:

- **Workflow Not Found**: Ensure that your workflow is correctly registered and that the `_type` in your configuration file matches the workflow's `_type`.

- **Dependency Issues**: Verify that all required dependencies are listed in your `pyproject.toml` file and installed. If in doubt run `uv sync --all-groups --all-extras` from the root of the repository.

- **Environment Variables**: Double-check that your `NVIDIA_API_KEY` is correctly set.

- **[429] Too Many Requests**: This error might arise during executing workflows that involve LLM calls because of rate limiting on the LLM models. It is recommended to pause briefly and then attempt the operation again a few times.
