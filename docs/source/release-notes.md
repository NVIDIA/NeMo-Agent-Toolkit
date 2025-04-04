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

# NVIDIA AgentIQ Release Notes

## Release 1.0.0
### Summary
This is the first general release of AgentIQ.

## LLM APIs
- NIM
- OpenAI

## Supported LLM Frameworks
- LangChain
- LlamaIndex

## Known Issues
- Faiss is currently broken on Arm64. This is a known issue [#72](https://github.com/NVIDIA/AgentIQ/issues/72) caused by an upstream bug in the Faiss library [https://github.com/facebookresearch/faiss/issues/3936](https://github.com/facebookresearch/faiss/issues/3936).
- AgentIQ applications must use the same name for both the distribution and root package. This is a current implementation limitation and will be addressed in a future release.
- Refer to [https://github.com/NVIDIA/AgentIQ/issues](https://github.com/NVIDIA/AgentIQ/issues) for an up to date list of current issues.
