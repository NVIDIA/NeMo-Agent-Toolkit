# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for the `demo_workflow` example.

The top-level workflow is a ReAct agent (`_type: react_agent` from
`nvidia-nat-langchain`), which alternates reasoning and tool use. This package
does not register custom tools; the config uses stock tools: core
`current_datetime` and `current_timezone`, plus LangChain `wiki_search` for
Wikipedia snippets.
"""
