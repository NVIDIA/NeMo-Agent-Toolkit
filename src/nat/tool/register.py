# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# flake8: noqa

# Import any tools which need to be automatically registered here
from . import chat_completion
from . import datetime_tools
from . import document_search
from . import github_tools
from . import nvidia_rag
from . import retriever
from . import server_tools
from .code_execution import register
from .github_tools import create_github_commit
from .github_tools import create_github_issue
from .github_tools import create_github_pr
from .github_tools import get_github_file
from .github_tools import get_github_issue
from .github_tools import get_github_pr
from .github_tools import update_github_issue
from .mcp import mcp_client_impl
from .mcp import mcp_tool
from .memory_tools import add_memory_tool
from .memory_tools import delete_memory_tool
from .memory_tools import get_memory_tool
