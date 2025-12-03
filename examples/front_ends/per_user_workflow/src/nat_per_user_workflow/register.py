# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Registration module for the per-user workflow example.

This module imports all per-user functions and workflows to register them
with NAT's plugin system.
"""

# Import to trigger registration
from nat_per_user_workflow.per_user_functions import per_user_notepad  # noqa: F401
from nat_per_user_workflow.per_user_functions import per_user_preferences  # noqa: F401
from nat_per_user_workflow.per_user_workflow import per_user_assistant_workflow  # noqa: F401
