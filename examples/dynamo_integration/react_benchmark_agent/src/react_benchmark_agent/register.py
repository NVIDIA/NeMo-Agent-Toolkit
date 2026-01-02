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

# flake8: noqa

# Import the generated workflow function to trigger registration
from .react_benchmark_agent import react_benchmark_agent_function

# Import banking tools group
from .banking_tools import banking_tools_group_function

# Import self-evaluating agent wrappers (both modes from unified module)
# - self_evaluating_agent: Legacy mode, no feedback by default
# - self_evaluating_agent_with_feedback: Advanced mode with feedback
from .self_evaluating_agent_with_feedback import self_evaluating_agent_function
from .self_evaluating_agent_with_feedback import self_evaluating_agent_with_feedback_function

# Import custom evaluators
# from .evaluators import action_completion_evaluator_function # not used in this example, keeping for reference
from .evaluators import tsq_evaluator_function
