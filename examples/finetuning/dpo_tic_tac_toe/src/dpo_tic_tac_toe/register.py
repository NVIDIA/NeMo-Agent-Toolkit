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
"""
NAT component registration entry point for DPO Tic-Tac-Toe example.

This module imports all registered components to trigger their registration
with the NAT framework via the entry point in pyproject.toml.
"""

# ruff: noqa: F401

# Register the choose_move NAT Function (base move generator)
# Register TTC strategies
from .board_position_scorer import register_board_position_scorer  # SCORING
from .choose_move_function import choose_move_function

# Register the main DPO workflow
from .dpo_workflow import dpo_tic_tac_toe_workflow

# Register evaluators
from .evaluator_register import register_dpo_data_collector_evaluator
from .evaluator_register import register_game_outcome_evaluator
from .move_search_strategy import register_multi_candidate_move_search  # SEARCH

# Register the TTC move selector NAT Function (wraps search/score/select)
from .ttc_move_selector_function import ttc_move_selector_function
