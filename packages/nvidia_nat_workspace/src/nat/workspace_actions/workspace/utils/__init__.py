# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# re-export utilities
from nat.workspace_actions.workspace.utils.file_state import (
    clear_file_read_state,
    get_file_read_state,
    register_file_read,
)
from nat.workspace_actions.workspace.utils.levenshtein import find_similar_file, levenshtein_distance
from nat.workspace_actions.workspace.utils.path_utils import resolve_workspace_path, validate_within_root
from nat.workspace_actions.workspace.utils.replacers import has_multiple_matches, perform_advanced_replacement
from nat.workspace_actions.workspace.utils.bash_commands import (
    interpret_exit_code,
    is_long_running_command,
    is_read_only_command,
)
from nat.workspace_actions.workspace.utils.ripgrep_utils import get_ripgrep_path
