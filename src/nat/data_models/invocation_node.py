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

from pydantic import BaseModel


class InvocationNode(BaseModel):
    """
    Represents a node in an invocation call graph.

    The InvocationNode class encapsulates the details of a specific function
    invocation within a call graph. It stores the unique identifier of the
    invocation, the function name, and optional details about the parent
    node (if any). This class is useful for tracing the execution flow
    in a system or application.

    Attributes:
        function_id (str): Unique identifier for the function invocation.
        function_name (str): Name of the function invoked.
        parent_id (str | None): Unique identifier of the parent invocation, if applicable. Defaults to None.
        parent_name (str | None): Name of the parent function invoked, if applicable. Defaults to None.
    """
    function_id: str
    function_name: str
    parent_id: str | None = None
    parent_name: str | None = None
