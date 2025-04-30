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


class RequestManagerBase:
    pass

    @staticmethod
    async def default_request_handler(request: "RequestManagerBase") -> None:
        """
        Default request callback handler for making API request. This is a pass-through function
        that returns None if no callback function has been explicitly set.

        Args:
            request (RequestManagerBase): Request Manager Interface.
        """
        return None
