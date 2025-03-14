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

import logging

from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class RequestHeaders(BaseModel):
    """Represents a data model for REST registry handler request headers.

    Args:
        accept (str): Specifies the media types the client can accept. Defaults to 'application/json'
        content_type (str): Describes the format of the request body data. Defaults to 'application/json'
        authorization (str): Contains authentication credentials for accessing a protected resource.
    """

    accept: str = Field("application/json", alias="Accept")
    content_type: str = Field("application/json", alias="Content-Type")
    authorization: str = Field(..., alias="Authorization")


class ResponseHeaders(BaseModel):
    """Placehoder data model for REST registry handler resopnse headers.
    """

    pass
