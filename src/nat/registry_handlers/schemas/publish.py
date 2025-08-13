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

from nat.data_models.component import ComponentEnum
from nat.data_models.discovery_metadata import DiscoveryMetadata
from nat.registry_handlers.schemas.status import StatusMessage

logger = logging.getLogger(__name__)


class BuiltArtifact(BaseModel):
    """A NAT artifact including base64 encoded string of wheel package and corrosponding discovery metadata.

    Args:
        whl (str): A base64 encoded string of a NAT package wheel (.whl).

        metadata (dict[ComponentEnum, list[DiscoveryMetadata]]): Provides rich discover metadata for developers to
        quickly find useful components.
    """

    whl: str
    metadata: dict[ComponentEnum, list[DiscoveryMetadata]]


class Artifact(BaseModel):
    """A NAT artifact including base64 encoded string of wheel package and corrosponding discovery metadata.

    Args:
        artifact (BuiltArtifact): A NAT artifact including base64 encoded string of wheel package and
        corrosponding discovery metadata.

        whl_path (str): A local path to the built wheel package.
    """

    artifact: BuiltArtifact | None = None
    whl_path: str


class PublishResponse(BaseModel):
    """The expected response from a publish request denoting status information.

    Args:
        status (StatusMessage): Provides metadata describing the success or errors that occurred when
            making a publish request.
    """

    status: StatusMessage


# Compatibility aliases with previous releases
BuiltAIQArtifact = BuiltArtifact
AIQArtifact = Artifact
