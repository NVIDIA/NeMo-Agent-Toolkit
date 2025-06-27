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

import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class ConversationMessage(BaseModel):
    """A single message in a conversation."""
    model_config = ConfigDict(extra="forbid")
    
    role: str = Field(..., description="The role of the message sender (user, assistant, system)")
    content: str = Field(..., description="The content of the message")
    timestamp: Optional[datetime.datetime] = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="When the message was created"
    )


class Conversation(BaseModel):
    """A conversation containing multiple messages."""
    model_config = ConfigDict(extra="forbid")
    
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    messages: List[ConversationMessage] = Field(default_factory=list, description="List of messages in the conversation")
    mode: str = Field(default="DEFAULT", description="The mode this conversation is running in")
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="When the conversation was created"
    )
    updated_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="When the conversation was last updated"
    ) 