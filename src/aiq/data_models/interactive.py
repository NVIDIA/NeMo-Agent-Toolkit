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

import re
import typing
from enum import Enum

from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import Field
from pydantic import field_validator


class HumanPromptModelType(str, Enum):
    """
    Represents the type of an interaction model.
    """
    TEXT = "text"
    NOTIFICATION = "notification"
    BINARY_CHOICE = "binary_choice"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"


class BinaryChoiceOptionsType(str, Enum):
    """
    Represents the types of system interaction binary choice content
    """
    CONTINUE = "continue"
    CANCEL = "cancel"


class MultipleChoiceOptionType(str, Enum):
    """
    Represents the types of system interaction multiple choice content
    """
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"


class BinaryHumanPromptOption(BaseModel):
    """
    Represents a choice for a binary interaction.
    """
    id: str = Field(default="default", description="The ID of the choice.")
    label: str = Field(default="default", description="Label of the choice")
    value: typing.Any = Field(default="default", description="The value of the choice.")


class MultipleChoiceOption(BaseModel):
    id: str = Field(default="default", description="The ID of the choice.")
    label: str = Field(default="default", description="The label for the multiple choice interaction.")
    value: str = Field(default="default", description="The value for the multiple choice interaction.")
    description: str = Field(default="default", description="The description for the multiple choice interaction.")


class HumanResponseText(BaseModel):
    """
    Represents a text response to an interaction.
    """
    type: typing.Literal[HumanPromptModelType.TEXT] = HumanPromptModelType.TEXT
    text: str = Field(description="The text of the response.")


class HumanResponseNotification(BaseModel):
    """
    Represents a notification response to an interaction.
    """
    type: typing.Literal[HumanPromptModelType.NOTIFICATION] = HumanPromptModelType.NOTIFICATION
    text: str = Field(default="Notification acknowledgement.", description="Default notification response text.")


class HumanResponseBinary(BaseModel):
    """
    Represents a binary response to an interaction.
    """
    type: typing.Literal[HumanPromptModelType.BINARY_CHOICE] = HumanPromptModelType.BINARY_CHOICE
    selected_option: BinaryHumanPromptOption = Field(description="The selected binary response.")


class HumanResponseRadio(BaseModel):
    """
    Represents a multiple choice radio response to an interaction.
    """
    type: typing.Literal[HumanPromptModelType.RADIO] = HumanPromptModelType.RADIO
    selected_option: MultipleChoiceOption = Field(description="The selected multiple choice radio response.")


class HumanResponseCheckbox(BaseModel):
    """
    Represents a multiple choice checkbox response to an interaction.
    """
    type: typing.Literal[HumanPromptModelType.CHECKBOX] = HumanPromptModelType.CHECKBOX
    selected_option: MultipleChoiceOption = Field(description="The selected multiple choice checkbox response.")


class HumanResponseDropdown(BaseModel):
    """
    Represents a multiple choice dropdown response to an interaction.
    """
    type: typing.Literal[HumanPromptModelType.DROPDOWN] = HumanPromptModelType.DROPDOWN
    selected_option: MultipleChoiceOption = Field(description="The selected multiple choice dropdown response.")


HumanResponse = typing.Annotated[HumanResponseText | HumanResponseBinary | HumanResponseNotification
                                 | HumanResponseRadio | HumanResponseCheckbox
                                 | HumanResponseDropdown,
                                 Discriminator("type")]


class HumanPromptBase(BaseModel):
    """
    Base interaction model to derive from
    """
    text: str = Field(description="Text prompt that will be displayed to the user.")


class HumanPromptText(HumanPromptBase):
    """
    Represents a text interaction.
    """
    input_type: typing.Literal[HumanPromptModelType.TEXT] = HumanPromptModelType.TEXT
    placeholder: str | None = Field(description="The placeholder for the text.")
    required: bool = Field(default=True, description="Whether the interaction is required.")


class HumanPromptNotification(HumanPromptBase):
    """
    Represents a notification interaction.
    """
    input_type: typing.Literal[HumanPromptModelType.NOTIFICATION] = HumanPromptModelType.NOTIFICATION


class HumanPromptBinary(HumanPromptBase):
    """
    Represents a binary interaction.
    """
    input_type: typing.Literal[HumanPromptModelType.BINARY_CHOICE] = HumanPromptModelType.BINARY_CHOICE
    options: list[BinaryHumanPromptOption] = Field(description="The options for the binary interaction.")

    # Field validator to make sure len(options) == 2
    @field_validator("options", mode="before")
    @classmethod
    def validate_options(cls, options):
        if len(options) != 2:
            raise ValueError("Binary interactions must have exactly two options.")
        return options


class HumanPromptMultipleChoiceBase(HumanPromptBase):
    """
    Represents a multiple choice interaction.
    """
    options: list[MultipleChoiceOption] = Field(description="The options for the multiple choice interaction.")


class HumanPromptRadio(HumanPromptMultipleChoiceBase):
    """
    Represents a radio interaction.
    """
    input_type: typing.Literal[HumanPromptModelType.RADIO] = HumanPromptModelType.RADIO


class HumanPromptCheckbox(HumanPromptMultipleChoiceBase):
    """
    Represents a checkbox interaction.
    """
    input_type: typing.Literal[HumanPromptModelType.CHECKBOX] = HumanPromptModelType.CHECKBOX


class HumanPromptDropdown(HumanPromptMultipleChoiceBase):
    """
    Represents a dropdown interaction.
    """
    input_type: typing.Literal[HumanPromptModelType.DROPDOWN] = HumanPromptModelType.DROPDOWN


HumanPrompt = typing.Annotated[HumanPromptText | HumanPromptNotification | HumanPromptBinary | HumanPromptRadio
                               | HumanPromptCheckbox | HumanPromptDropdown,
                               Discriminator("input_type")]


class InteractionStatus(str, Enum):
    """
    Represents the status of an interaction.
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class InteractionBase(BaseModel):
    """
    Represents a system-human interaction.
    """
    id: str = Field(description="The ID of the interaction.")
    type: str = Field(default="system_human_interaction", description="The type of the interaction.")
    thread_id: str | None = Field(description="The thread ID of the interaction.", default=None)
    parent_id: str | None = Field(description="The parent ID of the interaction.", default=None)
    status: InteractionStatus = Field(description="The status of the interaction.", default=InteractionStatus.PENDING)
    timestamp: str = Field(description="The timestamp of the interaction.")

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, timestamp):
        if not re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", timestamp):
            raise ValueError("Timestamp must be in the format 2025-01-13T10:00:03Z")
        return timestamp


class InteractionPrompt(InteractionBase):
    """
    Represents a system-human interaction with a prompt.
    """
    content: HumanPrompt = Field(description="The content of the interaction.")


class InteractionResponse(InteractionBase):
    """
    Represents a system-human interaction with a response.
    """
    content: HumanResponse = Field(description="The content of the interaction.")
