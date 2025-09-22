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

import pytest

from nat.builder.context import ContextState
from nat.builder.user_interaction_manager import UserInteractionManager
from nat.data_models.api_server import TextContent
from nat.data_models.interactive import BinaryHumanPromptOption
from nat.data_models.interactive import HumanPromptBinary
from nat.data_models.interactive import HumanPromptModelType
from nat.data_models.interactive import HumanPromptText
from nat.data_models.interactive import HumanResponseText
from nat.data_models.interactive import InteractionPrompt

# ------------------------------------------------------------------------------
# Tests for Interactive Data Models
# ------------------------------------------------------------------------------


def test_human_prompt_text_creation():
    """
    Verify that a TextInteraction can be created and its type is correctly set.
    """
    prompt = HumanPromptText(text="Please enter your name:", placeholder="Your name here", required=True)
    assert prompt.input_type == HumanPromptModelType.TEXT
    assert prompt.text == "Please enter your name:"
    assert prompt.placeholder == "Your name here"


def test_human_prompt_binary_valid():
    """
    Verify that a BinaryChoiceInteraction with exactly two options is valid.
    """
    options = [
        BinaryHumanPromptOption(id="yes", label="Yes", value=True),
        BinaryHumanPromptOption(id="no", label="No", value=False),
    ]
    prompt = HumanPromptBinary(text="Can I proceed continue or cancel?", options=options)
    assert prompt.input_type == HumanPromptModelType.BINARY_CHOICE
    assert len(prompt.options) == 2
    # Also check that each option’s label and value are as expected
    assert prompt.options[0].label == "Yes"
    assert prompt.options[1].value is False


def test_human_prompt_binary_invalid():
    """
    Verify that creating a BinaryChoiceInteraction with a number of options other than two raises ValueError.
    """
    # Try with one option
    options = [BinaryHumanPromptOption(id="yes", label="Yes", value=True)]
    with pytest.raises(ValueError, match=r"Binary interactions must have exactly two options"):
        HumanPromptBinary(text="Do you agree?", options=options, required=True)
    # Try with three options
    options = [
        BinaryHumanPromptOption(id="yes", label="Yes", value=True),
        BinaryHumanPromptOption(id="no", label="No", value=False),
        BinaryHumanPromptOption(id="maybe", label="Maybe", value="maybe"),
    ]
    with pytest.raises(ValueError, match=r"Binary interactions must have exactly two options"):
        HumanPromptBinary(text="Select one:", options=options, required=True)


def test_human_response_discriminator_text():
    """
    Verify that a dictionary with type 'text' is correctly parsed as a HumanResponseText.
    """
    data = {"type": "text", "text": "Hello, world!"}
    # Pydantic discriminator should create a HumanResponseText
    response = TextContent.model_validate(data)
    assert isinstance(response, TextContent)
    assert response.text == "Hello, world!"


# ------------------------------------------------------------------------------
# Tests for UserInteractionManager (callback handler)
# ------------------------------------------------------------------------------


async def test_prompt_user_input_text():
    """
    Test that UserInteractionManager.prompt_user_input correctly wraps a
    user-input callback that returns a text response.
    """

    # Define a dummy async callback that returns a HumanResponseText
    async def dummy_text_callback(interaction_prompt: InteractionPrompt) -> HumanResponseText:
        # For testing, simply return a HumanResponseText with a fixed answer.
        return HumanResponseText(text="dummy answer")

    # Get the singleton context state and override the user_input_callback.
    state = ContextState.get()
    token = state.user_input_callback.set(dummy_text_callback)

    try:
        manager = UserInteractionManager(context_state=state)
        # Create a TextInteraction instance as the prompt content.
        prompt_content = HumanPromptText(text="What is your favorite color?", placeholder="Enter color")
        # Call prompt_user_input
        response = await manager.prompt_user_input(prompt_content)
        # And the content should be our HumanResponseText with the dummy answer.
        assert isinstance(response.content, HumanResponseText)
        assert response.content.text == "dummy answer"
    finally:
        # Always reset the token so as not to affect other tests.
        state.user_input_callback.reset(token)
