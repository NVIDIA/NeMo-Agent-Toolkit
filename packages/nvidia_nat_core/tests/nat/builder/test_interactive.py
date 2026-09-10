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

import pytest
from pydantic import ValidationError

from nat.builder.context import ContextState
from nat.builder.user_interaction_manager import UserInteractionManager
from nat.data_models.api_server import TextContent
from nat.data_models.interactive import BinaryHumanPromptOption
from nat.data_models.interactive import HumanPromptBinary
from nat.data_models.interactive import HumanPromptModelType
from nat.data_models.interactive import HumanPromptText
from nat.data_models.interactive import HumanResponseText
from nat.data_models.interactive import InteractionPrompt
from nat.data_models.interactive import InteractionStatus
from nat.data_models.interactive import _HumanPromptOAuthConsent
from nat.utils import providers

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


async def test_prompt_user_input_uses_installed_providers():
    """
    prompt_user_input stamps the outgoing InteractionPrompt and the returned
    InteractionResponse with the id and timestamp from the installed providers.
    """
    fixed_id = "12345678-1234-4321-8765-123456789abc"
    # prompt_user_input reads the time provider once for the prompt and once for the response, so an advancing
    # provider proves each timestamp comes from its own current_time() call rather than being copied.
    times = iter([1700000000.5, 1700000003.5])
    # time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(...)) for the two epoch times above.
    expected_prompt_timestamp = "2023-11-14T22:13:20Z"
    expected_response_timestamp = "2023-11-14T22:13:23Z"

    captured_prompts: list[InteractionPrompt] = []

    async def dummy_text_callback(interaction_prompt: InteractionPrompt) -> HumanResponseText:
        captured_prompts.append(interaction_prompt)
        return HumanResponseText(text="dummy answer")

    state = ContextState.get()
    token = state.user_input_callback.set(dummy_text_callback)
    previous_id_provider = providers.set_id_provider(lambda: fixed_id)
    previous_time_provider = providers.set_time_provider(lambda: next(times))

    try:
        manager = UserInteractionManager(context_state=state)
        prompt_content = HumanPromptText(text="What is your favorite color?", placeholder="Enter color")
        response = await manager.prompt_user_input(prompt_content)
    finally:
        # Always restore the providers and reset the token so as not to affect other tests.
        providers.set_id_provider(previous_id_provider)
        providers.set_time_provider(previous_time_provider)
        state.user_input_callback.reset(token)

    # The prompt handed to the callback carries the provider-generated id and timestamp.
    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert prompt.id == fixed_id
    assert prompt.timestamp == expected_prompt_timestamp
    assert prompt.status == InteractionStatus.IN_PROGRESS
    assert prompt.content == prompt_content

    # The response reuses the prompt id and stamps its own timestamp from the time provider.
    assert response.id == fixed_id
    assert response.timestamp == expected_response_timestamp
    assert response.status == InteractionStatus.COMPLETED
    assert isinstance(response.content, HumanResponseText)
    assert response.content.text == "dummy answer"


# ------------------------------------------------------------------------------
# Tests for HITL timeout and error (HumanPromptBase)
# ------------------------------------------------------------------------------


def test_human_prompt_text_timeout_and_error_defaults():
    """HumanPromptText without timeout/error uses HumanPromptBase defaults."""
    prompt = HumanPromptText(text="Prompt", required=True)
    assert prompt.timeout is None
    assert prompt.error == "This prompt is no longer available."


def test_human_prompt_text_with_timeout_and_error():
    """HumanPromptText accepts timeout and error."""
    prompt = HumanPromptText(
        text="Confirm?",
        required=True,
        placeholder="yes/no",
        timeout=60,
        error="Approval window has expired.",
    )
    assert prompt.timeout == 60
    assert prompt.error == "Approval window has expired."


def test_human_prompt_base_timeout_validation_gt_zero():
    """HumanPromptBase timeout must be > 0 when set."""
    with pytest.raises(ValidationError):
        HumanPromptText(text="x", required=True, timeout=0)
    with pytest.raises(ValidationError):
        HumanPromptText(text="x", required=True, timeout=-1)


# ------------------------------------------------------------------------------
# Tests for _HumanPromptOAuthConsent
# ------------------------------------------------------------------------------


def test_human_prompt_oauth_consent_defaults():
    """_HumanPromptOAuthConsent defaults: input_type is OAUTH_CONSENT."""
    prompt = _HumanPromptOAuthConsent(text="https://auth.example.com/authorize")
    assert prompt.input_type == HumanPromptModelType.OAUTH_CONSENT


def test_human_prompt_oauth_consent_text_preserved():
    """_HumanPromptOAuthConsent stores the authorization URL in the text field."""
    url = "https://auth.example.com/authorize?client_id=abc&state=xyz"
    prompt = _HumanPromptOAuthConsent(text=url)
    assert prompt.text == url
