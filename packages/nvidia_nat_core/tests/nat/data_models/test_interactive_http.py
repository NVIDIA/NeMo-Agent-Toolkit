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
"""Tests for HTTP interactive data models."""

import json

import pytest

from nat.data_models.interactive import BinaryHumanPromptOption
from nat.data_models.interactive import HumanPromptBinary
from nat.data_models.interactive import HumanPromptCheckbox
from nat.data_models.interactive import HumanPromptDropdown
from nat.data_models.interactive import HumanPromptNotification
from nat.data_models.interactive import HumanPromptRadio
from nat.data_models.interactive import HumanPromptText
from nat.data_models.interactive import HumanResponseBinary
from nat.data_models.interactive import HumanResponseCheckbox
from nat.data_models.interactive import HumanResponseDropdown
from nat.data_models.interactive import HumanResponseNotification
from nat.data_models.interactive import HumanResponseRadio
from nat.data_models.interactive import HumanResponseText
from nat.data_models.interactive import MultipleChoiceOption
from nat.data_models.interactive_http import ExecutionAcceptedInteraction
from nat.data_models.interactive_http import ExecutionAcceptedOAuth
from nat.data_models.interactive_http import ExecutionCompletedStatus
from nat.data_models.interactive_http import ExecutionFailedStatus
from nat.data_models.interactive_http import ExecutionInteractionRequiredStatus
from nat.data_models.interactive_http import ExecutionOAuthRequiredStatus
from nat.data_models.interactive_http import ExecutionRunningStatus
from nat.data_models.interactive_http import ExecutionStatus
from nat.data_models.interactive_http import InteractionResponseRequest
from nat.data_models.interactive_http import StreamInteractionEvent
from nat.data_models.interactive_http import StreamOAuthEvent

# ---------------------------------------------------------------------------
# Helpers: prompt and response fixtures for every interaction type
# ---------------------------------------------------------------------------

_YES = BinaryHumanPromptOption(id="yes", label="Yes", value="yes")
_NO = BinaryHumanPromptOption(id="no", label="No", value="no")

_OPTION_A = MultipleChoiceOption(id="a", label="Option A", value="a", description="First option")
_OPTION_B = MultipleChoiceOption(id="b", label="Option B", value="b", description="Second option")

ALL_PROMPTS = [
    HumanPromptText(text="Enter your name", required=True, placeholder="Name"),
    HumanPromptNotification(text="Workflow paused"),
    HumanPromptBinary(text="Continue?", options=[_YES, _NO]),
    HumanPromptRadio(text="Pick one", options=[_OPTION_A, _OPTION_B]),
    HumanPromptCheckbox(text="Select all that apply", options=[_OPTION_A, _OPTION_B]),
    HumanPromptDropdown(text="Choose from list", options=[_OPTION_A, _OPTION_B]),
]

ALL_RESPONSES = [
    HumanResponseText(text="Alice"),
    HumanResponseNotification(),
    HumanResponseBinary(selected_option=_YES),
    HumanResponseRadio(selected_option=_OPTION_A),
    HumanResponseCheckbox(selected_option=_OPTION_B),
    HumanResponseDropdown(selected_option=_OPTION_A),
]

# ---------------------------------------------------------------------------
# ExecutionStatus
# ---------------------------------------------------------------------------


def test_execution_status_values():
    assert ExecutionStatus.RUNNING == "running"
    assert ExecutionStatus.INTERACTION_REQUIRED == "interaction_required"
    assert ExecutionStatus.OAUTH_REQUIRED == "oauth_required"
    assert ExecutionStatus.COMPLETED == "completed"
    assert ExecutionStatus.FAILED == "failed"


# ---------------------------------------------------------------------------
# ExecutionStatusResponse variants (discriminated union)
# ---------------------------------------------------------------------------


def test_execution_running_status():
    resp = ExecutionRunningStatus(execution_id="abc")
    assert resp.execution_id == "abc"
    assert resp.status == ExecutionStatus.RUNNING


def test_execution_completed_status():
    resp = ExecutionCompletedStatus(execution_id="abc", result={"answer": 42})
    assert resp.result == {"answer": 42}
    assert resp.status == ExecutionStatus.COMPLETED


def test_execution_failed_status():
    resp = ExecutionFailedStatus(execution_id="abc", error="Something went wrong")
    assert resp.error == "Something went wrong"
    assert resp.status == ExecutionStatus.FAILED


@pytest.mark.parametrize("prompt", ALL_PROMPTS, ids=lambda p: p.input_type)
def test_execution_interaction_required_status_all_prompt_types(prompt):
    resp = ExecutionInteractionRequiredStatus(
        execution_id="abc",
        interaction_id="int-1",
        prompt=prompt,
        response_url="/executions/abc/interactions/int-1/response",
    )
    assert resp.interaction_id == "int-1"
    assert resp.prompt == prompt
    assert resp.status == ExecutionStatus.INTERACTION_REQUIRED

    # Roundtrip: serialize and re-parse to verify the discriminated HumanPrompt
    data = json.loads(resp.model_dump_json())
    assert data["prompt"]["input_type"] == prompt.input_type


def test_execution_oauth_required_status():
    resp = ExecutionOAuthRequiredStatus(
        execution_id="abc",
        auth_url="https://auth.example.com/authorize?state=xyz",
        oauth_state="xyz",
    )
    assert resp.auth_url.startswith("https://")
    assert resp.oauth_state == "xyz"
    assert resp.status == ExecutionStatus.OAUTH_REQUIRED


def test_execution_status_serialization_roundtrip():
    """Each variant serializes with the correct ``status`` discriminator."""
    running = ExecutionRunningStatus(execution_id="r1")
    data = json.loads(running.model_dump_json())
    assert data["status"] == "running"
    assert set(data.keys()) == {"execution_id", "status"}

    failed = ExecutionFailedStatus(execution_id="f1", error="boom")
    data = json.loads(failed.model_dump_json())
    assert data["status"] == "failed"
    assert "error" in data


# ---------------------------------------------------------------------------
# ExecutionAcceptedResponse variants (discriminated union)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", ALL_PROMPTS, ids=lambda p: p.input_type)
def test_execution_accepted_interaction_all_prompt_types(prompt):
    resp = ExecutionAcceptedInteraction(
        execution_id="abc",
        status_url="/executions/abc",
        interaction_id="int-1",
        prompt=prompt,
        response_url="/executions/abc/interactions/int-1/response",
    )
    data = json.loads(resp.model_dump_json())
    assert data["execution_id"] == "abc"
    assert data["status"] == "interaction_required"
    assert data["interaction_id"] == "int-1"
    assert data["status_url"] == "/executions/abc"
    assert data["prompt"]["input_type"] == prompt.input_type


def test_execution_accepted_oauth():
    resp = ExecutionAcceptedOAuth(
        execution_id="abc",
        status_url="/executions/abc",
        auth_url="https://auth.example.com/authorize",
        oauth_state="xyz",
    )
    data = json.loads(resp.model_dump_json())
    assert data["status"] == "oauth_required"
    assert data["auth_url"] == "https://auth.example.com/authorize"


# ---------------------------------------------------------------------------
# InteractionResponseRequest – all interaction response types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("response", ALL_RESPONSES, ids=lambda r: r.type)
def test_interaction_response_request_all_types(response):
    body = InteractionResponseRequest(response=response)
    assert body.response.type == response.type


@pytest.mark.parametrize("response", ALL_RESPONSES, ids=lambda r: r.type)
def test_interaction_response_request_serialization_roundtrip(response):
    body = InteractionResponseRequest(response=response)
    raw = body.model_dump_json()
    parsed = InteractionResponseRequest.model_validate_json(raw)
    assert parsed.response.type == response.type


# ---------------------------------------------------------------------------
# StreamInteractionEvent – all prompt types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", ALL_PROMPTS, ids=lambda p: p.input_type)
def test_stream_interaction_event_serialization_all_prompt_types(prompt):
    event = StreamInteractionEvent(
        execution_id="exec-1",
        interaction_id="int-1",
        prompt=prompt,
        response_url="/executions/exec-1/interactions/int-1/response",
    )
    sse = event.get_stream_data()
    assert sse.startswith("event: interaction_required\n")
    assert "data:" in sse
    # Verify JSON payload
    data_line = [line for line in sse.split("\n") if line.startswith("data:")][0]
    data = json.loads(data_line[len("data: "):])
    assert data["event_type"] == "interaction_required"
    assert data["execution_id"] == "exec-1"
    assert data["prompt"]["input_type"] == prompt.input_type


# ---------------------------------------------------------------------------
# StreamOAuthEvent
# ---------------------------------------------------------------------------


def test_stream_oauth_event_serialization():
    event = StreamOAuthEvent(
        execution_id="exec-2",
        auth_url="https://auth.example.com/authorize?state=xyz",
        oauth_state="xyz",
    )
    sse = event.get_stream_data()
    assert sse.startswith("event: oauth_required\n")
    data_line = [line for line in sse.split("\n") if line.startswith("data:")][0]
    data = json.loads(data_line[len("data: "):])
    assert data["event_type"] == "oauth_required"
    assert data["oauth_state"] == "xyz"
