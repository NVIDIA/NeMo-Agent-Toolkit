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
from pydantic import TypeAdapter
from pydantic import ValidationError

from nat.data_models.api_server import AuthPayload
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import Message
from nat.data_models.api_server import OAuthMode
from nat.data_models.api_server import OAuthModePreferencePayload


def test_oauth_mode_preference_payload_parses_via_union():
    adapter = TypeAdapter(AuthPayload)
    payload = adapter.validate_python({"method": "oauth_mode_preference", "mode": "popup"})
    assert isinstance(payload, OAuthModePreferencePayload)
    assert payload.mode is OAuthMode.POPUP


def test_oauth_mode_preference_rejects_unknown_mode():
    adapter = TypeAdapter(AuthPayload)
    with pytest.raises(ValidationError):
        adapter.validate_python({"method": "oauth_mode_preference", "mode": "iframe"})


@pytest.mark.parametrize("model", [ChatRequest, ChatRequestOrMessage])
def test_chat_request_rejects_empty_messages(model):
    # The min_length used to sit inside Annotated as a conlist type, which pydantic
    # ignores, so an empty conversation was accepted.
    with pytest.raises(ValidationError) as exc_info:
        model(messages=[])

    assert exc_info.value.errors()[0]["type"] == "too_short"


def test_chat_request_accepts_non_empty_messages():
    request = ChatRequest(messages=[Message(role="user", content="hi")])

    assert len(request.messages) == 1
    assert ChatRequestOrMessage(input_message="hi").messages is None
