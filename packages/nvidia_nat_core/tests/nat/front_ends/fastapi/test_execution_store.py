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
from nat.data_models.interactive_http import ExecutionStatus
from nat.front_ends.fastapi.execution_store import ExecutionStore

# ---------------------------------------------------------------------------
# Helpers: prompt / response fixtures for every interaction type
# ---------------------------------------------------------------------------

_YES = BinaryHumanPromptOption(id="yes", label="Yes", value="yes")
_NO = BinaryHumanPromptOption(id="no", label="No", value="no")
_OPT_A = MultipleChoiceOption(id="a", label="A", value="a", description="first")
_OPT_B = MultipleChoiceOption(id="b", label="B", value="b", description="second")

ALL_PROMPT_RESPONSE_PAIRS = [
    (HumanPromptText(text="Name?", required=True), HumanResponseText(text="Alice")),
    (HumanPromptNotification(text="FYI"), HumanResponseNotification()),
    (HumanPromptBinary(text="Continue?", options=[_YES, _NO]), HumanResponseBinary(selected_option=_YES)),
    (HumanPromptRadio(text="Pick", options=[_OPT_A, _OPT_B]), HumanResponseRadio(selected_option=_OPT_A)),
    (HumanPromptCheckbox(text="Check", options=[_OPT_A, _OPT_B]), HumanResponseCheckbox(selected_option=_OPT_B)),
    (HumanPromptDropdown(text="Drop", options=[_OPT_A, _OPT_B]), HumanResponseDropdown(selected_option=_OPT_A)),
]


# ---------------------------------------------------------------------------
# Creation and lookup
# ---------------------------------------------------------------------------
async def test_create_execution_returns_record_with_running_status():
    store = ExecutionStore()
    record = await store.create_execution()
    assert record.execution_id is not None
    assert record.status == ExecutionStatus.RUNNING


async def test_get_returns_none_for_unknown_id():
    store = ExecutionStore()
    result = await store.get("nonexistent")
    assert result is None


async def test_get_returns_created_record():
    store = ExecutionStore()
    record = await store.create_execution()
    fetched = await store.get(record.execution_id)
    assert fetched is record


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


async def test_set_interaction_required():
    store = ExecutionStore()
    record = await store.create_execution()
    prompt = HumanPromptText(text="What?", required=True)
    pending = await store.set_interaction_required(record.execution_id, prompt)

    assert record.status == ExecutionStatus.INTERACTION_REQUIRED
    assert record.pending_interaction is pending
    assert pending.prompt is prompt
    assert not pending.future.done()


async def test_set_oauth_required():
    store = ExecutionStore()
    record = await store.create_execution()
    await store.set_oauth_required(record.execution_id, auth_url="https://auth.example.com", oauth_state="abc")

    assert record.status == ExecutionStatus.OAUTH_REQUIRED
    assert record.pending_oauth is not None
    assert record.pending_oauth.auth_url == "https://auth.example.com"
    assert record.pending_oauth.oauth_state == "abc"


async def test_set_running_clears_pending():
    store = ExecutionStore()
    record = await store.create_execution()
    prompt = HumanPromptText(text="Q?", required=True)
    await store.set_interaction_required(record.execution_id, prompt)
    await store.set_running(record.execution_id)

    assert record.status == ExecutionStatus.RUNNING
    assert record.pending_interaction is None
    assert record.pending_oauth is None


async def test_set_completed():
    store = ExecutionStore()
    record = await store.create_execution()
    await store.set_completed(record.execution_id, result={"answer": 42})

    assert record.status == ExecutionStatus.COMPLETED
    assert record.result == {"answer": 42}
    assert record.completed_at is not None


async def test_set_failed():
    store = ExecutionStore()
    record = await store.create_execution()
    await store.set_failed(record.execution_id, error="boom")

    assert record.status == ExecutionStatus.FAILED
    assert record.error == "boom"
    assert record.completed_at is not None


async def test_first_outcome_is_set_on_interaction_required():
    store = ExecutionStore()
    record = await store.create_execution()
    assert not record.first_outcome.is_set()

    prompt = HumanPromptText(text="Q?", required=True)
    await store.set_interaction_required(record.execution_id, prompt)

    assert record.first_outcome.is_set()


async def test_first_outcome_is_set_on_completed():
    store = ExecutionStore()
    record = await store.create_execution()
    await store.set_completed(record.execution_id, result="done")
    assert record.first_outcome.is_set()


# ---------------------------------------------------------------------------
# Interaction resolution
# ---------------------------------------------------------------------------


async def test_resolve_interaction_sets_future_result():
    store = ExecutionStore()
    record = await store.create_execution()
    prompt = HumanPromptText(text="Name?", required=True)
    pending = await store.set_interaction_required(record.execution_id, prompt)

    response = HumanResponseText(text="Alice")
    await store.resolve_interaction(record.execution_id, pending.interaction_id, response)

    assert pending.future.done()
    assert pending.future.result() == response


@pytest.mark.parametrize("prompt, response",
                         ALL_PROMPT_RESPONSE_PAIRS,
                         ids=[p.input_type for p, _ in ALL_PROMPT_RESPONSE_PAIRS])
async def test_resolve_interaction_all_types(prompt, response):
    """Every HumanPrompt / HumanResponse pair can round-trip through the store."""
    store = ExecutionStore()
    record = await store.create_execution()
    pending = await store.set_interaction_required(record.execution_id, prompt)

    assert record.status == ExecutionStatus.INTERACTION_REQUIRED
    assert pending.prompt == prompt

    await store.resolve_interaction(record.execution_id, pending.interaction_id, response)
    assert pending.future.done()
    assert pending.future.result() == response


async def test_resolve_interaction_raises_on_unknown_execution():
    store = ExecutionStore()
    with pytest.raises(KeyError, match="not found"):
        await store.resolve_interaction("bad", "bad", HumanResponseText(text="x"))


async def test_resolve_interaction_raises_on_wrong_interaction_id():
    store = ExecutionStore()
    record = await store.create_execution()
    prompt = HumanPromptText(text="Q?", required=True)
    await store.set_interaction_required(record.execution_id, prompt)

    with pytest.raises(KeyError, match="not found"):
        await store.resolve_interaction(record.execution_id, "wrong-id", HumanResponseText(text="x"))


async def test_resolve_interaction_raises_on_already_resolved():
    store = ExecutionStore()
    record = await store.create_execution()
    prompt = HumanPromptText(text="Q?", required=True)
    pending = await store.set_interaction_required(record.execution_id, prompt)

    await store.resolve_interaction(record.execution_id, pending.interaction_id, HumanResponseText(text="first"))

    with pytest.raises(ValueError, match="already been resolved"):
        await store.resolve_interaction(record.execution_id, pending.interaction_id, HumanResponseText(text="second"))


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


async def test_cleanup_expired_removes_old_completed():
    store = ExecutionStore(ttl_seconds=0)
    record = await store.create_execution()
    await store.set_completed(record.execution_id, result="done")

    removed = await store.cleanup_expired()
    assert removed == 1
    assert await store.get(record.execution_id) is None


async def test_cleanup_does_not_remove_running():
    store = ExecutionStore(ttl_seconds=0)
    record = await store.create_execution()

    removed = await store.cleanup_expired()
    assert removed == 0
    assert await store.get(record.execution_id) is not None


async def test_remove():
    store = ExecutionStore()
    record = await store.create_execution()
    await store.remove(record.execution_id)
    assert await store.get(record.execution_id) is None


# ---------------------------------------------------------------------------
# Status transition errors
# ---------------------------------------------------------------------------


async def test_set_interaction_required_raises_on_unknown():
    store = ExecutionStore()
    with pytest.raises(KeyError, match="not found"):
        await store.set_interaction_required("bad-id", HumanPromptText(text="Q?", required=True))


async def test_set_running_raises_on_unknown():
    store = ExecutionStore()
    with pytest.raises(KeyError, match="not found"):
        await store.set_running("bad-id")
