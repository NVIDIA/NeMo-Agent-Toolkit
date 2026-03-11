# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Validates Open Question #2 from NAT_NATIVE_DESIGN.md:

    "Can @register_per_user_function_group provide per-item stub isolation
    in the eval context?"

The eval runner generates a unique user_id per item when per_input_user_id=True
(the default). This causes SessionManager.session(user_id=unique_id) to create
a fresh PerUserWorkflowBuilder per item, which in turn creates fresh
@register_per_user_function_group instances with clean state.

This test validates that mechanism directly using the SessionManager API — the
same path the eval runner takes — without requiring a live LLM endpoint.

Architecture note:
- Per-user function groups are ONLY instantiated when the workflow is also
  registered as per-user (@register_per_user_function).  When the workflow is
  per-user, SessionManager creates a PerUserWorkflowBuilder for each unique
  user_id, and that builder instantiates all per-user function groups fresh.
- The echo workflow here is therefore registered with @register_per_user_function
  to match the real eval-runner pattern.

Expected outcome:
- Each unique user_id gets a fresh function group instance (counter starts at 0)
- State accumulated in one user's session does NOT bleed into another user's session
- This proves per-item isolation is available out-of-the-box for benchmarks that
  use @register_per_user_function_group.
"""

import asyncio
from uuid import uuid4

import pytest
from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_per_user_function
from nat.cli.register_workflow import register_per_user_function_group
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.runtime.session import SessionManager

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class IncrementInput(BaseModel):
    n: int = Field(default=1, description="Amount to add to the counter")


class CounterOutput(BaseModel):
    count: int = Field(description="Counter value after operation")
    instance_id: str = Field(description="Unique ID of this function group instance")


class EchoInput(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Config types (unique names to avoid conflicts with other tests)
# ---------------------------------------------------------------------------


class IsolationCounterGroupConfig(FunctionGroupBaseConfig, name="isolation_counter_group"):
    """Per-user function group with a counter — validates fresh-instance-per-user."""
    initial_value: int = Field(default=0)


class EchoWorkflowConfig(FunctionBaseConfig, name="isolation_echo_workflow"):
    """
    Minimal per-user echo workflow.

    Must be registered with @register_per_user_function so that SessionManager
    sets _is_workflow_per_user=True.  Only then does it create a
    PerUserWorkflowBuilder per unique user_id, which in turn instantiates each
    @register_per_user_function_group fresh for that user.
    """
    pass


# ---------------------------------------------------------------------------
# Component registrations
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _register_components():
    """Register test components once for the module."""

    @register_per_user_function_group(config_type=IsolationCounterGroupConfig)
    async def isolation_counter_group(config: IsolationCounterGroupConfig, builder: Builder):
        """
        Per-user function group. Each user_id gets its own isolated counter
        and a unique instance_id so we can detect which instance responded.
        """
        # State is local to this generator invocation — fresh per user_id
        counter = {"value": config.initial_value}
        instance_id = str(uuid4())

        group = FunctionGroup(config=config)

        async def increment(inp: IncrementInput) -> CounterOutput:
            counter["value"] += inp.n
            return CounterOutput(count=counter["value"], instance_id=instance_id)

        async def get_count(inp: IncrementInput) -> CounterOutput:
            return CounterOutput(count=counter["value"], instance_id=instance_id)

        group.add_function(name="increment",
                           fn=increment,
                           input_schema=IncrementInput,
                           description="Increment counter by n")
        group.add_function(name="get_count",
                           fn=get_count,
                           input_schema=IncrementInput,
                           description="Get current count without incrementing")

        yield group

    # The workflow must be per-user so that SessionManager creates a
    # PerUserWorkflowBuilder per unique user_id (which instantiates per-user
    # function groups fresh for each user).
    @register_per_user_function(config_type=EchoWorkflowConfig, input_type=EchoInput, single_output_type=str)
    async def echo_workflow(config: EchoWorkflowConfig, builder: Builder):

        async def _echo(inp: EchoInput) -> str:
            return inp.text

        yield FunctionInfo.from_fn(_echo)


# ---------------------------------------------------------------------------
# Helper: build a minimal Config using the per-user function group
# ---------------------------------------------------------------------------


def _make_config() -> Config:
    return Config(
        general=GeneralConfig(),
        workflow=EchoWorkflowConfig(),
        function_groups={"counter": IsolationCounterGroupConfig()},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPerUserFunctionGroupEvalIsolation:
    """
    Validates that @register_per_user_function_group provides fresh instances
    per user_id — matching what the eval runner does with per_input_user_id=True.
    """

    @pytest.mark.asyncio
    async def test_different_user_ids_get_independent_counters(self):
        """
        Two sessions with different user_ids must have independent counter state.
        This mirrors what the eval runner does: user_id = base_id + "-" + uuid4()
        for each eval item when per_input_user_id=True (the default).
        """
        config = _make_config()

        async with WorkflowBuilder.from_config(config) as shared_builder:
            session_manager = await SessionManager.create(config=config, shared_builder=shared_builder)

            user_id_a = f"eval-item-{uuid4()}"
            user_id_b = f"eval-item-{uuid4()}"

            # User A increments their counter twice
            async with session_manager.session(user_id=user_id_a) as session_a:
                fns_a = await session_a.workflow.function_groups["counter"].get_accessible_functions()
                increment_a = fns_a["counter__increment"]
                await increment_a.ainvoke(IncrementInput(n=1))
                result_a1 = await increment_a.ainvoke(IncrementInput(n=1))

            # User B increments their counter once
            async with session_manager.session(user_id=user_id_b) as session_b:
                fns_b = await session_b.workflow.function_groups["counter"].get_accessible_functions()
                increment_b = fns_b["counter__increment"]
                result_b1 = await increment_b.ainvoke(IncrementInput(n=1))

            await session_manager.shutdown()

        # A's counter should be 2, B's should be 1 — isolated
        assert result_a1.count == 2, f"User A counter expected 2, got {result_a1.count}"
        assert result_b1.count == 1, f"User B counter expected 1, got {result_b1.count}"

        # Instance IDs must be different — they are separate objects
        assert result_a1.instance_id != result_b1.instance_id, (
            "Different user_ids must produce different function group instances")

    @pytest.mark.asyncio
    async def test_same_user_id_shares_state(self):
        """
        Two calls within the same user_id session share state.
        Confirms that the per-user isolation is at the user_id boundary,
        not per-call.
        """
        config = _make_config()

        async with WorkflowBuilder.from_config(config) as shared_builder:
            session_manager = await SessionManager.create(config=config, shared_builder=shared_builder)

            user_id = f"eval-item-{uuid4()}"

            async with session_manager.session(user_id=user_id) as session:
                fns = await session.workflow.function_groups["counter"].get_accessible_functions()
                increment = fns["counter__increment"]
                get_count = fns["counter__get_count"]

                await increment.ainvoke(IncrementInput(n=5))
                result = await get_count.ainvoke(IncrementInput(n=0))

            await session_manager.shutdown()

        # Within the same session, state accumulates
        assert result.count == 5, f"Expected count 5, got {result.count}"

    @pytest.mark.asyncio
    async def test_eval_runner_pattern_unique_user_ids_give_fresh_instances(self):
        """
        Simulates exactly what the eval runner does with per_input_user_id=True
        (the default, per EvalGeneralConfig line 124-128):

            user_id = config.user_id + "-" + str(uuid4())  # per eval item

        Each item gets user_id += f"-{uuid4()}" — so N items -> N unique user_ids
        -> N fresh function group instances. This test runs 3 simulated items
        concurrently and verifies complete isolation.
        """
        config = _make_config()
        base_user_id = "eval-run-001"

        async with WorkflowBuilder.from_config(config) as shared_builder:
            session_manager = await SessionManager.create(config=config, shared_builder=shared_builder)

            async def simulate_eval_item(item_index: int) -> dict:
                # Mimic: user_id = config.user_id + "-" + str(uuid4())
                user_id = f"{base_user_id}-{uuid4()}"

                async with session_manager.session(user_id=user_id) as session:
                    fns = await session.workflow.function_groups["counter"].get_accessible_functions()
                    increment = fns["counter__increment"]

                    # Each item increments by item_index+1 times
                    result = None
                    for _ in range(item_index + 1):
                        result = await increment.ainvoke(IncrementInput(n=1))

                return {"item": item_index, "count": result.count, "instance_id": result.instance_id}

            # Run 3 items concurrently (mimics eval runner's asyncio.gather)
            results = await asyncio.gather(
                simulate_eval_item(0),  # should end up with count=1
                simulate_eval_item(1),  # should end up with count=2
                simulate_eval_item(2),  # should end up with count=3
            )

            await session_manager.shutdown()

        # Each item should have its own isolated count
        counts = {r["item"]: r["count"] for r in results}
        assert counts[0] == 1, f"Item 0 expected count=1, got {counts[0]}"
        assert counts[1] == 2, f"Item 1 expected count=2, got {counts[1]}"
        assert counts[2] == 3, f"Item 2 expected count=3, got {counts[2]}"

        # All instance_ids must be distinct — 3 separate objects
        instance_ids = {r["instance_id"] for r in results}
        assert len(instance_ids) == 3, (
            f"Expected 3 distinct function group instances, got {len(instance_ids)}: {instance_ids}")
