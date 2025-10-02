# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.data_models.span import Span
from nat.observability.processor.processor import Processor
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


class CrossWorkflowProcessor(Processor[Span, Span]):
    """
    Processor that enriches spans with cross-workflow observability information.

    This processor adds information about workflow chains, trace relationships,
    and cross-workflow attributes to spans to enable better observability
    across multiple workflow executions.
    """

    @override
    async def process(self, item: Span) -> Span:
        """
        Process a span by adding cross-workflow observability information.

        Args:
            item: The original span

        Returns:
            Enhanced span with cross-workflow information
        """
        try:
            # Get current observability context
            # Import here to avoid circular import
            from nat.builder.context import Context
            context = Context.get()
            obs_context = context.observability_context

            if obs_context:
                # Add cross-workflow attributes to the span
                # Add trace information
                item.set_attribute("observability.trace_id", obs_context.trace_id)
                item.set_attribute("observability.root_span_id", obs_context.root_span_id)
                item.set_attribute("observability.current_span_id", obs_context.current_span_id)

                # Add workflow chain information
                workflow_chain = obs_context.workflow_chain
                if workflow_chain:
                    item.set_attribute("observability.workflow_depth", len(workflow_chain))

                    # Add current workflow information
                    current_workflow = workflow_chain[-1]
                    item.set_attribute("observability.workflow_name", current_workflow.workflow_name)
                    item.set_attribute("observability.workflow_id", current_workflow.workflow_id)
                    item.set_attribute("observability.workflow_status", current_workflow.status)

                    if current_workflow.parent_workflow_id:
                        item.set_attribute("observability.parent_workflow_id", current_workflow.parent_workflow_id)

                    # Add timing information if available
                    if current_workflow.start_time:
                        item.set_attribute("observability.workflow_start_time", current_workflow.start_time)
                    if current_workflow.end_time:
                        item.set_attribute("observability.workflow_end_time", current_workflow.end_time)
                        if current_workflow.start_time:
                            item.set_attribute("observability.workflow_duration",
                                               current_workflow.end_time - current_workflow.start_time)

                    # Add workflow tags
                    for key, value in current_workflow.tags.items():
                        item.set_attribute(f"observability.workflow_tag.{key}", value)

                    # Add workflow chain as a serialized attribute
                    workflow_chain_info = []
                    for i, workflow in enumerate(workflow_chain):
                        workflow_info = {
                            "position": i,
                            "name": workflow.workflow_name,
                            "id": workflow.workflow_id,
                            "status": workflow.status
                        }
                        if workflow.parent_workflow_id:
                            workflow_info["parent_id"] = workflow.parent_workflow_id
                        workflow_chain_info.append(workflow_info)

                    item.set_attribute("observability.workflow_chain", str(workflow_chain_info))

                # Add custom attributes from observability context
                for key, value in obs_context.custom_attributes.items():
                    item.set_attribute(f"observability.custom.{key}", value)

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            # If there's any error in processing, log it but don't fail the span
            logger.warning(f"Error processing cross-workflow observability data: {e}", exc_info=True)
            item.set_attribute("observability.processing_error", str(e))

        return item


class WorkflowRelationshipProcessor(Processor[Span, Span]):
    """
    Processor that adds relationship information between workflows.

    This processor analyzes the observability context to add explicit
    parent-child relationships and workflow hierarchy information.
    """

    @override
    async def process(self, item: Span) -> Span:
        """
        Process a span by adding workflow relationship information.

        Args:
            item: The original span

        Returns:
            Enhanced span with relationship information
        """
        try:
            # Get current observability context
            # Import here to avoid circular import
            from nat.builder.context import Context
            context = Context.get()
            obs_context = context.observability_context

            if obs_context and obs_context.workflow_chain:
                workflow_chain = obs_context.workflow_chain

                # Add relationship information
                if len(workflow_chain) > 1:
                    # This is a child workflow
                    parent_workflow = workflow_chain[-2]
                    current_workflow = workflow_chain[-1]

                    item.set_attribute("relationship.type", "child_workflow")
                    item.set_attribute("relationship.parent_workflow_name", parent_workflow.workflow_name)
                    item.set_attribute("relationship.parent_workflow_id", parent_workflow.workflow_id)
                    item.set_attribute("relationship.child_workflow_name", current_workflow.workflow_name)
                    item.set_attribute("relationship.child_workflow_id", current_workflow.workflow_id)

                    # Add hierarchy path
                    hierarchy_path = " -> ".join([w.workflow_name for w in workflow_chain])
                    item.set_attribute("relationship.hierarchy_path", hierarchy_path)

                    # Add depth information
                    item.set_attribute("relationship.nesting_level", len(workflow_chain) - 1)

                else:
                    # This is a root workflow
                    item.set_attribute("relationship.type", "root_workflow")
                    item.set_attribute("relationship.nesting_level", 0)

        except (AttributeError, IndexError, TypeError) as e:
            # If there's any error in processing, log it but don't fail the span
            logger.warning(f"Error processing workflow relationship data: {e}", exc_info=True)
            item.set_attribute("relationship.processing_error", str(e))

        return item
