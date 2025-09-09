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

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class ChatFeedbackTool(FunctionBaseConfig, name="chat_feedback"):
    """
    A tool that allows adding reactions/feedback to Weave calls. This tool retrieves a Weave call
    by its ID and adds a reaction (like thumbs up/down) to provide feedback on the call's output.
    The tool automatically configures the Weave project from the builder's telemetry exporters.
    """
    pass


@register_function(config_type=ChatFeedbackTool)
async def chat_feedback(config: ChatFeedbackTool, builder: Builder):

    async def _add_chat_feedback(weave_call_id: str, reaction_type: str) -> str:
        import weave

        # Get the weave project configuration from the builder's telemetry exporters
        weave_project = None

        # Handle both ChildBuilder and WorkflowBuilder
        workflow_builder = getattr(builder, '_workflow_builder', builder)

        if hasattr(workflow_builder, '_telemetry_exporters'):
            for exporter_config in workflow_builder._telemetry_exporters.values():
                if hasattr(exporter_config.config, 'project'):
                    # Construct project name in the same format as the weave exporter
                    entity = getattr(exporter_config.config, 'entity', None)
                    project = exporter_config.config.project
                    weave_project = f"{entity}/{project}" if entity else project
                    break

        client = weave.init(weave_project)
        call = client.get_call(weave_call_id)
        call.feedback.add_reaction(reaction_type)

        return f"Added reaction '{reaction_type}' to call {weave_call_id}"

    yield FunctionInfo.from_fn(
        _add_chat_feedback,
        description="Adds a reaction/feedback to a Weave call using the provided call ID and reaction type.")
