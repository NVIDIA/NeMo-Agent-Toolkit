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

import asyncio
import logging

import click
from colorama import Fore

from nat.data_models.interactive import HumanPromptModelType
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import HumanResponseText
from nat.data_models.interactive import InteractionPrompt
from nat.front_ends.console.authentication_flow_handler import ConsoleAuthenticationFlowHandler
from nat.front_ends.console.console_front_end_config import ConsoleFrontEndConfig
from nat.front_ends.simple_base.simple_front_end_plugin_base import SimpleFrontEndPluginBase
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


async def prompt_for_input_cli(question: InteractionPrompt) -> HumanResponse:
    """
    A simple CLI-based callback.
    Takes question as str, returns the typed line as str.
    """

    if question.content.input_type == HumanPromptModelType.TEXT:
        user_response = click.prompt(text=question.content.text)

        return HumanResponseText(text=user_response)

    raise ValueError("Unsupported human prompt input type. The run command only supports the 'HumanPromptText' "
                     "input type. Please use the 'serve' command to ensure full support for all input types.")


class ConsoleFrontEndPlugin(SimpleFrontEndPluginBase[ConsoleFrontEndConfig]):

    def __init__(self, full_config):
        super().__init__(full_config=full_config)

        # Set the authentication flow handler
        self.auth_flow_handler = ConsoleAuthenticationFlowHandler()

    async def pre_run(self):
        if (self.front_end_config.input_query is not None and self.front_end_config.input_file is not None):
            raise click.UsageError("Must specify either --input or --input_file, not both")
        if (self.front_end_config.input_query is None and self.front_end_config.input_file is None):
            raise click.UsageError("Must specify either --input or --input_file")

    async def run_workflow(self, session_manager: SessionManager):

        assert session_manager is not None, "Session manager must be provided"
        runner_outputs = None

        if (self.front_end_config.input_query):

            async def run_single_query(query):

                async with session_manager.session(
                        user_input_callback=prompt_for_input_cli,
                        user_authentication_callback=self.auth_flow_handler.authenticate) as session:
                    async with session.run(query) as runner:
                        base_output = await runner.result(to_type=str)

                        return base_output

            # Convert to a list
            input_list = list(self.front_end_config.input_query)
            logger.debug("Processing input: %s", self.front_end_config.input_query)

            # Make `return_exceptions=False` explicit; all exceptions are raised instead of being silenced
            runner_outputs = await asyncio.gather(*[run_single_query(query) for query in input_list],
                                                  return_exceptions=False)

        elif (self.front_end_config.input_file):

            # Run the workflow
            with open(self.front_end_config.input_file, encoding="utf-8") as f:

                async with session_manager.workflow.run(f) as runner:
                    runner_outputs = await runner.result(to_type=str)
        else:
            assert False, "Should not reach here. Should have been caught by pre_run"

        line = f"{'-' * 50}"
        prefix = f"{line}\n{Fore.GREEN}Workflow Result:\n"
        suffix = f"{Fore.RESET}\n{line}"

        logger.info(f"{prefix}%s{suffix}", runner_outputs)

        # (handler is a stream handler) => (level > INFO)
        effective_level_too_high = all(
            type(h) is not logging.StreamHandler or h.level > logging.INFO for h in logging.getLogger().handlers)
        if effective_level_too_high:
            print(f"{prefix}{runner_outputs}{suffix}")
