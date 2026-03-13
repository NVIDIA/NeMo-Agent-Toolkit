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

import asyncio
import logging
import re
import select
import sys
import unicodedata

import click
from colorama import Fore
from pydantic import SecretStr

from nat.data_models.interactive import HumanPromptModelType
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import HumanResponseText
from nat.data_models.interactive import InteractionPrompt
from nat.data_models.user_info import BasicUserInfo
from nat.data_models.user_info import UserInfo
from nat.front_ends.console.authentication_flow_handler import ConsoleAuthenticationFlowHandler
from nat.front_ends.console.console_front_end_config import ConsoleFrontEndConfig
from nat.front_ends.simple_base.simple_front_end_plugin_base import SimpleFrontEndPluginBase
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)

_RE_UNICODE_WHITESPACE = re.compile(r'[\u00a0\u2000-\u200a\u202f\u205f\u3000]')
_RE_ZERO_WIDTH = re.compile(r'[\u200b-\u200d\u2060\ufeff]')
_RE_UNICODE_DASHES = re.compile(r'[\u2010-\u2015\ufe58\ufe63\uff0d]')
_RE_SINGLE_QUOTES = re.compile(r'[\u2018\u2019\u201a\u201b]')
_RE_DOUBLE_QUOTES = re.compile(r'[\u201c\u201d\u201e\u201f]')


def _normalize_unicode(text: str) -> str:
    """Replace common Unicode whitespace and punctuation with ASCII equivalents for clean console display."""
    text = _RE_UNICODE_WHITESPACE.sub(' ', text)
    text = _RE_ZERO_WIDTH.sub('', text)
    text = _RE_UNICODE_DASHES.sub('-', text)
    text = _RE_SINGLE_QUOTES.sub("'", text)
    text = _RE_DOUBLE_QUOTES.sub('"', text)
    text = text.replace('\u2026', '...')
    return unicodedata.normalize('NFKC', text)


def _format_output(runner_outputs) -> str:
    """Format workflow outputs as human-readable text with normalized Unicode."""
    if isinstance(runner_outputs, list):
        return "\n".join(_normalize_unicode(str(item)) for item in runner_outputs)
    return _normalize_unicode(str(runner_outputs))


async def prompt_for_input_cli(question: InteractionPrompt) -> HumanResponse:
    """
    A simple CLI-based callback.
    Takes question as str, returns the typed line as str.
    """

    if question.content.input_type == HumanPromptModelType.TEXT:
        timeout: int | None = question.content.timeout
        prompt_text: str = question.content.text

        if timeout is None:
            user_response = click.prompt(text=prompt_text)
            return HumanResponseText(text=user_response)

        # Countdown on its own line, input prompt below
        sys.stdout.write(f"[{timeout}s remaining]\n{prompt_text}: ")
        sys.stdout.flush()

        remaining: int = timeout
        while remaining > 0:
            ready, _, _ = select.select([sys.stdin], [], [], 1)
            if ready:
                user_response: str = sys.stdin.readline().strip()
                return HumanResponseText(text=user_response)
            remaining -= 1
            # Save cursor position, update countdown line, restore cursor position
            sys.stdout.write(f"\033[s\033[A\r[{remaining}s remaining]\033[K\033[u")
            sys.stdout.flush()

        error_msg: str = question.content.error or "This prompt is no longer available."
        click.echo(f"\n{Fore.RED}{error_msg}{Fore.RESET}")
        raise TimeoutError(f"HITL prompt timed out after {timeout}s waiting for human response")

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

        run_user_id: str = UserInfo(
            basic_user=BasicUserInfo(username="nat_run_user", password=SecretStr("nat_run_user"))).get_user_id()

        if (self.front_end_config.input_query):

            async def run_single_query(query):

                async with session_manager.session(
                        user_id=run_user_id,
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
                input_content = f.read()
            async with session_manager.session(user_id=run_user_id) as session:
                async with session.run(input_content) as runner:
                    runner_outputs = await runner.result(to_type=str)
        else:
            assert False, "Should not reach here. Should have been caught by pre_run"

        line = f"{'-' * 50}"
        prefix = f"{line}\n{Fore.GREEN}Workflow Result:\n"
        suffix = f"{Fore.RESET}\n{line}"

        display_output = _format_output(runner_outputs)

        logger.info(f"{prefix}%s{suffix}", display_output)

        # (handler is a stream handler) => (level > INFO)
        effective_level_too_high = all(
            type(h) is not logging.StreamHandler or h.level > logging.INFO for h in logging.getLogger().handlers)
        if effective_level_too_high:
            print(f"{prefix}{display_output}{suffix}")
