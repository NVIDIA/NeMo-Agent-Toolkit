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

import logging
from abc import ABC
from abc import abstractmethod
from io import StringIO

import click

from nat.builder.front_end import FrontEndBase
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.front_end import FrontEndConfigT
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class SimpleFrontEndPluginBase(FrontEndBase[FrontEndConfigT], ABC):

    async def pre_run(self):
        pass

    async def run(self):

        await self.pre_run()

        # Must yield the workflow function otherwise it cleans up
        async with WorkflowBuilder.from_config(config=self.full_config) as builder:

            if logger.isEnabledFor(logging.INFO):
                stream = StringIO()

                self.full_config.print_summary(stream=stream)

                click.echo(stream.getvalue())

            workflow = await builder.build()
            session_manager = SessionManager(workflow)
            await self.run_workflow(session_manager)

    @abstractmethod
    async def run_workflow(self, session_manager: SessionManager):
        pass
